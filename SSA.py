import numpy as np
import pydicom
import os
import nibabel as nib
import torch
from datetime import datetime
from tqdm import tqdm
import scipy


def extract_info(path, dicom_format = True):
    """
    extract data array, affine matrix and slice thickness from dicom/nifti header information

    :path (str): the path of label map
    :dicom_format (boolean): whether the input file is dicom or not
    :return: a list of data array, affine matrix and slice thickness
    :data_array (numpy array): 2D array of labelmap slice
    :data_affine (numpy array ,4x4): affine matrix of the data
    :data_thickness (int): thickness of slices
    """
    if dicom_format:
        ds = pydicom.dcmread(path)
        data_array = ds.pixel_array.astype(np.float64)
        data_affine = np.eye(4)
        data_affine[:3, 0] = np.array(ds.ImageOrientationPatient[:3])
        data_affine[:3, 1] = np.array(ds.ImageOrientationPatient[3:])
        data_affine[:3, 2] = np.cross(data_affine[:3, 0], data_affine[:3, 1])
        data_affine[:3, 3] = np.array(ds.ImagePositionPatient)
        data_thickness = ds.SliceThickness

        return [data_array, data_affine, data_thickness]

    else:
        data = nib.load(path)
        data_array = data.get_fdata()
        data_affine = data.get_qform()
        data_thickness = data.header['pixdim'][3]

        return [data_array, data_affine, data_thickness]

def intersection_resample(slice_a, slice_b):
    """
    calculate the intersection area by resampling slice_a into empty slice_b (same shape as slice_b)

    :slice_a (list): a list of data array, affine matrix and slice thickness for slice a
    :slice_b (list): a list of data array, affine matrix and slice thickness for slice b
    :return: a resampled intersection area in the empty slice_B
    :intersection_result (numpy array): 2D array of resampled intersection area (same shape as slice_b)
    """

    # read slice A and slice B information
    array_a = slice_a[0]
    affine_a = slice_a[1]
    # predefined the resampling thickness
    thickness_a = 2
    array_b = slice_b[0]
    affine_b = slice_b[1]

    # create block A with given thickness
    a_with_thickness = np.zeros((int(thickness_a), array_a.shape[0], array_a.shape[1]))
    a_with_thickness[:, ...] = array_a
    # affine matrix for block A
    affine_a_with_thickness = np.zeros(affine_a.shape)
    affine_a_with_thickness[affine_a != 0] = affine_a[affine_a != 0]
    affine_a_with_thickness[:, -1] -= affine_a_with_thickness[:, 2] * (a_with_thickness.shape[0]//2)


    # coordinate transformation, ijk to xyz
    # https://medium.com/redbrick-ai/dicom-coordinate-systems-3d-dicom-for-computer-vision-engineers-pt-1-61341d87485f
    ij_index = (np.array(np.meshgrid(np.arange(0, array_b.shape[1]), np.arange(0, array_b.shape[0]), indexing='ij')).T.reshape(array_b.shape[0], array_b.shape[1], 2))
    ijk_index = np.zeros((array_b.shape[0], array_b.shape[1], 4))
    ijk_index[:, :, :2] = ij_index
    # for matrix multiplication, make the last row as all 1
    ijk_index[:, :, -1] = 1
    # each entry for matrix ijk_index is in (i, j, 0, 1) format
    ijk_index = ijk_index.T.reshape(4, array_b.shape[0] * array_b.shape[1])
    # affine matrix for intersection area, which will be used for defining grid later
    affine_inter = np.dot(np.linalg.inv(affine_a_with_thickness), affine_b)
    # xyz coordinate system
    xyz_index_temp = np.dot(affine_inter, ijk_index)
    # remove the last row (all 1)
    xyz_index = xyz_index_temp[:3, :].reshape(3, array_b.shape[1], array_b.shape[0]).T

    # input tensor [N, C, D_in, H_in, W_in] and grid tensor [N, D_out, H_out, W_out, 3] for the grid_sampling function
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    input_tensor = torch.from_numpy(a_with_thickness.reshape((1, 1, a_with_thickness.shape[0], a_with_thickness.shape[1], a_with_thickness.shape[2])))
    grid = torch.from_numpy(xyz_index.reshape((1, 1, array_b.shape[0], array_b.shape[1], 3))).type(torch.DoubleTensor)

    # normalization to [-1, 1] for grid tensor
    norm_factor_0 = (input_tensor.shape[2] - 1) / 2
    norm_factor_1 = (input_tensor.shape[3] - 1) / 2
    norm_factor_2 = (input_tensor.shape[4] - 1) / 2

    grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - norm_factor_2) / (norm_factor_2 + 0.5)
    grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - norm_factor_1) / (norm_factor_1 + 0.5)
    grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - norm_factor_0) / (norm_factor_0 + 0.5)

    # resampled intersection area [N, C, D_out, H_out, W_out]
    intersection_result = torch.nn.functional.grid_sample(input_tensor, grid, mode='nearest', padding_mode='zeros',
                                                          align_corners=False)[0, 0, 0, ...].numpy()

    return intersection_result

def find_max_and_indices(arr):
    """
    given an array, find the maximum and its corresponding index
    """

    max_value = float('-inf')
    max_indices = []

    # Traverse the 2D array to find the maximum value and its indices
    first_dim_half = int(arr.shape[0] // 2)
    second_dim_half = int(arr.shape[1] // 2)
    for i in range(first_dim_half-10, first_dim_half+10):
        for j in range(second_dim_half-10, second_dim_half+10):
            if arr[i][j] - max_value > 1e-5:
                max_value = arr[i][j]
                max_indices = [(i, j)]  # Reset indices list when a new max is found
            elif abs(arr[i][j] - max_value) < 1e-5:
                max_indices.append((i, j))  # Add index to list if it matches the current max

    return max_value, max_indices

def fft_search_new(template_img, moving_img):
    """
    apply fast Fourier transform (FFT) to find the in-plane shift between moving image and template image

    :template_img (numpy array): 2D array of intersection image, which is the accumulated intersection from all other slices
    :moving_img (numpy array): 2D array of moving image, which will be applied with in-plane shift
    :return: in-plane shift to maximize the overlapping between template_img and moving_img
    :best_direction (tuple): [left/right, up/down]
    """

    fft_new = 0
    label_list = np.unique(moving_img)
    for l in range(1, len(label_list)):
        # *1 makes boolean vector as numbers
        moving_binary = (moving_img == label_list[l])*1
        template_binary = (template_img == label_list[l])*1
        tmp_fft_new = scipy.signal.correlate(template_binary, moving_binary,  method='fft')
        fft_new += tmp_fft_new

    fft_ref = scipy.signal.correlate(template_img, template_img, method='fft')
    peak_index_ref = find_max_and_indices(fft_ref)[1]
    peak_index = find_max_and_indices(fft_new)[1]

    # initialize, 10 pixels shift for each dimension
    max_shift_abs = 20
    best_direction = (0, 0)
    for i in range(len(peak_index)):
        current_index = peak_index[i]
        ref_index = peak_index_ref[0]
        tmp_dir = (current_index[1] - ref_index[1], ref_index[0] - current_index[0])
        abs_diff = abs(tmp_dir[0]) + abs(tmp_dir[1])
        if abs_diff == 0:
            max_shift_abs = abs_diff
            best_direction = tmp_dir
            break
        elif abs_diff < max_shift_abs:
            max_shift_abs = abs_diff
            best_direction = tmp_dir

    return best_direction

def apply_affine_transformation(path_data, dicom_file_name, direction, iter, dicom_format = True):
    """
    apply in-plane transformation (i.e., slice shifting) to the current slice

    :path_data (str): location of current data
    :dicom_file_name (str): name of current moving image/slice of a specific data
    :direction (tuple): [left/right, up/down]
    :iter (int): iteration number
    :return: new dicom file with same header information as before but updated content
    """
    if dicom_format:
        path_dicom = os.path.join(path_data, dicom_file_name)
        dataset = pydicom.dcmread(path_dicom)
        moving_img = dataset.pixel_array.astype(np.float64)
        corrected_img = np.zeros(moving_img.shape)

        # slice shifting by changing the content for the slice
        # right
        if direction[0] > 0:
            # up
            if direction[1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
                np.abs(direction[0]): moving_img.shape[1]] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                             0: moving_img.shape[1] - np.abs(direction[0])]

            # down
            else:
                corrected_img[np.abs(direction[1]):moving_img.shape[0],
                np.abs(direction[0]): moving_img.shape[1]] = moving_img[0: moving_img.shape[0] - np.abs(direction[1]),
                                                             0: moving_img.shape[1] - np.abs(direction[0])]

        # left
        else:
            # up
            if direction[1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
                0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                                 np.abs(direction[0]): moving_img.shape[1]]

            # down
            else:
                corrected_img[np.abs(direction[1]):moving_img.shape[0],
                0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[
                                                                 0: moving_img.shape[0] - np.abs(direction[1]),
                                                                 np.abs(direction[0]): moving_img.shape[1]]

        # save new dicom file with same header information as before
        moving_img = np.short(corrected_img)
        dataset.PixelData = moving_img.tobytes()
        dataset.save_as(path_dicom)
    else:
        path_nifti = os.path.join(path_data, dicom_file_name)
        data = nib.load(path_nifti)
        moving_img =data.get_fdata()[..., 0].T
        corrected_img = np.zeros(moving_img.shape)

        # slice shifting by changing the content for the slice
        # right
        if direction[0] > 0:
            # up
            if direction[1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
                np.abs(direction[0]): moving_img.shape[1]] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                             0: moving_img.shape[1] - np.abs(direction[0])]

            # down
            else:
                corrected_img[np.abs(direction[1]):moving_img.shape[0],
                np.abs(direction[0]): moving_img.shape[1]] = moving_img[0: moving_img.shape[0] - np.abs(direction[1]),
                                                             0: moving_img.shape[1] - np.abs(direction[0])]

        # left
        else:
            # up
            if direction[1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
                0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                                 np.abs(direction[0]): moving_img.shape[1]]

            # down
            else:
                corrected_img[np.abs(direction[1]):moving_img.shape[0],
                0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[
                                                                 0: moving_img.shape[0] - np.abs(direction[1]),
                                                                 np.abs(direction[0]): moving_img.shape[1]]

        # save new nifti file with same header information as before
        moving_img_new = corrected_img.T
        new_nifti = nib.Nifti1Image(moving_img_new[:, :, np.newaxis], data.affine)
        nib.save(new_nifti, path_nifti)

    return








if __name__ == '__main__':
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("\n" + "Start Time =", current_time)

    ##################################################################
    # modify here
    path_data = '/media/yx22/DATA/STACOM_code/toy_data_after_SSA'
    ##################################################################
    data_list = sorted(os.listdir(path_data))
    num_iter = 5

    # iterate over data (different patients)
    for i in tqdm(range(len(data_list))):
        path_dicom = os.path.join(path_data, data_list[i])
        print("\n" + data_list[i])
        dicom_list = sorted(os.listdir(path_dicom))

        # number of iteration for convergence (=3 in my case)
        for j in range(num_iter):
            print('----------------------------------------------------------')
            # iterate over dicom file under each data, usually start with LAX-4CH
            for k in range(len(dicom_list)):
                moving_img_name = dicom_list[k]
                dicom_list.remove(moving_img_name)
                moving_img_list = extract_info(os.path.join(path_dicom, moving_img_name), dicom_format=True)
                # moving_img_list[0] = moving_img_list[0].T
                template_img = np.zeros(moving_img_list[0].shape)
                # we might ignore the slice (like apex) containing nothing
                if len(np.unique(moving_img_list[0])) != 1:
                    for s in range(len(dicom_list)):
                        tmp_slice_name = dicom_list[s]
                        tmp_slice_list = extract_info(os.path.join(path_dicom, tmp_slice_name), dicom_format=True)
                        # tmp_slice_list[0] = tmp_slice_list[0].T
                        single_intersection = intersection_resample(tmp_slice_list, moving_img_list)
                        template_img = np.maximum(template_img, single_intersection)


                in_plane_shift = fft_search_new(template_img, moving_img_list[0])
                print('iteration:' + str(j+1) + '; slice name:' + moving_img_name + '; current shift:' + str(in_plane_shift))
                # update dicom file
                apply_affine_transformation(path_dicom, moving_img_name, in_plane_shift, j, True)
                # add back the slice name
                dicom_list.append(moving_img_name)
                dicom_list.sort()
            print('iteration' + str(j+1) + 'finished')


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("\n" + "End Time =", current_time)
    print('##############################################################################')
    print('SSA FINISHED')
    print('##############################################################################')


