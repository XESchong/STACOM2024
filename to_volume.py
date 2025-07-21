#################################################################################
# resample all 2D slice files (DICOM/NIFTI) into a 3D sparse volume
#################################################################################
from SSA import extract_info
import nibabel as nib
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pydicom
import os
import random
import torch

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

input_path = '/local/path/to/toy_data'
output_path = '/local/path/to/sparse_3D'
os.makedirs(output_path, exist_ok=True)
data_list = sorted(os.listdir(input_path))

for d in tqdm(range(len(data_list))):
    data_path = os.path.join(input_path, data_list[d])
    # each folder should have either multiple DICOMs or Nifti files (for SAXs+LAXs)
    dicom_files = os.listdir(data_path)
    # output shape
    final_3d_sparse_vol = np.zeros((160, 160, 160))
    for i in range(len(dicom_files)):
        dicom_path_each = os.path.join(data_path, dicom_files[i])
        # set the 'dicom_format = False' for the NIFTI files, default is True
        dicom_info_list = extract_info(dicom_path_each)
        tmp_array = dicom_info_list[0]
        tmp_affine = dicom_info_list[1]
        # thickness of each slice in final output 3D sparse volume
        thickness = 3
        # resample the 2D slices into the 3D space based on their affine matrices (by grid sampling)
        array_with_thickness = np.zeros((thickness, tmp_array.shape[0], tmp_array.shape[1]))
        array_with_thickness[:, ...] = tmp_array
        affine_with_thickness = np.zeros(tmp_affine.shape)
        affine_with_thickness[tmp_affine != 0] = tmp_affine[tmp_affine != 0]
        ijk_index = (np.array(
            np.meshgrid(np.arange(0, 160), np.arange(0, 160), np.arange(0, 160), indexing='ij')).T.reshape(160,
                                                                                                           160,
                                                                                                           160,
                                                                                                           3))
        ijk_mtx = np.zeros((160, 160, 160, 4))
        ijk_mtx[:, :, :, :3] = ijk_index
        ijk_mtx[:, :, :, -1] = 1
        ijk_mtx = ijk_mtx.T.reshape(4, 160 ** 3)
        # the affine for the sparse volume is user-defined. We choose the identity matrix for simplicity here.
        affine_sparse = np.eye(4)
        affine_inter = np.dot(np.linalg.inv(tmp_affine), affine_sparse)
        xyz_index_temp = np.dot(affine_inter, ijk_mtx)
        xyz_index = xyz_index_temp[:3, :].reshape(3, 160, 160, 160).T
        input_tensor = torch.from_numpy(
            array_with_thickness.reshape((1, 1, array_with_thickness.shape[0], array_with_thickness.shape[1],
                                         array_with_thickness.shape[2])))
        grid = torch.from_numpy(xyz_index.reshape((1, 160, 160, 160, 3))).type(torch.DoubleTensor)
        norm_factor_0 = (input_tensor.shape[2] - 1) / 2
        norm_factor_1 = (input_tensor.shape[3] - 1) / 2
        norm_factor_2 = (input_tensor.shape[4] - 1) / 2
        grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - norm_factor_2) / (norm_factor_2 + 0.5)
        grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - norm_factor_1) / (norm_factor_1 + 0.5)
        grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - norm_factor_0) / (norm_factor_0 + 0.5)

        tmp_img = torch.nn.functional.grid_sample(input_tensor, grid, mode='nearest', padding_mode='zeros',
                                        align_corners=False)[0, 0, ...].numpy()
        # elementwise maximisation for each resampled 2D slice
        final_3d_sparse_vol = np.maximum(tmp_img, final_3d_sparse_vol)
    # save the final 3D sparse volume as NIFTI format
    final_nifti = nib.Nifti1Image(np.transpose(final_3d_sparse_vol), affine=np.eye(4))
    nib.save(final_nifti, os.path.join(output_path, data_list[d] + '.nii.gz'))

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


#################################################################################
# resample all 2D slice files (DICOM/NIFTI) into a 3D sparse volume (for real data)
#################################################################################
from SSA import extract_info
import nibabel as nib
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pydicom
import os
import random
import torch
import scipy
from scipy.spatial import distance
from skimage.measure import label

def getLargestCC(segmentation):
    """
    remove the isolated island and find the largest connected component for each foreground label mask
    """
    labels = label(segmentation)
    # assume at least 1 connected component
    assert( labels.max() != 0 )
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def landmark_gen(path, output_size = 160):
    """
    find the landmarks (apex, mvc, tvc, lvc, rvc, coh) for the CT data

    :path (str): local path of CT data
    :output_size (int): the shape of resampled data, we assume the output should be isotropic, i.e., same spacing for all dimensions

    :return: landmarks in voxel coordinate space and patient coordinate space;
             new spacing and affine matrix for the resampled data under cardiac coordinate space;
             maximum distance for CT data (defined by mvc and apex)
    """

    # required label values:  LVM-1; LV-2; RV-3; RA-4; LA-5
    seg = nib.load(path).get_fdata()
    seg_LV_tmp = (seg == 2).astype(seg.dtype)
    # clear unwanted segmentation (find the largest connected component)
    seg_LV = getLargestCC(seg_LV_tmp).astype(seg.dtype)

    seg_RV_tmp = (seg == 3).astype(seg.dtype)
    seg_RV = getLargestCC(seg_RV_tmp).astype(seg.dtype)
    seg_RA_tmp = (seg == 4).astype(seg.dtype)
    seg_RA = getLargestCC(seg_RA_tmp).astype(seg.dtype)
    seg_LA_tmp = (seg == 5).astype(seg.dtype)
    seg_LA = getLargestCC(seg_LA_tmp).astype(seg.dtype)
    seg_binary_tmp = (seg != 0).astype(seg.dtype)
    seg_binary = getLargestCC(seg_binary_tmp).astype(seg.dtype)

    # mitral valve center
    seg_LA_dilation = scipy.ndimage.binary_dilation(seg_LA).astype(seg.dtype)
    mvc = np.mean(np.where(np.multiply(seg_LV, seg_LA_dilation) == 1), axis=1)
    # apex
    LV_points = np.where(seg_LV == 1)
    max_d = float('-inf')
    apex = None
    for i in range(len(LV_points[0])):
        if len(seg.shape) == 3:
            tmp_point = np.array([LV_points[0][i], LV_points[1][i], LV_points[2][i]])
        else:
            tmp_point = np.array([LV_points[0][i], LV_points[1][i]])

        tmp_distance = distance.euclidean(mvc, tmp_point)
        if tmp_distance > max_d :
            max_d = tmp_distance
            apex = tmp_point
    # centroid's of LV and RV
    lvc = np.mean(np.where(seg_LV == 1), axis=1)
    rvc = np.mean(np.where(seg_RV == 1), axis=1)
    # tricuspid valve center
    seg_RA_dilation = scipy.ndimage.binary_dilation(seg_RA).astype(seg.dtype)
    tvc = np.mean(np.where(np.multiply(seg_RV, seg_RA_dilation) == 1), axis=1)
    # center of the heart
    coh = np.mean(np.where(seg_binary == 1), axis=1)

    # transform landmarks into 3D patient coordinate space
    landmarks_list_voxel_space = [apex, mvc, tvc, lvc, rvc, coh]
    landmarks = np.array(landmarks_list_voxel_space).T
    landmarks_mtx = np.ones((4, landmarks.shape[1]))
    landmarks_mtx[:3, :] = landmarks
    # affine matrix from the original data
    affine_mtx = nib.load(path).get_qform()
    landmarks_patient_space = np.dot(affine_mtx, landmarks_mtx)[:3, :]

    apex_ps = landmarks_patient_space[:, 0]
    mvc_ps = landmarks_patient_space[:, 1]
    tvc_ps = landmarks_patient_space[:, 2]
    lvc_ps = landmarks_patient_space[:, 3]
    rvc_ps = landmarks_patient_space[:, 4]
    coh_ps = landmarks_patient_space[:, 5]

    # find the maximum distance (defined by mvc and apex), which will be used for defining the new spacing,
    # i.e.,  max_distance * 1.3 / output_size, where 30% more is for the background.
    vec_apex_mvc = apex - mvc
    surface_foreground = scipy.ndimage.binary_dilation(seg_binary).astype(seg.dtype) - seg_binary
    surface_foreground_points = np.where(surface_foreground == 1)
    foreground_points_ls = []
    for i in range(len(surface_foreground_points[0])):
        tmp_foreground_point = np.array(
            [surface_foreground_points[0][i], surface_foreground_points[1][i], surface_foreground_points[2][i]])
        foreground_points_ls.append(tmp_foreground_point)
    foreground_points_array = np.array(foreground_points_ls)
    vec_foreground = foreground_points_array - mvc
    dot_res = np.dot(vec_foreground, vec_apex_mvc)
    max_idx = np.argmax(dot_res)
    min_idx = np.argmin(dot_res)
    # closest_point = foreground_points_array[max_idx, :]
    # # map the point into patient coordinate space
    # closest_point_ps = np.dot(affine_mtx, np.append(closest_point, 1))[:3]
    # min_distance = distance.euclidean(closest_point_ps, apex_ps)
    farthest_point = foreground_points_array[min_idx, :]
    # map the point into patient coordinate space
    farthest_point_ps = np.dot(affine_mtx, np.append(farthest_point, 1))[:3]
    max_distance = distance.euclidean(farthest_point_ps, apex_ps)

    # cardiac coordinate space (based on 3D points)
    # x, left to right; mvc to tvc; tvc - mvc; define by v1 in the code
    # y, posterior to anterior; cross product between z and x; defined by v2 in the code
    # z, inferior to superior; apex to mvc; mvc - apex; defined by v3 in the code

    vec_three = mvc_ps - apex_ps
    vec_tmp = tvc_ps - mvc_ps
    vec_two_tmp = np.cross(vec_three, vec_tmp)
    vec_two_final = vec_two_tmp/np.linalg.norm(vec_two_tmp)
    vec_one_tmp = np.cross(vec_two_tmp, vec_three)
    vec_one_final = vec_one_tmp/np.linalg.norm(vec_one_tmp)
    vec_three_tmp = np.cross(vec_one_tmp, vec_two_tmp) # recalculate the vector three (more accurate)
    vec_three_final = vec_three_tmp/np.linalg.norm(vec_three_tmp)

    rotation_mtx = np.zeros((3, 3))
    new_affine_mtx = np.eye(4)
    # new spacing for CT data in cardiac coordinate space, 30% for background
    new_space = (np.round(max_distance) * 1.3) / output_size
    rotation_mtx[:, 0] = vec_one_final * new_space
    rotation_mtx[:, 1] = vec_two_final * new_space
    rotation_mtx[:, 2] = vec_three_final * new_space
    # specify translation (last column of affine matrix)
    center_of_voxel = np.array([80, 80, 80])
    translation_col = coh_ps - np.dot(rotation_mtx, center_of_voxel)
    new_affine_mtx[:3, :3] = rotation_mtx
    new_affine_mtx[:3, 3] = translation_col

    return landmarks_list_voxel_space, landmarks_patient_space, new_space, max_distance, new_affine_mtx


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

input_path = '/local/path/to/toy_data'
output_path_id_aff = '/local/path/to/sparse_3D_id_affine'
os.makedirs(output_path_id_aff , exist_ok=True)
output_path_new_aff = '/local/path/to/sparse_3D_new_affine'
os.makedirs(output_path_new_aff , exist_ok=True)
data_list = sorted(os.listdir(input_path))

for d in tqdm(range(len(data_list))):
    data_path = os.path.join(input_path, data_list[d])
    nii_files = os.listdir(data_path)
    # output shape
    final_3d_sparse_vol = np.zeros((160, 160, 160))
    # find the 4-chamber view NIFTI file
    mri_4ch_path = os.path.join(input_path, data_list[d], [s for s in nii_files if '4ch' in s.lower()][0])
    # generate the landmarks and new affine matrix based on the 4-chamber view
    landmarks_list_voxel_space, landmarks_patient_space, new_space, max_distance, new_affine_mtx = landmark_gen(
        mri_4ch_path, output_size=160)
    for i in range(len(nii_files)):
        nii_path_each = os.path.join(data_path, nii_files[i])
        dicom_info_list = extract_info(nii_path_each, False)
        data_array = dicom_info_list[0]
        data_affine = dicom_info_list[1]
        # normalize the affine matrix (slice thickness)
        data_affine[:3,2] = data_affine[:3, 2]/ np.linalg.norm(data_affine[:3, 2])
        thickness = 3
        data_with_thickness = np.zeros((thickness, data_array.shape[1], data_array.shape[0]))
        data_with_thickness[:, ...] = data_array[..., 0].T
        affine_with_thickness = np.zeros(data_affine.shape)
        affine_with_thickness[data_affine != 0] = data_affine[data_affine != 0]
        ijk_index = (np.array(
            np.meshgrid(np.arange(0, 160), np.arange(0, 160), np.arange(0, 160), indexing='ij')).T.reshape(160,
                                                                                                           160,
                                                                                                           160,
                                                                                                           3))
        ijk_mtx = np.zeros((160, 160, 160, 4))
        ijk_mtx[:, :, :, :3] = ijk_index
        ijk_mtx[:, :, :, -1] = 1
        ijk_mtx = ijk_mtx.T.reshape(4, 160 ** 3)
        # Use the new affine matrix to resample the 2D slices into the 3D space based on their affine matrices
        affine_sparse = new_affine_mtx
        affine_inter = np.dot(np.linalg.inv(data_affine), affine_sparse)
        xyz_index_temp = np.dot(affine_inter, ijk_mtx)
        xyz_index = xyz_index_temp[:3, :].reshape(3, 160, 160, 160).T
        input_tensor = torch.from_numpy(
            data_with_thickness.reshape((1, 1, data_with_thickness.shape[0], data_with_thickness.shape[1],
                                         data_with_thickness.shape[2])))
        grid = torch.from_numpy(xyz_index.reshape((1, 160, 160, 160, 3))).type(torch.DoubleTensor)
        norm_factor_0 = (input_tensor.shape[2] - 1) / 2
        norm_factor_1 = (input_tensor.shape[3] - 1) / 2
        norm_factor_2 = (input_tensor.shape[4] - 1) / 2
        grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - norm_factor_2) / (norm_factor_2 + 0.5)
        grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - norm_factor_1) / (norm_factor_1 + 0.5)
        grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - norm_factor_0) / (norm_factor_0 + 0.5)
        tmp_img = torch.nn.functional.grid_sample(input_tensor, grid, mode='nearest', padding_mode='zeros',
                                                  align_corners=False)[0, 0, ...].numpy()

        final_3d_sparse_vol = np.maximum(tmp_img, final_3d_sparse_vol)
    # save the final 3D sparse volume as NIFTI format
    label_nifti_id_aff = nib.Nifti1Image(np.transpose(final_3d_sparse_vol), affine=np.eye(4))
    label_nifti_new_aff = nib.Nifti1Image(np.transpose(final_3d_sparse_vol), affine=new_affine_mtx)
    nib.save(label_nifti_id_aff, os.path.join(output_path_id_aff, data_list[d] + '.nii.gz'))
    nib.save(label_nifti_new_aff, os.path.join(output_path_new_aff, data_list[d] + '.nii.gz'))

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)
