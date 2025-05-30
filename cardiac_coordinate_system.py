import os
import scipy
import nibabel as nib
import numpy as np
from scipy.spatial import distance
from skimage.measure import label
from scipy.interpolate import RegularGridInterpolator
from motion_correction_function import extract_info
from tqdm import tqdm

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


####################################################################
# label value for CT data : LVM-1; LV-2; RV-3; RA-4; LA-5 (required)
# modify the path here
data_path = '/media/yx22/DATA/test-ccd/input_ct'
output_path_new_aff = '/media/yx22/DATA/test-ccd/output_new'
output_path_id_aff = '/media/yx22/DATA/test-ccd/output_id'
####################################################################

data_list = sorted(os.listdir(data_path))
output_size = 160
for d in tqdm(range(len(data_list))):
    final_3d_dense_vol = np.zeros((output_size, output_size, output_size))
    nifti_path_each = os.path.join(data_path, data_list[d])
    nifti_info_list = extract_info(nifti_path_each, False)
    info_ls = [nifti_info_list[0], nifti_info_list[1], data_list[d][:-7]]
    # Interpolator based on the data array
    x_index = np.linspace(0, info_ls[0].shape[0] - 1, info_ls[0].shape[0])
    y_index = np.linspace(0, info_ls[0].shape[1] - 1, info_ls[0].shape[1])
    z_index = np.linspace(0, info_ls[0].shape[2] - 1, info_ls[0].shape[2])
    f = RegularGridInterpolator((x_index, y_index, z_index), info_ls[0], method="nearest", bounds_error=False,
                                fill_value=0)
    # find the landmarks and define new affine matrix based on the cardiac coordinate space
    landmarks_list_voxel_space, landmarks_patient_space, new_space, max_distance, new_affine_mtx = landmark_gen(
        nifti_path_each, output_size=160)
    # define the sample points based on the affine matrix from the CT data and cardiac coordinate space
    ijk_index = (
        np.array(np.meshgrid(np.arange(0, output_size), np.arange(0, output_size), np.arange(0, output_size),
                             indexing='ij')).T.reshape(output_size, output_size, output_size, 3))
    ijk_mtx = np.zeros((output_size, output_size, output_size, 4))
    ijk_mtx[:, :, :, :3] = ijk_index
    ijk_mtx[:, :, :, -1] = 1
    # each entry for ijk_mtx is in (i, j, 0, 1) format
    ijk_mtx = ijk_mtx.T.reshape(4, output_size ** 3)
    affine_mtx = info_ls[1]
    ijk_index_in_original = np.dot(np.dot(np.linalg.inv(affine_mtx), new_affine_mtx), ijk_mtx)
    sample_points = []
    for j in range(ijk_index_in_original.shape[-1]):
        sample_points.append(ijk_index_in_original[:3, j])

    # interpolation result for all sample points
    fp = f(sample_points)
    # rearrange into a 3D array
    for kx in range(output_size):
        for ky in range(output_size):
            for kz in range(output_size):
                final_3d_dense_vol[kx, ky, kz] = fp[kx * output_size * output_size + ky * output_size + kz]

    # save the resampled data under new affine
    label_nifti = nib.Nifti1Image(final_3d_dense_vol, affine=new_affine_mtx)
    nib.save(label_nifti, os.path.join(output_path_new_aff, data_list[d][:-7] + '_cardiac_coordinate_space_new_affine.nii.gz'))
    # save the resampled data with identity matrix
    # it should match with the defined cardiac coordinate system if we load into 3DSlicer for the visualisation
    label_nifti = nib.Nifti1Image(final_3d_dense_vol, affine=np.eye(4))
    nib.save(label_nifti,
             os.path.join(output_path_id_aff, data_list[d][:-7] + '_cardiac_coordinate_space_id_affine.nii.gz'))