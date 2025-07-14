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
