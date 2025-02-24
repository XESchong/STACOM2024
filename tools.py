##########################################################################################
# resample all dicom files into a 3D sparse volume
##########################################################################################
from SSA import extract_info
import nibabel as nib
from tqdm import tqdm
from datetime import datetime
import numpy as np
import os
import torch

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

#######################################################################################
# Modify here
data_path = '/media/yx22/DATA/STACOM_code/toy_data_after_SSA'
sparse_volume_path = '/media/yx22/DATA/STACOM_code/toy_data_sparse_volume_after_SSA'
#######################################################################################

data_list = sorted(os.listdir(data_path))
# remember sort the list if you needed
for d in tqdm(range(len(data_list))):
    data_motion_path = os.path.join(data_path, data_list[d])
    dicom_files = os.listdir(data_motion_path)
    final_3d_sparse_vol = np.zeros((160, 160, 160))

    for i in range(len(dicom_files)):
        # if you need to filter out some slices
        if '5ch' not in dicom_files[i]:
            dicom_path_each = os.path.join(data_motion_path, dicom_files[i])
            dicom_info_list = extract_info(dicom_path_each, True)
            data_array = dicom_info_list[0]
            data_affine = dicom_info_list[1]

            thickness = 3
            data_with_thickness = np.zeros((thickness, data_array.shape[1], data_array.shape[0]))
            data_with_thickness[:, ...] = data_array
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

            affine_inter = np.linalg.inv(data_affine)
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

    label_nifti = nib.Nifti1Image(np.transpose(final_3d_sparse_vol), affine=np.eye(4))
    nib.save(label_nifti, os.path.join(sparse_volume_path, data_list[d] + '.nii.gz'))


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)




########################################################################################################################
# one-hot encoding offline
########################################################################################################################
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

def one_hot_labelmap_with_mask(labelmap, smoothing_sigma=0, file_name = None, save_path = None):
    """Converts a single channel labelmap to a one-hot labelmap."""

    lab_array = sitk.GetArrayFromImage(labelmap)
    labels = np.unique(lab_array)
    labels.sort()

    labelmap_size = list(labelmap.GetSize()[::-1])
    labelmap_size.append(labels.size)

    lab_array_one_hot = np.zeros(labelmap_size).astype(float)
    for idx, lab in enumerate(labels):
        if smoothing_sigma > 0:
            lab_array_one_hot[..., idx] = gaussian_filter((lab_array == lab).astype(float), sigma=smoothing_sigma, mode='nearest')
        else:
            lab_array_one_hot[..., idx] = lab_array == lab

    labelmap_one_hot = sitk.GetImageFromArray(lab_array_one_hot, isVector=True)
    labelmap_one_hot.CopyInformation(labelmap)
    sitk.WriteImage(labelmap_one_hot, os.path.join(save_path, file_name))


    return labelmap_one_hot


# path for each data
csv_path = '/media/yx22/DATA/SSA_LTN_DSTN_pack/3d_sparse.csv'
# saving path the data after one-hot encoding
output_path = '/media/yx22/DATA/SSA_LTN_DSTN_pack/toy_data_sparse_volume_oh'
label_data_path = pd.read_csv(csv_path)

for i in tqdm(range(len(label_data_path))):
    img_fname = os.path.basename(label_data_path.iloc[i, 0])
    labelmap = sitk.ReadImage(label_data_path.iloc[i, 0], sitk.sitkInt64)
    labelmap.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    labelmap_new = sitk.Cast(one_hot_labelmap_with_mask(labelmap, smoothing_sigma=0, file_name=img_fname, save_path = output_path), sitk.sitkVectorFloat32)

























