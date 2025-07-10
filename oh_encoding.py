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

























