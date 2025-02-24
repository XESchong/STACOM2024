import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import os
import nibabel as nib
import edt
import torch.nn as nn
import random

def zero_mean_unit_var(image, mask=None, fill_value=0):
    """Normalizes an image to zero mean and unit variance."""

    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array.astype(np.float32)

    msk_array = np.ones(img_array.shape)

    if mask is not None:
        msk_array = sitk.GetArrayFromImage(mask)

    mean = np.mean(img_array[msk_array>0])
    std = np.std(img_array[msk_array>0])

    if std > 0:
        img_array = (img_array - mean) / std
        img_array[msk_array==0] = fill_value

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised


def range_matching(image, mask=None, low_percentile=4, high_percentile=96, fill_value=0):
    """Normalizes an image by mapping the low_percentile to zero, and the high_percentile to one."""

    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array.astype(np.float32)

    msk_array = np.ones(img_array.shape)

    if mask is not None:
        msk_array = sitk.GetArrayFromImage(mask)

    lo_p = np.percentile(img_array[msk_array>0], low_percentile)
    hi_p = np.percentile(img_array[msk_array>0], high_percentile)

    img_array = (img_array - lo_p) / (hi_p - lo_p)
    img_array[msk_array == 0] = fill_value

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised


def zero_one(image, mask=None, fill_value=0):
    """Normalizes an image by mapping the min to zero, and max to one."""

    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array.astype(np.float32)

    msk_array = np.ones(img_array.shape)

    if mask is not None:
        msk_array = sitk.GetArrayFromImage(mask)

    min_value = np.min(img_array[msk_array>0])
    max_value = np.max(img_array[msk_array>0])

    img_array = (img_array - min_value) / (max_value - min_value)
    img_array[msk_array == 0] = fill_value

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised


def threshold_zero(image, mask=None, fill_value=0):
    """Thresholds an image at zero."""

    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array > 0
    img_array = img_array.astype(np.float32)

    msk_array = np.ones(img_array.shape)

    if mask is not None:
        msk_array = sitk.GetArrayFromImage(mask)

    img_array[msk_array == 0] = fill_value

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised


def threshold_mid(image, mask=None, fill_value=0):
    """Thresholds an image at mid point."""

    img_array = sitk.GetArrayFromImage(image)

    msk_array = np.ones(img_array.shape)

    if mask is not None:
        msk_array = sitk.GetArrayFromImage(mask)

    min_value = np.min(img_array[msk_array>0])
    max_value = np.max(img_array[msk_array>0])

    img_array = img_array > ((max_value - min_value) / 2.0)
    img_array = img_array.astype(np.float32)

    img_array[msk_array == 0] = fill_value

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised


def same_image_domain(image1, image2):
    """Checks whether two images cover the same physical domain."""

    same_size = image1.GetSize() == image2.GetSize()
    same_spacing = image1.GetSpacing() == image2.GetSpacing()
    same_origin = image1.GetOrigin() == image2.GetOrigin()
    same_direction = image1.GetDirection() == image2.GetDirection()

    return same_size and same_spacing and same_origin and same_direction


def reorient_image(image):
    """Reorients an image to standard radiology view."""

    dir = np.array(image.GetDirection()).reshape(len(image.GetSize()), -1)
    ind = np.argmax(np.abs(dir), axis=0)
    new_size = np.array(image.GetSize())[ind]
    new_spacing = np.array(image.GetSpacing())[ind]
    new_extent = new_size * new_spacing
    new_dir = dir[:, ind]

    flip = np.diag(new_dir) < 0
    flip_diag = flip * -1
    flip_diag[flip_diag == 0] = 1
    flip_mat = np.diag(flip_diag)

    new_origin = np.array(image.GetOrigin()) + np.matmul(new_dir, (new_extent * flip))
    new_dir = np.matmul(new_dir, flip_mat)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing.tolist())
    resample.SetSize(new_size.tolist())
    resample.SetOutputDirection(new_dir.flatten().tolist())
    resample.SetOutputOrigin(new_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    return resample.Execute(image)


def resample_image_to_ref(image, ref, is_label=False, pad_value=0):
    """Resamples an image to match the resolution and size of a given reference image."""

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref)
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(image)


def resample_image(image, out_spacing=(1.0, 1.0, 1.0), out_size=None, is_label=False, pad_value=0):
    """Resamples an image to given element spacing and output size."""

    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    if out_size is None:
        out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
    else:
        out_size = np.array(out_size)

    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
    original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)

    original_center = np.matmul(original_direction, original_center)
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(image.GetOrigin()) + (original_center - out_center)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        return resample.Execute(image)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
        if image.GetNumberOfComponentsPerPixel() == 1:
            return resample.Execute(sitk.Cast(image, sitk.sitkFloat32))
        else:
            return resample.Execute(sitk.Cast(image, sitk.sitkVectorFloat32))


def extract_patch(image, pixel, out_spacing=(1.0, 1.0, 1.0), out_size=(32, 32, 32), is_label=False, pad_value=0):
    """Extracts a patch of given resolution and size at a specific location."""

    original_spacing = np.array(image.GetSpacing())

    out_size = np.array(out_size)

    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)

    pos = np.matmul(original_direction, np.array(pixel) * np.array(original_spacing)) + np.array(image.GetOrigin())
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(pos - out_center)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    if image.GetNumberOfComponentsPerPixel() == 1:
        return resample.Execute(sitk.Cast(image, sitk.sitkFloat32))
    else:
        return resample.Execute(sitk.Cast(image, sitk.sitkVectorFloat32))



class diceloss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(diceloss, self).__init__()

    def forward(self, gt, pred, smooth=0.01):

        # flatten label and prediction tensors
        dice_loss_batch = 0
        for i in range(gt.size()[0]):
            dice_loss_channel = 0
            for j in range(1, 6):
                inputs = gt[i, j, ...].contiguous().view(-1)
                targets = pred[i, j, ...].contiguous().view(-1)
                intersection = (inputs * targets).sum()
                dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
                dice_loss_channel += 1 - dice

            mean_dice_loss_channel = dice_loss_channel / 5
            dice_loss_batch += mean_dice_loss_channel
        mean_dice_loss_batch = dice_loss_batch / gt.size()[0]

        return mean_dice_loss_batch





def one_hot_labelmap_with_mask(labelmap, smoothing_sigma=0, sdm_option= True, sparse_option = True, motion_option = False, file_name = None):
    """Converts a single channel labelmap to a one-hot labelmap."""

    slice_dir = '/media/yx22/DATA/atlas-istn-main/data/Largre_scale_1700_cases_DICOM_NIFTY/mask_dicom_motion_corrected_new'
    # one-hot label map
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


    if sdm_option:
        if sparse_option:
            # aug = np.random.randint(10)
            # slice_file = os.path.join(slice_dir, str(aug), file_name[:9])
            slice_file = os.path.join(slice_dir, file_name[:9])
            slice_list = os.listdir(slice_file)
            SDM_final = np.zeros(lab_array_one_hot.shape)
            #one_hot_final = np.zeros(lab_array_one_hot.shape)
            for k in range(len(slice_list)):
                #print(k)
                if 'ch5' not in slice_list[k] and 'ch6' not in slice_list[k] and 'ch7' not in slice_list[k]:
                    mask = sitk.ReadImage(os.path.join(slice_file, slice_list[k]), sitk.sitkInt64)
                    current_slice_data = sitk.GetArrayFromImage(mask)
                    mask_repeat = np.repeat(current_slice_data[:, :, :, np.newaxis], 6, axis=3)
                else:
                    continue
                #print(k)


                if motion_option:
                    # introduce motion artifact
                    lab_array_one_hot_motion = np.zeros(lab_array_one_hot.shape)
                    uni_ax_motion_mean = 0  # from UKBioBank Slice Shift estimation
                    uni_ax_motion_std = 3.5  # from UKBioBank Slice Shift estimation

# 1.13 & 1.17
                    fx = random.gauss(0, 1)
                    rx = np.round((fx / np.linalg.norm(fx)) * random.gauss(uni_ax_motion_mean, uni_ax_motion_std)).astype(int)
                    fy = random.gauss(0, 1)
                    ry = np.round((fy / np.linalg.norm(fy)) * random.gauss(uni_ax_motion_mean, uni_ax_motion_std)).astype(int)
                    fz = random.gauss(0, 1)
                    rz = np.round((fz / np.linalg.norm(fz)) * random.gauss(uni_ax_motion_mean, uni_ax_motion_std)).astype(int)
                    lab_array_one_hot_motion[..., 0] = motion_artifact_sym(lab_array_one_hot[..., 0], rx, ry, rz, background=True)
                    for i in range(1, 6):
                        lab_array_one_hot_motion[..., i] = motion_artifact_sym(lab_array_one_hot[..., i], rx, ry, rz, background=False)
                else:
                    lab_array_one_hot_motion = lab_array_one_hot



                SDM_temp_1 = 1 - mask_repeat
                one_hot_mask = np.multiply(lab_array_one_hot_motion, mask_repeat)
                SDM_temp_1[SDM_temp_1 == 0] = one_hot_mask[SDM_temp_1 == 0]

                SDM_inside = np.zeros(lab_array_one_hot_motion.shape)
                SDM_inside[:, :, :, 0] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 0])
                SDM_inside[:, :, :, 1] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 1])
                SDM_inside[:, :, :, 2] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 2])
                SDM_inside[:, :, :, 3] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 3])
                SDM_inside[:, :, :, 4] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 4])
                SDM_inside[:, :, :, 5] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 5])

                SDM_temp_2 = 1 - mask_repeat
                SDM_temp_2[SDM_temp_2 == 0] = (1 - one_hot_mask)[SDM_temp_2 == 0]
                SDM_outside = np.zeros(lab_array_one_hot_motion.shape)
                SDM_outside[:, :, :, 0] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 0])
                SDM_outside[:, :, :, 1] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 1])
                SDM_outside[:, :, :, 2] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 2])
                SDM_outside[:, :, :, 3] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 3])
                SDM_outside[:, :, :, 4] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 4])
                SDM_outside[:, :, :, 5] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 5])

                SDM_final_inside = np.multiply(SDM_inside, mask_repeat) * (-1)
                SDM_final_outside = np.multiply(SDM_outside, mask_repeat)
                SDM_final_slice = SDM_final_inside + SDM_final_outside
                SDM_final_slice[SDM_final_slice > 2] = 2
                SDM_final_slice[SDM_final_slice < -2] = -2
                SDM_final[SDM_final_slice != 0] = SDM_final_slice[SDM_final_slice != 0]
                #one_hot_final[one_hot_mask != 0] = one_hot_mask[one_hot_mask != 0]

            labelmap_one_hot = sitk.GetImageFromArray(SDM_final, isVector=True)
            #labelmap_one_hot = sitk.GetImageFromArray(one_hot_final, isVector=True)
            labelmap_one_hot.CopyInformation(labelmap)
            # sitk.WriteImage(labelmap_one_hot, os.path.join('/media/yx22/DATA/atlas-istn-main/data/1700_cases_synthetic_data/SDM_representation_shifted', file_name))
        else:
            temp = np.zeros(lab_array_one_hot.shape)
            for i in range(6):
                temp[:, :, :, i] = edt.sdf(lab_array_one_hot[:, :, :, i]) * (-1)
                temp[temp > 2] = 2
                temp[temp < -2] = -2

            labelmap_one_hot = sitk.GetImageFromArray(temp, isVector=True)
            labelmap_one_hot.CopyInformation(labelmap)
            # sitk.WriteImage(labelmap_one_hot, os.path.join('/media/yx22/DATA/atlas-istn-main/data/1700_cases_synthetic_data/SDM_atlas_dense', file_name))


    else:
        # print('yes')
        # print('haha')
        labelmap_one_hot = sitk.GetImageFromArray(lab_array_one_hot, isVector=True)
        labelmap_one_hot.CopyInformation(labelmap)
        sitk.WriteImage(labelmap_one_hot, os.path.join('/media/yx22/DATA/atlas-istn-main/data/reorientated_MRI_with_motion_artefact_large_scale/atlas_oh', file_name))


    return labelmap_one_hot



def one_hot_labelmap(labelmap, smoothing_sigma=0, sdm_option= True, sparse_option = True, motion_option = False, file_name = None):
    # for syn-2 only!!!!!!!!!because all data share same mask information (basically indexing with i,j,k)
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

    L = sitk.GetImageFromArray(lab_array_one_hot, isVector=True)
    # labelmap_one_hot = sitk.GetImageFromArray(one_hot_final, isVector=True)
    L.CopyInformation(labelmap)
    # sitk.WriteImage(L, os.path.join('/media/yx22/DATA/atlas-istn-main/data/my_synthetic_MRI_datasetset/one_hot_representation/all_data', file_name))


    if sdm_option:
        if sparse_option:
            slice_file = '/media/yx22/DATA/atlas-istn-main/output/Slice_shifting_results/results/shifted_result_for_later_comparison/fair_comparison(for creating shifted Syn_2)/mask_with_thickness'
            slice_list = os.listdir(slice_file)
            SDM_final = np.zeros(lab_array_one_hot.shape)
            #one_hot_final = np.zeros(lab_array_one_hot.shape)
            for k in range(len(slice_list)):
                #print(k)
                if 'ch5' not in slice_list[k] and 'ch6' not in slice_list[k] and 'ch7' not in slice_list[k]:
                    mask = sitk.ReadImage(os.path.join(slice_file, slice_list[k]), sitk.sitkInt64)
                    current_slice_data = sitk.GetArrayFromImage(mask)
                    mask_repeat = np.repeat(current_slice_data[:, :, :, np.newaxis], 6, axis=3)
                else:
                    continue
                #print(k)

                # motion_option is always false in this case
                if motion_option:
                    # introduce motion artifact
                    lab_array_one_hot_motion = np.zeros(lab_array_one_hot.shape)
                    uni_ax_motion_mean = 0  # from UKBioBank Slice Shift estimation
                    uni_ax_motion_std = 3.5  # from UKBioBank Slice Shift estimation

# 1.13 & 1.17
                    fx = random.gauss(0, 1)
                    rx = np.round((fx / np.linalg.norm(fx)) * random.gauss(uni_ax_motion_mean, uni_ax_motion_std)).astype(int)
                    fy = random.gauss(0, 1)
                    ry = np.round((fy / np.linalg.norm(fy)) * random.gauss(uni_ax_motion_mean, uni_ax_motion_std)).astype(int)
                    fz = random.gauss(0, 1)
                    rz = np.round((fz / np.linalg.norm(fz)) * random.gauss(uni_ax_motion_mean, uni_ax_motion_std)).astype(int)
                    lab_array_one_hot_motion[..., 0] = motion_artifact_sym(lab_array_one_hot[..., 0], rx, ry, rz, background=True)
                    for i in range(1, 6):
                        lab_array_one_hot_motion[..., i] = motion_artifact_sym(lab_array_one_hot[..., i], rx, ry, rz, background=False)
                else:
                    lab_array_one_hot_motion = lab_array_one_hot



                SDM_temp_1 = 1 - mask_repeat
                one_hot_mask = np.multiply(lab_array_one_hot_motion, mask_repeat)
                SDM_temp_1[SDM_temp_1 == 0] = one_hot_mask[SDM_temp_1 == 0]

                SDM_inside = np.zeros(lab_array_one_hot_motion.shape)
                SDM_inside[:, :, :, 0] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 0])
                SDM_inside[:, :, :, 1] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 1])
                SDM_inside[:, :, :, 2] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 2])
                SDM_inside[:, :, :, 3] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 3])
                SDM_inside[:, :, :, 4] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 4])
                SDM_inside[:, :, :, 5] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 5])

                SDM_temp_2 = 1 - mask_repeat
                SDM_temp_2[SDM_temp_2 == 0] = (1 - one_hot_mask)[SDM_temp_2 == 0]
                SDM_outside = np.zeros(lab_array_one_hot_motion.shape)
                SDM_outside[:, :, :, 0] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 0])
                SDM_outside[:, :, :, 1] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 1])
                SDM_outside[:, :, :, 2] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 2])
                SDM_outside[:, :, :, 3] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 3])
                SDM_outside[:, :, :, 4] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 4])
                SDM_outside[:, :, :, 5] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 5])

                SDM_final_inside = np.multiply(SDM_inside, mask_repeat) * (-1)
                SDM_final_outside = np.multiply(SDM_outside, mask_repeat)
                SDM_final_slice = SDM_final_inside + SDM_final_outside
                SDM_final_slice[SDM_final_slice > 2] = 2
                SDM_final_slice[SDM_final_slice < -2] = -2
                SDM_final[SDM_final_slice != 0] = SDM_final_slice[SDM_final_slice != 0]
                #one_hot_final[one_hot_mask != 0] = one_hot_mask[one_hot_mask != 0]

            labelmap_one_hot = sitk.GetImageFromArray(SDM_final, isVector=True)
            #labelmap_one_hot = sitk.GetImageFromArray(one_hot_final, isVector=True)
            labelmap_one_hot.CopyInformation(labelmap)
            # sitk.WriteImage(labelmap_one_hot, os.path.join('/media/yx22/DATA/atlas-istn-main/data/my_synthetic_MRI_datasetset/SDM_representation/data_motion_artefact', file_name))
        else:
            temp = np.zeros(lab_array_one_hot.shape)
            for i in range(6):
                temp[:, :, :, i] = edt.sdf(lab_array_one_hot[:, :, :, i]) * (-1)
                temp[temp > 2] = 2
                temp[temp < -2] = -2

            labelmap_one_hot = sitk.GetImageFromArray(temp, isVector=True)
            labelmap_one_hot.CopyInformation(labelmap)
            # sitk.WriteImage(labelmap_one_hot, os.path.join('/media/yx22/DATA/atlas-istn-main/data/my_synthetic_MRI_datasetset/SDM_representation/data_dense', file_name))


    else:
        labelmap_one_hot = sitk.GetImageFromArray(lab_array_one_hot, isVector=True)
        labelmap_one_hot.CopyInformation(labelmap)


    return labelmap_one_hot





def lax_plane_jit(kf_img, xi, yi, zi):
    kf_img_pad = np.pad(kf_img, 1, 'constant', constant_values=0)
    kf_img_mod = kf_img_pad[xi:xi + 160, yi:yi + 160, zi:zi + 160]
    return kf_img_mod


def motion_artifact_sym(lab_in, rx, ry, rz, background=False):
    fov = 160
    # motion artifact
    # print([rx, ry, rz])
    if background:
        lab_out = np.ones(np.shape(lab_in))
    else:
        lab_out = np.zeros(np.shape(lab_in))
    if rx >= 0:
        if ry >= 0:
            if rz >= 0:
                lab_out[0:fov - rx, 0:fov - ry, 0:fov - rz] = lab_in[rx:fov, ry:fov, rz:fov]
            else:
                lab_out[0:fov - rx, 0:fov - ry, np.abs(rz):fov] = lab_in[rx:fov, ry:fov, 0:fov - np.abs(rz)]
        else:
            if rz >= 0:
                lab_out[0:fov - rx, np.abs(ry):fov, 0:fov - rz] = lab_in[rx:fov, 0:fov - np.abs(ry), rz:fov]
            else:
                lab_out[0:fov - rx, np.abs(ry):fov, np.abs(rz):fov] = lab_in[rx:fov, 0:fov - np.abs(ry), 0:fov - np.abs(rz)]
    else:
        if ry >= 0:
            if rz >= 0:
                lab_out[np.abs(rx):fov, 0:fov - ry, 0:fov - rz] = lab_in[0:fov - np.abs(rx), ry:fov, rz:fov]
            else:
                lab_out[np.abs(rx):fov, 0:fov - ry, np.abs(rz):fov] = lab_in[0:fov - np.abs(rx), ry:fov, 0:fov - np.abs(rz)]
        else:
            if rz >= 0:
                lab_out[np.abs(rx):fov, np.abs(ry):fov, 0:fov - rz] = lab_in[0:fov - np.abs(rx), 0:fov - np.abs(ry), rz:fov]
            else:
                lab_out[np.abs(rx):fov, np.abs(ry):fov, np.abs(rz):fov] = lab_in[0:fov - np.abs(rx), 0:fov - np.abs(ry), 0:fov - np.abs(rz)]
    return lab_out




'''

                # introduce motion artifact
                lab_array_one_hot_motion = np.zeros(lab_array_one_hot.shape)
                uni_ax_motion_mean = 0  # from UKBioBank Slice Shift estimation
                uni_ax_motion_std = 3.5  # from UKBioBank Slice Shift estimation
                fx = random.gauss(0, 1)
                rx = np.round((fx / np.linalg.norm(fx)) * random.gauss(uni_ax_motion_mean, uni_ax_motion_std)).astype(int)
                fy = random.gauss(0, 1)
                ry = np.round((fy / np.linalg.norm(fy)) * random.gauss(uni_ax_motion_mean, uni_ax_motion_std)).astype(int)
                fz = random.gauss(0, 1)
                rz = np.round((fz / np.linalg.norm(fz)) * random.gauss(uni_ax_motion_mean, uni_ax_motion_std)).astype(int)
                lab_array_one_hot_motion[..., 0] = motion_artifact_sym(lab_array_one_hot[..., 0], rx, ry, rz, background= True)
                for i in range(1,6):
                    lab_array_one_hot_motion[..., i] = motion_artifact_sym(lab_array_one_hot[..., i], rx, ry, rz, background= False)
'''



'''
                SDM_motion = np.zeros(SDM_final_slice.shape)
                SDM_motion_final = np.zeros(SDM_final_slice.shape)

                uni_ax_motion_mean = 0  # from UKBioBank Slice Shift estimation
                uni_ax_motion_std = 3.5  # from UKBioBank Slice Shift estimation
                fx = random.gauss(0, 1)
                rx = np.round((fx / np.linalg.norm(fx)) * random.gauss(uni_ax_motion_mean, uni_ax_motion_std)).astype(
                    int)
                fy = random.gauss(0, 1)
                ry = np.round((fy / np.linalg.norm(fy)) * random.gauss(uni_ax_motion_mean, uni_ax_motion_std)).astype(
                    int)
                fz = random.gauss(0, 1)
                rz = np.round((fz / np.linalg.norm(fz)) * random.gauss(uni_ax_motion_mean, uni_ax_motion_std)).astype(
                    int)
                if 'ch' in slice_list[k]:
                    xi = np.random.choice([0, 1, 2])
                    yi = np.random.choice([0, 1, 2])
                    zi = np.random.choice([0, 1, 2])
                    for i in range(6):
                        SDM_motion[..., i] = lax_plane_jit(SDM_final_slice[..., i], xi, yi, zi)
                        SDM_motion_final[..., i] = np.multiply(SDM_motion[..., i], motion_artifact_sym(SDM_final_slice[..., i], rx, ry, rz))
                    SDM_final[SDM_motion_final != 0] = SDM_motion_final[SDM_motion_final != 0]
                else:
                    for i in range(6):
                        SDM_motion_final[..., i] = np.multiply(SDM_final_slice[..., i], motion_artifact_sym(SDM_final_slice[..., i], rx, ry, rz))
                    SDM_final[SDM_motion_final != 0] = SDM_motion_final[SDM_motion_final != 0]
'''

'''
scipy SDM implementation for the sparse and dense one-hot labelmap
    if sdm_option:
        if sparse_option:
            aug = np.random.randint(10)
            slice_file = os.path.join(slice_dir, str(aug), file_name[:9])
            slice_list = os.listdir(slice_file)
            SDM_final = np.zeros(lab_array_one_hot.shape)
            for k in range(len(slice_list)):
                current_slice_mask = nib.load(os.path.join(slice_file, slice_list[k]))
                current_slice_data = current_slice_mask.get_fdata()
                mask_repeat = np.repeat(current_slice_data[:, :, :, np.newaxis], 6, axis=3)
                SDM_temp_1 = 1 - mask_repeat
                one_hot_mask = np.multiply(lab_array_one_hot, mask_repeat)
                SDM_temp_1[SDM_temp_1 == 0] = one_hot_mask[SDM_temp_1 == 0]
                SDM_inside = np.zeros(lab_array_one_hot.shape)
                SDM_inside[:, :, :, 0] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 0])
                SDM_inside[:, :, :, 1] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 1])
                SDM_inside[:, :, :, 2] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 2])
                SDM_inside[:, :, :, 3] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 3])
                SDM_inside[:, :, :, 4] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 4])
                SDM_inside[:, :, :, 5] = ndimage.distance_transform_edt(SDM_temp_1[:, :, :, 5])

                SDM_temp_2 = 1 - mask_repeat
                SDM_temp_2[SDM_temp_2 == 0] = (1 - one_hot_mask)[SDM_temp_2 == 0]
                SDM_outside = np.zeros(lab_array_one_hot.shape)
                SDM_outside[:, :, :, 0] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 0])
                SDM_outside[:, :, :, 1] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 1])
                SDM_outside[:, :, :, 2] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 2])
                SDM_outside[:, :, :, 3] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 3])
                SDM_outside[:, :, :, 4] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 4])
                SDM_outside[:, :, :, 5] = ndimage.distance_transform_edt(SDM_temp_2[:, :, :, 5])

                SDM_final_inside = np.multiply(SDM_inside, mask_repeat) * (-1)
                SDM_final_outside = np.multiply(SDM_outside, mask_repeat)
                SDM_final_slice = SDM_final_inside + SDM_final_outside
                SDM_final[SDM_final > 2] = 2
                SDM_final[SDM_final < -2] = -2
                SDM_final[SDM_final_slice != 0] = SDM_final_slice[SDM_final_slice != 0]
            labelmap_one_hot = sitk.GetImageFromArray(SDM_final, isVector=True)
            labelmap_one_hot.CopyInformation(labelmap)
        else:
            temp_1 = np.zeros(lab_array_one_hot.shape)
            temp_2 = np.zeros(lab_array_one_hot.shape)
            temp_3 = np.zeros(lab_array_one_hot.shape)
            for i in range(6):
                # temp_1[:, :, :, i] = ndimage.distance_transform_edt(lab_array_one_hot[:, :, :, i])
                # temp_1[temp_1 >6] = 6
                temp_1[:, :, :, i] = ndimage.distance_transform_edt(lab_array_one_hot[:, :, :, i]) * (-1)
                temp_2[:, :, :, i] = ndimage.distance_transform_edt(1 - lab_array_one_hot[:, :, :, i])
                temp_3[:, :, :, i] = temp_1[:, :, :, i] + temp_2[:, :, :, i]
                temp_3[temp_3 > 2] = 2
                temp_3[temp_3 < -2] = -2


            labelmap_one_hot = sitk.GetImageFromArray(temp_3, isVector=True)
            labelmap_one_hot.CopyInformation(labelmap)


'''

'''
edt SDM implementation for the sparse and dense one-hot labelmap 

    
'''


'''
# SDM implementation for the sparse one-hot labelmap 
    if sdm_option:
        aug = np.random.randint(10)
        slice_file = os.path.join(slice_dir, str(aug), file_name[:9])
        slice_list = os.listdir(slice_file)
        SDM_final = np.zeros(lab_array_one_hot.shape)
        for k in range(len(slice_list)):
            current_slice_mask = nib.load(os.path.join(slice_file, slice_list[k]))
            current_slice_data = current_slice_mask.get_fdata()
            mask_repeat = np.repeat(current_slice_data[:,:,:,np.newaxis], 6, axis=3)
            SDM_temp = np.ones(lab_array_one_hot.shape)
            SDM_temp[np.multiply(lab_array_one_hot, (1 - mask_repeat)) == 0] = 0
            SDM_temp1 = ndimage.distance_transform_edt(SDM_temp)
            SDM_temp2 = ndimage.distance_transform_edt(1 - SDM_temp)
            SDM_final1 = np.multiply(SDM_temp1, mask_repeat) * (-1)
            SDM_fianl2 = np.multiply(SDM_temp2, mask_repeat)
            SDM_final_slice = SDM_final1 + SDM_fianl2
            # SDM_final[SDM_final > 6] = 6
            # SDM_final[SDM_final < -6] = -6
            SDM_final[SDM_final_slice != 0] = SDM_final_slice[SDM_final_slice!= 0]
        labelmap_one_hot = sitk.GetImageFromArray(SDM_final, isVector=True)
        labelmap_one_hot.CopyInformation(labelmap)
'''


'''
# SDM implementation for the dense one-hot labelmap 
        # Sign distance labelmap added 20-Feb-2023
        temp_1 = np.zeros(lab_array_one_hot.shape)
        temp_2 = np.zeros(lab_array_one_hot.shape)
        temp_3 = np.zeros(lab_array_one_hot.shape)
        for i in range(6):
            #temp_1[:, :, :, i] = ndimage.distance_transform_edt(lab_array_one_hot[:, :, :, i])
            #temp_1[temp_1 >6] = 6
            temp_1[:, :, :, i] = ndimage.distance_transform_edt(lab_array_one_hot[:, :, :, i]) * (-1)
            temp_2[:, :, :, i] = ndimage.distance_transform_edt(1 - lab_array_one_hot[:, :, :, i])
            temp_3[:, :, :, i] = temp_1[:, :, :, i] + temp_2[:, :, :, i]
            temp_3[temp_3 > 6] = 6
            temp_3[temp_3 < -6] = -6

        mask_path = '/media/yx22/DATA/atlas-istn-main/data/reorientated_MRI_with_motion_artifact/mask_matrix'
        mask_data = nib.load(os.path.join(mask_path, file_name[:9] + '_mask_matrix.nii.gz'))
        mask = mask_data.get_fdata()
        mask_repeat = np.repeat(mask[:, :, :, np.newaxis], 6, axis=3)
        final_sparse_input = np.multiply(temp_3, mask_repeat)



        labelmap_one_hot = sitk.GetImageFromArray(final_sparse_input, isVector=True)
        #labelmap_one_hot = sitk.GetImageFromArray(temp_3, isVector=True)
        labelmap_one_hot.CopyInformation(labelmap)
'''