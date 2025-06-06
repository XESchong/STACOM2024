########################################################################
# synthetic DICOM data generation
########################################################################
import pydicom
from scipy.interpolate import RegularGridInterpolator
from SSA import apply_affine_transformation_data
from skimage.measure import label
import scipy
from scipy.spatial import distance
from datetime import datetime
import nibabel as nib
import numpy as np
import random
from scipy.ndimage import rotate
import os
from tqdm import tqdm

def getLargestCC(segmentation = None):
    """
    remove the isolated island and find the largest connected component for each foreground label mask
    """
    labels = label(segmentation)
    # assume at least 1 connected component
    assert( labels.max() != 0 )
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def gen_dicom(path = None, path_rotated = None, output_path = None, dicom_template_path = None):

    """
    generate dicom file from dense data

    :path (str): the path of dense data
    :path_rotated (str): the path of the folder containing rotated dense datas
    :output_path (str): saving path for the synthetic dicom files
    :dicom_template_path (str): dicom template from Sunnybrook Cardiac Data (SCD)


    :return: synthetic dicom files (2CH LAX, 3CH LAX , 4CH LAX and SAXs)
    """

    rotation_data_path = os.path.join(path_rotated, os.path.basename(path))
    dense_label_rotated = nib.load(rotation_data_path).get_fdata()
    dense_label = nib.load(path).get_fdata()
    # label format: LVM-1, LV-2, RV-3, RA-4, LA-5
    seg_LV_tmp = (dense_label == 2).astype(dense_label.dtype)
    # clear unwanted segmentation (find the largest connected component)
    seg_LV = getLargestCC(seg_LV_tmp).astype(dense_label.dtype)
    seg_LA_tmp = (dense_label == 5).astype(dense_label.dtype)
    seg_LA = getLargestCC(seg_LA_tmp).astype(dense_label.dtype)
    seg_RV_tmp = (dense_label == 3).astype(dense_label.dtype)
    seg_RV = getLargestCC(seg_RV_tmp).astype(dense_label.dtype)
    seg_RA_tmp = (dense_label == 4).astype(dense_label.dtype)
    seg_RA = getLargestCC(seg_RA_tmp).astype(dense_label.dtype)
    # binary label for the whole heart
    seg_binary_tmp = (dense_label != 0).astype(dense_label.dtype)
    seg_binary = getLargestCC(seg_binary_tmp).astype(dense_label.dtype)

    # mitral valve center
    seg_LA_dilation = scipy.ndimage.binary_dilation(seg_LA).astype(dense_label.dtype)
    mvc = np.mean(np.where(np.multiply(seg_LV, seg_LA_dilation) == 1), axis=1)
    # tricuspid valve center
    seg_RA_dilation = scipy.ndimage.binary_dilation(seg_RA).astype(dense_label.dtype)
    tvc = np.mean(np.where(np.multiply(seg_RV, seg_RA_dilation) == 1), axis=1)
    # apex
    LV_points = np.where(seg_LV == 1)
    max_d = float('-inf')
    apex = None
    for i in range(len(LV_points[0])):
        if len(dense_label.shape) == 3:
            tmp_point = np.array([LV_points[0][i], LV_points[1][i], LV_points[2][i]])
        else:
            tmp_point = np.array([LV_points[0][i], LV_points[1][i]])

        tmp_distance = distance.euclidean(mvc, tmp_point)
        if tmp_distance > max_d:
            max_d = tmp_distance
            apex = tmp_point
    # center of the heart
    coh = np.mean(np.where(seg_binary == 1), axis=1)
    # centroid of LV and RV
    lvc = np.mean(np.where(seg_LV == 1), axis=1)
    rvc = np.mean(np.where(seg_RV == 1), axis=1)

    output_loc = os.path.join(output_path, os.path.basename(path)[:6])
    # for each dense label, create two types of synthetic dicom files, i.e., motion corrupted and motion free
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
        os.makedirs(output_loc + '/motion_corrupted')
        os.makedirs(output_loc + '/motion_free')

    # Get dicom header information from real CMR data
    # dicom_template_path = '/path/to/SCD_IMAGES_05/SCD0004001/CINELAX_301/IM-0003-0001.dcm'
    dataset = pydicom.dcmread(dicom_template_path)

    # Interpolation function
    xi = np.arange(0, 160, 1)
    yj = np.arange(0, 160, 1)
    zk = np.arange(0, 160, 1)
    f = RegularGridInterpolator((xi, yj, zk), dense_label_rotated, method="nearest", bounds_error=False, fill_value=0)

    ############################################################
    # 4CH-LAX, no rotation
    ############################################################
    # needed landmarks: apex, mvc, tvc, coh
    # based on the landmarks, we define the Image orientation patient (IOP) and image position patient (IPP) for 4CH-LAX
    vec_one = apex - mvc
    vec_two_tmp = tvc - mvc
    vec_three = np.cross(vec_one, vec_two_tmp)
    vec_two = np.cross(vec_three, vec_one)

    iop_one = vec_one / np.linalg.norm(vec_one)
    iop_two = vec_two/ np.linalg.norm(vec_two)

    plane_normal = np.cross(iop_one, iop_two)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    vector = coh - mvc
    distance_vec = np.dot(vector, plane_normal)
    projected_point = coh - distance_vec * plane_normal
    ipp = projected_point

    # generate all possible pixels inside this plane based on IOP and IPP with unit spacing
    points = []
    for x in np.arange(-80, 80, 1):
        for y in np.arange(-80, 80, 1):
            tmp_p = ipp + x * 1 * iop_one + y * 1 * iop_two
            points.append(tmp_p)

    # Interpolate values based on learnt interpolator
    fp = f(points)
    # resize into a (160, 160) plane
    img_4ch = np.zeros((160, 160))
    for kx in range(160):
        for ky in range(160):
            img_4ch[kx, ky] = fp[kx * 160 + ky]

    dicom_array = dataset.pixel_array
    gt = img_4ch.T

    # motion_free and motion_corrupted data saving (no motion simulation for 4CH)
    shift_pixel_1 = 0
    shift_pixel_2 = 0
    direction = np.array([shift_pixel_1, shift_pixel_2])
    motion_data = apply_affine_transformation_data(gt, direction)
    dicom_array = np.short(motion_data)
    dataset.ImagePositionPatient = list(points[0])
    dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
    dataset.PixelData = dicom_array.tobytes()
    # unit spacing
    dataset.PixelSpacing = [1, 1]
    dataset.Rows, dataset.Columns = dicom_array.shape
    dataset.SliceThickness = 1
    dataset.SpacingBetweenSlices = 1
    dataset.save_as(os.path.join(output_loc + '/motion_free', os.path.basename(path)[:6] + '_LAX_4ch_' + str(shift_pixel_1) + '_'
                                 + str(shift_pixel_2) + '.dcm'))
    dataset.save_as(os.path.join(output_loc + '/motion_corrupted', os.path.basename(path)[:6] + '_LAX_4ch_' + str(shift_pixel_1) + '_'
                                 + str(shift_pixel_2) + '.dcm'))

    ############################################################
    # 3CH-LAX, rotation degree range: [-3, 3]
    ############################################################
    # needed landmarks: apex, mvc, avc (aortic valve centroid), coh

    data_name = os.path.basename(path)
    # find the aortic valve centroid from dense data with more labels (including aorta)
    #################################################################################
    # REMOVE 3CH-LAX generation if you do not have dense data containing aorta label!
    ##################################################################################
    original_folder = '/path/to/folder/containing/dense/data/with/more/labels'
    cardiac_folder = '/path/to/folder/containing/dense/data/with/more/labels/under/cardiac/coordinate/space'
    ##################################################################################
    # Modify this part based on your data name
    ##################################################################################
    # original data means the dense data containing the aorta label (aorta label value: 6)
    original_data_name = data_name.replace('_cardiac_coordinate_space_id_affine', '')
    # cardiac data is the original data under defined cardiac coordinate space
    cardiac_data_name = data_name.replace('_id_affine', '')

    original_path = os.path.join(original_folder, original_data_name)
    cardiac_path = os.path.join(cardiac_folder, cardiac_data_name)
    original_data = nib.load(original_path).get_fdata()
    original_affine = nib.load(original_path).affine
    cardiac_affine = nib.load(cardiac_path).affine

    seg_LV_tmp_original = (original_data == 1).astype(original_data.dtype)
    seg_LV_original = getLargestCC(seg_LV_tmp_original).astype(original_data.dtype)
    seg_aorta_tmp_original = (original_data == 6).astype(original_data.dtype)
    seg_aorta_original = getLargestCC(seg_aorta_tmp_original).astype(original_data.dtype)
    seg_aorta_dilation = scipy.ndimage.binary_dilation(seg_aorta_original).astype(original_data.dtype)
    avc = np.mean(np.where(np.multiply(seg_LV_original, seg_aorta_dilation) == 1), axis=1)
    # since other landmarks are based on the cardiac coordinate space, we compute aortic valve centroid under the cardiac coordinate space
    avc_cardiac = np.dot(np.linalg.inv(cardiac_affine), np.dot(np.append(avc, 1), original_affine))[:3]
    # based on the landmarks, we define the Image orientation patient (IOP) and image position patient (IPP) for 3CH-LAX
    vec_one_tmp = apex - mvc
    vec_two = avc_cardiac - mvc
    vec_three = np.cross(vec_one_tmp, vec_two)
    vec_one = np.cross(vec_two, vec_three)

    iop_one = vec_one / np.linalg.norm(vec_one)
    iop_two = vec_two/ np.linalg.norm(vec_two)

    plane_normal = np.cross(iop_one, iop_two)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    vector = coh - mvc
    distance_vec = np.dot(vector, plane_normal)
    projected_point = coh - distance_vec * plane_normal
    ipp = projected_point

    # random rotation for 3CH-LAX
    theta = random.choice([-1, 1]) * (np.pi / random.randint(60, 180))
    # rotation matrix R_x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    # rotation matrix R_y
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])

    A_LAX = np.zeros((3, 3))
    third_col = np.cross(iop_one, iop_two)
    third_col = third_col/np.linalg.norm(third_col)
    A_LAX[:, 0] = iop_one
    A_LAX[:, 1] = iop_two
    A_LAX[:, 2] = third_col

    # 80% for rotation
    if random.randint(-1, 8) > 0:
        A_LAX = np.matmul(A_LAX, Rx)
    elif random.randint(-1, 8) > 0:
        A_LAX = np.matmul(A_LAX, Ry)

    # new IOP after rotation
    iop_one = A_LAX[:, 0]
    iop_two = A_LAX[:, 1]

    # generate all possible points inside this plane based on new IOP and original IPP with unit spacing
    points = []
    for x in np.arange(-80, 80, 1):
        for y in np.arange(-80, 80, 1):
            tmp_p = ipp + x * 1 * iop_one + y * 1 * iop_two
            points.append(tmp_p)

    # Interpolate values based on learnt interpolator
    fp = f(points)
    # resize into a (160, 160) plane
    img_3ch = np.zeros((160, 160))
    for kx in range(160):
        for ky in range(160):
            img_3ch[kx, ky] = fp[kx * 160 + ky]

    dicom_array = dataset.pixel_array
    gt = img_3ch.T

    # motion_free data saving
    shift_pixel_1 = 0
    shift_pixel_2 = 0
    direction = np.array([shift_pixel_1, shift_pixel_2])
    motion_data = apply_affine_transformation_data(gt, direction)
    dicom_array = np.short(motion_data)
    dataset.ImagePositionPatient = list(points[0])
    dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
    dataset.PixelData = dicom_array.tobytes()
    dataset.PixelSpacing = [1, 1]
    dataset.Rows, dataset.Columns = dicom_array.shape
    dataset.SliceThickness = 1
    dataset.SpacingBetweenSlices = 1
    dataset.save_as(os.path.join(output_loc + '/motion_free', os.path.basename(path)[:6] + '_LAX_3ch_' + str(shift_pixel_1) + '_'
                                 + str(shift_pixel_2) + '.dcm'))


    # motion_corrupted data saving
    shift_pixel_1 = int(np.random.normal(0, 3.5, 1))
    shift_pixel_2 = int(np.random.normal(0, 3.5, 1))
    direction = np.array([shift_pixel_1, shift_pixel_2])
    motion_data = apply_affine_transformation_data(gt, direction)
    dicom_array = np.short(motion_data)
    dataset.ImagePositionPatient = list(points[0])
    dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
    dataset.PixelData = dicom_array.tobytes()
    dataset.PixelSpacing = [1, 1]
    dataset.Rows, dataset.Columns = dicom_array.shape
    dataset.SliceThickness = 1
    dataset.SpacingBetweenSlices = 1
    dataset.save_as(os.path.join(output_loc + '/motion_corrupted', os.path.basename(path)[:6] + '_LAX_3ch_' + str(shift_pixel_1) + '_'
                                 + str(shift_pixel_2) + '.dcm'))

    ############################################################
    # 2CH-LAX, rotation degree range: [-3, 3]
    ############################################################
    # needed landmarks: apex, mvc, rvc, coh
    vec_one = apex - mvc
    vec_three = rvc - mvc
    vec_two = np.cross(vec_three, vec_one)

    iop_one = vec_one/np.linalg.norm(vec_one)
    iop_two = vec_two/np.linalg.norm(vec_two)

    plane_normal = np.cross(iop_one, iop_two)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    vector = coh - mvc
    distance_vec = np.dot(vector, plane_normal)
    projected_point = coh - distance_vec * plane_normal
    ipp = projected_point

    theta = np.random.choice([-1, 1]) * (np.pi / random.randint(60, 180))

    # rotation matrix R_x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    # rotation matrix R_y
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])


    # original affine matrix for SAX
    A_LAX = np.zeros((3, 3))
    third_col = np.cross(iop_one, iop_two)
    third_col = third_col/ np.linalg.norm(third_col)
    A_LAX[:, 0] = iop_one
    A_LAX[:, 1] = iop_two
    A_LAX[:, 2] = third_col

    # 80% rotation
    if random.randint(-1, 8) > 0:
        A_LAX = np.matmul(A_LAX, Rx)
    elif random.randint(-1, 8) > 0:
        A_LAX = np.matmul(A_LAX, Ry)

    # Corresponding new IOP
    iop_one = A_LAX[:, 0]
    iop_two = A_LAX[:, 1]

    # generate all possible pixels inside this plane based on new IOP and original IPP with unit spacing
    points = []
    for x in np.arange(-80, 80, 1):
        for y in np.arange(-80, 80, 1):
            tmp_p = ipp + x * 1 * iop_one + y * 1 * iop_two
            points.append(tmp_p)

    # Interpolate values based on learnt interpolator
    fp = f(points)
    # resize into a (160, 160) plane
    img_2ch = np.zeros((160, 160))
    for kx in range(160):
        for ky in range(160):
            img_2ch[kx, ky] = fp[kx * 160 + ky]

    dicom_array = dataset.pixel_array
    gt = img_2ch.T

    # motion_free data saving
    shift_pixel_1 = 0
    shift_pixel_2 = 0
    direction = np.array([shift_pixel_1, shift_pixel_2])
    motion_data = apply_affine_transformation_data(gt, direction)
    dicom_array = np.short(motion_data)
    dataset.ImagePositionPatient = list(points[0])
    dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
    dataset.PixelData = dicom_array.tobytes()
    dataset.PixelSpacing = [1, 1]
    dataset.Rows, dataset.Columns = dicom_array.shape
    dataset.SliceThickness = 1
    dataset.SpacingBetweenSlices = 1
    dataset.save_as(os.path.join(output_loc + '/motion_free', os.path.basename(path)[:6] + '_LAX_2ch_' + str(shift_pixel_1) + '_'
                                 + str(shift_pixel_2) + '.dcm'))

    # motion_corrupted data saving
    shift_pixel_1 = int(np.random.normal(0, 3.5, 1))
    shift_pixel_2 = int(np.random.normal(0, 3.5, 1))
    direction = np.array([shift_pixel_1, shift_pixel_2])
    motion_data = apply_affine_transformation_data(gt, direction)
    dicom_array = np.short(motion_data)
    dataset.ImagePositionPatient = list(points[0])
    dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
    dataset.PixelData = dicom_array.tobytes()
    dataset.PixelSpacing = [1, 1]
    dataset.Rows, dataset.Columns = dicom_array.shape
    dataset.SliceThickness = 1
    dataset.SpacingBetweenSlices = 1
    dataset.save_as(os.path.join(output_loc + '/motion_corrupted', os.path.basename(path)[:6] + '_LAX_2ch_' + str(shift_pixel_1) + '_'
                                 + str(shift_pixel_2) + '.dcm'))

    ############################################################
    # Stack of SAXs, rotation degree range: [-10, -3] & [3, 10]
    ############################################################
    # needed landmarks: apex, mvc, tvc, coh
    vec_one = tvc - mvc
    vec_three_tmp = apex - mvc
    vec_two = np.cross(vec_three_tmp, vec_one)
    iop_one = vec_one / np.linalg.norm(vec_one)
    iop_two = vec_two / np.linalg.norm(vec_two)
    theta = random.choice([-1, 1]) * (np.pi / random.randint(18, 60))

    # rotation matrix R_x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    # rotation matrix R_y
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])

    # original affine matrix for SAX
    A_SAX = np.zeros((3, 3))
    third_col = np.cross(iop_one, iop_two)
    third_col = third_col / np.linalg.norm(third_col)
    A_SAX[:, 0] = iop_one
    A_SAX[:, 1] = iop_two
    A_SAX[:, 2] = third_col

    # 80% probability
    if random.randint(-1, 8) > 0:
        A_SAX = np.matmul(A_SAX, Rx)
    if random.randint(-1, 8) > 0:
        A_SAX = np.matmul(A_SAX, Ry)

    # Corresponding new IOP
    iop_one = A_SAX[:, 0]
    iop_two = A_SAX[:, 1]

    # basal and apex slice drop out (50%)
    drop_out_val_basal = random.randint(0, 1)
    drop_out_val_apex = random.randint(0, 1)

    number_of_slices = random.randint(7, 11)
    plane_normal = np.cross(iop_one, iop_two)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    vector_basal = coh - mvc
    distance_vec_basal = np.dot(vector_basal, plane_normal)
    ipp_basal = coh - distance_vec_basal * plane_normal
    print('ipp_basal:' + str(ipp_basal))
    vector_apex = coh - apex
    distance_vec_apex = np.dot(vector_apex, plane_normal)
    ipp_apex = coh - distance_vec_apex * plane_normal
    print('ipp_apex:' + str(ipp_apex))

    # calculate the max length between basal and apical
    max_length_lv = []
    slices_gap = []
    for d in range(3):
        dis_one_dim = int(np.round(ipp_apex[d])) - int(np.round(ipp_basal[d]))
        max_length_lv.append(dis_one_dim)
        slices_gap.append(int(np.round(dis_one_dim / number_of_slices)))

    for i in range(number_of_slices + 1):
        ipp = np.array(
            [ipp_basal[0] + slices_gap[0] * i, ipp_basal[1] + slices_gap[1] * i, ipp_basal[2] + slices_gap[2] * i])

        # generate all possible points inside this plane based on new IOP and original IPP with unit spacing
        points = []
        for x in np.arange(-dense_label_rotated.shape[0] / 2, dense_label_rotated.shape[0] / 2, 1):
            for y in np.arange(-dense_label_rotated.shape[1] / 2, dense_label_rotated.shape[1] / 2, 1):
                tmp_p = ipp + x * 1 * iop_one + y * 1 * iop_two
                points.append(tmp_p)

        # Interpolate values based on learnt interpolator
        fp = f(points)
        # resize into a (160, 160) plane
        img_sax = np.zeros((dense_label_rotated.shape[0], dense_label_rotated.shape[1]))
        for kx in range(dense_label_rotated.shape[0]):
            for ky in range(dense_label_rotated.shape[1]):
                img_sax[kx, ky] = fp[kx * dense_label_rotated.shape[1] + ky]

        dicom_array = dataset.pixel_array
        gt = img_sax.T

        # motion_free data saving
        shift_pixel_1 = 0
        shift_pixel_2 = 0
        direction = np.array([shift_pixel_1, shift_pixel_2])
        motion_data = apply_affine_transformation_data(gt, direction)
        dicom_array = np.short(motion_data)
        dataset.ImagePositionPatient = list(points[0])
        dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
        dataset.PixelData = dicom_array.tobytes()
        dataset.PixelSpacing = [1, 1]
        dataset.Rows, dataset.Columns = dicom_array.shape
        dataset.SliceThickness = 1
        dataset.SpacingBetweenSlices = 1
        # random dropout the basal slice and apex slice
        if i == 0 and drop_out_val_apex:
            output_file = os.path.join(output_loc + '/motion_free',
                                       os.path.basename(path)[:6] + '_SAX_' + str(i + 1) + '_' + str(shift_pixel_1)
                                       + '_' + str(shift_pixel_2) + '_apex.nii.gz')
            write_nifti(dataset, dicom_array, output_file)

        elif i == number_of_slices and drop_out_val_basal:
            output_file = os.path.join(output_loc + '/motion_free',
                                       os.path.basename(path)[:6] + '_SAX_' + str(i + 1) + '_' + str(shift_pixel_1)
                                       + '_' + str(shift_pixel_2) + '_basal.nii.gz')
            write_nifti(dataset, dicom_array, output_file)
        elif 0 < i < number_of_slices and len(np.unique(motion_data)) != 1:
            output_file = os.path.join(output_loc + '/motion_free',
                                       os.path.basename(path)[:6] + '_SAX_' + str(i + 1) + '_' + str(shift_pixel_1)
                                       + '_' + str(shift_pixel_2) + '.nii.gz')
            write_nifti(dataset, dicom_array, output_file)

        # motion_corrupted data saving
        shift_pixel_1 = int(np.random.normal(0, 3.5, 1)[0])
        shift_pixel_2 = int(np.random.normal(0, 3.5, 1)[0])
        direction = np.array([shift_pixel_1, shift_pixel_2])
        motion_data = apply_affine_transformation_data(gt, direction)
        dicom_array = np.short(motion_data)
        dataset.ImagePositionPatient = list(ipp)
        dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
        dataset.PixelData = dicom_array.tobytes()
        dataset.PixelSpacing = [1, 1]
        dataset.Rows, dataset.Columns = dicom_array.shape
        dataset.SliceThickness = 1
        dataset.SpacingBetweenSlices = 1
        # random dropout the basal slice and apex slice
        if i == 0 and drop_out_val_apex:
            output_file = os.path.join(output_loc + '/motion_corrupted',
                                       os.path.basename(path)[:6] + '_SAX_' + str(i + 1) + '_' + str(shift_pixel_1)
                                       + '_' + str(shift_pixel_2) + '_apex.nii.gz')
            print(os.path.basename(path)[:6] + ':apex slices saved')
            write_nifti(dataset, dicom_array, output_file)

        elif i == number_of_slices and drop_out_val_basal:
            output_file = os.path.join(output_loc + '/motion_corrupted',
                                       os.path.basename(path)[:6] + '_SAX_' + str(i + 1) + '_' + str(shift_pixel_1)
                                       + '_' + str(shift_pixel_2) + '_basal.nii.gz')
            print(os.path.basename(path)[:6] + ':basal slice saved')
            write_nifti(dataset, dicom_array, output_file)

        elif 0 < i < number_of_slices and len(np.unique(motion_data)) != 1:
            output_file = os.path.join(output_loc + '/motion_corrupted',
                                       os.path.basename(path)[:6] + '_SAX_' + str(i + 1) + '_' + str(shift_pixel_1)
                                       + '_' + str(shift_pixel_2) + '.nii.gz')
            write_nifti(dataset, dicom_array, output_file)

    return


#######################################################################################
# Step 1: Apply random rotation for the dense data (CT in the cardiac coordinate space)
#######################################################################################
#######################################
# MODIFY HERE
#######################################
dense_label_path = '/path/to/your/CT/data'
output_path = '/save/path/to/your/rotated/CT/data'
dense_data_ls = os.listdir(dense_label_path)
count = 0

for i in tqdm(range(len(dense_data_ls))):
    tmp_path= os.path.join(dense_label_path, dense_data_ls[i])
    tmp_nifti = nib.load(tmp_path)
    tmp_arr = tmp_nifti.get_fdata()
    tmp_aff = tmp_nifti.get_qform()

    # 10% probability
    if random.randint(-1, 8) < 0:
        # angle in degree
        angle = random.randint(-5, 5)
        tmp_new = rotate(tmp_arr, angle, axes= (1, 2), reshape= False, order = 0, mode="constant")
        count += 1
        output_nifti = nib.Nifti1Image(tmp_new, affine=tmp_aff)
    else:
        output_nifti = nib.Nifti1Image(tmp_arr, affine=tmp_aff)

    nib.save(output_nifti, os.path.join(output_path, dense_data_ls[i]))

#######################################################################################
# Step 2: Generate synthetic dicom from rotated dense data
#######################################################################################
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

#######################################
# MODIFY HERE
#######################################
dense_path = '/path/to/dense/data'
rotated_dense_path = '/path/to/rotated/dense/data'
output_dicom_path = '/save/path/to/synthetic/dicom'

# Download link : https://www.cardiacatlas.org/sunnybrook-cardiac-data/
# DICOM image batch 5 (6 cases) is used for the original code
dicom_template_path = '/path/to/the/dicom/format/template'

data_list = sorted(os.listdir(rotated_dense_path))

for k in tqdm(range(len(data_list))):
    path = os.path.join(rotated_dense_path, data_list[k])
    rotated_path = os.path.join(rotated_dense_path, data_list[k])
    gen_dicom(path, rotated_path, output_dicom_path, dicom_template_path)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)
print('#################################################################')
print('synthetic dicom generation is finished')
print('#################################################################')