import os
import json
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import numpy as np
import matplotlib as mpl
from datetime import datetime
import SimpleITK as sitk

back_end = mpl.get_backend()
try:
    mpl.use('module://backend_interagg')
    import matplotlib.pyplot as plt

    print('Set matplotlib backend to interagg')
except ImportError:
    print('Cannot set matplotlib backend to interagg, resorting to default backend {}'.format(back_end))
    mpl.use(back_end)
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print('Cannot set matplotlib backend to interagg, resorting to default backend {}'.format(back_end))
    mpl.use(back_end)
    import matplotlib.pyplot as plt



from nets.stn import FullSTN2D, FullSTN3D
from nets.gauss_conv import GaussianSmoothing

from img.processing import zero_mean_unit_var
from img.processing import range_matching
from img.processing import zero_one
from img.processing import threshold_zero
from img.transforms import Resampler
from img.transforms import Normalizer
from img.datasets_disk import ImageSegmentationOneHotDatasetFromDisk

import utils.metrics as mira_metrics
import utils.tensorboard_helpers as mira_th
import utils.vis_helpers as mira_vis

from tensorboardX import SummaryWriter
from attrdict import AttrDict

separator = '----------------------------------------'

# torch.autograd.set_detect_anomaly(True)

def write_images(writer, phase, image_dict, n_iter, mode3d):
    for name, image in image_dict.items():
        if mode3d:
            if image.size(1) == 1:
                writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[0, :, int(image.size(2)/2), ...]), n_iter)

            elif image.size(1) > 3:
                writer.add_image('{}/{}'.format(phase, name),
                                 torch.clamp(image[0, 3:6, int(image.size(2) / 2), ...], 0, 1), n_iter,
                                 dataformats='CHW')
            else:
                writer.add_image('{}/{}'.format(phase, name),
                                 mira_th.normalize_to_0_1(image[0, 1, int(image.size(2) / 2), ...]), n_iter,
                                 dataformats='HW')
        else:
            if image.size(1) ==  1:
                writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[0, ...]), n_iter)
            elif image.size(1) > 3:
                writer.add_image('{}/{}'.format(phase, name), torch.clamp(image[0, 1:4, ...], 0, 1), n_iter,
                                 dataformats='CHW')
            else:
                writer.add_image('{}/{}'.format(phase, name), mira_th.normalize_to_0_1(image[0, 1, ...]), n_iter, dataformats='HW')


def write_values(writer, phase, value_dict, n_iter):
    for name, value in value_dict.items():
        writer.add_scalar('{}/{}'.format(phase, name), value, n_iter)


def set_up_model_and_preprocessing(phase, args):
    print(separator)
    print('Starting {}...'.format(phase))
    print(separator)

    with open(args.config) as f:
        config = json.load(f)

    print('Config from file: ' + str(config))

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + args.dev if use_cuda else "cpu")

    print('Device: ' + str(device))
    if use_cuda:
        print('GPU: ' + str(torch.cuda.get_device_name(int(args.dev))))

    if args.stn == 'f':
        if args.mode3d:
            stn_model = FullSTN3D
        else:
            stn_model = FullSTN2D
    else:
        raise NotImplementedError('STN {} not supported'.format(args.stn))

    print('STN: ' + str(stn_model))

    resampler_img = Resampler(config['spacing'], config['size'])
    resampler_seg = Resampler(config['spacing'], config['size'], is_label=True)

    if config['normalizer'] == 'zero_mean_unit_var':
        normalizer = Normalizer(zero_mean_unit_var)
    elif config['normalizer'] == 'range_matching':
        normalizer = Normalizer(range_matching)
    elif config['normalizer'] == 'zero_one':
        normalizer = Normalizer(zero_one)
    elif config['normalizer'] == 'threshold_zero':
        normalizer = Normalizer(threshold_zero)
    elif config['normalizer'] == 'none':
        normalizer = None
    else:
        raise NotImplementedError('Normalizer {} not supported'.format(config['normalizer']))

    stn_input_channels = 2 * (config['num_classes'] - 1)

    gauss_conv = GaussianSmoothing(config['num_classes'], kernel_size=3, sigma=1, dim=3).to(device)

    stn = stn_model(input_size=config['size'], input_channels=stn_input_channels, device=device).to(device)
    parameters = list(stn.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config['learning_rate'])
    gamma = 0.5 ** (1 / config['epoch_decay_steps'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    config_dict = {'config': config,
                   'device': device,
                   'normalizer': normalizer,
                   'resampler_img': resampler_img,
                   'resampler_seg': resampler_seg,
                   'stn': stn,
                   'gaussian_conv': gauss_conv,
                   'optimizer': optimizer,
                   'scheduler': scheduler
                   }
    print('File config: {}'.format(config_dict))

    return AttrDict(config_dict)

def process_batch(config, batch_samples, atlas_img, atlas_lab, omega):

    labelmap_dense  = batch_samples['image'].to(config.device)
    labelmap_sparse  = batch_samples['labelmap'].to(config.device)
    atlas_labelmap = torch.from_numpy(sitk.GetArrayFromImage(atlas_lab)).permute(3, 0, 1, 2).unsqueeze(0).to(config.device)

    # matrix operation for multiple cases
    repeats = np.ones(len(image.size()))
    repeats[0] = image.size(0)
    atlas_labelmap = atlas_labelmap.repeat(tuple(repeats.astype(int)))

    # 1:: for removing the background channel
    source = labelmap_sparse[:, 1::, ...]
    target = atlas_labelmap[:, 1::, ...]
    config.stn(torch.cat((source, target), dim=1))

    warped_labelmap = config.stn.warp_image(labelmap_dense)
    warped_atlas_labelmap = config.stn.warp_inv_image(atlas_labelmap)

    grid = mira_vis.make_grid_image(config.config['size'], 4, device=config.device)
    grid = grid.repeat(tuple(repeats.astype(int)))
    warp_img2atl = config.stn.warp_image(grid, padding='zeros')
    warp_atl2img = config.stn.warp_inv_image(grid, padding='zeros')

    loss_atl2seg_mse = F.mse_loss(labelmap_dense[:, 1::, ...], warped_atlas_labelmap[:, 1::, ...])
    loss_seg2atl_mse = F.mse_loss(warped_labelmap[:, 1::, ...], atlas_labelmap[:, 1::, ...])

    # Regularization term
    reg_weight = config.config['lambda']
    reg_term = config.stn.regularizer()

    loss_train = omega * (loss_atl2seg_mse + loss_seg2atl_mse + reg_weight * reg_term)


    values_dict = {'01_loss': loss_train.item(),
                   '02_loss_atl2seg_mse': loss_atl2seg_mse.item(),
                   '03_loss_seg2atl_mse': loss_seg2atl_mse.item(),
                   '04_reg_term': reg_term.item(),
                   }


    images_dict = {'01_labelmap_dense': labelmap_dense,
                   '02_labelmap_sparse': labelmap_sparse,
                   '03_atlas_labelmap': atlas_labelmap,
                   '04_warped_atlas_labelmap': warped_atlas_labelmap,
                   '05_warped_labelmap': warped_labelmap,
                   '06_warp_atl2img': warp_atl2img,
                   '07_warp_img2atl': warp_img2atl}

    return loss_train, images_dict, values_dict

def process_batch_test(config, config_stn, batch_samples, atlas_img, atlas_lab):

    labelmap_sparse = batch_samples['labelmap'].to(config.device)
    atlas_labelmap = torch.from_numpy(sitk.GetArrayFromImage(atlas_lab)).permute(3, 0, 1, 2).unsqueeze(0).to(
        config.device)

    repeats = np.ones(len(labelmap_sparse.size()))
    repeats[0] = labelmap_sparse.size(0)

    atlas_labelmap = atlas_labelmap.repeat(tuple(repeats.astype(int)))
    source = labelmap_sparse[:, 1::, ...]
    target = atlas_labelmap[:, 1::, ...]

    config.stn(torch.cat((source, target), dim=1))

    warped_atlas_labelmap = config.stn.warp_inv_image(atlas_labelmap)

    transform = config_stn.stn.get_T()
    transform_inv = config_stn.stn.get_T_inv()

    loss_test = 0
    values_dict = {'01_loss': loss_test}

    images_dict = {'01_labelmap_sparse': labelmap_sparse,
                   '02_atlas_labelmap': atlas_labelmap,
                   '03_warped_atlas_labelmap': warped_atlas_labelmap,
                   '04_transform': transform,
                   '05_transform_inv': transform_inv}

    return  loss_test, images_dict, values_dict

def train(args):
    config = set_up_model_and_preprocessing('TRAINING', args)

    writer = SummaryWriter('{}/tensorboard'.format(args.out))
    global_step = 0

    print(separator)
    print('TRAINING data...')
    print(separator)

    num_w = 8

    # augmentation = False
    dataset_train = ImageSegmentationOneHotDatasetFromDisk(args.train, args.train_sparse, normalizer=config.normalizer,binarize=config.config['binarize'], augmentation=False)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.config['batch_size'], shuffle=True, num_workers= num_w)

    dataset_atlas = ImageSegmentationOneHotDatasetFromDisk(args.atlas, args.atlas_seg, normalizer=config.normalizer, binarize=config.config['binarize'], augmentation=True)
    dataloader_atlas = torch.utils.data.DataLoader(dataset_atlas, batch_size=1, shuffle=True, num_workers= num_w)

    if args.val is not None:
        print(separator)
        print('VALIDATION data...')
        print(separator)

        dataset_val = ImageSegmentationOneHotDatasetFromDisk(args.val, args.val_sparse, normalizer=config.normalizer, binarize=config.config['binarize'], augmentation=False)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers= num_w)

    # Create output directory
    out_dir = os.path.join(args.out, 'train')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    # Note: Must match those used in process_batch()
    loss_names = ['01_loss', '02_loss_atl2seg_mse', '03_loss_seg2atl_mse', '04_reg_term']


    train_logger = mira_metrics.Logger('TRAIN', loss_names)
    validation_logger = mira_metrics.Logger('VALID', loss_names)

    model_dir = args.model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    sample = dataset_atlas.get_sample(0)
    atlas_image = sample['image']
    atlas_labelmap = sample['labelmap']

    sitk.WriteImage(atlas_image, model_dir + '/atlas_image_initial.nii.gz')
    sitk.WriteImage(atlas_labelmap, model_dir + '/atlas_labelmap_initial.nii.gz')


    for epoch in range(1, config.config['epochs'] + 1):
        # config.itn.train()
        config.stn.train()

        if config.config['epoch_loss_fading'] != -1:
            omega = 1 / (1 + np.exp(-(epoch - config.config['epoch_loss_fading']) / 25))
        else:
            omega = 1

        for batch_idx, batch_samples in enumerate(tqdm(dataloader_train, desc='Epoch {}'.format(epoch))):
            global_step += 1
            config.optimizer.zero_grad()
            loss, images_dict, values_dict = process_batch(config, batch_samples, atlas_image, atlas_labelmap, omega)
            loss.backward()
            config.optimizer.step()
            train_logger.update_epoch_logger(values_dict)


        # iterate learning rate decay
        if config.config['epoch_decay_steps']:
            config.scheduler.step()

        train_logger.update_epoch_summary(epoch)
        write_values(writer, 'train', value_dict=train_logger.get_latest_dict(), n_iter=global_step)
        write_images(writer, 'train', image_dict=images_dict, n_iter=global_step, mode3d=args.mode3d)

        # Validation
        if args.val is not None and (epoch == 1 or epoch % config.config['val_interval'] == 0):
            config.stn.eval()

            with torch.no_grad():
                for batch_idx, batch_samples in enumerate(dataloader_val):
                    loss, images_dict, values_dict = process_batch(config, batch_samples, atlas_image, atlas_labelmap, omega)
                    validation_logger.update_epoch_logger(values_dict)

            validation_logger.update_epoch_summary(epoch)
            write_values(writer, phase='val', value_dict=validation_logger.get_latest_dict(), n_iter=global_step)
            write_images(writer, phase='val', image_dict=images_dict, n_iter=global_step, mode3d=args.mode3d)

            print(separator)
            train_logger.print_latest()
            validation_logger.print_latest()
            print(separator)

            torch.save(config.stn.state_dict(), model_dir + '/stn_' + str(epoch) + '.pt')


    torch.save(config.stn.state_dict(), model_dir + '/stn.pt')

    sitk.WriteImage(atlas_image, model_dir + '/atlas_image_final.nii.gz')
    sitk.WriteImage(atlas_labelmap, model_dir + '/atlas_labelmap_final.nii.gz')

    print(separator)
    print('Finished TRAINING... Plotting Graphs\n\n')
    for loss_name, colour in zip(['01_loss'], ['b']):
        plt.plot(train_logger.epoch_number_logger, train_logger.epoch_summary[loss_name], c=colour,
                 label='train {}'.format(loss_name))
        plt.plot(validation_logger.epoch_number_logger, validation_logger.epoch_summary[loss_name], c=colour,
                 linestyle=':',
                 label='val {}'.format(loss_name))

    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.show()
    plt.savefig('SSTN_lambda_2000.png')

def pred(args):
    config = set_up_model_and_preprocessing('TESTING', args)

    current_time = datetime.now()
    print(current_time)

    dataset_test = ImageSegmentationOneHotDatasetFromDisk(args.test, args.test_sparse, normalizer=config.normalizer, binarize=config.config['binarize'], augmentation=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

    loss_names = ['01_loss']

    test_logger = mira_metrics.Logger('TEST', loss_names)

    # Create output directory
    out_dir = os.path.join(args.out, 'test')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    ####################################################################################################################
    # Modify here
    atlas_image = sitk.ReadImage('/media/yx22/DATA/SSA_LTN_DSTN_pack/Dense Spatial Transformer Network (DSTN)/atlas_image_final.nii.gz')
    atlas_labelmap = sitk.ReadImage('/media/yx22/DATA/SSA_LTN_DSTN_pack/Dense Spatial Transformer Network (DSTN)/atlas_labelmap_final.nii.gz')
    config.stn.load_state_dict(torch.load('/media/yx22/DATA/SSA_LTN_DSTN_pack/Dense Spatial Transformer Network (DSTN)/stn_50.pt'))
    ####################################################################################################################
    config.stn.eval()

    with torch.no_grad():
        for index, batch_samples in enumerate(tqdm(dataloader_test)):
            print(index)
            loss, images_dict, values_dict = process_batch_test(config, config, batch_samples, atlas_image, atlas_labelmap)
            test_logger.update_epoch_logger(values_dict)
            file_name = dataset_test.get_sample(index)['fname']

            labelmap = sitk.GetImageFromArray(images_dict['01_labelmap_sparse'].cpu().squeeze().detach().permute(1, 2, 3, 0).numpy(), isVector=True)
            labelmap_argmax = sitk.GetImageFromArray(torch.argmax(images_dict['01_labelmap_sparse'], dim=1).cpu().squeeze().detach().numpy().astype(np.float32))
            warped_atlas_labelmap = sitk.GetImageFromArray(images_dict['03_warped_atlas_labelmap'].cpu().squeeze().detach().permute(1, 2, 3, 0).numpy(), isVector=True)
            warped_atlas_labelmap_argmax = sitk.GetImageFromArray(torch.argmax(images_dict['03_warped_atlas_labelmap'], dim=1).cpu().squeeze().detach().numpy().astype(np.float32))
            transform = sitk.GetImageFromArray(images_dict['04_transform'].cpu().squeeze().detach().numpy(), isVector=True)
            transform_inv = sitk.GetImageFromArray(images_dict['05_transform_inv'].cpu().squeeze().detach().numpy(), isVector=True)

            warped_atlas_labelmap.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            # sitk.WriteImage(warped_atlas_labelmap, os.path.join(out_dir, file_name + '_warped_atlas_labelmap.nii.gz'))

            warped_atlas_labelmap_argmax.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            sitk.WriteImage(warped_atlas_labelmap_argmax, os.path.join(out_dir, file_name + '_warped_atlas_labelmap_argmax.nii.gz'))

            labelmap.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            # sitk.WriteImage(labelmap, os.path.join(out_dir, file_name + '_labelmap.nii.gz'))

            labelmap_argmax.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            # sitk.WriteImage(labelmap_argmax, os.path.join(out_dir, file_name + '_labelmap_argmax.nii.gz'))

            transform.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            # sitk.WriteImage(transform, os.path.join(out_dir, file_name + '_transform.nii.gz'))

            transform_inv.CopyInformation(dataset_test.get_sample(index)['labelmap'])
            # sitk.WriteImage(transform_inv, os.path.join(out_dir, file_name + '_transform_inv.nii.gz'))


        with open(os.path.join(out_dir, 'test_results.yml'), 'w') as outfile:
            yaml.dump(test_logger.get_epoch_logger(), outfile)
    test_logger.update_epoch_summary(0)





if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='atlas warping')
    parser.add_argument('--save_temp', default=True, action='store_true', help='save temporary files (default: True)')
    parser.add_argument('--dev', default='0', help='cuda device (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    # Data args
    parser.add_argument('--train', default='data/SCOTHEART_data/csv_file_parameter_lambda/train_seg.csv', help='training data csv file')
    parser.add_argument('--train_sparse', default='data/SCOTHEART_data/csv_file_parameter_lambda/train_img_no_SSA.csv', help='training data csv file')

    parser.add_argument('--val', default='data/SCOTHEART_data/csv_file_parameter_lambda/valid_seg.csv', help='validation data csv file')
    parser.add_argument('--val_sparse', default='data/SCOTHEART_data/csv_file_parameter_lambda/valid_img_no_SSA.csv', help='validation data csv file')


    ###################################################################################################################
    # Modify here
    parser.add_argument('--test', default='3d_sparse_oh.csv', help='testing data csv file')
    parser.add_argument('--test_sparse', default='3d_sparse_oh.csv', help='testing data csv file')
    # Network args
    parser.add_argument('--mode3d', default=True, action='store_true', help='enable 3D mode', )
    parser.add_argument('--config', default="Dense Spatial Transformer Network (DSTN)/config.json", help='config file')
    ###################################################################################################################



    parser.add_argument('--atlas', default='data/SCOTHEART_data/csv_file_parameter_lambda/atlas_seg.csv', help='atlas data csv file')
    parser.add_argument('--atlas_dense', default='data/SCOTHEART_data/csv_file_parameter_lambda/atlas_seg.csv', help='atlas data csv file')




    # Logging args
    #####################################################################################################
    # Modify here
    parser.add_argument('--out', default='DSTN_output', help='output root directory')
    parser.add_argument('--model', default='DSTN_output', help='model directory')
    #####################################################################################################

    # change the default version from "f" to "s"
    parser.add_argument('--stn', default="f",
                        help='stn type, f=full',
                        choices=['f'])

    args = parser.parse_args()

    # Run training
    if args.train is None:
        train(args)
        
    # Run testing
    if args.test is not None:
        pred(args)


