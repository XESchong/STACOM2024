import os
import numpy as np
import random
import pandas as pd
#from skimage.measure import centroid
import nibabel as nib
from tqdm import tqdm
import random
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import SimpleITK as sitk
#from bbk_simu_online import io_gen
from tqdm import tqdm
from datetime import datetime


# network
class UNet3d(nn.Module):
    def contracting_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(out_channels),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            #torch.nn.Sigmoid()
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet3d, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channel, 16, 32)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=3, in_channels=128, out_channels=128, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(128),
            torch.nn.Conv3d(kernel_size=3, in_channels=128, out_channels=256, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(256),
            torch.nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)
        )
        # Decode
        self.conv_decode3 = self.expansive_block(128+256, 128, 128)
        self.conv_decode2 = self.expansive_block(64+128, 64, 64)
        self.final_layer = self.final_block(32+64, 32, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=False)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=False)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=False)
        final_layer = self.final_layer(decode_block1)
        return final_layer


# data preparation



def train_model(epoch_init, epoch_end, model_folder, init_model= None, lr=0.001):
    device = torch.device("cuda")
    training_loss = 0
    validating_loss = 0

    unet = UNet3d(in_channel=6, out_channel=6)

    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
    if init_model is not None:
        print(init_model)
        checkpoint = torch.load(init_model)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    unet.to(device, dtype=torch.float)
    unet.cuda()
    #loss_func = torch.nn.BCELoss()
    loss_func = torch.nn.MSELoss()

    trn_x_csv = pd.read_csv('/media/yx22/DATA/atlas-istn-main/data/sdm_csv_file/sdm_8/train_img_sdm_8.csv')
    trn_y_csv = pd.read_csv('/media/yx22/DATA/atlas-istn-main/data/sdm_csv_file/sdm_8/train_seg_sdm_8.csv')
    val_x_csv = pd.read_csv('/media/yx22/DATA/atlas-istn-main/data/sdm_csv_file/sdm_8/valid_img_sdm_8.csv')
    val_y_csv = pd.read_csv('/media/yx22/DATA/atlas-istn-main/data/sdm_csv_file/sdm_8/valid_seg_sdm_8.csv')


    for epoch in tqdm(range(epoch_init, epoch_end, 1)):
        print('Epoch: ' + str(epoch + 1))
        print("----------------------------------training----------------------------------")
        trn_idx = random.sample(list(range(len(trn_x_csv))), len(trn_x_csv))
        c_trn = 0
        c_val = 0
        for k_trn in trn_idx:
            print(str(c_trn + 1) + '/' + str(len(trn_x_csv)), end='\n')
            c_trn += 1
            try:
                # train_x_tmp = sitk.ReadImage(trn_x_csv.iloc[k_trn, 0])
                # train_y_tmp = sitk.ReadImage(trn_y_csv.iloc[k_trn, 0])
                train_x_tmp = sitk.ReadImage(os.path.join('/media/yx22/DATA/atlas-istn-main', trn_x_csv.iloc[k_trn, 0]))
                train_y_tmp = sitk.ReadImage(os.path.join('/media/yx22/DATA/atlas-istn-main', trn_y_csv.iloc[k_trn, 0]))
                # if k_trn%2 == 0:
                #     print(k_trn)
                #     print(os.path.join('/media/yx22/DATA/atlas-istn-main', trn_x_csv.iloc[k_trn, 0]))
                #     print(os.path.join('/media/yx22/DATA/atlas-istn-main', trn_y_csv.iloc[k_trn, 0]))
                trn_x = sitk.GetArrayFromImage(train_x_tmp)
                trn_y = sitk.GetArrayFromImage(train_y_tmp)
                train_x = trn_x[np.newaxis, ...]
                train_y = trn_y[np.newaxis, ...]



                t_x = Variable(torch.from_numpy(train_x).permute(0, 4, 1, 2, 3).float().cuda())
                t_y = Variable(torch.from_numpy(train_y).permute(0, 4, 1, 2, 3).float().cuda())
                output = unet(t_x)
                loss = loss_func(output, t_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
                del train_x, train_y
                del t_x, t_y, output, loss
                torch.cuda.empty_cache()
                #print(k_trn, 'yes')
            except:
                print(k_trn, 'Missed.')

        val_list = list(np.arange(0, 100, 1))
        print("----------------------------------validation----------------------------------")
        for k_val in val_list:
            print(str(c_val + 1) + '/' + str(len(val_list)), end='\n')
            c_val += 1
            try:
                # valid_x_tmp = sitk.ReadImage(val_x_csv.iloc[k_val, 0])
                # valid_y_tmp = sitk.ReadImage(val_y_csv.iloc[k_val, 0])
                valid_x_tmp = sitk.ReadImage(os.path.join('/media/yx22/DATA/atlas-istn-main', val_x_csv.iloc[k_val, 0]))
                valid_y_tmp = sitk.ReadImage(os.path.join('/media/yx22/DATA/atlas-istn-main', val_y_csv.iloc[k_val, 0]))
                val_x = sitk.GetArrayFromImage(valid_x_tmp)
                val_y = sitk.GetArrayFromImage(valid_y_tmp)
                valid_x = val_x[np.newaxis, ...]
                valid_y = val_y[np.newaxis, ...]

                v_x = Variable(torch.from_numpy(valid_x).permute(0, 4, 1, 2, 3).float().cuda())
                v_y = Variable(torch.from_numpy(valid_y).permute(0, 4, 1, 2, 3).float().cuda())
                output = unet(v_x)
                loss = loss_func(output, v_y)
                validating_loss += loss.item()
                del valid_x, valid_y
                del v_x, v_y, output, loss
                torch.cuda.empty_cache()
                #print(k_trn, 'yes')
            except:
                print(k_val, ': Missed.')

        print('[%d] training_loss: %.4f, validating_loss: %.4f' % (epoch + 1, training_loss / len(trn_x_csv), validating_loss / len(val_list)))
        np.savez(os.path.join(model_folder, 'exp_1', 'loss', 'epoch_' + str(epoch + 1) + '.npz'),
                 trn=training_loss / len(trn_x_csv), val=validating_loss / len(val_list))
        latest_file = os.path.join(model_folder, 'exp_1', 'epoch_' + str(epoch + 1) + '_params.pth')
        torch.save({'epoch': epoch + 1, 'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, latest_file)

        training_loss = 0
        validating_loss = 0
    return


def loss_eval():

    loss_folder = '/media/yx22/DATA/atlas-istn-main/output/SSA_LTN_sdm_8_v2/exp_1/loss'
    epoch_n = len(os.listdir(loss_folder))
    trn_list = []
    val_list = []
    for ki in range(epoch_n):
        loss_i = np.load(os.path.join(loss_folder, 'epoch_' + str(ki + 1) + '.npz'))
        trn_i = loss_i['trn']
        val_i = loss_i['val']
        trn_list.append(trn_i)
        val_list.append(val_i)
    plt.figure()
    plt.plot(trn_list)
    plt.plot(val_list)
    plt.savefig('/media/yx22/DATA/atlas-istn-main/output/SSA_LTN_sdm_8_v2/exp_1/loss_plot_LTN_shifted_sdm_8_v2.png')
    return

# loss_eval()


def eval_model():
    ############################################################################################
    # Modify here
    model_file = '/media/yx22/DATA/SSA_LTN_DSTN_pack/Label Transfer Network (LTN)/LTN_params.pth'
    ############################################################################################
    device = torch.device("cuda")

    unet = UNet3d(in_channel=6, out_channel=6)
    checkpoint = torch.load(model_file)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.to(device, dtype=torch.float)
    unet.cuda()


    ############################################################################################
    # Modify here
    test_x_csv = pd.read_csv('/media/yx22/DATA/SSA_LTN_DSTN_pack/3d_sparse_oh.csv')
    ##########################################################################################

    tst_list = list(np.arange(0, len(test_x_csv), 1))
    for k_tst in tst_list:
        ############################################################################################
        # Modify here
        test_x_tmp = sitk.ReadImage(os.path.join('/media/yx22/DATA/SSA_LTN_DSTN_pack', test_x_csv.iloc[k_tst, 0]))
        ##########################################################################################
        print(os.path.basename(test_x_csv.iloc[k_tst, 0]))
        tst_x = sitk.GetArrayFromImage(test_x_tmp)
        test_x = tst_x[np.newaxis, ...]




        t_x = Variable(torch.from_numpy(test_x).permute(0, 4, 1, 2, 3).float().cuda())
        output = unet(t_x)
        prd = output.cpu().detach().numpy()
        del test_x, tst_x, output
        torch.cuda.empty_cache()
        lab = np.argmax(prd, axis=1)[0, ...]
        lab = np.transpose(lab, (2, 1, 0))
        prd_nif = nib.Nifti1Image(lab.astype(float), affine=np.eye(4))
        ############################################################################################
        # Modify here
        nib.save(prd_nif, os.path.join('/media/yx22/DATA/SSA_LTN_DSTN_pack/LTN_output', 'pred_' + os.path.basename(test_x_csv.iloc[k_tst, 0])))
        ##########################################################################################
    return

eval_model()

