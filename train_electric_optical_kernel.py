# -*- coding: utf-8 -*-
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import time
import torch
import math
import platform
import torchsummary
from optical_layer import train_complex
from utils import padding, tile_kernels
from importlib.machinery import SourceFileLoader
import logging
import math
########################################################################

if platform.system().lower() == 'windows':
    server_dir = './'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
elif platform.system().lower() == 'linux':
    server_dir = '/data/xiaoyun/Elec-Opt-D2NN/'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

device_cpu = torch.device('cpu')
device_gpu = torch.device('cuda:0')
model_idx = '3bs'

whole_dim = 2000
phase_dim = 1200
wave_lambda = 532e-9
focal_length = 14.5e-2
pixel_size = 8e-6
factor = 100
train_epoch = 12000
dataset_input_shape = (3, 32, 32) # for cifar-10 and imagenet32 dataset
fc_in = 324
fc_out = 10

################################################################################
##########               fashionmnist and cifar-10 ONN         #################
################################################################################
# load network
if model_idx == '3bs':
    # Three layers best model
    folder_to_fit = server_dir + 'results_nc/cifar_3layers_best_0713_091820'
    test_epoch = 190
    foo = SourceFileLoader(
        "a", folder_to_fit+"/electric_network.py").load_module()
    CNN = foo.CNN_complex_3layers_best_small

################################################################################
##########                   Load ONN, start to fit                 ############
################################################################################
with torch.no_grad():
    net = CNN(fc_in, fc_out)
    net.load_state_dict(torch.load(
        folder_to_fit + '/models/%03d.pth' % test_epoch))
    net.eval()
    net_gpu = net.to(device_gpu)
    # net = net.to(device_cpu)
    netsummary = torchsummary.summary(net_gpu, dataset_input_shape)

    now = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))
    logging_file = folder_to_fit+"/Elec_to_optical_fitting__" + now + ".log"
    if os.path.isfile(logging_file):
        os.remove(logging_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_file),
            logging.StreamHandler()
        ])

    layer_idx = 0
    for layer_key, layer_value in netsummary.items():
        if layer_key.startswith('ComplexConv2d'):
            # get parameters of layers
            num_group = layer_value['groups']
            num_padding = layer_value['padding']
            output_shape_total = layer_value['output_shape']
            input_shape_total = layer_value['input_shape']
            complex_kernel_total = layer_value['complex_kernel'].cpu().detach()
            w = layer_value['w'].cpu().detach()

            for group_idx in range(num_group):
                # re-calculate shape
                input_shape = (input_shape_total[0], int(
                    input_shape_total[1] / num_group), input_shape_total[2], input_shape_total[3])
                output_shape = (output_shape_total[0], int(
                    output_shape_total[1] / num_group), output_shape_total[2], output_shape_total[3])

                kernel_out_step = int(
                    complex_kernel_total.shape[0] / num_group)
                complex_kernel = complex_kernel_total[group_idx * kernel_out_step:(
                    group_idx + 1) * kernel_out_step, :, :, :]

                # get required size and padding kernel size
                kernel_shape = complex_kernel.shape
                imgsize = input_shape[2] + 2*num_padding
                psf = padding(complex_kernel, imgsize)
                psf_shape = psf.shape
                print(input_shape, kernel_shape, psf_shape)
                # get input and output channels
                input_ch_num = input_shape[1]
                output_ch_num = output_shape[1]
                input_ch_len = int(math.ceil(math.sqrt(input_ch_num)))
                output_ch_len = int(math.ceil(math.sqrt(output_ch_num)))
                # padding kernel channels to square
                if input_ch_len ** 2 - psf_shape[1] > 0:
                    psf_padding = torch.zeros(
                        psf_shape[0], input_ch_len ** 2 - psf_shape[1], psf_shape[2], psf_shape[3])
                    psf = torch.cat((psf, psf_padding), dim=1)
                psf_shape = psf.shape
                if output_ch_len ** 2 - psf_shape[0] > 0:
                    psf_padding = torch.zeros(
                        output_ch_len ** 2 - psf_shape[0], psf_shape[1], psf_shape[2], psf_shape[3])
                    psf = torch.cat((psf, psf_padding), dim=0)
                psf_shape = psf.shape
                # tile kernels
                psf = torch.transpose(psf, 0, 1)
                psf = tile_kernels(psf, input_ch_len, input_ch_len)
                psf_to_fit = tile_kernels(
                    psf, output_ch_len, output_ch_len).unsqueeze(0).detach()

                psf_fitting_size = psf_to_fit.shape[1] + psf.shape[1]
                # rounded_size = int(math.ceil(psf_fitting_size / 50) * 50)
                rounded_size = psf_fitting_size

                # print layer fit
                print(
                    f'Fitting layer {layer_key}, group {group_idx}, psf {tuple(psf_to_fit.shape)}, fitting size {psf_fitting_size}, rounded size {rounded_size}')

                # start to fit
                compute_loss_region = psf_fitting_size  # rounded_size
                out_layer_name = layer_key + '_group_%d_' % group_idx
                folder_prefix = folder_to_fit + '/' + out_layer_name
                print('Fitting layer %s group %d ...' % (layer_key, group_idx))
                with torch.enable_grad():
                    folder_fitting, loss = train_complex(psf_to_fit, folder_prefix, train_epoch, whole_dim, phase_dim, wave_lambda, focal_length, pixel_size, compute_loss_region, factor)
                    params = {
                        'groups': num_group,
                        'paddings': num_padding,
                        'input_shape': input_shape_total,
                        'output_shape': output_shape_total,
                        'input_ch_len': input_ch_len,
                        'output_ch_len': output_ch_len,
                        'psf_shape': complex_kernel_total.shape,
                        'psf_to_fit_shape': psf_to_fit.shape,
                        'fitting_region': psf_fitting_size
                    }
                    print(params)
                    torch.save(params, folder_fitting + '/params_kernels.pt')
                logging.info('%s    %s    %f    %f' %
                            (out_layer_name, folder_fitting, w.numpy()[0], loss))
                layer_idx = layer_idx + 1
