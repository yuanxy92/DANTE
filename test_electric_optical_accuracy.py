# -*- coding: utf-8 -*-
import os
import time
from tokenize import group
from joblib import parallel_backend
import torch
import torchvision
import torchvision.transforms as transforms
import platform
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from optical_layer import FourierConvComplex
from utils import *
import math
from importlib.machinery import SourceFileLoader
from scipy.io import savemat
import argparse
from os import listdir
from os.path import isfile, join
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
########################################################################
if platform.system().lower() == 'windows':
    server_dir = './'
    dl_num_worker = 4
    imagenet_dir = 'D:/Data/ImageNet/64x64'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
elif platform.system().lower() == 'linux':
    server_dir = '/data/xiaoyun/Elec-Opt-D2NN/'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    dl_num_worker = 0

is_debug_size = False
loaded_epoch = 15000
batch_size = 32

# set the parameter to generate intermediate results for paper figures
test_for_fig = False
if test_for_fig == True:
    data_dir = server_dir + 'results/cifar10'

# load fitted phase masks
def read_elec_to_optical_log(file):
    f = open(file, "r")
    fitting_dirs = []
    while True:
        # Get next line from file
        line = f.readline()
        # if line is empty
        # end of file is reached
        if not line:
            break
        fitting_dirs.append(line.split()[1])
    return fitting_dirs

def load_model_from_folder(model, folder, epoch):
    params = edict(torch.load(folder+'/params.pt'))
    whole_dim = params.whole_dim
    phase_dim = params.phase_dim
    wave_lambda = params.wave_lambda
    focal_length = params.focal_length
    pixel_size = params.pixel_size
    params_kernel = edict(torch.load(folder+'/params_kernels.pt'))
    # find the weights w/ the maximum epoch number
    files_all = os.listdir(folder + '/results')
    files_pth = [i for i in files_all if i.endswith('.pth')]
    files_pth.sort(reverse=True)
    weight_name = files_pth[0]
    weights = torch.load(folder+'/results/%s' % weight_name) 
    print(f'----- Layer {folder}, load weight {weight_name} -----')
    onn = model(whole_dim, phase_dim, pixel_size,
                focal_length, wave_lambda, weights)
    return onn, params, params_kernel

# generate input from raw data
def get_inputs(images, params, padding_size, input_ch_len):
    if images.shape[1] < input_ch_len ** 2:
        images = torch.cat((images, torch.zeros(
            images.shape[0], input_ch_len ** 2 - images.shape[1], images.shape[2], images.shape[3]).to(device)), dim=1)
    images = images.transpose(0, 1)
    phase = torch.sigmoid(images) * 1.999 * math.pi
    images = torch.complex(torch.cos(phase), torch.sin(phase))
    images = padding(images, padding_size)
    images = tile_kernels(images, input_ch_len, input_ch_len)
    valid_region_size = images.shape
    images = padding(images, params.whole_dim)
    return images, valid_region_size

# get cropped area
def get_cropped_area(results, scalar, output_size, output_sidenum, crop_size):
    whole_dim = results.shape[-1]
    offset = whole_dim//2-output_size//2
    crop_offset = output_size//output_sidenum//2-crop_size//2

    output_optics = results[:, offset:-offset, offset:-offset]
    if crop_size % 2 == 0:
        crop_output_optics = split_kernels(output_optics, output_sidenum, output_sidenum)[
        :, :, crop_offset:-crop_offset, crop_offset:-crop_offset]
    else:
        crop_output_optics = split_kernels(output_optics, output_sidenum, output_sidenum)[
        :, :, crop_offset:-(crop_offset - 1), crop_offset:-(crop_offset - 1)]

    all_intensity = torch.sum(torch.abs(results)**2)/(whole_dim**2)
    output_intensity = torch.sum(torch.abs(output_optics)**2)/(output_size**2)
    cropped_intensity = torch.sum(
        torch.abs(crop_output_optics)**2)/(crop_size**2)/(output_sidenum**2)

    crop_output_optics = torch.abs(
        crop_output_optics*scalar)**2  # +layer1_bias
    return torch.flip(crop_output_optics, dims=[2, 3]), all_intensity, output_intensity, cropped_intensity


if __name__ == '__main__':
        # load dataset, choose dataset based on the model_idx
    parser = argparse.ArgumentParser(description='Experiment fig1 params')
    parser.add_argument('-e', '--exp_idx', type=str, help='experiment index')
    parser.add_argument('-g', '--gpu_idx', type=str, help='GPU index')
    args = parser.parse_args()
    model_idx = args.exp_idx
    gpu_idx = args.gpu_idx
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx) 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device used is {device}')

    # set model parameters
    if model_idx == '3bs':
        folder_to_fit = '/data/xiaoyun/Elec-Opt-D2NN/results_nc/cifar_3layers_best_0713_091820'
        model_name = 'CNN_complex_3layers_best_small'
        model_structure = ['conv1', 'pooling', 'conv2',
                        'pooling', 'conv3', 'pooling', 'flatten', 'fc']
        optical_impl = [1, 0, 1, 0, 1, 0, 0, 0]
        logtime = '0713_093717'
        best_epoch = 190
        fc_in = 324
        fc_out = 10
        loaded_epoch = 12000
        batch_size = 24
    

    # set cifar-10 dataset
    norm_mean = [0.485, 0.456, 0.406]  # mean
    norm_std = [0.229, 0.224, 0.225]  # std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    testset = torchvision.datasets.CIFAR10(
        root=server_dir + "/data/", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=dl_num_worker)


    # load model
    foo = SourceFileLoader(
        "a", folder_to_fit+"/electric_network.py").load_module()
    CNN = eval('foo.%s' % (model_name))
    net = CNN(fc_in, fc_out)
    net.to(device)
    net.load_state_dict(torch.load(
        folder_to_fit + '/models/%03d.pth' % best_epoch))
    net.eval()

    # load optical fitting masks
    fitting_log_file = folder_to_fit + '/Elec_to_optical_fitting__%s.log' % (logtime)
    fitting_dirs = read_elec_to_optical_log(fitting_log_file)

    # set output dir
    if test_for_fig == True:
        now = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))
        outdir = folder_to_fit + '/optical_test_'
        # outdir = folder_to_fit + '/optical_test_' + now
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    # load all the layers
    conv_idx = 0
    onn_layers = [0] * len(model_structure)
    for layer_idx in range(len(model_structure)):
        layer_name = model_structure[layer_idx]
        # print(layer_idx)
        # print(layer_name)
        if layer_name == 'flatten':
            onn_layers[layer_idx] = {'onn': nn.Flatten(start_dim=1)}
            continue
        layer = eval('net.%s' % layer_name)
        if type(layer).__name__ == 'ConvBlock':
            num_group = layer.layers[0].groups
            if optical_impl[layer_idx] == 1:
                # get number of groups
                onn_layers[layer_idx] = [0] * num_group
                for group_idx in range(num_group):
                    # load fitted phase masks
                    fitting_dir = fitting_dirs[conv_idx]
                    # print(fitting_dir)
                    onn_layer, params, params_kernel = load_model_from_folder(FourierConvComplex, fitting_dir, loaded_epoch)
                    onn_layer = onn_layer.to(device)
                    onn_layer.eval()
                    onn_layers[layer_idx][group_idx] = {'onn': onn_layer, 'params': params, 'params_kernel': params_kernel, 'scalar': layer.layers[0].w_scalar, 'kshape': layer.layers[0].complex_kernel.shape, 'groups': layer.layers[0].groups, 'padding': layer.layers[0].padding}
                    conv_idx = conv_idx + 1
                # check batch normalization layers
                if len(layer.layers) > 1 and type(layer.layers[1]).__name__ == 'BatchNorm2d':
                    onn_layers[layer_idx][0]['bn'] = layer.layers[1]
            else:
                onn_layers[layer_idx] = {'onn': layer}
                conv_idx = conv_idx + num_group
        elif layer_name == 'pooling' or layer_name == 'maxpool':
            onn_layers[layer_idx] = {'onn': layer}
        elif layer_name == 'shuffle2' or layer_name == 'shuffle4' or layer_name == 'shuffle8' or layer_name == 'shuffle16' or layer_name == 'ch_repeat':
            onn_layers[layer_idx] = {'onn': layer}
        elif layer_name == 'fc':
            onn_layers[layer_idx] = {'onn': layer}
        else:
            print(f'Unknown layer {layer_name} encountered in reading weights!')
            exit()

    # iterate all the layers
    # total, correct = 0, 0
    correct, correct_top5, total = 0, 0, 0
    # set no grad
    with torch.no_grad():
        for j, data in enumerate(testloader):
            print(f'{j}/{len(testloader)}')
            images, labels = data[0].to(device), data[1].to(device)
            optical_field = images
            ## intermediate results array initialization
            if test_for_fig == True:
                intermediate_result_ = [[] for i in range(images.shape[0])]
                intermediate_type_ = [[] for i in range(images.shape[0])]
                intermediate_param_ = [[] for i in range(images.shape[0])]
                intermediate_result_label_ = [0] * images.shape[0]
                for idx_in_batch in range(images.shape[0]):
                    intermediate_result_[idx_in_batch].append(images[idx_in_batch].cpu().detach().numpy())
                    intermediate_result_label_[idx_in_batch] = labels[idx_in_batch].cpu().detach().numpy()
                    intermediate_param_[idx_in_batch].append({})
                    intermediate_type_[idx_in_batch].append({'type':'image'})
            ############################# layer 1 ##########################
            for layer_idx in range(len(model_structure)):
                layer_name = model_structure[layer_idx]
                # print(layer_idx)
                if layer_name.startswith('conv'):
                    if optical_impl[layer_idx] == 1:
                        # get number of groups
                        num_groups = len(onn_layers[layer_idx])
                        optical_field_sub = [0] * num_groups
                        for group_idx in range(num_groups):
                            # load layer
                            onn_layer = onn_layers[layer_idx][group_idx]['onn']
                            # tile inputs
                            params_kernel = onn_layers[layer_idx][group_idx]['params_kernel']
                            input_shape = params_kernel['input_shape'][2]
                            # append intermediate param 
                            # split inputs
                            input_step = int(params_kernel['input_shape'][1] / num_groups)
                            optical_field_temp = optical_field[:, group_idx * input_step:(group_idx + 1) * input_step, :, :]

                            optical_field_temp, valid_input_shape = get_inputs(
                                images=optical_field_temp, params=onn_layers[layer_idx][group_idx]['params'],
                                padding_size=input_shape +
                                params_kernel['paddings'] * 2,
                                input_ch_len=params_kernel['input_ch_len'])
                            params_kernel['valid_input_shape'] = valid_input_shape
                            # save input optical field
                            if test_for_fig == True:
                                for idx_in_batch in range(images.shape[0]):
                                    intermediate_result_[idx_in_batch].append(optical_field_temp[idx_in_batch].cpu().detach().numpy())
                                    intermediate_param_[idx_in_batch].append(params_kernel)
                                    intermediate_type_[idx_in_batch].append({'type':'in'})
                            # crop results from the optical computing results
                            output_size = params_kernel['psf_to_fit_shape'][1]
                            optical_field_temp = onn_layer(optical_field_temp)
                            # save output optical field
                            if test_for_fig == True:
                                for idx_in_batch in range(images.shape[0]):
                                    intermediate_result_[idx_in_batch].append(optical_field_temp[idx_in_batch].cpu().detach().numpy())
                                    intermediate_param_[idx_in_batch].append(params_kernel)
                                    intermediate_type_[idx_in_batch].append({'type':'out'})
                            optical_field_temp, a, b, c = get_cropped_area(optical_field_temp, onn_layers[layer_idx][group_idx]['scalar'], output_size,  params_kernel['output_ch_len'], input_shape - params_kernel['psf_shape'][2] + 1 + params_kernel['paddings'] * 2)
                            optical_field_sub[group_idx] = optical_field_temp.transpose(0, 1)
                            num_del_ch = int(params_kernel['output_ch_len'] ** 2 - params_kernel['psf_shape'][0] / num_groups)
                            if num_del_ch > 0:
                                optical_field_sub[group_idx] = optical_field_sub[group_idx][:, 0:-num_del_ch, :, :]
                        optical_field = torch.cat(optical_field_sub, dim=1)
                        if 'bn' in onn_layers[layer_idx][0]:
                            optical_field = onn_layers[layer_idx][0]['bn'](optical_field)
                        # delete the channels used for padding
                        if is_debug_size:
                            print(f'optical_field shape {optical_field.shape}')
                            print(params_kernel['output_ch_len'])
                            print(params_kernel['psf_shape'])
                            print(f'number of channels needed to be removed {num_del_ch}')
                            print(f'optical_field_sub shape {optical_field_sub[0].shape}')
                    else:
                        # load layer
                        onn_layer = onn_layers[layer_idx]['onn']
                        optical_field = onn_layer(optical_field)
                        # append intermediate param 
                        if test_for_fig == True:
                            for idx_in_batch in range(images.shape[0]):
                                intermediate_param_[idx_in_batch].append({})
                elif layer_name == 'pooling' or layer_name == 'flatten' or layer_name == 'fc' or layer_name == 'maxpool' or layer_name == 'ch_repeat' or layer_name == 'shuffle2'  or layer_name == 'shuffle4' or layer_name == 'shuffle8' or layer_name == 'shuffle16':
                    # load layer
                    onn_layer = onn_layers[layer_idx]['onn']
                    if layer_name == 'fc' and test_for_fig == True:
                        for idx_in_batch in range(images.shape[0]):
                            intermediate_result_[idx_in_batch].append(optical_field[idx_in_batch].cpu().detach().numpy())
                            intermediate_param_[idx_in_batch].append({})
                            intermediate_type_[idx_in_batch].append({'type':'fc_in'})
                    optical_field = onn_layer(optical_field)
                    if layer_name == 'fc' and test_for_fig == True:
                        for idx_in_batch in range(images.shape[0]):
                            intermediate_result_[idx_in_batch].append(optical_field[idx_in_batch].cpu().detach().numpy())
                            intermediate_param_[idx_in_batch].append({})
                            intermediate_type_[idx_in_batch].append({'type':'fc_out'})

                else:
                    print('Unknown layer encountered in inference!')
                    exit()
                if is_debug_size:
                    print(f'optical_field shape {optical_field.shape}')
            # save output optical field
            
            # save output optical field to files
            if test_for_fig == True:
                for idx_in_batch in range(images.shape[0]):
                    print(f'Saving result for batch {j}, index {idx_in_batch} ...')
                    savemat(f'{outdir}/{idx_in_batch + j * batch_size}.mat', {'results': np.asanyarray(intermediate_result_[idx_in_batch]), 'labels': np.asanyarray(intermediate_result_label_[idx_in_batch]), 'params': intermediate_param_[idx_in_batch], 'types': intermediate_type_[idx_in_batch]})
                    
            # evaluate the output of optical neural networks        
            _, predicted = torch.max(optical_field, 1)
            total += labels.size(0)
            correct += (predicted == labels.reshape(predicted.shape)).sum().item()
            test_acc = 100 * correct / total
            print(
            f'Top-1 accuracy of the network on the {total} test images: {test_acc} %')
            
        print(
            f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
