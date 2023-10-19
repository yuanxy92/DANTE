# from this import d
import imp
import os
import math
from re import L
import torch
from torch import nn
import torchvision
from torch.nn.functional import relu
from torch.nn import MaxPool2d
from electric_unit import ChannelRepeat, ComplexConv2d, ConvBlock, ChannelShuffle
import torch.nn.functional as F
import torchsummary
from utils import *

# Utility function for channel shuffle in group convolution
def shuffle_channel(x, num_groups):
    # channel shuffle
    batch_size, num_channels, height, width = x.size()
    assert num_channels % num_groups == 0

    x = x.view(batch_size, num_groups, num_channels // num_groups, height, width)
    x = x.permute(0, 2, 1, 3, 4)
    return x.contiguous().view(batch_size, num_channels, height, width)

# Utility function for optical fitting size computation
def compute_fitting_size(net, input_size):
    netsummary = torchsummary.summary(net, input_size)
    layers = []
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

            print(f'---------------------------- layer {layer_idx + 1} ----------------------------')
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
                rounded_size = int(math.ceil(psf_fitting_size / 50) * 50)
                
                # start to fit
                print(f'Fitting layer {layer_key}, group {group_idx}, psf {tuple(psf_to_fit.shape)}, fitting size {psf_fitting_size}, rounded size {rounded_size}')
                layers.append({'layer index': layer_idx, 'group index': group_idx, 'psf shpae': psf_to_fit.shape})
            print(' ')

            layer_idx = layer_idx + 1
    return layers


############## cifar-10 neural network best performance ################
class CNN_complex_3layers_best_small(nn.Module):
    def __init__(self, fc_in=324, fc_out=10):
        super().__init__()
        self.conv1 = ConvBlock(3, 25, kernel_size=5, padding=0,batch_norm=False, kernel_complex=True)

        self.conv2 = ConvBlock(25, 36, kernel_size=3, padding=0,batch_norm=False, kernel_complex=True)

        self.conv3 = ConvBlock(36, 81, kernel_size=3, padding=0,batch_norm=False, kernel_complex=True)

        # self.bn = nn.BatchNorm1d(10) 
        self.pooling = nn.AvgPool2d(2, 2)  
        self.fc = nn.Linear(fc_in, fc_out, bias=False)

    def forward(self, input_field):
        x1 = self.conv1(input_field)
        x1 = self.pooling(x1)

        x2 = self.conv2(x1)
        x2 = self.pooling(x2)

        x3 = self.conv3(x2)
        x3 = self.pooling(x3)

        x4 = torch.flatten(x3, 1)
        x4 = self.fc(x4)
        # x4 = self.bn(x4)
        return x4

