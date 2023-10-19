import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from easydict import EasyDict as edict


def padding(array, whole_dim):
    # pad square array
    array_size = array.shape[-1]
    pad_size1 = (whole_dim-array_size)//2
    pad_size2 = whole_dim-array_size-pad_size1
    array_pad = F.pad(array, (pad_size1, pad_size2, pad_size1, pad_size2))
    return array_pad


def tile_kernels(array, numx, numy):
    # array should be [C*M*N]
    temp_list = []
    for i in range(numx):
        temp_list.append(torch.cat(
            [array[i*numx+j] for j in range(numy)], -1))

    newarray = torch.cat(temp_list, -2)
    return newarray


def split_kernels(array, numx, numy):
    l = torch.split(array, array.shape[-2]//numx, dim=-2)
    new_l = []
    for a in l:
        l = torch.split(a, a.shape[-1]//numy, dim=-1)
        new_l += l
    return torch.stack(new_l)


def load_model_from_folder(model, folder, epoch):
    params = edict(torch.load(folder+'/params.pt'))
    whole_dim = params.whole_dim
    phase_dim = params.phase_dim
    wave_lambda = params.wave_lambda
    focal_length = params.focal_length
    pixel_size = params.pixel_size
    params_kernel = edict(torch.load(folder+'/params_kernels.pt'))
    # find the weights w/ the maximum epoch number
    
    weights = torch.load(folder+'/results/weights_%05d.pth' % epoch)
    onn = model(whole_dim, phase_dim, pixel_size,
                focal_length, wave_lambda, weights)
    return onn, params, params_kernel


def make_DDNN_labels(unit, interval, whole_dim):
    labels = np.zeros((10, 4*unit+3*interval, 4*unit+3*interval))
    offset = unit+interval
    for i in range(3):
        tempx, tempy = offset//2, offset*i+offset//2
        labels[i, tempx:tempx+unit, tempy:tempy+unit] = 1

    for i in range(3, 7):
        tempx, tempy = offset//2+offset, (i-3)*offset
        labels[i, tempx:tempx+unit, tempy:tempy+unit] = 1

    for i in range(7, 10):
        tempx, tempy = offset//2+offset*2, offset*(i-7)+offset//2
        labels[i, tempx:tempx+unit, tempy:tempy+unit] = 1

    labels = padding(torch.tensor(labels), whole_dim)
    return labels


def make_DDNN_labels2(resize_dim, whole_dim):
    labels = np.zeros((10, 28, 28))
    for i in range(10):
        img = Image.open("temp/Downloads/%d.png" % i).convert("L")
        labels[i] = np.array(img) / 255.0
        transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.Pad((whole_dim-resize_dim)//2),
        ])
    return transform(torch.tensor(labels))
