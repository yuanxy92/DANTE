from utils import *
import os
import pickle
import math
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets.vision import VisionDataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from os import listdir
from os.path import isfile, join

class Retile(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = np.array(sample)
        img = np.tile(image, (self.output_size[0], self.output_size[1]))
        image = Image.fromarray(np.uint8(img))
        return image

class DirDataset(VisionDataset):
    train_list = ['train_data_batch_{}'.format(i + 1) for i in range(10)]
    val_list = ['val_data']

    def __init__(self, root="data", size=32, train=True, transform=None, classes=None):
        super().__init__(root, transform=transform)
        self.data = []
        self.targets = []
        file_list = [f for f in listdir(root) if isfile(join(root, f))]
        for filename in file_list:
            # process filename to get labels
            fname_ = filename[:-4]
            idx, label = fname_.split('_label_')
            filename = os.path.join(self.root, filename)
            print(f'Load image {filename}')
            img = torchvision.io.read_image(filename)
            self.data.append(img.reshape(-1, 3, size, size))
            self.targets.append(label)
        self.data = np.vstack(self.data).transpose((0, 2, 3, 1))
        self.targets = np.vstack(self.targets).astype(int).reshape(self.data.shape[0],)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

class MyCIFAR10_readout(Dataset):
    def __init__(self, server_dir, train, whole_dim, transform, num_used=0):
        cifar_ds = datasets.CIFAR10(root=server_dir,
                                    train=train,
                                    download=True)
        self.origin = cifar_ds if num_used == 0 else Subset(
            cifar_ds, np.arange(num_used))
        self.whole_dim = whole_dim
        self.transform = transform

    def __len__(self):
        return len(self.origin)

    def __getitem__(self, index):
        img, label = self.origin[index]
        img = self.transform(img)
        img = torch.cat((img, torch.zeros(1, 32, 32)), dim=0)
        phase = torch.sigmoid(img) * 1.999 * math.pi
        img = torch.complex(torch.cos(phase), torch.sin(phase))
        img = tile_kernels(img, 2, 2)
        img = padding(img, self.whole_dim)

        return img, torch.tensor(label)