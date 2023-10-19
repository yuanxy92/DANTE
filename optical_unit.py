import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.fft import fftshift, fft2, ifft2, ifftshift
from torchvision import transforms
from utils import *


class Lens(nn.Module):
    def __init__(self, whole_dim, pixel_size, focal_length, wave_lambda):
        super(Lens, self).__init__()
        # basic parameters
        temp = np.arange((-np.ceil((whole_dim - 1) / 2)),
                        np.floor((whole_dim - 1) / 2)+0.5)
        x = temp * pixel_size
        xx, yy = np.meshgrid(x, x)
        lens_function = np.exp(
            -1j * math.pi / wave_lambda / focal_length * (xx ** 2 + yy ** 2))
        self.lens_function = torch.tensor(
            lens_function, dtype=torch.complex64).cuda()

    def forward(self, input_field):
        out = torch.mul(input_field, self.lens_function)
        return out
    
class AngSpecProp(nn.Module):  
    def __init__(self, whole_dim, pixel_size, focal_length, wave_lambda):
        super(AngSpecProp, self).__init__()
        k = 2*math.pi/wave_lambda  # optical wavevector
        df1 = 1 / (whole_dim*pixel_size)
        f = np.arange((-np.ceil((whole_dim - 1) / 2)),
                      np.floor((whole_dim - 1) / 2)+0.5) * df1
        fxx, fyy = np.meshgrid(f, f)
        fsq = fxx ** 2 + fyy ** 2

        self.Q2 = torch.tensor(
            np.exp(-1j*(math.pi**2)*2*focal_length/k*fsq), dtype=torch.complex64).cuda()
        self.pixel_size = pixel_size
        self.df1 = df1

    def ft2(self, g, delta):
        return fftshift(fft2(ifftshift(g))) * (delta ** 2)

    def ift2(self, G, delta_f):
        N = G.shape[1]
        return ifftshift(ifft2(fftshift(G))) * ((N * delta_f)**2)

    def forward(self, input_field):
        # compute the propagated field
        Uout = self.ift2(self.Q2 * self.ft2(input_field,
                        self.pixel_size), self.df1)
        return Uout


class PhaseMask(nn.Module):
    def __init__(self, whole_dim, phase_dim, phase=None):
        super(PhaseMask, self).__init__()
        self.whole_dim = whole_dim
        phase = torch.randn(1, phase_dim, phase_dim, dtype=torch.float32) if phase is None else torch.tensor(
            phase, dtype=torch.float32)
        self.w_p = nn.Parameter(phase)
        pad_size = (whole_dim-phase_dim)//2
        self.paddings = (pad_size, pad_size, pad_size, pad_size)

    def forward(self, input_field):
        mask_phase = torch.sigmoid(self.w_p) * 1.999 * math.pi
        mask_whole = F.pad(torch.complex(torch.cos(mask_phase),
                                        torch.sin(mask_phase)), self.paddings).cuda()
        output_field = torch.mul(input_field, mask_whole)
        return output_field


class NonLinear_Int2Phase(nn.Module):
    def __init__(self):
        super(NonLinear_Int2Phase, self).__init__()

    def forward(self, input_field):
        phase = torch.sigmoid(input_field) * 1.999 * math.pi
        phase = torch.complex(torch.cos(phase), torch.sin(phase)).cuda()
        return phase

class Incoherent_Int2Complex(nn.Module):
    def __init__(self):
        super(Incoherent_Int2Complex, self).__init__()

    def forward(self, input_field):
        x = torch.complex(input_field, torch.zeros(input_field.shape)).cuda()
        return x


class Sensor(nn.Module):
    def __init__(self):
        super(Sensor, self).__init__()

    def forward(self, input_field):
        x = torch.square(torch.real(input_field)) + \
            torch.square(torch.imag(input_field))
        return x


class ReadOut(nn.Module):
    def __init__(self, center_dim, crop_num, crop_size, pooling, if_tile=True):
        super(ReadOut, self).__init__()
        self.crop_num = crop_num
        self.crop_offset = center_dim//crop_num//2-crop_size//2
        self.crop = transforms.CenterCrop(center_dim)
        self.pooling = pooling
        self.if_tile = if_tile

    def forward(self, input_field):
        x = self.crop(input_field)
        x = split_kernels(x, self.crop_num, self.crop_num)
        x = x[:, :, self.crop_offset:-self.crop_offset,
            self.crop_offset:-self.crop_offset]
        x = self.pooling(torch.flip(x, dims=[2, 3]))
        if self.if_tile:
            return tile_kernels(x, self.crop_num, self.crop_num)
        else:
            return x.transpose(0,1)


