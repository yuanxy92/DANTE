from telnetlib import Telnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from optical_unit import *
from electric_unit import *
from optical_layer import FourierConv

class DDNN_w_PhaseNL_NP_1ch_3layer_readout(nn.Module):
    def __init__(self, whole_dim, phase_dim, pixel_size, focal_length, wave_lambda):
        super(DDNN_w_PhaseNL_NP_1ch_3layer_readout, self).__init__()
        self.whole_dim = whole_dim
        self.fconv1 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv2 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv3 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)

        self.pooling = nn.AvgPool2d(2, 2)
        # self.readout1 = ReadOut(64*5, 5, 28, self.pooling)
        # self.readout2 = ReadOut(70*6, 6, 12, self.pooling)
        self.readout3 = ReadOut(36*9, 9, 4, self.pooling, False)

        self.fc = nn.Linear(324, 10)
        self.nl = NonLinear_Int2Phase()
        # self.scalar = torch.randn(1, dtype=torch.float32) * 0.001 if scalar is None else torch.tensor(
        # scalar, dtype=torch.float32)
        # self.w_scalar = nn.Parameter(self.scalar)

    def forward(self, input_field):
        # first layer
        x1 = self.nl(self.fconv1(input_field))
        x2 = self.nl(self.fconv2(x1))
        x3 = self.fconv3(x2)
        x3 = self.readout3(x3)

        out = self.fc(torch.flatten(x3, 1))
        return out

# 
class DDNN_w_PhaseNL_NP_3ch_7mask_readout(nn.Module):
    def __init__(self, whole_dim, phase_dim, pixel_size, focal_length, wave_lambda):
        super().__init__()
        self.whole_dim = whole_dim
        self.fconv11 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv12 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv13 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)

        self.fconv21 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv22 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv23 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
            
        self.fconv3 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.conv = nn.Conv2d(9, 1, 1)

        self.pooling = nn.AvgPool2d(2, 2)
        # self.readout1 = ReadOut(64*5, 5, 28, self.pooling)
        # self.readout2 = ReadOut(70*6, 6, 12, self.pooling)
        self.readout3 = ReadOut(36*9, 9, 4, self.pooling, False)
        
        self.fc = nn.Linear(324, 10)
        self.nl = NonLinear_Int2Phase()
        # self.scalar = torch.randn(1, dtype=torch.float32) * 0.001 if scalar is None else torch.tensor(
        # scalar, dtype=torch.float32)
        # self.w_scalar = nn.Parameter(self.scalar)

    def forward(self, input_field):
        # first layer
        x11 = self.nl(self.fconv11(input_field))
        x12 = self.nl(self.fconv12(input_field))
        x13 = self.nl(self.fconv13(input_field))

        x21_11 = self.fconv21(x11)
        x21_12 = self.fconv21(x12)
        x21_13 = self.fconv21(x13)
        x22_11 = self.fconv22(x11)
        x22_12 = self.fconv22(x12)
        x22_13 = self.fconv22(x13)
        x23_11 = self.fconv23(x11)
        x23_12 = self.fconv23(x12)
        x23_13 = self.fconv23(x13)

        x2 = torch.stack((x21_11, x21_12, x21_13, x22_11, x22_12, x22_13, x23_11, x23_12, x23_13), 1)
        x2 = self.nl(torch.squeeze(self.conv(x2), 1))

        x3 = self.fconv3(x2)
        x3 = self.readout3(x3)

        out = self.fc(torch.flatten(x3, 1))
        return out


# original DDNN
class DDNN(nn.Module):
    def __init__(self, whole_dim, phase_dim, pixel_size, focal_length, wave_lambda, scalar=None):
        super(DDNN, self).__init__()
        self.prop = AngSpecProp(
            whole_dim, pixel_size, focal_length, wave_lambda)
        self.phase1 = PhaseMask(whole_dim, phase_dim)
        self.phase2 = PhaseMask(whole_dim, phase_dim)
        self.phase3 = PhaseMask(whole_dim, phase_dim)
        self.phase4 = PhaseMask(whole_dim, phase_dim)
        self.phase5 = PhaseMask(whole_dim, phase_dim)
        self.scalar = torch.randn(1, dtype=torch.float32) if scalar is None else torch.tensor(
            scalar, dtype=torch.float32)
        self.w_scalar = nn.Parameter(self.scalar)

    def forward(self, input_field):
        x = self.prop(input_field)
        x = self.phase1(x)
        x = self.prop(x)
        x = self.phase2(x)
        x = self.prop(x)
        x = self.phase3(x)
        x = self.prop(x)
        x = self.phase4(x)
        x = self.prop(x)
        x = self.phase5(x)
        x = self.prop(x)
        output_amp = torch.square(self.w_scalar*x.abs())
        return output_amp


# Network structure with the same number of phase masks as the Nature Photonics
class DDNN_w_PhaseNL_NP(nn.Module):
    def __init__(self, whole_dim, phase_dim, pixel_size, focal_length, wave_lambda, scalar=None):
        super(DDNN_w_PhaseNL_NP, self).__init__()
        self.fconv1_1 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv1_2 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv1_3 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)

        self.fconv2_1 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv2_2 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv2_3 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)

        self.conv = nn.Conv2d(9, 1, 1)

        self.fconv3_1 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)

        self.nl = NonLinear_Int2Phase()
        self.sensor = Sensor()
        self.scalar = torch.randn(1, dtype=torch.float32) * 0.001 if scalar is None else torch.tensor(
            scalar, dtype=torch.float32)
        self.w_scalar = nn.Parameter(self.scalar)

    def forward(self, input_field):
        x = self.nl(input_field)
        # first layer
        x1_1 = self.fconv1_1(x)
        x1_2 = self.fconv1_2(x)
        x1_3 = self.fconv1_3(x)
        x1_1 = self.nl(x1_1)
        x1_2 = self.nl(x1_2)
        x1_3 = self.nl(x1_3)
        # second layer
        x2_1_1 = self.fconv2_1(x1_1)
        x2_1_2 = self.fconv2_1(x1_2)
        x2_1_3 = self.fconv2_1(x1_3)
        x2_2_1 = self.fconv2_2(x1_1)
        x2_2_2 = self.fconv2_2(x1_2)
        x2_2_3 = self.fconv2_2(x1_3)
        x2_3_1 = self.fconv2_3(x1_1)
        x2_3_2 = self.fconv2_3(x1_2)
        x2_3_3 = self.fconv2_3(x1_3)
        # merge layer
        x2 = torch.stack((x2_1_1, x2_1_2, x2_1_3, x2_2_1,
                        x2_2_2, x2_2_3, x2_3_1, x2_3_2, x2_3_3), 1)
        x3 = self.conv(x2)
        x3 = torch.squeeze(x3, 1)
        # third layer
        x3 = self.nl(x3)
        x3 = self.fconv3_1(x3)
        output_amp = self.w_scalar * x3
        return output_amp

# Network structure with the same number of phase masks as the Nature Photonics
#
class DDNN_w_PhaseNL_NP_readout(nn.Module):
    def __init__(self, whole_dim, phase_dim, pixel_size, focal_length, wave_lambda, scalar=None):
        super(DDNN_w_PhaseNL_NP_readout, self).__init__()
        self.fconv1_1 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv1_2 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv1_3 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)

        self.fconv2_1 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv2_2 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv2_3 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)

        self.conv = nn.Conv2d(9, 1, 1)

        # self.fconv3_1 = FourierConv(whole_dim, phase_dim, pixel_size,focal_length, wave_lambda)
        self.readout = ReadOut(whole_dim, whole_dim // 2, 25)
        self.fc = nn.Linear((whole_dim // 2 // 25) ** 2, 10)

        self.nl = NonLinear_Int2Phase()
        self.sensor = Sensor()

        # self.scalar = torch.randn(1, dtype=torch.float32) * 0.001 if scalar is None else torch.tensor(
        # scalar, dtype=torch.float32)
        # self.w_scalar = nn.Parameter(self.scalar)

    def forward(self, input_field):
        x = self.nl(input_field)
        # first layer
        x1_1 = self.fconv1_1(x)
        x1_2 = self.fconv1_2(x)
        x1_3 = self.fconv1_3(x)
        x1_1 = self.nl(x1_1)
        x1_2 = self.nl(x1_2)
        x1_3 = self.nl(x1_3)
        # second layer
        x2_1_1 = self.fconv2_1(x1_1)
        x2_1_2 = self.fconv2_1(x1_2)
        x2_1_3 = self.fconv2_1(x1_3)
        x2_2_1 = self.fconv2_2(x1_1)
        x2_2_2 = self.fconv2_2(x1_2)
        x2_2_3 = self.fconv2_2(x1_3)
        x2_3_1 = self.fconv2_3(x1_1)
        x2_3_2 = self.fconv2_3(x1_2)
        x2_3_3 = self.fconv2_3(x1_3)
        # merge layer
        x2 = torch.stack((x2_1_1, x2_1_2, x2_1_3, x2_2_1,
                        x2_2_2, x2_2_3, x2_3_1, x2_3_2, x2_3_3), 1)
        x3 = self.conv(x2)
        x3 = torch.squeeze(x3, 1)
        # readout layer
        x3 = self.readout(x3)
        output_amp = self.fc(x3)
        return output_amp


# Network structure with the same number of phase masks as the Nature Photonics
#
class DDNN_w_PhaseNL_NP_3ch_readout(nn.Module):
    def __init__(self, whole_dim, phase_dim, pixel_size, focal_length, wave_lambda):
        super(DDNN_w_PhaseNL_NP_3ch_readout, self).__init__()
        self.fconv1_1 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv1_2 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv1_3 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)

        self.fconv2_1 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv2_2 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv2_3 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)

        self.conv = nn.Conv2d(9, 1, 1)

        # self.fconv3_1 = FourierConv(whole_dim, phase_dim, pixel_size,focal_length, wave_lambda)
        self.readout = ReadOut(48*8, 8, 4, 2)
        self.fc = nn.Linear(8*8*2*2, 10)

        self.nl = NonLinear_Int2Phase()
        self.sensor = Sensor()

        # self.scalar = torch.randn(1, dtype=torch.float32) * 0.001 if scalar is None else torch.tensor(
        # scalar, dtype=torch.float32)
        # self.w_scalar = nn.Parameter(self.scalar)

    def forward(self, input_field):
        x1_1 = self.nl(input_field[:, 0, :, :])
        x1_2 = self.nl(input_field[:, 1, :, :])
        x1_3 = self.nl(input_field[:, 2, :, :])
        # first layer
        x1_1 = self.fconv1_1(x1_1)
        x1_2 = self.fconv1_2(x1_2)
        x1_3 = self.fconv1_3(x1_3)
        x1_1 = self.nl(x1_1)
        x1_2 = self.nl(x1_2)
        x1_3 = self.nl(x1_3)
        # second layer
        x2_1_1 = self.fconv2_1(x1_1)
        x2_1_2 = self.fconv2_1(x1_2)
        x2_1_3 = self.fconv2_1(x1_3)
        x2_2_1 = self.fconv2_2(x1_1)
        x2_2_2 = self.fconv2_2(x1_2)
        x2_2_3 = self.fconv2_2(x1_3)
        x2_3_1 = self.fconv2_3(x1_1)
        x2_3_2 = self.fconv2_3(x1_2)
        x2_3_3 = self.fconv2_3(x1_3)
        # merge layer
        x2 = torch.stack((x2_1_1, x2_1_2, x2_1_3, x2_2_1,
                        x2_2_2, x2_2_3, x2_3_1, x2_3_2, x2_3_3), 1)
        x3 = self.conv(x2)
        x3 = torch.squeeze(x3, 1)
        # readout layer
        x3 = self.readout(x3)
        output_amp = self.fc(x3)
        return output_amp

# Small Network structure
class DDNN_w_PhaseNL_small_3ch_readout(nn.Module):
    def __init__(self, whole_dim, phase_dim, pixel_size, focal_length, wave_lambda, scalar=None):
        super(DDNN_w_PhaseNL_small_3ch_readout, self).__init__()
        self.fconv1_1 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv1_2 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
        self.fconv1_3 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)

        self.conv = nn.Conv2d(3, 1, 1)

        self.fconv2 = FourierConv(
            whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)

        # self.fconv3_1 = FourierConv(whole_dim, phase_dim, pixel_size,focal_length, wave_lambda)
        self.readout = ReadOut(whole_dim, whole_dim // 2, 25)
        self.fc = nn.Linear((whole_dim // 2 // 25) ** 2, 10)
        self.softmax = nn.Softmax(dim=1)

        self.nl = NonLinear_Int2Phase()
        self.sensor = Sensor()

        # self.scalar = torch.randn(1, dtype=torch.float32) * 0.001 if scalar is None else torch.tensor(
        # scalar, dtype=torch.float32)
        # self.w_scalar = nn.Parameter(self.scalar)

    def forward(self, input_field):
        x1_1 = self.nl(input_field[:, 0, :, :])
        x1_2 = self.nl(input_field[:, 1, :, :])
        x1_3 = self.nl(input_field[:, 2, :, :])
        # first layer
        x1_1 = self.fconv1_1(x1_1)
        x1_2 = self.fconv1_2(x1_2)
        x1_3 = self.fconv1_3(x1_3)
        # merge layer
        x1_merge = torch.stack((x1_1, x1_2, x1_3), 1)
        x2 = self.conv(x1_merge)
        x2 = torch.squeeze(x2, 1)
        # second layer
        x3 = self.nl(x2)
        x3 = self.fconv2(x3)
        # readout layer
        x4 = self.readout(x3)
        output_amp = self.softmax(self.fc(x4))
        return output_amp
