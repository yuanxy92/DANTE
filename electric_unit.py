import torch
import torch.nn as nn
import torch.nn.functional as F
from optical_unit import NonLinear_Int2Phase

class ChannelRepeat(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
    def forward(self, input_field):
        return input_field.repeat([1, self.k, 1, 1])

class ChannelShuffle(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        # channel shuffle
        batch_size, num_channels, height, width = x.size()
        assert num_channels % self.num_groups == 0

        x = x.view(batch_size, self.num_groups, num_channels // self.num_groups, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        return x.contiguous().view(batch_size, num_channels, height, width)

class ComplexConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding=0, bias=False, groups=1):
        super(ComplexConv2d, self).__init__()
        assert(in_c % groups == 0 and out_c % groups == 0)
        self.complex_kernel = nn.Parameter(torch.randn(
            out_c, in_c//groups, kernel_size, kernel_size, dtype=torch.complex64))
        self.scalar = torch.ones(1, dtype=torch.float32)*0.1
        self.w_scalar = nn.Parameter(self.scalar)
        self.groups = groups

        self.bias = bias
        if self.bias:
            self.bias_init = torch.ones(1, dtype=torch.float32)*0.0001
            self.bias_scalar = nn.Parameter(self.bias_init)

        self.padding = padding

    def forward(self, input_field):
        k_real = torch.real(self.complex_kernel)
        k_imag = torch.imag(self.complex_kernel)

        x_real = torch.real(input_field.type(torch.complex64))
        x_imag = torch.imag(input_field.type(torch.complex64))

        y_real = F.conv2d(x_real, k_real, bias=None, padding=self.padding, groups=self.groups) - \
            F.conv2d(x_imag, k_imag, bias=None, padding=self.padding, groups=self.groups)
        y_imag = F.conv2d(x_imag, k_real, bias=None, padding=self.padding, groups=self.groups) + \
            F.conv2d(x_real, k_imag, bias=None, padding=self.padding, groups=self.groups)
        y = torch.complex(y_real, y_imag)

        output_field = torch.square(self.w_scalar*y.abs())
        if self.bias:
            output_field += self.bias_scalar
        return output_field

class ComplexConv2d_PurePhase(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding=0, bias=False, groups=1):
        super(ComplexConv2d, self).__init__()
        assert(in_c % groups == 0 and out_c % groups == 0)
        self.phase = nn.Parameter(torch.randn(
            out_c, in_c//groups, kernel_size, kernel_size, dtype=torch.float32))
        self.scalar = torch.ones(1, dtype=torch.float32)*0.1
        self.w_scalar = nn.Parameter(self.scalar)
        self.groups = groups

        self.bias = bias
        if self.bias:
            self.bias_init = torch.ones(1, dtype=torch.float32)*0.0001
            self.bias_scalar = nn.Parameter(self.bias_init)

        self.padding = padding

    def forward(self, input_field):
        # ifft kernel
        k = torch.fft.ifft2(torch.complex(
            torch.cos(self.phase), torch.sin(self.phase)), dim=(2, 3))
        k_real = torch.real(k)
        k_imag = torch.imag(k)

        x_real = torch.real(input_field.type(torch.complex64))
        x_imag = torch.imag(input_field.type(torch.complex64))

        y_real = F.conv2d(x_real, k_real, bias=None, padding=self.padding, groups=self.groups) - \
            F.conv2d(x_imag, k_imag, bias=None, padding=self.padding, groups=self.groups)
        y_imag = F.conv2d(x_imag, k_real, bias=None, padding=self.padding, groups=self.groups) + \
            F.conv2d(x_real, k_imag, bias=None, padding=self.padding, groups=self.groups)
        y = torch.complex(y_real, y_imag)

        output_field = torch.square(self.w_scalar*y.abs())
        if self.bias:
            output_field += self.bias_scalar
        return output_field

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=5, padding=2, bias=False, activation=False, groups=1, batch_norm=True, kernel_complex=True, input_complex=True, pure_phase=False):
        super(ConvBlock, self).__init__()
        if kernel_complex and pure_phase:
            layers = [ComplexConv2d_PurePhase(in_c, out_c, kernel_size=kernel_size,
                                    padding=padding, bias=bias, groups=groups)]
        elif kernel_complex and not pure_phase:
            layers = [ComplexConv2d(in_c, out_c, kernel_size=kernel_size,
                                    padding=padding, bias=bias, groups=groups)]
            
        if batch_norm:
            layers += [nn.BatchNorm2d(num_features=out_c)]
        if activation:
            layers += [nn.ReLU()]

        self.layers = nn.Sequential(*layers)

        self.input_complex = input_complex
        if self.input_complex:
            self.nl = NonLinear_Int2Phase()

    def forward(self, input_field):
        x = self.nl(input_field) if self.input_complex else input_field
        return self.layers(x)
