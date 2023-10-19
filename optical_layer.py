from optical_unit import AngSpecProp, Lens, PhaseMask, Incoherent_Int2Complex
from utils import padding, tile_kernels, split_kernels
import shutil
import time
import numpy as np
import os
import torch
import math
from torch import nn
import matplotlib.pyplot as plt

class FourierConv(nn.Module):
    def __init__(self, whole_dim, phase_dim, pixel_size, focal_length, wave_lambda, weights=None):
        super(FourierConv, self).__init__()
        self.prop = AngSpecProp(
            whole_dim, pixel_size, focal_length, wave_lambda)
        self.lens = Lens(whole_dim, pixel_size, focal_length, wave_lambda)

        phase_init = weights["phase"] if weights else None
        self.phase = PhaseMask(whole_dim, phase_dim, phase=phase_init)

        scalar_init = weights["scalar"] if weights else None
        self.scalar = torch.ones(1, dtype=torch.float32)*0.1 if scalar_init is None else torch.tensor(
            weights["scalar"], dtype=torch.float32)
        self.w_scalar = nn.Parameter(self.scalar)

    def forward(self, input_field):
        x = self.prop(input_field)
        x = self.lens(x)
        x = self.prop(x)
        x = self.phase(x)
        x = self.prop(x)
        x = self.lens(x)
        x = self.prop(x)
        output = torch.abs(self.w_scalar*x)**2
        return output

class FourierConvComplex(nn.Module):
    def __init__(self, whole_dim, phase_dim, pixel_size, focal_length, wave_lambda, weights=None):
        super(FourierConvComplex, self).__init__()
        self.prop = AngSpecProp(whole_dim, pixel_size, focal_length, wave_lambda)
        self.lens = Lens(whole_dim, pixel_size, focal_length, wave_lambda)

        phase_init = weights["phase"] if weights else None
        self.phase = PhaseMask(whole_dim, phase_dim, phase=phase_init)

        scalar_init = weights["scalar"] if weights else None
        self.scalar = torch.ones(1, dtype=torch.float32)*0.1 if scalar_init is None else torch.tensor(
            weights["scalar"], dtype=torch.float32)
        self.w_scalar = nn.Parameter(self.scalar)

    def forward(self, input_field):
        x = self.prop(input_field)
        x = self.lens(x)
        x = self.prop(x)
        x = self.phase(x)
        x = self.prop(x)
        x = self.lens(x)
        x = self.prop(x)
        output = self.w_scalar*x
        return output

def train_complex(label_nopad, folder_prefix, epoch, whole_dim, phase_dim, wave_lambda, focal_length, pixel_size, compute_loss_region, factor):
    image = np.zeros((1, whole_dim, whole_dim))
    image[0, whole_dim//2, whole_dim//2] = 1
    image = torch.tensor(image, dtype=torch.complex64)
    label = padding(label_nopad, whole_dim)

    loss_slice = slice(whole_dim//2-compute_loss_region//2, whole_dim//2+compute_loss_region//2)

    def cropped_loss(output, target):
        diff = (output-target)[:, loss_slice, loss_slice]
        return torch.mean(torch.abs(diff)**2)

    onn = FourierConvComplex(whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
    onn, image, label = onn.cuda(), image.cuda(), label.cuda()
    optimizer = torch.optim.Adam(onn.parameters(), lr=0.5)
    # scheduler = LambdaLR(optimizer, lr_lambda=adjust_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=150)

    now = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))
    folder = folder_prefix + now
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(folder+'/results')
        shutil.copy(__file__, folder)
        params = {
            'whole_dim': whole_dim,
            'phase_dim': phase_dim,
            'wave_lambda': wave_lambda,
            'focal_length': focal_length,
            'pixel_size': pixel_size,
            'compute_loss_region': compute_loss_region,
            'factor': factor
        }
        torch.save(params, folder+'/params.pt')

    logging_file = folder+"/Fitting_loss.log"
    log_f = open(logging_file, 'w')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for epoch in range(1, epoch+1):
        # Compute prediction and loss
        pred = onn(image*factor)
        loss = cropped_loss(pred, label*factor)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        scheduler.step(loss)

        log_f.write(f"epoch: {epoch} loss: {loss:>8f} Current learning rate: {optimizer.param_groups[0]['lr']}\r\n")

        if epoch % 3000 == 0:
            print(f"epoch: {epoch} loss: {loss:>8f} Current learning rate: {optimizer.param_groups[0]['lr']}")
            print(f"Saving results for epoch {epoch}")
            end.record()
            torch.cuda.synchronize()
            print(
            f'Running time for 3000 epoch: {start.elapsed_time(end) / 1000} s')

            # print(f"Learning rate: {scheduler.get_lr()}")
            temp_phase = onn.phase.w_p.cpu().detach().numpy()
            temp_scalar = onn.w_scalar.cpu().detach().numpy()
            weights_dict = {
                'phase': temp_phase,
                'scalar': temp_scalar,
                'pred': pred,
                'label': label
            }
            torch.save(weights_dict, folder +
                '/results/weights_%05d.pth' % epoch)
            pred_show = torch.abs(pred).cpu().detach().numpy()[
                0, loss_slice, loss_slice]
            label_show = (torch.abs(label).cpu().detach().numpy()[
                0, loss_slice, loss_slice])*factor
            plt.subplot(131)
            plt.imshow(pred_show)
            plt.colorbar()
            plt.subplot(132)
            plt.imshow(label_show)
            plt.colorbar()
            plt.subplot(133)
            plt.imshow(pred_show-label_show)
            plt.colorbar()
            plt.savefig(folder+'/results/output_%05d.png' % epoch)

            if loss < 0.01:
                break
            start.record()
    
    log_f.close()
    return folder, loss

