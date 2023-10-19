from optical_network import *
from utils import *
import shutil
import time
import numpy as np
import os
import torch
import argparse
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import platform
from optical_dataset import MyCIFAR10_readout
import logging
from os import listdir
from os.path import isfile, join
# ###########################################
# whole_dim = 400
# phase_dim = 400
# wave_lambda = 532e-9
# focal_length = 5e-3
# pixel_size = 4e-6
# ###########################################

# outdir = 'DDNN_w_PhaseNL_small_readout_cifar10_'
# net = DDNN_w_PhaseNL_small_3ch_readout

if platform.system().lower() == 'windows':
    server_dir = '.'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    dl_batch_size = 8
    dl_num_worker = 4
elif platform.system().lower() == 'linux':
    server_dir = '/data/xiaoyun/Elec-Opt-D2NN/'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    dl_batch_size = 64
    dl_num_worker = 16

batch_size = dl_batch_size
num_workers = dl_num_worker
pre_train_dir = []
start_epoch = 0


whole_dim = 1000
phase_dim = 800
wave_lambda = 532e-9
focal_length = 6e-2
pixel_size = 8e-6


if __name__ == '__main__':
    ########################################################################
    # read args
    parser = argparse.ArgumentParser(description='Experiment fig1 params')
    parser.add_argument('-e', '--exp_idx', type=str, help='experiment index')
    parser.add_argument('-g', '--gpu_idx', type=int, help='GPU index', default=0)
    parser.add_argument('-l', '--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('-bs', '--batch_size', type=int, help='batch size', default=64)

    args = parser.parse_args()
    experiment_idx = args.exp_idx
    gpu_idx = args.gpu_idx
    lr = args.lr
    batch_size = args.batch_size
    dl_batch_size = batch_size
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx) 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device used for computing is %s' % device)

    ## decide experiment index
    # 3 layers neural network
    if experiment_idx == '33l':
        # cifar 3 layers best
        whole_dim = 2000
        phase_dim = 1200
        wave_lambda = 532e-9
        focal_length = 14.5e-2
        pixel_size = 8e-6
        net = DDNN_w_PhaseNL_NP_1ch_3layer_readout 
        out_sub_dir = 'results_nc/DDNN_end2end_cifar_3layers_2000_'
    elif experiment_idx == '37l':
        # cifar 3 layers 7 masks 
        whole_dim = 2000
        phase_dim = 1200
        wave_lambda = 532e-9
        focal_length = 14.5e-2
        pixel_size = 8e-6
        net = DDNN_w_PhaseNL_NP_3ch_7mask_readout 
        out_sub_dir = 'results_nc/DDNN_end2end_cifar_3layers_7mask_2000_'

    # define transforms and datasets
    norm_mean = [0.485, 0.456, 0.406]  # 均值
    norm_std = [0.229, 0.224, 0.225]  # 方差
    train_transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize(norm_mean, norm_std),
        transforms.RandomHorizontalFlip(),  # 
        transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 
        transforms.RandomCrop(32, padding=4)  # 
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    data_train = MyCIFAR10_readout(
        server_dir, train=True, whole_dim=whole_dim, transform=train_transform, num_used=0)
    data_test = MyCIFAR10_readout(
        server_dir, train=False, whole_dim=whole_dim, transform=test_transform, num_used=0)
    trainloader = DataLoader(data_train, batch_size=dl_batch_size, shuffle=True, num_workers=dl_num_worker, pin_memory=True)
    testloader = DataLoader(data_test, batch_size=dl_batch_size, shuffle=True, num_workers=dl_num_worker, pin_memory=True)

    ########################################################################
    # define neural network
    ddnn = net(whole_dim, phase_dim, pixel_size, focal_length, wave_lambda)
    # ddnn = ddnn.cuda()
    ddnn.to(device)

    #
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddnn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=20)

    if pre_train_dir == []:
        now = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))
        folder = server_dir + out_sub_dir + now
        print('Save results to dir: %s' % folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(folder+'/models')
            shutil.copy(__file__, folder)
            shutil.copy('optical_network.py', folder)
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
                handlers=[
                    logging.FileHandler(folder+"/training.log"),
                    logging.StreamHandler()
                ])
            params = {
                'whole_dim': whole_dim,
                'phase_dim': phase_dim,
                'wave_lambda': wave_lambda,
                'focal_length': focal_length,
                'pixel_size': pixel_size
            }
            torch.save(params, folder+'/params.pt')
    else:
        folder = pre_train_dir
        print('Save results to dir: %s' % folder)
        # get the latest model
        model_files = [f for f in listdir(folder + '/models') if isfile(join(folder + 'models', f))]
        model_files.sort(reverse=True)
        model_file = folder + '/models/' + model_files[0]
        ddnn.load_state_dict(torch.load(model_file))
        start_epoch = int(model_files[0][:-4])
        logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
                handlers=[
                    logging.FileHandler(folder+"/training.log"),
                    logging.StreamHandler()
                ])

    ########################################################################
    # start training
    best_epoch, best_test_acc = 0, 0
    for epoch in range(start_epoch, 200):  # training
        running_loss = 0.0
        correct, total = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = ddnn(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            # loss_optimizer.step()
            # calculate training data accuracy
            # logits = loss_func.get_logits(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.reshape(predicted.shape)).sum().item()
            # print statistics
            running_loss += loss.item()
            if i % 25 == 24:    # print every 2000 mini-batches
                logging.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 25:.3f}')
                running_loss = 0.0
        end.record()

        logging.info(f"Epoch {epoch}:")
        logging.info(
            f'Accuracy of the network on the 50000 training images: {100 * correct / total} %')
        logging.info(f'Current learning rate: {optimizer.param_groups[0]["lr"]}')
        
        torch.cuda.synchronize()
        print(
            f'Running time for one epoch: {start.elapsed_time(end) / 1000} s')

        scheduler.step(loss.item())

        correct, total = 0, 0
        with torch.no_grad():
            torch.save(ddnn.state_dict(), folder + '/models/%03d.pth' % epoch)
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = ddnn(images)
                # the class with the highest energy is what we choose as prediction
                # logits = loss_func.get_logits(outputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.reshape(predicted.shape)).sum().item()

            test_acc = 100 * correct / total
            logging.info(
                f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
            logging.info(
                f'Best epoch: {best_epoch}, best test acc: {best_test_acc} %')
        logging.info(
            f'Running time for one epoch: {start.elapsed_time(end) / 1000} s')

    logging.info('Finished Training')