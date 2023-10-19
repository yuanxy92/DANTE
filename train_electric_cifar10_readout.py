# -*- coding: utf-8 -*-
import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import platform
from electric_network import *
import shutil
from pytorch_metric_learning.losses import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
import argparse

if platform.system().lower() == 'windows':
    server_dir = './'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    dl_batch_size = 128
    dl_num_worker = 4
elif platform.system().lower() == 'linux':
    server_dir = '/data/xiaoyun/Elec-Opt-D2NN/'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    dl_batch_size = 512
    dl_num_worker = 16

batch_size = dl_batch_size
num_workers = dl_num_worker
experiment_id = 3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment fig1 params')
    parser.add_argument('-e', '--exp_idx', type=str, help='experiment index')
    parser.add_argument('-g', '--gpu_idx', type=int, help='GPU index', default=0)
    parser.add_argument('-l', '--lr', type=float, help='learning rate', default=0.005)
    parser.add_argument('-ep', '--epoch', type=int, help='total epoch number', default=200)
    parser.add_argument('-opt', '--optimization', type=str, help='optimization method', default='adam')

    args = parser.parse_args()
    experiment_idx = args.exp_idx
    gpu_idx = args.gpu_idx
    lr = args.lr
    epoch_num = args.epoch
    optm = args.optimization
    print(f'experiment_idx:{experiment_idx}, gpu_idx:{gpu_idx}, lr:{lr}, epoch_num:{epoch_num}, optm:{optm}')
    print ('Waiting 1 seconds before continuing ...')
    time.sleep(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx) 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device used for computing is %s' % device)
    print('Experimental index is %s' % experiment_idx)

    norm_mean = [0.485, 0.456, 0.406]  # 
    norm_std = [0.229, 0.224, 0.225]  # 
    train_transform = transforms.Compose([
        transforms.ToTensor(),  # 
        transforms.Normalize(norm_mean, norm_std),
        transforms.RandomHorizontalFlip(),  # 
        transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 
        transforms.RandomCrop(32, padding=4)  # 
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    ########################################################################
    # 3 layers neural network
    if experiment_idx == '3b':
        # cifar 3 layers best
        CNN = CNN_complex_3layers_best_small 
        out_sub_dir = 'results_nc/cifar_3layers_best_'
        fc_in = 324
        fc_out = 10
        batch_size = 512
    else:
        print('Wrong experiment id is given !!!')
        exit(1)

    # load dataset
    trainset = torchvision.datasets.CIFAR10(
        root=server_dir + "/data/", train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(
        root=server_dir + "/data/", train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # init optical network
    net = CNN(fc_in, fc_out)
    net.to(device)
    # compute optical fitting size
    layers = compute_fitting_size(net, (3, 32, 32))
    print ('Waiting 5 seconds before continuing ...')
    time.sleep(5)

    ########################################################################
    loss_func = nn.CrossEntropyLoss()

    if optm == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=30)
    elif optm == 'sgd':
        optimizer = torch.optim.SGD(
                net.parameters(),
                lr=lr, weight_decay=1e-4,
                momentum=0.9,
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=30)

    now = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))
    folder = server_dir + out_sub_dir + now
    print('Save results to dir: %s' % folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(folder+'/models')
        shutil.copy(__file__, folder)
        shutil.copy('electric_network.py', folder)
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(folder+"/training.log"),
                logging.StreamHandler()
            ])

    ########################################################################
    best_epoch, best_test_acc = 0, 0
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        running_loss = 0.0
        correct, total = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.reshape(predicted.shape)).sum().item()
            # print statistics
            running_loss += loss.item()
            if i % 40 == 39:    # print every 2000 mini-batches
                logging.info(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
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
            torch.save(net.state_dict(), folder + '/models/%03d.pth' % epoch)
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
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
    print('All the results saved to dir: %s' % folder)
