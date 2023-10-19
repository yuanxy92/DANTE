# Implementation of Manuscript "Training large-scale optoelectronic neural networks with dual-neuron optical-artificial learning"

## How to use

### Our DANTE approach:

1. Global artificial learning: 

Modify the `server_dir` based on your own environment. Then, run:
`python train_electric_cifar10_readout.py -e 3b -g gpu_idx -l 0.01 -ep 200`

2. Local optical learning: 

Modify the `server_dir`,  `folder_to_fit`, and gpu index based on your own environment. Then, run:
`python train_electric_optical_kernel.py`

3. test accuracy:

Modify the  `folder_to_fit` based on your own environment. Then, run:
`python test_electric_optical_kernel.py -e 3bs -g gpu_index`

### Existing approach:

Modify the `server_dir` based on your own environment. 

run:
`python train_end2end_cifar10_readout.py -e 33l -g gpu_idx -l 0.01 -bs 32` for ONN-3-3

run:
`python train_end2end_cifar10_readout.py -e 37l -g gpu_idx -l 0.01 -bs 8` for ONN-3-7