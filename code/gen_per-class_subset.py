# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(sys.path[0],'../..'))
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import pandas as pd
import time
import os.path as op
import utils
import json
import torch.backends.cudnn as cudnn


eval_train = False

def parse_args():
    parser = argparse.ArgumentParser(description="Train a convolutional neural network on a specified dataset.")
    device_help = "CUDA device ID. If not specified: GPU 0 will be used if available, else CPU"
    # parser.add_argument("--device", help=device_help, type=int, nargs='+')


    parser.add_argument("-trn", "--trainset_name", help="Name of training dataset (required)", required=True)
    trainset_file_help = "Training dataset class file. Optional if -trn is MNIST or CIFAR10, required otherwise. " \
                         "Must inherit from PyTorch's Dataset class. Must implement __init__(), __len()__, and " \
                         "__getitem()__ methods. Must return samples in dictionary format {'data': data, 'label': label}"
    parser.add_argument("-trf", "--trainset_file", help=trainset_file_help)
    trainset_params_help = "Parameters to be passed to training set class. These should be specified in the form of a " \
                           "dictionary '{key1: val1, key2: val2, ...}'. These are the keyword arguments **kwargs taken " \
                           "by the __init(**kwargs)__ method of the dataset file specified in -trf. Optional if -trn " \
                           "is MNIST or CIFAR10. Required otherwise."
    parser.add_argument("-trp", "--trainset_params", help=trainset_params_help)
    parser.add_argument("--train_frac", type=float, nargs='+',
                        help="Percentage of samples to include per-class. Percentage should be specified for each class.")
    parser.add_argument("-b", "--batch_size", help="Batch size (default: 64)", type=int, default=64)
    parser.add_argument("-o", "--output_file", help="Output Numpy file containing indices of subset")

    # subset_train_group = params_parser.add_mutually_exclusive_group()
    # include_classes_help = "List of classes to be included from base dataset. Needed only if the network is to be " \
    #                         "trained on a subset of classes"
    # subset_train_group.add_argument("--include_classes", type=int, help=include_classes_help, nargs='+')
    # exclude_classes_help = "List of classes to be excluded from base dataset. Needed only if the network is to be " \
    #                         "trained on a subset of classes"
    # subset_train_group.add_argument("--exclude_classes", type=int, help=exclude_classes_help, nargs='+')
    # noremap_help = "Do not remap classes. Use only with include/exclude classes argument. This is mainly of interest " \
    #                "in a class-incremental learning scenario. Its behavior in other situations may lead to " \
    #                "unexpected results"
    # params_parser.add_argument("--noremap", help=noremap_help, action="store_true")

    args = parser.parse_args()

    # Error checking
    try:
        test_strings = [args.trainset_params]
        for s in test_strings:
            if s:
                json.loads(s.replace('\'', '"'))
    except ValueError:
        parser.error('Invalid argument for {}. Argument must be a valid JSON string.'.format(s))

    return args


# plt.ion()
args = parse_args()
batch_size = args.batch_size
trainset_name = args.trainset_name


trainset_file = args.trainset_file
train_params_string = args.trainset_params
trainset, classes = utils.load_dataset(trainset_name, trainset_file, train_params_string)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)


ground_truth = torch.zeros(len(trainset), dtype=torch.int64)

idx = 0
for d in trainloader:
    num_els = len(d['label'])
    ground_truth[idx:idx+num_els] = d['label']
    idx += num_els
ground_truth=ground_truth.numpy()

num_classes = len(classes)
class_totals = np.zeros(num_classes)
for i in range(num_classes):
    #class_totals[i] = torch.sum(ground_truth == i).item()
    class_totals[i] = np.sum(ground_truth == i)
    #print("class_totals = ",class_totals[i])	
class_perc = class_totals/len(trainset)

train_subset_totals = np.zeros(num_classes)
val_subset_totals = np.zeros(num_classes)

full_ind = np.arange(len(trainset))

train_ind = np.array([])
train_fracs = np.zeros(len(args.train_frac))
# suffix = ''
for i, e in enumerate(args.train_frac):
    # suffix = '{}{}'.format(suffix, e)
    if e == 0:
        train_fracs[i] = 1
    else:
        train_fracs[i] = e/100

for i in range(num_classes):
    class_full_ind = full_ind[ground_truth == i]
    if train_fracs[i] < 1:
        class_train_ind = np.random.choice(class_full_ind, int(train_fracs[i]*len(class_full_ind)), replace=False)
    else:
        class_train_ind = class_full_ind
    train_ind = np.append(train_ind, class_train_ind)
    train_subset_totals[i] = len(class_train_ind)
    
train_subset_perc = train_subset_totals / len(trainset)

output_file = args.output_file
with open(output_file, 'wb') as fp:
    np.save(fp, train_ind)
