# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(sys.path[0],'../..'))
import probfeat
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import pandas as pd
import time
import os.path as op
import utils
import json
from sklearn import metrics
from tqdm import tqdm
import yaml
import statistics
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Train a convolutional neural network on a specified dataset.")
    subparsers = parser.add_subparsers()
    file_parser = subparsers.add_parser('file', help="Use a configuration file to specify parameters")
    command_file_help = "Text file containing list of parameters. These are expected to be in the " \
                        "same format as the command line parameters. See '{} params -h' for help on " \
                        "parameters".format(sys.argv[0])
    file_parser.add_argument("file", help=command_file_help)
    command_line_help = "Use command line parameters. Enter '{} params -h' for help on parameters".format(sys.argv[0])
    params_parser = subparsers.add_parser('params', help=command_line_help)
    model_help = "Model to be tested. Model file must have two copies of network: 1. Named Net() which returns final " \
                 "softmax output. 2. Named Net_inner(), must have the same structure as Net() but returns outputs of " \
                 "all fully connected layers."
    params_parser.add_argument("-m", "--model", help=model_help, required=True)
    model_params_help = "Parameters to be passed to the model. These should be specified in the form of a "\
                        "dictionary '{key1: val1, key2: val2, ...}'. These are the keyword arguments **kwargs taken "\
                        "by the __init(**kwargs)__ method of the model file specified in --model. "
    params_parser.add_argument("--model_params", help=model_params_help)
    device_help = "CUDA device ID. If not specified: GPU 0 will be used if available, else CPU"
    params_parser.add_argument("--device", help=device_help, type=int, nargs='+')
    output_path_help = "Output path. All generated files will be stored in <output_path>/<valset_name>/<model>/"
    params_parser.add_argument("-o", "--output_folder", help=output_path_help)
    params_parser.add_argument("--weights", help="Path to weights to initialize network")
    params_parser.add_argument("--load_lenient", help="Use if weights being loaded are mismatched with model",
                               action='store_true')
    # params_parser.add_argument("--weights", help="Path to weights to initialize network", required=True)
    params_parser.add_argument("-b", "--batch_size", help="Batch size (default: 4)", type=int, default=64)

    dataset_group = params_parser.add_argument_group('Dataset command line parameters')
    dataset_group.add_argument("--valset_name", help="Name of training dataset (required)", required=True)
    valset_file_help = "Training dataset class file. Optional if -trn is MNIST or CIFAR10, required otherwise. " \
                         "Must inherit from PyTorch's Dataset class. Must implement __init__(), __len()__, and " \
                         "__getitem()__ methods. Must return samples in dictionary format {'data': data, 'label': label}"
    dataset_group.add_argument("--valset_file", help=valset_file_help)
    valset_params_help = "Parameters to be passed to training set class. These should be specified in the form of a " \
                           "dictionary '{key1: val1, key2: val2, ...}'. These are the keyword arguments **kwargs taken " \
                           "by the __init(**kwargs)__ method of the dataset file specified in -trf. Optional if -trn " \
                           "is MNIST or CIFAR10. Required otherwise."
    dataset_group.add_argument("--valset_params", help=valset_params_help)
    subset_group = params_parser.add_mutually_exclusive_group()
    include_classes_help = "List of classes to be included from base dataset. Needed only if the network is to be " \
                           "trained on a subset of classes"
    subset_group.add_argument("--include_classes", type=int, help=include_classes_help, nargs='+')
    exclude_classes_help = "List of classes to be excluded from base dataset. Needed only if the network is to be " \
                           "trained on a subset of classes"
    subset_group.add_argument("--exclude_classes", type=int, help=exclude_classes_help, nargs='+')


    root_args = parser.parse_args()
    args = None
    if 'file' in root_args:
        param_string_quoted = utils.get_args_from_file(root_args.file)
        args = params_parser.parse_args(param_string_quoted)
    else:
        args = root_args

    # Error checking
    try:
        if args.valset_params is not None:
            json.loads(args.valset_params.replace('\'', '"'))
    except ValueError:
        parser.error('Invalid argument for --valset_params. Argument must be a valid JSON string.')

    try:
        if args.model_params is not None:
            json.loads(args.model_params.replace('\'', '"'))
    except ValueError:
        parser.error('Invalid argument for --model_params. Argument must be a valid JSON string.')
    return args


def eval_net(dataloader):
    net.eval()
    # Let us look at how the network performs on the whole dataset
    softmax_mean_values=[0]*len(classes)
    logits_numpy=np.empty(0)
    softmax_numpy=np.empty(0)
    all_logits_numpy=np.empty(0)
    all_softmax_numpy=np.empty(0)
    correct = 0
    total = 0
    num_classes = len(classes)
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    y_true = np.zeros(len(dataloader.dataset))
    y_pred = np.zeros(len(dataloader.dataset))
    TruePositve=0
    TruePositve_count=0
    TrueNegative=0
    TrueNegative_count=0
    FalsePositive=0
    FalsePositive_count=0
    FalseNegative=0
    FalseNegative_count=0
    TruePositiveList=[];TrueNegativeList=[];FalsePositiveList=[];FalseNegativeList=[]
    with torch.no_grad():
        for data in tqdm(dataloader):
            images = data['data']
            if subset_labels:
                labels = torch.tensor(classes_subset.get_subset_label(data['label'].tolist()))
            else:
                labels = data['label']
            minibatch_size = len(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            p=torch.nn.functional.softmax(outputs , dim=1)          #softmax values from logits
            logits_val , predicted = torch.max(outputs.data, 1)      #taking max logits in logits_val
            softmax_val , predicted = torch.max(p.data, 1)           #taking  max softmax values in softmax_num
            
            for itr in range(len(predicted)):
                if labels[itr].item()==predicted[itr].item():
                    softmax_mean_values[labels[itr].item()]+=p[itr][labels[itr]].item()
                    
            classs=9  #specify the class make it as a command line argument
            for itr in range(len(predicted)):
                #TruePositive
                if labels[itr].item()==classs:
                    if labels[itr].item()==predicted[itr].item():
                        TruePositve+=outputs[itr][classs]
                        TruePositve_count+=1
                        TruePositiveList.append(outputs[itr][classs].item())

                #TrueNegative
                if labels[itr].item()!=classs and predicted[itr].item()!=classs:  
                    TrueNegative+=outputs[itr][classs]
                    TrueNegative_count+=1
                    TrueNegativeList.append(outputs[itr][classs].item())

                #FalsePositive
                if predicted[itr].item()==classs and labels[itr].item()!=classs:
                    FalsePositive+=outputs[itr][classs]
                    FalsePositive_count+=1
                    FalsePositiveList.append(outputs[itr][classs].item())

                #FalseNegative
                if predicted[itr].item()!=classs and labels[itr].item()==classs:
                    FalseNegative+=outputs[itr][classs]
                    FalseNegative_count+=1
                    FalseNegativeList.append(outputs[itr][classs].item())
            
            logits_num=logits_val.cpu().data.numpy()
            softmax_num=softmax_val.cpu().data.numpy()
            
            
            for itr in range(minibatch_size): 
                if labels[itr] == 1:
                    all_logits_numpy=np.append(all_logits_numpy,logits_num[itr])
                    all_softmax_numpy=np.append(all_softmax_numpy,softmax_num[itr])
                    if predicted[itr] == labels[itr]:
                        logits_numpy=np.append(logits_numpy , logits_num[itr])
                        softmax_numpy=np.append(softmax_numpy , softmax_num[itr])

            total += labels.size(0)
            y_true[total - labels.size(0): total] = labels.cpu().numpy()
            y_pred[total - labels.size(0): total] = predicted.cpu().numpy()
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(minibatch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    accuracy = correct/total

    print("all_softmax_numpy : ", np.mean(all_softmax_numpy, axis = None))
    print("softmax_numy : ", np.mean(softmax_numpy, axis = None))
    print("all_logits_numpy : ", np.mean(all_logits_numpy, axis = None))
    print("logits_numy : ", np.mean(logits_numpy, axis = None))
    print(" ")

    print("std_dev_all_softmax : ", statistics.pstdev(all_softmax_numpy))
    print("std_dev_softmax : ", statistics.pstdev(softmax_numpy))
    print("std_dev_all_logits : ", statistics.pstdev(all_logits_numpy))
    print("std_dev_logits_numpy : ", statistics.pstdev(logits_numpy))

    class_accuracy = np.array(class_correct) / np.array(class_total)
    #print("class_accuracy : ",class_accuracy)
    cm = metrics.confusion_matrix(y_true, y_pred)
    #print(cm)
    #print("softmax_mean_values : ",softmax_mean_values)
    softmax_mean_values=np.array(softmax_mean_values, dtype=float)
    softmax_mean_values/=1000
    for j in range(len(class_accuracy)):
        softmax_mean_values[j]/=class_accuracy[j]
    
    tp_numpy=np.array(TruePositiveList)
    tn_numpy=np.array(TrueNegativeList)
    fp_numpy=np.array(FalsePositiveList)
    fn_numpy=np.array(FalseNegativeList)

    return accuracy, class_accuracy, cm, softmax_mean_values


# plt.ion()
args = parse_args()

valset_name = args.valset_name
model_name = args.model
weights_path = args.weights
weight_file_name = os.path.basename(weights_path).split(".")[0] if weights_path else 'pretrained'
batch_size = args.batch_size
include_classes = args.include_classes
exclude_classes = args.exclude_classes
load_strict = not args.load_lenient

out_folder = None
if args.output_folder:
    out_folder = args.output_folder
elif weights_path:
    out_folder = os.path.dirname(weights_path)
else:
    print("Error: Either output folder or weights folder must be specified")
    sys.exit(-1)

device, device_list = utils.check_cuda_devices(args.device)
batch_multiplier = len(device_list)

valset_file = args.valset_file
val_params_string = args.valset_params
valset_base, classes = utils.load_dataset(valset_name, valset_file, val_params_string)
classes_subset = utils.ClassSubset(classes=classes, include_classes=include_classes, exclude_classes=exclude_classes)
subset_labels = classes_subset.subset_labels

if args.include_classes or args.exclude_classes:
    indices_train = classes_subset.get_subset_indices(valset_base)
    valset = torch.utils.data.Subset(valset_base, indices_train)
    num_categories = len(subset_labels)
else:
    valset = valset_base
    num_categories = len(classes)

valloader = torch.utils.data.DataLoader(valset, batch_size=batch_multiplier * batch_size, shuffle=False)

# model_params_string = args.model_params
model_params = utils.params_from_command_line(args.model_params)
net = utils.load_network(model_name, model_params, weights_path, device, strict=load_strict)
# net = utils.load_network(model_name, model_params_string, weights_path, device, strict=load_strict)
net.to(device)

print('Loaded pretrained weights from {}'.format(weights_path))

if len(device_list) > 1:
    net = nn.DataParallel(net, device_ids=device_list)
net.to(device)

criterion = nn.CrossEntropyLoss()

# Base dir is based on the training set (MNIST/CIFAR10 etc)
if not op.isdir(out_folder):
    print('Warning: output folder {} does not exist. Creating one.'.format(out_folder))
    os.makedirs(out_folder)

# Train the network
t0 = time.time()
time_stamps = [time.time()]

# Evaluate trained network
net.eval()

print("Evaluating {} ...".format(valset_name), end="", flush=True)
val_accuracy, class_val_accuracy, confusion_matrix, softmax_mean_values = eval_net(valloader)
time_stamps.append(time.time())
utils.print_elapsed_time(time_stamps)


# torch.save(net.state_dict(), net_out_path)
d = {'Val': 100*class_val_accuracy}
df = pd.DataFrame(d, index=classes)
df.loc['Total'] = {'Val': 100 * val_accuracy}
out_path = op.join(out_folder, '{}_{}_accuracies.csv'.format(valset_name, weight_file_name))
df.to_csv(out_path)




