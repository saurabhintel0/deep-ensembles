# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(sys.path[0],'../..'))
import probfeat
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
from sklearn import metrics
import yaml
import pickle
from probfeat.networks.network import Network
from sklearn.metrics import confusion_matrix
from singleclass_gaussian import singleclass_gaussian


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
    model_help = "YAML file specifying set of models."
    params_parser.add_argument("-m", "--models", help=model_help, required=True)
    device_help = "CUDA device ID. If not specified: GPU 0 will be used if available, else CPU"
    params_parser.add_argument("--device", help=device_help, type=int, nargs='+')
    output_path_help = "Output path. All generated files will be stored in <output_path>/<dataset_name>/<model>/"
    params_parser.add_argument("-o", "--output_folder", help=output_path_help)
    params_parser.add_argument("-b", "--batch_size", help="Batch size (default: 4)", type=int, default=64)

    dataset_group = params_parser.add_argument_group('Dataset command line parameters')
    dataset_group.add_argument("--dataset_name", help="Name of evaluation dataset (required)", required=True)
    dataset_file_help = "Training dataset class file. Optional if -trn is MNIST or CIFAR10, required otherwise. " \
                         "Must inherit from PyTorch's Dataset class. Must implement __init__(), __len()__, and " \
                         "__getitem()__ methods. Must return samples in dictionary format {'data': data, 'label': label}"
    dataset_group.add_argument("--dataset_file", help=dataset_file_help)
    dataset_params_help = "Parameters to be passed to training set class. These should be specified in the form of a " \
                           "dictionary '{key1: val1, key2: val2, ...}'. These are the keyword arguments **kwargs taken " \
                           "by the __init(**kwargs)__ method of the dataset file specified in -trf. Optional if -trn " \
                           "is MNIST or CIFAR10. Required otherwise."
    dataset_group.add_argument("--dataset_params", help=dataset_params_help)

    root_args = parser.parse_args()
    args = None
    if 'file' in root_args:
        param_string_quoted = utils.get_args_from_file(root_args.file)
        args = params_parser.parse_args(param_string_quoted)
    else:
        args = root_args

    # Error checking
    try:
        if args.dataset_params is not None:
            json.loads(args.dataset_params.replace('\'', '"'))
    except ValueError:
        parser.error('Invalid argument for --dataset_params. Argument must be a valid JSON string.')

    return args


def compute_scores(dataset_name, dataloader, classes_subset=None):
    len_dataset = len(dataloader.dataset)
    softmax = torch.zeros(num_categories, len_dataset)
    gt = torch.zeros(len_dataset)
    # ll = np.zeros((num_layers, num_categories, len_dataset))

    scores = dict()
    for l in layer_names:
        scores[l] = np.zeros((num_categories, len_dataset))
    scores['softmax'] = torch.zeros(num_categories, len_dataset)
    # dists = np.zeros((num_layers, batch_size, categories))

    with torch.no_grad():
        count = 0
        time_inf = 0
        time_score = 0
        for k, data in enumerate(tqdm(dataloader, desc=dataset_name)):
            # print(k)
            t0 = time.time()
            inputs = data['data'].to(device)
            # labels = data['label'].to(device)
            if classes_subset:
                labels = torch.tensor(classes_subset.get_subset_label(data['label'].tolist()))
            else:
                labels = data['label']
            labels = labels.to(device)
            num_im = inputs.shape[0]
            output_softmax, outputs_inner = net_inner(inputs)
            time_inf += time.time() - t0
            t1 = time.time()
            for l in layer_names:
                oi = outputs_inner[l].cpu().numpy()
                # print(oi.shape)
                if 'pca' in distribution[l]:
                    oi = distribution[l]['pca'].transform(oi)
                    # print(oi.shape)
                if distribution['type'] == 'normal':  # TODO: Harmonize API calls across all distribution types
                    dists_local = distribution[l]['model'].score_samples(oi, regularize)
                elif distribution['type'] == 'gmm':
                    dists_local = np.zeros((num_im, num_categories))
                    for cc in distribution[l]['model']:
                        dists_local[:, cc] = distribution[l]['model'][cc].score_samples(oi)
                    # print(dists_local.shape)
                scores[l][:, count: count + num_im] = dists_local.T
            scores['softmax'][:, count: count + num_im] = output_softmax.t()
            time_score += time.time() - t1
            # print(labels.shape)
            gt[count:count + num_im] = labels
            count += num_im
        gt = gt.numpy()
        scores['softmax'] = scores['softmax'].numpy()
        # print('Inference', time_inf)
        # print('Scoring', time_score)
        # softmax = softmax.numpy()
        # ll = ll.numpy()
    return gt, scores



# plt.ion()
args = parse_args()

dataset_name = args.dataset_name
yaml_file = args.models
batch_size = args.batch_size
out_folder = None
if args.output_folder:
    out_folder = args.output_folder

device, device_list = utils.check_cuda_devices(args.device)
batch_multiplier = len(device_list)

dataset_file = args.dataset_file
val_params_string = args.dataset_params
dataset, classes = utils.load_dataset(dataset_name, dataset_file, val_params_string)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_multiplier * batch_size, shuffle=False)

with open(yaml_file) as file:
    models_dict = yaml.load(file, Loader=yaml.SafeLoader)

net_list = dict()
net_inner_list = dict()
class_labels = dict()
pred_local = dict()
pred_global = dict()
maxvals = dict()
distribution = dict()
# layer_names = dict()
# pool_factors = dict()
scores = dict()
len_dataset = len(dataset)
num_categories = dict()
no_of_models=len(models_dict)

mdl=-1
load_network_list=[]
for model in models_dict:
    mdl+=1
    model_name = models_dict[model]['name']
    model_params = models_dict[model]['params']
    weights = models_dict[model]['weights']
    
    load_network_list.append(utils.load_network(model_name, model_params, weights))
    load_network_list[-1].to(device)
    load_network_list[-1].eval()



count = 0
ground_truth = -1*np.ones(len_dataset)

regularize = -1
model_no=-1
number_of_classes=models_dict["model0"]["params"]["num_classes"]
outputs_list=[]
class_correct = dict()
predlist=torch.zeros(0,dtype=torch.long,device='cpu')
lbllist=torch.zeros(0,dtype=torch.long,device='cpu')
classwise_accuracy_mv = [0]*number_of_classes       #list for classwise accuracy for majority vote.. 
classwise_data_list = [0]*number_of_classes
with torch.no_grad():
    correct_softmax_avg=0
    correct_majority_vote=0
    total=0
    for data in dataloader:
        model_no=-1
        images = data['data']
        labels = data['label']
        minibatch_size = len(labels)
        images, labels = images.to(device), labels.to(device)
        num_items = len(labels)
        ground_truth[count:count + num_items] = labels.cpu()
        for model in models_dict:
            model_no+=1
            outputs_list.append(load_network_list[model_no](images))


        #AVERAGE VOTE.........
        output_batch_tensor=torch.zeros([minibatch_size,number_of_classes])
        majority_vote_tensor=torch.zeros([minibatch_size,number_of_classes])
        majority_vote_list=[]
        for img in range(minibatch_size):   
            for itr in range(number_of_classes): 
                for mdl_no in range(len(outputs_list)):   
                    output_batch_tensor[img][itr]+=outputs_list[mdl_no][img][itr]
        
        final_values , predicted = torch.max(output_batch_tensor.data, 1)
        predlist = torch.cat([predlist , predicted.view(-1).cpu()]) #for confusion matrix in avg vote of softmax value
        lbllist = torch.cat([lbllist , labels.view(-1).cpu()])
        #print("predicted : ",predicted)
        np_predicted=predicted.cpu().data.numpy()
        np_labels=labels.cpu().data.numpy()
        correct_softmax_avg += np.sum(np_predicted == np_labels)


        #MAJORITY VOTE.........
        for modl in range(minibatch_size):   # batch_size times..
            for ml in range(len(outputs_list)):
                majority_vote_list.append(torch.argmax(outputs_list[ml][modl]))
            mv_tensor_pi=torch.stack(majority_vote_list)   #mv_tensor_pi=majority vote tensor per test image
            majority_vote_list.clear()
            pred_labels , frequency = torch.mode(mv_tensor_pi)
            np_pred_labels=pred_labels.cpu().data.numpy()
            classwise_data_list[np_labels[modl]]+=1
            if np_pred_labels == np_labels[modl]:
                correct_majority_vote+=1
                classwise_accuracy_mv[np_labels[modl]]+=1

        outputs_list.clear()


print("correct_softmax_avg: ",correct_softmax_avg)
print("correct_majority_vote : ",correct_majority_vote)
conf_mat = confusion_matrix(lbllist.numpy() , predlist.numpy())
class_accuracy = 100*conf_mat.diagonal()/conf_mat.sum(1)

print("class_accuracy softmax value acc : " , class_accuracy)
print("class accuracy majority vote : " , classwise_accuracy_mv)
