# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(sys.path[0],'../..'))
#print(os.getcwd())
import probfeat
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as sched
import argparse
import torch.optim as optim
import importlib
import pandas as pd
import time
from datetime import datetime as dt
import os.path as op
import utils
import json
import pprint
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]='1'

eval_train = False

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
    output_path_help = "Output path. All generated files will be stored in <output_path>/<trainset_name>/<model>/"
    params_parser.add_argument("-op", "--output_path", help=output_path_help, required=True)
    params_parser.add_argument("--weights", help="Path to weights to initialize network")
    params_parser.add_argument("--init", help="Initialization function", default='xavier_uniform')
    params_parser.add_argument("--init_params", help="Parameters to be passed to initialization function")
    layers_help = "Provide a list of layers to be trained. Default: all"
    params_parser.add_argument("--layers", nargs='+', help=layers_help)


    dataset_group = params_parser.add_argument_group('Dataset command line parameters')
    dataset_group.add_argument("-trn", "--trainset_name", help="Name of training dataset (required)", required=True)
    trainset_file_help = "Training dataset class file. Optional if -trn is MNIST or CIFAR10, required otherwise. " \
                         "Must inherit from PyTorch's Dataset class. Must implement __init__(), __len()__, and " \
                         "__getitem()__ methods. Must return samples in dictionary format {'data': data, 'label': label}"
    dataset_group.add_argument("-trf", "--trainset_file", help=trainset_file_help)
    trainset_params_help = "Parameters to be passed to training set class. These should be specified in the form of a " \
                           "dictionary '{key1: val1, key2: val2, ...}'. These are the keyword arguments **kwargs taken " \
                           "by the __init(**kwargs)__ method of the dataset file specified in -trf. Optional if -trn " \
                           "is MNIST or CIFAR10. Required otherwise."
    dataset_group.add_argument("-trp", "--trainset_params", help=trainset_params_help)
    dataset_group.add_argument("--train_ind", help="Numpy file of indices of elements from training dataset. Use for "
                                                   "training with subset.")
    dataset_group.add_argument("--test_ind", help="Numpy file of indices of elements from test dataset. Use for "
                                                   "testing with subset.")
    dataset_group.add_argument("--testset_name", help="Name of training dataset (required)")
    testset_file_help = "Test dataset class file. Same format as -trf"
    testset_params_help = "Parameters to be passed to test set class. Same format as -trp."
    dataset_group.add_argument("-tef", "--testset_file", help=testset_file_help)
    dataset_group.add_argument("-tep", "--testset_params", help=testset_params_help)
    subset_train_group = params_parser.add_mutually_exclusive_group()
    include_classes_help = "List of classes to be included from base dataset. Needed only if the network is to be " \
                            "trained on a subset of classes"
    subset_train_group.add_argument("--include_classes", type=int, help=include_classes_help, nargs='+')
    exclude_classes_help = "List of classes to be excluded from base dataset. Needed only if the network is to be " \
                            "trained on a subset of classes"
    subset_train_group.add_argument("--exclude_classes", type=int, help=exclude_classes_help, nargs='+')
    noremap_help = "Do not remap classes. Use only with include/exclude classes argument. This is mainly of interest " \
                   "in a class-incremental learning scenario. Its behavior in other situations may lead to " \
                   "unexpected results"
    params_parser.add_argument("--noremap", help=noremap_help, action="store_true")
    subset_test_group = params_parser.add_mutually_exclusive_group()
    subset_test_group.add_argument("--include_classes_test", type=int, help=include_classes_help, nargs='+')
    subset_test_group.add_argument("--exclude_classes_test", type=int, help=exclude_classes_help, nargs='+')
    params_parser.add_argument("--noremap_test", help=noremap_help, action="store_true")

    hyperparams_group = params_parser.add_argument_group('Training hyper-parameters')
    hyperparams_group.add_argument("--learning_rate", help="Learning Rate (default: 0.02)", type=float, default=0.02)
    hyperparams_group.add_argument("--epochs", help="Number of epochs (default: 10)", type=int, default=10)
    hyperparams_group.add_argument("-b", "--batch_size", help="Batch size (default: 4)", type=int, default=4)
    hyperparams_group.add_argument("--optimizer", choices={'sgd', 'adam'}, help="Optimizer type (default SGD)",
                                   default='sgd')
    hyperparams_group.add_argument("--save_interval", type=int, help="Save interval")
    hyperparams_group.add_argument("--save_last_epochs", type=int, help="Saves model for last few epochs")
    hyperparams_group.add_argument("--sched_steps", type=int, nargs='+',
                                   help="Scheduler steps for decayed learning rate (default [10, 20, 50, 80]")
    hyperparams_group.add_argument("--momentum", help="SGD momentum", type=float, default=0.4)
    hyperparams_group.add_argument("--weight_decay", help="Weight decay", type=float, default=0.0)
    hyperparams_group.add_argument("--lr_decay", help="Learning rate decay", type=float, default=0.1)

    root_args = parser.parse_args()
    args = None
    if 'file' in root_args:
        param_string_quoted = utils.get_args_from_file(root_args.file)
        args = params_parser.parse_args(param_string_quoted)
    else:
        args = root_args

    # Error checking
    try:
        test_strings = [args.trainset_params]
        test_strings.append(args.testset_params)
        test_strings.append(args.model_params)
        test_strings.append(args.init_params)
        for s in test_strings:
            if s:
                json.loads(s.replace('\'', '"'))
    except ValueError:
        parser.error('Invalid argument for {}. Argument must be a valid JSON string.'.format(s))

    return args

def eval_net(dataloader, classes_subset):
    net.eval()
    ind=np.empty(0)
    # Let us look at how the network performs on the whole dataset
    correct = 0
    total = 0
    subset_labels = classes_subset.subset_labels

    class_correct = dict()
    class_total = dict()
    if subset_labels:
        num_classes = len(subset_labels)
        for k in subset_labels:
            class_correct[subset_labels[k]] = 0
            class_total[subset_labels[k]] = 0
    else:
        num_classes = len(classes)
        for i in range(num_classes):
            class_correct[i] = 0
            class_total[i] = 0
    with torch.no_grad():
        for data in dataloader:
            images = data['data']
            if subset_labels:
                labels = torch.tensor(classes_subset.get_subset_label(data['label'].tolist()))
            else:
                labels = data['label']
            minibatch_size = len(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)


            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(minibatch_size):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
    accuracy = correct/total
    class_accuracy = dict()
    for k in class_total:
        class_accuracy[k] = 100*class_correct[k]/class_total[k]
    return accuracy, class_accuracy


# plt.ion()
args = parse_args()

batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
trainset_name = args.trainset_name
testset_name = args.trainset_name if args.testset_name is None else args.testset_name
model_name = args.model
base_dir = args.output_path
weights_path = args.weights
optim_gamma = args.lr_decay
sched_milestones = [10, 20, 50, 80] if args.sched_steps is None else args.sched_steps
include_classes = args.include_classes
exclude_classes = args.exclude_classes
noremap = args.noremap


include_classes_test = args.include_classes_test
exclude_classes_test = args.exclude_classes_test
noremap_test = args.noremap_test

device, device_list = utils.check_cuda_devices(args.device)
batch_multiplier = len(device_list)

params = {'batch_size': batch_size,
          'num_gpu': batch_multiplier,
          'epochs': epochs,
          'learning_rate': learning_rate,
          'sched_milestones': sched_milestones,
          'optim_gamma': optim_gamma,
          'optimizer': args.optimizer,
          'weight_decay': args.weight_decay,
          'momentum': args.momentum,
          'weights': args.weights,
          'layers': args.layers,
          'init': args.init
          }


trainset_file = args.trainset_file
train_params_string = args.trainset_params
trainset_base, classes = utils.load_dataset(trainset_name, trainset_file, train_params_string)
classes_subset_train = utils.ClassSubset(classes=classes, include_classes=include_classes,
                                         exclude_classes=exclude_classes, noremap=noremap)
subset_labels_train = classes_subset_train.subset_labels

if args.train_ind:#train
    indices_train = np.load(args.train_ind).astype('int64')
    trainset = torch.utils.data.Subset(trainset_base, indices_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_multiplier * batch_size, shuffle=True)
elif include_classes or exclude_classes:
    indices_train = classes_subset_train.get_subset_indices(trainset_base)
    trainset = torch.utils.data.Subset(trainset_base, indices_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_multiplier*batch_size, shuffle=True)
    # classes = include_classes
else:
    trainset = trainset_base
    trainloader = torch.utils.data.DataLoader(trainset_base, batch_size=batch_multiplier*batch_size, shuffle=True)

testset_file = args.testset_file
test_params_string = args.testset_params
testset_base, _ = utils.load_dataset(testset_name, testset_file, test_params_string)
classes_subset_test = utils.ClassSubset(classes=classes, include_classes=include_classes_test,
                                        exclude_classes=exclude_classes_test, noremap=noremap_test)
subset_labels_test = classes_subset_test.subset_labels

if args.test_ind:#test
    indices_test = np.load(args.test_ind).astype('int64')
    testset = torch.utils.data.Subset(trainset_base, indices_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_multiplier*batch_size, shuffle=True)
elif include_classes_test or exclude_classes_test:
    indices_test = classes_subset_test.get_subset_indices(testset_base)
    testset = torch.utils.data.Subset(testset_base, indices_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_multiplier*batch_size, shuffle=False)
    # classes = include_classes
else:
    testset = testset_base
    testloader = torch.utils.data.DataLoader(testset_base, batch_size=batch_multiplier*batch_size, shuffle=True)

print('Model hyperparams:')
pprint.pprint(params)

time_stamp = '{:%d%b%y_%H%M%S}'.format(dt.now())

# Load network with pretrained weights (if any)
model_params = utils.params_from_command_line(args.model_params)
net = utils.load_network(model_name, model_params)
if weights_path is None:
    if args.init_params:
        init_params = utils.params_from_command_line(args.init_params)
    else:
        init_params = None
    init_func = utils.get_init_func(args.init, init_params)
    net.apply(init_func)
    # net.apply(utils.conv_init)
else:
    pretrained_weights = torch.load(weights_path, map_location=device)
    network_params = dict(net.named_parameters())
    ignore_params = []
    # Check for size mismatches between similarly named parameters
    for pp in pretrained_weights:
        if pp in network_params:
            if pretrained_weights[pp].shape != network_params[pp].shape:
                ignore_params.append(pp)
    for pp in ignore_params:
        print('WARNING: size mismatch for {}. Ignoring {} from pretrained weights.'.format(pp, pp))
        pretrained_weights.pop(pp)

    print('Loaded pretrained weights from {}'.format(weights_path))
    print('WARNING: {}'.format(net.load_state_dict(pretrained_weights, strict=False)))

if len(device_list) > 1:
    net = nn.DataParallel(net, device_ids=device_list)
    cudnn.benchmark = True
net.to(device)

criterion = nn.CrossEntropyLoss()

# Train paramters of specified layers only. Freeze rest.
# Set requires_grad = True only for layers to be trained.
if args.layers:
    for param in net.parameters():
        param.requires_grad = False
    d = dict(net.named_modules())
    params_list = nn.ParameterList()
    for layer_name in args.layers:
        if layer_name in d:
            layer = d[layer_name]
            for param in layer.parameters():
                param.requires_grad = True
            params_list.extend(layer.parameters())
        else:
            print("Warning: Layer {} not found".format(layer_name))
else:
    params_list = net.parameters()

if args.optimizer is None or args.optimizer == 'sgd':
    optimizer = optim.SGD(params_list, lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(params_list, lr=learning_rate, weight_decay=args.weight_decay)
scheduler = sched.MultiStepLR(optimizer, milestones=sched_milestones, gamma=optim_gamma)

# Base dir is based on the training set (MNIST/CIFAR10 etc)
subset_suffix = classes_subset_train.subset_suffix
if subset_suffix is not None:
    model_name = '{}_{}'.format(model_name, subset_suffix)
out_folder = os.path.join(base_dir, trainset_name, model_name, time_stamp)
if not op.isdir(base_dir):
    print('Warning: output folder {} does not exist. Creating one.')
if not op.isdir(out_folder):
    os.makedirs(out_folder)
weight_file_name = '{}_{}_{}'.format(trainset_name, model_name, time_stamp)

# Train the network
t0 = time.time()
net.train()

print('Beginning training')
time_stamps = [time.time()]

train_accuracies = []
test_accuracies = []
training_loss = []
epoch_list = []
best_test_accuracy = 0.0
best_train_accuracy = 0.0


classwise_acc_train=np.arange(0, 12)
classwise_acc=np.arange(0, 12)
for epoch in range(epochs):  # loop over the dataset multiple times
    count=0
    classwise_accuracy_train = np.zeros(10)
    epoch_list.append(epoch)
    print('Epoch {}'.format(epoch + 1))
    num_minibatch = int(len(trainset)/(batch_multiplier*batch_size)+1)
    total = 0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs = data['data']
        if subset_labels_train:
            labels = torch.tensor(classes_subset_train.get_subset_label(data['label'].tolist()))
        else:
            labels = data['label']

        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        training_loss.append(loss.item())
        loss.backward() 
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += torch.sum(predicted == labels).item()
        total += len(labels)
        sys.stdout.write('\r')
        sys.stdout.write('Minibatch {}/{}: Loss {:.3f}'.format(i + 1, num_minibatch, loss.item()))
        sys.stdout.flush()

    

    runtime_accuracy = correct/total
    test_accuracy, class_test_accuracy = eval_net(testloader, classes_subset_test)

    res_array = np.array(list(class_test_accuracy.values()))
    res_array = np.append(res_array,np.average(res_array))
    res_array = np.append(res_array,loss.item())
    classwise_acc=np.vstack((classwise_acc,res_array))

    
    train_accuracy , class_train_accuracy = eval_net(trainloader , classes_subset_train)
    res_array_train = np.array(list(class_train_accuracy.values()))
    res_array_train = np.append(res_array_train,np.average(res_array_train))
    res_array_train = np.append(res_array_train,loss.item())
    classwise_acc_train=np.vstack((classwise_acc_train,res_array_train))

    sys.stdout.write(', Accuracy {:.3f}%, Test accuracy {:.3f}%\n'.format(runtime_accuracy*100, test_accuracy*100))
    sys.stdout.flush()
    train_accuracies.append(runtime_accuracy)
    test_accuracies.append(test_accuracy)
    correct = 0
    total = 0
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        print("Saving checkpoint with {:.3f}% accuracy".format(100*test_accuracy))
        net_out_path = op.join(out_folder, '{}.pt'.format(weight_file_name))
        utils.save_model_dict(net, net_out_path)
        best_train_accuracy = runtime_accuracy
    if args.save_interval is not None:
        if epoch%args.save_interval == 0:
            print("Saving epoch {}".format(epoch))
            net_out_path = op.join(out_folder, '{}_{}.pt'.format(weight_file_name, epoch))
            utils.save_model_dict(net, net_out_path)

    if args.save_last_epochs is not None:
        if epoch >= (args.epochs-args.save_last_epochs):
            net_out_path = op.join(out_folder, '{}_{}_epoch_.pt'.format(weight_file_name, epoch+1))
            utils.save_model_dict(net, net_out_path)

    time_stamps.append(time.time())
    print('Epoch time ', end='')
    utils.print_elapsed_time(time_stamps)
    scheduler.step()
    net.train()

classwise_acc = pd.DataFrame(classwise_acc)
classwise_acc.to_csv(op.join(out_folder, 'classwise_accuracies_test_{}.csv'.format(time_stamp)))

classwise_acc_train = pd.DataFrame(classwise_acc_train)
classwise_acc_train.to_csv(op.join(out_folder, 'classwise_accuracies_train_{}.csv'.format(time_stamp)))

print('Finished Training')

t1 = time.time()

# Save training outputs - losses, accuracies
dict_accuracies = {'Train': 100*np.array(train_accuracies), 'Test': 100*np.array(test_accuracies)}
df_accuracies = pd.DataFrame(dict_accuracies, index=epoch_list)
df_accuracies.to_csv(op.join(out_folder, 'training_accuracies_{}.csv'.format(time_stamp)))

df_training_loss = pd.DataFrame(training_loss)
df_training_loss.to_csv(op.join(out_folder, 'training_loss_{}.csv'.format(time_stamp)))

# Evaluate trained network
net.eval()
train_accuracy, class_train_accuracy = eval_net(trainloader, classes_subset_train)
test_accuracy, class_test_accuracy = eval_net(testloader, classes_subset_test)


print('Total training time', t1-t0)
net_out_path = op.join(out_folder, '{}.pt'.format(weight_file_name))
# torch.save(net.state_dict(), net_out_path)
utils.save_model_dict(net, net_out_path)
# d_train = dict()
# d_test = dict()
d = {'Train': dict(), 'Test': dict()}
if subset_labels_train:
    for k in subset_labels_train:
        d['Train'][k] = class_train_accuracy[subset_labels_train[k]]
else:
    for k in class_train_accuracy:
        d['Train'][k] = class_train_accuracy[k]
if subset_labels_test:
    for k in subset_labels_test:
        d['Test'][k] = class_test_accuracy[subset_labels_test[k]]
else:
    for k in class_test_accuracy:
        d['Test'][k] = class_test_accuracy[k]
df = pd.DataFrame(d)

df.loc['Total'] = {'Test': 100 * best_test_accuracy, 'Train': 100 * best_train_accuracy}
df.to_csv(op.join(out_folder, '{}_accuracies.csv'.format(weight_file_name, time_stamp)))
# print(df)

hyper_params_name = 'hyper_params_' + time_stamp + '.txt'
hyper_params_file = os.path.join(out_folder, 'hyper_params_{}.txt'.format(time_stamp))
if args.model_params is not None:
    # model_params = json.loads(model_params_string.replace('\'', '"'))
    params = {**params, **model_params}
if train_params_string is not None:
    train_params = json.loads(train_params_string.replace('\'', '"'))
    params = {**params, **train_params}
if args.train_ind:
    params['train_ind'] = args.train_ind

with open(hyper_params_file, 'w') as fid:
    fid.write(json.dumps(params))

