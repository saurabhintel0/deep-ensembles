# Deep Ensembles

**Required Packages:**


* Cuda (10.1)(Driver version 418.39)
* pytorch (1.4.0)
* matplotlib (2.2.4)
* numpy (1.12.0)
* pandas (0.23.4)
* pip (9.0.1)
* python (3.6.0)
* pydot (1.0.29)
* python-dateutil (2.7.3)
* PyYAML (3.12)
* scipy (0.19.0)
* setuptools (40.2.0)
* wheel (0.31.1)
* argparse (1.1)
* json (2.0.9)
* sklearn (0.19.2)


**How To Run Project**
<br/>
```
usage: train_network.py [-h] [-m MODEL] [-w WEIGHTS] [-t TRAIN] [-ti TRAIN_INDICES] [-m MODEL] 
                [-b] [-o OUTPUT] [-e EPOCH]
                [-l LR] [-opt OPTIMIZER] [-ss] [-ld]

arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The directory where the model/yaml located.
  -w WEIGHTS, --weights WEIGHTS
                        Path to the file which contains the weights
  -t TRAIN, --trn TRAIN
                        Path to the train dataset.
  -ti TRAIN_INDICES --train_indices TRAIN_INDICES
                        Path to selected train indices.
  -m MODEL, --m MODEL   Model name.
  -b BATCH, -b BATCH
                        Batch size.
  -o OUTPUT, --output OUTPUT
                        Path of model performance (and weight file)
  -e EPOCH, --epoch EPOCH
                        no of Epoch
  -l LR, --lr LR        learning rate
  -opt OPTIMIZER, -opt OPTIMIZER
                        Optmizer used.
  -ss SCHEDULED_STPES, -sched_steps SCHEDULED_STEPS
                        From which epoch number will learning rate be decremented.
  -ld LEARNING_RATE_DECAY, --lr_decay LEARNING_RATE_DECAY
                        By which factor the learning rate should decay.
  
```

```
Example : python3 train_network.py params -trn CIFAR10 -trp "{'transform':'cifar10_train'}" -tep "{'train':false}" -op output_file_path -m resnet20 
                            -b 32 --epochs 80 --learning_rate 0.01 --optimizer sgd --sched_steps 35 70 --momentum 0.9 --lr_decay 0.1

Output: Saves model.pt (weightfile) and perfomance of the model.

```

```
usage: eval_network.py [-h] [-m MODEL] [-w WEIGHTS] [--valset_name]
                [-b] [--valset_params]
                
    
arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The directory where the model/yaml located.
  -w WEIGHTS, --weights WEIGHTS
                        Path to the file which contains the weights
  -v VALSET_NAME, --valset_name  
                        Name/Path of dataset the model is trained on
                        Path to the train dataset. 
  -b BATCH, -b BATCH
                        Batch size.
```


```
Example : python3 eval_network.py params -m resnet20 --valset_name CIFAR10 --valset_params "{'train':false}" --weights 'specify weight file path' -b 64
                            

Script used for extracting mean, standard deviation values of softmax and logits and accuracy of the model. 

```



```
usage: eval_ensemble.py  [-h] [--dataset_name ] [--dataset_params] [--models]
                
    
arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        The directory where the model/yaml located.
                        
  --dataset_name
                        Name of the dataset of all models in ensemble
                        
  --dataset_params
                        Use train or test data
                        
  --models
                        Path of the .yaml file which contains weight files of all the models in ensemble.
                        
```


```
Example : python3 eval_ensemble.py params --dataset_name CIFAR10 --dataset_aprams "{'train' : false}"  --models 'Path of the .yaml file which has weights of all models in the ensemble'
                            

Script used for evaluating softmax average and majority vote ensemble

```


```
usage: gen_per-class_subset.py  [-h] [-trn CIFAR10] [-trp] [--train_frac] [-o]
                
    
arguments:
  -h, --help            show this help message and exit
  --trn
                        Dataset name
  --train_frac 
                        Percentage of data (indices) to be selected for each class 
  
                        
```


```
Example : python3gen_per-class_subset.py -trn CIFAR10 -trp "{'train': false}" --train_frac 40 40 40 40 40 40 40 40 40 40 -o 'Specify the output path'
                            

Script used for selecting percentage of data for each class for training

```