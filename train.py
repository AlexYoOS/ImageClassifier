import time
import numpy as np

import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models

import matplotlib.pyplot as plt
import argparse

import utility_functions

ap = argparse.ArgumentParser(description='Train.py')

# Command Line ardguments
ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs


trainloader, v_loader, testloader = utility_functions.load_data(where)
utility_functions.load_data(where)

model_conv, optimizer_conv, criterion = utility_functions.nn_setup(structure,dropout,hidden_layer1,lr,power)

utility_functions.nn_setup(structure,dropout,hidden_layer1,lr,power)
utility_functions.train_network(model_conv, optimizer_conv, criterion, epochs, 10, trainloader, power)


utility_functions.save_checkpoint(path,structure,hidden_layer1,dropout,lr,epochs)


print("Model finished training")
