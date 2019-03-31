#imports Pytorch Tutorial

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json
import argparse
plt.ion()   # interactive 

# for Classifier 
from torch import nn

from collections import OrderedDict

# for image processing
import PIL
from PIL import Image

# for prediction
import torch.nn.functional as F
# sanity check
from matplotlib.ticker import FormatStrFormatter
structure = {"vgg16":25088, "alexnet":9216, "densenet121":1024}
arch = {"vgg16":25088, "alexnet":9216, "densenet121":1024}
structures = {"vgg16":25088, "alexnet":9216, "densenet121":1024}
def load_data(where  = "./flowers" ):
    
    '''
    Arguments : Path to data
    Returns : Dataloaders for train, validation and test datasets
    Function receives the location of the image files,and transforms ( + data augmentation on train set) and gives out tensor for CNN
    '''

    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    
    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])


    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # The data loaders are going to use to load the data to the NN(no shit Sherlock)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    vloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)



    return trainloader , vloader, testloader


# Defining Classifier Networks

def nn_setup(structure='vgg16',dropout=0.5, hidden_layer1 = 120,lr = 0.001, power='gpu'):
    
    
    if structure == 'vgg16':
        model_conv = models.vgg16(pretrained=True)        
    elif structure == 'alexnet':
        model_conv = models.alexnet(pretrained = True)
    elif structure == 'densenet121':
        model_conv = models.densenet121(pretrained=True)
    
    else:
        print("Exercise only trains and accepts Classifier-Networks vgg16, alexnet, and densenet121.")
        
    
        
    for param in model_conv.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(structures[structure], hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))
        
        
        model_conv.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer_conv = optim.Adam(model_conv.classifier.parameters(), lr )
        model_conv.cuda()
        
        return model_conv, criterion, optimizer_conv 

    
model_conv, criterion, optimizer_conv = nn_setup(structure='vgg16',dropout=0.5, hidden_layer1 = 120,lr = 0.001, power='gpu')

data_dir = "./flowers"
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # The data loaders are going to use to load the data to the NN(no shit Sherlock)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
vloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)


def train_network(model_conv, criterion, optimizer_conv, epochs = 3, print_every=20, loader=trainloader, power='gpu'):
    '''
    Arguments: The model, the criterion, the optimizer, the number of epochs, teh dataset, and whether to use a gpu or not
    Returns: Nothing
    This function trains the model over a certain number of epochs and displays the training,validation and accuracy every "print_every" step using cuda if specified. The training method is specified by the criterion and the optimizer which are NLLLoss and Adam respectively
    '''
    steps = 0
    running_loss = 0

    print("Training in progress")
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(loader):
            steps += 1
            if torch.cuda.is_available() and power=='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer_conv.zero_grad()

            # Forward and backward passes
            outputs = model_conv.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_conv.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model_conv.eval()
                vlost = 0
                accuracy=0


                for ii, (inputs2,labels2) in enumerate(vloader):
                    optimizer_conv.zero_grad()
                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                        model_conv.to('cuda:0')

                    with torch.no_grad():
                        outputs = model_conv.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                vlost = vlost / len(vloader)
                accuracy = accuracy /len(vloader)



                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(vlost),
                       "Accuracy: {:.4f}".format(accuracy))


                running_loss = 0


    print("--------- Training finished")
    print("----------Epochs: {}".format(epochs))
    print("----------Steps: {}".format(steps))

def save_checkpoint(path='checkpoint.pth',structure ='densenet121', hidden_layer1=120,dropout=0.5,lr=0.001,epochs=10):
    '''
    Arguments: The saving path and the hyperparameters of the network
    Returns: Nothing
    This function saves the model at a specified checkpoint
    '''
    model_conv.class_to_idx = train_data.class_to_idx
    model_conv.cpu
    torch.save({'structure' :structure,
                'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'nb_of_epochs':epochs,
                'state_dict':model_conv.state_dict(),
                'class_to_idx':model_conv.class_to_idx},
                path)
    
    
def load_checkpoint(path='checkpoint.pth'):
    '''
    Arguments: The path of the checkpoint file
    Returns: The Neural Netowrk with all hyperparameters, weights and biases
    '''
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']

    model_conv,_,_ = nn_setup(structure , dropout,hidden_layer1,lr)

    model_conv.class_to_idx = checkpoint['class_to_idx']
    model_conv.load_state_dict(checkpoint['state_dict'])
  
# TODO: Process a PIL image for use in a PyTorch model

def process_image(image):
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor
    
    
def predict(image_path, model_conv, topk=5,power='gpu'):
    '''
    Arguments: The path to the image, the model, the number of prefictions and whether cuda will be used or not
    Returns: The "topk" most probable choices that the network predicts
    '''
    model_conv.class_to_idx =train_data.class_to_idx

    ctx = model_conv.class_to_idx
    if torch.cuda.is_available() and power=='gpu':
        model_conv.to('cuda:0')

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if power == 'gpu':
        with torch.no_grad():
            output = model_conv.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model_conv.forward(img_torch)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)
