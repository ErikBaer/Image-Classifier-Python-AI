# define transforms
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import sub
import main

from collections import OrderedDict
from PIL import Image
import json


#define transforms

data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])]),
                   'valid': transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])}


# import data

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    dataset = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
                'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])}

    dataloader = {'train': torch.utils.data.DataLoader(dataset['train'], batch_size=32, shuffle=True),
                  'valid': torch.utils.data.DataLoader(dataset['valid'], batch_size=32, shuffle=True),
                  'test': torch.utils.data.DataLoader(dataset['test'], batch_size=32, shuffle=True)}

    return dataloader, dataset['train'].class_to_idx


# process image for inference

def process_image(image):
    im = Image.open(image)


    image = data_transforms['valid'](im)
    return image


# set device to cpu / gpu

def set_device(device):

    if device == 'gpu':
        if torch.cuda.is_available:
            device = torch.device('cuda')
            print ('Cuda is available - Running model on GPU!')
        else:
            device = torch.device('cpu')
            print('Cuda is NOT available - Running model on CPU!')
    else:
        device = torch.device('cpu')
        print('Running model on CPU!')



    return device


# define architecture

def set_arch(arch):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)

    elif arch == 'densenet':
        model = models.densenet(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)

    return model


# label mapping

def create_map_dict(json_file):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)

    name_to_cat = {x: z for z, x in cat_to_name.items()}
    return cat_to_name, name_to_cat


def mapping(top_p, top_class, map_dict):
    index = []
    probabilities = []

    for x in top_class[0]:
        idx = str(np.array(x))
        idx = map_dict[idx]
        index.append(idx)

    for p in np.array(top_p[0]):
        probabilities.append(p)

    return probabilities, index