#imports

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import sub
import main
import argparse

from collections import OrderedDict
from PIL import Image
import json


#define user inputs -  args

def process_inputs_predict():
    parser = argparse.ArgumentParser(description="That`s your Flower! Isn`t it ?")
    parser.add_argument('--checkpoint', type=str, default = 'checkpoint.pth',help='path to a checkpoint')
    parser.add_argument('--path_to_image', type=str, help='path and name of Image')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='category mapper')
    parser.add_argument('--gpu', type=str, default='gpu', help='run model on gpu')
    parser.add_argument('--top_k', type=int, default='3')

    args = parser.parse_args()

    return args

args_prd = process_inputs_predict()

device = sub.set_device(args_prd.gpu)

#build model from checkpoint

model, class_to_idx = main.load_checkpoint(args_prd.checkpoint)


#predict

def predict(device, image_path, model, class_to_idx, topk):

    model.class_to_idx = class_to_idx

    image = sub.process_image(image_path)
    image.unsqueeze_(0)
    image = torch.FloatTensor(image)
    image = image.to(device)
    model.to(device)
    with torch.no_grad():
        model.eval()
        output = model.forward(image)
        ps = torch.exp(output)

        top_p, top_class = ps.topk(topk, dim=1)

        return top_p, top_class

top_p, top_class = predict(device, args_prd.path_to_image, model, class_to_idx, args_prd.top_k)
top_p, top_class = torch.Tensor.cpu(top_p), torch.Tensor.cpu(top_class)
top_p, top_class = np.array(top_p), np.array(top_class)


#map class_to_index (make dict index_to_class)

def mapping(top_p, top_class, map_dict):
    index = []
    probabilities = []

    with open(map_dict, 'r') as f:
        map_dict = json.load(f)


    for x in top_class[0]:
        idx = str(np.array(x))
        idx = map_dict[idx]
        index.append(idx)

    for p in np.array(top_p[0]):
        probabilities.append(p)

    return probabilities, index

top_p, top_class = mapping(top_p, top_class, args_prd.category_names)

#output results (topk3)



def print_results(top_p, top_class, args):
    print('')
    print('The {} predicted topk classes and corresponding probabilities for {} are:'.format(args.top_k, args.path_to_image))
    print('')
    print('')
    for i in range(len(top_p)):
        Pos = i+1
        print('Position {}: {}'.format(Pos,top_class[i]),
              'Probability: {} '.format(top_p[i]))
        print('')

print_results(top_p, top_class, args_prd)