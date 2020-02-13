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


# process inputs

def process_inputs():
    parser = argparse.ArgumentParser(description='Do you know your Flowers?')
    parser.add_argument('--arch', type=str, default='vgg13',
                        help='pick architecture:vgg13, alexnet,densenet')  # ,required true
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--hidden_units', type=int, default=512, help='number of hidden units')
    parser.add_argument('--learnrate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--data_dir', type=str, default='assets')
    parser.add_argument('--save_path', type=str, default='checkpoint.pth', help='path and name of checkpoint')
    parser.add_argument('--gpu', type=str, default='gpu', help='run model on gpu')


    args = parser.parse_args()

    return args


# build_model


def build_model(arch, hidden_units):
    model = sub.set_arch(arch)

    input_size = model.classifier[0].in_features

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


# train_model


def train_model(model, dataloader, epochs, learnrate, optimizer,criterion, device):
    running_loss = 0
    running_total = 0
    steps = 0
    print_every = 50

    train_losses = []
    test_losses = []



    model.to(device)

    for epoch in range(epochs):
        model.train()
        print('Epoch: {}'.format(epoch))
        for images, labels in dataloader['train']:
            model.train()
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                with torch.no_grad():
                    model.eval()
                    test_loss = 0
                    accuracy = 0
                    for images, labels in dataloader['valid']:
                        images, labels = images.to(device), labels.to(device)

                        output = model.forward(images)
                        loss = criterion(output, labels)
                        test_loss += loss.item()

                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss / len(dataloader['valid'])),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(dataloader['valid'])))
                running_loss = 0
                model.train()

    print('Final benchmark:')
    print('Epochs {}:'.format(epochs),
          "Training Loss: {:.3f}.. ".format(running_loss / print_every),
          "Validation Loss: {:.3f}.. ".format(test_loss / len(dataloader['valid'])),
          "Validation Accuracy: {:.3f}".format(accuracy / len(dataloader['valid'])))


# save_model

def save_checkpoint(save_path, model,class_to_idx, args, optimizer, criterion):
    checkpoint = {'arch': args.arch,
                  'model.class_to_idx': class_to_idx,
                  'epochs': args.epochs,
                  'learnrate': args.learnrate,
                  'optimizer': optimizer,
                  'criterion': criterion,
                  'input_size': model.classifier[0].in_features,
                  'hidden_size': args.hidden_units,
                  'output_size': model.classifier[3].out_features,
                  'state.dict': model.state_dict(),
                  'optimizer.state_dict': optimizer.state_dict()}


    torch.save(checkpoint, save_path)

    print('checkpoint succefully saved to {}'.format(save_path))

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = main.build_model(checkpoint['arch'], checkpoint['hidden_size'])

    Criterion = checkpoint['criterion']
    Optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state.dict'])

    return model, checkpoint['model.class_to_idx']


