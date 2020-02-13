
#Process inputs

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import sub
import main

#define input

args = main.process_inputs()

# load / transform data

dataloader, class_to_idx = sub.load_data(args.data_dir)

#build model

model = main.build_model(args.arch, args.hidden_units)

# set device = cpu/gpu

device = sub.set_device(args.gpu)

# train/validate model

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=args.learnrate)
main.train_model(model, dataloader, args.epochs, args.learnrate, optimizer, criterion, device)

# return final statistics (after printing during process)
# save

main.save_checkpoint(args.save_path, model, class_to_idx, args, optimizer, criterion)

