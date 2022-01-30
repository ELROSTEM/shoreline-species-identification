# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import division, print_function

import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from trains import StorageManager, Task

plt.ion()   # interactive mode


def tensor_to_img(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=[6,10])

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                img = tensor_to_img(inputs.cpu().data[j])
                fig.add_subplot(num_images / 2, 2, images_so_far)
                plt.title('predicted: {}'.format(class_names[preds[j]]))
                plt.imshow(img)

                if images_so_far == num_images:
                    plt.show()
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def train_model(model, criterion, optimizer, scheduler, dataloaders, logger, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # If we are checking baseline, only perform validation, otherwise perform training and validation
        if optimizer is None:
            phases = ['val']
        else:
            phases = ['train', 'val']

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients - not done in baseline mode
                if optimizer:
                    optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            logger.report_scalar(title=phase,series='epoch_loss', iteration=epoch, value=epoch_loss)
            logger.report_scalar(title=phase, series='epoch_accuracy', iteration=epoch, value=epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_finetune(params):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=params['initial_lr'], momentum=params['momentum'])

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, logger, num_epochs=params['epochs'])
    return model_ft


def train_fixed_feature_extractor(params):
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer = optim.SGD(model.fc.parameters(), lr=params['initial_lr'], momentum=params['momentum'])

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, logger, num_epochs=params['epochs'])
    return model_ft


def train_baseline(params):
    model_bl = models.resnet18(pretrained=True)
    num_ftrs = model_bl.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_bl.fc = nn.Linear(num_ftrs, 2)

    model = model_bl.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = None
    exp_lr_scheduler = None
    model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, logger, num_epochs=params['epochs'])
    return model_ft


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

task = Task.init(project_name='Pytorch Transfer Learning',task_name='fine tuning')
logger = task.get_logger()

# Define input parameters for training
args = {'initial_lr': 0.001,'momentum': 0.9,'step_size': 7, 'gamma': 0.1,'epochs': 2, 'batch_size': 4, 'workers': 4,'network': 'finetune'}

task.connect(args)

# Download data
# local_data_folder = StorageManager.get_local_copy(
#         remote_url="https://download.pytorch.org/tutorial"
#                    "/hymenoptera_data.zip",
#         name="dataset",
#     )

data_dir = './data/hymenoptera_data'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args['batch_size'],
                                             shuffle=True, num_workers=args['workers'])
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args['network'] == 'finetune':
    output_model = train_finetune(args)

elif args['network'] == 'fixed fe':
    output_model = train_fixed_feature_extractor(args)
else:
    output_model = train_baseline(args)


visualize_model(output_model)

model_scripted = torch.jit.script(output_model) # Export to TorchScript
model_scripted.save('./model/model_scripted.pt') # Save
