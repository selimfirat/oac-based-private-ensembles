'''Train CIFAR10 with PyTorch. Took parts of the code from: https://github.com/kuangliu/pytorch-cifar''' 
import os
from turtle import forward
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from utils import seed_everything
seed_everything(1)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.utils import shuffle
import argparse

from utils import progress_bar
from argparse import ArgumentParser


class Model(nn.Module):
    
    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.mobilenet = torchvision.models.mobilenet_v3_large(pretrained=True)
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, num_classes)
    
    def forward(self, x):
        
        res = self.mobilenet(x)
        res = nn.functional.softmax(res, dim=1)

        return res
        
# Training
def train(epoch, trainloader, net, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def eval_on_data(dataloader, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    y_pred_beliefs = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            y_pred.append(predicted)
            y_true.append(targets)
            y_pred_beliefs.append(outputs)

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    res = torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0), torch.cat(y_pred_beliefs, dim=0)
    
    print(res[0].shape, res[1].shape, res[2].shape)
    
    return res

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def gray2rgb(image):
    return image.repeat(3, 1, 1)

rgb_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

gray_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(gray2rgb),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

datasets = {
    "cifar10": {
        "num_classes": 10,
        "cls": torchvision.datasets.CIFAR10,
        "transform": rgb_transform,
    },
    "cifar100": {
        "num_classes": 100,
        "cls": torchvision.datasets.CIFAR100,
        "transform": rgb_transform,
    },
    "mnist": {
        "num_classes": 10,
        "cls": torchvision.datasets.MNIST,
        "transform": gray_transform,
    },
    "fashionmnist": {
        "num_classes": 10,
        "cls": torchvision.datasets.FashionMNIST,
        "transform": gray_transform,
    },
}

def train_and_save(data_name, num_devices, num_repeats, num_epochs):
    seed_everything(1)
    dataset = datasets[data_name]
    
    trainset = dataset["cls"](root='./data', train=True, download=True, transform=dataset["transform"])
    
    shuffled_indices = shuffle(np.arange(len(trainset)))
    
    num_traindata = int(len(shuffled_indices)*0.9)
    
    val_inds = shuffled_indices[num_traindata:]
    valset = Subset(trainset, val_inds)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)


    testset = dataset["cls"](root='./data', train=False, download=True, transform=dataset["transform"])
        
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)
    
    for seed_idx in range(num_repeats):
        seed_everything(seed_idx)

        train_indices = np.array_split(shuffled_indices[:num_traindata], num_devices)
        
        for device_idx, inds in enumerate(train_indices):
            seed_everything(seed_idx)

            print("Device", device_idx)
            trainloader = torch.utils.data.DataLoader(
                Subset(trainset, inds), batch_size=128, shuffle=True, num_workers=2)
            
            # Model
            #net = MobileNetV2(num_classes=dataset["num_classes"], in_channels = dataset["num_channels"])
            net = Model(num_classes = dataset["num_classes"])
            net = net.to(device)
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

            for epoch in range(num_epochs):
                train(epoch, trainloader, net, criterion, optimizer)
                scheduler.step()
            
            y_train_true, y_train_pred, y_train_pred_beliefs = eval_on_data(trainloader, net, criterion)
            y_val_true, y_val_pred, y_val_pred_beliefs = eval_on_data(valloader, net, criterion)
            y_test_true, y_test_pred, y_test_pred_beliefs = eval_on_data(testloader, net, criterion)

            res = {
                "model": net.state_dict(),
                "inds": inds,
                "device_idx": device_idx,
                "y_train_true": y_train_true,
                "y_train_pred": y_train_pred,
                "y_train_pred_beliefs": y_train_pred_beliefs,
                "y_val_true": y_val_true,
                "y_val_pred": y_val_pred,
                "y_val_pred_beliefs": y_val_pred_beliefs,
                "y_test_true": y_test_true,
                "y_test_pred": y_test_pred,
                "y_test_pred_beliefs": y_test_pred_beliefs
            }

            targetdir = f"results/{data_name}_{num_devices}devices_seed{seed_idx}"
            if not os.path.isdir(targetdir):
                os.makedirs(targetdir)
            
            torch.save(res, f'{targetdir}/{device_idx}.pth')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", choices=["cifar10", "fashionmnist", "mnist", "cifar100"])
    parser.add_argument("--num_repeats", default=5, type=int)
    parser.add_argument("--num_devices", default=20, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    
    cfg = vars(parser.parse_args())
    
    train_and_save(cfg["data"], cfg["num_devices"], cfg["num_repeats"], cfg["num_epochs"])