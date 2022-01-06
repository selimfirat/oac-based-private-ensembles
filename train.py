'''Train CIFAR10 with PyTorch. Took parts of the code from: https://github.com/kuangliu/pytorch-cifar''' 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

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

from mobilenetv2 import MobileNetV2
from utils import progress_bar


# Training
def train(epoch):
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


def eval_on_data(dataloader):
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

datasets = {
    "cifar10": {
        "num_classes": 10,
        "cls": torchvision.datasets.CIFAR10,
        "num_epochs": 100,
        "transform": transform
    },
    "cifar100": {
        "num_classes": 100,
        "cls": torchvision.datasets.CIFAR100,
        "num_epochs": 100,
        "transform": transform,
    }
}

"""
"mnist": {
    "num_classes": 10,
    "cls": torchvision.datasets.MNIST,
    "num_epochs": 100,
    "transform": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.2,)),
    ])
},
"fashionmnist": {
    "num_classes": 100,
    "cls": torchvision.datasets.FashionMNIST,
    "num_epochs": 100,
    "transform": transform
},
"""
num_devices = 20
num_repeats = 5

if __name__ == "__main__":
    for data_name, dataset in datasets.items():
        print(data_name, dataset)
        seed_everything(1)
        trainset = dataset["cls"](root='./data', train=True, download=True, transform=dataset["transform"])

        shuffled_indices = shuffle(np.arange(len(trainset)))
        num_traindata = int(len(shuffled_indices)*0.9)
        train_indices = np.array_split(shuffled_indices[:num_traindata], 20)
        val_inds = shuffled_indices[num_traindata:]
        
        valset = Subset(trainset, val_inds)

        for seed_idx in range(num_repeats):
            for device_idx, inds in enumerate(train_indices):
                print("Device", device_idx)
                seed_everything(1)
                trainloader = torch.utils.data.DataLoader(
                    Subset(trainset, inds), batch_size=128, shuffle=True, num_workers=2)
                
                valloader = torch.utils.data.DataLoader(
                    valset, batch_size=128, shuffle=False, num_workers=2)

                testset = dataset["cls"](
                    root='./data', train=False, download=True, transform=transform)
                testloader = torch.utils.data.DataLoader(
                    testset, batch_size=128, shuffle=False, num_workers=2)

                seed_everything(seed_idx)
                # Model
                net = MobileNetV2(num_classes=dataset["num_classes"])
                net = net.to(device)
                if device == 'cuda':
                    net = torch.nn.DataParallel(net)
                    cudnn.benchmark = True

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(net.parameters(), lr=0.1,
                                    momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=dataset["num_epochs"])

                for epoch in range(dataset["num_epochs"]):
                    train(epoch)
                    scheduler.step()
                
                y_train_true, y_train_pred, y_train_pred_beliefs = eval_on_data(trainloader)
                y_val_true, y_val_pred, y_val_pred_beliefs = eval_on_data(valloader)
                y_test_true, y_test_pred, y_test_pred_beliefs = eval_on_data(testloader)

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
