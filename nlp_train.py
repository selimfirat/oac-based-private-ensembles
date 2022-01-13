'''Train CIFAR10 with PyTorch. Took parts of the code from: https://github.com/kuangliu/pytorch-cifar''' 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

from mobilenetv2 import MobileNetV2
from utils import progress_bar
from argparse import ArgumentParser

def eval_on_data(dataloader, net):
    net.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    y_pred_beliefs = []
    print(vars(dataloader))
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["labels"] = batch["label"]
            del batch["label"]
            targets = batch["labels"]
            
            outputs = net(**batch).logits

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            y_pred.append(predicted)
            y_true.append(targets)
            y_pred_beliefs.append(outputs)

            progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)'
                        % (100.*correct/total, correct, total))

    res = torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0), torch.cat(y_pred_beliefs, dim=0)
    
    print(res[0].shape, res[1].shape, res[2].shape)
    
    return res

device = 'cuda' if torch.cuda.is_available() else 'cpu'

datasets = {
    "yelp_review_full": {
        "num_classes": 5,
        "cls": torchvision.datasets.CIFAR10,
        "num_epochs": 100,
    }
}

def train_and_save(data_name, num_devices, num_repeats):
    seed_everything(1)
    dataset = datasets[data_name]
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification
    from transformers import Trainer
    from transformers import TrainingArguments
    from transformers import AutoModelForSequenceClassification
    import numpy as np
    from datasets import load_metric

    metric = load_metric("accuracy")
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    raw_datasets = load_dataset(data_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets.set_format("torch")
    print(tokenized_datasets.keys())
    
    trainset = tokenized_datasets["train"]
    testset = tokenized_datasets["test"]

    shuffled_indices = shuffle(np.arange(len(trainset)))
    
    num_traindata = int(len(shuffled_indices)*0.9)
    
    val_inds = shuffled_indices[num_traindata:]
    valset = trainset.select(val_inds)
    valset.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])
    testset.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])
    trainset.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])
    valloader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=False)

    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)
    
    for seed_idx in range(num_repeats):
        seed_everything(seed_idx)

        train_indices = np.array_split(shuffled_indices[:num_traindata], num_devices)
        
        for device_idx, inds in enumerate(train_indices):
            seed_everything(seed_idx)

            device_trainset = trainset.select(inds)

            trainloader = torch.utils.data.DataLoader(device_trainset, batch_size=8, shuffle=False)

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=dataset["num_classes"])
    
            training_args = TrainingArguments(output_dir=f"tmp/{data_name}_{seed_idx}_{device_idx}", save_strategy="no", seed=seed_idx, report_to="none", per_device_train_batch_size=8)
            trainer = Trainer(
                model=model, args=training_args, train_dataset=device_trainset
            )
            trainer.train()
            
            y_train_true, y_train_pred, y_train_pred_beliefs = eval_on_data(trainloader, model)
            y_val_true, y_val_pred, y_val_pred_beliefs = eval_on_data(valloader, model)
            y_test_true, y_test_pred, y_test_pred_beliefs = eval_on_data(testloader, model)

            res = {
                "model": model.state_dict(),
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



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", choices=["yelp_review_full", "imdb"], default="imdb")
    parser.add_argument("--num_repeats", default=5, type=int)
    parser.add_argument("--num_devices", default=20, type=int)
    
    cfg = vars(parser.parse_args())
    
    train_and_save(cfg["data"], cfg["num_devices"], cfg["num_repeats"])