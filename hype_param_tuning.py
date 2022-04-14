import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Dataset

import pandas as pd
import numpy as np
import optuna

from sklearn.model_selection import train_test_split

from model import dataset as ds, TextClassificationModel, device, collate_batch

# TODO: Possible params to train:
# emsize
# scheduler

# Global params
EPOCHS = 3
val_prop = 0.2

def build_model(params):
    num_class = 2 # num of labels
    vocab = ds.vocab_list.get('combined_description')
    vocab_size = len(vocab)
    emsize = 128

    return TextClassificationModel(vocab_size, emsize, num_class).to(device)

def train(dataloader, model, criterion, optimizer, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax() == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                    '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader), total_acc/total_count), elapsed)
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count
    

def train_and_evaluate(param, model):
    val_size = int(len(ds) * val_prop)
    train_ds, val_ds = random_split(ds, [len(ds) - val_size, val_size])

    BATCH_SIZE = param['batch_size']
    
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr=param['learning_rate'])


    train(train_dataloader, model, criterion, optimizer, EPOCHS)

    return evaluate(val_dataloader, model, criterion)

# Defines the parameters to optimize
def objective(trial):
    
    params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
            'optimizer': trial.suggest_categorical('optimizer', ['Adagrad', 'SGD']),
            'batch_size': trial.suggest_categorical('BATCH_SIZE', [32, 64]),
            }

    model = build_model(params)

    accuracy = train_and_evaluate(params, model)

    return accuracy

def hype_tune_trainer():

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=30)

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print(f"{key}: {value}")

