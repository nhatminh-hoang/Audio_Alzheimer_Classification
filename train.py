import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm
from torch.cuda.amp import autocast
import tqdm
from contextlib import nullcontext 

scaler = torch.cuda.amp.GradScaler() 


def accuracy(output, label):
    with torch.no_grad():
        if output.size(1) > 1:
            # Multi-class: output shape (batch_size, num_classes)
            pred = output.argmax(dim=1)
            true = label.argmax(dim=1) if label.ndim > 1 else label
        else:
            # Binary classification: output shape (batch_size, 1)
            pred = (output > 0.5).long().squeeze(1)
            true = label.long().squeeze(1)

        correct = (pred == true).sum().item()
        total = true.size(0)
        return correct / total if total > 0 else 0.0


def get_lr(it, warmup_steps, max_step, max_lr=1e-3, min_lr=1e-5):
    if it < warmup_steps:
        lr = (max_lr - min_lr) / warmup_steps * it + min_lr
        return lr
    if it > max_step:
        lr = min_lr
        return lr
    
    decay_ratio = (it - warmup_steps) / (max_step - warmup_steps)
    coeff = 0.5 * (1 + torch.cos(torch.tensor(decay_ratio) * np.pi))
    lr = min_lr + (max_lr - min_lr) * coeff
    return lr

def train(model,dt_loader,optimizer,criterion, device,scaler=None):
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    use_autocast = scaler is not None
    autocast_context = autocast(device_type=device.type, enabled=use_autocast)

    data_iter = tqdm(dt_loader, desc='Train' , leave=False)
#-------------------------------------------------------------- lấy data
    for waveform, label in data_iter:
        waveform = waveform.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        batch_size = label.size(0) #lấy kích thước batch_size để tí tính toán loss , acc
        if label.ndim == 1 or (label.ndim == 2 and label.size(1) == 1):
            label = label.unsqueeze(-1).float()

#--------------------------------------------------------------- Zeros the optimizer’s gradients
        optimizer.zero_grad(set_to_none=True)

#----------------------------------------------------------------- Make predictions for this batch
        with autocast_context:
                output = model(waveform)
                loss = criterion(output, label)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

#-------------------------------------------------------------------Adjust learning weights
            optimizer.step()

#------------------------------------------------------------------
        acc = accuracy(output, label)  # thay đổi tùy theo bài toán
        running_loss += loss.item() * batch_size
        running_acc += acc * batch_size
        total_samples += batch_size

        data_iter.set_postfix(loss=running_loss / total_samples, acc=running_acc / total_samples)

    avg_loss = running_loss / total_samples
    avg_acc = running_acc / total_samples

    return avg_loss, avg_acc 

def valua(model,dt_loader,device,criterion):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

#------------------------------------------------------------------Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for waveform, label in tqdm(dt_loader,desc = "val",leave = True):
            waveform, label = waveform.to(device), label.to(device)
            
            if label.ndim == 1 or (label.ndim == 2 and label.size(1) == 1):
                label = label.unsqueeze(-1).float()

            batch_size = label.size(0)
            output = model(waveform)

            loss += criterion(output, label).cpu().detach().numpy()
            acc += accuracy(output, label)

            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            total_samples += batch_size

    avg_loss = running_loss / total_samples
    avg_acc = running_acc / total_samples
    return avg_loss, avg_acc 
