import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt  # 📌 Thêm matplotlib để vẽ biểu đồ

from contextlib import nullcontext
from model import *
from torch.utils.data import TensorDataset, DataLoader, random_split
from load_data import train_loader, val_loader, test_loader

scaler = GradScaler(device='cuda')


def accuracy(output, label):
    with torch.no_grad():
        if output.size(1) > 1:
            pred = output.argmax(dim=1)
            true = label.argmax(dim=1) if label.ndim > 1 else label
        else:
            pred = (output > 0.5).long().squeeze(1)
            true = label.long().squeeze(1)

        correct = (pred == true).sum().item()
        total = true.size(0)
        return correct / total if total > 0 else 0.0
    
class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        # Đảm bảo target là float (bắt buộc với BCEWithLogitsLoss)
        target = target.float()
        # Áp dụng label smoothing đúng chuẩn
        smoothed_target = target * (1 - self.smoothing) + (1 - target) * self.smoothing
        return self.bce_loss(output, smoothed_target)


def train(model, train_loader, optimizer, criterion, device, scaler=None, scheduler=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    use_autocast = scaler is not None
    autocast_context = autocast(device_type='cuda', enabled=use_autocast)

    data_iter = tqdm(train_loader, desc='Train', leave=False)

    for waveform, label in data_iter:
        waveform = waveform.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        batch_size = label.size(0)
        if label.ndim == 1 or (label.ndim == 2 and label.size(1) == 1):
            label = label.unsqueeze(-1).float()

        optimizer.zero_grad(set_to_none=True)

        with autocast_context:
            output = model(waveform)
            loss = criterion(output, label)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        acc = accuracy(output, label)
        running_loss += loss.item() * batch_size
        running_acc += acc * batch_size
        total_samples += batch_size

        data_iter.set_postfix(loss=running_loss / total_samples, acc=running_acc / total_samples)

    avg_loss = running_loss / total_samples
    avg_acc = running_acc / total_samples
    return avg_loss, avg_acc


def validate(model, val_loader, device, criterion):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for waveform, label in tqdm(val_loader, desc="Validation", leave=True):
            waveform = waveform.to(device)
            label = label.to(device)

            if label.ndim == 1 or (label.ndim == 2 and label.size(1) == 1):
                label = label.unsqueeze(-1).float()

            batch_size = label.size(0)
            output = model(waveform)

            loss = criterion(output, label).item()
            acc = accuracy(output, label)

            running_loss += loss * batch_size
            running_acc += acc * batch_size
            total_samples += batch_size

    avg_loss = running_loss / total_samples
    avg_acc = running_acc / total_samples
    return avg_loss, avg_acc


def test(model, test_loader, device, criterion):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for waveform, label in tqdm(test_loader, desc="Testing", leave=True):
            waveform = waveform.to(device)
            label = label.to(device)

            if label.ndim == 1 or (label.ndim == 2 and label.size(1) == 1):
                label = label.unsqueeze(-1).float()

            batch_size = label.size(0)
            output = model(waveform)

            loss = criterion(output, label).item()
            acc = accuracy(output, label)

            running_loss += loss * batch_size
            running_acc += acc * batch_size
            total_samples += batch_size

    avg_loss = running_loss / total_samples
    avg_acc = running_acc / total_samples
    print(f"\nTest Results - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc


# --- Cài đặt ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OptimizedBinaryCNN().to(device=device)

criterion = BCEWithLogitsLossWithLabelSmoothing()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scaler = GradScaler(device='cuda')

num_epochs = 100
patience = 10
best_val_acc = 0.0
patience_counter = 0

os.makedirs("checkpoints", exist_ok=True)

# 📌 Danh sách lưu để vẽ plot
train_losses, val_losses, test_losses = [], [], []
train_accs, val_accs, test_accs = [], [], []

# --- Huấn luyện ---
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, scaler, scheduler)
    val_loss, val_acc = validate(model, val_loader, device, criterion)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print("✅ Saved best model.")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"⏳ EarlyStopping counter: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("🛑 Early stopping.")
            break

# --- Đánh giá test ---
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
test_loss, test_acc = test(model, test_loader, device, criterion)

test_losses = [test_loss] * len(train_losses)
test_accs = [test_acc] * len(train_losses)

# --- Vẽ biểu đồ ---
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

# 📊 Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.plot(epochs, test_losses, label='Test Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

# 📊 Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, label='Train Acc')
plt.plot(epochs, val_accs, label='Val Acc')
plt.plot(epochs, test_accs, label='Test Acc', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()

plt.tight_layout()
plt.savefig("checkpoints/training_plot.png")
plt.show()
