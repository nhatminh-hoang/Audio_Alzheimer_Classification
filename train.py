import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from contextlib import nullcontext
from model import *
from torch.utils.data import TensorDataset, DataLoader, random_split
from load_data import train_loader, val_loader, test_loader
import random
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import math

import torch
import numpy as np
from torch.optim import Optimizer


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # nếu dùng nhiều GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_weights_with_seed(model, seed=42):
    """Initialize model weights deterministically."""
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, _BatchNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def accuracy(output, label):
    """Calculate accuracy between model output and labels."""
    with torch.no_grad():
        if output.size(1) > 1:
            pred = output.argmax(dim=1)
            true = label.argmax(dim=1) if label.ndim > 1 else label
        else:
            pred = (output > 0.5).long().squeeze(1)
            true = label.long().squeeze(1) if label.ndim > 1 else label

        correct = (pred == true).sum().item()
        total = true.size(0)
        return correct / total if total > 0 else 0.0

class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    """BCE loss with label smoothing for better generalization."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        # Đảm bảo target là float (bắt buộc với BCEWithLogitsLoss)
        target = target.float()
        # Áp dụng label smoothing
        smoothed_target = target * (1 - self.smoothing) + (1 - target) * self.smoothing
        return self.bce_loss(output, smoothed_target)

def train(model, train_loader, optimizer, criterion, device, scaler=None, scheduler=None):
    """Train the model for one epoch."""
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

        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
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
    """Validate the model on validation set."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for waveform, label in tqdm(val_loader, desc="Validation", leave=False):
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

def test(model, dataloader, criterion, device, is_binary=False, class_names=None, save_dir="checkpoints"):
    """Evaluate model on test set with detailed metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Ensure correct label format for binary case
            if is_binary and (labels.ndim == 1 or (labels.ndim == 2 and labels.size(1) == 1)):
                labels = labels.unsqueeze(-1).float()
            
            outputs = model(inputs)
            
            # Handle loss calculation correctly for binary vs multiclass
            if is_binary:
                loss = criterion(outputs, labels)
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs >= 0.5).long()
                all_probs.extend(probs.cpu().numpy())
            else:
                loss = criterion(outputs, labels)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.extend(probs.cpu().numpy())

            total_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Ensure arrays are properly shaped
    all_labels = np.array(all_labels).squeeze()
    all_preds = np.array(all_preds).squeeze()
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    
    # F1 scores
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    # Classification report
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names if class_names else None, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report)

    # Save to CSV
    os.makedirs(save_dir, exist_ok=True)
    df_report.to_csv(f"{save_dir}/classification_report.csv")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else ["Class 0", "Class 1"],
                yticklabels=class_names if class_names else ["Class 0", "Class 1"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close()

    # AUC-ROC curve for binary classification
    if is_binary and len(np.unique(all_labels)) == 2:
        try:
            # Ensure shapes are correct for ROC calculation
            if all_probs.ndim > 1 and all_probs.shape[1] > 1:
                probs_for_roc = all_probs[:, 1]  # For multi-class, use prob of positive class
            else:
                probs_for_roc = all_probs
                
            auc_score = roc_auc_score(all_labels, probs_for_roc)
            fpr, tpr, _ = roc_curve(all_labels, probs_for_roc)

            print(f"\nAUC-ROC Score: {auc_score:.4f}")

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{save_dir}/roc_curve.png")
            plt.close()
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(all_labels, probs_for_roc)
            avg_precision = average_precision_score(all_labels, probs_for_roc)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f"AP = {avg_precision:.4f}")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{save_dir}/precision_recall_curve.png")
            plt.close()
            
            print(f"Average Precision Score: {avg_precision:.4f}")
        except Exception as e:
            print(f"Warning: Could not calculate ROC/PR curves: {e}")

    return avg_loss, acc, f1_macro, f1_micro, f1_weighted

def main():
    # Set up
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Directory for checkpoints
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize model
    model = ImprovedBinaryCNN(hidden_size=16, drop_out=0.5).to(device)
    initialize_weights_with_seed(model, seed=42)
    
    # Loss function, optimizer, scheduler
    criterion = BCEWithLogitsLossWithLabelSmoothing(smoothing=0.1)
    w = 0.03 * math.sqrt(16/780*100)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3,weight_decay= w)
    
    # Scheduler options (choose one)
    # Option 1: CosineAnnealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 =5,T_mult =2)
    
    # Option 2: OneCycleLR (alternatively)
    # total_steps = len(train_loader) * num_epochs
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps)
    
    # Mixed precision training
    scaler = GradScaler(device='cuda') if torch.cuda.is_available() else None
    
    # Training parameters
    num_epochs = 100  # Reduced from 10000 to a more reasonable value
    patience = 100
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Lists for tracking metrics
    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []
    test_f1s = []
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device, criterion)
        
        # Step scheduler (if not OneCycleLR which steps every batch)
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
        
        # Save metrics for plotting
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Check best model criteria (based on lowest val_loss only)
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print(f"✅ Saved best model (Loss: {val_loss:.4f}, Acc: {val_acc:.4f})")
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1
            print(f"⏳ EarlyStopping counter: {patience_counter}/{patience}")

        # Optionally evaluate on test set when we have a new best model
        if is_best:
            print("\nEvaluating on test set:")
            model.load_state_dict(torch.load(f"{save_dir}/best_model.pth"))
            test_loss, test_acc, f1_macro, _, _ = test(
                model, test_loader, criterion, device, 
                is_binary=True, 
                class_names=["Class 0", "Class 1"],
                save_dir=save_dir
            )
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_f1s.append(f1_macro)

            # Save latest test metrics to all previous epochs (fill-in)
            while len(test_losses) < len(train_losses):
                test_losses.append(test_loss)
                test_accs.append(test_acc)
                test_f1s.append(f1_macro)

        # Early stopping
        if patience_counter >= patience:
            print("🛑 Early stopping triggered.")
            break

    
    # Final evaluation on test set
    print("\n=== Final Evaluation on Test Set ===")
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pth"))
    test_loss, test_acc, f1_macro, f1_micro, f1_weighted = test(
        model, test_loader, criterion, device,
        is_binary=True,
        class_names=["Class 0", "Class 1"],
        save_dir=save_dir
    )
    
    # Plot metrics
    plot_training_history(
        train_losses, val_losses, test_losses,
        train_accs, val_accs, test_accs,
        test_f1s,
        save_dir
    )

def plot_training_history(train_losses, val_losses, test_losses, 
                        train_accs, val_accs, test_accs,
                        test_f1s, save_dir="checkpoints"):
    """Plot and save training metrics history."""
    # Make sure test metrics lists are the same length as training
    if test_losses and len(test_losses) < len(train_losses):
        last_test_loss = test_losses[-1]
        last_test_acc = test_accs[-1]
        last_test_f1 = test_f1s[-1]
        
        test_losses.extend([last_test_loss] * (len(train_losses) - len(test_losses)))
        test_accs.extend([last_test_acc] * (len(train_accs) - len(test_accs)))
        test_f1s.extend([last_test_f1] * (len(train_accs) - len(test_f1s)))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(18, 6))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    if test_losses:
        plt.plot(epochs, test_losses, 'g--', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-', label='Val Acc')
    if test_accs:
        plt.plot(epochs, test_accs, 'g--', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1 Score plot
    plt.subplot(1, 3, 3)
    if test_f1s:
        plt.plot(epochs, test_f1s, 'purple', label='Test F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Test F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_metrics_plot.png")
    plt.close()
    
    print(f"📊 Training history plots saved to {save_dir}/training_metrics_plot.png")

if __name__ == "__main__":
    main()