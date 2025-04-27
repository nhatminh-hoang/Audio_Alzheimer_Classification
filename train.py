import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np
from collections import Counter

from load_data import train_loader, test_loader
from model import *

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== TRAIN FUNCTION ====================
def train_epoch(model, train_loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc="Training")

    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)  # Ensure shape [batch_size, 1]

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(inputs)  # No sigmoid here
            loss = criterion(outputs, labels)

        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(train_loader)

# ==================== EVALUATION FUNCTION ====================
def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)  # threshold 0.5
            all_preds.append(preds)
            all_labels.append(labels.long().squeeze(1))  # ensure same shape

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    print("Dự đoán theo lớp:", dict(zip(*np.unique(all_preds, return_counts=True))))
    print("Ground truth theo lớp:", dict(zip(*np.unique(all_labels, return_counts=True))))

    return running_loss / len(test_loader), accuracy, report

# ==================== MAIN TRAINING FUNCTION ====================
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=100, patience=5):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'accuracy': []}
    scaler = GradScaler()

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, accuracy, report = evaluate(model, test_loader, criterion)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(accuracy)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
        print(f"Classification Report:\n{report}")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            print(f"Validation loss decreased from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'accuracy': accuracy,
            }, best_model_path)
        else:
            epochs_without_improvement += 1
            print(f"Validation loss did not improve. {patience-epochs_without_improvement} epochs until early stopping.")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break

    print(f"Training completed. Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, history

# ==================== MAIN ====================
def main():
    try:
        # Thống kê phân bố nhãn
        labels = []
        for _, targets in test_loader:
            labels.extend(targets.tolist())

        counter = Counter(labels)
        print("Phân bố nhãn trong tập test:")
        for label, count in counter.items():
            print(f"  Nhãn {label}: {count} mẫu")

        # Initialize model
        model = DeiTBinary().to(device)

        # Setup training
        criterion = nn.BCEWithLogitsLoss()  # Khác BCELoss nhé!
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

        # Train model
        model, history = train_and_evaluate(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs=9999,
            patience=50
        )

        # Final evaluation
        print("\nFinal Evaluation:")
        _, accuracy, report = evaluate(model, test_loader, criterion)
        print(f"Final Accuracy: {accuracy*100:.2f}%")
        print(f"Classification Report:\n{report}")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
