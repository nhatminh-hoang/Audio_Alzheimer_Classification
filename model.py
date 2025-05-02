import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedBinaryCNN(nn.Module):
    def __init__(self):
        super(OptimizedBinaryCNN, self).__init__()
        
        # Giảm số filter và thêm BatchNorm để cải thiện hiệu suất
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)  # Giảm từ 16 xuống 8
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)  # Giảm từ 32 xuống 16
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Giảm từ 64 xuống 32
        self.bn3 = nn.BatchNorm2d(32)
        
        # Thêm Dropout để tránh overfitting
        self.dropout = nn.Dropout(0.5)
        
        # Giảm kích thước lớp fully connected
        self.fc1 = nn.Linear(32 * 28 * 28, 64)  # Giảm từ 128 xuống 64
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x







model = OptimizedBinaryCNN()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")
