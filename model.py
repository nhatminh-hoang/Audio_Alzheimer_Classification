import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedBinaryCNN(nn.Module):
    def __init__(self, input_size=(224, 224)):
        super(OptimizedBinaryCNN, self).__init__()
        
        # Feature extraction layers
        self.fe = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
            
        
        # Tính toán kích thước đầu vào cho lớp tuyến tính
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *input_size)
            dummy_output = self.fe(dummy_input)
            self._to_linear = dummy_output.view(1, -1).shape[1]
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1))
    
    def forward(self, x):
        features = self.fe(x)
        features = features.view(features.size(0), -1)  # Flatten
        output = self.classifier(features)
        return output
    
    def forward(self, x):
            features = self.fe(x)
            features = features.view(features.size(0), -1)  # Flatten
            output = self.classifier(features)
            return output
#-------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedConv2d(nn.Module):
    """Grouped Convolution Layer (Split-Transform-Merge)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=4):
        super().__init__()
        self.groups = groups
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride,
            padding=kernel_size//2,
            groups=groups,
            bias=False
        )

    def forward(self, x):
        return self.conv(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedBinaryCNN(nn.Module):
    def __init__(self, input_size=(224, 224), dropout_rate=0.5):
        super(ImprovedBinaryCNN, self).__init__()
        
        self.input_bn = nn.BatchNorm2d(3)
        #self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 1. Sử dụng các block residual để cải thiện learning
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        
        # Residual Block 1
        self.res_block1 = ResidualBlock(16, 16)
        
        # Transition + Downsample
        self.transition1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Residual Block 2
        self.res_block2 = ResidualBlock(32, 32)
        
        # Transition + Downsample
        self.transition2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Residual Block 3
        self.res_block3 = ResidualBlock(64, 64)
        
        # Transition + Downsample
        self.transition3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 2. Spatial Attention
        self.attention = SpatialAttention()
        
        # Tính toán kích thước đầu vào cho lớp tuyến tính
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *input_size)
            dummy_features = self._extract_features(dummy_input)
            #dummy_features = self.gap(dummy_features) 
            self._to_linear = dummy_features.view(1, -1).shape[1]
        
        # 3. Cải thiện classifier với batch normalization và dropout thích ứng
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),  # Giảm dropout ở lớp cuối
            nn.Linear(128, 1)
        )
        
        # 4. Khởi tạo trọng số tốt hơn
        self._initialize_weights()
    
    def _extract_features(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.transition1(x)
        x = self.res_block2(x)
        x = self.transition2(x)
        x = self.res_block3(x)
        x = self.transition3(x)
        x = self.attention(x)
        return x
    
    def forward(self, x):
        features = self._extract_features(x)
        #features = self.gap(features)
        features = features.view(features.size(0), -1)  # Flatten
        output = self.classifier(features)
        return output
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection (identity) nếu số kênh đầu vào và đầu ra giống nhau
        self.skip = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.activation(self.conv_block(x) + self.skip(x))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Tạo hai feature map theo chiều không gian
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Nối hai feature map
        attention = torch.cat([avg_out, max_out], dim=1)
        
        # Tính attention map
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        # Nhân attention map với đầu vào
        return x * attention


# Thêm phương thức cho dự đoán với threshold

# Kiểm tra model


