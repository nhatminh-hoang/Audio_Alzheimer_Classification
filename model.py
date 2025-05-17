import torch
import torch.nn as nn

class ImprovedBinaryCNN(nn.Module):
    def __init__(self, input_size=(224, 224), dropout_rate=0.5):
        super(ImprovedBinaryCNN, self).__init__()

        # Tối ưu số kênh
        channels = [16, 24, 40, 64]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )

        self.res_block1 = ResidualBlock(channels[0], channels[0])

        self.transition1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.res_block2 = ResidualBlock(channels[1], channels[1])

        self.transition2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.res_block3 = ResidualBlock(channels[2], channels[2])

        self.transition3 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[3]),  # ✅ Đã thêm
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.attention = SpatialAttention()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # ✅ GAP để flatten ổn định

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *input_size)
            dummy_features = self._extract_features(dummy_input)
            self._to_linear = dummy_features.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 1)
        )

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
        #x = self.gap(x)  # ✅ GAP
        return x

    def forward(self, x):
        features = self._extract_features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


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
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        return x * attention


# Kiểm tra số tham số
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = ImprovedBinaryCNN()
print(f"Tổng số tham số: {count_parameters(model):,}")
