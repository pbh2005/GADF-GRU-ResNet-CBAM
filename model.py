import torch.nn as nn
import torch
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_channels = max(in_channels // reduction_ratio, 3)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=16, kernel_size=7):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels, reduction_ratio, kernel_size)

        self.short_cut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out = out + self.short_cut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, group_depth, num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels = group_depth
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.layer1 = self.make_layer(block, 4, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 8, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 16, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 32, num_blocks[3], stride=2)

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

def ResNet18(init_channels, num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], init_channels, num_classes)

def ResNet34(init_channels, num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], init_channels, num_classes)

class ResNetCBAMGRUModel(nn.Module):
    def __init__(self, gru_input_dim, hidden_layer_sizes, output_dim, resnet_type, init_channels):
        super(ResNetCBAMGRUModel, self).__init__()

        if resnet_type == 18:
            self.resnet = ResNet18(init_channels, output_dim)
        elif resnet_type == 34:
            self.resnet = ResNet34(init_channels, output_dim)

        self.space_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_layers = len(hidden_layer_sizes)
        self.gru_layers = nn.ModuleList()
        self.gru_layers.append(nn.GRU(gru_input_dim, hidden_layer_sizes[0], batch_first=True))
        for i in range(1, self.num_layers):
            self.gru_layers.append(
                nn.GRU(hidden_layer_sizes[i - 1], hidden_layer_sizes[i], batch_first=True))

        self.classifier = nn.Linear(32 + hidden_layer_sizes[-1], output_dim)

    def forward(self, x_seq, x_img):
        batch_size = x_seq.size(0)

        resnet_features = self.resnet(x_img)
        space_features = self.space_adaptive_pool(resnet_features)
        space_features = space_features.view(batch_size, -1)

        gru_out = x_seq.view(batch_size, 32, 32)
        for gru in self.gru_layers:
            gru_out, hidden = gru(gru_out)
        time_features = gru_out[:, -1, :]

        output_features = torch.cat((space_features, time_features), dim=1)
        output = self.classifier(output_features)
        return output

class GRUOnlyModel(nn.Module):
    def __init__(self, gru_input_dim, hidden_layer_sizes, num_classes, resnet_type=18, init_channels=3):
        super(GRUOnlyModel, self).__init__()
        self.num_layers = len(hidden_layer_sizes)
        self.gru_layers = nn.ModuleList()
        self.gru_layers.append(nn.GRU(gru_input_dim, hidden_layer_sizes[0], batch_first=True))
        for i in range(1, self.num_layers):
            self.gru_layers.append(nn.GRU(hidden_layer_sizes[i-1], hidden_layer_sizes[i], batch_first=True))
        self.classifier = nn.Linear(hidden_layer_sizes[-1], num_classes)

    def forward(self, x_seq, x_img=None):
        batch_size = x_seq.size(0)
        gru_out = x_seq.view(batch_size, 32, 32)
        for gru in self.gru_layers:
            gru_out, hidden = gru(gru_out)
        time_features = gru_out[:, -1, :]
        output = self.classifier(time_features)
        return output

class ResNetCBAMOnlyModel(nn.Module):
    def __init__(self, gru_input_dim=32, hidden_layer_sizes=[32,64], num_classes=10, resnet_type=18, init_channels=3):
        super(ResNetCBAMOnlyModel, self).__init__()
        if resnet_type == 18:
            self.resnet = ResNet18(init_channels, num_classes)
        elif resnet_type == 34:
            self.resnet = ResNet34(init_channels, num_classes)
        self.space_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x_seq=None, x_img=None):
        batch_size = x_img.size(0)
        resnet_features = self.resnet(x_img)
        space_features = self.space_adaptive_pool(resnet_features)
        space_features = space_features.view(batch_size, -1)
        output = self.classifier(space_features)
        return output
