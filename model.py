import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.05):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.drop = nn.Dropout2d(dropout)
        
    def forward(self, x):
        return self.drop(self.bn(F.relu(self.conv(x))))

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.pool(x)
        return F.relu(self.conv1x1(x))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = ConvBlock(1, 16)  # 28x28x1 > 26x26x16
        self.conv2 = ConvBlock(16, 16) # 26x26x16 > 24x24x16
        self.conv3 = ConvBlock(16, 32) # 24x24x16 > 22x22x32
        
        # Transition Block 1
        self.trans1 = TransitionBlock(32, 8)  # 22x22x32 > 11x11x8
        
        self.conv4 = ConvBlock(8, 16)  # 11x11x8 > 9x9x16
        self.conv5 = ConvBlock(16, 16)  # 9x9x16 > 7x7x16
        self.conv6 = nn.Conv2d(16, 32, 3)  # 7x7x16 > 5x5x32
        
        # Final Layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Transition 1
        x = self.trans1(x)
        
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.relu(self.conv6(x))
        
        # GAP and FC
        x = self.gap(x)
        x = x.view(-1, 32) # flatten
        x = self.fc(x)
        
        # Use log_softmax with more stable computation
        return F.log_softmax(x, dim=1).clamp(min=-100)  # Clamp to prevent extreme negative values

if __name__ == "__main__":
    model = Net()

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

    # Print model summary
    from torchsummary import summary

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    summary(model, input_size=(1, 28, 28))

