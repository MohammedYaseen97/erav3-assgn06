import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.05):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-7)
        self.drop = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return self.drop(x)

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
        
        self.conv1 = ConvBlock(1, 8)  # 28x28x1 > 28x28x8
        self.conv2 = ConvBlock(8, 16)  # 28x28x8 > 28x28x16
        self.conv3 = ConvBlock(16, 32)  # 28x28x16 > 28x28x32
        
        # Transition Block 1
        self.trans1 = TransitionBlock(32, 4)  # 28x28x32 > 14x14x4
        
        self.conv4 = ConvBlock(4, 8)  # 14x14x4 > 14x14x8
        self.conv5 = ConvBlock(8, 16)  # 14x14x8 > 14x14x16
        self.conv6 = ConvBlock(16, 32)  # 14x14x16 > 14x14x32

        # Transition Block 2
        self.trans2 = TransitionBlock(32, 4)  # 14x14x32 > 7x7x4
        
        # Output Block
        self.conv7 = nn.Conv2d(4, 8, 3, padding=1)  # 7x7x4 > 7x7x8
        
        # Final Layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Transition 1
        x = self.trans1(x)
        
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        # Transition 2
        x = self.trans2(x)
        
        x = F.relu(self.conv7(x))
        
        # GAP and FC
        x = self.gap(x)
        x = x.view(-1, 8)
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

