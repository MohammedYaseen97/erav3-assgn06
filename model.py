import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) # 28x28x1 > 28x28x8
        self.bn1 = nn.BatchNorm2d(8)
        self.drop1 = nn.Dropout2d(0.1)
        
        # CONV 2
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # 28x28x8 > 28x28x16
        self.bn2 = nn.BatchNorm2d(16)
        self.drop2 = nn.Dropout2d(0.1)
        
        # Transition Block 1
        self.pool1 = nn.MaxPool2d(2, 2) # 28x28x16 > 14x14x16
        self.conv1x1_1 = nn.Conv2d(16, 4, 1) # 14x14x16 > 14x14x4
        
        # CONV 3
        self.conv3 = nn.Conv2d(4, 8, 3, padding=1) # 14x14x4 > 14x14x8
        self.bn3 = nn.BatchNorm2d(8)
        self.drop3 = nn.Dropout2d(0.1)
        
        # CONV 4
        self.conv4 = nn.Conv2d(8, 16, 3, padding=1) # 14x14x8 > 14x14x16
        self.bn4 = nn.BatchNorm2d(16)
        self.drop4 = nn.Dropout2d(0.1)
        
        # Transition Block 2
        self.pool2 = nn.MaxPool2d(2, 2) # 14x14x16 > 7x7x16
        self.conv1x1_2 = nn.Conv2d(16, 4, 1) # 7x7x16 > 7x7x4
        
        # CONV 5
        self.conv5 = nn.Conv2d(4, 8, 3, padding=1) # 7x7x4 > 7x7x8
        self.bn5 = nn.BatchNorm2d(8)
        self.drop5 = nn.Dropout2d(0.1)
        
        # Output Block
        self.conv6 = nn.Conv2d(8, 16, 3, padding=1) # 7x7x8 > 7x7x16
        
        # Final Layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # Input Block
        x = self.drop1(self.bn1(F.relu(self.conv1(x))))
        
        # CONV 2
        x = self.drop2(self.bn2(F.relu(self.conv2(x))))
        
        # Transition 1
        x = self.pool1(x)
        x = F.relu(self.conv1x1_1(x))
        
        # CONV 3
        x = self.drop3(self.bn3(F.relu(self.conv3(x))))
        
        # CONV 4
        x = self.drop4(self.bn4(F.relu(self.conv4(x))))
        
        # Transition 2
        x = self.pool2(x)
        x = F.relu(self.conv1x1_2(x))
        
        # CONV 5
        x = self.drop5(self.bn5(F.relu(self.conv5(x))))
        
        # Output Block
        x = F.relu(self.conv6(x))
        
        # GAP and FC
        x = self.gap(x)
        x = x.view(-1, 16)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1) 