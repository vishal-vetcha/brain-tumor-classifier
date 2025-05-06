import torch.nn as nn
import torch.nn.functional as F

class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) #Three convolutional layers (nn.Conv2d) with kernel size 3x3 and padding 1.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #Max pooling layers (nn.MaxPool2d) with kernel size 2x2 and stride 2 to reduce spatial dimensions
        self.fc1 = nn.Linear(128 * 28 * 28, 512) #Two fully connected layers (nn.Linear) for classification.
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5) #dropout regularization

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # applying relu activation function on layer 1
        x = self.pool(F.relu(self.conv2(x))) # relu 2
        x = self.pool(F.relu(self.conv3(x))) #relu 3(ignores negitive values)
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x   #it passed layer by layer and finally returns the output after the last layer
