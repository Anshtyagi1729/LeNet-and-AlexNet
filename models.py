import torch.nn as nn
import torch
class LeNet(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,6,kernel_size=5,padding=2),nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(576,120),nn.Sigmoid(),
            nn.Linear(120,84),nn.Sigmoid(),
            nn.Linear(84,num_classes)
        )
    def forward(self,x):
        return self.net(x)
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),  # changed from 96 and 11x11
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(256*4*4, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.net(x)
class BNLeNet(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,6,kernel_size=5,padding=2),nn.BatchNorm2d(6),
            nn.Sigmoid(),nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),nn.BatchNorm2d(16),nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,stride=2)
            )
        with torch.no_grad():
            dummy=torch.zeros(1,3,32,32)
            out=self.features(dummy)
            self.flattened_dim=out.view(1,-1).size(1)
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim,120),nn.BatchNorm1d(120),
            nn.Sigmoid(),nn.Linear(120,84),nn.BatchNorm1d(84),
            nn.sigmoid(),nn.Linear(84,num_classes)
        )
    def forward(self,x):
        x=self.net(x)
        x=self.classifier(x)
        return x
