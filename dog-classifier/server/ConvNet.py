import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_pool_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5), #(100,12,150,150)
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #(100,12, 75, 75)

            nn.Conv2d(6, 16, 5), 
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=86528, out_features=500), #params: (16x5x5+1)x10
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(500, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 5),
            nn.ReLU()
        )

    def forward(self, x):
        convolved = self.conv_pool_stack(x)
        flattened = self.flatten(convolved)
        logits = self.linear_relu_stack(flattened)
        return logits