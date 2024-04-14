import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5,self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(), # different with original LeNet5(Tanh)
            nn.MaxPool2d(kernel_size=2), # different with original LeNet5(Avgpool)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        )

        self.Classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_classes)
        )

    def forward(self, img):
        features = self.feature_extractor(img)
        features = torch.flatten(features, 1)
        outputs = self.Classifier(features)

        return outputs


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self, n_classes):
        super(CustomMLP,self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(6), # Add batch normalization to improve gradient flow
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.BatchNorm2d(120)
        )

        self.Classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(0.3), # Add dropout layer to apply regularization effect 
            nn.Linear(in_features=84, out_features=n_classes)
        )

    def forward(self, img):
        features = self.feature_extractor(img)
        features = torch.flatten(features, 1)
        outputs = self.Classifier(features)

        return outputs