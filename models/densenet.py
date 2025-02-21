import torch.nn as nn
from torchvision import models


class DenseNetMultiLabel(nn.Module):
    """
    CNN model based on DenseNet121 for multi-label classification tasks.

    Args:
        num_classes (int): Number of output classes.
    """

    def __init__(self, num_classes: int, weights: str | None = "DEFAULT"):
        super(DenseNetMultiLabel, self).__init__()
        self.densenet = models.densenet121(weights="DEFAULT")
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(nn.Linear(num_features, num_classes))

    def forward(self, x):
        x = self.densenet(x)
        return x
