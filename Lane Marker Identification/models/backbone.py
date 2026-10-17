import torch.nn as nn
import torchvision.models as tv

class ResNet18Feature(nn.Module):
    """Return C5 feature map (stride 32) from ResNet-18."""
    def __init__(self, pretrained=True):
        super().__init__()
        m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.out_channels = 512

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)  # /4
        x = self.layer2(x)  # /8
        x = self.layer3(x)  # /16
        x = self.layer4(x)  # /32 -> C5
        return x
