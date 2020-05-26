import torch.nn as nn
import torchvision

class VanillaVGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = torchvision.models.vgg19(pretrained=True)
        self.backbone.classifier[6] = nn.Linear(4096, num_classes, bias=True)

    def forward(self, x):
        return self.backbone(x)

class CosSimVGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = torchvision.models.vgg19(pretrained=True)
        self.backbone.classifier[6] = nn.Linear(4096, num_classes, bias=True)

    def forward(self, x):
        return self.backbone(x)


