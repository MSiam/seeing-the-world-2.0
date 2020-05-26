import torch.nn as nn
import torchvision
import torch

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
        temp_backbone = torchvision.models.vgg19(pretrained=True)
        self.backbone_features = temp_backbone.features
        self.backbone_classifier = temp_backbone.classifier[:-1]
        self.cls_score = nn.Linear(4096, num_classes, bias=True)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.backbone_features(x)
        x = self.backbone_classifier(x.view(-1, 512*7*7))

        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = torch.norm(self.cls_score.weight.data,
                               p=2, dim=1).unsqueeze(1).expand_as(
            self.cls_score.weight.data)
        self.cls_score.weight.data = \
                self.cls_score.weight.data.div(temp_norm + 1e-5)

        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        return scores


