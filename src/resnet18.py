import torchvision.models as models
import torch.nn as nn
import torch

def build_resnet18_fer(num_classes=7):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    old_conv_weight = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    with torch.no_grad():
        model.conv1.weight = nn.Parameter(old_conv_weight.sum(dim=1, keepdim=True) / 3.0)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_resnet18_fer(num_classes=7).to(device)

for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)