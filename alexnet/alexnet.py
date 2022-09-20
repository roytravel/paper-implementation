import torch
import torch.nn as nn
import torch.optim
#from tqdm import tqdm

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        print (f"[*] Num of Classes: {self.num_classes}")
        self.layers = nn.Sequential(
            nn.Conv2d(kernel_size=11, in_channels=3, out_channels=96, stride=4, padding=0),
            nn.ReLU(), # inplace=True mean it will modify input. effect of this action is reducing memory usage. but it removes input.
            nn.LocalResponseNorm(alpha=1e-3, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(kernel_size=5, in_channels=96, out_channels=256, padding=2, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-3, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=384, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(kernel_size=3, in_channels=384, out_channels=384, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(kernel_size=3, in_channels=384, out_channels=256, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.avgpool = nn.AvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=self.num_classes))

        self._init_bias()


    def _init_bias(self):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.layers[4].bias, 1)
        nn.init.constant_(self.layers[10].bias, 1)
        nn.init.constant_(self.layers[12].bias, 1)
        nn.init.constant_(self.classifier[1].bias, 1)
        nn.init.constant_(self.classifier[4].bias, 1)
        nn.init.constant_(self.classifier[6].bias, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 1차원화
        x = self.classifier(x)
        return x