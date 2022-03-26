import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data

CONFIGURES = {
    "VGG11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "VGG13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "VGG19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGGNet(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True, vgg_name: str = "VGG19") -> None:
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        self.features = self._make_layers(CONFIGURES[vgg_name], batch_norm=False)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1000),
        )

        if init_weights:
            self._init_weight()


    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu") # fan out: neurons in output layer
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        # print (x.size()) # torch.Size([2, 512, 7, 7])
        x = x.view(x.size(0), -1) # return torch.Size([2, 1000])
        x = self.classifier(x)
        return x

    def _make_layers(self, CONFIGURES:list, batch_norm: bool = False) -> nn.Sequential:
        layers: list = []
        in_channels = 3
        for value in CONFIGURES:
            if value == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                #value = cast(int, value)
                conv2d = nn.Conv2d(in_channels=in_channels, out_channels=value, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(vgg), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

                in_channels = value
        return nn.Sequential(*layers)


if __name__ == "__main__":
    # for simple test
    # x = torch.randn(1, 3, 256, 256)
    # y = vggnet(x)
    # print (y.size(), torch.argmax(y))

    # set hyper-parameter
    BATCH_SIZE= 256
    NUM_EPOCHS = 74 # (?)
    LEARNING_RATE = 0.01
    CHECKPOINT_PATH = "./checkpoint"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vggnet = VGGNet(num_classes=1000, init_weights=True, vgg_name="VGG19")

    preprocess = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48235, 0.45882, 0.40784), std=(1.0/255.0, 1.0/255.0, 1.0/255.0))
    ])

    train_dataset = datasets.STL10(root='./data', download=True, split='train', transform=preprocess)
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = datasets.STL10(root='./data', download=True, split='test', transform=preprocess)
    test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(lr=LEARNING_RATE, weight_decay=5e-3, params=vggnet.parameters(), momentum=0.9)
    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=10)
    torch.nn.parallel.DistributedDataParallel(vggnet, device_ids=[0, ])

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for idx, data in enumerate(train_dataloader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = vggnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    print (time.time() - start_time)
    print ('Finished Training')
    torch.save(vggnet.state_dict(), './checkpoint')