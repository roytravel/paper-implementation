import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Conv2d(kernel_size=11, in_channels=3, out_channels=96, stride=4, padding=0),
            nn.ReLU(inplace=True), # inplace=True mean it will modify input. effect of this action is reducing memory usage. but it removes input.
            nn.LocalResponseNorm(alpha=1e-3, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(kernel_size=5, in_channels=96, out_channels=256, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(alpha=1e-3, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=384, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size=3, in_channels=384, out_channels=384, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size=3, in_channels=384, out_channels=256, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.avgpool = nn.AvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
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


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(add_help=True)
    # parser.add_argument()
    seed = torch.initial_seed()
    print (f'[*] Seed : {seed}')

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (device)
    alexnet = AlexNet(num_classes=10)
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=[0,])

    # hyper parameters
    NUM_EPOCHS = 90
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    TRAIN_IMG_DIR = "../data"
    CHECKPOINT_PATH = "./checkpoint"

    # dataset
    transform = transforms.Compose(
        [transforms.CenterCrop(227),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transform=transform)
    print (dataset.data.shape)
    print ('[*] Dataset Created')

    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=8,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE
    )
    print ('[*] DataLoader Created')

    optimizer = torch.optim.SGD(momentum=0.9, weight_decay=5e-4, params=alexnet.parameters(), lr=LEARNING_RATE)
    print ('[*] Optimizer Created')

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    print ('[*] Learning Scheduler created')

    steps = 1
    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step(metrics=1)
        for images, classes in dataloader:
            print (images.shape)
            images, classes = images.to(device), classes.to(device)

            output = alexnet(images)
            loss = F.cross_entropy(output, classes)
            loss.backward()
            optimizer.step()

            if steps % 10 == 0:
                print (steps)
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)
                    print ('Epoch: {} \tStep: {}\tLoss: {:.4f} \tAccuracy: {}'.format(epoch+1, steps, loss.item(), accuracy.item()))

            steps = steps + 1

        checkpoint_path = os.path.join(CHECKPOINT_PATH)
        state = {
            'epoch': epoch,
            'steps': steps,
            'optimizer': optimizer.state_dict(),
            'model': alexnet.state_dict(),
            'seed': seed,
        }

        torch.save(state, checkpoint_path)







