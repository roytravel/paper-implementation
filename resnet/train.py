import os
import torch
import torch.nn as nn
import numpy as np
import argparse
# import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
# from torchsummary import summary
from resnet import Model

def do_transform(train_mean, train_std, test_mean, test_std):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([train_mean[0], train_mean[1], train_mean[2]], [train_std[0], train_std[1], train_std[2]]),
        # transforms.RandomResizedCrop((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=(0.2, 2), contrast=(0.3, 2), saturation=(0.2, 2), hue=(-0.3, 0.3)),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([test_mean[0], test_mean[1], test_mean[2]], [test_std[0], test_std[1], test_std[2]]),
        transforms.Resize(224)])

    return train_transform, test_transform


def do_mean_std(train_data, test_data):
    train_mean_rgb = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_data]
    train_std_rgb = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_data]

    train_mean_r = np.mean([m[0] for m in train_mean_rgb])
    train_mean_g = np.mean([m[1] for m in train_mean_rgb])
    train_mean_b = np.mean([m[2] for m in train_mean_rgb])

    train_std_r = np.mean([s[0] for s in train_std_rgb])
    train_std_g = np.mean([s[1] for s in train_std_rgb])
    train_std_b = np.mean([s[2] for s in train_std_rgb])

    test_mean_rgb = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in test_data]
    test_std_rgb = [np.std(x.numpy(), axis=(1, 2)) for x, _ in test_data]

    test_mean_r = np.mean([m[0] for m in test_mean_rgb])
    test_mean_g = np.mean([m[1] for m in test_mean_rgb])
    test_mean_b = np.mean([m[2] for m in test_mean_rgb])

    test_std_r = np.mean([s[0] for s in test_std_rgb])
    test_std_g = np.mean([s[1] for s in test_std_rgb])
    test_std_b = np.mean([s[2] for s in test_std_rgb])

    train_mean = [train_mean_r, train_mean_g, train_mean_b]
    train_std = [train_std_r, train_std_g, train_std_b]
    test_mean = [test_mean_r, test_mean_g, test_mean_b]
    test_std = [test_std_r, test_std_g, test_std_b]

    return train_mean, train_std, test_mean, test_std


def get_dataloader(train_data, test_data):
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.01, type=float, required=False)
    parser.add_argument('--weight_decay', default=0.0001, type=float, required=False)
    parser.add_argument('--momentum', default=0.9, type=float, required=False)
    parser.add_argument('--batch_size', default=128, type=int, required=False)
    parser.add_argument('--num_epochs', default=1, type=int, required=False)
    args = parser.parse_args()

    # 1. load data
    train_data = datasets.STL10(root='../data', split="train", download=False, transform=transforms.ToTensor())
    test_data = datasets.STL10(root='../data', split="test", download=False, transform=transforms.ToTensor())

    # 2. preprocess
    train_mean, train_std, test_mean, test_std = do_mean_std(train_data, test_data)
    train_transform, test_transform = do_transform(train_mean, train_std, test_mean, test_std)
    train_data.transform = train_transform
    test_data.transform = test_transform

    # 3. prepare data loader
    train_dataloader, test_dataloader = get_dataloader(train_data, test_data)

    # 4. set hyper-parameter
    np.random.seed(1)
    seed = torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 5. load model, loss function, optimizer and scheduler
    model = Model().resnet152().to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, verbose=True)

    # 6. train epoch
    for epoch in range(args.num_epochs):
        model.train()
        correct, count = 0, 0
        train_loss = 0
        for batch_idx, (images, labels) in enumerate(train_dataloader, start=1):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(output, 1)
            count += labels.size(0)
            correct += preds.eq(labels).sum().item() # torch.sum(preds == labels)
            print (f"[*] Epoch: {epoch} \tStep: {batch_idx}/{len(train_dataloader)}\tTrain accuracy: {round((correct/count), 4)} \tTrain Loss: {round((train_loss/count), 4)}")


        model.eval()
        correct, count = 0, 0
        valid_loss = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_dataloader, start=1):
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                valid_loss += loss.item()
                _, preds = torch.max(output, 1)
                count += labels.size(0)
                correct += preds.eq(labels).sum().item() # torch.sum(preds == labels)
                print (f"[*] Step: {batch_idx}/{len(test_dataloader)}\tValid accuracy: {round((correct/count), 4)} \tValid Loss: {round((valid_loss/count), 4)}")

        # if epoch % 10 == 0:
        #     if not os.path.isdir('../checkpoint'):
        #         os.makedirs('../checkpoint', exists_ok=True)
        #     checkpoint_path = os.path.join(f"../checkpoint/ResNet{epoch}.pth")
        #     state = {
        #         'epoch': epoch,
        #         'optimizer': optimizer.state_dict(),
        #         'model': model.state_dict(),
        #         'seed': seed
        #     }
        #     torch.save(state, checkpoint_path)
            
        scheduler.step()