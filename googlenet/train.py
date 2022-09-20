import os
import numpy as np 
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from googlenet import GoogLeNet

def load_dataset():
    # preprocess
    transform = transforms.Compose([    
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load train, test data
    train = datasets.CIFAR10(root="../data", train=True, transform=transform, download=True)
    test = datasets.CIFAR10(root="../data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    # set hyperparameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', action='store', type=int, default=100)
    parser.add_argument('--learning_rate', action='store', type=float, default='0.0002')
    parser.add_argument('--n_epochs', action='store', type=int, default=100)
    parser.add_argument('--plot', action='store', type=bool, default=True)
    args = parser.parse_args()
    
    np.random.seed(1)
    seed = torch.manual_seed(1)

    # load dataset
    train_loader, test_loader = load_dataset()

    # model, loss, optimizer
    losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNet(aux_logits=False, num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # train
    for epoch in range(args.n_epochs):
        model.train()
        train_loss = 0
        correct, count = 0, 0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(output, 1)
            count += labels.size(0)
            correct += preds.eq(labels).sum().item() # torch.sum(preds == labels)

            if batch_idx % 100 == 0:
                print (f"[*] Epoch: {epoch} \tStep: {batch_idx}/{len(train_loader)}\tTrain accuracy: {round((correct/count), 4)} \tTrain Loss: {round((train_loss/count)*100, 4)}")
        
        # valid
        model.eval()
        correct, count = 0, 0
        valid_loss = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader, start=1):
                images, labels = images.to(device), labels.to(device)
                output = model.forward(images)
                loss = criterion(output, labels)
                valid_loss += loss.item()
                _, preds = torch.max(output, 1)
                count += labels.size(0)
                correct += preds.eq(labels).sum().item() # torch.sum(preds == labels)
                if batch_idx % 100 == 0:
                    print (f"[*] Step: {batch_idx}/{len(test_loader)}\tValid accuracy: {round((correct/count), 4)} \tValid Loss: {round((valid_loss/count)*100, 4)}")

        # if epoch % 10 == 0:
        #     if not os.path.isdir('../checkpoint'):
        #         os.makedirs('../checkpoint', exists_ok=True)
        #     checkpoint_path = os.path.join(f"../checkpoint/googLeNet{epoch}.pth")
        #     state = {
        #         'epoch': epoch,
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'seed': seed,
        #     }
        #     torch.save(state, checkpoint_path)