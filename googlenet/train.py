import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import googlenet

def load_dataset():
    train = datasets.CIFAR10(root="data/", train=True, transform=transform, download=True)
    test = datasets.CIFAR10(root="data/", train=False, transform=transform, download=True)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader

def inference():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for image, label in test_loader:
            x = image.to(device)
            y = label.to(device)
            output = model.forward(x)
            _, preds = torch.max(output,1)
            total += label.size(0)
            correct += (preds == y).sum().float()
        print(f"[*] Accuracy: {(correct / total)*100}%")


if __name__ == "__main__":
    # set hyperparameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', action='store', type=int, default=100)
    parser.add_argument('--learning_rate', action='store', type=float, default='0.0002')
    parser.add_argument('--n_epochs', action='store', type=int, default=100)
    parser.add_argument('--inference', action='store', type=bool, default=True)
    parser.add_argument('--plot', action='store', type=bool, default=True)
    args = parser.parse_args()

    # preprocess
    transform = transforms.Compose([    
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load dataset
    train_loader, test_loader = load_dataset()
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


    # model, loss, optimizer
    losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = googlenet.GoogLeNet(aux_logits=False, num_classes=10).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # train
    for i in range(args.n_epochs):
        for j, [image, label] in enumerate(train_loader):
            x = image.to(device)
            y = label.to(device)
            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
        
        if i % 10 == 0:
            print(loss)
            losses.append(loss.cpu().detach().numpy())

    # plot
    if args.plot:
        plt.plot(losses)
        plt.show()

    # test
    if args.inference:
        inference()