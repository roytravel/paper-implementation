import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from googlenet.googlenet import GoogLeNet

def load_dataset():
    # preprocess
    transform = transforms.Compose([    
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load train, test data
    train = datasets.CIFAR10(root="data/", train=True, transform=transform, download=True)
    test = datasets.CIFAR10(root="data/", train=False, transform=transform, download=True)
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

    # load dataset
    train_loader, test_loader = load_dataset()

    # model, loss, optimizer
    losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNet(aux_logits=False, num_classes=10).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # train
    for epoch in range(args.n_epochs):
        model.train()
        for batch_idx, (image, label) in enumerate(train_loader, start=1):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = model.forward(image)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
        
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            with torch.no_grad():
                for batch_idx, (image, label) in enumerate(test_loader, start=1):
                    image, label = image.to(device), label.to(device)
                    output = model.forward(image)
                    _, preds = torch.max(output, 1)
                    total += label.size(0)
                    correct += (preds == label).sum().float()
                print(f"[*] Accuracy: {(correct / total)*100}%")

    # plot
    if args.plot:
        plt.plot(losses)
        plt.show()