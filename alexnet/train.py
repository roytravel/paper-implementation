import os
import argparse
import logging
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
from glob import glob
from alexnet.alexnet import AlexNet


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--args.num_epochs', default=1000, type=int, required=False)
    parser.add_argument('--batch_size', default=128, type=int, required=False)
    parser.add_argument('--learning_rate', default=0.00005, type=float, required=False)
    args = parser.parse_args()

    seed = torch.initial_seed()
    print (f'[*] Seed : {seed}')
    NUM_CLASSES = 1000
    TRAIN_IMG_DIR = "C:/github/paper-implementation/data/ILSVRC2012_img_train/"
    #VALID_IMG_DIR = "<INPUT VALID IMAGE DIR>"
    CHECKPOINT_PATH = "../checkpoint/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (f'[*] Device : {device}')

    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
    checkpoints = glob(CHECKPOINT_PATH+'*.pth') # Is there a checkpoint file?
    if checkpoints:
        checkpoint = torch.load(checkpoints[-1])
        alexnet.load_state_dict(checkpoint['model'])
    #alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=[0,]) # for distributed training using multi-gpu

    transform = transforms.Compose(
        [transforms.CenterCrop(227),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    train_dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transform=transform)
    print ('[*] Dataset Created')

    train_dataloader = data.DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=False, # more training speed but more memory
        num_workers=8,
        drop_last=True,
        batch_size=args.batch_size
    )
    print ('[*] DataLoader Created')

    #optimizer = torch.optim.SGD(momentum=0.9, weight_decay=5e-4, params=alexnet.parameters(), lr=args.learning_rate) # SGD used in original paper
    optimizer = torch.optim.Adam(params=alexnet.parameters(), lr=args.learning_rate)
    print ('[*] Optimizer Created')

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, verbose=True, patience=4) # used if valid error doesn't improve.
    print ('[*] Learning Scheduler created')

    for epoch in range(50, args.num_epochs):
        logging.info(f" training on epoch {epoch}...")        
        for batch_idx, (images, classes) in enumerate(train_dataloader):
            images, classes = images.to(device), classes.to(device)
            output = alexnet(images)
            loss = F.cross_entropy(input=output, target=classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)
                    print ('[*] Epoch: {} \tStep: {}\tLoss: {:.4f} \tAccuracy: {}'.format(epoch+1, batch_idx, loss.item(), accuracy.item() / args.batch_size))

        lr_scheduler.step(metrics=loss)

        if epoch % 5 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_PATH, "AlexNet{}.pth".format(epoch))
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'model': alexnet.state_dict(),
                'seed': seed
            }
            torch.save(state, checkpoint_path)