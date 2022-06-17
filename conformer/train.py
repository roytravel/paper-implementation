import argparse
import torch
import torch.nn as nn
from model import Conformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout_rate', type=float, default='0.1', required=True)
    parser.add_argument('--learning_rate', type=float, default=0.05, required=True)
    parser.add_argument('--warmup_steps', type=int, default=4096, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(lr=args.learning_rate, betas=(0.9, 0.98), eps=10e-9)
    criterion = nn.CTCLoss().to(device)

    model = Conformer()