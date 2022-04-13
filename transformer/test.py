import argparse
import torch
import torch.nn as nn
from transformer import Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #  model parameter
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--device', type=int, default=DEVICE)
    # parser.add_argument('--d_k', type=int, default=64)
    # parser.add_argument('--d_v', type=int, default=64)

    # optimizer parameter
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--learning_rate', type=float) # default
    parser.add_argument('--num_epochs', type=int, default=200) # ?
    parser.add_argument('--warmup_steps', type=int, default=4000)

    args = parser.parse_args()