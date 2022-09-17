# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# import argparse
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batch_size', type=int, default=3, required=False)
#     parser.add_argument('--d_model', type=int, default=80, required=False)
#     parser.add_argument('--seq_len', type=int, default=12345, required=False)
#     parser.add_argument('--learning_rate', type=float, default=0.05, required=False)
#     parser.add_argument('--dropout_p', type=float, default=0.1, required=False)
#     args = parser.parse_args()

import torch
import torch.nn as nn
from model import Conformer

batch_size, sequence_length, dim = 3, 12345, 80
cuda = torch.cuda.is_available()  
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(batch_size, sequence_length, dim).to(device)
input_lengths = torch.LongTensor([12345, 12300, 12000])
targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                            [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                            [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
target_lengths = torch.LongTensor([9, 8, 7])

model = Conformer(num_class=10,
                  input_dim=dim,
                  encoder_dim=32,
                  num_encoder_layer=3).to(device)

criterion = nn.CTCLoss().to(device)
optimizer = torch.optim.SGD(params=model.parameters())
outputs, output_lengths = model(inputs, input_lengths)
loss = criterion(outputs.transpose(0, 1), targets, output_lengths, target_lengths)
print (loss)