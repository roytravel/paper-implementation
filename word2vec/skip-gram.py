# -*- coding:utf-8 -*-
# Reference: https://github.com/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec-Skipgram(Softmax).py
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def get_random_batch():
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), args.batch_size, replace=False)
    
    for i in random_index:
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])
        random_labels.append(skip_grams[i][1])
    
    return random_inputs, random_labels


class Word2Vec(nn.Module):
    def __init__(self) -> None:
        super(Word2Vec, self).__init__()
        # skip-gram
        self.W = nn.Linear(voc_size, args.embedding_size, bias=False)
        self.WT = nn.Linear(args.embedding_size, voc_size, bias=False)

    def forward(self, X):
        hidden_layer = self.W(X)
        output_layer = self.WT(hidden_layer)
        return output_layer

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--batch_size", type=int, default=30, help='size of batch size')
    parser.add_argument('--embedding_size', type=int, default=5, help='size of embedding size')
    args = parser.parse_args()
    
    sentences = ["philosophy is what we have to pursue", "book is valuable thing to spread out our think",
                 "physics is a discipline for knowing the world intuitively"]
    
    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    print (word_dict)
    voc_size = len(word_list)
    
    skip_grams = []
    for i in range(2, len(word_sequence) -2):
        target_word = word_dict[word_sequence[i]]
        context_word = [word_dict[word_sequence[i-2]], word_dict[word_sequence[i-1]], word_dict[word_sequence[i+1]], word_dict[word_sequence[i+2]]]
        for word in context_word:
            skip_grams.append([target_word, word])
    
    print (skip_grams)
    
    model = Word2Vec() # model
    criterion = nn.CrossEntropyLoss() # loss
    optimizer = optim.Adam(model.parameters(), lr=0.001) # optimizer
    
    for epoch in range(50000):
        input_batch, target_batch = get_random_batch() # batch
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)
        
        optimizer.zero_grad()
        output = model(input_batch)
        
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print (f"Epoch: {epoch+1} cost = "'{:.6f}'.format(loss))
        
        loss.backward()
        optimizer.step()
    
    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext = (5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
    