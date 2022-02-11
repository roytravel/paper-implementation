# -*- coding:utf-8
# Reference: https://github.com/nzw0301/keras-examples/blob/master/CBoW.ipynb

#%% 
import numpy as np
import argparse
import easydict
import gensim
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_file
from tensorflow.python.keras.utils import np_utils # well work with python.
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
tf.config.list_physical_devices("GPU")

#%% 
# parser = argparse.ArgumentParser(add_help=True)
# parser.add_argument('--dim', type=int, default=100, help='size of dimension')
# parser.add_argument('--window_size', type=int, default=2, help='size of window size')
# args = parser.parse_args()
args = easydict.EasyDict({
    "dim": 100,
    "window_size": 2,
})

#%%
path = get_file("alice.txt", origin='http://www.gutenberg.org/files/11/11-0.txt')
with open(path, mode='r', encoding='utf-8') as c:
    corpus = c.readlines()[:]
    corpus = [sentence for sentence in corpus if sentence.count(' ')>=2]
    tokenizer = Tokenizer() # 토큰화
    tokenizer.fit_on_texts(corpus) # char to list: {'love': '764', 'nine': 767}
    corpus = tokenizer.texts_to_sequences(corpus) # char to int squence: [326, 11, 741, ~], [744, 52, 15, ~], ...
    nb_samples = sum(len(s) for s in corpus)
    vocab = len(tokenizer.word_index) + 1
    dimension = 100
    window_size = 2

#%%
def generate_data(corpus, window_size, vocab):
    for words in corpus:
        for index, word in enumerate(words):
            contexts, labels = [], []
            start = index - window_size
            end = index + window_size + 1
            contexts.append([words[i] for i in range(start, end) if 0 <= i <len(words) and i != index])
            labels.append(word)
            
            x = sequence.pad_sequences(contexts, maxlen=window_size * 2)
            y = np_utils.to_categorical(labels, vocab)
            yield (x, y)
        
# %% Create model
model = Sequential()
model.add(Embedding(input_dim=vocab, output_dim=args.dim, input_length=window_size * 2))
model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(args.dim)))
model.add(Dense(vocab, activation='softmax'))

# %% Compile model
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

# %% Training model
for iter in range(5):
    loss = 0.
    for x, y in generate_data(corpus, window_size, vocab):
        loss += model.train_on_batch(x, y)
    print (iter, loss)

# %% Save result
f = open('vectors.txt', 'w', encoding='utf-8')
f.write('{} {}\n'.format(vocab-1, args.dim))

# %% Get weights of word
vectors = model.get_weights()[0]
for word, i in tokenizer.word_index.items():
    str_vec = " ".join(map(str, list(vectors[i, :])))
    f.write(f"{word} {str_vec}\n")


# %% Load vectors for using w2v model
w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)

# %% Get word similarity from model
w2v.most_similar(positive=['king'])
