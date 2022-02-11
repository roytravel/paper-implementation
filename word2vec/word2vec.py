# Reference: https://wikidocs.net/50739
# %% for English word2vec
import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
# %% for Korean word2vec
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
from tqdm import tqdm
# %% for pretrained word2vec
import gensim

# %% 
class EnWord2Vec:
    def __init__(self) -> None:
        # Download training data
        urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")
    
    def preprocess(self) -> None:
        # Preprocessing on training data
        targetXML = open('ted_en-20160408.xml', 'r', encoding='utf-8')
        target_text = etree.parse(targetXML)

        # Get content between <content> and </content> from xml file
        parse_text = '\n'.join(target_text.xpath('//content/text()'))

        # delete if content start with bracket and
        content_text = re.sub(r'\([^)]*\)', '', parse_text)
        sent_text = sent_tokenize(content_text)
        normalized_text = []
        for string in sent_text:
            tokens = re.sub(r"[^a-z0-9]+", " ", string.lower()) # substitution
            normalized_text.append(tokens)
        self.result = [word_tokenize(sentence) for sentence in normalized_text]
        print (f"[*] total count of sample: {len(self.result)}")

    def train(self) -> None:
        # Training word2vec model
        self.model = Word2Vec(sentences=self.result, size=100, window=5, min_count=5, workers=4, sg=0) # sg=0 is CBOW, sg=1 is Skip-gram
        
        # %% show result by checking similarity with random word
        model_result = self.model.wv.most_similar("man")
        print(model_result)
        

    def save(self) -> None:
        self.model.wv.save_word2vec_format("eng_w2v")
        

    def load(self) -> None:
        self.loaded_model = KeyedVectors.load_word2vec_format("eng_w2v")


    def test(self) -> None:
        model_result = self.loaded_model.most_similar("man")
        print(model_result)

# %% 
class KoWord2Vec:
    def __init__(self) -> None:
        urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
        self.train_data = pd.read_table('ratings.txt')
        print (self.train_data[:5])
        print (f"[BEFOR] total count of review: {len(self.train_data)}\n")
    
    
    def eda(self) -> None:
        print (f"[*] is null? : {self.train_data.isnull().values.any()}") # check if NULL --> True
        self.train_data = self.train_data.dropna(how='any') # delete row that null has
        print (f"[AFTER] total count of review: {len(self.train_data)}")
        print (f"[*] is null? : {self.train_data.isnull().values.any()}\n") # check, is null delete? --> False
    
        
    def preprocessing(self) -> None:
        # remain only char
        self.train_data['document'] = self.train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
        print (self.train_data[:5], end='\n')

        # delete stopwords
        stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
        okt = Okt()
        self.tokenized_data = []
        for sentence in tqdm(self.train_data['document']):
            tokenized_sent = okt.morphs(sentence, stem=True) # Tokenize
            stopwords_removed_sent = [word for word in sentence if not word in stopwords]
            self.tokenized_data.append(stopwords_removed_sent)
        
        print (f'[*] Max length of review : {max(len(review) for review in self.tokenized_data)}')
        print (f'[*] Average length of review : {sum(map(len, self.tokenized_data))/len(self.tokenized_data)}')
        plt.hist([len(review) for review in self.tokenized_data], bins=50)
        plt.xlabel('length of samples')
        plt.ylabel('number of samples')
        plt.show()
        
    def train(self) -> None:
        self.model = Word2Vec(sentences=self.tokenized_data, vector_size=100, window= 5, min_count=5, workers=4, sg=0)
        print ("[*] size of completed embedding matrix \n {self.model.wv.vectors.shape}")
    
    def get_similar_word(self, word) -> None:
        print (self.model.wv.most_similar(word))
        

if __name__ == "__main__":
    
    # %% 
    # 1. create English word2vec model 
    # EW2V = EnWord2Vec()
    # EW2V.preprocess()
    # EW2V.train()
    # EW2V.save()
    # EW2V.load()
    # EW2V.test()
    
    # %% 
    # 2. create Korean word2vec model
    KW2V = KoWord2Vec()
    KW2V.eda()
    KW2V.preprocessing()
    KW2V.train()
    # %% 
    KW2V.get_similar_word("느와르")
    
    # %% 
    # 3. use pre-trained word2vec model
    urllib.request.urlretrieve("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", filename="GoogleNews-vectors-negative300.bin.gz")
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    print(word2vec_model.vectors.shape)
    print(word2vec_model.similarity('love', 'hate'))
    print(word2vec_model.similarity('like', 'love'))
    # ptW2V = ptWord2Vec()
    # ptW2V.calc_similarity('this', 'is')
    # ptW2V.calc_similarity('love', 'hate')
    # ptW2V.calc_similarity('love', 'like')
# %%
