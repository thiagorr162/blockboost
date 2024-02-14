from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD

import fasttext
from torchtext.data import get_tokenizer


class SIFEmbedding():
    # sif_weighting_param is a parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    # the SIF paper set the default value to 1e-3
    # remove_pc is a Boolean parameter that controls whether to remove the first principal component or not
    # min_freq: if a word is too infrequent (ie frequency < min_freq), set a SIF weight of 1.0 else apply the formula
    # The SIF paper applies this formula if the word is not the top-N most frequent
    def __init__(self, sif_weighting_param=1e-3, min_freq=0):
        super().__init__()
        print("Loading FastText model")

        self.word_embedding_model = fasttext.load_model("research/data/wiki-news-300d-1M-subword.bin")
        self.dimension_size = 300

        self.tokenizer = get_tokenizer("basic_english")

        #Word to frequency counter
        self.word_to_frequencies = Counter()

        #Total number of distinct tokens
        self.total_tokens = 0

        self.sif_weighting_param = sif_weighting_param
        self.min_freq = min_freq
        
        self.token_weight_dict = {}

    #There is no pre processing needed for Average Embedding
    def preprocess(self, list_of_tuples):
        for tuple_as_str in tqdm(list_of_tuples, desc= "Getting token frequencies"):
            self.word_to_frequencies.update(self.tokenizer(tuple_as_str))

        #Count all the tokens in each tuples
        self.total_tokens = sum(self.word_to_frequencies.values())

        #Compute the weight for each token using the SIF scheme
        a = self.sif_weighting_param        
        for word, frequency in tqdm(self.word_to_frequencies.items(), desc= "Computing weight of each token"):
            if frequency >= self.min_freq:
                self.token_weight_dict[word] = a / (a + frequency / self.total_tokens)
            else:
                self.token_weight_dict[word] = 1.0


    #list_of_strings is an Iterable of tuples as strings
    # See the comments of AverageEmbedding's get_tuple_embedding for details about how this works
    def get_tuple_embedding(self, list_of_tuples):

        num_tuples = len(list_of_tuples)
        tuple_embeddings = np.zeros((num_tuples, self.dimension_size))

        for index, _tuple in tqdm(enumerate(list_of_tuples), desc= "Getting text embeddings", total=num_tuples):
            #Compute a weighted average using token_weight_dict
            words = [self.word_embedding_model.get_word_vector(token) *  self.token_weight_dict[token] for token in self.tokenizer(_tuple)]
            if words != []:
                tuple_embeddings[index, :] = np.mean(np.array(words), axis=0)

        return  tuple_embeddings
    
    
    def get_pc(self, tuple_embeddings):
        #From the code of the SIF paper at 
        # https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(tuple_embeddings)
        self.pc = svd.components_

    
    def remove_pc(self, tuple_embeddings):
        return tuple_embeddings - tuple_embeddings.dot(self.pc.transpose()) * self.pc


def get_SIFEmbedding(df, text_cols, remove_pc = True):
    concatenated_df = df[text_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)

    embedding = SIFEmbedding()
    embedding.preprocess(concatenated_df)

    tuple_embeddings = embedding.get_tuple_embedding(concatenated_df)

    if remove_pc:
        embedding.get_pc( tuple_embeddings )
        tuple_embeddings = embedding.remove_pc(tuple_embeddings)
    
    return tuple_embeddings

