# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from collections import defaultdict


# build a simple class of text dataset
class TextDataset(Dataset):
    
    def __init__(self, x, y, w2i, size):
        super(TextDataset, self).__init__()
        self.size = size
        if x and y:
            self.x = [sent2tensor(words, size, w2i) for words in x]
            self.y = [int(c)-1 for c in y]
            self.w2i = w2i
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        wi = self.x[index]
        return wi, self.y[index]
        
    # return a subset of dataset of given range
    def get_subset(self, start, end):
        subset = TextDataset(None, None, None, self.size)
        subset.x = self.x[start:end]
        subset.y = self.y[start:end]
        return subset

# build word to index mapping through vocab from word2vec model
def words2indexes(w2v_model, words_filtered = None):
    
    embed_dim = w2v_model.wv.vectors.shape[1]
    embed_matrix = [np.zeros(embed_dim)] # index 0 as unkonwn words
    w2i = defaultdict(int)
    i2w = defaultdict(str)
    index = 1
    
    for w in w2v_model.wv.vocab:
        if words_filtered and w in words_filtered:
            continue
        vector = w2v_model.wv[w]
        embed_matrix.append(vector)
        w2i[w] = index
        i2w[index] = w
        index += 1
        
    return (np.array(embed_matrix), w2i, i2w)


# given w2i convert sentence to tensor
def sent2tensor(sent, max_length, w2i):
    # unknown words have index 0, same as paddings
    encoding_sent = np.zeros(max_length)

    for i in range(max_length):
        if i < len(sent) and sent[i] in w2i: 
            encoding_sent[i] = w2i[sent[i]]

            
    return torch.LongTensor(encoding_sent)