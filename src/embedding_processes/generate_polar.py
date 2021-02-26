# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import re
import torchtext
import torch


def getSimilariry(v1, v2):
    # calculate cosine similarity
    return np.dot(v1, v2) \
          / np.linalg.norm(v1,2) \
          / np.linalg.norm(v2,2)

def mostSimilar(antonymDict, v, k=10):
    topK = []
    for i in antonymDict:
        simi = getSimilariry(antonymDict[i], v)
        if len(topK) <= k:
            topK.append((i,simi ))
            topK = sorted(topK, key=lambda x:x[1], reverse=True)
        else:
            topK[-1] = (i, simi)
            topK = sorted(topK, key=lambda x:x[1], reverse=True)
    return topK

embed_matrix = pickle.load(open('../../data/embeddings/glove/sst/embed.pkl', 'rb'))
w2i = pickle.load(open('../../data/embeddings/glove/sst/w2i.pkl', 'rb'))
i2w = pickle.load(open('../../data/embeddings/glove/sst/i2w.pkl', 'rb'))
antonyms =  [x[1] for x in pickle.load(open('../../data/embeddings/polar/selected1275_yelp/b1275.pkl', 'rb'))]
# print(list(w2i.keys())[0:100])
print(len(embed_matrix), len(w2i))

antonyms_added = [('want','refuse'), ('nice', 'gross'), ('sometimes','always'), ('we', 'they'), ('happy','sad'), 
('mystery', 'known'), ('carefully', 'careless'), ('and', 'but')]

antonyms = antonyms + antonyms_added
# print((embed_matrix[16]))

current_model = embed_matrix
in_set_antonyms = set()
antonym_vector = []


# generate antonym vectors
for each_word_pair in antonyms:
    w1 = each_word_pair[0]
    w2 = each_word_pair[1]
    if w1 in w2i and w2 in w2i and ((w2,w1) not in in_set_antonyms):
        in_set_antonyms.add(each_word_pair)
        diff = current_model[w2i[each_word_pair[0]]]- current_model[w2i[each_word_pair[1]]]
        diff = diff / np.linalg.norm(diff)
        antonym_vector.append(diff)
print(len(embed_matrix), len(w2i))
antonym_vector = np.array(antonym_vector)

in_set_antonyms = list(in_set_antonyms)
print(antonym_vector.shape, len(in_set_antonyms))
print(in_set_antonyms[:20])

# transform to polar space
antonym_matrix = np.matrix(antonym_vector)
antonym_matrix_inv= np.linalg.pinv((antonym_matrix))
polar_embedding = np.matmul(embed_matrix, antonym_matrix_inv) 
print(polar_embedding.shape, len(embed_matrix), len(w2i))
print(np.linalg.norm(polar_embedding[122]), np.linalg.norm(embed_matrix[122]))

# save
pickle.dump(in_set_antonyms, open('../../data/embeddings/polar/temp513_sst/a513.pkl', 'wb'))
pickle.dump(polar_embedding, open('../../data/embeddings/polar/temp513_sst/embed.pkl', 'wb'))



