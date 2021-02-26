# -*- coding: utf-8 -*-
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import re
import os

# read data preprocessed
def readData(class_file, text_file):
    f_cls = open(class_file, 'r')
    f_txt = open(text_file, 'r')
    l1 = f_txt.readline()
    text = []
    cls = []

    while l1:
        text.append(l1.split())
        l1 = f_txt.readline()
    cls = (f_cls.readline().split())
    
    return (cls, text)


def preprocessYelpPolarity(input_path, output_path=''):
    # can be replaced by nltk.corpus.stopwords
    stop_words = set(['i', 'it', 'you', 'they', 'the', 'and', 'or', 'do', 'have', 'had', 'has', 'here',
                      'he', 'she', 'me', 'we', 'there', 'these', 'those', 'to', 'any', 'some', 'of',
                      'a', 'an', 'that', 'u', 'did', 'when', 'where', 'what', 'been', 'were', 'was', 
                      'this', 'is', 'are', 'with', 'our', 'their', 'my', 'your', 'his', 'her', 'as',
                      'may', 'be', 'for', 'from', 'at', 'because', 'about', 'so', 'such', 'by', 'on'])
    lemmatizer = WordNetLemmatizer()

    # classes and texts
    cls = []
    texts = []

    # read the data
    with open(input_path, "r") as f:
        for line in f:
            cls.append(line[1])
            # only for yelp data set
            line = line[5:-2]
            line = line.lower()
            line = re.sub(r"\\\w", '', line)
#           # get rid of the punctuation except "?" and "!"
            line = re.sub(r"[^\w\?\! ']", ' ', line)
#           # get rid of the numbers in the review text
            line = re.sub(r'\d+', '', line)
            # tokenize
            words = word_tokenize(line)
            # lemmatize
            words = [lemmatizer.lemmatize(w) for w in words]
            words = [w for w in words if w not in stop_words]
            texts.append(words)
            
    # save the classes and texts to file
    if output_path:
        f_cls = open(output_path +"_classes.data", "w")
        f_txt = open(output_path +"_texts.data", "w")
        for i in range(len(cls)):
            for w in texts[i]:
                f_txt.write(w+ ' ')
            f_cls.write(cls[i] + ' ')
            f_txt.write('\n')
        f_cls.close()
        f_txt.close()
    
    return (cls, texts)

def preprocessImdb(input_path, output_path=''):
    # can be replaced by nltk.corpus.stopwords
    stop_words = set(['i', 'it', 'you', 'they', 'the', 'and', 'or', 'do', 'have', 'had', 'has', 'here',
                      'he', 'she', 'me', 'we', 'there', 'these', 'those', 'to', 'any', 'some', 'of',
                      'a', 'an', 'that', 'u', 'did', 'when', 'where', 'what', 'been', 'were', 'was', 
                      'this', 'is', 'are', 'with', 'our', 'their', 'my', 'your', 'his', 'her', 'as',
                      'may', 'be', 'for', 'from', 'at', 'because', 'about', 'so', 'such', 'by', 'on'])
    lemmatizer = WordNetLemmatizer()

    # classes and texts
    cls = []
    texts = []
    all_file_names = [(s,2) for s in os.listdir(input_path + '/pos')] + \
                     [(s,1) for s in os.listdir(input_path + '/neg')]
    files_set = set(all_file_names)
    shuffled_files = []

    while files_set: #12500 train 12500 test
        file_name, class_value = files_set.pop()
        cls.append(class_value)
        class_string = '/pos/' if class_value==2 else '/neg/'
        shuffled_files.append("%s %s"%(file_name, class_value))
        # print(shuffled_files[-1])
        with open(input_path + class_string + file_name, "r") as f:
            for line in f:
                line = line.lower()
    #           # get rid of the punctuation except "?" and "!"
                line = re.sub(r"[^\w\?\!\- ']", ' ', line)
    #           # get rid of the numbers in the review text
                line = re.sub(r'\d+', '', line)
                # tokenize
                words = word_tokenize(line)
                words = [w for w in words if w not in stop_words]
                texts.append(words)

            
    # save the classes and texts to file
    if output_path:
        f_cls = open(output_path +"_classes.data", "w")
        f_txt = open(output_path +"_texts.data", "w")
        f_names = open(output_path+"_file_names", "w")
        for i in range(len(shuffled_files)):
            for w in texts[i]:
                f_txt.write(w+ ' ')
            f_cls.write(str(cls[i]) + ' ')
            f_txt.write('\n')
            f_names.write(shuffled_files[i] + '\n')
        f_cls.close()
        f_txt.close()
    
    return (cls, texts)

