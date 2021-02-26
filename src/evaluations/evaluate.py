# -*- coding: utf-8 -*-

import numpy as np
import pickle
import json
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
# import torch.onnx
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import os
import threading
import time
import matplotlib.pyplot as plt

# my modules
sys.path.append(os.getcwd() + '/../')

from models import ShallowCNN
from models import DPCNN
from dataset import dataset
from preprocess import preprocess
from util import readConfig
from train import *
from plot import plots
from scipy import stats

from captum.attr._core import (
    input_x_gradient,
    guided_grad_cam,
    gradient_shap,
    kernel_shap
)

from captum.attr._core.layer import grad_cam

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Saliency
)


# load the config get the model and the data
def init(config_path):
    print('Read config file %s.'%config_path)
    args = readConfig.readConfig(config_path)
    print(args)
    # load embeddings
    embed_matrix = pickle.load(open(args['embed_path']+'/embed.pkl', 'rb'))
    print("Load embedding of size %s"%embed_matrix.shape[1])
    w2i = pickle.load(open(args['w2i']+'/w2i.pkl', 'rb'))
    i2w = pickle.load(open(args['i2w']+'/i2w.pkl', 'rb'))
    # load data
    cls, texts =preprocess.readData(args['input_path']+"/test_classes.data", args['input_path']+"/test_texts.data") 
    test_set = dataset.TextDataset(texts, cls, w2i, int(args['max_sen_len']))
    model = None
    
    model_args = {
        'vocab_size': embed_matrix.shape[0], # add unkown word
        'max_len': int(args['max_sen_len']),
        'n_class': int(args['num_class']),
        'dim': embed_matrix.shape[1],
        'dropout': float(args['dropout']),
        'freeze': True,
        'kernel_num': int(args['channel_size']),
        'embedding_matrix': embed_matrix,
        'forward_embed': True
        
    }
    if args['model']=='shallowCNN':
            print("Start training shallowCNN")
            model = ShallowCNN.ShallowCNN(model_args)
    else:
        model = DPCNN.DPCNN(model_args)
    model.load(args['load_model_from'])
    print(model.eval)
    model.setDropout(0)
    
    return (args, w2i, i2w, cls, texts, test_set, model_args, model)

args, w2i, i2w, cls, texts, test_set, model_args, model = init("../train/abspath_train_config")

print("success!")
