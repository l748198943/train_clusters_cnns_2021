# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
from .BasicModule import BasicModule
from torch.autograd import Variable
from torch.nn import functional as F

class DPCNN_(BasicModule):
    def __init__(self,args):
        super(DPCNN, self).__init__()
        vocab_size = args['vocab_size']
        dim = args['dim']
        n_class = args['n_class']
        max_len = args['max_len']
        dropout = args['dropout']
        self.param = args
        embedding_matrix=args['embedding_matrix']
        
        self.freeze_embed = args['freeze']
        self.channel_size = args['kernel_num']
        self.dropout = nn.Dropout(dropout)
        self.nl = 3
        
        # initialize the weights of embeddings with the trained word2vec model
        # we set the freeze to false so that the weights will also be trained here
        # self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze = self.freeze_embed)
        # capture 3-gram context info
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, dim), stride=1)
        # used in block conv
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.fc1 = nn.Linear(self.nl*self.channel_size, n_class)
        
        
    def forward(self, x):
        # x = self.embedding_layer(x)
        batch = x.size(0)
        x = x.view(batch, 1, self.param['max_len'], self.param['dim'])
        # Region embedding
        x = self.conv_region_embedding(x)   
        # [batch_size, channel_size, text_length-2, 1]
        x = self.padding_conv(x)
        # [batch_size, channel_size, text_length, 1]
        x = F.relu(x)
        x = self.conv3(x)
        # [batch_size, channel_size, text_length-2, 1]
        x = self.padding_conv(x)
        # [batch_size, channel_size, text_length, 1]
        x = F.relu(x)
        x = self.conv3(x)
        # [batch_size, channel_size, text_length-2, 1]

        while x.size()[-2] > 3:
            x = self._block(x)

        x = x.view(batch, self.nl*self.channel_size)
        x = self.dropout(x)
        logit = F.log_softmax(self.fc1(x), dim=1)

        return logit
    
    def setDropout(self, dropout):
        self.dropout = nn.Dropout(dropout)
        

    def _block(self, x):
        # Pooling
        # [batch_size, channel_size, text_length-2, 1]
        # [batch_size, channel_size, text_length-1, 1]

        x = self.padding_pool(x)
#         print(x.shape)
        px = self.pooling(x)
        # [batch_size, channel_size, text_length-1-2 / 2 + 1, 1]
#         print(px.size(2))

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)
#         print(x.shape)

        # Short Cut
        x = x + px

        return x

