# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
from .BasicModule import BasicModule
from torch.autograd import Variable
from torch.nn import functional as F

class ShallowCNN(BasicModule):
    def __init__(self,args):
        super(ShallowCNN, self).__init__()
        vocab_size = args['vocab_size']
        dim = args['dim']
        n_class = args['n_class']
        max_len = args['max_len']
        dropout = args['dropout']
        self.param = args
        embedding_matrix=args['embedding_matrix']
        
        # self.k = args['k'] # for k max pooling
        self.kernel_num = args['kernel_num']
        self.dropout = nn.Dropout(dropout)
        self.forwardEmbedLayer = args.get('forward_embed', True)
        
        # initialize the weights of embeddings with the pretrained embeddings
        self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze = args['freeze'])
        
        # convolution layer with 6 filters: (3,4,5) * embedding_dimention
        self.conv11 = nn.Conv2d(1, self.kernel_num, (3, dim))
        self.conv12 = nn.Conv2d(1, self.kernel_num, (3, dim))
        self.conv13 = nn.Conv2d(1, self.kernel_num, (4, dim))
        self.conv14 = nn.Conv2d(1, self.kernel_num, (4, dim))
        self.conv15 = nn.Conv2d(1, self.kernel_num, (5, dim))
        self.conv16 = nn.Conv2d(1, self.kernel_num, (5, dim))

        self.fc1 = nn.Linear(6 * self.kernel_num, n_class)

    def _conv_and_pool(self, x, conv):
        # x: (batch, 1, sentence_length, dim)
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         x = self.kmax_pooling(x, 2, k=self.k)
#         x = x.view(x.size(0), x.size(1) * x.size(2))
        #  (batch, kernel_num * k)
        return x
    
    def setDropout(self, dropout):
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # x: (batch, sentence_length)
        if self.forwardEmbedLayer:
            x = self.embedding_layer(x)
        # x: (batch, sentence_length, embed_dim)
        # x = x.view(x.size(0),1, self.param['max_len'], self.param['dim'])
        x1 = self._conv_and_pool(x, self.conv11)  # (batch, kernel_num * k)
        x2 = self._conv_and_pool(x, self.conv12)  # (batch, kernel_num * k)
        x3 = self._conv_and_pool(x, self.conv13)  # (batch, kernel_num * k)
        x4 = self._conv_and_pool(x, self.conv14)  # (batch, kernel_num * k)
        x5 = self._conv_and_pool(x, self.conv15)  # (batch, kernel_num * k)
        x6 = self._conv_and_pool(x, self.conv16)  # (batch, kernel_num * k)

        
        x = torch.cat((x1, x2, x3, x4, x5, x6), 1)  # (batch, 6 * kernel_num * k)
        x = self.dropout(x)
        logit = F.log_softmax(self.fc1(x), dim=1)
        return logit
    
#     def kmax_pooling(self, x, dim, k):
#         index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
#         return x.gather(dim, index)