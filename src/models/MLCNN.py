# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
from .BasicModule import BasicModule
from torch.autograd import Variable
from torch.nn import functional as F


class ConvBlock(nn.Module):

    def __init__(self, conv_layer):
        super(ConvBlock, self).__init__()
        self.conv1 = conv_layer
        self.conv2 = conv_layer
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))

    def forward(self, input):
        output = self.padding_pool(input)
        output = self.pooling(output)
        residual = output
        output = self.padding_conv(output)
        output = self.relu1(output)
        output = self.conv1(output)
        output = self.padding_conv(output)
        output = self.relu2(output)
        output = self.conv2(output)
        output += residual
        return output



class MLCNN(BasicModule):
    def __init__(self,args, embed_layer, fc_layer, conv_region_layer, conv3_layer,num_blocks):
        super(MLCNN, self).__init__()
        vocab_size = args['vocab_size']
        dim = args['dim']
        n_class = args['n_class']
        max_len = args['max_len']
        dropout = args['dropout']
        self.param = args
        embedding_matrix=args['embedding_matrix']
        self.forwardEmbedLayer = args.get('forward_embed', True)
        
        self.freeze_embed = args['freeze']
        self.channel_size = args['kernel_num']
        self.dropout = nn.Dropout(dropout)
        self.nl = 3
        
        # initialize the weights of embeddings with the trained model
        self.embedding_layer = embed_layer
        # capture 3-gram context info
        self.conv_region_embedding = conv_region_layer

        # used in block conv
        self.conv_layer1 = conv3_layer
        self.conv_layer2 = conv3_layer
        layers = []
        for _ in range(num_blocks):
            layers.append(ConvBlock(self.conv_layer1))
        self.layers = nn.Sequential(*layers)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.fc1 = fc_layer
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        
        
    def forward(self, x):
        if self.forwardEmbedLayer:
            x = self.embedding_layer(x)
        batch = x.size(0)
        # print('------------\n', x.shape)
        # x = x.view(batch, 1, self.param['max_len'], self.param['dim'])
        # Region embedding
        x = self.conv_region_embedding(x)   
        # [batch_size, channel_size, text_length-2, 1]
        x = self.padding_conv(x)
        # [batch_size, channel_size, text_length, 1]
        x = self.relu3(x)
        x = self.conv_layer1(x)
        # [batch_size, channel_size, text_length-2, 1]
        x = self.padding_conv(x)
        # [batch_size, channel_size, text_length, 1]
        x = self.relu4(x)
        x = self.conv_layer2(x)
        # [batch_size, channel_size, text_length-2, 1]



        x = self.layers(x)
        x = x.view(batch, self.nl*self.channel_size)
        x = self.dropout(x)
        logit = F.log_softmax(self.fc1(x), dim=1)

        return logit
    
    def setDropout(self, dropout):
        self.dropout = nn.Dropout(dropout)
        

#     def _block(self, x, i):

#         # Pooling
#         # [batch_size, channel_size, text_length-2, 1]
#         # [batch_size, channel_size, text_length-1, 1]

#         x = self.padding_pool(x)
# #         print(x.shape)
#         px = self.pooling(x)
#         # [batch_size, channel_size, text_length-1-2 / 2 + 1, 1]
# #         print(px.size(2))

#         # Convolution
#         x = self.padding_conv(px)
#         x = F.relu(x)
#         x = self.conv3_layers[2*i](x)

#         x = self.padding_conv(x)
#         x = F.relu(x)
#         x = self.conv3_layers[2*i+1](x)
# #         print(x.shape)

#         # Short Cut
#         x = x + px

#         return x


