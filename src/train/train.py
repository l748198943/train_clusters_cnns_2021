# -*- coding: utf-8 -*-
import numpy as np
import pickle
import json
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import os
sys.path.append(os.getcwd() + '/..')

from models import ShallowCNN
from models import DPCNN
from dataset import dataset
from preprocess import preprocess
from util import readConfig
from plot import plots
from scipy import stats

from captum.attr._core import (
    input_x_gradient,
    guided_grad_cam
)

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel
)

# test the model and print confusion matrix and accuracy
def test(model, test_dataset, batch_size = 64, start=0, end=0):
    
    test_dataloader = DataLoader(test_dataset.get_subset(start, end), batch_size=batch_size)
    total = 0
    correct = 0
    matrx = np.zeros((2,2))
        
    for inputs, labels in test_dataloader:
        inp = inputs.unsqueeze(1)
        inp = Variable(inp)
        out = model(inp)
        labels = labels.tolist()
        predicted = torch.argmax(out, dim=1).tolist()
        total += batch_size
        for i in range(len(labels)):
            correct += 1 if labels[i] == predicted[i] else 0
            matrx[predicted[i]][labels[i]] += 1

    print(matrx)
    return round(correct/total * 100, 3)


# training implemented with train only a subset of dataset given start and end
def train(model, train_dataset, epochs, batch_size, cv=0.1, learning_rate=0.001, start = 0, end = 0):
    
    subset = train_dataset.get_subset(start, end)
    n_training_samples = len(subset) * (1-cv)
    n_val_samples = len(subset) * cv
    train_loader =torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                              sampler=SubsetRandomSampler(
                                                  np.arange(n_training_samples, dtype=np.int64)
                                              ),
                                              num_workers=3)
    val_loader =torch.utils.data.DataLoader(subset, batch_size=100,
                                              sampler=SubsetRandomSampler(
                                                  np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64)
                                              ), num_workers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss() # cross_entropy_loss = logsoftmax + NLLLoss
    model.float()

    print("Train %s samples."%n_training_samples)
    
    for _ in range(epochs):
        # train loss
        epoch_train_loss = 0.0
        for inp, labels in train_loader:
            inp = inp.unsqueeze(1)
            inp = Variable(inp)
            out = model(inp)

            labels = Variable(torch.LongTensor(labels))
            loss = criterion(out, labels)
            epoch_train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
        # validation loss
#         epoch_val_loss = 0.0
#         for inp, labels in val_loader:
#             inp = inp.unsqueeze(1)
#             out = model(inp)        
#             loss = criterion(out, labels)
#             epoch_val_loss += loss


        print(str(epoch_train_loss.tolist()) + ' ')
#         print(str(epoch_val_loss.tolist()) + '\n')


def main(argv):
    print('Read config file %s.'%argv[0])
    args = readConfig.readConfig(argv[0])
    print(args)
    # load embeddings
    embed_matrix = pickle.load(open(args['embed_path']+'/embed.pkl', 'rb'))
    print("Load embedding of size %s"%embed_matrix.shape[1])
    w2i = pickle.load(open(args['w2i']+'/w2i.pkl', 'rb'))
    i2w = pickle.load(open(args['i2w']+'/i2w.pkl', 'rb'))
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
    if args['action'] == 'train':
        # load the preprocessed data
        cls, texts =preprocess.readData(args['input_path']+"/train_classes.data", args['input_path']+"/train_texts.data") 
        train_set = dataset.TextDataset(texts, cls, w2i, int(args['max_sen_len']))
        print(len(texts))
        del cls
        del texts

        if args['model']=='shallowCNN':
            print("Start training shallowCNN")
            model = ShallowCNN.ShallowCNN(model_args)
            model.load(args['load_model_from'])
            print(model.eval)
            train(model, train_set, epochs = int(args['epoch']), batch_size=int(args['batch']), cv=float(args['cv']), start=0, end=len(train_set))
            model.save(args['output_path']+ 'polar_'+(args['epoch'])+ '.model2')
            # for e in range(int(args['epoch'])): # the true num of epochs
            #     for i in range(0,140):
            #         train(model, train_set, epochs = 1, batch_size=int(args['batch']), cv=float(args['cv']), start=i*4000, end=(i+1)*4000)
            #         if i == 139 or i == 0:
            #             model.save(args['output_path']+ 'new_polar1275-'+ str(e)+ '-' + str(i) + '.model')

        elif args['model']=='DPCNN':
            print("Start training DPCNN")
            model = DPCNN.DPCNN(model_args)
            model.load(args['load_model_from'])
            print(model.eval)
            train(model, train_set, epochs = int(args['epoch']), batch_size=int(args['batch']), cv=float(args['cv']), start=0, end=len(train_set))
            model.save(args['output_path']+ 'glove_'+(args['epoch'])+ '.model2')
            # for e in range(int(args['epoch'])): # the true num of epochs
            #     for i in range(0,140):
            #         train(model, train_set, epochs = 1, batch_size=int(args['batch']), cv=float(args['cv']), start=i*4000, end=(i+1)*4000)
            #         if i == 139 or i == 0:
            #             model.save(args['output_path']+ 'new_polar1275-'+ str(e)+ '-' + str(i) + '.model')
            
    elif args['action'] == 'test':
        # load the preprocessed data
        cls, texts =preprocess.readData(args['input_path']+"/test_classes.data", args['input_path']+"/test_texts.data") 
        test_set = dataset.TextDataset(texts, cls, w2i, int(args['max_sen_len']))
        print(len(test_set[0][0]), (test_set[0][1]))
        print(len(texts))
        # del cls
        # del texts

        if args['model']=='DPCNN':
            print("Start testing DPCNN")
            model = DPCNN.DPCNN(model_args)
        elif args['model'] == 'shallowCNN':
            print("Start testing shallowNN")
            model = ShallowCNN.ShallowCNN(model_args)

        model.load(args['load_model_from'])
        print(model.eval)
        model.setDropout(0)

        # Test with captum
        np.random.seed(1) # used to be 1
        attr_folder = '../../data/numbers/ig/'
        grad_threshold = 1 / 100000

        if args.get('gradients', '') == 'True':
            # In order to save original sentences in json file
            text_sentences = []
            with open(args['input_path'] + '/texts_sentences.data', 'r') as sent_f:
                for line in sent_f:
                    text_sentences.append(line.split('<_SEP_>'))

            for p in np.random.randint(300,size=20):
                model.forwardEmbedLayer = False
                num_rows = min(129, len(texts[p]))
                ig = IntegratedGradients(model)

                input = test_set[p][0]
                label = test_set[p][1]
                input = model.embedding_layer(input)
                input = input.view(1, 1, model_args['max_len'], model_args['dim'])
                out = model(input)
                predicted = torch.argmax(out, dim=1).tolist()[0]
                baseline = torch.zeros((model_args['max_len'],model_args['dim']))
                baseline = baseline.view(1, 1, model_args['max_len'], model_args['dim'])
                # attributions_out = input_x_gradient.InputXGradient(model).attribute(input, target=label)
                # attributions_out = guided_grad_cam.GuidedGradCam(model, model.conv3).attribute(input, target=label)
                attributions_out, delta = ig.attribute(input, baseline, target=label, return_convergence_delta=True)


                attributions = attributions_out.view(model_args['max_len'], model_args['dim']).detach().numpy()

                # flaten
                flat_attribution = attributions_out.view((model_args['max_len'] * model_args['dim'], )).detach().numpy()
                flat_attribution += np.ones(model_args['max_len']*model_args['dim']) / 100000
                # normalize with z-score
                flat_attribution = stats.zscore(flat_attribution, ddof=1)


                # print('IG Attributions:', attributions.shape)

                # check top K contributing words
                K = min(20,len(texts[p]))
                # use L2 norm
                word_contributions = [(i, np.sum(attributions[i])) for i in range(num_rows) ]
                word_contributions = sorted(word_contributions, key=lambda x: x[1], reverse=True)
                topk_words = [i for i,v in word_contributions[:K]]
                print([texts[p][x] for x in topk_words])

                # load dim names
                antonym_names = [b for a,b in pickle.load(open(args['load_dim_names'], 'rb'))]
                # build sorted list of attributions
                flat_attr_sorted = []
                # remember position in flat attribution list
                pos_counter = 0 
                word_counter = {}
                for i in range(num_rows):
                    w = texts[p][i]
                    word_counter[w] = word_counter.get(w, 0) + 1
                    # print(pos_counter / model_args['dim'])
                    row = flat_attribution[pos_counter: pos_counter+model_args['dim']]
                    temp_list = []
                    if i in topk_words:
                        for t in range(model_args['dim']):
                            attr_value = row[t]
                            a1 = antonym_names[t][0]
                            a2 = antonym_names[t][1]
                            word_pair = (a1,a2) if attr_value < 0 else (a2,a1)
                            temp_list.append( (w, word_counter[w], word_pair, abs(attr_value)) )

                        temp_list = sorted(temp_list, key=lambda x: x[3], reverse=True)
                        flat_attr_sorted.append(temp_list[:20])
                    pos_counter += model_args['dim']
                
                # construct dict then convert it to json
                dict_to_json = {
                    'Text' : text_sentences[p],
                    'Prediction' : 'Positive' if predicted > 0 else 'Negative',
                    'Label' : 'Positive' if label > 0 else 'Negative'
                }
                for index in range(len(flat_attr_sorted)):
                    word_dict = {}
                    antonym_dict = {}
                    for w, wc, antonym_pair, v in flat_attr_sorted[index]:
                        if not word_dict:
                            word_dict['Word'] = w 
                            word_dict['Contribution'] = str(word_contributions[topk_words[index]][1])
                            pos = [0,0]

                            for sent_index in range(len(text_sentences[p])):
                                pos[0] = sent_index
                                words = text_sentences[p][sent_index].split(' ')
                                for word_index in range(len(words)):
                                    pos[1] = word_index
                                    if words[word_index].lower() == w:
                                        wc -= 1
                                        # print(word, type(wc), type(pos[1]))
                                        if wc == 0:
                                            # print(word)
                                            word_dict['Position'] = pos
                                            break
                                if word_dict.get('Position'):
                                    break

                        antonym_dict[antonym_pair[0]+','+ antonym_pair[1]] = str(v)
                    word_dict['Antonyms'] = antonym_dict
                    dict_to_json[word_dict['Word']+"_"+str(index)] = word_dict
                    
                # write to json file
                # with open(attr_folder+'ig-'+str(p)+'.json', 'w') as output_file:
                    # json.dump(dict_to_json, output_file)



                # *********************plotting*****************************
                # # sort dims
                # sorted_dims = {}
                # for i in range(attributions.shape[1]):
                #     grad_sum = 0
                #     for j in range(attributions.shape[0]):
                #         grad_sum += attributions[j][i]
                #     sorted_dims[i] = grad_sum
                # sorted_dims = sorted(sorted_dims.items(), key= lambda x:x[1], reverse=True)

                # selected_dims = [d for d,_ in sorted_dims][:30]

                # attr_to_plot = np.array([attributions.T[i] for i in selected_dims]).T
                # dim_to_plot = [antonym_names[i] for i in selected_dims]
                # print('class is ', 'positive' if label>0 else 'negative')
                # # plots.show_polar_heat_map(attr_to_plot, dim_to_plot, texts[p], max_words=len(texts[p]), width=22, height=15+len(texts[p])//20, 
                # #     save_path='')#'temp_test_'+str(p)+'_'+str(label)+'_'+ str(predicted) +'.png')
                # # print('Convergence Delta:', delta)
                # ****************************************************
        else:
            print(test(model, test_set, batch_size=200, start = 0, end = len(test_set)))


if __name__ == "__main__":
    main(sys.argv[1:])



