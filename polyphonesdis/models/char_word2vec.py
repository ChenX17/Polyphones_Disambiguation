#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import math


class CHARW2VNet(nn.Module):
    """The BiLSTM model for sequence tagging.

    Attributes:
        device (:obj:`device`):
        vocab_size (int): The input word vocab size.
        tags_size (int): The tag size.
        embedding_dim (int): The size of input word embedding.
        num_layers (int): The number of BiLSTM layers.
        hidden_dim (int): The size of BiLSTM hidden vector.
                         Note: the hidden size of each LSTM cell is hidden_dim // 2.
        pretrained_embedding (:obj:`Embedding`, optional):

    """

    def __init__(self,
                 device='cuda:0',
                 vocab_size=4941,
                 tags_size=204,
                 embedding_dim=64,
                 num_layers=1,
                 hidden_dim=64,
                 pretrained_embedding=None,
                 pretrained_embedding_dim=200):
        super(CHARW2VNet, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.tagset_size = tags_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pretrained_embedding = pretrained_embedding

        if pretrained_embedding:
            self.word_embeds = pretrained_embedding
            self.pretrained_embedding_dim = pretrained_embedding_dim
        else:
            self.word_embeds = nn.Embedding(vocab_size,
                                            self.embedding_dim).to(device)
            self.pretrained_embedding_dim = pretrained_embedding_dim
        self.dropout_rate = 0.0
        self.bilstm = nn.LSTM(input_size=self.embedding_dim+self.pretrained_embedding_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers,
                              dropout=self.dropout_rate,
                              bidirectional=True,
                              batch_first=True).to(device)

        self.layernorm = nn.LayerNorm([self.embedding_dim]).to(device)
        self.layernorm_2 = nn.LayerNorm([self.hidden_dim*2]).to(device)                      
        self.dropout = nn.Dropout(0.3)
        self.dropout_linear = nn.Dropout(0.1)
        self.linear = nn.Linear(self.hidden_dim * 2,
                                    self.hidden_dim * 4).to(device)
        self.activation = nn.ReLU()
        # Maps the output of BiLSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim * 4,
                                    self.tagset_size).to(device)

    
    def ce_loss(self, input_features, tags, seq_lens):
        celoss = nn.CrossEntropyLoss(reduction='none')
        loss = celoss(input_features.transpose(1, 2), tags)
        mask = tags > 1
        loss = (loss * mask.float()).sum()/mask.sum().item()

        return loss
    
    def forward(self, inputs, target, seq_lens):
        """Returns the bilstm output features given the input sentence.

        :param batch_input:
        :return:
        """
        embedding_inputs = inputs['word2vecs']
        mask = inputs['mask']

        batch_size = embedding_inputs.shape[0]

        input_features = self.word_embeds(inputs['char'])
        input_features = self.dropout(self.layernorm(input_features))
        input_features = torch.cat((input_features,embedding_inputs.float()), 2)


        packed_input = pack_padded_sequence(input_features,
                                            seq_lens,
                                            batch_first=True)

        h0 = torch.zeros(self.num_layers * 2,
                         batch_size,
                         self.hidden_dim,
                         device=self.device)
        c0 = torch.zeros(self.num_layers * 2,
                         batch_size,
                         self.hidden_dim,
                         device=self.device)

        # blstm
        y, _ = self.bilstm(packed_input, (h0, c0))
        y, batch_sizes = y.data, y.batch_sizes
        y = self.layernorm_2(y)
        y = self.dropout(y)

        # linear
        y = self.linear(y)
        y = self.activation(y)
        y = self.dropout_linear(y)

        # classifier
        y = self.hidden2tag(y)
        y = PackedSequence(y, batch_sizes)

        y, _ = pad_packed_sequence(y, batch_first=True)
        mask = (torch.unsqueeze(1 - mask, dim=1)*(-1e5)).float().to(self.device)
        y = y + mask
        return y
