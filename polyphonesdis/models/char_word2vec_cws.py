#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import math


class CHARW2VCWSNet(nn.Module):
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

    def __init__(self, cfg):
        super(CHARW2VCWSNet, self).__init__()
        self.device = cfg.DEVICE
        self.vocab_size = cfg.VOCAB_SIZE
        self.tagset_size = cfg.TAGS_SIZE
        self.embedding_dim = cfg.EMBEDDING_DIM
        self.num_layers = cfg.NUM_LAYERS
        self.hidden_dim = cfg.HIDDEN_DIM
        self.pretrained_embedding_dim = cfg.PRETRAINED_EMBEDDING_DIM

        self.word_embeds = nn.Embedding(self.vocab_size,
                                        self.embedding_dim).to(self.device)
        self.cws_embeds = nn.Embedding(cws_size,
                                        self.embedding_dim).to(self.device)
        self.dropout_rate = cfg.DROPOUT_INTER_BLSTM
        
        self.bilstm = nn.LSTM(input_size=2*self.embedding_dim+self.pretrained_embedding_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers,
                              dropout=self.dropout_rate,
                              bidirectional=True,
                              batch_first=True).to(self.device)

        self.layernorm = nn.LayerNorm([self.embedding_dim]).to(self.device)                      
        self.dropout_embedding = nn.Dropout(cfg.DROPOUT_EMBEDDING)
        self.dropout_blstm = nn.Dropout(cfg.DROPOUT_BLSTM)
        self.dropout_linear = nn.Dropout(cfg.DROPOUT_LINEAR)
        self.linear = nn.Linear(self.hidden_dim * 2,
                                    self.hidden_dim * 4).to(self.device)
        self.activation = nn.ReLU()
        # Maps the output of BiLSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim * 4,
                                    self.tagset_size).to(self.device)
    
    def forward(self, inputs, target, seq_lens):
        """Returns the bilstm output features given the input sentence.

        :param batch_input:
        :return:
        """
        embedding_inputs = inputs['word2vecs']
        mask = inputs['mask']

        batch_size = embedding_inputs.shape[0]

        input_features = self.word_embeds(inputs['char'])
        input_features = self.dropout_embedding(self.layernorm(input_features))

        cws_features = self.cws_embeds(inputs['cws'])
        cws_features = self.dropout_embedding(self.layernorm(cws_features))

        input_features = torch.cat((input_features, embedding_inputs.float(), cws_features), 2)


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
        y = self.dropout_blstm(y)

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
