"""CHAR&WORD2VEC dataset"""

import os
import re

import numpy as np
import torch
import pickle

import polyphonesdis.core.logging as logging
from polyphonesdis.datasets.utils import load_vocab, build_tags_from_file, gen_mask, find_word, load_poly_lexicon


class CHARW2CDataSet(torch.utils.data.Dataset):
    """CHAR&WORD2VEC dataset"""
    def __init__(self, data_path, split, vocab_path, tag_path):
        f = open(data_path+'/'+split+'files.txt')
        self.data_list = f.readlines()
        f.close()
        self.data_list = [data_path+'/'+'total_'+split+'/'+item.strip() for item in self.data_list]
        self.feature_to_index, self.index_to_feature = load_vocab(vocab_path)
        self.tag_to_index, self.index_to_tag = build_tags_from_file(tag_path)
        self.unk_feat_id = 1
        self.pad_tag_id = 0
        self.poly_lexicon = load_poly_lexicon()


    
    def __getitem__(self, index):
        lines = pickle.load(open(self.data_list[index], 'rb'))
        words = list(lines['texts'])
        words = ['ENG' if item=='E' else item for item in words]
        tags = lines['labels'].split(' ')
        #word2vecs = lines['embeddings']
        word2vecs = lines['vec']

        word_ids = [
            self.feature_to_index['word'][w]
            if w in self.feature_to_index['word'] else self.unk_feat_id for w in words
        ]
        tags_ids = [self.tag_to_index[t] for t in tags]
        word = find_word(words, tags)
        pinyins = self.poly_lexicon[word]
        mask = gen_mask([self.tag_to_index[t] for t in pinyins], len(self.tag_to_index))
        return word_ids, word2vecs, tags_ids, mask

    def pad_and_sort_by_len(self, batch):
        """Prepare batch input for packing.

        Apply padding and sort data by length in descending order.
        Here we simply use unk feature ids as padding ids.
        Args:
            batch (list of x, y tuples): batch input, for example [([x1, x2], [y1, y2]), ...]
        Returns:
            (Tensor) the padded x, y and seq_lens
        """
        sorted_batch = sorted(batch, key=lambda x_y: len(x_y[0]), reverse=True)
        seq_lens = torch.tensor([len(features[0]) for features in sorted_batch],
                                )
        maxlen = max(seq_lens).item()
        padded_x = torch.tensor([
            x_y[0] + [self.unk_feat_id] * (maxlen - len(x_y[0]))
            for x_y in sorted_batch
        ],
                                dtype=torch.long,
                                )
        padded_y = torch.tensor([
            x_y[2] + [self.pad_tag_id] * (maxlen - len(x_y[2])) for x_y in sorted_batch
        ],
                                dtype=torch.long,
                                )
        padding_f4 = np.zeros((len(sorted_batch), sorted_batch[0][1].shape[0], sorted_batch[0][1].shape[1]))
        for i,x_y in enumerate(sorted_batch):
            if seq_lens[i] != x_y[1].shape[0]:
                import pdb;pdb.set_trace()
            padding_f4[i, :seq_lens[i], :]=x_y[1]
        padding_f4 = torch.tensor(padding_f4, dtype=torch.float16,)
        mask = torch.tensor([x_y[3] for x_y in sorted_batch])
        features_dict = {'char': padded_x, 'word2vecs': padding_f4, 'mask': mask}
        return features_dict, padded_y, seq_lens
    
    def __len__(self):
        return len(self.data_list)

