"""CHAR&WORD2VEC dataset"""

import os
import re

import numpy as np
import torch
import pickle

import polyphonesdis.core.logging as logging
from polyphonesdis.datasets.utils import load_vocab, build_tags_from_file, gen_mask, find_word


class CHARW2CPOSCWSFLAGDataSet(torch.utils.data.Dataset):
    """CHAR&WORD2VEC dataset"""
    def __init__(self, data_path, split, vocab_path, tag_path):
        f = open(data_path+'/'+split+'files.txt')
        self.data_list = f.readlines()
        f.close()
        self.data_list = [data_path+'/'+'total_'+split+'/'+item.strip() for item in self.data_list]
        self.feature_to_index, self.index_to_feature = load_vocab(vocab_path)
        self.tag_to_index, self.index_to_tag = build_tags_from_file(tag_path)


    
    def __getitem__(self, index):
        lines = pickle.load(open(self.data_list[index], 'rb'))
        words = list(lines['texts'])
        words = ['ENG' if item=='E' else item for item in words]
        tags = lines['labels'].split(' ')
        word2vecs = lines['embeddings']
        pos = lines['pos']
        cws = lines['cws']
        flag = lines['flag']
        import pdb;pdb.set_trace()

        word_ids = [
            self.feature_to_index['word'][w]
            if w in self.feature_to_index['word'] else self.unk_feat_id for w in words
        ]
        tags_ids = [self.tag_to_index[t] for t in tags]
        pos_ids = [self.feature_to_index['pos'][w] if w in self.feature_to_index['pos'] else self.unk_feat_id for w in pos]
        cws_ids = [self.feature_to_index['position'][w] if w in self.feature_to_index['position'] else self.unk_feat_id for w in cws]
        word = find_word(words, tags)
        pinyins = self.poly_lexicon[word]
        mask = gen_mask([self.tag_to_index[t] for t in pinyins], len(self.tag_to_index))
        return word_ids, word2vecs, tags_ids, mask, pos_ids, cws_ids, flag

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
                                device=self.device)
        maxlen = max(seq_lens).item()
        padded_x = torch.tensor([
            x_y[0] + [self.unk_feat_id] * (maxlen - len(x_y[0]))
            for x_y in sorted_batch
        ],
                                dtype=torch.long,
                                device=self.device)
        padded_y = torch.tensor([
            x_y[2] + [self.pad_tag_id] * (maxlen - len(x_y[2])) for x_y in sorted_batch
        ],
                                dtype=torch.long,
                                device=self.device)
        word2vecs = np.zeros((len(sorted_batch), sorted_batch[0][1].shape[0], sorted_batch[0][1].shape[1]))
        for i,x_y in enumerate(sorted_batch):
            if seq_lens[i] != x_y[1].shape[0]:
                import pdb;pdb.set_trace()
            word2vecs[i, :seq_lens[i], :]=x_y[1]
        word2vecs = torch.tensor(word2vecs, dtype=torch.float16, device=self.device)
        mask = torch.tensor([x_y[3] for x_y in sorted_batch])

        padded_pos = torch.tensor([
            x_y[4] + [self.pad_tag_id] * (maxlen - len(x_y[4])) for x_y in sorted_batch
        ],
                                dtype=torch.long,
                                device=self.device)
        padded_cws = torch.tensor([
            x_y[5] + [self.pad_tag_id] * (maxlen - len(x_y[5])) for x_y in sorted_batch
        ],
                                dtype=torch.long,
                                device=self.device)
        padded_flag = torch.tensor([
            x_y[6] + [self.pad_tag_id] * (maxlen - len(x_y[6])) for x_y in sorted_batch
        ],
                                dtype=torch.long,
                                device=self.device)
        features_dict = {'char': padded_x, 'word2vecs': word2vecs, 'mask': mask, 'pos': padded_pos, 'cws': padded_cws, 'flag': padded_flag}
        return features_dict, padded_y, seq_lens
    
    def __len__(self):
        return len(self.data_list)

