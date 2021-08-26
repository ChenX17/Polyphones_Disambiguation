import os
import re
import json

def load_vocab(path_vocab):
    feature_to_index = dict()
    index_to_feature = dict()
    with open(path_vocab, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        if re.search('^[^\t]', line):
            name = line.strip()
            index = 0
            feature_to_index[name] = dict()
            index_to_feature[name] = dict()
            continue
        feature_to_index[name][line.strip()] = index
        index_to_feature[name][index] = line.strip()
        index += 1
    return feature_to_index, index_to_feature

def load_poly_lexicon():
    poly_lexicon_path = 'preprocess/poly_lexocon.json'
    f = open(poly_lexicon_path, 'r')
    poly_lexicon = json.load(f)
    f.close()
    return poly_lexicon

def build_tags_from_file(tags_file, padding_tag=0):
    with open(tags_file) as fp:
        lines = fp.readlines()
    tags = []
    for line in lines:
        tags.append(line.strip())
    tag_to_index = {'<pad>': padding_tag}
    for tag in tags:
        tag_to_index[tag] = len(tag_to_index)
    index_to_tag = {index: tag for tag, index in tag_to_index.items()}
    return tag_to_index, index_to_tag

def gen_mask(ids, vocab_size):
    mask = [0] * vocab_size
    for i_id in ids:
        mask[i_id] = 1
    return mask

def find_word(words, tags):
    for i, item in enumerate(tags):
        if item != '_':
            return words[i]
    return '_'
