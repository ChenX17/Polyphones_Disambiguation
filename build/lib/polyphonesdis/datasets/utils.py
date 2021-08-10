import os

def load_vocab(path_vocab):
    """Load vocabulary from a file containing one word per line.

    Args:
        path_vocab (str): Path of vocab file.

    Returns: A map from word to index, and a map from index to word.

    """
    word_to_index = {}
    index_to_word = {}
    with open(path_vocab, "r") as fin:
        index = 0
        for line in fin:
            word_to_index[line.strip()] = index
            index_to_word[index] = line.strip()
            index += 1
    return word_to_index, index_to_word


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