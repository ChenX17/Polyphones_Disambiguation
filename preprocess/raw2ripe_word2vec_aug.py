# -*- coding: utf-8 -*
import re
import glob
import pickle
from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models import KeyedVectors
import jieba
import jieba.posseg
from multiprocessing import Pool
import numpy as np
import copy
import math

# raw_dir = 'raw_cut0802/'
# ripe_dir = 'ripe_char_word2vec_pos_cws_flag/'

raw_dir = 'raw_cut_0824/'
ripe_dir = 'ripe_aug/'
re_en = re.compile('E{2,}')

base = 10000
print(glob.glob(raw_dir+'*.txt'))

def convert_eng(text):
    ori_text = text
    re_en = re.compile(r'[a-zA-Z]+')
    text = re_en.sub('E', text)
    return text

def convert_digits(text):
    re_d = re.compile(r'\d+')
    ori_text = text
    text = re_d.sub(u'一', text)
    return text

def train_word2vec(sentences, ripe_dir='ripe', mode='char'):
    model = Word2Vec(sentences, min_count=1)
    model.save(ripe_dir+mode+'.model')
    return model

def re_cut(model, words):
    words = [words]
    new_cut = []
    max_length = max([len(item) for item in words])
    new_pos = []
    while True:
        new_cut = []
        flag = True
        for item in words:
            if re_en.search(item):
                for char in item:
                    new_cut.append(char)
            elif item not in model.key_to_index and len(item) >= 3:
                jieba.del_word(item)
                flag=False
                re_cut_words = list(jieba.cut(item))
                for word in re_cut_words:
                    new_cut.append(word)

                new_cut.append(item)
        words = new_cut
        if flag:
            break
    return words

def get_cws(cuted_text):
    cws = []
    for item in cuted_text:
        if len(item)== 1:
            cws.append('S')
        elif len(item)==2:
            cws.append('B')
            cws.append('E')
        else:
            count = len(item)-2
            cws.append('B')
            while count >0:
                cws.append('M')
                count -= 1
            cws.append('E')
    return cws

def get_pos(texts ,pos_list):
    new_pos = []
    if len(texts) != len(pos_list):
        import pdb;pdb.set_trace()
    for i, item in enumerate(texts):
        count = len(item)
        while count > 0:
            new_pos.append(pos_list[i])
            count -= 1
    return new_pos

def get_flag(label):
    flag = [0] * len(label)
    for i,item in enumerate(label):
        if item != '_':
            flag[i] = 1
    return flag

def get_vec(model, text, pos, idx=0):
    vecs = []
    cuted_texts = []
    cuted_pos = []
    while ' ' in text:
        text.remove(' ')
    for char in text:
        count = len(char)
        while count > 0:
            try:
                vecs.append(model[char])
                count -= 1
                if count == 0:
                    cuted_texts.append(char)
            except:
                items = [char]
                length_sum = sum([len(item) for item in items])
                count -= length_sum

                for item in items:
                    len_item = len(item)
                    while len_item > 0:
                        try:
                            vecs.append(model[item])
                            len_item -= 1
                            if len_item == 0:
                                cuted_texts.append(item)
                        except:
                            # 词不在model的vocab中
                            if item[-len_item] not in model.key_to_index:
                                vecs.append(np.zeros_like(model[u'中']))
                            else:
                                vecs.append(model[item[-len_item]])
                            cuted_texts.append(item[-len_item])
                            len_item -= 1
    
    new_pos = [tmp.flag for tmp in list(jieba.posseg.cut(' '.join(cuted_texts)))]
    new_texts = [tmp.word for tmp in list(jieba.posseg.cut(' '.join(cuted_texts)))]

    processed_pos = []
    flag = True
    processed_pos.append(new_pos[0])
    for i, item in enumerate(new_texts):
        if not flag:
            continue
        if i == len(new_texts)-1:
            continue
        elif new_texts[i] == ' ' and new_texts[i+1] == ' ':
            processed_pos.append(new_pos[i+1])
        elif new_texts[i] == ' ' and new_texts[i+1] != ' ':
            processed_pos.append(new_pos[i+1])
        else:
            continue


    if len(processed_pos) != len(cuted_texts):
        print(new_texts)
        print(cuted_texts)
        import pdb;pdb.set_trace()
    if len(vecs) != len(''.join(text)):
        import pdb;pdb.set_trace()
    assert len(vecs) == len(''.join(text)), 'different length'
    return np.array(vecs), processed_pos, cuted_texts

re_cn = re.compile(r'[\u4e00-\u9fa5]')


def gen_aug(model_words, word_list, times, char):
    times = min(len(word_list)-1, times)
    res_dict = {}
    for i in range(len(word_list)):
        if char in word_list[i]:
            continue
        else:
            try:
                if not re_cn.search(word_list[i]):
                    continue
                similar_words = model_words.similar_by_word(word_list[i], topn=times)
                for j in range(times):
                    aug_word_list = [item for item in word_list]
                    aug_word_list[i] = similar_words[j][0]
                    # skip 股票代码
                    if not re_cn.search(aug_word_list[i]):
                        continue
                    res_dict[' '.join(aug_word_list) + '\t' + str(i) + '\t'+word_list[i] + '\t' + similar_words[j][0]] = similar_words[j][1]
            except:
                continue

    sorted_res = sorted(res_dict.items(), key=lambda x: x[1], reverse=True)
    if times > len(sorted_res):
        times = min(times,len(sorted_res))
    result = sorted_res[:times]
    return result



def read_file(filename, pro=0):
    re_cut = re.compile(r'(【.】)')
    f = open(filename, 'r', encoding='utf-8')
    texts = []
    labels = []
    texts_lines = f.readlines()
    f.close()

    batch_size = 128

    tmp = []
    tmp_label = []
    idx = 0
    data = []
    sentences = []
    phrases = []
    filelist = []
    model_words = KeyedVectors.load_word2vec_format("embeddings/70000-small.txt")
    aug_model_words = KeyedVectors.load_word2vec_format("embeddings/70000-small.txt")
    size_dict = {}
    count_dict = {}
    word_max_count = {}
    for line in texts_lines:
        if len(line.strip().split('\t'))!=4:
            import pdb;pdb.set_trace()
        uttid, word, label, text = line.strip().split('\t')
        if (word+':'+label) not in count_dict.keys():
            count_dict[word+':'+label]=0
            count_dict[word+':'+label]+=1
        else:
            count_dict[word+':'+label]+=1
    for key in count_dict.keys():
        if key.split(':')[0] not in word_max_count.keys():
            word_max_count[key.split(':')[0]]=count_dict[key]
        elif word_max_count[key.split(':')[0]] < count_dict[key]:
            word_max_count[key.split(':')[0]]=count_dict[key]
        else:
            continue
    total_aug_times = 0
    for key in count_dict.keys():
        #size_key = round(math.log(base/count_dict[key]))
        size_key = round(math.log(word_max_count[key.split(':')[0]]/count_dict[key]))
        if size_key < 0:
            size_key = 0
        size_dict[key] = size_key
        total_aug_times += size_key * count_dict[key]
    print('total_aug_times is :', total_aug_times)
    f = open('auged_text_new.txt', 'a+')
    for line in texts_lines:
        if len(line.strip().split('\t'))!=4:
            import pdb;pdb.set_trace()
        uttid, word, label_i, text = line.strip().split('\t')
        if word+':'+label_i not in count_dict.keys():
            continue
        elif size_dict[word+':'+label_i] == 0:
            continue
        else:
            times = size_dict[word+':'+label_i]

        parts = re_cut.split(text)
        label = ['_']*len(parts[0]) + [label_i] + ['_']*len(parts[2])
        text = parts[0] + word + parts[2]
        text = list(text)
        ori_text = copy.deepcopy(text)

        space_num = 0

        for i,tmp_char in enumerate(ori_text):
            if tmp_char == ' ':
                text.pop(i-space_num)
                label.pop(i-space_num)
                
                space_num+=1

        text = ''.join(text)
        label = ' '.join(label)

        word_pos = list(jieba.posseg.cut(text))
        
        word_list = [item.word for item in word_pos]
        pos_list = [item.flag for item in word_pos]

        auged_texts = gen_aug(aug_model_words, word_list, times=times, char=word)

        auged_ids = 0

        for i in range(len(auged_texts)):
            idx += 1
            new_word_list = auged_texts[i][0].split('\t')[0].split(' ')
            word_idx = int(auged_texts[i][0].split('\t')[1])
            ori_label_idx = len(''.join(word_list[:word_idx]))
            label_idx = len(''.join(word_list[:word_idx+1]))
            new_label_idx = len(''.join(new_word_list[:word_idx+1]))
            
            if label_idx != new_label_idx:
                new_label = label.split(' ')
                new_label = new_label[:ori_label_idx] + ['_'] * len(new_word_list[word_idx]) + new_label[label_idx:]
                new_label = ' '.join(new_label)
                flag = get_flag(new_label.split(' '))
            else:
                flag = get_flag(label.split(' '))
                new_label = label

            vec, pos, cuted_text = get_vec(model_words, new_word_list, pos_list)
            poly_idx = 0
            TMP_LABEL = new_label.split(' ')
            for k in range(len(TMP_LABEL)):
                if TMP_LABEL[k] != '_':
                    poly_idx = k
            
            pos = get_pos(cuted_text, pos)
            cws = get_cws(cuted_text)
            
            text = ''.join(cuted_text)
            if not len(text)==len(vec)==len(pos)==len(cws)==len(flag):
                import pdb;pdb.set_trace()

            assert len(text)==len(vec)==len(pos)==len(cws)==len(flag)
            f.write(uttid+"_aug_"+str(i)+'\t'+word+'\t'+label_i+'\t'+text[:poly_idx]+'【'+text[poly_idx]+'】'+text[poly_idx+1:]+'\n')
            '''
            df_data = {
                        "texts": text,
                        "labels": new_label,
                        "embeddings": vec,
                        "cws": cws,
                        "pos": pos,
                        "flag": flag,
                        }
            save_path = ripe_dir+filename.split('.')[0].split('/')[-1]+'/'+uttid+"_aug_"+str(i)+".pkl"
            filelist.append(uttid+"_aug_"+str(i)+".pkl")
            f = open(save_path, 'wb')
            df_data = pickle.dump(df_data, f)
            f.close()
            if idx %10 ==0:
                print(idx, " sentences!")
            '''
            if idx %200 ==0:
                print(idx, " sentences!")
    f.close()
if __name__=='__main__':
    from multiprocessing import Process
    file_list = sorted(glob.glob(raw_dir+'*.txt'))[::-1]
    for item in file_list:
        if 'train' not in item:
            continue
        #process_list = []
        #for i in range(5):
        #    p = Process(target=read_file,args=(item,i,))
        #    p.start()
        #    process_list.append(p)

        #for i in process_list:
        #    p.join()
        read_file(item)
        #p.close()
        #p.join()
