import re
import math
from collections import Counter
import random
import jieba
import codecs
import numpy as np
jieba.initialize()
def replace_digits_by_zero(s):
    """用0替换字符串中的每个数字"""
    return re.sub('\d', '0', s)

def create_vocabs(file_path, vocab_path, label_path, vocab_size, is_RDBZ=False):
    all_text = []
    all_label = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            if is_RDBZ:
                all_text.append(replace_digits_by_zero(line[0]))
            else:
                all_text.append(line[0])
            all_label.append(line[-1])
    # Counter()统计all_text每个字的频数
    char_counter = Counter(all_text)
    count_pairs = char_counter.most_common(vocab_size - 2)  # most_common
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>', '<UNK>'] + list(words)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words) + '\n')
    # 构建标签
    word_to_id = dict(zip(words, range(len(words))))
    tag_counter = Counter(all_label)
    tags = list(tag_counter.keys())
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tags) + '\n')
    tag_to_id = dict(zip(tags, range(len(tags))))
    return word_to_id, tag_to_id

def load_vocabs(vocab_path, label_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        words = [w for w in f.read().split('\n') if len(w) == 1 or len(w) == 5]
    word_to_id = dict(zip(words, range(len(words))))
    with open(label_path, 'r', encoding='utf-8') as f:
        tags = [w for w in f.read().split('\n') if len(w) == 1]
    tag_to_id = dict(zip(tags, range(len(tags))))
    return word_to_id, tag_to_id

def read_data(path, is_RDBZ=False):
    labels, texts = list(), list()
    text, label = str(), str()
    line_idx = 0
    for line in open(path, 'r', encoding='utf-8').readlines():
        o_line = line
        line_idx += 1
        line = line.rstrip().lstrip()
        line = replace_digits_by_zero(line) if is_RDBZ else line
        arr = line.split()
        if len(arr) == 2:
            text += arr[0]
            label += arr[-1]
        elif len(arr) == 0:
            if len(text) == 0 or len(label) == 0:
                text, label = str(), str()
                continue
            texts.append(text)
            labels.append(label)
            text, label = str(), str()
        else:
            print('data_process format wrong. at ', line_idx, [o_line], arr)
    if len(text) == len(label) != 0:
        texts.append(text)
        labels.append(label)
    return texts, labels

class DataSet:
    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        self.num_batch = int(math.ceil(len(data) / batch_size))
        # 以文本的长度来排序
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(self.num_batch):
            batch_data.append(self.pad_data(sorted_data[int(i*batch_size): int((i+1)*batch_size)]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        # print("abc:" + str(self.len_data))
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]

def load_sentences(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num += 1
        line = line.rstrip()
        # print(list(line))
        if not line:  # None,False,0,空列表[],空字典{},空元祖(),都相当于false
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word = line.split()  # 一个字一个字的分割，转成列表
            # assert len(word) >= 2, print([word[0]])
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    none_index = tag_to_id["S"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])

    return data

def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs

def full_to_half(s):
    """
    Convert full-width character to half-width one
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)

def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)

def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    """
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature

def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    #old_weights是随机的初始权重
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    new_weights = old_weights
    print('Loading pretrained embeddings      from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        #print(len(line))
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
        #print(pre_trained)
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    temp = {}
    for key,value in id_to_word.items():
        temp[value] = key
    for i in range(n_words):
        word = temp[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
        c_found, c_lower, c_zeros
    ))
    return new_weights