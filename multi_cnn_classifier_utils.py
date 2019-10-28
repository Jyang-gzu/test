import json
from collections import Counter
import numpy as np
import jieba
jieba.initialize()

def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences, entities, labels = list(), list(), list()
        for line in f.readlines():
            js = json.loads(line)
            entities.append(js['entity'])
            sentences.append(js['left']+'^'+js['entity']+'$'+js['right'])
            labels.append(js['label'])
    return sentences, entities, labels

def read_vocabs(vocab_path, label_path):
    word_to_id = read_to_dict(vocab_path)
    tag_to_id = read_to_dict(label_path)
    return word_to_id, tag_to_id

def read_to_dict(file_path):
    items = dict()
    with open(file_path, 'r', encoding='utf-8') as f:
        idx = 0
        for item in f.readlines():
            item = item.rstrip()
            if item != '':
                items[item] = idx
                idx += 1
    return items

def build_vocabs(file_path, vocab_path, label_path, vocab_size):
    sentences, entities, labels = read_data(file_path)

    all_data = []
    for content in sentences:
        all_data.extend(content)

    char_counter = Counter(all_data)
    count_pairs = char_counter.most_common(vocab_size - 2)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本padding为同一长度
    words = ['<PAD>', '<UNK>'] + list(words)
    with open(vocab_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write('\n'.join(words) + '\n')
    word_to_id = dict(zip(words, range(len(words))))

    tag_counter = Counter(labels)
    tags = list(tag_counter.keys())
    with open(label_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write('\n'.join(tags) + '\n')
    tag_to_id = dict(zip(tags, range(len(tags))))

    return word_to_id, tag_to_id

class DataStruct(object):
    def __init__(self, file_path, word_to_id, tag_to_id, seq_length=None):
        self.word_to_id = word_to_id
        self.tag_to_id = tag_to_id
        self.seq_length = seq_length
        sentences, entities, labels = read_data(file_path)
        self.data_size = len(labels)
        self.entities_pad = self.pad_sequences(entities)
        self.sentences_pad = self.pad_sequences(sentences)
        self.labels_onehot = self.onehot_labels(labels)

    def pad_sequences(self, sequences):
        if self.seq_length is None:
            max_length = 0
            for sequence in sequences:
                if len(sequence) > max_length:
                    max_length = len(sequence)
        else:
            max_length = self.seq_length
        sequences_id = list()
        for i in range(len(sequences)):
            sequence_id = list()
            sequence = sequences[i]
            for j in range(max_length):
                if j >= len(sequence):
                    break
                if sequence[j] in self.word_to_id:
                    sequence_id.append(self.word_to_id[sequence[j]])
                else:
                    sequence_id.append(self.word_to_id['<UNK>'])
            if len(sequence_id) < max_length:
                sequence_id = [self.word_to_id['<PAD>']] * (max_length-len(sequence_id)) + sequence_id
            sequences_id.append(sequence_id)
        return sequences_id

    def onehot_labels(self, tags):
        onehots = list()
        for tag in tags:
            onehot = [0] * len(self.tag_to_id)
            onehot[self.tag_to_id[tag]] = 1
            onehots.append(onehot)
        return onehots

    def batch_iter(self, batch_size=10):
        """生成批次数据"""
        self.num_batch = int((self.data_size - 1) / batch_size) + 1
        # indices = np.random.permutation(np.arange(self.data_size, dtype=np.int16))
        indices = np.arange(self.data_size, dtype=np.int16)
        entities_shuffle = np.asarray(self.entities_pad)[indices]
        sentences_shuffle = np.asarray(self.sentences_pad)[indices]
        labels_shuffle = np.asarray(self.labels_onehot)[indices]
        # labels_shuffle = np.asarray(self.labels_onehot)[self.data_size]
        for i in range(self.num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, self.data_size)
            yield (sentences_shuffle[start_id:end_id], entities_shuffle[start_id:end_id], labels_shuffle[start_id:end_id])
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
