import os
import json
import string
import wikiwords
import unicodedata
import numpy as np

from collections import Counter
from nltk.corpus import stopwords

flatten = lambda l: [item for sublist in l for item in sublist]

punc = frozenset(string.punctuation)
stop_word_set = set()

def get_stop_words():
    stop_words_dir = './data/vocabulary/stopwords'
    ls = os.listdir(stop_words_dir)

    for fn in ls:
        file_path = os.path.join(stop_words_dir, fn)
        with open(file_path, encoding='utf8') as file:
            for line in file.readlines():
                stop_word_set.add(line.strip())

get_stop_words()

def is_stopword(w):
    return w.lower() in stop_word_set



data_path = './data/vocabulary'
# science term
science_terms = set()
with open(os.path.join(data_path, 'v1_geo_specific.txt'), 'r',encoding='utf8') as f:
    for l in f.readlines():
        l = l.strip().split()
        if len(l) == 1:
            science_terms.add(l[0])


def is_science_term(w):
    return w in science_terms


baseline = wikiwords.freq('the')


def get_idf(w):
    return np.log(baseline / (wikiwords.freq(w.lower()) + 1e-10))


# no need to word indices for elmo
def load_data(path, elmo=None):
    from doc import Example
    data = []
    for line in open(path, 'r', encoding='utf-8'):
        try:
            data.append(Example(json.loads(line), elmo))
        except Exception as e:
            print(e)
            print(json.loads(line)['id'])
        # data.append(Example(json.loads(line)))
    print('Load %d examples from %s...' % (len(data), path))
    return data


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}  # NULL is padding
        self.ind2tok = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens


vocab, pos_vocab, ner_vocab, rel_vocab = Dictionary(), Dictionary(), Dictionary(), Dictionary()


def build_vocab(data=None):
    global vocab, pos_vocab, ner_vocab, rel_vocab
    # build word vocabulary 
    if os.path.exists(os.path.join(data_path, 'vocab')):
        print('Load vocabulary from ./data/vocab...')
        for w in open(os.path.join(data_path, 'vocab'), encoding='utf-8'):
            vocab.add(w.strip())
        print('Vocabulary size: %d' % len(vocab))

    # build part-of-speech vocabulary
    if os.path.exists(os.path.join(data_path, 'pos_vocab')):
        print('Load pos vocabulary from ./data/pos_vocab...')
        for w in open(os.path.join(data_path, 'pos_vocab'), encoding='utf-8'):
            pos_vocab.add(w.strip())
        print('POS vocabulary size: %d' % len(pos_vocab))

    # build named entity vocabulary
    if os.path.exists(os.path.join(data_path, 'ner_vocab')):
        print('Load ner vocabulary from ./data/ner_vocab...')
        for w in open(os.path.join(data_path, 'ner_vocab'), encoding='utf-8'):
            ner_vocab.add(w.strip())
        print('NER vocabulary size: %d' % len(ner_vocab))

    # Load conceptnet relation vocabulary
    if not os.path.exists(os.path.join(data_path, 'rel_vocab')):
        os.system("cut -d' ' -f1 data/concept.filter | sort | uniq > ./data/rel_vocab")
    print('Load relation vocabulary from ./data/rel_vocab...')

    for w in open(os.path.join(data_path, 'rel_vocab'), encoding='utf-8'):
        rel_vocab.add(w.strip())
    print('Rel vocabulary size: %d' % len(rel_vocab))


def gen_debug_file(data, prediction):
    writer = open('./data/output.log', 'w', encoding='utf-8')
    cur_pred, cur_choices = [], []
    for i, ex in enumerate(data):
        if i + 1 == len(data):
            cur_pred.append(prediction[i])
            cur_choices.append(ex.choice)
        if (i > 0 and ex.id[:-1] != data[i - 1].id[:-1]) or (i + 1 == len(data)):
            writer.write('Passage: %s\n' % data[i - 1].passage)
            writer.write('Question: %s\n' % data[i - 1].question)
            for idx, choice in enumerate(cur_choices):
                writer.write('%s  %f\n' % (choice, cur_pred[idx]))
            writer.write('\n')
            cur_pred, cur_choices = [], []
        cur_pred.append(prediction[i])
        cur_choices.append(ex.choice)

    writer.close()
