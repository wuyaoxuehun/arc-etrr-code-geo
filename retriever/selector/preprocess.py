import os
import sys
import spacy
import copy
import json
import math
import glob
import wikiwords
import re

import string

exclude = set(string.punctuation)
import pickle

from collections import Counter
from tqdm import tqdm


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


import re


class JiebaTokenizer():
    ENT_NAMES = ['ng', 'nr', 'ns', 'nz']

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        import jieba
        jieba.set_dictionary('./data/vocabulary/dict.txt.small')
        # jieba.load_userdict('./data/vocabulary/v1.txt')
        # jieba.set_dictionary('./data/vocabulary/vocab')
        import jieba.posseg as pseg
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        self.nlp = pseg

    def tokenize(self, text):
        # We don't treat new lines as tokens. Besides, need to remove space at the beginning.
        clean_text = text.replace('\n', ' ').replace('\t', ' ').replace('/', ' / ').strip()
        # clean_text = re.sub(r'[^\x00-\x7F]+', ' ', clean_text)
        clean_text = re.sub(r'。$', '', clean_text)
        clean_text.strip()
        clean_text = re.sub(r'[(（].{0,4}[）)]$', '', clean_text)

        clean_text = clean_text.strip(' ')
        # remove consecutive spaces
        clean_text = ''.join(clean_text.split())
        tokens = self.nlp.cut(clean_text)

        data = []
        for word, flag in tokens:
            # Get whitespace
            data.append((
                word,
                0,  # width not used
                (0, 0),  # span not used
                flag,
                0,  # lemma not used
                flag if flag.lower() in JiebaTokenizer.ENT_NAMES else '',
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={'non_ent': ''})


TOK = None


def init_tokenizer():
    global TOK
    TOK = JiebaTokenizer(annotators={'pos', 'lemma', 'ner'})


def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': tokens.words(),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'ner': tokens.entities()
    }
    return output


#
from utils import is_stopword, is_science_term

from wordfreq import word_frequency


def compute_features(q_dict, c_dict):
    # in_c, lemma_in_c, tf
    c_words_set = set([w.lower() for w in c_dict['words']])
    in_c = [int(w.lower() in c_words_set and not is_stopword(w)) for w in q_dict['words']]

    # tf = [0.1 * math.log(wikiwords.N * wikiwords.freq(w.lower()) + 10) for w in q_dict['words']] 
    tf = [word_frequency(w.lower(), 'zh') for w in q_dict['words']]
    # tf = [float('%.2f' % v) for v in tf]

    # q_words = Counter(filter(lambda w: not is_punc(w), q_dict['words']))
    from conceptnet import concept_net
    q_c_relation = concept_net.p_q_relation(q_dict['words'], c_dict['words'])
    assert len(tf) == len(in_c) == len(q_c_relation)

    q_is_science_term = [is_science_term(w) for w in q_dict['words']]
    q_is_cand = [1 if not is_stopword(w) else 0 for w in q_dict['words']]

    return {
        'in_c': in_c,
        'tf': tf,
        'q_c_relation': q_c_relation,
        'q_is_science_term': q_is_science_term,
        'q_is_cand': q_is_cand
    }


## do not use d_dict (i.e. information from paragraph)
def get_selector_example(q_id, q_dict, c_dict, label):
    return {
        'id': q_id,
        'q_words': ' '.join(q_dict['words']),
        'q_pos': q_dict['pos'],
        'q_ner': q_dict['ner'],
        'c_words': ' '.join(c_dict['words']),
        'label': label
        # 'answer': answer
    }


import utils


def preprocess_selector_dataset(path, is_test_set=False):
    filename = path.split('\\')[-1]
    print(path)
    print(filename)
    # write to ./data
    writer = open('./data/' + filename.replace('.json', '') + '-processed.json', 'w', encoding='utf8')
    ex_cnt = 0
    with open(path, encoding='utf8') as file:
        reader = json.load(file)
        for q_id, d in enumerate(reader):
            question_toks = tokenize(d['background'] + ' ' + d['question'])  # tokenize the question
            # term_toks = []  # tokenize each words
            term_toks = []
            for term in d['keywords']:
                term_toks.extend(tokenize(term)['words'])
            c_dict = tokenize(' '.join([d['a'], d['b'], d['c'], d['d']]))
            label = [(True if (tok in term_toks and tok not in exclude) else False) for tok in question_toks['words']]
            assert len(label) == len(question_toks['words'])
            q_dict = question_toks
            example = get_selector_example(d['questionID'], q_dict, c_dict, label)

            # get all pos tags, only once for all
            # for item in example['q_pos']:
            #     utils.pos_vocab.add(item)

            example.update(compute_features(q_dict, c_dict))  # compute features
            print(example)
            writer.write(json.dumps(example, ensure_ascii=False))
            writer.write('\n')
            ex_cnt += 1

    print('Found %d examples in %s...' % (ex_cnt, path))
    writer.close()
    # write pos_vocab
    # writer = open('./data/vocabulary/pos_vocab', 'w', encoding='utf-8')
    # writer.write('\n'.join(utils.pos_vocab.tokens()))
    # writer.close()


# build from all json file of multiple datasets
def build_vocabulary():
    jieba_dic_small_file = './data/vocabulary/dict.txt.small'

    with open(jieba_dic_small_file, encoding='utf8') as file:
        for line in file.readlines():
            utils.vocab.add(line.split()[0])

    # geo_dic_file = './data/vocabulary/v1.txt'
    # with open(geo_dic_file, encoding='utf8') as file:
    #     for line in file.readlines():
    #         utils.vocab.add(line.strip())

    print('Vocabulary size: %d' % len(utils.vocab))
    writer = open('./data/vocabulary/vocab', 'w', encoding='utf-8')
    writer.write('\n'.join(utils.vocab.tokens()))
    writer.close()


def preprocess_conceptnet(path):
    # build vocab from arc dataset
    build_vocabulary()

    print("vocab built...")
    writer = open('./data/concept.filter', 'w', encoding='utf-8')

    def _get_lan_and_w(arg):
        arg = arg.strip('/').split('/')
        return arg[1], arg[2]

    for line in open(path, 'r', encoding='utf-8'):
        fs = line.split('\t')
        relation, arg1, arg2 = fs[1].split('/')[-1], fs[2], fs[3]
        lan1, w1 = _get_lan_and_w(arg1)
        if lan1 != 'zh' or not all(w in utils.vocab for w in w1.split('_')):
            continue
        lan2, w2 = _get_lan_and_w(arg2)
        if lan2 != 'zh' or not all(w in utils.vocab for w in w2.split('_')):
            continue
        obj = json.loads(fs[-1])
        if obj['weight'] < 1.0:
            continue
        writer.write('%s %s %s\n' % (relation, w1, w2))
    writer.close()


if __name__ == '__main__':
    init_tokenizer()

    # preprocess conceptnet
    if len(sys.argv) > 1 and sys.argv[1] == 'conceptnet':
        # will build vocab inside 
        concept_path = './data/conceptnet-assertions-5.5.5.csv/'
        print("preprocess conceptnet...")
        preprocess_conceptnet(os.path.join(concept_path, 'assertions.csv'))

    # preprocess the label data
    elif len(sys.argv) > 1 and sys.argv[1] == 'selector':
        selector_path = 'data'
        print("preprocess given vocab...")
        preprocess_selector_dataset(os.path.join(selector_path, 'keywordAndRelevancePassage_h.json'))
        # preprocess_selector_dataset(os.path.join(selector_path, 'keywordAndRelevancePassage_Test.json'), is_test_set=True)

        print("load data...")
        # import utils

        # train_data = utils.load_data(os.path.join(selector_path, 'train-terms-processed.json'))
        # dev_data = utils.load_data(os.path.join(selector_path, 'dev-terms-processed.json'))
        # test_data = utils.load_data(os.path.join(selector_path, 'test-terms-processed.json'))
        #
        # print("build other vocab...")
        # utils.build_vocab(
        #     train_data + dev_data + test_data)  # word vocab already exists, run to build pos/ner/rel vocab

    else:
        build_vocabulary()
