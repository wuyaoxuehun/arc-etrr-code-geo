import copy
import json
import math
import os
import sys

import utils


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



import wordfreq
from wordfreq import word_frequency
# 
from utils import is_stopword

word_count = len(list(wordfreq.iter_wordlist('zh')))


def compute_features(d_dicts, q_dict, c_dicts, q_terms):
    # compute features for each d_dict and c_dict
    in_qs, in_cs, lemma_in_qs, lemma_in_cs = [], [], [], []
    p_q_relations, p_c_relations = [], []
    tfs = []

    for d_dict, c_dict in zip(d_dicts, c_dicts):
        # in_q, in_c, lemma_in_q, lemma_in_c, tf
        q_words_set = set([w.lower() for w in q_dict['words']])
        in_q = [int(w.lower() in q_words_set and not is_stopword(w)) for w in d_dict['words']]
        in_qs.append(in_q)

        c_words_set = set([w.lower() for w in c_dict['words']])
        in_c = [int(w.lower() in c_words_set and not is_stopword(w)) for w in d_dict['words']]
        in_cs.append(in_c)

        tf = [0.1 * math.log(word_count * word_frequency(w.lower(), 'zh') + 5) for w in d_dict['words']]
        tf = [float('%.2f' % v) for v in tf]
        tfs.append(tf)
        # d_words = Counter(filter(lambda w: not is_stopword(w) and not is_punc(w), d_dict['words']))

        from conceptnet import concept_net
        p_q_relation = concept_net.p_q_relation(d_dict['words'], q_dict['words'])
        p_q_relations.append(p_q_relation)
        p_c_relation = concept_net.p_q_relation(d_dict['words'], c_dict['words'])
        p_c_relations.append(p_c_relation)

        assert len(in_q) == len(in_c) and len(tf) == len(in_q)
        assert len(tf) == len(p_q_relation) and len(tf) == len(p_c_relation)

    q_es = [True if w in q_terms else False for w in q_dict['words']]

        # update in_c, lemma_in_c and p_c_relation
    return {
        'in_qs': in_qs,
        'in_cs': in_cs,
        'tfs': tfs,
        'p_q_relations': p_q_relations,
        'p_c_relations': p_c_relations,
        'q_es': q_es
    }


def get_example(d_id, q_id, d_dicts, q_dict, c_dicts, label):
    return {
        'id': d_id + '_' + q_id,
        'd_words': [' '.join(d_dict['words']) for d_dict in d_dicts],  # all paras
        'd_pos': [d_dict['pos'] for d_dict in d_dicts],
        'd_ner': [d_dict['ner'] for d_dict in d_dicts],
        'q_words': ' '.join(q_dict['words']),
        'q_pos': q_dict['pos'],
        'c_words': [' '.join(c_dict['words']) for c_dict in c_dicts],  # all choices
        'label': label
    }


def preprocess_geo_dataset(path, topk=5):
    filename = path.split('/')[-1]
    writer = open('./data/' + filename.replace('.json', '') + '_reader_processed.json', 'w', encoding='utf-8')
    ex_cnt = 0
    from tqdm import tqdm
    with open(path, encoding="utf8") as reader:
        for obj in tqdm(json.load(reader)):
            d_dicts = []
            for idx in list('abcd'):
                # print([p for p in obj[str(idx) + '_paragraphs']])
                d_dicts.append(tokenize(' '.join([p['passage'] for p in obj[idx + '_paragraphs'][:topk]]).replace('\n', ' ')))

            d_id = '0'
            choices = [obj['a'], obj['b'], obj['c'], obj['d']]
            ans = obj['answer']
            ans = ord(ans) - ord('A')

            q_dict = tokenize(obj['background'] + ' ' + obj['question'])
            q_cnt = str(obj['questionID'])
            q_id = str(q_cnt)

            q_terms = set(obj['keywords'])
            # enumerate choices
            c_dicts = []
            for c_id, choice in enumerate(choices):
                choice_text = choice
                c_dict = tokenize(choice_text)
                c_dicts.append(c_dict)
            label = ans

            # make an example for each choice

            example = get_example(d_id, q_id, d_dicts, q_dict, c_dicts, label)
            # get all pos tags, only once for all
            # for item in example['q_pos']:
            #     utils.pos_vocab.add(item)

            example.update(compute_features(d_dicts, q_dict, c_dicts, q_terms))

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
    import utils
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
    if len(sys.argv) > 1 and sys.argv[1] == 'conceptnet':
        # will build vocab inside
        concept_path = '../data/conceptnet'
        preprocess_conceptnet(os.path.join(concept_path, 'conceptnet-assertions-5.6.0.csv'))

    elif len(sys.argv) > 1 and sys.argv[1] == 'reader':  # preprocess the data collected by reformed query
        selector_path = './data/'
        import time
        start_time = time.time()
        preprocess_geo_dataset(os.path.join(selector_path, 'keywordAndRelevancePassage.json'), topk=5)
        preprocess_geo_dataset(os.path.join(selector_path, 'keywordAndRelevancePassage_Test.json'), topk=5)
        print('time:'+str(time.time()-start_time))

    else:
        build_vocabulary()
