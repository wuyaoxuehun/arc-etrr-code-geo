def test_preprocess():
    import jieba
    import jieba.posseg as pseg
    jieba.set_dictionary('./data/vocabulary/v1.txt')
    words = pseg.cut('中华民族是个伟大的民族。熊爱明是个中国人，他非常热爱自己的祖国。中华人民共和国在1949年10月成立')
    for word, flag in words:
        print(word, flag)
    pass


def test_metric():
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    a = [0, 0, 1, 1, 2, 3]
    b = [0, 0, 1, 2, 2, 1]
    print(precision_recall_fscore_support(a, b,labels=[0,1,2]))



def hanlp_test():
    import hanlp
    tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
    tokens = tokenizer('中国的全称')
    tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)


def test_shuffle():
    import random

    for i in range(10):
        random.seed(1234)
        a = [1,2,3,4,5]
        print(random.shuffle(a))
        print(a)


test_shuffle()

# test_preprocess()

# 让json文件容易读
def test_test():
    import unicodedata
    a = ''
    print(unicodedata.normalize('NFD', a))
    import json
    data = json.load(open(r'data/keywordAndRelevancePassage.json', encoding='utf8'))
    newData = []
    for q in data:
        newData.append({'questionID': q['questionID'],
                        'background': q['background'],
                        'question': q['question'],
                        'a': q['a'],
                        'b': q['b'],
                        'c': q['c'],
                        'd': q['d'],
                        'keywords': q['keywords']})

    json.dump(newData, open(r'data/keywordAndRelevancePassage_h.json', 'w', encoding='utf8'), indent=4,
              ensure_ascii=False)


def test_wordfreq():
    from wordfreq import word_frequency
    for i in range(100):
        print(word_frequency('绿色食品', 'zh'))


def test_conceptnet():
    from conceptnet import concept_net
    q_c_relation = concept_net.p_q_relation('国家', '中国')
    print(q_c_relation)
