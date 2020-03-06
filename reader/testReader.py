def test_json():
    writer = open('./data/keywordAndRelevancePassage-reader-clean-processed.json', 'w', encoding='utf8')
    with open('./data/keywordAndRelevancePassage-reader-processed.json', encoding='utf8') as file:
        import json
        for line in file.readlines():
            data = json.loads(line)
            writer.write(json.dumps(data, ensure_ascii=False))
            writer.write('\n')

    file.close()
    writer.close()


def test_num():
    file = open('./data/keywordAndRelevancePassage.json', encoding='utf8')
    import json
    print(len(json.load(file)))


def test_answer_distributed():
    import json
    distributed = {0: 0, 1: 0, 2: 0, 3: 0}
    import jsonlines
    with jsonlines.open('./data/keywordAndRelevancePassage_reader_processed.json') as reader:
        for example in reader:
            distributed[example['label']] += 1

    print(distributed)


def test_ori_answer_distributed():
    import json
    files = ['./data/keywordAndRelevancePassage.json', './data/keywordAndRelevancePassage_Test.json']
    for file in files:
        distributed = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        with open(file, encoding='utf8') as file:
            for example in json.load(file):
                distributed[example['answer']] += 1

        print(distributed)


def test_new_files():
    import os
    path = './reader/checkpoint'
    model_path_list = os.listdir(path)
    model_path_list.sort(key=lambda fn: os.path.getmtime(path + '\\' + fn))
    print(model_path_list)

test_new_files()
# test_answer_distributed()
# test_ori_answer_distributed()
# test_json()
# test_num()
