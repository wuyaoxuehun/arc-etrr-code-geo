from config import args
from utils import load_data, build_vocab, gen_submission, gen_final_submission, eval_based_on_outputs
from model import Model
import os

if __name__ == '__main__':
    pretrained_dir = './reader/checkpoint'

    # if not args.pretrained:
    #     print('No pretrained model specified.')
    #     exit(0)
    build_vocab()
    # swap dev and test
    swap = False
    if swap:
        args.test_mode = False if args.test_mode else True

    args.test_mode = True
    if args.test_mode:
        dev_path = './data/keywordAndRelevancePassage_Test_reader_processed.json'
        print("load test data...")
    else:

        dev_path = './data/ARC-Challenge-Dev-question-reform.nn-qa-para.clean-processed.json'
        print("load dev data...")
        
    dev_data = load_data(dev_path)

    # ensemble models
    model_path_list = os.listdir(pretrained_dir)
    model_path_list.sort(key=lambda fn: os.path.getmtime(os.path.join(pretrained_dir, fn)))
    recent_k = args.test_recent_k
    # recent_k_models = os.path.join(args.pretrained, model_path_list[-recent_k])

    for k in range(recent_k):

        args.pretrained = os.path.join(pretrained_dir, model_path_list[-(k+1)])
        model = Model(args)
        # evaluate on development dataset
        dev_acc = model.evaluate(dev_data, debug=True)
        #dev_acc = model.evaluate(dev_data)
        print('test accuracy: %f' % dev_acc)

