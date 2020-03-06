### arc-etrr-code
This repo follows the following hierarchy:
```
arc-etrr-code
|---retriever
  |---selector
  |---arc-solver
|---reader
|---data

```

### selector

### arc-solver

### reader
+ 预处理json文件，进行分词、计算pos，ner，tf以及conceptnet导出的relation等：
    - 运行在根目录下运行：python reader/preprocess.py reader, 生成keywordAndRelevancePassage_reader_preprocessed.json以及keywordAndRelevancePassage_Test_reader_processed.json文件，分别用于训练（训练：验证8：2）和测试
    ```
    #在reader/preprocess.py最后用于修改要处理的文件
    elif len(sys.argv) > 1 and sys.argv[1] == 'reader': 
        selector_path = './data/'
        import time
        start_time = time.time()
        preprocess_geo_dataset(os.path.join(selector_path, 'keywordAndRelevancePassage.json'), topk=5)
        preprocess_geo_dataset(os.path.join(selector_path, 'keywordAndRelevancePassage_Test.json'), topk=5)
        print('time:'+str(time.time()-start_time)) 
    ```
    其中测试集处理的时间根据采取的前k段落变化，修改topk决定使用多少段落。
    
+ 根据产生的预处理文件进行训练
    - 在根目录下运行：python reader/main.py
    - 根据配置文件reader/config.py设置epoch、batch_size以及数据随机打乱shuffle_seed等
    - 最后模型将会出现在reader/checkpoint中以训练时间后缀为文件名的文件中
+ 在测试集上验证
    - 在根目录上运行：python reader/play_with_trained_model_reform.py --test_recent_k=1
    - test_recent_k 缺省1

