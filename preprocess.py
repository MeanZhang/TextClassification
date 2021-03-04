import logging
import os
import pickle
import jieba
from torchtext import data

# 取消jieba日志输出
jieba.setLogLevel(logging.INFO)


def get_classes(path):
    '''获取分类列表'''
    classes = []
    with open(os.path.join(path, 'classes.csv'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            classes.append(line.strip())
    return classes


def get_field(path):
    '''创建Field'''
    with open(os.path.join(path, 'stop_words.txt'), 'r', encoding='utf-8') as f:
        stop_words = [word.strip('\n') for word in f.readlines()]
    TEXT = data.Field(tokenize=jieba.lcut,
                      lower=True, stop_words=stop_words)
    LABEL = data.Field(sequential=False, use_vocab=False)
    return TEXT, LABEL


def preprocess(path, text, label, args):
    '''预处理'''

    # 创建Dataset
    train, val, test = data.TabularDataset.splits(
        path=path,
        train='train.tsv',
        validation='val.tsv',
        test='test.tsv',
        format='tsv',
        fields=[('label', label), ('text', text)]
    )

    text.build_vocab(train, val)  # 创建vocab
    # 保存vocab
    with open(os.path.join(path, 'vocab.pkl'), 'wb') as f:
        pickle.dump(text.vocab, f)

    # 创建Iterator
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test),
        batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        device=args.device,
        sort_key=lambda x: len(x.text)
    )
    return train_iter, val_iter, test_iter
