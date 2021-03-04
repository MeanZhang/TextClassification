import argparse
import logging
import time
import train
import preprocess
import torch
from module import TextCNN
import os
import pickle
import jieba
from torchtext import data

# 取消jieba日志输出
jieba.setLogLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='data', help="数据保存路径（默认'data'）")
parser.add_argument('--device_num', type=int, default=0, help='使用设备编号（默认0）')
parser.add_argument('--filter_sizes', type=str,
                    default='3,4,5', help='filter大小，默认(3,4,5)')
parser.add_argument('--lr', type=float, default=0.001, help='学习率（默认0.001）')
parser.add_argument('--batch_size', type=int,
                    default=128, help='batch大小（默认128）')
parser.add_argument('--filter_num', type=int,
                    default=128, help='fliter数量（默认128）')
parser.add_argument('--embedding_dim', type=int,
                    default=128, help='embedding维度（默认128）')
parser.add_argument('--dropout', type=float,
                    default=0.5, help='dropout（默认0.5）')

args = parser.parse_args()

args.filter_sizes = list(map(int, args.filter_sizes.split(',')))

# 设置使用的设备
if args.device_num != -1 and torch.cuda.is_available():
    args.device = torch.device('cuda:'+str(args.device_num))
    args.cuda = True
    torch.cuda.set_device(torch.device(args.device))
else:
    args.device = torch.device('cpu')
    args.cuda = False

args.classes = preprocess.get_classes(args.data_path)  # 获取分类列表
args.class_num = len(args.classes)

def test(data_file, args):
    '''选择任意数据集进行测试'''
    with open(os.path.join(args.data_path, 'stop_words.txt'), 'r', encoding='utf-8') as f:
        stop_words = [word.strip('\n') for word in f.readlines()]
    TEXT, LABEL = preprocess.get_field(args.data_path)
    with open(os.path.join(args.data_path, 'vocab.pkl'), 'rb') as f:
        TEXT.vocab = pickle.load(f)
    args.vocab_size = len(TEXT.vocab)
    text_cnn = TextCNN(args)
    text_cnn.load_state_dict(torch.load(
        os.path.join(args.data_path, 'saved_model.pt')))
    if args.cuda:
        text_cnn.cuda()
    test_dataset = data.TabularDataset(
        path=os.path.join(args.data_path, data_file),
        format='tsv',
        fields=[('label', LABEL), ('text', TEXT)]
    )
    test_iter = data.Iterator(test_dataset, args.batch_size)
    start=time.time()
    train.test(test_iter,text_cnn,args)
    end=time.time()
    print('\ntest took {:.2f}s'.format(end-start))

if __name__ == "__main__":
    test('test.tsv',args)