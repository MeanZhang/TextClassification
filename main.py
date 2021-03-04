import argparse
import os
import pickle
import torch
import preprocess
import module
import train

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='data', help="数据保存路径（默认'data'）")
parser.add_argument('--device_num', type=int, default=0, help='使用设备编号（默认0）')
parser.add_argument('--filter_sizes', type=str,
                    default='3,4,5', help='filter大小，默认(3,4,5)')
parser.add_argument('--lr', type=float, default=0.001, help='学习率（默认0.001）')
parser.add_argument('--batch_size', type=int,
                    default=128, help='batch大小（默认128）')
parser.add_argument('--epochs', type=int, default=20, help='epoch数（默认20）')
parser.add_argument('--filter_num', type=int,
                    default=128, help='fliter数量（默认128）')
parser.add_argument('--embedding_dim', type=int,
                    default=128, help='embedding维度（默认128）')
parser.add_argument('--dropout', type=float,
                    default=0.5, help='dropout（默认0.5）')
parser.add_argument('--show_steps', type=int, default=1,
                    help='每多少batch显示信息（默认1）')
parser.add_argument('--stop_improvements', type=int, default=300,
                    help='多少batch内loss无提高时停止（默认300）')
parser.add_argument('--predict', action='store_true', help='预测模式')

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

TEXT, LABEL = preprocess.get_field(args.data_path)  # 建立文本和标签Field

# 是否进入预测模式
if args.predict:
    # 预测模式
    if os.path.exists(os.path.join(args.data_path, 'vocab.pkl')) and os.path.exists(os.path.join(args.data_path, 'saved_model.pt')):
        # 加载词向量和模型
        s = input('plase enter a sentence in classes '+str(args.classes)+'\n')
        with open(os.path.join(args.data_path, 'vocab.pkl'), 'rb') as f:
            TEXT.vocab = pickle.load(f)
        args.vocab_size = len(TEXT.vocab)
        text_cnn = module.TextCNN(args)
        # 预测
        text_cnn.load_state_dict(torch.load(
            os.path.join(args.data_path, 'saved_model.pt')))
        result = train.predict(s, text_cnn, TEXT)
        print('[CLASS]'+args.classes[result])
    else:
        # 没有词向量，提示训练
        print("please TRAIN first")
else:
    # 训练模式
    print("loading data...")
    train_iter, val_iter, test_iter = preprocess.preprocess(
        args.data_path, TEXT, LABEL, args)
    args.vocab_size = len(TEXT.vocab)
    text_cnn = module.TextCNN(args)
    print('training...')
    train.train(train_iter, val_iter, text_cnn, args)  # 训练
    train.test(test_iter, text_cnn, args)  # 测试
