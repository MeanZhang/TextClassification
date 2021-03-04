import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        self.embed = nn.Embedding(
            args.vocab_size, args.embedding_dim)  # 随机词向量

        self.convs = nn.ModuleList([nn.Conv2d(
            1, args.filter_num, (i, args.embedding_dim)) for i in args.filter_sizes])  # 卷积层
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(args.filter_sizes) *
                            args.filter_num, args.class_num)  # 全连接层

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embed(x)

        out = out.unsqueeze(1)

        out = torch.cat([self.conv_and_pool(out, conv)
                         for conv in self.convs], 1)

        out = self.dropout(out)
        out = self.fc(out)
        return out
