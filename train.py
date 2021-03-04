import os
import numpy
import torch
import torch.nn.functional as F
from sklearn import metrics


def train(train_iter, dev_iter, model, args):
    '''训练'''

    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 优化器Adam
    steps = 0  # 训练步数
    best_dev_loss = float('inf')  # 验证集最优loss
    last_improve = 0  # 最优loss步数
    stop = False  # 停止标志
    for epoch in range(args.epochs):
        for batch in train_iter:
            model.train()
            text, label = batch.text.data.t(), batch.label.data.sub(1)
            if args.cuda:
                text, label = text.cuda(), label.cuda()
            optimizer.zero_grad()
            out = model(text)
            loss = F.cross_entropy(out, label)
            loss.backward()
            optimizer.step()
            steps += 1
            # 每多少步显示信息
            if steps % args.show_steps == 0:
                true = label.cpu()
                predict = torch.max(out, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                print('\rBatch: {:<4}  Train Loss: {:.6f}  Train Acc: {:.4f}  Epoch:{}'.format(
                    steps, loss.item(), train_acc, epoch), end='')
            # 每100步验证一次
            if steps % 100 == 0:
                dev_loss, dev_acc = evaluate(dev_iter, model, args)
                print('\nEvaluation     Val Loss: {:.6f}    Val Acc: {:.4f}\n'.format(
                    dev_loss, dev_acc))
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    last_improve = steps
                    print('saving best model...\n')
                    torch.save(model.state_dict(),
                               os.path.join(args.data_path, 'saved_model.pt'))
            # 提前停止
            if steps - last_improve >= args.stop_improvements:
                print('\nno optimization in {} steps, stop\nsaved model loss: {:.4f}\n'.format(
                    args.stop_improvements, best_dev_loss))
                stop = True
                break
        if stop:
            break


def evaluate(data_iter, model, args):
    '''评估'''

    model.eval()
    loss = 0.0  # 总loss
    predicts = numpy.array([], dtype=int)  # 预测值
    trues = numpy.array([], dtype=int)  # 正确值
    with torch.no_grad():
        for batch in data_iter:
            text, label = batch.text.data.t(), batch.label.data.sub(1)
            if args.cuda:
                text, label = text.cuda(), label.cuda()
            out = model(text)
            loss += F.cross_entropy(out, label)
            true = label.cpu().numpy()
            predict = torch.max(out, 1)[1].cpu().numpy()
            trues = numpy.append(trues, true)
            predicts = numpy.append(predicts, predict)
        acc = metrics.accuracy_score(trues, predicts)
        loss /= len(data_iter.dataset)  # 平均loss
    return loss, acc


def test(data_iter, model, args):
    '''测试'''

    print('\n********************TEST********************')
    model.eval()
    loss = 0.0
    predicts = numpy.array([], dtype=int)
    trues = numpy.array([], dtype=int)
    with torch.no_grad():
        for batch in data_iter:
            text, label = batch.text.data.t(), batch.label.data.sub(1)
            if args.cuda:
                text, label = text.cuda(), label.cuda()
            out = model(text)
            loss += F.cross_entropy(out, label)
            true = label.cpu().numpy()
            predict = torch.max(out, 1)[1].cpu().numpy()
            trues = numpy.append(trues, true)
            predicts = numpy.append(predicts, predict)
    acc = metrics.accuracy_score(trues, predicts)
    loss /= len(data_iter.dataset)
    report = metrics.classification_report(
        trues, predicts, target_names=args.classes)
    confusion_matrix = metrics.confusion_matrix(trues, predicts)
    print('Loss: {:.6f} Acc: {:.4f}'.format(loss, acc))
    print('[CLASSIFICATION REPORT]')
    print(report)
    print('[CONFUSION MATRIX]')
    print(confusion_matrix)


def predict(text, model, text_field):
    '''预测'''

    model.eval()
    text = text_field.preprocess(text)  # 文本预处理
    text = [[text_field.vocab.stoi[x] for x in text]]  # 转为向量
    text = torch.tensor(text)  # 转为tensor
    result = model(text)
    result = torch.max(result, 1)[1]
    return result
