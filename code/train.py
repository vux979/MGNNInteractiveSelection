import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MGNN_IS
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score
import pickle
import time
import math
import pandas as pd
from tqdm import tqdm
import logging


def train(args, data_info, show_loss):
    train_loader = data_info['train']
    val_loader = data_info['val']
    test_loader = data_info['test']
    feature_num = data_info['feature_num']
    train_num, val_num, test_num = data_info['data_num']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print('=' * 30, '\n', device)
    print('=' * 30)
    model = GMCF(args, feature_num, device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        weight_decay=args.l2_weight,
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                           patience=2)  # goal: maximize Dice score
    # 等待两个epoch指标没有改进（变高）学习率就*0.1
    crit = torch.nn.BCELoss()
    logname = 'last_h' + str(args.heads) + 'l' + str(args.layer) + 'dim' + str(args.dim) + 'XSimGCL'
    logging.basicConfig(filename=f'./log/{args.dataset}/{logname}.log', level=logging.DEBUG)
    # print([i.size() for i in filter(lambda p: p.requires_grad, model.parameters())])
    print('start training...')
    for step in range(args.n_epoch):
        # training
        loss_all = 0
        baseloss_all = 0
        edge_all = 0
        model.train()
        for data in tqdm(train_loader, desc='training--{}'.format(step + 1)):
            data = data.to(device)

            if args.cl == 1:
                (output, item_graphs_select_cl, item_graphs_select_cl_2, user_graphs_select_cl, user_graphs_select_cl_2) \
                    = model(data, is_training=True)
            else:
                output = model(data, is_training=True)

            label = data.y
            label = label.to(device)
            baseloss = crit(torch.squeeze(output), label)

            if args.cl == 1:
                cl_loss_i = InfoNCE(item_graphs_select_cl, item_graphs_select_cl_2, temperature=args.temp)
                cl_loss_u = InfoNCE(user_graphs_select_cl, user_graphs_select_cl_2, temperature=args.temp)
                loss = baseloss + 0.1 * (cl_loss_i + cl_loss_u)
            else:
                loss = baseloss
            loss_all += data.num_graphs * loss.item()
            baseloss_all += data.num_graphs * baseloss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cur_loss = loss_all / train_num
        cur_baseloss = baseloss_all / train_num

        # evaluation
        eval_name = 'val--' + str(step + 1)
        val_auc, val_logloss, val_ndcg5, val_ndcg10 = evaluate(model, val_loader, device, eval_name)
        # val_auc, val_logloss, val_ndcg5, val_ndcg10 = 0, 0, 0, 0
        scheduler.step(val_auc)
        eval_name = 'test--' + str(step + 1)
        test_auc, test_logloss, test_ndcg5, test_ndcg10 = evaluate(model, test_loader, device, eval_name)

        print(
            'Epoch: {:03d}, Loss: {:.5f}, baseloss: {:.5f}, AUC: {:.5f}/{:.5f}, Logloss: {:.5f}/{:.5f}, NDCG@5: {:.5f}/{:.5f} NDCG@10: {:.5f}/{:.5f}'.
            format(step + 1, cur_loss, cur_baseloss, val_auc, test_auc, val_logloss, test_logloss, val_ndcg5,
                   test_ndcg5, val_ndcg10,
                   test_ndcg10))
        logging.info(
            'Epoch: {:03d}, Loss: {:.5f}, baseloss: {:.5f}, AUC: {:.5f}/{:.5f}, Logloss: {:.5f}/{:.5f}, NDCG@5: {:.5f}/{:.5f} NDCG@10: {:.5f}/{:.5f}'.
            format(step + 1, cur_loss, cur_baseloss, val_auc, test_auc, val_logloss, test_logloss, val_ndcg5,
                   test_ndcg5, val_ndcg10,
                   test_ndcg10))


def evaluate(model, data_loader, device, eval_name):
    model.eval()

    predictions = []
    labels = []
    user_ids = []
    edges_all = [0, 0]
    with torch.no_grad():
        for data in tqdm(data_loader, desc=eval_name):
            _, user_id_index = np.unique(data.batch.detach().cpu().numpy(), return_index=True)
            user_id = data.x.detach().cpu().numpy()[user_id_index]
            user_ids.append(user_id)

            data = data.to(device)
            pred = model(data, is_training=False)
            pred = pred.squeeze().detach().cpu().numpy().astype('float64')
            if pred.size == 1:
                pred = np.expand_dims(pred, axis=0)
            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.concatenate(predictions, 0)
    labels = np.concatenate(labels, 0)
    user_ids = np.concatenate(user_ids, 0)

    ndcg5 = cal_ndcg(predictions, labels, user_ids, 5)
    ndcg10 = cal_ndcg(predictions, labels, user_ids, 10)
    auc = roc_auc_score(labels, predictions)
    logloss = log_loss(labels, predictions)

    return auc, logloss, ndcg5, ndcg10


def cal_ndcg(predicts, labels, user_ids, k):
    d = {'user': np.squeeze(user_ids), 'predict': np.squeeze(predicts), 'label': np.squeeze(labels)}
    df = pd.DataFrame(d)
    user_unique = df.user.unique()

    ndcg = []
    for user_id in user_unique:
        user_srow = df.loc[df['user'] == user_id]
        upred = user_srow['predict'].tolist()
        if len(upred) < 2:
            # print('less than 2', user_id)
            continue
        # supred = [upred] if len(upred)>1 else [upred + [-1]]  # prevent error occured if only one sample for a user
        ulabel = user_srow['label'].tolist()
        # sulabel = [ulabel] if len(ulabel)>1 else [ulabel +[1]]

        ndcg.append(ndcg_score([ulabel], [upred], k=k))

    return np.mean(np.array(ndcg))


def InfoNCE(view1, view2, temperature, b_cos=True):
    # view1: torch.Size([1919, 64]) view2:torch.Size([1919, 64])
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)  # torch.Size([1919])
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))  # torch.Size([1919, 1919])
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)  # torch.Size([1919, 64])
    cl_loss = -torch.log(pos_score / ttl_score + 10e-6)
    return torch.mean(cl_loss)
