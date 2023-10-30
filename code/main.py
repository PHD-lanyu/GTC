#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/3/21 15:20
# @Site :
# @File : main.py
# @Software: PyCharm
import os
import sys
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(BASE_DIR)
# sys.path.append(BASE_DIR)
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import numpy
import numpy as np
import torch
from dgl.dataloading import DataLoader
from dgl.dataloading import NeighborSampler
from module import GTC
import datetime
import random
from self_tools.data_tools import load_data, get_batch_pos
from self_tools.evaluate import evaluate_for_test, evaluate_for_train
from self_tools.params import set_params

args = set_params()
if torch.cuda.is_available() and args.device > -1:
    device = torch.device("cuda:0")
    torch.cuda.set_device(args.device)
else:
    device = torch.device("cpu")

## name of intermediate document ##

own_str = args.dataset
exp_num = 10


def make(config, dgl_graph, feats_dim_list, P, h_dict, category, all_node_idx,
         num_classes, mini_batch_flag=True):
    """
    the fuction of building the model, train_loader and optimizer
    :param config:
    :param dgl_graph:
    :param feats_dim_list:
    :param P:
    :param meta_path_adj:
    :param h_dict:
    :param category:
    :param all_node_idx:
    :param num_classes:
    :param mini_batch_flag:
    :return: model，train_loader,optimizer
    """
    print("seed ", config.seed)
    print("Dataset: ", config.dataset)
    print("The number of gnn_branch_num: ", config.gnn_branch_layer_num)
    # build the GTC model
    model = GTC(config.hidden_dim, feats_dim_list, config.feat_drop, P, config.tau, config.lam,
                t_hops=config.t_hops, t_n_class=num_classes, t_input_dim=h_dict[category].shape[1],
                t_pe_dim=config.t_pe_dim, t_n_layers=config.t_n_layers, t_num_heads=config.t_n_heads,
                t_dropout_rate=config.t_dropout,
                t_attention_dropout_rate=config.t_attention_dropout, rel_names=dgl_graph.etypes, category=category,
                gnn_branch_layer_num=config.gnn_branch_layer_num)
    # build the optimizer for GTC
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2_coef)

    # NeighborSampler and corresponding graph DataLoader for mini_batch training~
    # for more details for NeighborSampler and DataLoader, please see https://docs.dgl.ai/guide/minibatch.html#guide-minibatch
    fanouts = [20]  # first hop sample 20 neighbors for every node
    for i in range(1, config.gnn_branch_layer_num):
        fanouts.append(10)  # 2-gnn_branch_layer_num hop sample 10 neighbors for every node
    sampler = NeighborSampler(fanouts=fanouts)
    all_idx_dict = {category: all_node_idx}

    train_dataloader_4GTC = DataLoader(graph=dgl_graph, indices=all_idx_dict, graph_sampler=sampler,
                                       batch_size=config.batch_size,
                                       shuffle=True)

    return model, train_dataloader_4GTC, optimizer


def train_flow(model, train_loader, optimizer, config, category, pos, own_str, exp=0):
    cnt_wait = 0
    best = 1e9
    best_t = 0
    print('-' * 60)
    print('train_flow for exp-{}'.format(exp))
    starttime = datetime.datetime.now()
    for epoch in range(config.nb_epochs):
        model.train()
        loss_epoch = 0
        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            blocks = [block.to(config.device) for block in blocks]
            # for GNN_branch batch data
            if 'h' in blocks[0].srcdata:
                input_fea4GNN = blocks[0].srcdata['h']
            elif 'feature' in blocks[0].srcdata:
                input_fea4GNN = blocks[0].srcdata['feature']
            else:
                print('please specify the feature key!')
                return
            if not isinstance(input_fea4GNN, dict):
                input_fea4GNN = {category: input_fea4GNN}
            # deal with pos for mini-batch
            pos_batch = get_batch_pos(pos=pos, batch_node_id_x=output_nodes[category].numpy()).to(config.device)
            # [num_meta-paths,num_nodes,num_hops,feature_dim}
            multi_hop_features = blocks[-1].dstnodes[category].data['multi_hop_feature'].permute(1, 0, 2, 3)

            loss = model(g=blocks, feats=input_fea4GNN, multi_hop_features=multi_hop_features, pos=pos_batch,
                         mini_batch_flag=True)
            loss_epoch = loss_epoch + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("exp={}; epoch: {};batch-{}; loss {}".format(exp, epoch, batch_id, loss.data.cpu()))

        print(" epoch: {}; epoch_loss {}".format(epoch, loss_epoch.data.cpu()))
        if loss_epoch < best:
            print('best loss: {}->{}'.format(best, loss_epoch))
            best = loss_epoch
            best_t = epoch
            cnt_wait = 0
            # save better checkpoint~
            torch.save(model.state_dict(), '../data/GTC_' + own_str + '.pkl')
        else:
            cnt_wait += 1
            print('lost not improved~ {}'.format(cnt_wait))
        if cnt_wait >= config.patience:
            print('Early stopping at {} epoch!'.format(epoch))
            break
    print('best epoch is {} !'.format(best_t))
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print('Total train time {} s'.format(time))
    print('-' * 40)
    return best_t


def test(model, config, train_idx_list, val_idx_list, test_idx_list, labels, num_classes, fea_evalue, ma_dic_list,
         mi_dic_list, auc_dic_list):
    starttime = datetime.datetime.now()
    model.eval()
    emb = model.get_embeds(multi_hop_features=fea_evalue.permute(1, 0, 2, 3))
    for i in range(len(train_idx_list)):  # for different data splits for testing~
        ma, mi, auc = evaluate_for_train(config.hidden_dim, train_idx_list[i], val_idx_list[i], test_idx_list[i],
                                         labels, num_classes, config.device, config.dataset, config.eva_lr,
                                         config.eva_wd, batch_size=500, patience=config.patience, emb=emb)
        # record the result of this exp
        ma_dic_list['ma_{}'.format(config.ratio[i])].append(ma)
        mi_dic_list['mi_{}'.format(config.ratio[i])].append(mi)
        auc_dic_list['auc_{}'.format(config.ratio[i])].append(auc)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total evaluate time: ", time, "s")
    print('-' * 40)


def model_train(args):
    # record the result of each exp
    ma_dic_list = dict.fromkeys(['ma_20', 'ma_40', 'ma_60'])
    for key in ma_dic_list.keys():
        ma_dic_list[key] = []
    mi_dic_list = dict.fromkeys(['mi_20', 'mi_40', 'mi_60'])
    for key in mi_dic_list.keys():
        mi_dic_list[key] = []
    auc_dic_list = dict.fromkeys(['auc_20', 'auc_40', 'auc_60'])
    for key in auc_dic_list.keys():
        auc_dic_list[key] = []
    for exp in range(exp_num):  # every exp
        print('-' * 60)
        print('exp:{}'.format(exp))
        print('-' * 60)
        starttime = datetime.datetime.now()
        if torch.cuda.is_available() and args.device > -1:
            device = torch.device("cuda:0")
            torch.cuda.set_device(args.device)
        else:
            device = torch.device("cpu")

        # name of intermediate document
        own_str = args.dataset + '_' + str(exp)

        # random seed
        seed = args.seed
        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # load data~
        dgl_graph, category, all_node_idx, train_idx_list, val_idx_list, test_idx_list, \
        h_dict, labels, P, num_classes, pos = load_data(
            data_name=args.dataset, data_dir='../data/', t_hops=args.t_hops,
            cache_sub_dir='cache-opensource')

        feats_dim_list = [h_dict[key].shape[-1] for key in h_dict.keys()]

        # build the model, train_loader and optimizer
        model, train_loader, optimizer = make(args, dgl_graph, feats_dim_list, P, h_dict,
                                              category, all_node_idx, dgl_graph.etypes, num_classes)
        print(model)

        if torch.cuda.is_available() and args.device > -1:
            print('Using CUDA~')
            model.to(device)
            labels = labels.cuda()
            for index in range(len(train_idx_list)):
                train_idx_list[index] = train_idx_list[index].long().cuda()
                val_idx_list[index] = val_idx_list[index].long().cuda()
                test_idx_list[index] = test_idx_list[index].long().cuda()

        # train the model~
        best_t = train_flow(model, train_loader, optimizer, args, category, pos, own_str, exp=exp)
        # test the model~
        print('-' * 40)
        print('test paradigm~')
        print('Loading {}th epoch'.format(best_t))
        # load checkpoint
        model.load_state_dict(torch.load('../data/GTC_' + own_str + '.pkl'))
        fea_evalue = dgl_graph.nodes[category].data['multi_hop_feature'].to(device)
        # test flow
        test(model, args, train_idx_list, val_idx_list, test_idx_list, labels, num_classes, fea_evalue, ma_dic_list,
             mi_dic_list, auc_dic_list)

        endtime = datetime.datetime.now()
        time = (endtime - starttime).seconds
        print("Total time: ", time, "s")

    # print the result
    for key in ma_dic_list.keys():
        lst = ma_dic_list[key]
        print('{}_mean:{},{}_var:{}'.format(key, np.mean(lst), key, np.std(lst)))
        # print('{}:{}'.format(key, lst))

    for key in mi_dic_list.keys():
        lst = mi_dic_list[key]
        print('{}_mean:{},{}_var:{}'.format(key, np.mean(lst), key, np.std(lst)))
        # print('{}:{}'.format(key, lst))

    for key in auc_dic_list.keys():
        lst = auc_dic_list[key]
        print('{}_mean:{},{}_var:{}'.format(key, np.mean(lst), key, np.std(lst)))
        # print('{}:{}'.format(key, lst))


def test_pre_trained_model(args):
    model = torch.load('../data/{}_model.pkl'.format(args.dataset))
    ## random seed ##
    seed = model.seed
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # load data
    dgl_graph, category, all_node_idx, train_idx_list, val_idx_list, test_idx_list, \
    h_dict, labels, P, num_classes, pos = load_data(
        data_name=args.dataset, data_dir='../data/', t_hops=model.t_hops,
        cache_sub_dir='cache-opensource')
    if torch.cuda.is_available() and args.device > -1:
        print('Using CUDA')
        model.to(device)
        labels = labels.cuda()
        for index in range(len(train_idx_list)):
            train_idx_list[index] = train_idx_list[index].long().cuda()
            val_idx_list[index] = val_idx_list[index].long().cuda()
            test_idx_list[index] = test_idx_list[index].long().cuda()

    starttime = datetime.datetime.now()
    model.eval()
    fea_evalue = dgl_graph.nodes[category].data['multi_hop_feature'].to(device)
    for i in range(len(train_idx_list)):
        evaluate_for_test(model.hidden_dim, train_idx_list[i], val_idx_list[i], test_idx_list[i],
                          labels,
                          num_classes, device,
                          args.dataset,
                          args.eva_lr, args.eva_wd, model=model, fea_evalue=fea_evalue,
                          patience=args.patience, batch_size=500)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")


if __name__ == '__main__':
    # if args.load_from_pretrained:  # test the pretrained model
    #     test_pre_trained_model(args)
    # else:  # train new model
    model_train(args)
