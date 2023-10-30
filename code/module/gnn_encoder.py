#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/3/15 10:04
# @Author : syd 
# @Site :  
# @File : gnn_encoder.py 
# @Software: PyCharm
import torch
from dgl.nn.pytorch import HeteroGraphConv, GraphConv


class rgcn_layer(torch.nn.Module):

    def __init__(self, in_feats, out_feats, rel_names):
        """
        RGCN layer
        support mini-batch training
        :param in_feats:
        :param out_feats:
        :param rel_names: hg.etypes
        """
        super(rgcn_layer, self).__init__()
        self.conv1 = HeteroGraphConv(
            {rel: GraphConv(in_feats, out_feats, norm='right', weight=True, bias=True) for rel in rel_names},
            aggregate='sum')

    def forward(self, g, feature):
        assert isinstance(feature, (dict))
        h = self.conv1(g, feature)
        h = {k: torch.nn.functional.relu(v) for k, v in h.items()}
        return h


class GNN_encoder(torch.nn.Module):
    """
    the GNN_encoder branch of GTC model
    """

    def __init__(self, in_feats, hid_feats, out_feats, rel_names, layer_nums=2, category=None):
        """

        :param in_feats:
        :param hid_feats:
        :param out_feats:
        :param rel_names:
        :param layer_nums:
        :param category:
        """
        super(GNN_encoder, self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.rel_names = rel_names
        self.layer_nums = layer_nums
        self.category = category
        self.gcn_layer_list = torch.nn.ModuleList()
        for i in range(layer_nums):
            if i == 0:  # the first layer
                if layer_nums == 1:  # only one layer
                    self.gcn_layer_list.append(
                        rgcn_layer(in_feats=self.in_feats, out_feats=self.out_feats,
                                   rel_names=self.rel_names))
                else:
                    self.gcn_layer_list.append(
                        rgcn_layer(in_feats=self.in_feats, out_feats=self.hid_feats,
                                   rel_names=self.rel_names))

            elif i == layer_nums - 1:  # the last layer
                self.gcn_layer_list.append(
                    rgcn_layer(in_feats=self.hid_feats, out_feats=self.out_feats,
                               rel_names=self.rel_names))
            else:
                self.gcn_layer_list.append(
                    rgcn_layer(in_feats=self.hid_feats, out_feats=self.hid_feats,
                               rel_names=self.rel_names))

    def forward(self, g, feat, mini_batch_flag=False):
        """
        the data flow~
        :param mini_batch_flag:
        :param g:
        :param feat:
        :return:
        """
        h = feat
        for layer_index in range(self.layer_nums):
            #  mini-batch
            if mini_batch_flag:
                h = self.gcn_layer_list[layer_index](g=g[layer_index], feature=h)
            else:
                h = self.gcn_layer_list[layer_index](g=g, feature=h)
        if self.category is not None:
            out = h[self.category]

        return out
