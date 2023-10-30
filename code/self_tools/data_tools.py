# -- coding: utf-8 --
import os
import warnings

import dgl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from .adj_utils import re_featuresv2

warnings.filterwarnings('ignore')


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def split(node_idx, labels, train_ratio=0.2):
    '''
    split the train and valid from the node_idx
    :param stratify:
    :param node_idx:
    :param labels:
    :param test_ratio:
    :param train_ratio: default=0.2
    :param val_ratio:
    :return:
    '''
    # randint = torch.randperm(len(node_idx))
    # r = int((train_ratio + val_ratio) / val_ratio)
    # val_idx = node_idx[randint[:len(node_idx) // r]]
    # valid_idx = train_idx[random_int[:len(train_idx) // 5]]
    # train_idx = node_idx[randint[len(node_idx) // r:]]

    train_idx, val_idx, train_y, val_y = train_test_split(node_idx, labels, test_size=1 - train_ratio,
                                                          stratify=labels)
    # train_idx, val_idx, train_y, val_y = train_test_split(node_idx, labels, test_size=1 - train_ratio)
    val_idx, test_idx, val_y, test_y = train_test_split(val_idx, val_y, test_size=0.5,
                                                        stratify=val_y)
    # val_idx, test_idx, val_y, test_y = train_test_split(val_idx, val_y, test_size=0.5)

    return train_idx, val_idx, test_idx


def split_v2(all_node_idx, labels, train_num_per_class=20, valid_num=1000, test_num=1000):
    # randint = torch.randperm(len(node_idx))
    # r = int((train_ratio + val_ratio) / val_ratio)
    # val_idx = node_idx[randint[:len(node_idx) // r]]
    # valid_idx = train_idx[random_int[:len(train_idx) // 5]]
    # train_idx = node_idx[randint[len(node_idx) // r:]]
    # label_value=
    # if labels isinstance(to)
    labels_np = labels.numpy().tolist()
    label_value = set(labels_np)
    train_idx = None
    # valid_idx =
    for lv in label_value:
        index = torch.nonzero(labels == lv, as_tuple=False).squeeze()
        randint = torch.randperm(len(index))
        # r = int((train_ratio + val_ratio) / val_ratio)
        _idx = index[randint[:train_num_per_class]].numpy()
        if train_idx is None:
            train_idx = _idx
        else:
            train_idx = np.hstack((train_idx, _idx))

    # all_node_idx_list = all_node_idx.numpy().tolist()
    # s = [1, 2, 3, 4, 5, 6]
    # s1 = s.pop([1, 2, 3])
    rest_nodes_idx = np.delete(all_node_idx.numpy(), train_idx)

    randint = torch.randperm(len(rest_nodes_idx))
    # r = int((train_ratio + val_ratio) / val_ratio)
    val_idx = rest_nodes_idx[randint[:valid_num]]
    test_idx = rest_nodes_idx[randint[valid_num:valid_num + test_num]]
    # valid_idx = train_idx[random_int[:len(train_idx) // 5]]
    # train_idx = node_idx[randint[len(node_idx) // r:]]

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def load_data(data_name, t_hops=7, data_dir='../data/', start_hop=0, cache_sub_dir='cache'):
    cache_dir = data_dir + cache_sub_dir + '/' + data_name + '/'
    if data_name == 'acm':  # the acm dataset
        het_graph, _ = dgl.load_graphs(data_dir + data_name + '/graph.bin')
        het_graph = het_graph[0]
        ndata = het_graph.ndata
        h_dict = ndata['h']
        category = 'paper'
        meta_path_num = 2
        all_node_idx = torch.arange(0, het_graph.num_nodes(category), dtype=het_graph.idtype)
        ratio = [20, 40, 60]
        path = data_dir + "acm/"
        train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
        test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
        val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
        train_idx_list = [torch.LongTensor(i) for i in train]
        val_idx_list = [torch.LongTensor(i) for i in val]
        test_idx_list = [torch.LongTensor(i) for i in test]

        if os.path.exists(cache_dir):
            multi_hop_features = torch.load(
                cache_dir + 'multi_hop_features_{}_{}.pt'.format(start_hop, t_hops))
            het_graph.nodes[category].data['multi_hop_feature'] = multi_hop_features
            labels = torch.load(cache_dir + 'labels.pt')
            het_graph.nodes[category].data['label'] = labels
            num_classes = labels.shape[1]
            pos = torch.load(cache_dir + 'pos.pt')
            print('load cached files finished~')
        else:

            labels = het_graph.nodes[category].data['label'].squeeze().long()
            labels = encode_onehot(labels)
            labels = torch.FloatTensor(labels)
            num_classes = labels.shape[1]
            het_graph.nodes[category].data['label'] = labels

            adj_dict = {e: het_graph.adj(etype=e) for e in het_graph.etypes}
            pap_adj = torch.matmul(adj_dict['paper_author'], adj_dict['author_paper'])
            psp_adj = torch.matmul(adj_dict['paper_subject'], adj_dict['subject_paper'])

            pap = pap_adj.to_dense()
            psp = psp_adj.to_dense()
            pos = ((pap + psp) >= meta_path_num).float().to_sparse()

            meta_path_adj_with_normalize_adj = {
                'pap': sparse_mx_to_torch_sparse_tensor(normalize_adj(pap.numpy())),
                'psp': sparse_mx_to_torch_sparse_tensor(normalize_adj(psp.numpy())),
            }

            print('start save cache files~')
            os.makedirs(cache_dir)
            torch.save(labels, cache_dir + 'labels.pt')
            torch.save(pos, cache_dir + 'pos.pt')
            for max_hop in range(1, 10):
                multi_hop_features_with_process_feature_with_normalize_adj = [
                    re_featuresv2(adj=meta_path_adj_with_normalize_adj[mp],
                                  features=h_dict[category], K=max_hop, start_hops=0)
                    for mp in meta_path_adj_with_normalize_adj]
                multi_hop_features_with_process_feature_with_normalize_adj = torch.stack(
                    multi_hop_features_with_process_feature_with_normalize_adj).permute(1, 0, 2, 3)
                torch.save(multi_hop_features_with_process_feature_with_normalize_adj,
                           cache_dir + 'multi_hop_features_{}_{}.pt'.format(0, max_hop))
            multi_hop_features = torch.load(
                cache_dir + 'multi_hop_features_{}_{}.pt'.format(0, t_hops))

            het_graph.nodes[category].data['multi_hop_feature'] = multi_hop_features
            print('save cache files finished~')

        print('dataset acm is loaded!')
        print(h_dict['paper'].shape)
        return het_graph, category, all_node_idx, train_idx_list, val_idx_list, test_idx_list, h_dict, labels, meta_path_num, num_classes, pos
    elif data_name == 'dblp':  # the dblp dataset
        category = 'author'
        ratio = [20, 40, 60]
        meta_path_num = 3
        path = data_dir + "dblp/"
        train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
        test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
        val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
        train = [torch.LongTensor(i) for i in train]
        val = [torch.LongTensor(i) for i in val]
        test = [torch.LongTensor(i) for i in test]

        if os.path.exists(cache_dir):
            het_graph = dgl.load_graphs(cache_dir + 'het_graph.bin')[0][0]
            all_node_idx = torch.arange(0, het_graph.num_nodes(category), dtype=het_graph.idtype)
            h_dict = {}
            multi_hop_features = torch.load(
                cache_dir + 'multi_hop_features_{}_{}.pt'.format(start_hop, t_hops))
            het_graph.nodes[category].data['multi_hop_feature'] = multi_hop_features
            h_dict['author'] = torch.load(cache_dir + 'h_dict_author.pt').to_dense()

            het_graph.nodes[category].data['feature'] = h_dict[category]
            labels = torch.load(cache_dir + 'labels.pt')
            num_classes = labels.shape[1]
            pos = torch.load(cache_dir + 'pos.pt')
            print('load cache files finished~')
        else:
            # The order of node types: 0 m 1 d 2 a 3 w
            path = data_dir + "dblp/"
            label = np.load(path + "labels.npy").astype('int32')
            feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
            apa = sp.load_npz(path + "apa.npz")
            apcpa = sp.load_npz(path + "apcpa.npz")
            aptpa = sp.load_npz(path + "aptpa.npz")
            apa_src_node = apa.row
            apcpa_src_node = apcpa.row
            aptpa_src_node = aptpa.row
            apa_dst_node = apa.col
            apcpa_dst_node = apcpa.col
            aptpa_dst_node = aptpa.col
            graph_data = {
                ('author', 'net_apa', 'author'): (torch.tensor(apa_src_node), torch.tensor(apa_dst_node)),
                ('author', 'net_apcpa', 'author'): (torch.tensor(apcpa_src_node), torch.tensor(apcpa_dst_node)),
                ('author', 'net_aptpa', 'author'): (torch.tensor(aptpa_src_node), torch.tensor(aptpa_dst_node))
            }
            het_graph = dgl.heterograph(graph_data)
            # feature
            feat_a_with_process_feature = torch.FloatTensor(preprocess_features(feat_a))
            het_graph.nodes[category].data['feature'] = feat_a_with_process_feature
            h_dict_with_process_feature = {
                'author': feat_a_with_process_feature,
            }
            pos = ((sparse_mx_to_torch_sparse_tensor(apa).to_dense()
                    + sparse_mx_to_torch_sparse_tensor(
                        apcpa).to_dense() + sparse_mx_to_torch_sparse_tensor(
                        aptpa).to_dense()) >= meta_path_num).float()

            meta_path_adj_with_normalize_adj = {
                'apa': sparse_mx_to_torch_sparse_tensor(normalize_adj(apa)),
                'apcpa': sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa)),
                'aptpa': sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
            }

            all_node_idx = torch.arange(0, het_graph.num_nodes(category), dtype=het_graph.idtype)

            # labels
            labels = encode_onehot(label)
            labels = torch.FloatTensor(labels)
            num_classes = labels.shape[1]
            het_graph.nodes[category].data['label'] = labels

            print('start save cache files~')
            os.makedirs(cache_dir)
            dgl.save_graphs(cache_dir + 'het_graph.bin', [het_graph])

            for key in h_dict_with_process_feature:
                torch.save(h_dict_with_process_feature[key].to_sparse(),
                           cache_dir + 'h_dict_{}.pt'.format(key))
            h_dict = h_dict_with_process_feature
            torch.save(labels, cache_dir + 'labels.pt')
            torch.save(pos.to_sparse(), cache_dir + 'pos.pt')
            for max_hop in range(1, 10):
                multi_hop_features_with_process_feature_with_normalize_adj = [
                    re_featuresv2(adj=meta_path_adj_with_normalize_adj[mp],
                                  features=h_dict[category], K=max_hop, start_hops=0)
                    for mp in meta_path_adj_with_normalize_adj]
                multi_hop_features_with_process_feature_with_normalize_adj = torch.stack(
                    multi_hop_features_with_process_feature_with_normalize_adj).permute(1, 0, 2, 3)
                torch.save(multi_hop_features_with_process_feature_with_normalize_adj,
                           cache_dir + 'multi_hop_features_{}_{}.pt'.format(0, max_hop))
            multi_hop_features = torch.load(
                cache_dir + 'multi_hop_features_{}_{}.pt'.format(0, t_hops))

            het_graph.nodes[category].data['multi_hop_feature'] = multi_hop_features
            print('save cache files finished~')

        print('dataset dblp is loaded!')
        return het_graph, category, all_node_idx, train, val, test, h_dict, labels, meta_path_num, num_classes, pos

    elif data_name == 'freebase':
        category = 'movie'
        meta_path_num = 3
        path = data_dir + "freebase/"
        ratio = [20, 40, 60]
        train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
        test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
        val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
        train_idx_list = [torch.LongTensor(i) for i in train]
        val_idx_list = [torch.LongTensor(i) for i in val]
        test_idx_list = [torch.LongTensor(i) for i in test]
        pos = sp.load_npz(path + "pos.npz")
        pos = sparse_mx_to_torch_sparse_tensor(pos)
        if os.path.exists(cache_dir):
            het_graph = dgl.load_graphs(cache_dir + 'het_graph.bin')[0][0]
            all_node_idx = torch.arange(0, het_graph.num_nodes(category), dtype=het_graph.idtype)
            h_dict = {}
            h_dict['movie'] = torch.load(cache_dir + 'h_dict_movie.pt').to_dense()
            labels = torch.load(cache_dir + 'labels.pt')
            num_classes = labels.shape[1]
            # pos = torch.load(cache_dir + 'pos.pt')
            multi_hop_features = torch.load(
                cache_dir + 'multi_hop_features_{}_{}.pt'.format(0, t_hops))
            het_graph.nodes[category].data['multi_hop_feature'] = multi_hop_features
            print('load cache files finished~')

        else:
            # The order of node types: 0 m 1 d 2 a 3 w
            path = data_dir + "freebase/"
            label = np.load(path + "labels.npy").astype('int32')
            feat_m = sp.eye(3492).tocoo()
            apa = sp.load_npz(path + "mam.npz")
            apvpa = sp.load_npz(path + "mdm.npz")
            mwm = sp.load_npz(path + "mwm.npz")
            # pos = sp.load_npz(path + "pos.npz")
            # pos = torch.from_numpy(pos.toarray()).float()
            apa_src_node = apa.row
            apvpa_src_node = apvpa.row
            mwm_src_node = mwm.row
            apa_dst_node = apa.col
            apvpa_dst_node = apvpa.col
            mwm_dst_node = mwm.col
            graph_data = {
                ('movie', 'net_mam', 'movie'): (torch.tensor(apa_src_node), torch.tensor(apa_dst_node)),
                ('movie', 'net_mdm', 'movie'): (torch.tensor(apvpa_src_node), torch.tensor(apvpa_dst_node)),
                ('movie', 'net_mwm', 'movie'): (torch.tensor(mwm_src_node), torch.tensor(mwm_dst_node))
            }
            het_graph = dgl.heterograph(graph_data)
            # feature
            feat_m = torch.FloatTensor(preprocess_features(feat_m))
            het_graph.nodes['movie'].data['feature'] = feat_m
            h_dict = {
                'movie': feat_m,
            }

            pos = ((sparse_mx_to_torch_sparse_tensor(apa).to_dense()
                    + sparse_mx_to_torch_sparse_tensor(
                        apvpa).to_dense() + sparse_mx_to_torch_sparse_tensor(mwm).to_dense()) >= meta_path_num).float()

            apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
            apvpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apvpa))
            mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))

            meta_path_adj = {'mam': apa,
                             'mdm': apvpa,
                             'mwm': mwm}

            adj_dict = {e: het_graph.adj(etype=e) for e in het_graph.etypes}
            all_node_idx = torch.arange(0, het_graph.num_nodes(category), dtype=het_graph.idtype)
            # labels
            labels = encode_onehot(label)
            labels = torch.FloatTensor(labels)
            num_classes = labels.shape[1]
            het_graph.nodes[category].data['label'] = labels
            print('start save cache files~')
            os.makedirs(cache_dir)
            dgl.save_graphs(cache_dir + 'het_graph.bin', [het_graph])
            for key in h_dict:
                torch.save(h_dict[key].to_sparse(), cache_dir + 'h_dict_{}.pt'.format(key))
            torch.save(labels, cache_dir + 'labels.pt')
            np.save(cache_dir + 'adj_dict', adj_dict)
            np.save(cache_dir + 'meta_path_adj', meta_path_adj)
            torch.save(pos.to_sparse(), cache_dir + 'pos.pt')
            for max_hop in range(1, 10):
                multi_hop_features_with_process_feature_with_normalize_adj = [
                    re_featuresv2(adj=meta_path_adj[mp],
                                  features=h_dict[category], K=max_hop, start_hops=0)
                    for mp in meta_path_adj]
                multi_hop_features_with_process_feature_with_normalize_adj = torch.stack(
                    multi_hop_features_with_process_feature_with_normalize_adj).permute(1, 0, 2, 3)
                torch.save(multi_hop_features_with_process_feature_with_normalize_adj,
                           cache_dir + 'multi_hop_features_{}_{}.pt'.format(0, max_hop))
            multi_hop_features = torch.load(
                cache_dir + 'multi_hop_features_{}_{}.pt'.format(start_hop, t_hops))

            het_graph.nodes[category].data['multi_hop_feature'] = multi_hop_features
            print('save cache files finished~')

        print('dataset freebase is loaded!')
        return het_graph, category, all_node_idx, train_idx_list, val_idx_list, test_idx_list, h_dict, labels, meta_path_num, num_classes, pos

    elif data_name == 'academic':
        path = data_dir + data_name+"/"
        het_graph, _ = dgl.load_graphs(
            path+'graph.bin')
        het_graph = het_graph[0]
        category = 'author'
        meta_path_num = 3 # APA (author-paper-author), APVPA (author-paper-venue-paper-author) and APPA (authorpaper-paper-author)
        ratio = [20, 40, 60]
        ndata = het_graph.ndata
        labels = ndata['label'][category].squeeze()
        N = labels.shape[0]
        for n in het_graph.ntypes:
            n_num = het_graph.number_of_nodes(n)
            print('type:{};num:{}'.format(n,n_num))
        train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
        val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
        test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
        train_idx_list = [torch.LongTensor(i) for i in train]
        val_idx_list = [torch.LongTensor(i) for i in val]
        test_idx_list = [torch.LongTensor(i) for i in test]
        if os.path.exists(cache_dir):
            het_graph = dgl.load_graphs(cache_dir + 'het_graph.bin')[0][0]
            all_node_idx = torch.arange(0, het_graph.num_nodes(category), dtype=het_graph.idtype)
            h_dict = {}
            h_dict[category] = torch.load(cache_dir + 'h_dict_{}.pt'.format(category)).to_dense()
            labels = torch.load(cache_dir + 'labels.pt')
            num_classes = labels.shape[1]
            pos = torch.eye(N).to_sparse()
            multi_hop_features = torch.load(
                cache_dir + 'multi_hop_features_{}_{}.pt'.format(0, t_hops))
            het_graph.nodes[category].data['multi_hop_feature'] = multi_hop_features
            print('load cache files finished~')
        else:
            label = np.load(path + "labels.npy").astype('int32')
            h_dict = {}
            # only dw_embedding
            for n in het_graph.ntypes:
                ndata = het_graph.nodes[n].data
                h_dict[n] = ndata['dw_embedding']
            feat_m=sp.coo_matrix(h_dict[category])
            adj_dict = {e: het_graph.adj(etype=e) for e in het_graph.etypes}

            apa_adj = torch.matmul(adj_dict['author-paper'], adj_dict['paper-author'])
            # mam = sp.load_npz(path + "mam.npz")
            apvpa_adj = torch.matmul(adj_dict['author-paper'], adj_dict['paper-venue'])
            apvpa_adj = torch.matmul(apvpa_adj, adj_dict['venue-paper'])
            apvpa_adj = torch.matmul(apvpa_adj, adj_dict['paper-author'])
            appa_adj =torch.matmul(adj_dict['author-paper'], adj_dict['cite'])
            appa_adj =torch.matmul(appa_adj, adj_dict['paper-author'])
            apa=sp.coo_matrix(apa_adj.to_dense().numpy())
            apvpa=sp.coo_matrix(apvpa_adj.to_dense().numpy())
            appa=sp.coo_matrix(appa_adj.to_dense().numpy())
            apa_src_node = apa.row
            apvpa_src_node = apvpa.row
            appa_src_node = appa.row
            apa_dst_node = apa.col
            apvpa_dst_node = apvpa.col
            appa_dst_node = appa.col
            graph_data = {
                ('author', 'net_apa', 'author'): (torch.tensor(apa_src_node), torch.tensor(apa_dst_node)),
                ('author', 'net_apvpa', 'author'): (torch.tensor(apvpa_src_node), torch.tensor(apvpa_dst_node)),
                ('author', 'net_appa', 'author'): (torch.tensor(appa_src_node), torch.tensor(appa_dst_node)),
            }
            het_graph = dgl.heterograph(graph_data)
            # feature
            feat_m = torch.FloatTensor(preprocess_features(feat_m))
            het_graph.nodes[category].data['feature'] = feat_m
            h_dict = {
                category: feat_m,
            }
            pos=torch.eye(N)
            # pos = ((sparse_mx_to_torch_sparse_tensor(apa).to_dense()
            #         + sparse_mx_to_torch_sparse_tensor(
            #             apvpa).to_dense()+ sparse_mx_to_torch_sparse_tensor(
            #             appa).to_dense()) >= meta_path_num).float()

            apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
            apvpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apvpa))
            appa = sparse_mx_to_torch_sparse_tensor(normalize_adj(appa))

            meta_path_adj = {'apa': apa,
                             'apvpa': apvpa,
                             'appa': appa,
                             }

            adj_dict = {e: het_graph.adj(etype=e) for e in het_graph.etypes}
            all_node_idx = torch.arange(0, het_graph.num_nodes(category), dtype=het_graph.idtype)
            # labels
            labels = encode_onehot(label)
            labels = torch.FloatTensor(labels)
            num_classes = labels.shape[1]
            het_graph.nodes[category].data['label'] = labels
            print('start save cache files~')
            os.makedirs(cache_dir)
            dgl.save_graphs(cache_dir + 'het_graph.bin', [het_graph])
            for key in h_dict:
                torch.save(h_dict[key].to_sparse(), cache_dir + 'h_dict_{}.pt'.format(key))
            torch.save(labels, cache_dir + 'labels.pt')
            np.save(cache_dir + 'adj_dict', adj_dict)
            np.save(cache_dir + 'meta_path_adj', meta_path_adj)
            torch.save(pos.to_sparse(), cache_dir + 'pos.pt')
            for max_hop in tqdm(range(1, 10)):
                multi_hop_features_with_process_feature_with_normalize_adj = [
                    re_featuresv2(adj=meta_path_adj[mp],
                                  features=h_dict[category], K=max_hop, start_hops=0)
                    for mp in meta_path_adj]
                multi_hop_features_with_process_feature_with_normalize_adj = torch.stack(
                    multi_hop_features_with_process_feature_with_normalize_adj).permute(1, 0, 2, 3)
                torch.save(multi_hop_features_with_process_feature_with_normalize_adj,
                           cache_dir + 'multi_hop_features_{}_{}.pt'.format(0, max_hop))
            multi_hop_features = torch.load(
                cache_dir + 'multi_hop_features_{}_{}.pt'.format(start_hop, t_hops))

            het_graph.nodes[category].data['multi_hop_feature'] = multi_hop_features
            print('save cache files finished~')

        print('dataset academic is loaded!')
        return het_graph, category, all_node_idx, train_idx_list, val_idx_list, test_idx_list, h_dict, labels, meta_path_num, num_classes, pos


def get_batch_pos(pos, batch_node_id_x):
    M = pos.shape[0]
    N = pos.shape[1]
    batch_num = len(batch_node_id_x)
    left_matrix_shape = [batch_num, M]
    right_matrix_shape = [N, batch_num]

    left_matrix_row_index = range(batch_num)
    left_matrix_col_index = batch_node_id_x

    value = torch.ones(batch_num)
    indices = torch.tensor([left_matrix_row_index,
                            left_matrix_col_index])
    left_matrix = torch.sparse_coo_tensor(indices, value, left_matrix_shape).cuda()

    ss1 = torch.matmul(left_matrix, pos.cuda())
    right_matrix_col_index = range(batch_num)
    right_matrix_row_index = batch_node_id_x
    indices = torch.tensor([right_matrix_row_index,
                            right_matrix_col_index])
    right_matrix = torch.sparse_coo_tensor(indices, value, right_matrix_shape).cuda()

    ss2 = torch.matmul(ss1, right_matrix)
    return ss2
