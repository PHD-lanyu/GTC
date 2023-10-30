import numpy as np
import torch
from torch.utils.data import DataLoader
from .logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax


def evaluate_for_test(hidden_dim, idx_train, idx_val, idx_test, label, nb_classes, device, dataset, lr, wd
                      , model=None, fea_evalue=None, batch_size=1000, patience=20, model_dir=None):
    import torch.utils.data as Data
    xent = nn.CrossEntropyLoss()
    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    auc_score_list = []
    train_dataset = Data.TensorDataset(idx_train, train_lbls)
    valid_dataset = Data.TensorDataset(idx_val, val_lbls)
    test_dataset = Data.TensorDataset(idx_test, test_lbls)
    for exp in range(10):
        log = LogReg(hidden_dim, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)
        val_accs = []
        val_micro_f1s = []
        val_macro_f1s = []
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        best_val_acc = 0
        patience_count = 0
        for epoch_ in range(200):
            # train
            log.train()
            for i, data in enumerate(train_loader):
                batch_id, labels = data
                bach_fea = fea_evalue[batch_id].permute(1, 0, 2, 3)
                batch_emb = model.get_embeds(multi_hop_features=bach_fea)
                logits = log(batch_emb)
                loss = xent(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
            # val
            log.eval()
            val_pred_list = []
            val_label_list = []
            for i, data in enumerate(valid_loader):
                batch_id, labels = data
                bach_fea = fea_evalue[batch_id].permute(1, 0, 2, 3)
                batch_emb = model.get_embeds(multi_hop_features=bach_fea)
                logits = log(batch_emb)
                preds = torch.argmax(logits, dim=1)
                val_pred_list = np.append(val_pred_list, preds.cpu())
                val_label_list = np.append(val_label_list, labels.cpu())
            val_pred_list = torch.FloatTensor(val_pred_list).to(device)
            val_label_list = torch.FloatTensor(val_label_list).to(device)

            val_acc = torch.sum(val_pred_list == val_label_list).float() / val_label_list.shape[0]
            if val_acc > best_val_acc:
                if model_dir is not None:
                    torch.save(log.state_dict(), model_dir + 'GTC_LogReg' + dataset + '_exp_{}.pkl'.format(exp))
                else:
                    torch.save(log.state_dict(), '../data/GTC_LogReg' + dataset + '_exp_{}.pkl'.format(exp))
                best_val_acc = val_acc
                patience_count = 0
            else:
                patience_count = patience_count + 1
                if patience_count >= patience:
                    break

            val_f1_macro = f1_score(val_label_list.cpu(), val_pred_list.cpu(), average='macro')
            val_f1_micro = f1_score(val_label_list.cpu(), val_pred_list.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

        if model_dir is not None:
            log.load_state_dict(torch.load(model_dir + 'GTC_LogReg' + dataset + '_exp_{}.pkl'.format(exp)))
        else:
            log.load_state_dict(torch.load('../data/GTC_LogReg' + dataset + '_exp_{}.pkl'.format(exp)))
        log.eval()
        test_pred_list = []
        test_label_list = []
        test_logits_list = None
        for i, data in enumerate(test_loader):
            batch_id, labels = data
            bach_fea = fea_evalue[batch_id].permute(1, 0, 2, 3)
            batch_emb = model.get_embeds(multi_hop_features=bach_fea)
            logits = log(batch_emb)
            preds = torch.argmax(logits, dim=1)
            test_pred_list = np.append(test_pred_list, preds.cpu())
            test_label_list = np.append(test_label_list, labels.cpu())
            if test_logits_list is None:
                test_logits_list = logits.detach().cpu().numpy()
            else:
                test_logits_list = np.vstack((test_logits_list, logits.detach().cpu().numpy()))

        test_label_list = torch.FloatTensor(test_label_list).to(device)
        test_pred_list = torch.FloatTensor(test_pred_list).to(device)
        test_logits_list = torch.FloatTensor(test_logits_list).to(device)

        test_acc = torch.sum(test_pred_list == test_label_list).float() / test_label_list.shape[0]
        test_f1_macro = f1_score(test_label_list.cpu(), test_pred_list.cpu(), average='macro')
        test_f1_micro = f1_score(test_label_list.cpu(), test_pred_list.cpu(), average='micro')

        accs.append(test_acc.detach().cpu().numpy())
        macro_f1s.append(test_f1_macro)

        micro_f1s.append(test_f1_micro)

        # auc
        if nb_classes <= 2:
            best_proba = softmax(test_logits_list, dim=1)[:, 1]
            auc_score_list.append(roc_auc_score(y_true=test_label_list.detach().cpu().numpy(),
                                                y_score=best_proba.detach().cpu().numpy()))
        else:
            best_proba = softmax(test_logits_list, dim=1)
            auc_score_list.append(roc_auc_score(y_true=test_label_list.detach().cpu().numpy(),
                                                y_score=best_proba.detach().cpu().numpy(),
                                                multi_class='ovr'
                                                ))

    print(
        "[Classification] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} "
        "auc {:.4f} var: {:.4f}"
            .format(
            np.mean(macro_f1s),
            np.std(macro_f1s),
            np.mean(micro_f1s),
            np.std(micro_f1s),
            np.mean(auc_score_list),
            np.std(auc_score_list)
        )
    )


def evaluate_for_train(hidden_dim, idx_train, idx_val, idx_test, label, nb_classes, device, dataset, lr, wd,
                       batch_size=1000, patience=20, emb=None):
    import torch.utils.data as Data

    xent = nn.CrossEntropyLoss()
    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    auc_score_list = []
    train_dataset = Data.TensorDataset(idx_train, train_lbls)
    valid_dataset = Data.TensorDataset(idx_val, val_lbls)
    test_dataset = Data.TensorDataset(idx_test, test_lbls)
    for exp in range(10):  # 10 times exp using LogReg
        log = LogReg(hidden_dim, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)
        val_accs = []
        val_micro_f1s = []
        val_macro_f1s = []
        # dataloader for mini-batch test
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        best_val_acc = 0
        patience_count = 0
        for epoch_ in range(1000):
            # train
            log.train()
            for i, data in enumerate(train_loader):
                batch_id, labels = data
                batch_emb = emb[batch_id]
                logits = log(batch_emb)
                loss = xent(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
            # val
            log.eval()
            val_pred_list = []
            val_label_list = []
            for i, data in enumerate(valid_loader):
                batch_id, labels = data
                batch_emb = emb[batch_id]
                logits = log(batch_emb)
                preds = torch.argmax(logits, dim=1)
                val_pred_list = np.append(val_pred_list, preds.cpu())
                val_label_list = np.append(val_label_list, labels.cpu())
            val_pred_list = torch.FloatTensor(val_pred_list).to(device)
            val_label_list = torch.FloatTensor(val_label_list).to(device)
            val_acc = torch.sum(val_pred_list == val_label_list).float() / val_label_list.shape[0]
            # early stop
            if val_acc > best_val_acc:
                torch.save(log.state_dict(), '../data/GTC_LogReg' + dataset + '_exp_{}.pkl'.format(exp))
                best_val_acc = val_acc
                patience_count = 0
            else:
                patience_count = patience_count + 1
                if patience_count >= patience:
                    break

            val_f1_macro = f1_score(val_label_list.cpu(), val_pred_list.cpu(), average='macro')
            val_f1_micro = f1_score(val_label_list.cpu(), val_pred_list.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)
        # test
        log.load_state_dict(torch.load('../data/GTC_LogReg' + dataset + '_exp_{}.pkl'.format(exp)))
        log.eval()
        test_pred_list = []
        test_label_list = []
        test_logits_list = None
        for i, data in enumerate(test_loader):
            batch_id, labels = data
            batch_emb = emb[batch_id]
            logits = log(batch_emb)
            preds = torch.argmax(logits, dim=1)
            test_pred_list = np.append(test_pred_list, preds.cpu())
            test_label_list = np.append(test_label_list, labels.cpu())
            if test_logits_list is None:
                test_logits_list = logits.detach().cpu().numpy()
            else:
                test_logits_list = np.vstack((test_logits_list, logits.detach().cpu().numpy()))

        test_label_list = torch.FloatTensor(test_label_list).to(device)
        test_pred_list = torch.FloatTensor(test_pred_list).to(device)
        test_logits_list = torch.FloatTensor(test_logits_list).to(device)

        test_acc = torch.sum(test_pred_list == test_label_list).float() / test_label_list.shape[0]
        test_f1_macro = f1_score(test_label_list.cpu(), test_pred_list.cpu(), average='macro')
        test_f1_micro = f1_score(test_label_list.cpu(), test_pred_list.cpu(), average='micro')
        accs.append(test_acc.detach().cpu().numpy())
        macro_f1s.append(test_f1_macro)
        micro_f1s.append(test_f1_micro)
        # auc
        if nb_classes <= 2:
            best_proba = softmax(test_logits_list, dim=1)[:, 1]
            auc_score_list.append(roc_auc_score(y_true=test_label_list.detach().cpu().numpy(),
                                                y_score=best_proba.detach().cpu().numpy()))
        else:
            best_proba = softmax(test_logits_list, dim=1)
            auc_score_list.append(roc_auc_score(y_true=test_label_list.detach().cpu().numpy(),
                                                y_score=best_proba.detach().cpu().numpy(),
                                                multi_class='ovr'
                                                ))

    max_iter = macro_f1s.index(max(macro_f1s))
    print(
        "[Classification] Macro-F1_mean: {:.4f}  Micro-F1_mean: {:.4f} "
        "auc {:.4f} "
            .format(macro_f1s[max_iter],
                    micro_f1s[max_iter],
                    auc_score_list[max_iter],
                    )
    )
    return macro_f1s[max_iter], micro_f1s[max_iter], auc_score_list[max_iter]
