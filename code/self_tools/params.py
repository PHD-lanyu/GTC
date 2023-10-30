import argparse
import sys

argv = sys.argv
dataset = argv[1]


def acm_params():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--load_from_pretrained', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=1000)
    parser.add_argument('--gnn_branch_layer_num', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=512)
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--lam', type=float, default=0.4)
    parser.add_argument('--feat_drop', type=float, default=0.2)
    # parser.add_argument('--attn_drop', type=float, default=0.1)
    # parser.add_argument('--sample_rate', nargs='+', type=int, default=[6])
    # transformer related parameters
    #
    parser.add_argument('--t_hops', type=int, default=2,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--t_pe_dim', type=int, default=40,
                        help='position embedding size')
    parser.add_argument('--t_n_layers', type=int, default=6,
                        help='Number of Transformer layers')
    parser.add_argument('--t_n_heads', type=int, default=6,
                        # parser.add_argument('--t_n_heads', type=int, default=6,
                        help='Number of Transformer heads')
    parser.add_argument('--t_dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--t_attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    # --------------------------------------------------------------------------------

    args, _ = parser.parse_known_args()

    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--load_from_pretrained', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=100)
    parser.add_argument('--gnn_branch_layer_num', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=512)
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--feat_drop', type=float, default=0.4)

    # transformer related parameters
    parser.add_argument('--t_hops', type=int, default=1,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--t_pe_dim', type=int, default=45,
                        help='position embedding size')
    parser.add_argument('--t_n_layers', type=int, default=5,
                        help='Number of Transformer layers')
    parser.add_argument('--t_n_heads', type=int, default=3,
                        # parser.add_argument('--t_n_heads', type=int, default=6,
                        help='Number of Transformer heads')
    parser.add_argument('--t_dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--t_attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    # --------------------------------------------------------------------------------

    args, _ = parser.parse_known_args()

    return args

def freebase_params():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--load_from_pretrained', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="freebase")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3838)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=1000)
    parser.add_argument('--gnn_branch_layer_num', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0016)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.7)
    parser.add_argument('--feat_drop', type=float, default=0.6)

    # transformer related parameters
    parser.add_argument('--t_hops', type=int, default=8,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--t_pe_dim', type=int, default=20,
                        help='position embedding size')
    parser.add_argument('--t_n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--t_n_heads', type=int, default=2,
                        # parser.add_argument('--t_n_heads', type=int, default=6,
                        help='Number of Transformer heads')
    parser.add_argument('--t_dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--t_attention_dropout', type=float, default=0.7,
                        help='Dropout in the attention layer')
    # --------------------------------------------------------------------------------

    args, _ = parser.parse_known_args()
    return args

def academic_params():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--load_from_pretrained', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="academic")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=1000)
    parser.add_argument('--gnn_branch_layer_num', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.009)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--feat_drop', type=float, default=0.7)

    # transformer related parameters
    parser.add_argument('--t_hops', type=int, default=4,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--t_pe_dim', type=int, default=20,
                        help='position embedding size')
    parser.add_argument('--t_n_layers', type=int, default=2,
                        help='Number of Transformer layers')
    parser.add_argument('--t_n_heads', type=int, default=4,
                        # parser.add_argument('--t_n_heads', type=int, default=6,
                        help='Number of Transformer heads')
    parser.add_argument('--t_dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--t_attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    # --------------------------------------------------------------------------------

    args, _ = parser.parse_known_args()
    return args
def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "freebase":
        args = freebase_params()
    elif dataset == "academic":
        args = academic_params()

    return args
