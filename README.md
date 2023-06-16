# GTC
This repo is for source code of paper "GTC: GNN-Transformer Co-contrastive Learning for Self-supervised Heterogeneous Graph
Representations". 

# A Gentle Introduction
<div align="center">
  <img src="https://github.com/PHD-lanyu/GTC/blob/main/framework.png">
</div>

Graph Neural Networks (GNNs) have emerged as the most powerful weapon for various graph tasks due to the message-passing mechanism’s great local information aggregation ability. However, over-smoothing has always hindered GNNs from going deeper and capturing multi-hop neighbors. Unlike GNNs, Transformers can model global information and multi-hop interactions via multi-head self-attention and a proper Transformer structure can show more immunity to the over-smoothing problem. So, **can we propose a novel framework to combine GNN and Transformer, integrating both of GNN's local information aggregation and Transformer's global information modeling ability to eliminate the over-smoothing problem?** To realize this, this paper proposes a collaborative learning scheme for GNN-Transformer and constructs GTC architecture. GTC leverages the GNN and Transformer branch to encode node information from different views respectively, and establishes contrastive learning tasks based on the encoded cross-view information to realize self-supervised heterogeneous graph representation. For the Transformer branch, we propose Metapath-aware Hop2Token and CG-Hetphormer, which can cooperate with GNN to attentively encode neighborhood information from different levels. As far as we know, this is the first attempt in the field of graph representation learning to utilize both GNN and Transformer to collaboratively capture different view information and conduct cross-view contrastive learning. The experiments on real datasets show that GTC exhibits superior performance compared with state-of-the-art methods.

## Environment Settings
> python==3.7 \
torch==1.12.0 \
scikit_learn==0.24.2 \
torch-geometric==2.2.0 \
torchvision==0.13.0 \
dgl==0.9.1 \
openhgnn==0.4.0

GPU: GeForce RTX 4090  24GB \
CPU: AMD 3900X @3.80GHz 24-core
# Usage

Fisrt, go into ./code, and then you can use the following commend to run our model: 

## Pre-trained Models
> python main.py dblp --load_from_pretrained 

## Train New Models
> python main.py dblp 

[//]: # (> python main.py acm )

[//]: # (> python main.py freebase )

Here, "dblp" can be replaced by "acm"  or "freebase".

[//]: # (## Some tips in parameters)

[//]: # (1. We suggest you to carefully select the *“pos_num”* &#40;existed in ./data/pos.py&#41; to ensure the threshold of postives for every node. This is very important to final results. Of course, more effective way to select positives is welcome.)

[//]: # (2. In ./code/utils/params.py, except "lr" and "patience", meticulously tuning dropout and tau is applaudable.)

[//]: # (3. In our experiments, we only assign target type of nodes with original features, but assign other type of nodes with one-hot. This is because most of datasets used only provide features of target nodes in their original version. So, we believe in that if high-quality features of other type of nodes are provided, the overall results will improve a lot. The AMiner dataset is an example. In this dataset, there are not original features, so every type of nodes are all asigned with one-hot. In other words, every node has the same quality of features, and in this case, our HeCo is far ahead of other baselines. So, we strongly suggest that if you have high-quality features for other type of nodes, try it!)

[//]: # (## Cite)

[//]: # (```)

[//]: # (@inproceedings{heco,)

[//]: # (  author    = {Xiao Wang and)

[//]: # (               Nian Liu and)

[//]: # (               Hui Han and)

[//]: # (               Chuan Shi},)

[//]: # (  title     = {Self-supervised Heterogeneous Graph Neural Network with Co-contrastive)

[//]: # (               Learning},)

[//]: # (  booktitle = {{KDD} '21: The 27th {ACM} {SIGKDD} Conference on Knowledge Discovery)

[//]: # (               and Data Mining, Virtual Event, Singapore, August 14-18, 2021},)

[//]: # (  pages     = {1726--1736},)

[//]: # (  year      = {2021})

[//]: # (})

[//]: # (```)
# License
This repository is released under the Apache 2.0 license.

# Acknowledgement
This repository is built upon [HeCo](https://github.com/liun-online/HeCo), [NAGphormer](https://github.com/JHL-HUST/NAGphormer) and [OpenHGNN](https://github.com/BUPT-GAMMA/OpenHGNN), we thank the authors for their open-sourced code.

# Contact
If you have any questions, please feel free to contact me with hitffmy@163.com
