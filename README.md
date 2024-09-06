# GTC
This repo is for source code of paper "GTC: GNN-Transformer Co-contrastive Learning for Self-supervised Heterogeneous Graph
Representation". 

# A Gentle Introduction
<div align="center">
  <img src="https://github.com/PHD-lanyu/GTC/blob/main/framework.png" alt="Framework">
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
Fisrt, download the dataset from [Google Drive](https://drive.google.com/file/d/1gtJ04O-9DKMVy1a_lM_ECbM1nnLruqil/view?usp=sharing), unzip the data.zip file and place the data folder in the root directory \
Second, go into ./code, and then you can use the following commend to run our model: 


## Train New Models
> python main.py dblp 

Here, "dblp" can be replaced by "acm", "freebase" or "academic".

# License
This repository is released under the Apache 2.0 license.

# Acknowledgement
This repository is built upon [HeCo](https://github.com/liun-online/HeCo), [NAGphormer](https://github.com/JHL-HUST/NAGphormer) and [OpenHGNN](https://github.com/BUPT-GAMMA/OpenHGNN), we thank the authors for their open-sourced code.
# Citation
If you find our codes useful, please cite our work. Thank you.
```bibtex
@article{sun2024gtc,
  title={GTC: GNN-Transformer co-contrastive learning for self-supervised heterogeneous graph representation},
  author={Sun, Yundong and Zhu, Dongjie and Wang, Yansong and Fu, Yansheng and Tian, Zhaoshuo},
  journal={Neural Networks},
  pages={106645},
  year={2024},
  publisher={Elsevier}
}
```
# Contact
If you have any questions, please feel free to contact me with hitffmy@163.com
