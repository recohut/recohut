# SR-GNN

SR-GNN stands for Session-based Recommendation with Graph Neural Networks.

:::info research paper

[Wu et. al., “*Session-based Recommendation with Graph Neural Networks*”. AAAI, 2019.](https://arxiv.org/abs/1811.00855)

> The problem of session-based recommendation aims to predict user actions based on anonymous sessions. Previous methods model a session as a sequence and estimate user representations besides item representations to make recommendations. Though achieved promising results, they are insufficient to obtain accurate user vectors in sessions and neglect complex transitions of items. To obtain accurate item embedding and take complex transitions of items into account, we propose a novel method, i.e. Session-based Recommendation with Graph Neural Networks, SR-GNN for brevity. In the proposed method, session sequences are modeled as graph-structured data. Based on the session graph, GNN can capture complex transitions of items, which are difficult to be revealed by previous conventional sequential methods. Each session is then represented as the composition of the global preference and the current interest of that session using an attention network. Extensive experiments conducted on two real datasets show that SR-GNN evidently outperforms the state-of-the-art session-based recommendation methods consistently.
> 

:::

## Architecture

![The workflow of the proposed SR-GNN method. We model all session sequences as session graphs. Then, each session graph is proceeded one by one and the resulting node vectors can be obtained through a gated graph neural network. After that, each session is represented as the combination of the global preference and current interests of this session using an attention net. Finally, we predict the probability of each item that will appear to be the next-click one for each session.](/img/content-models-raw-mp2-sr-gnn-untitled.png)

The workflow of the proposed SR-GNN method. We model all session sequences as session graphs. Then, each session graph is proceeded one by one and the resulting node vectors can be obtained through a gated graph neural network. After that, each session is represented as the combination of the global preference and current interests of this session using an attention net. Finally, we predict the probability of each item that will appear to be the next-click one for each session.

![An example of a session graph and the connection matrix $A_s$.](/img/content-models-raw-mp2-sr-gnn-untitled-1.png)

An example of a session graph and the connection matrix $A_s$.

SR-GNN with normalized global connections (**SR-GNN-NGC**) replaces the connection matrix with **edge weights** extracted from the global graph on the basis of SR-GNN.

![Untitled](/img/content-models-raw-mp2-sr-gnn-untitled-2.png)

Compared with SR-GNN, SR-GNN-NGC reduces the influence of edges that are connected to nodes. Such a fusion method notably affects the integrity of the current session, especially when the weight of the edge in the graph varies, leading to performance downgrade.

SR-GNN with full connections (**SR-GNN-FC**) represents all higher-order relationships using boolean weights and appends its corresponding connection matrix to that of SR-GNN.

![Untitled](/img/content-models-raw-mp2-sr-gnn-untitled-3.png)

Similarly, it is reported that SR-GNN-FC performs worse than SR-GNN, though the experimental results of the two methods are not of many differences. Such a **small difference** suggests that in most recommendation scenarios, not every high-order transitions can be directly converted to straight connections and intermediate stages between high-order items are **still** necessities.

## Links

- [https://github.com/mmaher22/iCV-SBR/tree/master/Source Codes/SRGNN_Pytorch](https://github.com/mmaher22/iCV-SBR/tree/master/Source%20Codes/SRGNN_Pytorch)
- [https://github.com/CRIPAC-DIG/SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN)