# GC-SAN

GC-SAN stands for Graph contextualized self-attention.

:::info research paper

[Xu et. al., “*Graph Contextualized Self-Attention Network for Session-based Recommendation*”. IJCAI, 2019.](https://www.ijcai.org/proceedings/2019/0547.pdf)

> Session-based recommendation, which aims to predict the user’s immediate next action based on anonymous sessions, is a key task in many online services (e.g., e-commerce, media streaming). Recently, Self-Attention Network (SAN) has achieved significant success in various sequence modeling tasks without using either recurrent or convolutional network. However, SAN lacks local dependencies that exist over adjacent items and limits its capacity for learning contextualized representations of items in sequences. In this paper, we propose a graph contextualized self-attention model (GC-SAN), which utilizes both graph neural network and self-attention mechanism, for session-based recommendation. In GC-SAN, we dynamically construct a graph structure for session sequences and capture rich local dependencies via graph neural network (GNN). Then each session learns long-range dependencies by applying the self-attention mechanism. Finally, each session is represented as a linear combination of the global preference and the current interest of that session. Extensive experiments on two real-world datasets show that GC-SAN outperforms state-of-the-art methods consistently.
> 

:::

Graph contextualized self-attention model (GC-SAN) utilizes both graph neural network and self-attention mechanism, for session-based recommendation. In GC-SAN, we dynamically construct a graph structure for session sequences and capture rich local dependencies via graph neural network (GNN). Then each session learns long-range dependencies by applying the self-attention mechanism. Finally, each session is represented as a linear combination of the global preference and the current interest of that session.

![Untitled](/img/content-models-raw-mp2-gc-san-untitled.png)

We first construct a directed graph of all session sequences. Based on the graph, we apply graph neural network to obtain all node vectors involved in the session graph. After that, we use a multi-layer self-attention network to capture long-range dependencies between items in the session. In prediction layer, we represent each session as a linear of the global preference and the current interest of that session. Finally, we compute the ranking scores of each candidate item for recommendation.

![Untitled](/img/content-models-raw-mp2-gc-san-untitled-1.png)