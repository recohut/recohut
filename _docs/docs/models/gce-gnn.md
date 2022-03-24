# GCE-GNN

GCE-GNN stands for Global Context Enhanced Graph Neural Networks. It exploit item transitions over all sessions in a more subtle manner for better inferring the user preference of the current session.

:::info research paper

[Zhang et. al., “*Global Context Enhanced Graph Neural Networks for Session-based Recommendation*”. arXiv, 2021.](https://arxiv.org/abs/2106.05081)

> Session-based recommendation (SBR) is a challenging task, which aims at recommending items based on anonymous behavior sequences. Almost all the existing solutions for SBR model user preference only based on the current session without exploiting the other sessions, which may contain both relevant and irrelevant item-transitions to the current session. This paper proposes a novel approach, called Global Context Enhanced Graph Neural Networks (GCE-GNN) to exploit item transitions over all sessions in a more subtle manner for better inferring the user preference of the current session. Specifically, GCE-GNN learns two levels of item embeddings from session graph and global graph, respectively: (i) Session graph, which is to learn the session-level item embedding by modeling pairwise item-transitions within the current session; and (ii) Global graph, which is to learn the global-level item embedding by modeling pairwise item-transitions over all sessions. In GCE-GNN, we propose a novel global-level item representation learning layer, which employs a session-aware attention mechanism to recursively incorporate the neighbors' embeddings of each node on the global graph. We also design a session-level item representation learning layer, which employs a GNN on the session graph to learn session-level item embeddings within the current session. Moreover, GCE-GNN aggregates the learnt item representations in the two levels with a soft attention mechanism. Experiments on three benchmark datasets demonstrate that GCE-GNN outperforms the state-of-the-art methods consistently.
> 

:::

## Architecture

![An overview of the proposed framework. Firstly, a global graph is constructed based on all training session sequences. Then for each session, a global feature encoder and local feature encoder will be used to extract node feature with global context and local context. Then the model incorporates position information to learn the contribution of each item to the next predicted item. Finally, candidate items will be scored.](/img/content-models-raw-mp1-gce-gnn-untitled.png)

An overview of the proposed framework. Firstly, a global graph is constructed based on all training session sequences. Then for each session, a global feature encoder and local feature encoder will be used to extract node feature with global context and local context. Then the model incorporates position information to learn the contribution of each item to the next predicted item. Finally, candidate items will be scored.