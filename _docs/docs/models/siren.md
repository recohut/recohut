# SiReN

Existing literature often ignores the negative feedback e.g. dislikes on YouTube videos, and only capture the homophily (or assortativity) patterns by positive feedback. This is a missed opportunity situation. Performance of GNN-based Recommender Systems can be improved by including negative feedbacks. Disassortivity patterns can be learned by negative feedback. LightGCN can capture the assortativity patterns. and the MLP network can capture the disassortivity patterns.

Let's say I watched movie A and rated it 4 on a 1-5 scale (where 5 is the best), It indicates that I liked the movie. Next time, I watched movie B and rated it 2 on the same scale, what does this indicates? Well, it is below the average, so we can take an assumption here, just like we did in the first case, that anything below (resp. above) an average (or a number *n* in general) would indicate that the user didn't liked (resp. liked) the movie. SiRen leverages this core assumption to provide the recommendation to users. It first constructs a signed bipartite graph $\mathcal{E}^s = (u,v,w^s_{uv})|w^s_{uv} = w_{uv} − w_o,(u,v,w_{uv}) ∈ \mathcal{E}$. Then it split this into 2 graphs. $\mathcal{E}^p = (u,v,w^s_{uv})|w^s_{uv} > 0,\  (u,v,w^s_{uv}) ∈ \mathcal{E}^s$, and $\mathcal{E}^n = (u,v,w^s_{uv})|w^s_{uv} < 0,\ (u,v,w^s_{uv}) ∈ \mathcal{E}^s$. The purpose of this graph partitioning is to make the graphs $G^p$ and $G^n$, respectively, assortative and disassortative so that each partitioned graph is used as input to the most appropriate learning model.

:::info research paper

[Seo et. al., “*Sign-Aware Recommendation Using Graph Neural Networks*”. arXiv, 2021.](https://arxiv.org/abs/2108.08735v1)

> In recent years, many recommender systems using network embedding (NE) such as graph neural networks (GNNs) have been extensively studied in the sense of improving recommendation accuracy. However, such attempts have focused mostly on utilizing only the information of positive user-item interactions with high ratings. Thus, there is a challenge on how to make use of low rating scores for representing users' preferences since low ratings can be still informative in designing NE-based recommender systems. In this study, we present SiReN, a new sign-aware recommender system based on GNN models. Specifically, SiReN has three key components: 1) constructing a signed bipartite graph for more precisely representing users' preferences, which is split into two edge-disjoint graphs with positive and negative edges each, 2) generating two embeddings for the partitioned graphs with positive and negative edges via a GNN model and a multi-layer perceptron (MLP), respectively, and then using an attention model to obtain the final embeddings, and 3) establishing a sign-aware Bayesian personalized ranking (BPR) loss function in the process of optimization. Through comprehensive experiments, we empirically demonstrate that SiReN consistently outperforms state-of-the-art NE-aided recommendation methods. [(read less)](https://paperswithcode.com/paper/siren-sign-aware-recommendation-using-graph#)
> 

:::

## Architecture

![For assortative relation learning, we use GNN network and for learning the disassortative relations, MLP Network is a better candidate. The resultant embeddings $Z^p$ and $Z^n$ are then reweighted using attention mechanism. Finally, the whole network parameters is optimized using a sign-aware BPR loss function.](/img/content-models-raw-mp1-siren-untitled.png)

For assortative relation learning, we use GNN network and for learning the disassortative relations, MLP Network is a better candidate. The resultant embeddings $Z^p$ and $Z^n$ are then reweighted using attention mechanism. Finally, the whole network parameters is optimized using a sign-aware BPR loss function.

:::note

For the graph with negative edges, we adopt a multi-layer perceptron (MLP) due to the fact that negative edges can weaken the homophily and thus message passing to such dissimilar nodes would not be feasible.

:::