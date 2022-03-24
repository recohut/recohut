# MB-GMN

MB-GMN stands for Multi-behavior pattern modeling with meta-knowledge learner.

:::info research paper

[Lianghao Xia, Yong Xu, Chao Huang, Peng Dai and Liefeng Bo, “*Graph Meta Network for Multi-Behavior Recommendation*”. SIGIR, 2021.](https://arxiv.org/abs/2110.03969)

> Modern recommender systems often embed users and items into low-dimensional latent representations, based on their observed interactions. In practical recommendation scenarios, users often exhibit various intents which drive them to interact with items with multiple behavior types (e.g., click, tag-as-favorite, purchase). However, the diversity of user behaviors is ignored in most of the existing approaches, which makes them difficult to capture heterogeneous relational structures across different types of interactive behaviors. Exploring multi-typed behavior patterns is of great importance to recommendation systems, yet is very challenging because of two aspects: i) The complex dependencies across different types of user-item interactions; ii) Diversity of such multi-behavior patterns may vary by users due to their personalized preference. To tackle the above challenges, we propose a Multi-Behavior recommendation framework with Graph Meta Network to incorporate the multi-behavior pattern modeling into a meta-learning paradigm. Our developed MB-GMN empowers the user-item interaction learning with the capability of uncovering type-dependent behavior representations, which automatically distills the behavior heterogeneity and interaction diversity for recommendations. Extensive experiments on three real-world datasets show the effectiveness of MB-GMN by significantly boosting the recommendation performance as compared to various state-of-the-art baselines. The source code is available at https://github.com/akaxlh/MB-GMN.
> 

:::

## Architecture

![The model architecture of MB-GMN. (a) Multi-behavior pattern modeling with meta-knowledge learner for behavior heterogeneity; (b) Meta graph neural network which preserves the behavior semantics with high-order connectivity; (c) Metaknowledge transfer networks that customize the parameter of prediction layer to capture cross-type behavior dependency.](/img/content-models-raw-mp1-mb-gmn-untitled.png)

The model architecture of MB-GMN. (a) Multi-behavior pattern modeling with meta-knowledge learner for behavior heterogeneity; (b) Meta graph neural network which preserves the behavior semantics with high-order connectivity; (c) Metaknowledge transfer networks that customize the parameter of prediction layer to capture cross-type behavior dependency.