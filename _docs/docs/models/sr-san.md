# SR-SAN

SR-SAN stands for Session-based Recommendation with Self-Attention Networks. 

:::info research paper

[Jun Fang, “*Session-based Recommendation with Self-Attention Networks*”. arXiv, 2021.](https://arxiv.org/abs/2102.01922)

> Session-based recommendation aims to predict user's next behavior from current session and previous anonymous sessions. Capturing long-range dependencies between items is a vital challenge in session-based recommendation. A novel approach is proposed for session-based recommendation with self-attention networks (SR-SAN) as a remedy. The self-attention networks (SAN) allow SR-SAN capture the global dependencies among all items of a session regardless of their distance. In SR-SAN, a single item latent vector is used to capture both current interest and global interest instead of session embedding which is composed of current interest embedding and global interest embedding. Some experiments have been performed on some open benchmark datasets. Experimental results show that the proposed method outperforms some state-of-the-arts by comparisons.
> 

:::

Firstly, a self-attention based model which captures and reserves the full dependencies among all items regardless of their distance is proposed without using RNNs or GNNs. Secondly, to generate session-based recommendations, the proposed method use a single item latent vector which jointly represents current interest and global interest instead of session embedding which is composed of current interest embedding and global interest embedding. In RNNs or GNNs based methods, the global interest embedding usually obtained by aggregating all items in the session with attention mechanism which is based on current interest embedding. However, this is redundant in SR-SAN which last item embedding is aggregating all items in the session with self-attention mechanism. In this way, the last item embedding in session can jointly represent current interest and global interest.

It utilizes the self-attention to learn global item dependencies. The multi-head attention mechanism is adopted to allow SR-SAN focus on different important part of the session. The latent vector of the last item in the session is used to jointly represents current interest and global interest with prediction layer.

![The architecture of SR-SAN.](/img/content-models-raw-mp2-sr-san-untitled.png)

The architecture of SR-SAN.

Session-based recommender system makes prediction based upon current user sessions data without accessing to the long-term preference profile. Let $V = \{v_1, v_2, . . ., v_{|V|}\}$ denote the set consisting of all unique items involved in all the sessions. An anonymous session sequence $S$ can be represented by a list $S = [s_1, s_2, . . ., s_n]$, where $s_i ∈ V$ represents a clicked item of the user within the session $S$. The task of session-based recommendation is to predict the next click $s_{n+1}$ for session $S$. Our models are constructed and trained as a classifier that learns to generate a score for each of the candidates in $V$. Let $\hat{y} = \{\hat{y}_1, \hat{y}_2, . . ., \hat{y}_{|V|}\}$ denote the output score vector, where $\hat{y}_i$ corresponds to the score of item $v_i$. The items with top-K values in $\hat{y}$ will be the candidate items for recommendation.

The proposed model is made up of two parts. The first part is obtaining item latent vectors with self-attention networks, the second part of the proposed model is making recommendation with prediction layer.

## Links

- [https://github.com/GalaxyCruiser/SR-SAN](https://github.com/GalaxyCruiser/SR-SAN)