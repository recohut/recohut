# Session-based Recommendation with Graph Neural Networks

Session-based recommendation tasks are performed based on the user's anonymous historical behavior sequence and implicit feedback data, such as clicks, browsing, purchasing, etc., rather than rating or comment data. The primary aim is to predict the next behavior based on a sequence of the historical sequence of the session. Session-based recommendation aims to predict which item a user will click next, solely based on the user’s current sequential session data without access to the long-term preference profile.

:::info

Session-based recommenders are gaining popularity due to user privacy concerns.

:::

 

The input to these systems are time-ordered logs of recorded user interactions, where the interactions are grouped into sessions. Such a session could, for example, correspond to a listening session on a music service, or a shopping session on an e-commerce site. One particularity of such approaches is that users are anonymous, which is a common problem on websites that deal with first-time users or users that are not logged in. The prediction task in this setting is to predict the next user action, given only the interactions of the ongoing session. Today, session-based recommendation is a highly active research area due to its practical relevance.

![/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled.png](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled.png)

Some of the major reason for the effectiveness and the growing popularity of session-based recommenders are: 1) Cookies and browser fingerprinting can not always recognize users correctly, especially across different devices and platforms, 2) Opt-out users are not tracked across sessions, 3) Privacy concerns for opt-in users, 4) Infrequent user visits, cookie expiration - Users visiting a site infrequently can not be recognized over long time, 5) Changing user intent across sessions - On many domains, users have session-based traits (short video sites, marketplaces, classified sites, etc.), and 6) Earlier solutions for handling sessions was only based on last user click and ignored user interactions in the session.

Session-based recommendation is an important task for domains such as e-commerce, news, streaming video and music services, where users might be untraceable, their histories can be short, and users can have rapidly changing tastes. Providing recommendations based purely on the interactions that happen in the current session.

But the problem of session-based recommendation aims to predict user actions based on anonymous sessions. Previous methods model a session as a sequence and estimate user representations besides item representations to make recommendations. Though achieved promising results, they are insufficient to obtain accurate user vectors in sessions and neglect complex transitions of items.

Session-based recommendations are challenging due to limited user-item interactions. Typical sequential models are not able to capture complex patterns from all previous interactions.

GNNs offer an intuitive approach to session-based recommendation since each session can be mapped into a graph’s chain. Each node of the graph represents an item, and each edge represents the interactions. The natural compatibility between data modeled in such a manner and GNNs allow this method to perform well.

Here is the general framework of session-based recommendations using GNNs:

![/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-1.png](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-1.png)

:::info fun fact

Representation of session with Graph neural networks was first introduced in the **SR-GNN** paper. In this method, session sequences are modeled as graph-structured data. Based on the session graph, GNN can capture complex transitions of items, which are difficult to be revealed by previous conventional sequential methods. Each session is then represented as the composition of the global preference and the current interest of that session using an attention network.

:::

## Datasets

Session data has many important characteristics:

- Session clicks and navigation are sequential by nature. The order of clicks as well as the navigational path may contain information about user intent.
- Viewed items often have metadata such as names, categories, and descriptions, which provides information about the user’s tastes and what they are looking for.
- Sessions are limited in time and scope. A session has a specific goal and generally ends once that goal is accomplished: rent a hotel for a business trip, find a restaurant for a romantic date, and so on. This means that the session has intrinsic informational power related to a specific item (such as the hotel or restaurant that’s eventually booked).

### Diginetica

Diginetica is a dataset that comes from CIKM Cup 2016. Its transaction data is suitable for session-based recommendation.

### Yoochoose

The dataset is session-based and each session contains a sequence of clicks and purchases. Since the Yoochoose dataset is too large, in some cases we only use its the most recent 1/64 fractions of the training sessions, denoted as Yoochoose 1/64.

## Preprocessing

In general, we do the following preprocessing operations on the sessions data - 

1. Drop all unit-length sessions.
2. Remove items with less than $k$ interactions.
3. Remove items that appear less than $n$ no. of times.
4. Take session of last $x$ days/weeks as test set.
5. For an existing session, generate a series of input session sequences and corresponding labels.
6. Filter out items from the test set which do not appear in the training set.
7. Keep only $M$ most popular items/users.
8. Partition the log into sessions by applying user inactivity threshold of time duration $t$.
9. Replace multiple consecutive clicks (or in general, event) on the same item by a single click on that item.

## Methods

### Baseline methods

| Model | Description |
| --- | --- |
| Item-KNN | Item-KNN is a neighborhood method that recommends items that are similar to the previous items in the current session, where the similarity between two items is defined by their cosine similarity. |
| Markov Chain | Markov chains predicts the user’s next action solely based on the previous action. But such a strong independence assumption suffers from noisy data. |
| FPMC | FPMC is a Markov-chain based method for next-basket recommendation. To adapt it for session-based recommendation, we consider the next item as the next basket. |
| GRU4Rec | GRU4Rec models short-term preferences with gated recurrent units (GRUs). |
| NARM | NARM employs RNNs with attention to capture user’s main purposes and sequential behaviors. Two RNN-based subsystems to capture users’ local and global preference respectively. |
| STAMP | STAMP extracts users’ potential interests using a simple multilayer perceptron model and an attentive network. |
| NextItNet | NextItNet is a CNN-based method for next-item recommendation. It uses dilated convolution to increase the receptive fields without using the lossy pooling operations. |

### GNN-based methods

| Model | Description |
| --- | --- |
| SR-GNN | SR-GNN first models all session sequences as session graphs. Then, each session graph is proceeded one by one and the resulting node vectors can be obtained through a gated graph neural network. After that, each session is represented as the combination of the global preference and current interests of this session using an attention net. Finally, it predict the probability of each item that will appear to be the next-click one for each session. |
| GC-SAN | GC-SAN uses a GGNN to extract local context information and a self-attention network (SAN) to capture global dependencies between distant positions. |
| FGNN | FGNN converts a session into a weighted directed graph where the edge weights are the counts of item transitions. An adapted multi-layered graph attention network (GAT) is used to extract item features and a modified Set2Set pooling operator is applied to generate session representations. |
| LESSR | Given a session, LESSR compute an edge-order preserving (EOP) multigraph and a shortcut graph. The graphs and the node representations are passed as input to multiple interleaved EOPA and SGAT layers. Each layer outputs the new node representations. The readout layer computes a graph-level representation, which is combined with the recent interests to form the session embedding. Finally, the prediction layer computes the probability distribution of the next item. |
| GCE_GNN | GCE-GNN first construct a global graph based on all training session sequences. Then for each session, a global feature encoder and local feature encoder is used to extract node feature with global context and local context. Then the model incorporates position information to learn the contribution of each item to the next predicted item. Finally, candidate items are scored. |
| DGTN | DGTN propagate the embeddings of neighbor items in the target session and the neighbor session set towards the next layer in the intra- and intersession channel, respectively. Then the embeddings of the final layer in the two channels are fed into the fusion function to obtain the final item embedding. Based on the learned item embeddings, it generate the session embedding via session pooling. Finally, a prediction layer is applied to generate the recommendation probability. |
| TAGNN | TAGNN first models all session sequences as session graphs. Then, graph neural networks capture rich item transitions in sessions. Lastly, from one session embedding vector, target-aware attention adaptively activates different user interests concerning varied target items to be predicted. |
| TAGNN++ | TAGNN models item interactions with GNN, and both local and global user interactions with  a Transformer. |

## SR-GNN

*[Wu et. al. Session-based Recommendation with Graph Neural Networks. AAAI, 2019.](https://arxiv.org/abs/1811.00855)*

SR-GNN first models all session sequences as session graphs. Then, each session graph is proceeded one by one and the resulting node vectors can be obtained through a gated graph neural network. After that, each session is represented as the combination of the global preference and current interests of this session using an attention net. Finally, it predict the probability of each item that will appear to be the next-click one for each session.

![The workflow of the proposed SR-GNN method.](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-2.png)

The workflow of the proposed SR-GNN method.

## GC-SAN

[*Xu et. al. Graph Contextualized Self-Attention Network for Session-based Recommendation. IJCAI, 2019.*](https://www.ijcai.org/proceedings/2019/0547.pdf)

Graph contextualized self-attention model (GC-SAN) utilizes both graph neural network and self-attention mechanism, for session-based recommendation. It dynamically construct a graph structure for session sequences and capture rich local dependencies via graph neural network (GNN). Then each session learns long-range dependencies by applying the self-attention mechanism. Finally, each session is represented as a linear combination of the global preference and the current interest of that session.

![Untitled](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-3.png)

## FGNN

*[Qiu et. al. Rethinking the Item Order in Session-based Recommendation with Graph Neural Networks. CIKM, 2019.](https://arxiv.org/abs/1911.11942)*

FGNN converts a session into a weighted directed graph where the edge weights are the counts of item transitions. An adapted multi-layered graph attention network (GAT) is used to extract item features and a modified Set2Set pooling operator is applied to generate session representations.

## LESSR

*[T. Chen and R. Wong, Handling Information Loss of Graph Neural Networks for Session-based Recommendation. KDD, 2020.](https://dl.acm.org/doi/pdf/10.1145/3394486.3403170)*

Given a session, LESSR compute an edge-order preserving (EOP) multigraph and a shortcut graph. The graphs and the node representations are passed as input to multiple interleaved EOPA and SGAT layers. Each layer outputs the new node representations. The readout layer computes a graph-level representation, which is combined with the recent interests to form the session embedding. Finally, the prediction layer computes the probability distribution of the next item.

## GCE-GNN

*[Wang etl. al. Global Context Enhanced Graph Neural Networks for Session-based Recommendation. arXiv, 2021.](https://arxiv.org/abs/2106.05081)*

GCE-GNN first construct a global graph based on all training session sequences. Then for each session, a global feature encoder and local feature encoder is used to extract node feature with global context and local context. Then the model incorporates position information to learn the contribution of each item to the next predicted item. Finally, candidate items are scored.

![An overview of the proposed framework.](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-4.png)

An overview of the proposed framework.

## DGTN

*[Zheng et. al. Dual-channel Graph Transition Network for Session-based Recommendation. arXiv, 2020.](https://arxiv.org/abs/2009.10002)*

DGTN propagate the embeddings of neighbor items in the target session and the neighbor session set towards the next layer in the intra- and intersession channel, respectively. Then the embeddings of the final layer in the two channels are fed into the fusion function to obtain the final item embedding. Based on the learned item embeddings, it generate the session embedding via session pooling. Finally, a prediction layer is applied to generate the recommendation probability.

![An illustration of the model architecture.](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-5.png)

An illustration of the model architecture.

## TAGNN

[*Yu et. al. Target Attentive Graph Neural Networks for Session-based Recommendation. SIGIR, 2020.*](https://arxiv.org/abs/2005.02844)

TAGNN first models all session sequences as session graphs. Then, graph neural networks capture rich item transitions in sessions. Lastly, from one session embedding vector, target-aware attention adaptively activates different user interests concerning varied target items to be predicted.

![An overview of the proposed TAGNN method.](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-6.png)

An overview of the proposed TAGNN method.

## TAGNN++

*[Mitheran et. al. Improved Representation Learning for Session-based Recommendation. arXiv, 2021.](https://arxiv.org/abs/2107.01516v2)*

TAGNN models item interactions with GNN, and both local and global user interactions with  a Transformer.

![Untitled](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-7.png)

## Challenges

Despite the convenient representation of sessions offered by GNNs, it lacks the ability to model long-range dependencies and complicated interactions. One common issue is the lossy session encoding in which some sequential information about item transitions is ignored because of the lossy encoding from sessions to graphs and the permutation-invariant aggregation during message passing. Another common issue is the GNN's ineffectiveness in capturing long-range dependencies within sessions due to the limited number of layers.

### Lossy Session Encoding Problem

It is due to their lossy encoding schemes that convert sessions to graphs. To process sessions using a GNN, the sessions need to be converted to graphs first. In these methods, each session is converted to a directed graph whose nodes are the unique items in the session and the edges are the transitions between items. The edges can be either weighted or unweighted.

![Untitled](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-8.png)

Two different sessions [$v_1, v_2, v_3, v_3, v_2, v_2, v_4$] and [$v_1, v_2, v_2, v_3, v_3, v_2, v_4$] are converted to the same graph.

Although in a particular dataset, the two sessions may produce the same next item, there may also exist a dataset in which the two sessions produce different next items. In the latter case, it is not possible for these GNN models to make correct recommendations for both sessions. Therefore, these models have a limitation in their modeling capacity.

Lossy conversion could be problematic because the information ignored may be important to determine the next item. We should let the model automatically learn to decide what information can be ignored instead of “blindly” making the decision using a lossy conversion method. Otherwise, the model is not flexible enough to fit complex datasets since its modeling capacity is limited by the lossy conversion method.

### Oversmoothing Problem

Over-smoothing is caused as they use only the embeddings updated through the last layer in the prediction layer. Specifically, as the number of layers increases, the embedding of a node will be influenced more from its neighbors’ embeddings. As a result, the embedding of a node in the last layer becomes similar to the embeddings of many directly/indirectly connected nodes. This phenomenon prevents most of the existing GCN-based methods from effectively utilizing the information of high-order neighborhood. Empirically, this is also shown by the fact that most of non-linear GCN-based methods show better performance when using only a few layers instead of deep networks.

Most common way to handle this is by **Residual Prediction** i.e. utilize the embeddings from all layers for prediction. After that, perform residual prediction, which predict each user’s preference to each item with the multiple embeddings from the multiple layers.

## Tutorials

### SR-GNN Session-based Recommendation on Sample dataset

[direct link to notebook →](https://nbviewer.org/gist/sparsh-ai/9cc4447495dfd6465698b8d99afc2316)

![Untitled](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-9.png)

### TAGNN Session-based Recommendation on Sample dataset

[direct link to notebook →](https://nbviewer.org/gist/sparsh-ai/64f5ed5f288b03693880ddf94b56b6c2)

:::note

The flow diagram of this is same as SR-GNN, the difference is in the implementation of `SessionGraph()` module mainly.

:::

![Untitled](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-10.png)

### TAGNN++ Session-based Recommendation on Diginetica and Yoochose dataset

[direct link to notebook (diginetica) →](https://nbviewer.org/gist/sparsh-ai/f506616b853cad89c76c8e3ca2e0c105)

[direct link to notebook (yoochoose) →](https://nbviewer.org/gist/sparsh-ai/339575ed65f760ee2d9d21f3f977bb67)

![Untitled](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-11.png)

### GC-SAN implementation in PyTorch

[direct link to notebook →](https://nbviewer.org/gist/sparsh-ai/25d725cd3ffb43ef711fcd3ac13390a6)

![Untitled](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-12.png)

### LESSR Session-based Recommendation on Sample dataset

[direct link to notebook →](https://nbviewer.org/gist/sparsh-ai/2187689c8f2ecda58a0e5d5510f077ef)

![Untitled](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-13.png)

### GCE-GNN Session-based Recommendation on Diginetica, Tmall and NowPlaying dataset

[direct link to notebook (diginetica)→](https://nbviewer.org/gist/sparsh-ai/0255e6fadc347db81ba360e8d2d571c4) 

[direct link to notebook (tmall)→](https://nbviewer.org/gist/sparsh-ai/ac9c8e27612e9a4656f06376aa260d6a)

[direct link to notebook (nowplaying)→](https://nbviewer.org/gist/sparsh-ai/83ed18af8d234d9745cfd7b2ddaf8da0)

![Untitled](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-14.png)

### FGNN Session-based Recommendation on Sample dataset

[direct link to notebook →](https://nbviewer.org/gist/sparsh-ai/642d10ae901e3bb8b8821070ad689b71)

![Untitled](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-15.png)

### Conversion of sessions into Session graph on Yoochoose dataset

[direct link to notebook →](https://nbviewer.org/gist/sparsh-ai/d732e7c3cf42dc5f26c551cd88b8641c)

### Session graph with attention in PyTorch

[direct link to notebook →](https://gist.github.com/sparsh-ai/a47acac562c6801f4bd37196e251fd88)

### Preprocessing of session datasets

:::note python notebooks
[Preprocessing of Sample session dataset](https://nbviewer.org/gist/sparsh-ai/12d1f5ca07add606f27b0f841b550a82)

[Preprocessing of Diginetica session dataset](https://nbviewer.org/gist/sparsh-ai/fbaf4627cbd3fe5b45efc2f6ab50920a)

[Preprocessing of Gowalla session dataset](https://nbviewer.org/gist/sparsh-ai/43b1bfb234971380b4f3179244bc25f5)

[Preprocessing of LastFM session dataset](https://nbviewer.org/gist/sparsh-ai/512750bd6427d98ca396d041abf421ac)
:::

## Appendix

### Session Graph

![An example of a session graph and the connection matrix $A_s$.](/img/content-tutorials-raw-session-based-recommendation-with-graph-neural-net-untitled-16.png)

An example of a session graph and the connection matrix $A_s$.