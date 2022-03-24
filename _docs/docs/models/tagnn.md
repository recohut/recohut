# TAGNN

TAGNN stands for Target Attentive Graph Neural Network. Session-based recommendations are challenging due to limited user-item interactions. Typical sequential models are not able to capture complex patterns from all previous interactions. SessionGraph (a graph representation of sessions) can capture the complex patterns from all previous interactions.

:::info research paper

[Yu et. al., “*Target Attentive Graph Neural Networks for Session-based Recommendation*”. SIGIR, 2020.](https://arxiv.org/abs/2005.02844)

> Session-based recommendation nowadays plays a vital role in many websites, which aims to predict users' actions based on anonymous sessions. There have emerged many studies that model a session as a sequence or a graph via investigating temporal transitions of items in a session. However, these methods compress a session into one fixed representation vector without considering the target items to be predicted. The fixed vector will restrict the representation ability of the recommender model, considering the diversity of target items and users' interests. In this paper, we propose a novel target attentive graph neural network (TAGNN) model for session-based recommendation. In TAGNN, target-aware attention adaptively activates different user interests with respect to varied target items. The learned interest representation vector varies with different target items, greatly improving the expressiveness of the model. Moreover, TAGNN harnesses the power of graph neural networks to capture rich item transitions in sessions. Comprehensive experiments conducted on real-world datasets demonstrate its superiority over state-of-the-art methods.
> 

:::

In TAGNN, we first construct session graphs using items in historical sessions. After that, we obtain corresponding embeddings using graph neural networks to capture complex item transitions based on session graphs. Given the item embeddings, we employ a target-aware attentive network to activate specific user interests with respect to a target item. Following that, we construct session embeddings. At last, for each session, we can infer the user’s next action based on item embeddings and the session embedding.

![An overview of the proposed TAGNN method. Session graphs are constructed based on sessions at first. Then, graph neural networks capture rich item transitions in sessions. Last, from one session embedding vector, target-aware attention adaptively activates different user interests concerning varied target items to be predicted.](/img/content-models-raw-mp2-tagnn-untitled.png)

An overview of the proposed TAGNN method. Session graphs are constructed based on sessions at first. Then, graph neural networks capture rich item transitions in sessions. Last, from one session embedding vector, target-aware attention adaptively activates different user interests concerning varied target items to be predicted.