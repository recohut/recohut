# AutoInt

:::info research paper

[Song et. al., “*Automatic Feature Interaction Learning via Self-Attentive Neural Networks*”. CIKM, 2018.](https://arxiv.org/abs/1810.11921)

> Click-through rate (CTR) prediction, which aims to predict the probability of a user clicking on an ad or an item, is critical to many online applications such as online advertising and recommender systems. The problem is very challenging since (1) the input features (e.g., the user id, user age, item id, item category) are usually sparse and high-dimensional, and (2) an effective prediction relies on high-order combinatorial features (\textit{a.k.a.} cross features), which are very time-consuming to hand-craft by domain experts and are impossible to be enumerated. Therefore, there have been efforts in finding low-dimensional representations of the sparse and high-dimensional raw features and their meaningful combinations. In this paper, we propose an effective and efficient method called the \emph{AutoInt} to automatically learn the high-order feature interactions of input features. Our proposed algorithm is very general, which can be applied to both numerical and categorical input features. Specifically, we map both the numerical and categorical features into the same low-dimensional space. Afterwards, a multi-head self-attentive neural network with residual connections is proposed to explicitly model the feature interactions in the low-dimensional space. With different layers of the multi-head self-attentive neural networks, different orders of feature combinations of input features can be modeled. The whole model can be efficiently fit on large-scale raw data in an end-to-end fashion. Experimental results on four real-world datasets show that our proposed approach not only outperforms existing state-of-the-art approaches for prediction but also offers good explainability. Code is available at: \url{[this https URL](https://github.com/DeepGraphLearning/RecommenderSystems)}.
> 

:::

## Architecture

![Untitled](/img/content-models-raw-mp1-autoint-untitled.png)

## Links

- [https://github.com/DaPenggg/AutoInt](https://github.com/DaPenggg/AutoInt)
- [https://github.com/shichence/AutoInt](https://github.com/shichence/AutoInt)
- [https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.autoint.html](https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.autoint.html)
- [https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.autoint.html](https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.autoint.html)
- [https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.autoint.html](https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.autoint.html)