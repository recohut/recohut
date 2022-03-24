# DeepFM

DeepFM stands for Deep Factorization Machines. It consists of an FM component and a deep component which are integrated in a parallel structure. The FM component is the same as the 2-way factorization machines which is used to model the low-order feature interactions. The deep component is a multi-layered perceptron that is used to capture high-order feature interactions and nonlinearities. These two components share the same inputs/embeddings and their outputs are summed up as the final prediction.

:::info research paper

[Huifeng Guo et al. “*DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*” in IJCAI 2017.](https://arxiv.org/abs/1703.04247)

> Learning sophisticated feature interactions behind user behaviors is critical in maximizing CTR for recommender systems. Despite great progress, existing methods seem to have a strong bias towards low- or high-order interactions, or require expertise feature engineering. In this paper, we show that it is possible to derive an end-to-end learning model that emphasizes both low- and high-order feature interactions. The proposed model, DeepFM, combines the power of factorization machines for recommendation and deep learning for feature learning in a new neural network architecture. Compared to the latest Wide \& Deep model from Google, DeepFM has a shared input to its "wide" and "deep" parts, with no need of feature engineering besides raw features. Comprehensive experiments are conducted to demonstrate the effectiveness and efficiency of DeepFM over the existing models for CTR prediction, on both benchmark data and commercial data.
> 

:::

It is worth pointing out that the spirit of DeepFM resembles that of the Wide & Deep architecture which can capture both memorization and generalization. The advantages of DeepFM over the Wide & Deep model is that it reduces the effort of hand-crafted feature engineering by identifying feature combinations automatically.

## Architecture

![/img/content-models-raw-mp1-deepfm-untitled.png](/img/content-models-raw-mp1-deepfm-untitled.png)

## Links

- [https://arxiv.org/abs/1703.04247](https://arxiv.org/abs/1703.04247)
- [https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.deepfm.html](https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.deepfm.html)
- [https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.deepfm.html](https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.deepfm.html)
- [http://d2l.ai/chapter_recommender-systems/deepfm.html](http://d2l.ai/chapter_recommender-systems/deepfm.html)
- [https://github.com/PaddlePaddle/PaddleRec/tree/release/2.1.0/models/rank/deepfm](https://github.com/PaddlePaddle/PaddleRec/tree/release/2.1.0/models/rank/deepfm)