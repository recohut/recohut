# NFM

NFM stands for Neural Factorization Machine.

:::info research paper

[He, X., & Chua, T.-S. (2017, August 16). *Neural factorization machines for sparse predictive analytics*. arXiv.org.](https://arxiv.org/abs/1708.05027)

> In this paper, we propose a novel model Neural Factorization Machine (NFM) for prediction under sparse settings. NFM seamlessly combines the linearity of FM in modelling second-order feature interactions and the non-linearity of neural network in modelling higher-order feature interactions. Conceptually, NFM is more expressive than FM since FM can be seen as a special case of NFM without hidden layers. Empirical results on two regression tasks show that with one hidden layer only, NFM significantly outperforms FM with a 7.3% relative improvement. Compared to the recent deep learning methods Wide&Deep and DeepCross, our NFM uses a shallower structure but offers better performance, being much easier to train and tune in practice.
> 

:::

## Architecture

![https://github.com/recohut/reco-static/raw/master/media/diagrams/nfm.svg](https://github.com/recohut/reco-static/raw/master/media/diagrams/nfm.svg)

Bi-interaction Pooling:

$$
f_{BI}(\mathcal{V}_x) = \dfrac{1}{2}\left[ (\sum_{i=1}^n x_iv_i)^2 - \sum_{i=1}^n(x_iv_i)^2 \right]
$$

## Links

- [https://arxiv.org/abs/1708.05027](https://arxiv.org/abs/1708.05027)
- [https://github.com/hexiangnan/neural_factorization_machine](https://github.com/hexiangnan/neural_factorization_machine)
- [https://dl.acm.org/doi/pdf/10.1145/3077136.3080777](https://dl.acm.org/doi/pdf/10.1145/3077136.3080777)
- [https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.nfm.html](https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.nfm.html)
- [https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.nfm.html](https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.nfm.html)
- [https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.nfm.html](https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.nfm.html)
- [https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rank/nfm/](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rank/nfm/)