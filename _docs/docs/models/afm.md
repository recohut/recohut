# AFM

AFM stands for Attentional Factorization Machines. It Improves FM by discriminating the importance of different feature interactions, and learns the importance of each feature interaction from data via a neural attention network. Empirically, it is shown on regression task that AFM performs betters than FM with a 8.6% relative improvement, and consistently outperforms the state-of-the-art deep learning methods Wide&Deep and DeepCross with a much simpler structure and fewer model parameters.

:::info research paper

[Jun Xiao et al. “*Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks*” in IJCAI 2017.](https://arxiv.org/abs/1708.04617)

> Factorization Machines (FMs) are a supervised learning approach that enhances the linear regression model by incorporating the second-order feature interactions. Despite effectiveness, FM can be hindered by its modelling of all feature interactions with the same weight, as not all feature interactions are equally useful and predictive. For example, the interactions with useless features may even introduce noises and adversely degrade the performance. In this work, we improve FM by discriminating the importance of different feature interactions. We propose a novel model named Attentional Factorization Machine (AFM), which learns the importance of each feature interaction from data via a neural attention network. Extensive experiments on two real-world datasets demonstrate the effectiveness of AFM. Empirically, it is shown on regression task AFM betters FM with a 8.6% relative improvement, and consistently outperforms the state-of-the-art deep learning methods Wide&Deep and DeepCross with a much simpler structure and fewer model parameters.
> 

:::

## Architecture

![Untitled](/img/content-models-raw-mp1-afm-untitled.png)

Formally, the AFM model can be defined as:

$$
\hat{y}_{AFM} (x) = w_0 + \sum_{i=1}^nw_ix_i + p^T\sum_{i=1}^n\sum_{j=i+1}^na_{ij}(v_i\odot v_j)x_ix_j
$$

## Links

- [https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.afm.html#afm](https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.afm.html#afm)
- [https://github.com/hexiangnan/attentional_factorization_machine](https://github.com/hexiangnan/attentional_factorization_machine)
- [https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.afm.html](https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.afm.html)
- [https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.afm.html](https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.afm.html)