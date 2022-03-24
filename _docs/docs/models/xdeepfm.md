# xDeepFM

xDeepFM stands for Extreme Deep Factorization Machines.

:::info research paper

[Jianxun Lian at al. “*xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems*” in SIGKDD 2018.](https://arxiv.org/abs/1803.05170)

> Combinatorial features are essential for the success of many commercial models. Manually crafting these features usually comes with high cost due to the variety, volume and velocity of raw data in web-scale systems. Factorization based models, which measure interactions in terms of vector product, can learn patterns of combinatorial features automatically and generalize to unseen features as well. With the great success of deep neural networks (DNNs) in various fields, recently researchers have proposed several DNN-based factorization model to learn both low- and high-order feature interactions. Despite the powerful ability of learning an arbitrary function from data, plain DNNs generate feature interactions implicitly and at the bit-wise level. In this paper, we propose a novel Compressed Interaction Network (CIN), which aims to generate feature interactions in an explicit fashion and at the vector-wise level. We show that the CIN share some functionalities with convolutional neural networks (CNNs) and recurrent neural networks (RNNs). We further combine a CIN and a classical DNN into one unified model, and named this new model eXtreme Deep Factorization Machine (xDeepFM). On one hand, the xDeepFM is able to learn certain bounded-degree feature interactions explicitly; on the other hand, it can learn arbitrary low- and high-order feature interactions implicitly. We conduct comprehensive experiments on three real-world datasets. Our results demonstrate that xDeepFM outperforms state-of-the-art models.
> 

:::

## Architecture

![The architecture of xDeepFM.](/img/content-models-raw-mp1-xdeepfm-untitled.png)

The architecture of xDeepFM.

## Compressed Interaction Network (CIN)

![Components and architecture of the Compressed Interaction Network (CIN).](/img/content-models-raw-mp1-xdeepfm-untitled-1.png)

Components and architecture of the Compressed Interaction Network (CIN).

## Links

- [https://arxiv.org/abs/1803.05170](https://arxiv.org/abs/1803.05170)
- [https://dl.acm.org/doi/pdf/10.1145/3219819.3220023](https://dl.acm.org/doi/pdf/10.1145/3219819.3220023)
- [https://github.com/Leavingseason/xDeepFM](https://github.com/Leavingseason/xDeepFM)
- [https://github.com/shenweichen/DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)
- [https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.xdeepfm.html](https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.xdeepfm.html)
- [https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.xdeepfm.html](https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.xdeepfm.html)
- [https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/models/rank/xdeepfm](https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/models/rank/xdeepfm)
- [https://towardsdatascience.com/extreme-deep-factorization-machine-xdeepfm-1ba180a6de78](https://towardsdatascience.com/extreme-deep-factorization-machine-xdeepfm-1ba180a6de78)