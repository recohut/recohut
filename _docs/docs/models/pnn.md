# PNN

PNN stands for Product-based Neural Network.

:::info research paper

[Qu Y et al. “Product-based neural networks for user response prediction.” in ICDM 2016.](https://arxiv.org/abs/1611.00144)

> Predicting user responses, such as clicks and conversions, is of great importance and has found its usage in many Web applications including recommender systems, web search and online advertising. The data in those applications is mostly categorical and contains multiple fields; a typical representation is to transform it into a high-dimensional sparse binary feature representation via one-hot encoding. Facing with the extreme sparsity, traditional models may limit their capacity of mining shallow patterns from the data, i.e. low-order feature combinations. Deep models like deep neural networks, on the other hand, cannot be directly applied for the high-dimensional input because of the huge feature space. In this paper, we propose a Product-based Neural Networks (PNN) with an embedding layer to learn a distributed representation of the categorical data, a product layer to capture interactive patterns between inter-field categories, and further fully connected layers to explore high-order feature interactions. Our experimental results on two large-scale real-world ad click datasets demonstrate that PNNs consistently outperform the state-of-the-art models on various metrics.
> 

:::

## Architecture

![Untitled](/img/content-models-raw-mp1-pnn-untitled.png)

## Links

- [https://arxiv.org/abs/1611.00144](https://arxiv.org/abs/1611.00144)
- [https://github.com/rixwew/pytorch-fm/blob/master/torchfm/model/pnn.py](https://github.com/rixwew/pytorch-fm/blob/master/torchfm/model/pnn.py)
- [https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/pnn.py](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/pnn.py)
- [https://github.com/Atomu2014/product-nets/blob/master/python/models.py](https://github.com/Atomu2014/product-nets/blob/master/python/models.py)
- [https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.pnn.html](https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.pnn.html)
- [https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rank/pnn/](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rank/pnn/)