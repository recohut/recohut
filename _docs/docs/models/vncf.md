# VNCF

VNCF stands for Variational Neural Collaborative Filtering. 

:::info research paper

[Bobadilla et. al., “*Deep Variational Models for Collaborative Filtering-based Recommender Systems*”. arXiv, 2021.](https://arxiv.org/abs/2107.12677v1)

> Deep learning provides accurate collaborative filtering models to improve recommender system results. Deep matrix factorization and their related collaborative neural networks are the state-of-art in the field; nevertheless, both models lack the necessary stochasticity to create the robust, continuous, and structured latent spaces that variational autoencoders exhibit. On the other hand, data augmentation through variational autoencoder does not provide accurate results in the collaborative filtering field due to the high sparsity of recommender systems. Our proposed models apply the variational concept to inject stochasticity in the latent space of the deep architecture, introducing the variational technique in the neural collaborative filtering field. This method does not depend on the particular model used to generate the latent representation. In this way, this approach can be applied as a plugin to any current and future specific models. The proposed models have been tested using four representative open datasets, three different quality measures, and state-of-art baselines. The results show the superiority of the proposed approach in scenarios where the variational enrichment exceeds the injected noise effect. Additionally, a framework is provided to enable the reproducibility of the conducted experiments.
> 

:::

## Architecture

![Proposed VDeepMF architecture. The NCF architecture will have identical ‘Embedding’ and ‘Variational’ layers to the VDeepMF one; it will just replace the ‘Dot’ layer for a ‘Concatenate’ layer, followed by an MLP.](/img/content-models-raw-mp1-vncf-untitled.png)

Proposed VDeepMF architecture. The NCF architecture will have identical ‘Embedding’ and ‘Variational’ layers to the VDeepMF one; it will just replace the ‘Dot’ layer for a ‘Concatenate’ layer, followed by an MLP.