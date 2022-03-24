# SVAE

SVAE stands for Sequential Variational Autoencoder. 

:::info research paper

[Noveen Sachdeva, Giuseppe Manco, Ettore Ritacco and Vikram Pudi, “*Sequential Variational Autoencoders for Collaborative Filtering*”. WSDM, 2019.](https://arxiv.org/abs/1811.09975)

> Variational autoencoders were proven successful in domains such as computer vision and speech processing. Their adoption for modeling user preferences is still unexplored, although recently it is starting to gain attention in the current literature. In this work, we propose a model which extends variational autoencoders by exploiting the rich information present in the past preference history. We introduce a recurrent version of the VAE, where instead of passing a subset of the whole history regardless of temporal dependencies, we rather pass the consumption sequence subset through a recurrent neural network. At each time-step of the RNN, the sequence is fed through a series of fully-connected layers, the output of which models the probability distribution of the most likely future preferences. We show that handling temporal information is crucial for improving the accuracy of the VAE: In fact, our model beats the current state-of-the-art by valuable margins because of its ability to capture temporal dependencies among the user-consumption sequence using the recurrent encoder still keeping the fundamentals of variational autoencoders intact.
> 

:::

## Links

- [https://github.com/khanhnamle1994/MetaRec/tree/master/Autoencoders-Experiments/SVAE-PyTorch](https://github.com/khanhnamle1994/MetaRec/tree/master/Autoencoders-Experiments/SVAE-PyTorch)
- [https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Sequential-Variational-Autoencoders-for-Collaborative-Filtering.pdf](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Sequential-Variational-Autoencoders-for-Collaborative-Filtering.pdf)