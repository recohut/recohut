# VSKNN

VSKNN stands for Vector Multiplication Session-Based kNN. The idea of this variant is to put more emphasis on the more recent events of a session when computing the similarities. Instead of encoding a session as a binary vector, we use real-valued vectors to encode the current session. Only the very last element of the session obtains a value of “1”; the weights of the other elements are determined using a linear decay function that depends on the position of the element within the session, where elements appearing earlier in the session obtain a lower weight. As a result, when using the dot product as a similarity function between the current weight-encoded session and a binary-encoded past session, more emphasis is given to elements that appear later in the sessions.

:::info research paper

[L. Malte and J. Dietmar, “*Evaluation of Session-based Recommendation Algorithms*”. arXiv, 2018.](https://arxiv.org/abs/1803.09587)

> Recommender systems help users find relevant items of interest, for example on e-commerce or media streaming sites. Most academic research is concerned with approaches that personalize the recommendations according to long-term user profiles. In many real-world applications, however, such long-term profiles often do not exist and recommendations therefore have to be made solely based on the observed behavior of a user during an ongoing session. Given the high practical relevance of the problem, an increased interest in this problem can be observed in recent years, leading to a number of proposals for session-based recommendation algorithms that typically aim to predict the user's immediate next actions. In this work, we present the results of an in-depth performance comparison of a number of such algorithms, using a variety of datasets and evaluation measures. Our comparison includes the most recent approaches based on recurrent neural networks like GRU4REC, factorized Markov model approaches such as FISM or FOSSIL, as well as simpler methods based, e.g., on nearest neighbor schemes. Our experiments reveal that algorithms of this latter class, despite their sometimes almost trivial nature, often perform equally well or significantly better than today's more complex approaches based on deep neural networks. Our results therefore suggest that there is substantial room for improvement regarding the development of more sophisticated session-based recommendation algorithms.
> 

:::

## References

- [https://arxiv.org/pdf/1803.09587.pdf](https://arxiv.org/pdf/1803.09587.pdf)
- [https://github.com/mmaher22/iCV-SBR/tree/master/Source Codes/VSKNN_Python](https://github.com/mmaher22/iCV-SBR/tree/master/Source%20Codes/VSKNN_Python)