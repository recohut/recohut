# CASER

CASER stands for Convolutional Sequence Embedding Recommendation. Top-N sequential recommendation models each user as a sequence of items interacted in the past and aims to predict top-N ranked items that a user will likely interact in a 'near future'. The order of interaction implies that sequential patterns play an important role where more recent items in a sequence have a larger impact on the next item. Convolutional Sequence Embedding Recommendation Model (Caser) address this requirement by embedding a sequence of recent items into an image' in the time and latent spaces and learn sequential patterns as local features of the image using convolutional filters. This approach provides a unified and flexible network structure for capturing both general preferences and sequential patterns. In other words, Caser adopts convolutional neural networks capture the dynamic pattern influences of users’ recent activities. 

:::info research paper

[Jiaxi Tang and Ke Wang, “*Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding*” in WSDM 2018.](https://arxiv.org/abs/1809.07426)

> Top-N sequential recommendation models each user as a sequence of items interacted in the past and aims to predict top-N ranked items that a user will likely interact in a near future. The order of interaction implies that sequential patterns play an important role where more recent items in a sequence have a larger impact on the next item. In this paper, we propose a Convolutional Sequence Embedding Recommendation Model (Caser) as a solution to address this requirement. The idea is to embed a sequence of recent items into an image in the time and latent spaces and learn sequential patterns as local features of the image using convolutional filters. This approach provides a unified and flexible network structure for capturing both general preferences and sequential patterns. The experiments on public datasets demonstrated that Caser consistently outperforms state-of-the-art sequential recommendation methods on a variety of common evaluation metrics.
> 

:::

## Architecture

The main component of Caser consists of a horizontal convolutional network and a vertical convolutional network, aiming to uncover the union-level and point-level sequence patterns, respectively. Point-level pattern indicates the impact of single item in the historical sequence on the target item, while union level pattern implies the influences of several previous actions on the subsequent target. For example, buying both milk and butter together leads to higher probability of buying flour than just buying one of them. Moreover, users’ general interests, or long term preferences are also modeled in the last fully-connected layers, resulting in a more comprehensive modeling of user interests.

![/img/content-models-raw-mp1-caser-untitled.png](/img/content-models-raw-mp1-caser-untitled.png)

## Links

- [PyTorch Implementation](https://github.com/graytowne/caser_pytorch)
- [D2L Tutorial Chapter](http://d2l.ai/chapter_recommender-systems/seqrec.html)
- [https://recbole.io/docs/recbole/recbole.model.sequential_recommender.caser.html](https://recbole.io/docs/recbole/recbole.model.sequential_recommender.caser.html)