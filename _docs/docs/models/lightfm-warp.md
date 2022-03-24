# LightFM WARP

LightFM is probably the only recommender package implementing the WARP (Weighted Approximate-Rank Pairwise) loss for implicit feedback learning-to-rank. Generally, it performs better than the more popular BPR (Bayesian Personalised Ranking) loss --- often by a large margin.

It was originally applied to image annotations in the Weston et al. [WSABIE paper](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf), but has been extended to apply to recommendation settings in the [2013 k-order statistic loss paper](http://www.ee.columbia.edu/~ronw/pubs/recsys2013-kaos.pdf) in the form of the k-OS WARP loss, also implemented in LightFM.

Like the BPR model, WARP deals with (user, positive item, negative item) triplets. Unlike BPR, the negative items in the triplet are not chosen by random sampling: they are chosen from among those negative items which would violate the desired item ranking given the state of the model. This approximates a form of active learning where the model selects those triplets that it cannot currently rank correctly.

This procedure yields roughly the following algorithm:

1. For a given (user, positive item pair), sample a negative item at random from all the remaining items. Compute predictions for both items; if the negative item's prediction exceeds that of the positive item plus a margin, perform a gradient update to rank the positive item higher and the negative item lower. If there is no rank violation, continue sampling negative items until a violation is found.
2. If you found a violating negative example at the first try, make a large gradient update: this indicates that a lot of negative items are ranked higher than positives items given the current state of the model, and the model must be updated by a large amount. If it took a lot of sampling to find a violating example, perform a small update: the model is likely close to the optimum and should be updated at a low rate.

While this is fairly hand-wavy, it should give the correct intuition. For more details, read the paper itself or a more in-depth blog post [here](https://building-babylon.net/2016/03/18/warp-loss-for-implicit-feedback-recommendation/). A similar approach for BPR is described in Rendle's 2014 [WSDM 2014 paper](http://webia.lip6.fr/~gallinar/gallinari/uploads/Teaching/WSDM2014-rendle.pdf).