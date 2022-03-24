# Wide and Deep

Wide and Deep Learning Model, proposed by Google, 2016, is a DNN-Linear mixed model, which combines the strength of memorization and generalization. It's useful for generic large-scale regression and classification problems with sparse input features (e.g., categorical features with a large number of possible feature values). It has been used for Google App Store for their app recommendation.

:::info research paper

[Cheng et. al., *Wide & Deep Learning for Recommender Systems*. RecSys, 2016.](https://arxiv.org/abs/1606.07792)

> Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. In this paper, we present Wide & Deep learning---jointly trained wide linear models and deep neural networks---to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow.
> 

:::

## Architecture

![Untitled](/img/content-models-raw-mp1-wide-and-deep-untitled.png)

To understand the concept of deep & wide recommendations, it’s best to think of it as two separate, but collaborating, engines. The wide model, often referred to in the literature as the linear model, memorizes users and their past product choices. Its inputs may consist simply of a user identifier and a product identifier, though other attributes relevant to the pattern (such as time of day) may also be incorporated.

![/img/content-models-raw-mp1-wide-and-deep-untitled-1.png](/img/content-models-raw-mp1-wide-and-deep-untitled-1.png)

The deep portion of the model, so named as it is a deep neural network, examines the generalizable attributes of a user and their product choices. From these, the model learns the broader characteristics that tend to favor users’ product selections.

Together, the wide and deep submodels are trained on historical product selections by individual users to predict future product selections. The end result is a single model capable of calculating the probability with which a user will purchase a given item, given both memorized past choices and generalizations about a user’s preferences. These probabilities form the basis for user-specific product rankings, which can be used for making recommendations.

The goal with wide and deep recommenders is to provide the same level of customer intimacy that, for example, our favorite barista does. This model uses explicit and implicit feedback to expand the considerations set for customers. Wide and deep recommenders go beyond simple weighted averaging of customer feedback found in some collaborative filters to balance what is understood about the individual with what is known about similar customers. If done properly, the recommendations make the customer feel understood and this should translate into greater value for both the customer and the business.

<iframe width="727" height="409" src="https://www.youtube.com/embed/Xmw9SWJ0L50" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The intuitive logic of the wide-and-deep recommender belies the complexity of its actual construction. Inputs must be defined separately for each of the wide and deep portions of the model and each must be trained in a coordinated manner to arrive at a single output, but tuned using optimizers specific to the nature of each submodel. Thankfully, the **[Tensorflow DNNLinearCombinedClassifier estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier)** provides a pre-packaged architecture, greatly simplifying the assembly of the overall model.

## Links

- [https://github.com/intel-analytics/analytics-zoo/tree/master/apps/recommendation-wide-n-deep](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/recommendation-wide-n-deep)
- [https://dl.acm.org/doi/pdf/10.1145/2988450.2988454](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454)
- [https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.widedeep.html](https://recbole.io/docs/recbole/recbole.model.context_aware_recommender.widedeep.html)
- [https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.wdl.html](https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.wdl.html)
- [https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.wdl.html](https://deepctr-torch.readthedocs.io/en/latest/deepctr_torch.models.wdl.html)
- [https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/models/rank/wide_deep](https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/models/rank/wide_deep)
- [https://docs.databricks.com/applications/machine-learning/reference-solutions/recommender-wide-n-deep.html](https://docs.databricks.com/applications/machine-learning/reference-solutions/recommender-wide-n-deep.html)
- [https://medium.com/analytics-vidhya/wide-deep-learning-for-recommender-systems-dc99094fc291](https://medium.com/analytics-vidhya/wide-deep-learning-for-recommender-systems-dc99094fc291)
- [https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)