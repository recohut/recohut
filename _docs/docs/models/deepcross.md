# DeepCross

:::info research paper

[Shan, Y., Hoens, T., Jiao, J., Wang, H., Yu, D. and Mao, J., 2016. *Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features*. [online] Kdd.org.](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)

> Manually crafted combinatorial features have been the “secret sauce” behind many successful models. For web-scale applications, however, the variety and volume of features make these manually crafted features expensive to create, maintain, and deploy. This paper proposes the Deep Crossing model which is a deep neural network that automatically combines features to produce superior models. The input of Deep Crossing is a set of individual features that can be either dense or sparse. The important crossing features are discovered implicitly by the networks, which are comprised of an embedding and stacking layer, as well as a cascade of Residual Units.
> 

> Deep Crossing is implemented with a modeling tool called the Computational Network Toolkit (CNTK), powered by a multi-GPU platform. It was able to build, from scratch, two web-scale models for a major paid search engine, and achieve superior results with only a subset of the features used in the production models. This demonstrates the potential of using Deep Crossing as a general modeling paradigm to improve existing products, as well as to speed up the development of new models with a fraction of the investment in feature engineering and acquisition of deep domain knowledge.
> 

:::

## Architecture

![Untitled](/img/content-models-raw-mp1-deepcross-untitled.png)