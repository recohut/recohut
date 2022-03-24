# Spotify Contextual Bandits

[Read more in this paper](https://dl.acm.org/doi/10.1145/3240323.3240354)

Spotify using contextual bandits to identify the best recommendation explanation (aka “recsplanations”) for users. The problem was how to jointly personalize music recommendations with their associated explanation, where the reward is user engagement on the recommendation. Contextual features include user region, product, and platform of the user device, user listening history (genres, playlist), etc.

An initial approach involved using logistic regression to predict user engagement from a recsplanation, given data about the recommendation, explanation, and user context. However, for logistic regression, the recsplanation that maximized reward was the same regardless of the user context.

To address this, they introduced higher-order interactions between recommendation, explanation, and user context, first by embedding them, and then introducing inner products on the embeddings (i.e., 2nd-order interactions). Then, the 2nd-order interactions are combined with first-order variables via a weighted sum, making it a 2nd-order factorization machine. They tried both 2nd and 3rd order factorization machines. (For more details of factorization machines in recommendations, see figure 2 and the “FM Component” section in the [DeepFM](https://arxiv.org/abs/1703.04247) paper).

To train their model, they adopt sample reweighting to account for the non-uniform probability of recommendations in production. (They didn’t have the benefit of uniform random samples like in the Netflix example.) During the offline evaluation, the 3rd-order factorization machine performed best. During online evaluation (i.e., A/B test), both 2nd and 3rd-order factorization machines did better than logistic regression and the baseline. Nonetheless, there was no significant difference between the 2nd and 3rd-order models.