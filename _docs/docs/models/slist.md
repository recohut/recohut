# SLIST

SLIST stands for Session-aware Linear Similarity/Transition. It is built by unifying two linear models - SLIS and SLIT.

### Session-aware Linear Item Similarity Model (SLIS)

In this model, we use a linear model using full session representation, focusing on the similarity between items. We reformulated the objective function of SLIM to accommodate the timeliness of sessions and repeated item consumption in the session.

### Session-aware Linear Item Transition Model (SLIT)

Using the partial session representation, we design a linear model that captures the sequential dependency across items. Unlike SLIS, each session is split into multiple partial sessions, forming different input and output matrices. Similar to SLIS, we also incorporate the weight of sessions to SLIT. Meanwhile, we ignore the constraint for diagonal elements as different input and output matrices are naturally free from the trivial solution.

### Unifying Two Linear Models

Since SLIS and SLIT capture various characteristics of sessions, we propose a unified model, called Session-aware Linear Similarity/Transition model (SLIST), by jointly optimizing both models:

$$
\argmin_B \alpha||W_{full} \odot (X-X.B)||_F^2 + (1-\alpha)||W_{par}\odot(T-S.B)||_F^2 + \lambda||B||_F^2
$$