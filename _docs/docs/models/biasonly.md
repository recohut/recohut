# BiasOnly

BiasOnly is a simple baseline that assumes no interactions between users and items. Formally, it learns: (1) a global bias 𝛼; (2) scalar biases $\beta_u$ for each user 𝑢 ∈ U; and (3) scalar biases $\beta_i$ for each item 𝑖 ∈ I. Ultimately, the rating/relevance for user 𝑢 and item 𝑖 is modeled as $\hat{r}_i^u = \alpha + \beta_u + \beta_i$.