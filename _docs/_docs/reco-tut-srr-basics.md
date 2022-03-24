---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    name: python3
---

```python id="GKUkECKocMyy"
import pandas as pd
import numpy as np
```

```python colab={"base_uri": "https://localhost:8080/", "height": 142} id="tRPsLl0KcW2K" outputId="037baffb-7c12-40eb-aa31-b770263b459b"
pd.DataFrame ([['Ivan', 'Borodinsky Bread', 1],
               ['Ivan', 'White Bread', 0],
               ['Vasily', 'Epica Yogurt', 1]],
              columns = ['user', 'item', 'purchase_fact'])
```

<!-- #region id="NRtDLdT1cwB0" -->
Wait, you can add features to user (average bill, number of purchases in categories, etc.), to item (price, number of sales per week, etc.), and solve the classification problem. What is the difference between RecSys and classification?

- Many predictions for 1 user (extreme classification)
- Much larger amount of data: 100K users, 10K items -> 1B predictions
- Most of the products the user has never seen -> Did not interact -> 0 does not mean "did not like"
- There is no explicit target. It is not clear what it means "(not) liked"
- Feedback loop
- The order of recommendations is always important
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="pwMi6yLkc2Ra" outputId="2645c117-a458-43f9-90c6-fc907005250e"
pd.DataFrame([['Ivan','BreadBorodinsky',1],
              ['Ivan','WhiteBread',0],
              ['Ivan','EpicaYogurt',"?"],
              ['Vasily','BorodinskyBread',"?"],
              ['Vasily','WhiteBread',"?"],
              ['Vasily','EpicaYogurt',1]],
             columns = ['user','item','purchase_fact'])
```

<!-- #region id="PSyjFtONc88L" -->
The main types of tasks: 
- Recommended top-K products : Given a list of products. Recommend K products to the user that they like
    - e-mail newsletter
    - push notifications
    - Recommendations in a separate window on the site
    - Ranking of goods : Given a list of goods. You need to rank it in descending order of interest for the user
- Ranking of the product catalog
    - Ranking feed
    - Search engine ranking
    - Ranking of "carousels" of goods
- Search for similar products : Given 1 product. You need to find the most similar products
    - "You May Also Like"
    - Similar users liked
    - You may be familiar
- Additional product recommendation . Given 1 product. Find products that are buying with this product
    - Frequently bought with this product
<!-- #endregion -->
