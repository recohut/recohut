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

<!-- #region id="O_VupUUIr0ff" -->
# The importance of Rating Normalization
> Understanding the concept of rating normalization and user-based, item-based similarity with example

- toc: true
- badges: true
- comments: true
- categories: [Concept, Preprocessing]
- image:
<!-- #endregion -->

```python id="zj1_hePrjSqk"
import numpy as np
import pandas as pd
```

```python id="6BHnJCSoo5vz"
!wget http://static.preferred.ai/tutorials/recommender-systems/sample_data.csv
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="ukcj5vtbobGB" outputId="df4b1469-24c9-4ba6-dad4-2e6fe705fcc2"
df = pd.read_csv("sample_data.csv", sep=",", names=["UserID", "ItemID", "Rating"])
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="e02SnEywobef" outputId="f05d444b-ae05-41d4-ee67-02b001f0bba1"
df = pd.pivot_table(df, 'Rating', 'UserID', 'ItemID')
df["Mean Rating"] = df.mean(axis=1)
df
```

<!-- #region id="gtr81Hdbpu8R" -->
One concern about rating data is its subjectivity. In particular, different users may use different ranges. Some users are lenient and tend to assign higher ratings. Others are strict and tend to assign lower ratings. A commonly adopted approach to 'normalize' the ratings is to take the mean of the ratings by a user and subtract the mean from the individual ratings of the said user.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="YAdx2Uc4oebI" outputId="617aa205-adde-4f88-e56b-bc9e8dff8733"
ratings = df[df.columns.difference(["Mean Rating"])].values
means = df["Mean Rating"].values[:, np.newaxis]
df[df.columns.difference(["Mean Rating"])] = (ratings - means)
df.drop(columns="Mean Rating")
```

<!-- #region id="AuML2S-MqSZK" -->
### User-based method
<!-- #endregion -->

<!-- #region id="9_O9E_DMqYEu" -->
For each user, mean rating is calculated as follows:

$$ \mu_u = \frac{\Sigma_{k \in \mathcal{I}_u} r_{uk}}{|\mathcal{I}_u|} \ \ \forall u \in \{1 \dots m\} $$


Two common approaches to measure similarity between two users $\mathrm{Sim}(u, v)$ are *Cosine similarity* and *Pearson correlation coefficient*:

\begin{align*}
\mathrm{Cosine}(u,v) &= \frac{\Sigma_{k \in \mathcal{I}_u \cap \mathcal{I}_v} r_{uk} * r_{vk}}{\sqrt{\Sigma_{k \in \mathcal{I}_u \cap \mathcal{I}_v} r_{uk}^2} * \sqrt{\Sigma_{k \in \mathcal{I}_u \cap \mathcal{I}_v} r_{vk}^2}} \\
\mathrm{Pearson}(u,v) &= \frac{\Sigma_{k \in \mathcal{I}_u \cap \mathcal{I}_v} (r_{uk} - \mu_u) * (r_{vk} - \mu_v)}{\sqrt{\Sigma_{k \in \mathcal{I}_u \cap \mathcal{I}_v} (r_{uk} - \mu_u)^2} * \sqrt{\Sigma_{k \in \mathcal{I}_u \cap \mathcal{I}_v} (r_{vk} - \mu_v)^2}}
\end{align*}


For example, given the original rating matrix, between *User 1* and *User 3* we have their similarities as:

\begin{align*}
\mathrm{Cosine}(1,3) &= \frac{6*3+7*3+4*1+5*1}{\sqrt{6^2+7^2+4^2+5^2} * \sqrt{3^2+3^2+1^2+1^2}} = 0.956 \\
\mathrm{Pearson}(1,3) &= \frac{(6 - 5.5) * (3 - 2) + (7 - 5.5) * (3 - 2) + (4 - 5.5) * (1 - 2) + (5 - 5.5) * (1 - 2)}{\sqrt{0.5^2 + 1.5^2 + (-1.5)^2 + (-0.5)^2} * \sqrt{1^2 + 1^2 + (-1)^2 + (-1)^2}} = 0.894
\end{align*}
<!-- #endregion -->

<!-- #region id="-kDjrZOxq5DQ" -->
The overall neighborhood-based *prediction function* is as follows:

$$ \hat{r}_{uj} = \mu_u + \frac{\Sigma_{v \in P_u(j)} \mathrm{Sim}(u,v) * (r_{vj} - \mu_v)}{\Sigma_{v \in P_u(j)} |\mathrm{Sim}(u,v)|} $$


For example, to calculate the predicted rating given by *User 3* to *Item 1* and *Item 6*, where the ratings are based on the two nearest neighbors (*User 1* and *User 2*):

\begin{align*}
\hat{r}_{31} &= 2 + \frac{1.5*0.894+1.2*0.939}{0.894 + 0.939} = 3.35 \\
\hat{r}_{36} &= 2 + \frac{-1.5*0.894-0.8*0.939}{0.894 + 0.939} = 0.86
\end{align*}
<!-- #endregion -->

<!-- #region id="CnuPWmy_q9H5" -->
### Item-based method
<!-- #endregion -->

<!-- #region id="Ang7mHyqq_Bh" -->
The *Cosine* and *Pearson* similarities can be applied for item-based methods as well, except that the feature vectors are now columns instead of rows as we measure similarity between items. 

If *Cosine* similarity is based on the mean-centered rating matrix, we have a variant called *AdjustedCosine*.  The *adjusted* cosine similarity between the items (columns) *i* and *j* is defined as follows:

$$ \mathrm{AdjustedCosine}(i,j) = \frac{\Sigma_{u \in \mathcal{U}_i \cap \mathcal{U}_j} s_{ui} * s_{uj}}{\sqrt{\Sigma_{u \in \mathcal{U}_i \cap \mathcal{U}_j} s_{ui}^2} * \sqrt{\Sigma_{u \in \mathcal{U}_i \cap \mathcal{U}_j} s_{uj}^2}} $$

where $s_{ui}$ is the mean-centered rating that user $u$ gives to item $i$. 

For example, we calculate *adjusted* cosine between *Item 1* and *Item 3* in the small sample dataset above as follows:

$$ \mathrm{AdjustedCosine}(1,3) = \frac{1.5 * 1.5 + (-1.5) * (-0.5) + (-1) * (-1)}{\sqrt{1.5^2 + (-1.5)^2 + (-1)^2} * \sqrt{1.5^2 + (-0.5)^2 + (-1)^2}} = 0.912 $$
<!-- #endregion -->

<!-- #region id="_o-xjaj_rUm4" -->
For prediction, we use the same form of prediction function as in user-based methods but aggregate the user's ratings on neighboring items:

$$ \hat{r}_{ut} = \mu_u + \frac{\Sigma_{j \in Q_t(u)} \mathrm{Sim}(j,t) * (r_{uj} - \mu_u)}{\Sigma_{j \in Q_t(u)} |\mathrm{Sim}(j,t)|} $$


For example, below we predict the ratings that *User 3* would give to *Item 1* and *Item 6*. The rating for *Item 1* is based on two nearest neighbors *Item 2* and *Item 3*, while the rating for *Item 6* is based on *Item 4* and *Item 5*.

\begin{align*}
\hat{r}_{31} &= 2 + \frac{1*0.735 + 1*0.912}{0.735 + 0.912} = 3 \\
\hat{r}_{36} &= 2 + \frac{(-1)*0.829 + (-1)*0.730}{0.829 + 0.730} = 1
\end{align*}
<!-- #endregion -->
