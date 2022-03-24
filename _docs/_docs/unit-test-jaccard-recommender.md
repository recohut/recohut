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

<!-- #region id="5NR6O-NIyXvO" -->
# Unit Test - Jaccard Similarity based Recommender
> This test verifies the functionality of JaccardRecommender

- toc: true
- badges: true
- comments: true
- categories: [JaccardModel, UnitTest, Concept]
- image:
<!-- #endregion -->

```python id="RsZQz_Usw06d" executionInfo={"status": "ok", "timestamp": 1625725064242, "user_tz": -330, "elapsed": 521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import sys
import pathlib
import numpy as np
from scipy.sparse.csc import csc_matrix
```

<!-- #region id="LT8i_1Zxw1uF" -->
## Base Recommender
<!-- #endregion -->

```python id="Y7X4Ulcwwfzo" executionInfo={"status": "ok", "timestamp": 1625725038309, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class Recommender(object):
    def __init__(self):
        raise NotImplementedError

    def recommend(self, userid=None, username=None, N=10):
        raise NotImplementedError

    def recommend_all(self, userids, num_recs, **kwargs):
        recs = {}
        with tqdm(total=len(userids), leave=True) as progress:
            for u in userids:
                recs[u] = self.recommend(userid=u, N=num_recs, **kwargs)
                progress.update(1)

        return recs
```

<!-- #region id="ksGAr1gPw3bp" -->
## Jaccard Recommender
<!-- #endregion -->

```python id="UD-h1S5ZwqIs" executionInfo={"status": "ok", "timestamp": 1625725051482, "user_tz": -330, "elapsed": 797, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class JaccardRecommender(Recommender):
    def __init__(self, implicit_matrix, p2i, t2i, i2t, i2p, n2i, u2i, i2u):
        self.implicit_matrix = implicit_matrix
        self.p2i = p2i
        self.t2i = t2i
        self.i2t = i2t
        self.i2p = i2p
        self.n2i = n2i
        self.i2p = i2p
        self.u2i = u2i
        self.i2u = i2u

    def jaccard_multiple(self, page_indices, exclude_index=None):
        X = self.implicit_matrix.astype(bool).astype(int)
        if exclude_index is None:
            intrsct = X.dot(X[page_indices, :].T)
            totals = X[page_indices, :].sum(axis=1).T + X.sum(axis=1)
        else:
            use_indices = np.full(X.shape[1], True)
            use_indices[exclude_index] = False
            # print(X[:, use_indices].shape)
            # print(X[page_indices, :][:, use_indices].T.shape)

            intrsct = X[:, use_indices].dot(X[page_indices, :][:, use_indices].T)
            totals = X[page_indices, :][:, use_indices].sum(axis=1).T + X[
                :, use_indices
            ].sum(axis=1)

        return intrsct / (totals - intrsct)

    def recommend(
        self,
        N=10,
        userid=None,
        user=None,
        num_lookpage_pages=None,
        recent_pages_dict=None,
        interactions=None,
    ):
        if user is not None:
            user_index = self.n2i[user]
        elif userid is not None:
            user_index = self.u2i[userid]
        else:
            raise ValueError("Either user or userid must be non-null")

        recent_pages = recent_pages_dict[self.i2u[user_index]][:num_lookpage_pages]

        user_page_indices = [self.p2i[p] for p in recent_pages]
        d = self.jaccard_multiple(user_page_indices, exclude_index=user_index)

        d = np.nan_to_num(d)
        d[d == 1] = np.nan

        mean_jaccard = np.nanmean(d, axis=1).A.squeeze()
        order = np.argsort(mean_jaccard)[::-1]
        return [self.i2p[o] for o in order[:N]]

    def item_to_item(self, N=10, title=None, pageid=None):
        if title is not None:
            page_index = self.t2i.get(title, None)
        elif pageid is not None:
            page_index = self.p2i.get(pageid, None)
        else:
            raise ValueError("Either title or pageid must be non-null")

        if page_index is None:
            raise ValueError(
                "Page {} not found".format(pageid if title is None else title)
            )

        target_page_editors = np.flatnonzero(
            self.implicit_matrix[page_index, :].toarray()
        )
        # print("target_page_editors {}".format(target_page_editors))

        num_target_editors = len(target_page_editors)

        edited_indices = np.flatnonzero(
            np.sum(self.implicit_matrix[:, target_page_editors] > 0, axis=1)
        )

        # print("edited_indices {}".format(edited_indices))

        num_shared_editors = np.asarray(
            np.sum(self.implicit_matrix[:, target_page_editors] > 0, axis=1)[
                edited_indices
            ]
        ).squeeze()

        # print("num_shared_editors {}".format(num_shared_editors))

        num_item_editors = np.asarray(
            np.sum(self.implicit_matrix[edited_indices, :] > 0, axis=1)
        ).squeeze()

        # print("num_item_editors {}".format(num_item_editors))
        # print("Type num_item_editors {}".format(type(num_item_editors)))
        # print("num_item_editors dims {}".format(num_item_editors.shape))

        jaccard_scores = (
            num_shared_editors.astype(float)
            / ((num_target_editors + num_item_editors) - num_shared_editors)
        ).squeeze()

        # print("jaccard_scores {}".format(jaccard_scores))

        sorted_order = np.argsort(jaccard_scores)
        sorted_order = sorted_order.squeeze()

        rec_indices = edited_indices.squeeze()[sorted_order][::-1]
        sorted_scores = jaccard_scores.squeeze()[sorted_order][::-1]
        sorted_num_shared_editors = num_shared_editors.squeeze()[sorted_order][::-1]
        sorted_num_item_editors = num_item_editors.squeeze()[sorted_order][::-1]

        if title is None:
            return list(
                zip(
                    [self.i2p[i] for i in rec_indices[:N]],
                    sorted_scores[:N],
                    sorted_num_shared_editors[:N],
                    sorted_num_item_editors[:N],
                )
            )
        else:
            return list(
                zip(
                    [self.i2t[i] for i in rec_indices[:N]],
                    sorted_scores[:N],
                    sorted_num_shared_editors[:N],
                    sorted_num_item_editors[:N],
                )
            )
```

<!-- #region id="jQJKv6O9yS1o" -->
## Test data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="tj2e22n4xF3P" executionInfo={"status": "ok", "timestamp": 1625725248341, "user_tz": -330, "elapsed": 591, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="588f1118-cc30-4d47-98b8-a7831c0d089f"
implicit_matrix = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 0]])
assert implicit_matrix.shape == (3, 4)
print(implicit_matrix)
implicit_matrix = csc_matrix(implicit_matrix)
```

```python id="2N58ZOLXxKIa" executionInfo={"status": "ok", "timestamp": 1625725119369, "user_tz": -330, "elapsed": 511, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
n2i = {"huey": 0, "dewey": 1, "louie": 2, "chewy": 3}
t2i = {"Batman": 0, "Mystery Men": 1, "Taxi Driver": 2}
i2n = {v: k for k, v in n2i.items()}
i2t = {v: k for k, v in t2i.items()}
```

<!-- #region id="KCiz9b0oyUtn" -->
## Test
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dV28uFTRwWzp" executionInfo={"status": "ok", "timestamp": 1625725182156, "user_tz": -330, "elapsed": 559, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f18046bf-d3af-4d39-ec02-aac6e8bb693d"
jrec = JaccardRecommender(implicit_matrix, p2i=None, t2i=t2i, i2t=i2t, i2p=None, n2i=None, u2i=None, i2u=None)
print(jrec.item_to_item(N=10, title="Batman"))
```
