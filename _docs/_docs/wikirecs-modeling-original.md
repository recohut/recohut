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

```python id="vVKXpNyW0lq9"
!mv /content/reco-wikirecs/wikirecs.parquet.gz /content
```

```python id="Jep4VQyz3ZzE"
project_name="reco-wikirecs"; branch="master"; account="sparsh-ai"
```

```python id="ailHP5gi3ZzP" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625769898023, "user_tz": -330, "elapsed": 6373, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2f2a562b-cd52-426d-f6e7-b8db0592c49a"
!cp /content/drive/MyDrive/mykeys.py /content
import mykeys
!rm /content/mykeys.py
path = "/content/" + project_name; 
!mkdir "{path}"
%cd "{path}"
import sys; sys.path.append(path)
!git config --global user.email "sparsh@recohut.com"
!git config --global user.name  "colab-sparsh"
!git init
!git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
!git pull origin "{branch}"
```

```python id="WWDDXhuK9klF"
%cd /content/reco-wikirecs/
```

```python id="6HVnZkVW3ZzQ"
# !git status
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="LLMOakVK7lZg"
!pip install -r requirements.txt
```

<!-- #region id="wlWx6OrY3n_A" -->
---
<!-- #endregion -->

<!-- #region id="kiO7Fk7khazs" -->
## Setup
<!-- #endregion -->

```python id="wXZfzk8rTYX6"
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import itertools
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, coo_matrix

from recochef.datasets.wikirecs import WikiRecs

import implicit
from implicit.nearest_neighbours import bm25_weight
from implicit.nearest_neighbours import BM25Recommender

from utils import *
from wiki_pull import *
from recommenders import *
```

```python id="X1XNTud2orfP"
%matplotlib inline
%load_ext autoreload
%autoreload 2
```

<!-- #region id="2GtDLqdKsx1d" -->
## Loading artifacts
<!-- #endregion -->

```python id="cX7sQzl_nNx3"
p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t = load_pickle('lookup_tables.pickle')
userids, pageids = load_pickle('users_and_pages.pickle')
resurface_userids, discovery_userids = load_pickle('resurface_discovery_users.pickle')
implicit_matrix = load_pickle('implicit_matrix.pickle')
```

<!-- #region id="WIFZ5dHX-WM7" -->
## Test the matrix and indices
<!-- #endregion -->

```python id="MoCSRFwB-WM8" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625774716932, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9d0b7776-12b9-47cc-ca4b-d3ecb1a32ad7"
# Crude item to item recs by looking for items edited by the same editors (count how many editors overlap)
veditors = np.flatnonzero(implicit_matrix[t2i['Hamburger'],:].toarray())
indices =  np.flatnonzero(np.sum(implicit_matrix[:,veditors] > 0,axis=1))
totals = np.asarray(np.sum(implicit_matrix[:,veditors] > 0 ,axis=1)[indices])
sorted_order = np.argsort(totals.squeeze())

[i2t.get(i, "")  + " " + str(total[0]) for i,total in zip(indices[sorted_order],totals[sorted_order])][::-1][:10]
```

<!-- #region id="Jes7kvme-WNA" -->
# Implicit recommendation
<!-- #endregion -->

```python id="cupY14em-WNB"
implicit_matrix = load_pickle('implicit_matrix_2021-05-28.pickle')
p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t = load_pickle('lookup_tables_2021-05-28.pickle')
```

```python id="uspK0UVm-WNC"
bm25_matrix = bm25_weight(implicit_matrix, K1=100, B=0.25)
```

```python colab={"referenced_widgets": ["cebcfca877334675b349e9ff7d2fc5e8"]} id="vlglmBoU-WNC" outputId="30fd780e-fbbb-4c34-df66-cca88af7e87a"
num_factors =200
regularization = 0.01
os.environ["OPENBLAS_NUM_THREADS"] = "1"
model = implicit.als.AlternatingLeastSquares(
    factors=num_factors, regularization=regularization
)
model.fit(bm25_matrix)
```

```python id="dTMCTh4M-WND"
save_pickle(model,'als%d_bm25_model.pickle' % num_factors)
```

```python id="73spHZWZ-WNE"
model = wr.load_pickle('als200_bm25_model_2021-05-28.pickle')
```

```python id="10iE-M4_-WNE" outputId="e1459056-6875-437e-f3fb-7545638a073c"
results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]
```

```python id="InMxOoWV-WNF"
u = n2u["Rama"]
recommendations = model.recommend(u2i[u], bm25_matrix.tocsc(), N=1000, filter_already_liked_items=False)
[ ("*" if implicit_matrix[ind,u2i[u]]>0 else "") +
'%s %.4f' % (i2t[ind], score) + ' %d' % (implicit_matrix[ind,:]>0).sum()
 for ind, score in recommendations]
```

<!-- #region id="aa4arDRu-WNG" -->
## Grid search results
<!-- #endregion -->

```python id="lK9Tf-P9-WNH"
grid_search_results = wr.load_pickle("implicit_grid_search.pickle")
```

```python id="AXBN4kyT-WNI" outputId="be7bace9-f0ca-4bad-d5c0-70edb53e7b34"
pd.DataFrame(grid_search_results)
```


```python id="lYyee96H-WNP" outputId="6b9209b8-c9dc-438b-89d1-71c6be1f4d87"
pd.DataFrame([[i['num_factors'], i['regularization']] + list(i['metrics'].values()) for i in grid_search_results],
            columns = ['num_factors','regularization'] + list(grid_search_results[0]['metrics'].keys()))
```


```python id="HWKMB1RJ-WNS"
grid_search_results_bm25 = wr.load_pickle("implicit_grid_search_bm25.pickle")
```

```python id="sMb1ZJs2-WNT" outputId="d25ce83f-1991-4646-f5cd-515a4708e717"
pd.DataFrame([[i['num_factors'], i['regularization']] + list(i['metrics'].values()) for i in grid_search_results_bm25],
            columns = ['num_factors','regularization'] + list(grid_search_results_bm25[0]['metrics'].keys()))
```


<!-- #region id="qOsMziXy-WNY" -->
# B25 Recommendation
<!-- #endregion -->

```python id="hfTCx693-WNa" outputId="066ccff1-c1ce-4bbf-d97c-9b0c939a427c"
bm25_matrix = bm25_weight(implicit_matrix, K1=20, B=1)
bm25_matrix = bm25_matrix.tocsc()
sns.distplot(implicit_matrix[implicit_matrix.nonzero()],bins = np.arange(0,100,1),kde=False)

sns.distplot(bm25_matrix[bm25_matrix.nonzero()],bins = np.arange(0,100,1),kde=False)
```

```python id="-JZzKCom-WNa" colab={"referenced_widgets": ["789272e0be0841eab56c5e97832d6835"]} outputId="4561da2a-4398-4fd1-e30c-801d13c9527e"
K1 = 100
B = 0.25
model = BM25Recommender(K1, B)
model.fit(implicit_matrix)
```

```python id="iv6cnXyq-WNc"
save_pickle(model, 'bm25_model_2021-05-28.pkl')
```

```python id="36LmWlj8-WNc" outputId="15e84255-ce93-45a0-bbc2-0de481f508a6"
results = model.similar_items(t2i['Mark Hamill'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]
```

```python id="XL_iVwTX-WNd"
a = ['Steven Universe 429.4746',
 'List of Steven Universe episodes 178.4544',
 'Demon Bear 128.7237',
 'Legion of Super Heroes (TV series) 128.7237',
 'The Amazing World of Gumball 126.3522',
 'Steven Universe Future 123.9198']
```

```python id="5X5v4q6_-WNf" outputId="a67e0b49-4159-4ac6-e9c8-c6efaf8520f5"
results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]
```

```python id="LUl3lyS2-WNg" outputId="6a98ddb5-b997-4e5c-c5ad-491483da90c1"
results = model.similar_items(t2i['George Clooney'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]
```

```python id="aqpRMPPM-WNh" outputId="252b0efb-6888-445f-d81f-4babde73617c"
results = model.similar_items(t2i['Hamburger'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]
```

```python id="_u_TUsaj-WNj"
u = n2u["Rama"]
recommendations = model.recommend(u2i[u], implicit_matrix.astype(np.float32), N=1000, filter_already_liked_items=True)
```

```python id="Je7p-9V6-WNj"
[ ("*" if implicit_matrix[ind,u2i[u]]>0 else "") +
'%s %.4f' % (i2t[ind], score) 
 for ind, score in recommendations]
```

```python id="1eJyEd-F-WNl" outputId="372b5937-8b5b-4a1a-aae6-3363082d89e6"
plt.plot([ score for i,(ind, score) in enumerate(recommendations) if implicit_matrix[ind,u2i[u]]==0])
```

```python id="XSqgR7U3-WNm"
save_pickle(model, "b25_model.pickle")
```

```python id="wxMXiyqK-WNm"
model = load_pickle("b25_model.pickle")
```

<!-- #region id="wkIrfRRE-WNn" -->
# Evaluate models
<!-- #endregion -->

<!-- #region id="rJRxYGWJ-WNn" -->
## Item to item recommendation
<!-- #endregion -->

```python id="9V5YDLX_-WNp" outputId="07d757cd-26ae-4c3c-b396-4f301f26e860"
results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]
```

<!-- #region id="_lMIJmyg-WNp" -->
## User to item recommendations
<!-- #endregion -->

```python id="aedWb0g6-WNq" outputId="dd8b14e3-e741-40a7-ab08-6dd64b69e8e4"
# Check out a specific example

u = n2u["HyprMarc"]

print_user_history(clean_histories, userid=u)
```

```python id="SIEYvwMo-WNr"
u = n2u["HyprMarc"]
recommendations = model.recommend(u2i[u], implicit_matrix, N=100, filter_already_liked_items=False)
```

```python id="TIQI9XQ2-WNs"
[ ("*" if implicit_matrix[ind,u2i[u]]>0 else "") +
'%s %.4f' % (i2t[ind], score) 
 for ind, score in recommendations]
```

<!-- #region id="lWed-iyq-WNt" -->
# Visualize implicit embeddings
<!-- #endregion -->

```python id="C3UMYj97-WNt"
model = load_pickle('als150_model.pickle')
```

```python id="KqNxkJux-WNt"
# Only plot the ones with over 3 entries
indices = np.squeeze(np.asarray(np.sum(implicit_matrix[nonzero,:],axis=1))) > 3

indices = nonzero[indices]
```

```python id="8-xWlVsz-WNu" outputId="71022d2f-78b4-444f-a3c2-93defa98ca10"
len(indices)
```

```python id="MgZRXk5S-WNv"
# Visualize  the collaborative filtering item vectors, embedding into 2D space with UMAP
# nonzero = np.flatnonzero(implicit_matrix.sum(axis=1))
# indices = nonzero[::100]
embedding = umap.UMAP().fit_transform(model.item_factors[indices,:])
```

```python id="jYG7yAZ8-WNv" outputId="23e273fa-62d6-4854-8ead-75f1fbbf59f1"
plt.figure(figsize=(10,10))
plt.plot(embedding[:,0], embedding[:,1],'.')
# _ = plt.axis('square')
```

<!-- #region id="DdxY9hWV-WNw" -->
## Visualize actors in the embeddings space
<!-- #endregion -->

```python id="Vjv6octg-WNx"
edit_counts = np.squeeze(np.asarray(np.sum(implicit_matrix[indices,:],axis=1)))
log_edit_counts = np.log10(np.squeeze(np.asarray(np.sum(implicit_matrix[indices,:],axis=1))))

emb_df = pd.DataFrame({'dim1':embedding[:,0].squeeze(), 
                       'dim2':embedding[:,1].squeeze(),
                       'title':[i2t[i] for i in indices],
                       'edit_count':edit_counts,
                       'log_edit_count':log_edit_counts
                       })
```

```python id="hKQjcZ5p-WNx"
actors = ['Mark Hamill',
'Carrie Fisher',
'James Earl Jones',
'David Prowse',
'Sebastian Shaw (actor)',
'Alec Guinness',
'Jake Lloyd',
'Hayden Christensen',
'Ewan McGregor',
'William Shatner',
'Leonard Nimoy',
'DeForest Kelley',
'James Doohan',
'George Takei']
actor_indices = [t2i[a] for a in actors]
edit_counts = np.squeeze(np.asarray(np.sum(implicit_matrix[actor_indices,:],axis=1)))
log_edit_counts = np.log10(np.squeeze(np.asarray(np.sum(implicit_matrix[actor_indices,:],axis=1))))
embedding = umap.UMAP().fit_transform(model.item_factors[actor_indices,:])
emb_df = pd.DataFrame({'dim1':embedding[:,0].squeeze(), 
                       'dim2':embedding[:,1].squeeze(),
                       'title':[i2t[i] for i in actor_indices],
                       'edit_count':edit_counts,
                       'log_edit_count':log_edit_counts
                       })
key = np.zeros(len(actors))
key[:8] = 1
fig = px.scatter(data_frame=emb_df,
                 x='dim1',
                 y='dim2', 
                 hover_name='title',
                 color=key,
                 hover_data=['edit_count'])
fig.update_layout(
    autosize=False,
    width=600,
    height=600,)
fig.show()
```

```python id="8duesyS8-WNz"
# Full embedding plotly interactive visualization

emb_df = pd.DataFrame({'dim1':embedding[:,0].squeeze(), 
                       'dim2':embedding[:,1].squeeze(),
                       'title':[i2t[i] for i in indices],
                       'edit_count':edit_counts,
                       'log_edit_count':log_edit_counts
                       })

fig = px.scatter(data_frame=emb_df,
                 x='dim1',
                 y='dim2', 
                 hover_name='title',
                 color='log_edit_count',
                 hover_data=['edit_count'])
fig.update_layout(
    autosize=False,
    width=600,
    height=600,)
fig.show()
```

<!-- #region id="1JEnaCOD-WNz" -->
# Evaluate on test set
<!-- #endregion -->

```python id="1c17JTrC-WN0"
# Load the edit histories in the training set and the test set
histories_train = feather.read_feather('histories_train_2021-05-28.feather')
histories_test = feather.read_feather('histories_test_2021-05-28.feather')
histories_dev = feather.read_feather('histories_dev_2021-05-28.feather')

implicit_matrix = wr.load_pickle('implicit_matrix_2021-05-28.pickle')
p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t = wr.load_pickle('lookup_tables_2021-05-28.pickle')

userids, pageids = wr.load_pickle('users_and_pages_2021-05-28.pickle')

resurface_userids, discovery_userids   = wr.load_pickle('resurface_discovery_users_2021-05-28.pickle')

results = {}
```

```python id="zCvRQx01-WN0" outputId="2ae68511-595c-406a-eaf2-0aa8c339355b"
display_recs_with_history(
    recs,
    userids[:100],
    histories_test,
    histories_train,
    p2t,
    u2n,
    recs_to_display=5,
    hist_to_display=10,
)
```

```python id="mD5Cbidmyfn2"
import utils as wr
import numpy as np
from tqdm.auto import tqdm
import itertools
import pandas as pd
from implicit.nearest_neighbours import BM25Recommender


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


class PopularityRecommender(Recommender):
    def __init__(self, interactions):
        with wr.Timer("Building popularity table"):
            self.editors_per_page = (
                interactions.drop_duplicates(subset=["TITLE", "USER"])
                .groupby(["ITEMID", "TITLE"])
                .count()
                .USER.sort_values(ascending=False)
            )

    def recommend(self, N=10, userid=None, user=None):
        return self.editors_per_page.iloc[:N].index.get_level_values(0).values


class MostRecentRecommender(Recommender):
    """
    Recommend the most recently edited pages by the user in reverse chronological
    order. When those run out, go to most popular
    """

    def __init__(self, interactions):
        with wr.Timer("Building popularity table"):
            self.editors_per_page = (
                interactions.drop_duplicates(subset=["TITLE", "USER"])
                .groupby(["ITEMID", "TITLE"])
                .count()
                .USER.sort_values(ascending=False)
            )

    def all_recent_only(self, N=10, userids=None, interactions=None):
        recents = {}

        with tqdm(total=len(userids), leave=True) as progress:
            for u in userids:
                is_user_row = interactions.USERID == u
                recents[u] = (
                    interactions[is_user_row]
                    .drop_duplicates(subset=["ITEMID"])
                    .iloc[:N]
                    .ITEMID.values
                )
                progress.update(1)
        return recents

    def recommend(self, N=10, userid=None, user=None, interactions=None):
        if user is not None:
            is_user_row = interactions.user == user
        elif userid is not None:
            is_user_row = interactions.USERID == userid
        else:
            raise ValueError("Either user or userid must be non-null")

        deduped_pages = interactions[is_user_row].drop_duplicates(subset=["pageid"])
        if len(deduped_pages) == 1:
            recs = []
        else:
            # Don't take the most recent, because this dataset strips out repeated instance
            recs = deduped_pages.iloc[1:N].pageid.values

        # If we've run out of recs, fill the rest with the most popular entries
        if len(recs) < N:
            recs = np.concatenate(
                [
                    recs,
                    self.editors_per_page.iloc[: (N - len(recs))]
                    .index.get_level_values(0)
                    .values,
                ]
            )
        return recs


class MostFrequentRecommender(Recommender):
    """
    Recommend the most frequently edited pages by the user. When those run out, go to most popular
    """

    def __init__(self, interactions):
        with wr.Timer("Building popularity table"):
            self.editors_per_page = (
                interactions.drop_duplicates(subset=["TITLE", "USER"])
                .groupby(["pageid", "title"])
                .count()
                .USER.sort_values(ascending=False)
            )

    def recommend(self, N=10, userid=None, user=None, interactions=None):
        if user is not None:
            is_user_row = interactions.USER == user
        elif userid is not None:
            is_user_row = interactions.ITEMID == userid
        else:
            raise ValueError("Either user or userid must be non-null")

        recs = (
            interactions[is_user_row]
            .groupby("ITEMID")
            .USER.count()
            .sort_values(ascending=False)
            .index[:N]
            .values
        )

        # If we've run out of recs, fill the rest with the most popular entries
        if len(recs) < N:
            recs = np.concatenate(
                [
                    recs,
                    self.editors_per_page.iloc[: (N - len(recs))]
                    .index.get_level_values(0)
                    .values,
                ]
            )
        return recs


class ImplicitCollaborativeRecommender(Recommender):
    def __init__(self, model, implicit_matrix):
        self.model = model
        self.implicit_matrix = implicit_matrix

    def recommend(
        self,
        N=10,
        userid=None,
        user=None,
        u2i=None,
        n2i=None,
        i2p=None,
        filter_already_liked_items=False,
    ):
        if user is not None:
            user_index = n2i[user]
        elif userid is not None:
            user_index = u2i[userid]
        else:
            raise ValueError("Either user or userid must be non-null")

        recs_indices = self.model.recommend(
            user_index,
            self.implicit_matrix,
            N,
            filter_already_liked_items=filter_already_liked_items,
        )
        recs = [i2p[a[0]] for a in recs_indices]

        return recs

    def recommend_all(self, userids, num_recs, i2p, filter_already_liked_items=True):
        all_recs = self.model.recommend_all(
            self.implicit_matrix.T,
            num_recs,
            filter_already_liked_items=filter_already_liked_items,
        )
        recs = {
            userid: [i2p[i] for i in all_recs[i, :]] for i, userid in enumerate(userids)
        }

        return recs


class MyBM25Recommender(Recommender):
    def __init__(self, model, implicit_matrix):
        self.model = model

        self.implicit_matrix = implicit_matrix

    def recommend(
        self,
        N=10,
        filter_already_liked_items=True,
        userid=None,
        user=None,
        u2i=None,
        n2i=None,
        i2p=None,
    ):
        if user is not None:
            user_index = n2i[user]
        elif userid is not None:
            user_index = u2i[userid]
        else:
            raise ValueError("Either user or userid must be non-null")

        recs_indices = self.model.recommend(
            user_index,
            self.implicit_matrix.astype(np.float32),
            N,
            filter_already_liked_items=filter_already_liked_items,
        )
        recs = [i2p[a[0]] for a in recs_indices]

        if len(recs) <= 20:
            recs = recs + [recs[-1]] * (20 - len(recs))

        return recs


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


class InterleaveRecommender(Recommender):
    """
    Recommend for users by interleaving recs from multiple lists. When there is
    duplicates keeping only the first instance.
    """

    def __init__(self):
        pass

    def recommend_all(self, N=10, recs_list=[]):
        """
        Args:
            N (int): Number of recs to return
            recs_list: Array of recs, which are ordered lists of pageids in a dict keyed by a userid
        Returns:
            dict: Recommendations, as a list of pageids keyed by userid
        """

        def merge_page_lists(page_lists):
            return pd.unique(list(itertools.chain(*zip(*page_lists))))

        return {
            userid: merge_page_lists([recs.get(userid, []) for recs in recs_list])[:N]
            for userid in recs_list[0]
        }
```

```python id="9IDWPu-50FJB"
results = {}
```

<!-- #region id="elAIUYRp-WN2" -->
## Most popular
<!-- #endregion -->

```python id="YJKENiZx-WN2" colab={"base_uri": "https://localhost:8080/", "height": 134, "referenced_widgets": ["b9b52d73a3c14c22b1bf5e0d230bd2b5", "759abc1c3319406cbc5a74b476c37632", "50080ceae7e8446eb9dda1fd6f6dd2e2", "df20fec7e023489cac17b37e20c90039", "83764018203549919318661441b0f118", "6636044ffbcb49dc8f4acfb4b069384e", "c60409de1e0f4bb282082247f35dcded", "ebea5d3221194ff28c4462b694ad428b"]} executionInfo={"status": "ok", "timestamp": 1625776140792, "user_tz": -330, "elapsed": 22230, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cde13f3e-bf97-43d7-9543-b76bf6b33495"
%%time
K=20
rec_name = "Popularity"

prec = PopularityRecommender(histories_train)
precs = prec.recommend_all(userids, K)
# wr.save_pickle(precs, "../" + rec_name +"_recs.pickle")
```


```python id="f1GZYGkQ-WN3" outputId="45f1831d-0fcf-4b8d-ed5f-49fba497b568"
results[rec_name] = get_recs_metrics(histories_dev, precs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)

results[rec_name]
```


<!-- #region id="XXwTr-cr-WN3" -->
## Most recent
<!-- #endregion -->

```python id="gcRTnre8-WN4" colab={"referenced_widgets": ["d716ae7ee1874ee3a08b46a3a461e333"]} outputId="3dc7c2a3-8da0-4b0e-812d-a0ecfb24c369"
%%time
# Most recent
K=20
rrec = recommenders.MostRecentRecommender(histories_train)
rrecs = rrec.recommend_all(userids, K, interactions=histories_train)
rec_name = "Recent"
wr.save_pickle(rrecs, "../" + rec_name +"_recs.pickle")
```

```python id="d-t8VuIy-WN4" outputId="bc307379-2262-4b66-ec5c-009a81285fd3"
len(resurface_userids)
```

```python id="StZPGbXT-WN5"
results ={}
```

```python id="_a2xes5N-WN5" outputId="fd1da463-16f7-48c7-f842-0cd1eef7c490"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, rrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

<!-- #region id="3hHDox8J-WN7" -->
## Most frequent
<!-- #endregion -->

```python id="Cq2Nm6Dm-WN8" colab={"referenced_widgets": ["287b777493b948f2b0ce09f1ad79ea4d"]} outputId="435f5696-3120-4019-ba86-72da91b94e60"
%%time
# Sorted by frequency of edits
K=20
frec = recommenders.MostFrequentRecommender(histories_train)
frecs = frec.recommend_all(userids, K, interactions=histories_train)
rec_name = "Frequent"
wr.save_pickle(frecs, "../" + rec_name +"_recs.pickle")
```


```python id="GOG4_JVX-WN8" outputId="a2f5cdfd-3ba8-4a84-9f5f-06d276a464b8"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, frecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

<!-- #region id="g19lzJpJ-WN9" -->
## BM25
<!-- #endregion -->

```python id="tfHMxaCK-WN-"
%%time
K=20
brec = recommenders.MyBM25Recommender(model, implicit_matrix)
```

```python id="nSVV6hXJ-WN-" colab={"referenced_widgets": ["d3955e88957c46e18fe203a3227fb31d"]} outputId="d33c41bc-5d6d-4768-ab1c-5b45b52b2f83"
brecs = brec.recommend_all(userids, K, u2i=u2i, n2i=n2i, i2p=i2p, filter_already_liked_items=False)
rec_name = "bm25"
wr.save_pickle(brecs, "../" + rec_name +"_recs.pickle")
```

```python id="fICW4eHY-WN-" outputId="6c78d4bb-1173-4c57-e8bc-cb9827d1fc94"
# filter_already_liked_items = False
results[rec_name] = wr.get_recs_metrics(
    histories_dev, brecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

```python id="dLnQMU_k-WOA"
# filter_already_liked_items = True
rec_name = "bm25_filtered"
brecs_filtered = brec.recommend_all(userids, K, u2i=u2i, n2i=n2i, i2p=i2p, filter_already_liked_items=True)
wr.save_pickle(brecs_filtered, "../" + rec_name +"_recs.pickle")
```


```python id="r6ml95k3-WOA" outputId="ed54ace6-5f7a-479a-82a4-35eeb6d94ddb"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, recs['bm25_filtered'], K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

```python id="edOgJNpi-WOC" outputId="5377b292-9224-460c-c713-d77d697d3e2a"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, recs['bm25_filtered'], K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

<!-- #region id="72SeK4Ip-WOD" -->
## ALS Implicit collaborative filtering
<!-- #endregion -->

```python id="pTkjLsYp-WOD"
model_als = wr.load_pickle('../als200_bm25_model_2021-05-28.pickle')
```

```python id="MBdVKNcV-WOE" colab={"referenced_widgets": ["14ff8e978a09477e91aca95e1f3b28f2"]} outputId="0caa85df-989b-4eaf-a5fb-242857315be6"
%%time
rec_name = "als"
K=20
irec = recommenders.ImplicitCollaborativeRecommender(model_als, bm25_matrix.tocsc())
irecs = irec.recommend_all(userids, K, i2p=i2p, filter_already_liked_items=False)
wr.save_pickle(irecs, "../" + rec_name +"_recs.pickle")
```

```python id="LW_8B_2c-WOE" colab={"referenced_widgets": ["79ee3142902143588e76b1a03e7b2d86"]} outputId="68483952-1fe8-4ba8-fbe2-6248a16c2bd3"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, irecs, K, discovery_userids, resurface_userids, bm25_matrix.tocsc(), i2p, u2i)
results[rec_name]
```

```python id="ByYa78kc-WOF"
rec_name = "als_filtered"
K=20
irec = recommenders.ImplicitCollaborativeRecommender(model_als, bm25_matrix.tocsc())
irecs_filtered = irec.recommend_all(userids, K, i2p=i2p, filter_already_liked_items=True)
results[rec_name] = wr.get_recs_metrics(
    histories_dev, irecs_filtered, K, discovery_userids, resurface_userids, bm25_matrix.tocsc(), i2p, u2i)
results[rec_name]
```

```python id="meqpp6ZM-WOG"
wr.save_pickle(irecs_filtered, "../" + rec_name +"_recs.pickle")
```

```python id="vmw6tWlU-WOG" outputId="7bf4fffd-e822-4d9e-cd4d-07c673b67c92"
show(pd.DataFrame(results).T)
```

<!-- #region id="9OO8PATm-WOQ" -->
## Jaccard
<!-- #endregion -->

```python id="MFkof-Te-WOT" colab={"referenced_widgets": ["8612548500614215b3221bdcf6ee204f", "dbca2b89d81647ff9c4a910a574d0d40"]} outputId="f09bd895-701a-4ee6-c3ba-d3dd398bf5ec"
%%time
# Sorted by Jaccard
K=20
rrec = recommenders.MostRecentRecommender(histories_train)
recent_pages_dict = rrec.all_recent_only(K, userids,  interactions=histories_train)
jrec = recommenders.JaccardRecommender(implicit_matrix, p2i=p2i, t2i=t2i, i2t=i2t, i2p=i2p, n2i=n2i, u2i=u2i, i2u=i2u)
jrecs = jrec.recommend_all(userids, 
                                   K, 
                                   num_lookpage_pages=1, 
                                   recent_pages_dict=recent_pages_dict, 
                                   interactions=histories_train)
```

```python id="88o0wlQt-WOU"
wr.save_pickle(jrecs,"jaccard-1_recs.pickle")
```

```python id="MUmAmtsC-WOV" outputId="6ce910f4-1add-4cc7-d1d6-650b4f94493f"
rec_name = "Jaccard"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, jrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

```python id="g8iMXfQR-WOV" outputId="a625d11c-76a2-4eb1-d7c1-896f0ac7cad5"
wr.display_recs_with_history(
    jrecs,
    userids[:30],
    histories_test,
    histories_train,
    p2t,
    u2n,
    recs_to_display=5,
    hist_to_display=10,
)
```

```python id="YqhaNQSn-WOW" colab={"referenced_widgets": ["e8b374d98a2f401c9f48ff75e50b67e1"]} outputId="d05b74c4-98fb-420f-b0ce-67df1f1cc583"
%%time
# Sorted by Jaccard
K=5
jrec = recommenders.JaccardRecommender(implicit_matrix, p2i=p2i, t2i=t2i, i2t=i2t, i2p=i2p, n2i=n2i, u2i=u2i, i2u=i2u)
jrecs = jrec.recommend_all(userids[:1000], 
                                   10, 
                                   num_lookpage_pages=50, 
                                   recent_pages_dict=recent_pages_dict, 
                                   interactions=histories_train)
print("Jaccard")
```

```python id="KU6OEcZH-WOX" outputId="0f730a3b-a0d4-408f-d918-c19efff54733"
print("Recall @ %d: %.1f%%" % (K, 100*wr.recall(histories_test, jrecs, K)))
print("Prop resurfaced: %.1f%%" % (100*wr.prop_resurface(jrecs, K, implicit_matrix, i2p, u2i)))
print("Recall @ %d (discovery): %.1f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=discovery_userids)))
print("Recall @ %d (resurface): %.1f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=resurface_userids)))
```

<!-- #region id="C96Q99hE-WOX" -->
## Interleaved
<!-- #endregion -->

```python id="99l-aATC-WOY" outputId="686ff223-45b4-4ed1-b7a4-1f9d0e07b147"
recs.keys()
```

```python id="WEGzdF7j-WOY" outputId="d1bd7fa6-68bd-46da-96ec-10b32561b2d6"
# Interleaved jaccard and recent
K=20
rec_name = "Interleaved"
print(rec_name)
intrec = recommenders.InterleaveRecommender()
intrecs = intrec.recommend_all(K, [recs['Recent'], recs['bm25_filtered']])

wr.save_pickle(intrecs, "../" + rec_name +"_recs.pickle")
```

```python id="YnT1K_YB-WOY" outputId="8cd9a207-75fb-4d13-b016-60eb2fc49205"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, intrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

<!-- #region id="dr_p42yH-WOY" -->
# Report on evaluations results
<!-- #endregion -->

<!-- #region id="neVECcOt-WOZ" -->
## Hard coded metrics
<!-- #endregion -->

```python id="3HbF4BmQ-WOZ"
results = {}
results["Popularity"] = {'recall': 0.16187274312040842,
 'ndcg': 0.0005356797596941751,
 'resurfaced': 0.6213422985929523,
 'recall_discover': 0.11947959996459864,
 'recall_resurface': 0.2624396388830569,
 'ndcg_discover': 0.000410354483750028,
 'ndcg_resurface': 0.0008329819416998272}
results["Recent"] = {'recall': 22.618602913709378,
 'ndcg': 0.14306080818547054,
 'resurfaced': 71.13808990163118,
 'recall_discover': 0.03982653332153288,
 'recall_resurface': 76.18097837497375,
 'ndcg_discover': 0.00011494775493754298,
 'ndcg_resurface': 0.4821633227780786}
results["Frequent"] = {'recall': 20.834889802017184,
 'ndcg': 0.11356953338215306,
 'resurfaced': 76.10353629684971,
 'recall_discover': 0.035401362952473675,
 'recall_resurface': 70.17635943732941,
 'ndcg_discover': 9.90570471847343e-05,
 'ndcg_resurface': 0.38274923359395385}
results["ALS"] = {'recall': 5.488108579255385,
 'ndcg': 0.026193145556306998,
 'resurfaced': 16.251556468683848,
 'recall_discover': 1.146119125586335,
 'recall_resurface': 15.788368675204703,
 'ndcg_discover': 0.004817135435898367,
 'ndcg_resurface': 0.0769022655123215}
results["ALS_filtered"] = {'recall': 0.9027518366330469,
 'ndcg': 0.003856703716094881,
 'resurfaced': 0.0,
 'recall_discover': 1.2832994070271706,
 'recall_resurface': 0.0,
 'ndcg_discover': 0.005482465270193466,
 'ndcg_resurface': 0.0}
results["BM25"] = {'recall': 18.945336819823186,
 'ndcg': 0.1015175508656068,
 'resurfaced': 74.0469742248786,
 'recall_discover': 1.3939286662536507,
 'recall_resurface': 60.581566239764854,
 'ndcg_discover': 0.004204510293040833,
 'ndcg_resurface': 0.332367864833573}
results["BM25_filtered"] = {'recall': 1.8148424853691942,
 'ndcg': 0.008622285155255174,
 'resurfaced': 0.14848711243929774,
 'recall_discover': 2.522347110363749,
 'recall_resurface': 0.1364686122191896,
 'ndcg_discover': 0.011740495141426633,
 'ndcg_resurface': 0.0012251290280766518}
results["Interleaved"] = {'recall': 21.382766778732414,
 'ndcg': 0.12924273396038563,
 'resurfaced': 42.478676379031256,
 'recall_discover': 1.8364457031595716,
 'recall_resurface': 67.75141717404996,
 'ndcg_discover': 0.006943981897312752,
 'ndcg_resurface': 0.4193652616867473}
results_df = pd.DataFrame(results).T

results_df.reset_index(inplace=True)
```


<!-- #region id="dKS1jicv-WOZ" -->
## Table of results
<!-- #endregion -->

```python id="PgSRtbvp-WOa" outputId="56acb728-4559-4572-856d-14924062b711"
results_df
```

<!-- #region id="UoVyyHOz-WOa" -->
### FIG Table for post
<!-- #endregion -->

```python id="06tmkf3n-WOa"
def scatter_text(x, y, text_column, data, title, xlabel, ylabel):
    """Scatter plot with country codes on the x y coordinates
       Based on this answer: https://stackoverflow.com/a/54789170/2641825"""
    
    # Create the scatter plot
    p1 = sns.scatterplot(x, y, data=data, size = 8, legend=False)
    # Add text besides each point
    for line in range(0,data.shape[0]):
         p1.text(data[x][line]+0.01, data[y][line], 
                 data[text_column][line], horizontalalignment='left', 
                 size='medium', color='black', weight='semibold')
    # Set title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return p1


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]



results_df.sort_values("recall", ascending=False).style.apply(highlight_max, subset=["recall",
                                                                                    "ndcg",
                                                                                    "resurfaced",
                                                                                    "recall_discover",
                                                                                    "recall_resurface",
                                                                                    "ndcg_discover",
                                                                                    "ndcg_resurface",]).format({"recall": "{:.1f}%", 
                                             "ndcg": "{:.3f}",
                                             "resurfaced": "{:.1f}%", 
                                             "recall_discover": "{:.1f}%", 
                                              "recall_resurface": "{:.1f}%", 
                                            "ndcg_discover": "{:.3f}",
                                              "ndcg_resurface": "{:.3f}",
                                             })
```

```python id="cBtum3PP-WOb" outputId="981a08df-6523-4051-8e19-61e9dd11da77"
colnames = ["Recommender", "Recall@20", "nDCG@20","Resurfaced","Recall@20 discovery","Recall@20 resurface","nDCG@20 discovery","nDCG@20 resurface"]
#apply(highlight_max, subset=colnames[1:]).
results_df.columns = colnames
results_df.sort_values("Recall@20", ascending=False).style.\
    format({"Recall@20": "{:.1f}%", 
             "nDCG@20": "{:.3f}",
             "Resurfaced": "{:.1f}%", 
             "Recall@20 discovery": "{:.1f}%", 
             "Recall@20 resurface": "{:.1f}%", 
             "nDCG@20 discovery": "{:.3f}",
             "nDCG@20 resurface": "{:.3f}",
             })
```

<!-- #region id="omDGo1qC-WOb" -->
## Scatter plots (resurface vs discover)
<!-- #endregion -->

```python id="H4zugWVq-WOc" outputId="253e7f8e-483a-4265-e971-6f289595bf23"
fig = px.scatter(data_frame=results_df,
                 x='ndcg_discover',
                 y='ndcg_resurface',
                hover_name='index')
#                  hover_name='title',)
fig.show()
```

```python id="UFTr64Zc-WOc" outputId="c0edfddf-67b2-4cd3-d909-4533abc5b471"
fig = px.scatter(data_frame=results_df,
                 x='recall_discover',
                 y='recall_resurface',
                hover_name='index')
#                  hover_name='title',)
fig.show()
```

<!-- #region id="rvXW7Cpb-WOd" -->
### FIG Scatterplot for post
<!-- #endregion -->

```python id="ohEujuT7-WOd"
x = 2*[results_df.loc[results_df.Recommender == "Interleaved","Recall@20 resurface"].values[0]]
y = [0, results_df.loc[results_df.Recommender == "Interleaved","Recall@20 discovery"].values[0]]
```

```python id="bje7EUY7-WOd" outputId="6d57bb57-4359-429d-c8cb-06d4b6fe9ed2"
sns.set_theme(style="darkgrid")
matplotlib.rcParams.update({'font.size': 48, 'figure.figsize':(8,5), 'legend.edgecolor':'k'})


plt.figure(figsize=(12,7))
A = results_df.loc[:,'Recall@20 discovery']
B = results_df.loc[:,'Recall@20 resurface']

x = 2*[results_df.loc[results_df.Recommender == "Interleaved","Recall@20 discovery"].values[0]]
y = [-1, results_df.loc[results_df.Recommender == "Interleaved","Recall@20 resurface"].values[0]]
plt.plot(x,y,":k")
x[0] = 0
y[0] = y[1]
# plt.rcParams.update({'font.size': 48})
plt.rc('xtick', labelsize=3)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

plt.plot(x,y,":k")

plt.plot(A,B,'.', MarkerSize=15)


for xyz in zip(results_df.Recommender, A, B):                                       # <--
    plt.gca().annotate('%s' % xyz[0], xy=np.array(xyz[1:])+(0.05,0), textcoords='data', fontsize=18) # <--

for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in plt.gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(20)

plt.xlabel("Recall@20 discovery (%)",fontsize=20)
plt.ylabel("Recall@20 resurface (%)",fontsize=20)
plt.xlim([0,3])
plt.ylim([-2,85])
axes = plt.gca()
```

<!-- #region id="LcqUfW0Z-WOe" -->
## Read recs in from files
<!-- #endregion -->

```python id="vqDVP7RU-WOe"
recommender_names = ['Popularity', 'Recent', 'Frequent', 'ALS', 'ALS_filtered', 'BM25', 'BM25_filtered', 'Interleaved']
```

```python id="aAg8HSgl-WOe"
recs = {rname:wr.load_pickle("../" + rname + "_recs.pickle") for rname in recommender_names}
```

<!-- #region id="GEFTxZhy-WOf" -->
## Recall curves
<!-- #endregion -->

```python id="NcpKfVt4-WOf"
histories_dev = feather.read_feather('../histories_dev_2021-05-28.feather')
```

```python id="0uYSfuKV-WOf" outputId="e801a150-0328-421e-d4eb-c34312fc5555"
plt.figure(figsize=(15,10))
for rname in recommender_names:
    recall_curve = wr.recall_curve(histories_dev, recs[rname], 20)
#     print(recall_curve[-1])
    plt.plot(recall_curve,'.-')
plt.legend(recommender_names)
```

```python id="GQHyH7ez-WOg" outputId="794635c3-900b-455b-d88e-06dbd61c7067"
plt.figure(figsize=(15,10))
for rname in recommender_names:
    recall_curve = wr.recall_curve(histories_dev, recs[rname], 20, discovery_userids)
    plt.plot(recall_curve,'.-')
plt.legend(recommender_names)
```

```python id="S-j6WEnn-WOg" outputId="bb0dcb04-d7e8-490e-8feb-e2fe352b1cd9"
plt.figure(figsize=(15,10))
for rname in recommender_names:
    recall_curve = wr.recall_curve(histories_dev, recs[rname], 20, resurface_userids)
    plt.plot(recall_curve,'.-')
plt.legend(recommender_names)
```

<!-- #region id="mK-CVSCh-WOh" -->
### FIG Implicit vs BM25 figure
<!-- #endregion -->

```python id="GoMbt9vt-WOh" outputId="d556fe90-cf15-430a-fe17-e3389863ade2"
sns.set_theme(style="darkgrid")
matplotlib.rcParams.update({'font.size': 18, 'figure.figsize':(8,5), 'legend.edgecolor':'k'})
plt.figure(figsize=(10,6))
for rname in ["ALS","BM25"]:
    recall_curve = wr.recall_curve(histories_dev, recs[rname], 20, discovery_userids)
    plt.plot(np.array(recall_curve)*100,'.-',markersize=12)
plt.legend( ["ALS","BM25"],title="Algorithm", fontsize=16, title_fontsize=16, facecolor="w")
plt.xlabel("@N",fontsize=20)
plt.ylabel("Discovery recall (%)",fontsize=20)
_ = plt.xticks(np.arange(0,20,2),np.arange(0,20,2)+1)
# plt.gca().legend(prop=dict(size=20))
for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in plt.gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(20)
```


<!-- #region id="9ZY9R22s-WOh" -->
# User recommendation comparison
<!-- #endregion -->

```python id="q5qqTWaM-WOh"
recs_subset = ["Recent","Frequent","Popularity","Implicit","bm25","interleaved"]
```

```python id="wGBe0NhF-WOi" outputId="131bdc68-65a3-4dd3-c59b-4b362e824ead"
print("Next edit: " + histories_dev.loc[histories_dev.userid == userid].title.values[0])
```


<!-- #region id="iTS8NLfY-WOi" -->
## FIG Rama table
<!-- #endregion -->

```python id="XKjJCXbb-WOj"
def bold_viewed(val, viewed_pages):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    weight = 'bold' if val in  viewed_pages else 'normal'
    return 'font-weight: %s' % weight

def color_target(val, target_page):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val ==  target_page else 'black'
    return 'color: %s' % color

def display_user_recs_comparison(user_name, recs, recs_subset, train_set, test_set, N=20):
    userid = n2u[user_name]
    recs_table = pd.DataFrame({rec_name: [p2t[r] for r in recs[rec_name][userid][:N]] for rec_name in recs_subset})
    recs_table = recs_table.reset_index()
    recs_table.loc[:,"index"] = recs_table.loc[:,"index"]+1
    recs_table = recs_table.rename(columns={"index":""})
    viewed_pages = train_set.loc[train_set.userid == userid,["title"]].drop_duplicates(subset=["title"]).values.squeeze()
    target_page = test_set.loc[test_set.userid == userid].title.values[0]
#     print("Next edit: " + target_page)
    s = recs_table.style.applymap(bold_viewed, viewed_pages=viewed_pages).applymap(color_target, target_page=target_page)
    display(s)
```

```python id="Mf0T2iGr-WOj" outputId="d2bcd9bc-293c-4ce7-8440-58131b3d3791"
recs_subset = ["Recent","Frequent","Popularity","ALS","ALS_filtered","BM25","BM25_filtered"]

display_user_recs_comparison('Rama', recs, recs_subset, histories_train, histories_dev, N=10)
```

<!-- #region id="CP7azvNa-WOk" -->
## Other individuals tables
<!-- #endregion -->

```python id="i6n0sN0V-WOk" outputId="492dd490-55f6-43f0-e78c-19a688b1c224"
display_user_recs_comparison('Meow', recs, recs_subset, histories_train, histories_dev, N=10)
```

```python id="7Wa6EfVJ-WOl" outputId="9959aee4-4e8d-477d-891a-7c1f2fd45ee3"
display_user_recs_comparison('KingArti', recs, recs_subset, histories_train, histories_dev, N=10)
```

```python id="Z4U9tKmu-WOl" outputId="94127335-d727-4c3b-9dc0-dad0c0c709c6"
display_user_recs_comparison('Tulietto', recs, recs_subset, histories_train, histories_dev, N=10)
```

```python id="pWuk2vp2-WOn" outputId="fb913d5a-540d-4165-a495-e814c8900d45"
display_user_recs_comparison('Thornstrom', recs, recs_subset, histories_train, histories_dev, N=10)
```

<!-- #region id="D6C3ogOI-WOo" -->
## FIG Interleaved
<!-- #endregion -->

```python id="ppxfwRKn-WOo" outputId="17e0b29d-6b1c-4a2e-da1e-a4f45fd8accb"
display_user_recs_comparison('Rama', recs,['Interleaved'], histories_train, histories_dev, N=10)
```

```python id="52LAy96u-WOo" outputId="27a5021a-400b-454b-9ea7-492300392d5e"
display_user_recs_comparison('KingArti', recs,['Interleaved'], histories_train, histories_dev, N=10)
```

```python id="ViIjOnKo-WOp" outputId="402067b1-fe7b-45ff-c937-b638db7dea03"
N = 20
display(pd.DataFrame({rec_name: [p2t[r] for r in recs[rec_name][n2u['HenryXVII']]][:N] for rec_name in recs_subset}))
```

```python id="4sig0LjT-WOp"
persons_of_interest = [
    "DoctorWho42",
    "AxelSjögren",
    "Mighty platypus",
    "Tulietto",
    "LipaCityPH",
    "Hesperian Nguyen",
    "Thornstrom",
    "Meow",
    "HyprMarc",
    "Jampilot",
    "Rama"
]
N=10
```

```python id="Zh5wJ11u-WOq" colab={"referenced_widgets": ["df21373672484fc8a075c5acf4cf3e3b"]} outputId="3c260951-3c70-49db-810d-c9e757f83f96"
irec_500 = recommenders.ImplicitCollaborativeRecommender(model, implicit_matrix)
irecs_poi = irec_500.recommend_all([n2u[user_name] for user_name in persons_of_interest], N, u2i=u2i, n2i=n2i, i2p=i2p)
```


<!-- #region id="JWUHbf7A-WOq" -->
# Find interesting users
<!-- #endregion -->

```python id="hApo5gCO-WOq"
edited_pages = clean_histories.drop_duplicates(subset=['title','user']).groupby('user').userid.count()

edited_pages = edited_pages[edited_pages > 50]
edited_pages = edited_pages[edited_pages < 300]
```


```python id="HXcTZggh-WOr" outputId="680923a8-5977-425d-cf20-43d0f49f66c6"
clean_histories.columns
```

```python id="OtQQaL6m-WOr" outputId="c251d764-c4a0-4c6f-9384-016df0e22d38"
display_user_recs_comparison("Rama", recs, recs_subset, histories_train, histories_dev, N=20)
```


```python id="FA7J-tOZ-WOs" outputId="e7d8d17c-8850-42f4-b22d-dd338a69c7ba"
index = list(range(len(edited_pages)))
np.random.shuffle(index)

for i in index[:10]:
    user_name = edited_pages.index[i]
    print(user_name)
    display_user_recs_comparison(user_name, recs, recs_subset, histories_train, histories_dev, N=20)
    print("\n\n\n")
```

```python id="aayAbNCP-WOu"
index = list(range(len(edited_pages)))
np.random.shuffle(index)

for i in index[:10]:
    print(edited_pages.index[i])
    display_user_recs_comparison
    wr.print_user_history(user=edited_pages.index[i],all_histories=clean_histories)
    print("\n\n\n")
```

```python id="8C_wmTQ7-WOu"
sns.distplot(edited_pages,kde=False,bins=np.arange(0,2000,20))
```

<!-- #region id="NOyTdCF1-WOv" -->
# Repetition analysis
<!-- #endregion -->

```python id="oXiO82ES-WOv"
import itertools
```

```python id="HMcErsbP-WOw" outputId="649cb8ab-6af4-428d-9228-5843f52a61c4"
clean_histories.head()
```

```python id="eBzmdUjr-WOw" outputId="512498b6-6e12-49db-e49d-e9686cf01771"
clean_histories.iloc[:1000].values.tolist()
```

```python id="qPMpD2Wx-WOy" outputId="71285e49-f770-4953-b08d-2098e9a030e7"
df = clean_histories
dict(zip(df.columns, range(len(df.columns))))
```

```python id="iXBEtkWb-WOz"
def identify_runs(df):
    d  = df.loc[:,['userid','pageid']].values.tolist()
    return [(k, len(list(g))) for k,g in itertools.groupby(d)]
```

```python id="Bnkg2_ea-WOz" outputId="517a1562-3ed5-42c6-f9a7-6a2b4085f486"
%%time
runs = identify_runs(clean_histories)
```

```python id="L43DyR2D-WO0" outputId="2bc0a218-4798-4992-d6e8-235bd9ebb983"
lens = np.array([r[1] for r in runs])

single_edits = np.sum(lens==1)
total_edits = len(clean_histories)

print("Percent of edits that are part of a run: %.1f%%" % (100*(1-(float(single_edits)/total_edits))))

print("Percent of edits that are repetitions: %.1f%%" % (100*(1-len(runs)/total_edits)))
```
