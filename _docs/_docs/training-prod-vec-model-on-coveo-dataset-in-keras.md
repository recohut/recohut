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

<!-- #region id="KZ9YVOLXU5F8" -->
# In-session Recommendation in eCommerce
> Training prod2vec model on a small sample of coveo sigir21 retail dataset and covering cold-start as well as search query scoping scenarios

- toc: true
- badges: true
- comments: true
- categories: [ColdStart, Session, Sequential, Retail, Coveo, Prod2vec, Annoy, Keras, Embedding, Visualization, TSNE, Search, QueryScoping]
- image:
<!-- #endregion -->

<!-- #region id="r6hrx0AaU5F8" -->
| |  |
| :-: | -:|
| Vision | Improve customer's in-session experience |
| Mission | Improve search, learn better product embeddings, Effectively address the cold-start scenario|
| Scope | Model training and validation, Multi-modality, Search-based personalization, In-session recommendations |
| Task | Next-item Prediction, Cold-start Scenario, Search Query Scoping |
| Data | Coveo |
| Tool | Gensim, Keras, Colab, Python, Annoy |
| Technique | Improve search with query scoping technique, learn better product embeddings using prod2vec, Effectively address the cold-start scenario using multi-stage KNN scoring |
| Process | 1) Load data using recochef, 2) Build prod2vec model, 3) Improve low-count vectors to address cold-start, 4) Query scoping system |
| Takeaway | Prod2vec is simple yet effective model as can be seen in TSNE plot, Cold-start can be improved with strategies like we have seen in this case, query scoping is relevant |
| Credit | [Jacopo Tagliabue](https://github.com/jacopotagliabue) |
| Link | [link1](https://github.com/jacopotagliabue/retail-personalization-workshop), [link2](https://dl.acm.org/doi/10.1145/3366424.3386198) |
<!-- #endregion -->

<!-- #region id="Sw6E2dz8Q6cm" -->
## Environment setup
<!-- #endregion -->

```python id="FXvyKn2vgYlU"
!pip install gensim==4.0.1 keras==2.4.3 pydot==1.4.2 graphviz==0.16 annoy==1.17.0
!pip install git+https://github.com/sparsh-ai/recochef
```

```python id="0ccWbzTGfVVs"
import os
from random import choice
import time
import ast
import json
import numpy as np
import pandas as pd
import csv
from collections import Counter,defaultdict
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from IPython.display import Image 
import gensim
from gensim.similarities.annoy import AnnoyIndexer
from sklearn.model_selection import train_test_split
import hashlib
from copy import deepcopy

from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Concatenate
from keras.models import Sequential
from keras.layers import Input
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras import utils
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

from recochef.datasets.coveo import Coveo
```

```python id="m2qUbBeWhAce"
%matplotlib inline
```

```python colab={"base_uri": "https://localhost:8080/"} id="8MarALZkRCZv" executionInfo={"status": "ok", "timestamp": 1626639460375, "user_tz": -330, "elapsed": 5952, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3c175d37-2e1a-4d71-dbb1-a054e36fe1a0"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv
```

<!-- #region id="ABTmMAV0Q9ql" -->
## Data loading
<!-- #endregion -->

```python id="tmkYznGadRVG"
coveo_data = Coveo()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="bv4BXyyDeHBc" executionInfo={"status": "ok", "timestamp": 1626634283103, "user_tz": -330, "elapsed": 59216, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c2cf601a-a9e6-42c7-8b1c-8e4361f01976"
browsing_data = coveo_data.load_browsing_events()
browsing_data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="LXnpLallfE4w" executionInfo={"status": "ok", "timestamp": 1626634294403, "user_tz": -330, "elapsed": 11313, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c4994a0e-02e3-44e2-c616-6c2cf251b161"
labels = coveo_data.load_labels()
labels.keys()
```

```python colab={"base_uri": "https://localhost:8080/"} id="lB9LVxGB5CdX" executionInfo={"status": "ok", "timestamp": 1626634294407, "user_tz": -330, "elapsed": 47, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="de74ad32-14b8-47b7-adf1-b534930b39fe"
browsing_data.columns
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="vL1GbTXB5qwO" executionInfo={"status": "ok", "timestamp": 1626634294410, "user_tz": -330, "elapsed": 45, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6cafdae1-ea1b-4e0a-eee1-024240b53d4d"
browsing_data.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="MOCMk3dN4eqw" executionInfo={"status": "ok", "timestamp": 1626639258371, "user_tz": -330, "elapsed": 501, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="443b4e72-fb55-49e3-c9a8-a7482852ade2"
inverse_action_map = {v:k for k,v in labels['product_action'].items()}
browsing_data['ACTIONTYPE'] = browsing_data['ACTIONTYPE'].map(inverse_action_map)

inverse_event_map = {v:k for k,v in labels['event_type'].items()}
browsing_data['EVENTTYPE'] = browsing_data['EVENTTYPE'].map(inverse_event_map)

inverse_session_map = {v:k for k,v in labels['session_id_hash'].items()}
browsing_data['SESSIONID'] = browsing_data['SESSIONID'].map(inverse_session_map)

inverse_item_map = {v:k for k,v in labels['product_sku_hash'].items()}
browsing_data['ITEMID'] = browsing_data['ITEMID'].map(inverse_item_map.get, na_action='ignore')

inverse_url_map = {v:k for k,v in labels['hashed_url'].items()}
browsing_data['URLID'] = browsing_data['URLID'].map(inverse_url_map)

browsing_data.columns = ['session_id_hash','event_type','product_action',
                         'product_sku_hash','server_timestamp_epoch_ms', 'hashed_url']

browsing_data.head()
```

<!-- #region id="LXGikyCtRyrT" -->
## Build a prod2vec space

Know more - [blog](https://blog.coveo.com/clothes-in-space-real-time-personalization-in-less-than-100-lines-of-code/), [paper](https://arxiv.org/abs/2104.02061)
<!-- #endregion -->

```python id="7uCbrKlzfXJm"
N_ROWS = 500000  # how many rows we want to take (to avoid waiting too much for tutorial purposes)?
```

```python id="vjjx7qc16Y3o"
def read_sessions_from_training_file(df: pd.DataFrame):
    """
    Read the training file containing product interactions, up to K rows.
    
    :return: a list of lists, each list being a session (sequence of product IDs)
    """
    user_sessions = []
    current_session_id = None
    current_session = []
    for idx, row in df.iterrows():
        # just append "detail" events in the order we see them
        # row will contain: session_id_hash, product_action, product_sku_hash
        _session_id_hash = row['session_id_hash']
        # when a new session begins, store the old one and start again
        if current_session_id and current_session and _session_id_hash != current_session_id:
            user_sessions.append(current_session)
            # reset session
            current_session = []
        # check for the right type and append
        if row['product_action'] == 'detail':
            current_session.append(row['product_sku_hash'])
        # update the current session id
        current_session_id = _session_id_hash

    # print how many sessions we have...
    print("# total sessions: {}".format(len(user_sessions)))

    return user_sessions
```

```python id="NkYE64_u8XRR"
def train_product_2_vec_model(sessions: list,
                              min_c: int = 3,
                              size: int = 48,
                              window: int = 5,
                              iterations: int = 15,
                              ns_exponent: float = 0.75):
    """
    Train CBOW to get product embeddings. We start with sensible defaults from the literature - please
    check https://arxiv.org/abs/2007.14906 for practical tips on how to optimize prod2vec.

    :param sessions: list of lists, as user sessions are list of interactions
    :param min_c: minimum frequency of an event for it to be calculated for product embeddings
    :param size: output dimension
    :param window: window parameter for gensim word2vec
    :param iterations: number of training iterations
    :param ns_exponent: ns_exponent parameter for gensim word2vec
    :return: trained product embedding model
    """
    model =  gensim.models.Word2Vec(sentences=sessions,
                                    min_count=min_c,
                                    vector_size=size,
                                    window=window,
                                    epochs=iterations,
                                    ns_exponent=ns_exponent)

    print("# products in the space: {}".format(len(model.wv.index_to_key)))

    return model.wv
```

```python colab={"base_uri": "https://localhost:8080/"} id="gzNCgIHv8fRp" executionInfo={"status": "ok", "timestamp": 1626634539272, "user_tz": -330, "elapsed": 1704, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ed2ab49c-8807-440a-a981-d42919278455"
# get sessions
sessions = read_sessions_from_training_file(browsing_data.head(N_ROWS))

# get a counter on all items for later use
sku_cnt = Counter([item for s in sessions for item in s])

# print out most common SKUs
sku_cnt.most_common(3)
```

```python colab={"base_uri": "https://localhost:8080/"} id="SKmoC5Q_8rko" executionInfo={"status": "ok", "timestamp": 1626634563327, "user_tz": -330, "elapsed": 5239, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b83b7294-4052-4fea-c32a-2dca9748cf32"
# leave some sessions aside
idx = int(len(sessions) * 0.8)
train_sessions = sessions[0: idx]
test_sessions = sessions[idx:]
print("Train sessions # {}, test sessions # {}".format(len(train_sessions), len(test_sessions)))

# finally, train the p2vec, leaving all the default hyperparameters
prod2vec_model = train_product_2_vec_model(train_sessions)
```

```python colab={"base_uri": "https://localhost:8080/"} id="MeWK7xZn8sgq" executionInfo={"status": "ok", "timestamp": 1626634582114, "user_tz": -330, "elapsed": 415, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f2ef8a95-00e9-469a-dee2-7f2fd81e54da"
prod2vec_model.similar_by_word(sku_cnt.most_common(1)[0][0], topn=3)
```

```python id="0aMxnGxb-kAh"
def plot_scatter_by_category_with_lookup(title, 
                                         skus, 
                                         sku_to_target_cat,
                                         results, 
                                         custom_markers=None):
    groups = {}
    for sku, target_cat in sku_to_target_cat.items():
        if sku not in skus:
            continue

        sku_idx = skus.index(sku)
        x = results[sku_idx][0]
        y = results[sku_idx][1]
        if target_cat in groups:
            groups[target_cat]['x'].append(x)
            groups[target_cat]['y'].append(y)
        else:
            groups[target_cat] = {
                'x': [x], 'y': [y]
                }
    # DEBUG print
    print("Total of # groups: {}".format(len(groups)))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    for group, data in groups.items():
        ax.scatter(data['x'], data['y'], 
                   alpha=0.3, 
                   edgecolors='none', 
                   s=25, 
                   marker='o' if not custom_markers else custom_markers,
                   label=group)

    plt.title(title)
    plt.show()
    
    return
```

```python id="lYkof03Q-lO1"
def tsne_analysis(embeddings, perplexity=25, n_iter=1000):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
    return tsne.fit_transform(embeddings)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 255} id="0xsXjzFS-28G" executionInfo={"status": "ok", "timestamp": 1626634726687, "user_tz": -330, "elapsed": 7427, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1efcb037-3c51-48c8-a977-9d99b915910f"
sku_category = coveo_data.load_metadata()
sku_category.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 326} id="JQEtzdQu_DtF" executionInfo={"status": "ok", "timestamp": 1626635834986, "user_tz": -330, "elapsed": 472, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5fe94e59-b980-4e40-c130-2f196e88262b"
# inverse_item_map = {v:k for k,v in labels['product_sku_hash'].items()}
sku_category['ITEMID'] = sku_category['ITEMID'].map(inverse_item_map.get, na_action='ignore')

inverse_category_map = {v:k for k,v in labels['category_hash'].items()}
sku_category['CATEGORYID'] = sku_category['CATEGORYID'].map(inverse_category_map)

sku_category.columns = ['product_sku_hash','description_vector','category_hash',
                         'image_vector','price_bucket']

sku_category['category_hash'] = sku_category['category_hash'].fillna('None')

sku_category.head()
```

```python id="ofIWJBLs-m7I"
def get_sku_to_category_map(df, depth_index=1):
    """
    For each SKU, get category from catalog file (if specified)
    
    :return: dictionary, mapping SKU to a category
    """
    sku_to_cats = dict()
    for _, row in df.iterrows():
        _sku = row['product_sku_hash']
        category_hash = row['category_hash']
        if category_hash=='None':
            continue
        # pick only category at a certain depth in the tree
        # e.g. x/xx/xxx, with depth=1, -> xx
        branches = category_hash.split('/')
        target_branch = branches[depth_index] if depth_index < len(branches) else None
        if not target_branch:
            continue
        # if all good, store the mapping
        sku_to_cats[_sku] = target_branch
            
    return sku_to_cats
```

```python colab={"base_uri": "https://localhost:8080/"} id="_62k4npZ-oYm" executionInfo={"status": "ok", "timestamp": 1626635886720, "user_tz": -330, "elapsed": 6061, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b743f800-ca90-4811-96e4-698fca363f75"
sku_to_category = get_sku_to_category_map(sku_category)
print("Total of # {} categories".format(len(set(sku_to_category.values()))))
print("Total of # {} SKU with a category".format(len(sku_to_category)))
# debug with a sample SKU
print(sku_to_category[sku_cnt.most_common(1)[0][0]])
skus = prod2vec_model.index_to_key
print("Total of # {} skus in the model".format(len(skus)))
embeddings = [prod2vec_model[s] for s in skus]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 862} id="e1ZUDPwWAc99" executionInfo={"status": "ok", "timestamp": 1626635994969, "user_tz": -330, "elapsed": 86979, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="60727b54-e691-4497-a402-a9ab1f38c4ee"
# print out tsne plot with standard params
tsne_results = tsne_analysis(embeddings)
assert len(tsne_results) == len(skus)
plot_scatter_by_category_with_lookup('Prod2vec', skus, sku_to_category, tsne_results)
```

```python id="Nr5kt4uxDjEL"
# do a version with only top K categories
TOP_K = 5
cnt_categories = Counter(list(sku_to_category.values()))
top_categories = [c[0] for c in cnt_categories.most_common(TOP_K)]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 641} id="Ee76hA5kDqWL" executionInfo={"status": "ok", "timestamp": 1626635998152, "user_tz": -330, "elapsed": 3198, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="196ab27d-975b-4e57-ccef-a74a641c46f6"
# filter out SKUs outside of top categories
top_skus = []
top_tsne_results = []
for _s, _t in zip(skus, tsne_results):
    if sku_to_category.get(_s, None) not in top_categories:
        continue
    top_skus.append(_s)
    top_tsne_results.append(_t)
# re-plot tsne with filtered SKUs
print("Top SKUs # {}".format(len(top_skus)))
plot_scatter_by_category_with_lookup('Prod2vec (top {})'.format(TOP_K), 
                                     top_skus, sku_to_category, top_tsne_results)
```

```python colab={"base_uri": "https://localhost:8080/"} id="GmxgYdbEEWZS" executionInfo={"status": "ok", "timestamp": 1626636129573, "user_tz": -330, "elapsed": 2298, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="08252ca3-73c9-44cd-8b9d-5806cdbb6b19"
# Set up the model and vector that we are using in the comparison
annoy_index = AnnoyIndexer(prod2vec_model, 100)
test_sku = sku_cnt.most_common(1)[0][0]
# test all is good
print(prod2vec_model.most_similar([test_sku], topn=2, indexer=annoy_index))
print(prod2vec_model.most_similar([test_sku], topn=2))
```

```python colab={"base_uri": "https://localhost:8080/"} id="o-JVMlXhEYlv" executionInfo={"status": "ok", "timestamp": 1626636141961, "user_tz": -330, "elapsed": 3214, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="34a68431-eebd-4481-83a7-c4edde6c19b9"
def avg_query_time(model, annoy_index=None, queries=5000):
    """Average query time of a most_similar method over random queries."""
    total_time = 0
    for _ in range(queries):
        _v = model[choice(model.index_to_key)]
        start_time = time.process_time()
        model.most_similar([_v], topn=5, indexer=annoy_index)
        total_time += time.process_time() - start_time
        
    return total_time / queries

gensim_time = avg_query_time(prod2vec_model)
annoy_time = avg_query_time(prod2vec_model, annoy_index=annoy_index)
print("Gensim (s/query):\t{0:.5f}".format(gensim_time))
print("Annoy (s/query):\t{0:.5f}".format(annoy_time))
speed_improvement = gensim_time / annoy_time
print ("\nAnnoy is {0:.2f} times faster on average on this particular run".format(speed_improvement))
```

```python id="R4QIuD7VEbZ4"
def calculate_HR_on_NEP(model, sessions, k=10, min_length=3):
    _count = 0
    _hits = 0
    for session in sessions:
        # consider only decently-long sessions
        if len(session) < min_length:
            continue
        # update the counter
        _count += 1
        # get the item to predict
        target_item = session[-1]
        # get model prediction using before-last item
        query_item = session[-2]
        # if model cannot make the prediction, it's a failure
        if query_item not in model:
            continue
        predictions = model.similar_by_word(query_item, topn=k)
        # debug
        # print(target_item, query_item, predictions)
        if target_item in [p[0] for p in predictions]:
            _hits += 1
    # debug
    print("Total test cases: {}".format(_count))
    
    return _hits / _count
```

```python colab={"base_uri": "https://localhost:8080/"} id="Yt1nznjhEial" executionInfo={"status": "ok", "timestamp": 1626636189448, "user_tz": -330, "elapsed": 5694, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="66d5ec65-abcf-4a16-9489-9cbd23d1569f"
# we simulate a test with 3 values for epochs in prod2ve
iterations_values = [1, 10]
# for each value we train a model, and use Next Event Prediction (NEP) to get a quality assessment
for i in iterations_values:
    print("\n ======> Hyper value: {}".format(i))
    cnt_model = train_product_2_vec_model(train_sessions, iterations=i)
    # use hold-out to have NEP performance
    _hr = calculate_HR_on_NEP(cnt_model, test_sessions)
    print("HR: {}\n".format(_hr))
```

<!-- #region id="associate-fairy" -->
## Improving low-count vectors

*prod2vec in the cold start scenario*

Know more - [paper](https://dl.acm.org/doi/10.1145/3383313.3411477), [video](https://vimeo.com/455641121)
<!-- #endregion -->

```python id="capable-ultimate"
def build_mapper(pro2vec_dims=48):
    """
    Build a Keras model for content-based "fake" embeddings.
    
    :return: a Keras model, mapping BERT-like catalog representations to the prod2vec space
    """
    # input
    description_input = Input(shape=(50,))
    image_input = Input(shape=(50,))
    # model
    x = Dense(25, activation="relu")(description_input)
    y = Dense(25, activation="relu")(image_input)
    combined = Concatenate()([x, y])
    combined = Dropout(0.3)(combined)
    combined = Dense(25)(combined)
    output = Dense(pro2vec_dims)(combined)

    return Model(inputs=[description_input, image_input], outputs=output)
```

```python id="italian-stereo"
# get vectors representing text and images in the catalog
def get_sku_to_embeddings_map(df):
    """
    For each SKU, get the text and image embeddings, as provided pre-computed by the dataset
    
    :return: dictionary, mapping SKU to a tuple of embeddings
    """
    sku_to_embeddings = dict()
    for _, row in df.iterrows():
        _sku = row['product_sku_hash']
        _description = row['description_vector']
        _image = row['image_vector']
        # skip when both vectors are not there
        if not _description or not _image:
            continue
        # if all good, store the mapping
        sku_to_embeddings[_sku] = (json.loads(_description), json.loads(_image))
            
    return sku_to_embeddings
```

```python id="champion-simpson" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626636370037, "user_tz": -330, "elapsed": 8912, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="19a39ef2-eb79-4107-e4b3-d9b039edb45e"
sku_to_embeddings = get_sku_to_embeddings_map(sku_category)
print("Total of # {} SKUs with embeddings".format(len(sku_to_embeddings)))
# print out an example
_d, _i = sku_to_embeddings['438630a8ba0320de5235ee1bedf3103391d4069646d640602df447e1042a61a3']
print(len(_d), len(_i), _d[:5], _i[:5])
```

```python id="executive-cover" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626636381761, "user_tz": -330, "elapsed": 438, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="44fa4078-4b94-4c37-db76-4dfa020e183a"
# just make sure we have the SKUs in the model and a counter
skus = prod2vec_model.index_to_key
print("Total of # {} skus in the model".format(len(skus)))
print(sku_cnt.most_common(5))
```

<!-- #region id="8qU4OnA1UJrI" -->
Due to the long-tail of interactions, not all product vectors have the same quality. Plus, new products have no interactions!
<!-- #endregion -->

<!-- #region id="NNcl1tffTSvA" -->
**Solution - “Faking” high quality vectors through content**

1. First, we learn a mapping between meta-data and the space by using
popular products only.
2. Then, we apply the mapping to rare products and obtain new vectors! 
<!-- #endregion -->

<!-- #region id="gvl4z9-aTc8u" -->
<!-- #endregion -->

<!-- #region id="yY7bvlFpThRz" -->
<!-- #endregion -->

```python id="distinguished-satin"
# above which percentile of frequency we consider SKU popular enough to be our training set?
FREQUENT_PRODUCTS_PTILE = 80
```

```python id="introductory-entrepreneur" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626636392718, "user_tz": -330, "elapsed": 409, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="64e08c1c-fcec-49a1-ada5-3cfdbd5108e0"
_counts = [c[1] for c in sku_cnt.most_common()]
_counts[:3]
```

```python id="adult-handbook" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626636407960, "user_tz": -330, "elapsed": 402, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0dd01901-c979-4c57-b216-ed5574e24470"
# make sure we have just SKUS in the prod2vec space for which we have embeddings
popular_threshold = np.percentile(_counts, FREQUENT_PRODUCTS_PTILE)
popular_skus = [s for s in skus if s in sku_to_embeddings and sku_cnt.get(s, 0) > popular_threshold]
product_embeddings = [prod2vec_model[s] for s in popular_skus]
description_embeddings = [sku_to_embeddings[s][0] for s in popular_skus]
image_embeddings = [sku_to_embeddings[s][1] for s in popular_skus]
# debug
print(popular_threshold, len(skus), len(popular_skus))
# print(description_embeddings[:1][:3])
# print(image_embeddings[:1][:3])
```

```python id="physical-baptist"
# train the mapper now
training_data_X = [np.array(description_embeddings), np.array(image_embeddings)]
training_data_y = np.array(product_embeddings)
```

```python id="working-trade" colab={"base_uri": "https://localhost:8080/", "height": 644} executionInfo={"status": "ok", "timestamp": 1626636425114, "user_tz": -330, "elapsed": 2474, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6028239a-cc15-4263-c9f0-63905d1f08b8"
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)
# build and display model
rare_net = build_mapper()
plot_model(rare_net, show_shapes=True, show_layer_names=True, to_file='rare_net.png')
Image('rare_net.png')
```

```python id="hourly-poultry" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626636471508, "user_tz": -330, "elapsed": 25642, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1791e82d-4073-4466-9b33-8ef31bac0e09"
# train!
rare_net.compile(loss='mse', optimizer='rmsprop')
rare_net.fit(training_data_X, 
             training_data_y, 
             batch_size=200, 
             epochs=20000, 
             validation_split=0.2, 
             callbacks=[es])
```

```python id="enclosed-portal" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626636496386, "user_tz": -330, "elapsed": 438, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f9cb6518-1014-4063-c05f-00bdabd7e558"
# rarest_skus = [_[0] for _ in sku_cnt.most_common()[-500:]]
# test_skus = [s for s in rarest_skus if s in sku_to_embeddings]

# get to rare vectors
test_skus = [s for s in skus if s in sku_to_embeddings and sku_cnt.get(s, 0) < popular_threshold/2]
print(len(skus), len(test_skus))
# prepare embeddings for prediction
rare_description_embeddings = [sku_to_embeddings[s][0] for s in test_skus]
rare_image_embeddings = [sku_to_embeddings[s][1] for s in test_skus]
```

```python id="realistic-underwear"
# prepare embeddings for prediction
test_data_X = [np.array(rare_description_embeddings), np.array(rare_image_embeddings)]
predicted_embeddings = rare_net.predict(test_data_X)
# debug
# print(len(predicted_embeddings))
# print(predicted_embeddings[0][:10])
```

```python id="a9336162"
def calculate_HR_on_NEP_rare(model, sessions, rare_skus, k=10, min_length=3):
    _count = 0
    _hits = 0
    _rare_hits = 0
    _rare_count = 0
    for session in sessions:
        # consider only decently-long sessions
        if len(session) < min_length:
            continue
        # update the counter
        _count += 1
        # get the item to predict
        target_item = session[-1]
        # get model prediction using before-last item
        query_item = session[-2]

        # if model cannot make the prediction, it's a failure
        if query_item not in model:
            continue
        
        # increment counter if rare sku
        if query_item in rare_skus:
            _rare_count+=1
        
        predictions = model.similar_by_word(query_item, topn=k)
    
        # debug
        # print(target_item, query_item, predictions)    
        if target_item in [p[0] for p in predictions]:
            _hits += 1
            # track hits if query is rare sku
            if query_item in rare_skus:
                _rare_hits+=1
    # debug
    print("Total test cases: {}".format(_count))
    print("Total rare test cases: {}".format(_rare_count))
    
    return _hits / _count, _rare_hits/_rare_count
```

```python id="fa3894f3" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626636513513, "user_tz": -330, "elapsed": 1846, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="74253ca0-cb73-4c6e-d54f-22e2ad97d7dc"
# make copy of original prod2vec model
prod2vec_rare_model = deepcopy(prod2vec_model)
# update model with new vectors
prod2vec_rare_model.add_vectors(test_skus, predicted_embeddings, replace=True)
prod2vec_rare_model.fill_norms(force=True)
# check
assert np.array_equal(predicted_embeddings[0], prod2vec_rare_model[test_skus[0]])

# test new model
calculate_HR_on_NEP_rare(prod2vec_rare_model, test_sessions, test_skus)
```

```python id="e1f7a5f5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626636528744, "user_tz": -330, "elapsed": 1743, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7158537e-7382-461e-bf74-83b4ca401acb"
# test original model
calculate_HR_on_NEP_rare(prod2vec_model, test_sessions, test_skus)
```

<!-- #region id="chinese-flour" -->
## Query scoping

Know more - [paper](https://www.aclweb.org/anthology/2020.ecnlp-1.2), [repo](https://github.com/jacopotagliabue/session-path)
<!-- #endregion -->

<!-- #region id="DyIjJ-TeTtmk" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="M-sMrExqGfBg" executionInfo={"status": "ok", "timestamp": 1626638056393, "user_tz": -330, "elapsed": 6822, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="079c1cec-bf4e-478c-94fc-2569029f1034"
search_data = coveo_data.load_search_events()
search_data.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="-AX0EAXOGfBs" executionInfo={"status": "ok", "timestamp": 1626638110555, "user_tz": -330, "elapsed": 54176, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8684d910-d053-4b42-d074-9cb4890d40b7"
# inverse_item_map = {v:k for k,v in labels['product_sku_hash'].items()}
search_data['SESSIONID'] = search_data['SESSIONID'].map(inverse_session_map)
search_data['ITEMID_VIEW'] = search_data['ITEMID_VIEW'].apply(lambda x: [inverse_item_map[y] for y in x if not np.isnan(y)])
search_data['ITEMID_CLICKED'] = search_data['ITEMID_CLICKED'].apply(lambda x: [inverse_item_map[y] for y in x if not np.isnan(y)])

search_data.columns = ['session_id_hash','query_vector','clicked_skus_hash',
                         'product_skus_hash','server_timestamp_epoch_ms']

# search_data.clicked_skus_hash = search_data.clicked_skus_hash.apply(lambda y: np.nan if len(y)==0 else y)
# search_data.product_skus_hash = search_data.product_skus_hash.apply(lambda y: np.nan if len(y)==0 else y)

search_data.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="a51kUQk-MANF" executionInfo={"status": "ok", "timestamp": 1626638134408, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8ff7c301-fdc2-474c-b081-2c681447469a"
xx = search_data.head()
xx
```

```python id="accessible-talent"
from tensorflow.keras.utils import to_categorical

# get vectors representing text and images in the catalog
def get_query_to_category_dataset(df, cat_2_id, sku_to_category):
    """
    For each query, get a label representing the category in items clicked after the query.
    It uses as input a mapping "sku_to_category" to join the search file with catalog meta-data!
    
    :return: two lists, matching query vectors to a label
    """
    query_X = list()
    query_Y = list()
    for _, row in df.iterrows():
        _click_products = row['clicked_skus_hash']
        if not _click_products: # or _click_product not in sku_to_category:
            continue
        # clean the string and extract SKUs from array
        cleaned_skus = _click_products
        for s in cleaned_skus: 
            if s in sku_to_category:
                query_X.append(json.loads(row['query_vector']))
                target_category_as_int = cat_2_id[sku_to_category[s]]
                query_Y.append(to_categorical(target_category_as_int, num_classes=len(cat_2_id)))
            
    return query_X, query_Y
```

```python id="certain-wings" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626638502853, "user_tz": -330, "elapsed": 108412, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="159967fd-5d56-4427-d21b-688efb5f159c"
# sku_to_category = get_sku_to_category_map(sku_category)
print("Total of # {} categories".format(len(set(sku_to_category.values()))))
# cats = list(set(sku_to_category.values()))
# cat_2_id = {c: idx for idx, c in enumerate(cats)}
print(cat_2_id[cats[0]])
query_X, query_Y = get_query_to_category_dataset(search_data, 
                                                 cat_2_id,
                                                 sku_to_category)
print(len(query_X))
print(query_Y[0])
```

```python id="normal-draft"
x_train, x_test, y_train, y_test = train_test_split(np.array(query_X), np.array(query_Y), test_size=0.2)
```

```python id="subject-weapon"
def build_query_scoping_model(input_d, target_classes):
    print('Shape tensor {}, target classes {}'.format(input_d, target_classes))
    # define model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_d))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(target_classes, activation='softmax'))
    
    return model
```

```python id="french-guide" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626638505340, "user_tz": -330, "elapsed": 42, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e5a51445-429b-4b42-e9f2-e29d71749529"
query_model = build_query_scoping_model(x_train[0].shape[0], y_train[0].shape[0])
```

```python id="falling-confirmation" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626638714333, "user_tz": -330, "elapsed": 209026, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="db692385-9e6a-48d7-c0b8-35a29e0aca25"
# compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
query_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# train first
query_model.fit(x_train, y_train, epochs=10, batch_size=32)
# compute and print eval score
score = query_model.evaluate(x_test, y_test, batch_size=32)
score
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="i1Y_caXAQEnW" executionInfo={"status": "ok", "timestamp": 1626639196609, "user_tz": -330, "elapsed": 411, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dddee78b-a8f0-42f5-f0cd-cbb73c63f3b2"
browsing_data.head()
```

```python id="brave-activity"
# get vectors representing text and images in the catalog
def get_query_info(df):
    """
    For each query, extract relevant metadata of query and to match with session data

    :return: list of queries with metadata
    """
    queries = list()
    for _, row in df.iterrows():
        _click_products = row['clicked_skus_hash']
        if not _click_products: # or _click_product not in sku_to_category:
            continue
        # clean the string and extract SKUs from array
        cleaned_skus = _click_products
        queries.append({'session_id_hash' : row['session_id_hash'],
                        'server_timestamp_epoch_ms' : int(row['server_timestamp_epoch_ms']),
                        'clicked_skus' : cleaned_skus,
                        'query_vector' : json.loads(row['query_vector'])})
    print("# total queries: {}".format(len(queries)))        
    
    return queries

def get_session_info_for_queries(df: str, query_info: list, K: int = None):
    """
    Read the training file containing product interactions for sessions with query, up to K rows.
    
    :return: dict of lists with session_id as key, each list being a session (sequence of product events with metadata) 
    """
    user_sessions = dict()
    current_session_id = None
    current_session = []
    
    query_session_ids = set([ _['session_id_hash'] for _ in query_info])

    for idx, row in df.iterrows():
        # just append "detail" events in the order we see them
        # row will contain: session_id_hash, product_action, product_sku_hash
        _session_id_hash = row['session_id_hash']
        # when a new session begins, store the old one and start again
        if current_session_id and current_session and _session_id_hash != current_session_id:
            user_sessions[current_session_id] = current_session
            # reset session
            current_session = []
        # check for the right type and append event info
        if row['product_action'] == 'detail' and _session_id_hash in query_session_ids :
            current_session.append({'product_sku_hash': row['product_sku_hash'],
                                    'server_timestamp_epoch_ms' : int(row['server_timestamp_epoch_ms'])})
        # update the current session id
        current_session_id = _session_id_hash

    # print how many sessions we have...
    print("# total sessions: {}".format(len(user_sessions)))


    return dict(user_sessions)
```

```python id="3e2dc83f" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626639345672, "user_tz": -330, "elapsed": 44025, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="43320ea0-686a-4b3c-fd21-620957d985ad"
query_info = get_query_info(search_data)
session_info = get_session_info_for_queries(browsing_data.head(N_ROWS), query_info)
```

```python id="c55b2172"
def get_contextual_query_to_category_dataset(query_info, session_info, prod2vec_model, cat_2_id, sku_to_category):
    """
    For each query, get a label representing the category in items clicked after the query.
    It uses as input a mapping "sku_to_category" to join the search file with catalog meta-data!
    It also creates a joint embedding for input by concatenating query vector and average session vector up till
    when query was made
    
    :return: two lists, matching query vectors to a label
    """
    query_X = list()
    query_Y = list()
    
    for row in query_info:
        query_timestamp = row['server_timestamp_epoch_ms']
        cleaned_skus = row['clicked_skus']
        session_id_hash = row['session_id_hash']
        if session_id_hash not in session_info or not cleaned_skus: # or _click_product not in sku_to_category:
            continue            
            
        session_skus = session_info[session_id_hash]
        context_skus = [ e['product_sku_hash'] for e in session_skus if query_timestamp > e['server_timestamp_epoch_ms'] 
                                                                        and e['product_sku_hash'] in prod2vec_model]
        if not context_skus:
            continue
        context_vector = np.mean([prod2vec_model[sku] for sku in context_skus], axis=0).tolist()
        for s in cleaned_skus: 
            if s in sku_to_category:
                query_X.append(row['query_vector'] + context_vector)
                target_category_as_int = cat_2_id[sku_to_category[s]]
                query_Y.append(to_categorical(target_category_as_int, num_classes=len(cat_2_id)))
            
    return query_X, query_Y
```

```python id="0019ff85" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626639379683, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="82b04640-3980-4a9c-ab01-f0fb3432bd78"
context_query_X, context_query_Y = get_contextual_query_to_category_dataset(query_info, 
                                                                            session_info, 
                                                                            prod2vec_model, 
                                                                            cat_2_id, 
                                                                            sku_to_category)
print(len(context_query_X))
print(context_query_Y[0])
```

```python id="794c7429"
x_train, x_test, y_train, y_test = train_test_split(np.array(context_query_X), np.array(context_query_Y), test_size=0.2)
```

```python id="cccbdba0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626639387480, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7b01dd93-7504-446b-ac86-ff61c32036d6"
contextual_query_model = build_query_scoping_model(x_train[0].shape[0], y_train[0].shape[0])
```

```python id="420b0275" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626639393314, "user_tz": -330, "elapsed": 3012, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e44d0aea-9970-4a5f-a813-f0f90cd504b2"
# compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
contextual_query_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# train first
contextual_query_model.fit(x_train, y_train, epochs=10, batch_size=32)
# compute and print eval score
score = contextual_query_model.evaluate(x_test, y_test, batch_size=32)
score
```

<!-- #region id="KV0jiVxpUfqL" -->
## References

* ["An Image is Worth a Thousand Features": Scalable Product Representations for In-Session Type-Ahead Personalization](https://dl.acm.org/doi/10.1145/3366424.3386198)
* [Fantastic Embeddings and How to Align Them: Zero-Shot Inference in a Multi-Shop Scenario](https://arxiv.org/abs/2007.14906)
* [Shopping in the Multiverse: A Counterfactual Approach to In-Session Attribution](https://arxiv.org/pdf/2007.10087.pdf)
* [The Embeddings That Came in From the Cold: Improving Vectors for New and Rare Products with Content-Based Inference](https://dl.acm.org/doi/10.1145/3383313.3411477)
* [How to Grow a (Product) Tree: Personalized Category Suggestions for eCommerce Type-Ahead](https://www.aclweb.org/anthology/2020.ecnlp-1.2/)
<!-- #endregion -->
