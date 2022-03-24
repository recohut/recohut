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

```python id="86MgMsi_GD70" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628326501067, "user_tz": -330, "elapsed": 685, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="50479f6e-3232-4e78-fadb-3674d4756d7c"
import os
project_name = "reco-tut-ttr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="zzFVExkIFzDe" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628324637145, "user_tz": -330, "elapsed": 520, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a115c61d-fcc1-4120-91d8-b924c1cf89e0"
!git status
```

```python id="pXWJ6RWXjvEx" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628324649336, "user_tz": -330, "elapsed": 5539, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9c3500bd-5345-4ce0-fe6e-fcbd2023313b"
!git add . && git commit -m 'commit' && git push origin main
```

<!-- #region id="DqVtQ4T7Fz_l" -->
---
<!-- #endregion -->

<!-- #region id="_maNPpw-ADFa" -->
## Setup
<!-- #endregion -->

```python id="XJhLU7p1l0rL" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628320497409, "user_tz": -330, "elapsed": 16161, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ec03a9ec-9b76-47f0-ee50-d6adcd73ec7a"
!pip install -q git+https://github.com/sparsh-ai/recochef.git
```

```python id="8741PGXA-IaD"
#basic
import csv
from os import path, makedirs
import random
import requests
import math
import pickle
import gc
from time import time

# matrix
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import components

import matplotlib.pyplot as plt

# fancy python styling and error bars
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm

from dataclasses import dataclass
from recochef.utils._utils import download_yandex

from gensim.models import Word2Vec
import multiprocessing
```

```python id="HfgX6C7o-VhB"
RANDOM_SEED = 42
```

<!-- #region id="wBfkBPFcAAuP" -->
## Data loading
<!-- #endregion -->

```python id="pO8CL14YfRb_"
class TrivagoData:
    """download the trivago dataset"""

    def __init__(self, tempdata_path='/content/tempdata'):
        self.data_path = tempdata_path
        if not path.exists(self.data_path):
            makedirs(self.data_path)

    def train(self, name='train'):
        url = 'https://disk.yandex.ru/d/fFEvSaj6Cdplug'
        fpath = path.join(self.data_path, name+'.parquet.gzip')
        if not path.exists(fpath):
            download_yandex(url, fpath)
        return pd.read_parquet(fpath)

    def test(self, name='test'):
        url = 'https://disk.yandex.ru/d/_kAw9ajZuIC3rg'
        fpath = path.join(self.data_path, name+'.parquet.gzip')
        if not path.exists(fpath):
            download_yandex(url, fpath)
        return pd.read_parquet(fpath)
    
    def items(self, name='items'):
        url = 'https://github.com/recohut/reco-data/raw/master/trivago/v1/item_metadata.parquet.gz'
        fpath = path.join(self.data_path, name+'.parquet.gzip')
        if not path.exists(fpath):
            r = requests.get(url, allow_redirects=True)
            open(fpath, 'wb').write(r.content)
        return pd.read_parquet(fpath)
         
    def confirmation(self, name='confirmation'):
        url = 'https://github.com/recohut/reco-data/raw/master/trivago/v1/confirmation.parquet.gz'
        fpath = path.join(self.data_path, name+'.parquet.gzip')
        if not path.exists(fpath):
            r = requests.get(url, allow_redirects=True)
            open(fpath, 'wb').write(r.content)
        return pd.read_parquet(fpath)

    def validation(self, name='validation'):
        url = 'https://github.com/recohut/reco-data/raw/master/trivago/v1/validation.parquet.gz'
        fpath = path.join(self.data_path, name+'.parquet.gzip')
        if not path.exists(fpath):
            r = requests.get(url, allow_redirects=True)
            open(fpath, 'wb').write(r.content)
        return pd.read_parquet(fpath)
```

```python id="QKVVF1PyeSpL"
data_loader = TrivagoData()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="pz5f8mapi0-a" executionInfo={"status": "ok", "timestamp": 1628322161226, "user_tz": -330, "elapsed": 1542, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b5760c99-72ed-4257-c35d-443b0668cc8f"
item_metadata = data_loader.items()
item_metadata.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="MxFsjENhl7eB" executionInfo={"status": "ok", "timestamp": 1628322688771, "user_tz": -330, "elapsed": 29767, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a3e73123-749d-4dbc-9306-f59919e784b7"
train = data_loader.train()
train.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="g649Q9qRl_7z" executionInfo={"status": "ok", "timestamp": 1628322706067, "user_tz": -330, "elapsed": 11476, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0938c06a-28f6-4309-d8d7-18b9afdcd0b8"
test = data_loader.test()
test.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="ErrS-G7emKL2" executionInfo={"status": "ok", "timestamp": 1628322727139, "user_tz": -330, "elapsed": 2052, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="779da6a3-2683-4cbe-d8a5-299bc132f818"
confirmation = data_loader.confirmation()
confirmation.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="23dhEuzJmRhl" executionInfo={"status": "ok", "timestamp": 1628322750877, "user_tz": -330, "elapsed": 5245, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1604640f-bde0-4b65-ddd1-119429ea9269"
validation = data_loader.validation()
validation.head()
```

<!-- #region id="ut7qRGGVk8R9" -->
## Compute CTR
<!-- #endregion -->

<!-- #region id="Oy2UCscPk-kl" -->
Compute click-through ratios for hotels. Do this two ways.

First, by looping over the clickouts in the training set and computing for each hotel i: *ctr_i = (times hotel i is clicked)/(# times hotel i appears in impression results)*.

Second -- intuitively, we'd like the click through ratio to be penalized most when the first-rank item is not clicked and rewarded most when it is clicked at a higher rank.
<!-- #endregion -->

```python id="PlTQTdVElpYx"
compute_ctr: Dict = defaultdict(lambda: {"num":0, "num_pw":0, "denom":0, "denom_pw": 0})
rank_click_prob: Dict[int,float] = {}
```

```python id="Xi6LxaHJltHG"
# helpers for main function -- these get applied to df rows
def create_entry(interaction_row: Tuple[str,int]) -> None:
    # unpack tuple. (item_id, rank_appeared)
    item_id = interaction_row[0]
    rank_appeared = interaction_row[1]
    entry = compute_ctr[item_id]
    entry["num"] += 1
    entry["num_pw"] += rank_click_prob[rank_appeared]
```

```python id="pBA9S6jolwCR"
def count_denom(L: List) -> None:
    for idx, id in enumerate(L):
        compute_ctr[id]["denom"] +=1
        compute_ctr[id]["denom_pw"] += rank_click_prob[idx + 1] # rank = index + 1
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="_Mjvy799mgUc" executionInfo={"status": "ok", "timestamp": 1628322848455, "user_tz": -330, "elapsed": 2714, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e276e5bb-3f5f-4594-968c-8b4b863ba8d8"
df = train.copy()
df.head()
```

```python id="Uie-JbSNm4wc"
df = df[df.action_type == "clickout item"]
df["impressions"] = df["impressions"].str.split("|").map(list)

# meanwhile: get the index of the reference in the impression. Adding one makes this the rank.
get_rank = lambda x: 1 + x[1].index(x[0]) if x[0] in x[1] else 0
df["rank_appeared"] = df[["reference","impressions"]].apply(get_rank,axis=1)
# apply this over ref and impressions (x[0] is reference, x[1] is impressions)
df = df[df.rank_appeared != 0]  # if the reference is not in the impressions list -- remove it
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="3mk41akYnC-0" executionInfo={"status": "ok", "timestamp": 1628322995546, "user_tz": -330, "elapsed": 1177, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e5fba6b2-7f6e-48f2-bac0-4704eff222a0"
plt.plot(df["rank_appeared"].value_counts().sort_index())
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="nzJqEbTPoNW9" executionInfo={"status": "ok", "timestamp": 1628323536864, "user_tz": -330, "elapsed": 428, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c8760a0c-c20e-4928-8f84-47712e328aa3"
gc.collect()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="t7hmiL1gnlDO" executionInfo={"status": "ok", "timestamp": 1628323295365, "user_tz": -330, "elapsed": 76624, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8791d848-8969-4ed4-c769-8764fc98b7f4"
# i want the clicks df to look like:
# id | ctr | ctr_prob
# AB | 0.1 | 0.03

denominator = df.shape[0] # how many observations am I working with
rank: int
for rank, count in df["rank_appeared"].value_counts().iteritems():
    rank_click_prob[rank] = 1 - (count/denominator)

#  add these interaction counts to my default dict with this apply trick
df[["reference","rank_appeared"]].apply(create_entry,axis=1)
df["impressions"].apply(count_denom)

# then do the division num/denom and output to a new dataframe
safe_divide = lambda x,y: x / y if (y > 0) else 0
record_format  = [{"item_id": id, "ctr":safe_divide(dict["num"],dict["denom"]), "ctr_prob": safe_divide(dict["num_pw"],dict["denom_pw"])} for id,dict in compute_ctr.items()]
clicks = pd.DataFrame.from_records(record_format)

clicks.head()
```

```python id="jk7bP7HJogIE"
silver_layer = './data/silver'
if not path.exists(silver_layer):
    os.makedirs(silver_layer)
```

```python id="HbMj1nEElbE9"
# save it
clicks.to_parquet('./data/silver/item_clicks.parquet.gzip', compression='gzip')
```

<!-- #region id="Jk0iCyRqkTPe" -->
## Extract Hotel features
<!-- #endregion -->

```python id="JY0sZ9ybpFfd"
RATING_MAP = {"Satisfactory Rating": 1, "Good Rating": 2, "Very Good Rating": 3, "Excellent Rating": 4}
STAR_MAP = {"1 Star": 1, "2 Star": 2, "3 Star": 3, "4 Star": 4, "5 Star": 5}
HOTEL_CAT = {
    "Hotel": "hotel",
    "Resort": "resort",
    "Hostal (ES)": "hostal",
    "Motel": "motel",
    "House / Apartment": "house",
}
```

```python id="J4ZYiAjlpGfj"
def densify(d: Dict[str,Any], properties):
    """
    Apply the mapping d to each row of properties
    """
    values = [None] * properties.shape[0]
    for i, p in enumerate(properties):
        for k in d:
            if k in p:
                values[i] = d[k]
    return values
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="N0ESX9r3piXv" executionInfo={"status": "ok", "timestamp": 1628323604830, "user_tz": -330, "elapsed": 2847, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d88208e4-c5c0-4e32-f590-648e7efd39ef"
df = item_metadata.copy()
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="OqT3pnrhpwwY" executionInfo={"status": "ok", "timestamp": 1628323675981, "user_tz": -330, "elapsed": 21801, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="232d5554-491d-45f9-90d0-aeda16389a73"
df["properties_temp"] = df["properties"].str.split("|").map(set)
df["rating"] = densify(RATING_MAP, df["properties_temp"])
df["stars"] = densify(STAR_MAP, df["properties_temp"])
df["cat"] = densify(HOTEL_CAT, df["properties_temp"])
df['item_id'] = df['item_id'].astype('str')
df = df.drop(columns=["properties_temp"])
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="sNn6GZjpqO8o" executionInfo={"status": "ok", "timestamp": 1628323886384, "user_tz": -330, "elapsed": 1216, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1f7738c2-50d6-4f7e-b1aa-427ef6662134"
df2 = pd.read_parquet('./data/silver/item_clicks.parquet.gzip')
df2.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Ht8gkft2kVun" executionInfo={"status": "ok", "timestamp": 1628324065533, "user_tz": -330, "elapsed": 3001, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f808cdc5-4e7e-42e6-d7f3-f896c5099af8"
merged = pd.merge(df,df2,on="item_id",how="left")
merged = merged.fillna(value=0)
merged.head()
```

```python id="z1trlMuBrkla"
merged['cat'] = merged['cat'].astype('str')
```

```python id="coD2YjaVq4RW"
merged.to_parquet('./data/silver/item_metadata_dense.parquet.gzip', compression='gzip')
```

<!-- #region id="4uSl6IJH9Zp4" -->
**We have a Trivago dataset with user sessions. What task we are trying to solve?**

We will leverage user profile and session data to create a model which estimates the probability that a person will click-out on a hotel, and use this probability to sort search results.

**What do my labels look like?**
    - Positive labels (1) are click-out events and negative labels are ones which a user did not click (0). So, because there are 25 resuls and one click-out event, there will be 1 positive example for 24 negatives.

**What information do we have while crafting features?**
    - all of the information which precedes the clickout action in the session.
    - entire user profile which a user has previously looked at.

Extract labels and features as follows:
    - for each session id: create an ordered list of sessions.
    - for each user id: construct a profile of all sessions done by that user
    - roll through each session
<!-- #endregion -->

```python id="0-z0FWfKr7HU"
# helpers --
def safe_mean(input: List[float]) -> float:
    if len(input) == 0:
        return 0.0
    return sum(input) / len(input)

def sigmoid(x: float) -> float:
  return 1 / (1 + math.exp(-x))

def jaccard(lhs: Set[str], rhs: Set[str]) -> float:
    isect_size = sum(1 for x in lhs if x in rhs)
    union_size = len(lhs.union(rhs))
    return isect_size / union_size

def hotel_sim(id_x: str, id_y: str) -> float:
    """
    Look up the ids of the hotels from the dict and return the jaccard similarity of their properties
    """
    try:
        x_props = id_to_hotel[id_x].properties
        y_props = id_to_hotel[id_y].properties
        return jaccard(x_props,y_props)
    except KeyError:
        # fix these weird dictionary misses
        # print(id_x,id_y)
        return 0

def hotel_sim_embed(id_x: str, id_y: str) -> float:
    """
    Look up the ids of the hotels from the dict and return their embeddding similarity
    """
    try:
        sim: float = w2v_model.wv.similarity(id_x, id_y)
        return sim
    except KeyError:
        return 0.0


# First define classes to make handling data a little easier.
@dataclass
class Hotel:
    item_id: str
    properties: Set[str]
    cat: str
    rating: float
    stars: float
    ctr: float
    ctr_prob: float


@dataclass
class Interaction:
    timestamp: int
    action_type: str
    action_on: str
    is_clickout: bool
    # if it's a clickout event we have more data.
    # what's the best way to say this with dataclasses?
    impressions: List[str]
    prices: List[float]

    def get_clicked_idx(self) -> int:
        """
        Helper to get the index of the clicked item, which is helpful for feature extraction.
        Be careful with this method for 2 reasons:
        1. if it's called on an interaction instance which is not a clickout, raise an error.
        2. using this method improperly in the feature extractor can generate the label for the current prediction task.
            for example: adding this line gives us an auc of 1.
            cheat = (choice_idx == current_clickout.get_clicked_idx())
        """
        assert(self.is_clickout)
        idx: int
        try:
            idx = self.impressions.index(self.action_on)
        except ValueError:
            idx = -1
        return idx


@dataclass(unsafe_hash=True) #TypeError: unhashable type: 'Session' just don't use me as a key
class Session:
    start: int
    user_id: str
    session_id: str
    interactions: List[Interaction]


    def set_user_id(self,num: int) -> "Session":
        """
        This helps us with the *extra training data* strategy where we want copies of
        our session data which don't align with user profiles -- to prevent over-learning
        user-based features while training.
        """
        new_user_id: str = self.user_id + "abc" + str(num)
        new_session_id: str = self.session_id + "abc" + str(num)
        new_session: Session = Session(self.start,new_user_id,new_session_id,self.interactions)
        return new_session

    def append_interaction(self, o: Interaction) -> None:
        """ helper """
        self.interactions.append(o)


# makes less sense to make a dataclass here because user profiles change often
class UserProfile:

    def __init__(self, id: str):
        self.user_id = id
        self.sessions: Dict[str,Session] = {}
        self.unique_interactions: Set[str] = set([])

    def update(self, session_id: str, interaction: Interaction) ->  None:
        """
        Update a user profile requires two things.
        - First: add the reference (action_on) string to the unique_interactions set.
        - Second: find the session which the interaction belongs to and append it. If it doesn't
            exist make a new session with the interaction as the first entry
        """
        if interaction.action_on != "":
            self.unique_interactions.add(interaction.action_on)

        s: Session
        try:
            s = self.sessions[session_id] # try to find the session
            s.append_interaction(interaction)
        except KeyError:
            # but if it doesn't work, create a new user at that address.
            self.sessions[session_id] = Session(interaction.timestamp, self.user_id,session_id,[interaction])

# this one is a wrapper for training and test data types
@dataclass
class SessionData:
    data: pd.DataFrame
    qids: List[str]
    feature_names: List[str]

    def get_session(self, session_id: str) -> Session:
        """
        Helps me commpute mean reciprocal rank later on.
        """
        # access the session dict, then skip to the corresponding step in that session's interaction list
        s_temp: Session = sids_to_data[session_id]
        return s_temp


def create_session(df: List[Dict[str,str]]) -> Session:
    """
    This function takes a list of interactions and saves it as a session in object format.
    """
    interaction_list: List[Interaction] = []

    for d in df:
        i: Interaction
        is_clickout = ("clickout item" == d["action_type"])
        t = int(d["timestamp"])#datetime.fromtimestamp(int(d["timestamp"]))
        if is_clickout:
            i = Interaction(t,d["action_type"], str(d["reference"]),
            is_clickout, d["impressions"].split("|"), [float(i) for i in d["prices"].split("|")])
        else:
            i = Interaction(t,d["action_type"],str(d["reference"]),is_clickout,[],[])

        interaction_list.append(i)
    return Session(interaction_list[0].timestamp,df[0]["user_id"],df[0]["session_id"],interaction_list)


def extract_features(session: Session, step: int, choice_idx: int) -> Dict[str,Any]:
    """
    Feature extraction for one session step of action type 'clicked out'
    """
    current_clickout = session.interactions[step]
    current_timestamp = current_clickout.timestamp # shorthand I'll use this a lot lot
    current_price = current_clickout.prices[choice_idx]
    current_choice_id = current_clickout.impressions[choice_idx]

    prev_clickouts: List[Interaction] = [o for o in session.interactions[:step] if o.is_clickout]
    # last_clickout is really useful for extracting features
    # -- this will be set to None if there was no clickout
    last_clickout = prev_clickouts[-1] if len(prev_clickouts) else None

    item_exists: bool
    try:
        id_to_hotel[current_choice_id]
        item_exists = True
    except KeyError:
        item_exists = False
        #print("Key Error... {} does not exist in our dict".format(current_choice_id))


    user_exists: bool = True
    if session.user_id not in users.keys():
        user_exists = False

    hotel_sims = [hotel_sim(current_choice_id, o.action_on) for o in session.interactions[:step] if o.action_on.isnumeric()]
    hotel_sims_embed = [hotel_sim_embed(current_choice_id, o.action_on) for o in prev_clickouts] if item_exists else [0]

    user_hotel_unique_sims = [hotel_sim(current_choice_id, hotel_id) for hotel_id in users[session.user_id].unique_interactions] if user_exists else [0]
    user_hotel_unique_sims_embed = [hotel_sim_embed(current_choice_id, hotel_id) for hotel_id in users[session.user_id].unique_interactions] if user_exists else [0]


    dst_shortest_path: float
    try:
        if nx.has_path(G,current_choice_id, session.user_id):
            dst_shortest_path = sigmoid(nx.shortest_path_length(G,current_choice_id, session.user_id, weight = None))
        else:
            dst_shortest_path = sigmoid(100.0)
    except nx.NodeNotFound:
        dst_shortest_path = -1.0

    features: Dict[str,Any] = { #type:ignore
        # session-based features
        "time_since_start": current_timestamp - session.start,
        "time_since_last_clickout": current_timestamp - last_clickout.timestamp if last_clickout else 0,
        "diff_price_mean": current_price - safe_mean(session.interactions[step].prices),
        "last_price_diff": current_price - last_clickout.prices[last_clickout.get_clicked_idx()] if last_clickout else 0,
        "reciprocal_choice_rank": 1 / (choice_idx + 1), # rank starts at 1 index starts at
        # person clicks 1 then 4... maybe they will try 7 next
        "predicted_next_click": 2 * last_clickout.get_clicked_idx() - prev_clickouts[-2].get_clicked_idx() + 1 if len(prev_clickouts) > 1 else last_clickout.get_clicked_idx() + 2 if last_clickout else 1,
        # position to previous clickout position
        "delta_position_last_position": (choice_idx) - last_clickout.get_clicked_idx() if last_clickout else 0,
        # z-score (?) difference between price and average price of clicked hotels
        "avg_price_sim": current_price - safe_mean([o.prices[o.get_clicked_idx()] for o in prev_clickouts]) if last_clickout else 0,
        "prev_impression_sim": jaccard(set(last_clickout.impressions),set(current_clickout.impressions)) if last_clickout else 0,
        # user-item or item-item features --
        # roll through the previous items interacted in the session and average their jaccard similarity to the current item
        "item_item_sim": safe_mean(hotel_sims),
        "item_item_embed_sim": safe_mean(hotel_sims_embed),
        "item_ctr": id_to_hotel[current_choice_id].ctr if item_exists else 0,
        "item_ctr_prob": id_to_hotel[current_choice_id].ctr_prob if item_exists else 0,
        # user-based features (these build on previous sessions or in conjunction with current sessions) --
        "unique_item_interact_by_user_avg": safe_mean(user_hotel_unique_sims),
        "unique_item_interact_by_user_embed_avg": safe_mean(user_hotel_unique_sims_embed),
        "unique_item_interact_by_user_max": max(user_hotel_unique_sims),
        "unique_item_interact_by_user_min": min(user_hotel_unique_sims),
        "path_user_item": dst_shortest_path
    }
    return features


def build_graph(session_ids: List[str]) -> nx.Graph:
    """
    Build a graph for collaborative filtering using a list of session ids.
    """
    G = nx.Graph()

    sessions: List[Session] = [sids_to_data[s_id] for s_id in session_ids]
    for s in sessions:
        clickouts = [o for o in s.interactions if o.is_clickout]
        G.add_edges_from([(o.action_on, s.user_id) for o in clickouts])

    return G


def make_w2v(session_ids: List[str]) -> Word2Vec:
    sessions: List[Session] = [sids_to_data[s_id] for s_id in session_ids]
    sequences: List[str] = []
    for s in sessions:
        clickouts = [o for o in s.interactions if o.is_clickout]
        sequences += [o.impressions for o in clickouts]

    n_cores = multiprocessing.cpu_count()

    # 1. initialise model with params
    w2v_model = Word2Vec(min_count=1,
                     window=4,
                     vector_size=100,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=8,
                     workers=n_cores-1)

    # 2. create the vocabulary
    w2v_model.build_vocab(sequences, progress_per=10000)

    t = time()
    w2v_model.train(sequences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    return w2v_model


def collect(what: str, session_ids: List[str]) -> SessionData:
    """
    This function takes as input a filename and will return a SessionData object containing
    a list of clickout features, a list of labels, and methods to convert each into matrices.
    """
    sessions: List[Session] = [sids_to_data[s_id] for s_id in session_ids]

    print("Rolling through ({}) sessions/creating labeled feature vectors for {} sessions".format(len(sessions), what))

    examples: List[Dict[str,str]] = []
    qids: List[str] = []


    features: Dict[str,Any]

    for s in tqdm(sessions): # for each session -- --
        for step, o in enumerate(s.interactions): # for each interaction in the session

            # if it's of type "clickout", e.g. o.is_clickout
            if o.is_clickout:
                if o.get_clicked_idx() != -1:
                    # create a positive training example and 24 negatives... also extract features for each and add to x
                    for index, choice in enumerate(o.impressions):
                        label = (choice == o.action_on)

                        # feature extraction needs session, interaction info, and the index we were wondering about.
                        # we also pass whether we want to ignore user features. (defaults to false)
                        features = extract_features(s,step,index)
                        features["is_advantaged_user"] = 0 if s.user_id not in users.keys() else 1 if len(users[s.user_id].sessions) > 2 else 0
                        q_id = "{}/{}".format(s.session_id, step)
                        qids.append(q_id)
                        examples.append({"q_id": q_id, "choice_idx":index, **features, "y":label})

                    G.add_edge(o.action_on, s.user_id) # then add the edge to the graph.. aka build it in real time.

            # try to add each interaction to a user profile
            uid = s.user_id # (use this many times)
            user: UserProfile
            try:
                user = users[uid]
            except KeyError:
                # but if it doesn't work, create a new user at that address.
                user = UserProfile(uid)
                users[uid]= user
            user.update(s.session_id, o)

    feature_names: List[str] = [i for i in features.keys()]
    return SessionData(pd.DataFrame.from_records(examples), qids, feature_names)
```

```python id="iAjQynoXsZEZ"
def load_session_dict(dfpath: str) -> Dict[str,Session]:
    """
    This function be called on both train and test sets, so that I can get both a list of unique session ids to collect later,
    and a way to access these sessions (in collect) in  O(1) time.
    """

    sessions: List[Session] = []
    # nrows=1_000 for my laptop's sake
    df_interactions = pd.read_parquet(dfpath)
    # apply the "save_session" function to each grouped item/session
    # but first turn each group from a df into a list of dictionaries
    A = lambda x: sessions.append(create_session(x.to_dict("records"))) #type:ignore
    df_interactions.groupby(by="session_id").apply(A)

    sessions_dict: Dict[str,Session] = {}
    for s in sessions:
        sessions_dict[s.session_id] = s
    return sessions_dict
```

```python id="e-COHgCZslxy"
# globals
id_to_hotel: Dict[str,Hotel] = {}
users: Dict[str,UserProfile] = {} # this map ids to UserProfile objects (which are just sets of sessions)
# create an empty graph - nodes and edges will be built in real-time because we want to prevent crazy amounts of overfitting
G = nx.Graph()
```

<!-- #region id="YWf5cL51tMEK" -->
Generate features, splits, and quickly-loadable matrix files. There are two tasks:
1. make a parquet file for user-aware train/eval later
2. make another file to be used for a user-blind recommender. This is a little tricky because I still need to know if a user belongs to a 'advantaged' or 'disadvantaged' group. Pass in a parameter to     feature extraction to ignore the user ?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pljQeyQ0tbE0" outputId="5ecca977-e8d8-4701-9c60-d7e086b8ab22"
print("Loading Session Data")
sessions_train = load_session_dict('/content/tempdata/train.parquet.gzip')
gc.collect()
sessions_vali = load_session_dict('/content/tempdata/validation.parquet.gzip')
gc.collect()
sessions_test = load_session_dict('/content/tempdata/confirmation.parquet.gzip')
gc.collect()

session_ids_train = list(sessions_train.keys())
session_ids_vali = list(sessions_vali.keys())
session_ids_test = list(sessions_test.keys())

sids_to_data = {**sessions_train, **sessions_vali,**sessions_test}
```

```python colab={"base_uri": "https://localhost:8080/", "height": 221} id="gxWl0nDYstlB" executionInfo={"status": "ok", "timestamp": 1628325053924, "user_tz": -330, "elapsed": 45805, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8c83b795-79de-46e1-8ffa-cd6d60e28924"
print("Loading up item features")
hotel_features_df = pd.read_parquet('./data/silver/item_metadata_dense.parquet.gzip') #type:ignore
hotel_features_df["item_id"] = hotel_features_df["item_id"].apply(str) # make sure the id is of type str, not int
hotel_features_df["properties"] = hotel_features_df["properties"].str.split("|").map(set)
d: Dict = hotel_features_df.to_dict("records")
for h in d: # loop over the dictionary version of this df
    id_to_hotel[h["item_id"]] = Hotel(**h)
hotel_features_df.head()
```

```python id="HjYsAhpevG-s"
# get w2v
w2v_model_path = './model/query_sim_w2v.model'
if not path.exists(w2v_model_path):
    make_w2v(session_ids_train).save(w2v_model_path)
w2v_model = Word2Vec.load(w2v_model_path)
```

```python id="O93grmoRsWjE"
train = collect("train",session_ids_train)
session_ids_vali_test = session_ids_test + session_ids_vali
session_ids_vali_test.sort(key=lambda x: sids_to_data[x].start)
valitest = collect("vali/test", session_ids_vali_test)
#test = collect("test", session_ids_test)

train.data["grp"] = 0
vali_or_test = lambda x: 2 if x.split("/")[0] in session_ids_test else 1
valitest.data["grp"] = valitest.data["q_id"].apply(vali_or_test)
#test.data["grp"]  = 2


frames: List[pd.DataFrame] = [train.data, valitest.data]
df_out = pd.concat(frames)
print("Writing a df with pyarrow. It has dimensions {}. ".format(df_out.shape))

# dump dataset and put my experiments in a different file
df_out.to_parquet("data/trivago/data_all.parquet", engine="pyarrow")
```
