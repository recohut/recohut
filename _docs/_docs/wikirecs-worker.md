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

```python id="Jep4VQyz3ZzE" executionInfo={"status": "ok", "timestamp": 1625763697015, "user_tz": -330, "elapsed": 908, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
project_name="reco-wikirecs"; branch="master"; account="sparsh-ai"
```

```python colab={"base_uri": "https://localhost:8080/"} id="ailHP5gi3ZzP" executionInfo={"status": "ok", "timestamp": 1625763724038, "user_tz": -330, "elapsed": 6338, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e864fccf-5896-4abb-9f02-88c73ea22bb4"
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

```python colab={"base_uri": "https://localhost:8080/"} id="WWDDXhuK9klF" executionInfo={"status": "ok", "timestamp": 1625729526223, "user_tz": -330, "elapsed": 743, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b18d4705-83dd-402c-a6fc-3f3461c92d5b"
%cd /content/reco-wikirecs/
```

```python colab={"base_uri": "https://localhost:8080/"} id="6HVnZkVW3ZzQ" executionInfo={"status": "ok", "timestamp": 1625762573026, "user_tz": -330, "elapsed": 10791, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="84c27b46-9194-4aa6-8f63-90317b2a696c"
!git status
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="LLMOakVK7lZg" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625763802144, "user_tz": -330, "elapsed": 75926, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3272b0d6-ba68-4434-fd90-528e9ed01c06"
!pip install -r requirements.txt
```

<!-- #region id="wlWx6OrY3n_A" -->
---
<!-- #endregion -->

```python id="c_uAz5OS4sm3" colab={"base_uri": "https://localhost:8080/", "height": 17} executionInfo={"status": "ok", "timestamp": 1625763803215, "user_tz": -330, "elapsed": 1077, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d0464559-970d-4d7e-b0d6-69df582dcd5f"
import yaml
import os

from wiki_pull import *

%matplotlib inline
%reload_ext autoreload
%autoreload 2

from itables.javascript import load_datatables
load_datatables()
```

```python id="iT-JOJiF3G8M" executionInfo={"status": "ok", "timestamp": 1625763803216, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 521} id="tQPeT_sk7Mud" executionInfo={"status": "ok", "timestamp": 1625755605162, "user_tz": -330, "elapsed": 432198, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a248ea52-7dd1-4633-bd10-bc24d0201250"
get_sample_of_users(config['edit_lookback'], config['outfile'])
```

```python colab={"base_uri": "https://localhost:8080/"} id="qzPsvEDtEgwm" executionInfo={"status": "ok", "timestamp": 1625763805196, "user_tz": -330, "elapsed": 578, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="05c3d3b1-1bbf-4d24-ce7c-de272597b734"
!wget https://github.com/sparsh-ai/reco-wikirecs/raw/033f9d18a9a791b5ffa173b6dda9a2e0ac76e311/sampled_users_2021_07_08.csv
```

```python id="iRqIusXhEsUJ" executionInfo={"status": "ok", "timestamp": 1625763939656, "user_tz": -330, "elapsed": 665, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
import numpy as np
import requests
import time
import os
from tqdm import tqdm
from pyarrow import feather


def get_recent_changes(N):
    S = requests.Session()

    t = tqdm(total=N, position=0, leave=True)

    URL = "https://en.wikipedia.org/w/api.php"

    PARAMS = {
        "format": "json",
        "rcprop": "title|ids|sizes|flags|user|userid|timestamp",
        "rcshow": "!bot|!anon|!minor",
        "rctype": "edit",
        "rcnamespace": "0",
        "list": "recentchanges",
        "action": "query",
        "rclimit": "500",
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    RECENTCHANGES = DATA["query"]["recentchanges"]
    all_rc = RECENTCHANGES

    i = 500
    t.update(500)
    while i <= N:
        last_continue = DATA["continue"]
        PARAMS.update(last_continue)
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        RECENTCHANGES = DATA["query"]["recentchanges"]
        all_rc.extend(RECENTCHANGES)
        i = i + 500
        t.update(500)

    if len(all_rc) > N:
        all_rc = all_rc[:N]

    return all_rc


def get_sample_of_users(edit_lookback, outfile=None):
    """Get a sample of recently active users by pulling the most recent N edits
    Note that this will be biased towards highly active users.
    Args:
        edit_lookback: The number of edits to go back.
        outfile: Pickle file path to write the user list to
    Returns:
        Dataframe with user and user id columns
    """
    df = get_recent_changes(edit_lookback)

    # Drop missing userid entries
    df = pd.DataFrame(df).dropna(subset=["userid"])

    print("Earliest timestamp: {}".format(df.timestamp.min()))
    print("Latest timestamp: {}".format(df.timestamp.max()))
    print("Number of distinct users: {}".format(len(df.user.unique())))
    print(
        "Mean number of edits per user in timeframe: %.2f"
        % (len(df) / len(df.user.unique()))
    )
    print("Number of distinct pages edited: {}".format(len(df.pageid.unique())))
    print(
        "Mean number of edits per page in timeframe: %.2f"
        % (len(df) / len(df.pageid.unique()))
    )

    # Deduplicate to get
    sampled_users = df.loc[:, ["user", "userid"]].drop_duplicates()

    # Remove RFD
    sampled_users = sampled_users[np.invert(sampled_users.user == "RFD")]
    sampled_users = sampled_users.reset_index(drop=True)

    if outfile:
        sampled_users.to_csv(outfile, index=False)

    return sampled_users


def get_edit_history(
    userid=None, user=None, latest_timestamp=None, earliest_timestamp=None, limit=None):
    """For a particular user, pull their whole history of edits.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """

    S = requests.Session()
    S.headers.update(
        {"User-Agent": "WikiRecs (danielrsaunders@gmail.com) One-time pull"}
    )

    URL = "https://en.wikipedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "format": "json",
        "ucnamespace": "0",
        "list": "usercontribs",
        "ucuserids": userid,
        "ucprop": "title|ids|sizediff|flags|comment|timestamp",
        "ucshow=": "!minor|!new",
    }
    if latest_timestamp is not None:
        PARAMS["ucstart"] = latest_timestamp
    if earliest_timestamp is not None:
        PARAMS["ucend"] = earliest_timestamp
    if user is not None:
        PARAMS["ucuser"] = user
    if userid is not None:
        PARAMS["ucuserid"] = userid

    PARAMS["uclimit"] = 500

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    if "query" not in DATA:
        print(DATA)
        raise ValueError

    USERCONTRIBS = DATA["query"]["usercontribs"]
    all_ucs = USERCONTRIBS
    i = 500
    while i < 100000:
        if "continue" not in DATA:
            break
        last_continue = DATA["continue"]
        PARAMS.update(last_continue)
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        USERCONTRIBS = DATA["query"]["usercontribs"]
        all_ucs.extend(USERCONTRIBS)
        i = i + 500

    return all_ucs


def pull_edit_histories(
    sampled_users_file,
    edit_histories_file_pattern,
    users_per_chunk,
    earliest_timestamp,
    start=0):
    histories = []
    cols = ["userid", "user", "pageid", "title", "timestamp", "sizediff"]
    sampled_users = pd.read_csv(sampled_users_file)
    sampled_users.loc[:, "userid"].astype(int)

    sampled_users = sampled_users.reset_index()

    # Iterate through all the users in the list
    for i, (user, userid) in tqdm(
        iterable=enumerate(
            zip(sampled_users["user"][start:], sampled_users["userid"][start:]),
            start=start),
        total=len(sampled_users)): 

        # Get the history of edits for this userid
        thehistory = get_edit_history(
            userid=int(userid), earliest_timestamp=earliest_timestamp
        )

        # If no edits, skip
        if len(thehistory) == 0:
            continue

        thehistory = pd.DataFrame(thehistory)

        # Remove edits using automated tools by looking for the word "using" in the comments
        try:
            thehistory = thehistory[
                np.invert(thehistory.comment.astype(str).str.contains("using"))
            ]
        except AttributeError:
            continue

        if len(thehistory) == 0:
            continue

        histories.append(thehistory.loc[:, cols])

        # if np.mod(i, 50) == 0:
        #     print(
        #         "Most recent: {}/{} {} ({}) has {} edits".format(
        #             i, len(sampled_users), user, int(userid), len(thehistory)
        #         )
        #     )

        # Every x users save it out, for the sake of ram limitations
        if np.mod(i, users_per_chunk) == 0:
            feather.write_feather(
                pd.concat(histories), edit_histories_file_pattern.format(i)
            )

            histories = []
      
    # Get the last few users that don't make up a full chunk
    feather.write_feather(pd.concat(histories), edit_histories_file_pattern.format(i))
```

```python colab={"base_uri": "https://localhost:8080/"} id="-vNLJxSLB2m8" outputId="f87e9c2d-a0f7-4a45-e9d4-807f85bc07eb"
pull_edit_histories(
    config['outfile'],
    os.path.join(config['file_save_path'],config['edit_histories_file_pattern']),
    config['users_per_chunk'],
    config['earliest_timestamp'],
    start=10000,
    )
```

```python id="EwuprV5I3MIw"
!rm -r /content/drive/MyDrive/TempData/WikiRecs/edit_histories_2021_07_08_9*.feather
```

<!-- #region id="VjRxyUhCePFT" -->
### Baseline models

| Model | Type | Description |
| - | -:| ---:|
| Popularity | Rule-based | Most popular over the past year |
| Recent | Rule-based | Most recently edited by this user |
| Frequent | Rule-based | Most frequently edited by this user in the last year |
| BM25 | Collaborative-filtering | Okapi BM25, a simple variation on Jaccard similarity with TF-IDF that often has much better results |
| ALS | Collaborative-filtering | Alternating Least Squares matrix factorization of implicit training data, with BM25 pre-scaling |
<!-- #endregion -->

```python id="3g07BOYfbSUl"

```
