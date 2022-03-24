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

<!-- #region id="YrsgmsHmaDiA" -->
For fun and learning, I am building a recommender system for Wikipedia editors (aka Wikipedians) to suggest what article they should edit next, called WikiRecs. I’m especially interested in the interplay of timing and sequence of actions to predict what someone will do next in an app or on a website. There are several cutting edge AI techniques I am exploring to fully use all those sources of information, but from my background in science I knew that the first priority is trustworthy measurements, in this case via offline evaluation.
<!-- #endregion -->

<!-- #region id="ZC1IKW9faGcV" -->
Offline evaluation requires figuring out how to make a valid test set, what metrics to compute, and what baseline models to compare it to. I’ll give my tips for these below, focusing on what is practical for industry rather than the higher bar of academic publication, but first a bit about my dataset. Wikipedia doesn’t have star ratings or thumbs up/thumbs down, so this is inherently an implicit recommendation problem, where my signal of interest is the choice to edit an article. Based on a user’s history of edits, and any other information I can pull in, I want to build a picture of their interests, in particular their recent interests, and learn to rank a personalized list of articles.
<!-- #endregion -->

```python id="4YJJHWgwhzKc"
!pip install -q implicit
!pip install -q wikipedia
!pip install -q umap
!pip install -q itables
```

```python id="4bD-GNLNavbB"
import pandas as pd
import numpy as np
import requests
import time
import argparse
from tqdm import tqdm
from pyarrow import feather

import pandas as pd
import numpy as np
import requests
import argparse
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import logging
import wikipedia
import requests
import os

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, coo_matrix
from tqdm.auto import tqdm
import umap
import pickle
import collections
import plotly.express as px
from pyarrow import feather
import itertools
from itables import show
import matplotlib

import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
import collections
from sklearn.metrics import ndcg_score, label_ranking_average_precision_score
from tqdm.auto import tqdm

from tqdm.auto import tqdm
import itertools

import implicit
from implicit.nearest_neighbours import bm25_weight, BM25Recommender

%matplotlib inline

pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 100)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 17} id="MTbcVaO7h8cM" executionInfo={"status": "ok", "timestamp": 1625721346225, "user_tz": -330, "elapsed": 521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d6609b16-8c7e-4dd8-b40d-e86423473898"
from itables.javascript import load_datatables
load_datatables()
```

```python cellView="form" id="J3U5Zr9RiFWR"
#@markdown utils
class Timer:
    def __init__(self, segment_label=""):
        self.segment_label = segment_label

    def __enter__(self):
        print(" Entering code segment {}".format(self.segment_label))
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(" Code segment {} took {}".format(self.segment_label, self.interval))


def load_pickle(filename):
    with open(filename, "rb") as fh:
        return pickle.load(fh)


def save_pickle(theobject, filename):
    with open(filename, "wb") as fh:
        pickle.dump(theobject, fh)


def conv_wikipedia_timestamp(timestamp):
    return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")


# Look at one user's edits
def print_user_history(all_histories, user=None, userid=None):

    if user is not None:
        edits = all_histories[all_histories.user == user].copy()
    elif userid is not None:
        edits = all_histories[all_histories.userid == userid].copy()
    else:
        raise ValueError("Either user or userid must be non-null")

    if len(edits) == 0:
        print("User not found")

    last_page = -1
    last_date = ""
    for i in range(len(edits))[::-1]:
        row = edits.iloc[i]
        dt = conv_wikipedia_timestamp(row.first_timestamp)
        # Every day of edits, print out the date and reset the "last edited"
        if str(dt.date()) != last_date:
            print(dt.date())
            last_date = str(dt.date())
            last_page = -1

        # Only output when they edit a new page (print only the timestamp of a first of a string of edits)
        if row.pageid != last_page:
            print(" {} {}".format(str(dt.time()), row.title.strip()))
            last_page = row.pageid


def dataframe_set_subtract(df, to_subtract, by_cols=None):
    original_cols = df.columns
    if by_cols is None:
        merge_df = df.merge(to_subtract, how="left", indicator=True)
    else:
        merge_df = df.merge(to_subtract, how="left", indicator=True, on=by_cols)

    return df.loc[merge_df._merge == "left_only", original_cols]


def recall_curve(test_set, recs, max_k, userid_subset=None):
    recall_vals = []
    for K in np.arange(max_k) + 1:
        recall_vals.append(recall(test_set, recs, K, userid_subset))

    return recall_vals


def recall(test_set, recs, K=10, userid_subset=[]):
    """For a test set, compute the % of users who have a hit in the top K.
    Args:
        test_set: DF with an entry for each user with the target edit-to-be-predicted
        recs: Dict by userid of lists of pageid recs
        K: Number of recs to consider when looking for a hit
        userid_subset: Only compute for the userids in this list
    Returns:
        float of the mean number of test entries with hits in the top K
    """

    if userid_subset is None:
        userid_subset = []
    userid_subset = set(userid_subset)

    hits = [
        pageid in recs[userid][:K]
        for pageid, userid in zip(test_set.pageid, test_set.userid)
        if (len(userid_subset) == 0) or (userid in userid_subset)
    ]

    return np.mean(hits)


def prep_for_metrics(test_set, recs, K, userid_subset=None):
    test_set = test_set.drop(columns=["recs"], errors="ignore")
    test_set = test_set.merge(
        pd.DataFrame(
            [(u, recs[u]) for u in test_set.userid], columns=["userid", "recs"]
        ),
        on="userid",
    )
    if userid_subset is None:
        selected_rows = [True] * len(test_set)
    else:
        selected_rows = test_set.userid.isin(userid_subset)

    y_true = [
        (p == r[:K]).astype(int)
        for p, r in zip(
            test_set[selected_rows].pageid.values, test_set[selected_rows].recs.values
        )
    ]
    dummy_y_score = len(test_set[selected_rows]) * [list(range(K))[::-1]]

    test_set = test_set.drop(columns=["recs"])

    return y_true, dummy_y_score


def ndcg(test_set, recs, K=20, userid_subset=None):
    y_true, dummy_y_score = prep_for_metrics(test_set, recs, K, userid_subset)

    ## Print the individual scores
    # for yt in y_true:
    #     print(
    #         (
    #             yt,
    #             ndcg_score(
    #                 np.array(yt, ndmin=2), np.array(list(range(K))[::-1], ndmin=2)
    #             ),
    #         )
    #     )
    return ndcg_score(y_true, dummy_y_score)


def mrr(test_set, recs, K=20, userid_subset=None):
    y_true, dummy_y_score = prep_for_metrics(test_set, recs, K, userid_subset)
    return label_ranking_average_precision_score(y_true, dummy_y_score)


def get_recs_metrics(
    test_set, recs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i
):
    return {
        "recall": 100 * recall(test_set, recs, K),
        "ndcg": ndcg(test_set, recs, K),
        "resurfaced": 100 * prop_resurface(recs, K, implicit_matrix, i2p, u2i),
        "recall_discover": 100
        * recall(test_set, recs, K, userid_subset=discovery_userids),
        "recall_resurface": 100
        * recall(test_set, recs, K, userid_subset=resurface_userids),
        "ndcg_discover": ndcg(test_set, recs, K, userid_subset=discovery_userids),
        "ndcg_resurface": ndcg(test_set, recs, K, userid_subset=resurface_userids),
    }


def prop_resurface(recs, K=10, implicit_matrix=None, i2p=None, u2i=None):
    """What proportion of the top K recs are resurfaced pages (already edited by user)?
    Args:
    Returns:
        float of the mean number of resurfaced pages in the top K recs
    """
    prop_resurface = []
    for userid in recs.keys():
        past_pages = [i2p[i] for i in implicit_matrix[:, u2i[userid]].nonzero()[0]]
        rec_pages = recs[userid][:K]
        prop_resurface.append(np.mean(np.isin(rec_pages, past_pages)))

    return np.mean(prop_resurface)


def display_recs_with_history(
    recs,
    userids,
    histories_test,
    histories_train,
    p2t,
    u2n,
    recs_to_display=5,
    hist_to_display=10,
):
    """Return a dataframe to display showing the true next edit for each user along
    with the top n recs and the history
    Args:
        recs: List of recs per user
        userids: List of users to show this for
        histories_test: The true next edit for all users
        histories_train: The history of all users minus the test set
        recs_to_display: How many of the top recs to show
        hist_to_display: How many edits to look back
        p2t: Page id to title
        u2n: User id to username
    Returns:
        dataframe for display formatting all this information
    """
    display_dicts = collections.OrderedDict()

    index_labels = (
        ["True value"]
        + ["Rec " + str(a + 1) for a in range(recs_to_display)]
        + ["-"]
        + ["Hist " + str(a + 1) for a in range(hist_to_display)]
    )
    for u in userids:
        real_next = histories_test.loc[histories_test.userid == u].pageid.values[0]
        user_history = list(
            histories_train.loc[histories_train.userid == u].title.values[
                :hist_to_display
            ]
        )

        # If we don't have enough history, pad it out with "-"
        if len(user_history) < hist_to_display:
            user_history = user_history + ["-"] * (hist_to_display - len(user_history))
        display_dicts[u2n[u]] = (
            [p2t[real_next]]
            + [p2t[a] for a in recs[u][:recs_to_display]]
            + ["----------"]
            + user_history
        )

    return pd.DataFrame(display_dicts, index=index_labels)


def get_resurface_discovery(histories_train, histories_test):
    d = histories_train.merge(
        histories_test[["userid", "pageid"]], on="userid", suffixes=["", "target"]
    )
    d.loc[:, "is_target"] = d.pageid == d.pageidtarget
    d = d.groupby("userid").is_target.any()
    resurface_userids = d[d].index.values
    discovery_userids = d[~d].index.values

    return (resurface_userids, discovery_userids)
```

```python id="lv9GfnmhfjrB"
edit_lookback = 100 # How many edits to look back in Wikipedia history
suffix = "2021_07_08"
outfile = "sampled_users_{}.csv".format(suffix) # CSV file to write the resulting user names and ids to
file_save_path = "/content"
edit_histories_file_pattern = "edit_histories_"+suffix+"_{}.feather" # Output for edit histories (needs a {} for where to place the number)
users_per_chunk = 10 # How many users to pull before dumping to a file (for the sake of ram)
earliest_timestamp = "2020-07-08T02:59:44Z" # How far back to go (format e.g. 2021-02-11T05:35:34Z)
```

<!-- #region id="OW2hzVgJc1cx" -->
Using Wikipedia’s publicly available API, I sampled 1 million of the most recent edits for all users of English Wikipedia, which turned out to be only a few days worth of data. Then for each editor represented in the sample I pulled their last 1 year of edits, 21 million datapoints in total. Sampling like this only captures users who were active in this period — for one thing this will somewhat over-represent the most active editors, but that’s the population I am most interested in serving. I did some basic cleaning, e.g. removing bots and users with too many or too few edits, and then built my training set, test set, and baseline recommenders. The final training set covered 3 million Wikipedia pages (out of about 6.3 million total) and 32,000 users.
<!-- #endregion -->

```python cellView="form" id="VjRxyUhCePFT"
#@markdown def get_recent_changes(N)
def get_recent_changes(N):
    S = requests.Session()

    t = tqdm(total=N)

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
```

```python cellView="form" id="d9kmR5hTecLr"
#@markdown def get_sample_of_users(edit_lookback, outfile=None)
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
```

```python colab={"base_uri": "https://localhost:8080/", "height": 538} id="JkCYCiMjeG3t" executionInfo={"status": "ok", "timestamp": 1625720455573, "user_tz": -330, "elapsed": 807, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4197c46a-bd29-478b-a1c3-bb8d9593c9ae"
get_sample_of_users(edit_lookback, outfile)
```

```python cellView="form" id="xIJ4nsOta6vw"
#@markdown For a particular user, pull their whole history of edits
def get_edit_history(
    userid=None, user=None, latest_timestamp=None, earliest_timestamp=None, limit=None
):
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
```

```python cellView="form" id="3g07BOYfbSUl"
#@markdown For the given list of users, pull edit histories
def pull_edit_histories(
    sampled_users_file,
    edit_histories_file_pattern,
    users_per_chunk,
    earliest_timestamp,
    start=0,
):
    histories = []
    cols = ["userid", "user", "pageid", "title", "timestamp", "sizediff"]
    sampled_users = pd.read_csv(sampled_users_file)
    sampled_users.loc[:, "userid"].astype(int)

    sampled_users = sampled_users.reset_index()

    # Iterate through all the users in the list
    for i, (user, userid) in tqdm(
        iterable=enumerate(
            zip(sampled_users["user"][start:], sampled_users["userid"][start:]),
            start=start,
        ),
        total=len(sampled_users),
        initial=start,
    ):
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

        if np.mod(i, 50) == 0:
            print(
                "Most recent: {}/{} {} ({}) has {} edits".format(
                    i, len(sampled_users), user, int(userid), len(thehistory)
                )
            )

        # Every x users save it out, for the sake of ram limitations
        if np.mod(i, users_per_chunk) == 0:
            feather.write_feather(
                pd.concat(histories), edit_histories_file_pattern.format(i)
            )

            histories = []

    # Get the last few users that don't make up a full chunk
    feather.write_feather(pd.concat(histories), edit_histories_file_pattern.format(i))
```

```python id="r0ndk8FHasiP"
pull_edit_histories(
    outfile,
    edit_histories_file_pattern,
    users_per_chunk,
    earliest_timestamp,
    start=0,
    )
```

```python id="CURtDxUflnhK"
all_histories = []
for fname in os.listdir(file_save_path):
    if 'feather' in  fname:
        all_histories.append(feather.read_feather(fname))

all_histories = pd.concat(all_histories, ignore_index=True)
feather.write_feather(all_histories, "all_histories_{}.feather".format(suffix))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 462} id="G5J0gvODtgDT" executionInfo={"status": "ok", "timestamp": 1625724165642, "user_tz": -330, "elapsed": 506, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="12bcf168-b07f-47ee-c967-52c521b3bf83"
all_histories.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="R9CApu96nnnE" executionInfo={"status": "ok", "timestamp": 1625724138876, "user_tz": -330, "elapsed": 654, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bb6eb39f-1825-4240-e551-79600a09b3e9"
all_histories = feather.read_feather("all_histories_{}.feather".format(suffix))

print(all_histories.columns, len(all_histories), len(all_histories.pageid.unique()))
```

<!-- #region id="cbWTiNiSrUD0" -->
### EDA
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 530} id="SGrV5W1znv-J" executionInfo={"status": "ok", "timestamp": 1625722870895, "user_tz": -330, "elapsed": 1467, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1719b49d-abac-442b-9d46-803d282fb7dd"
# Look at the distribution of edit counts
edit_counts = all_histories.groupby('userid').userid.count().values

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
sns.distplot(edit_counts,kde=False,bins=np.arange(0,20000,200))
plt.xlabel('Number of edits by user')
plt.subplot(1,2,2)
sns.distplot(edit_counts,kde=False,bins=np.arange(0,200,1))
plt.xlim([0,200])
plt.xlabel('Number of edits by user')
num_counts = len(edit_counts)
print("Median edit counts: %d" % np.median(edit_counts))
thres = 5
over_thres = np.sum(edit_counts > thres)
print("Number over threshold %d: %d (%.f%%)" % (thres, over_thres, 100*over_thres/num_counts))
```

```python id="09CmQa64o-CS"
# Most edits by user
all_histories.groupby(['userid','user']).userid.count().sort_values(ascending=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="yE9_qDk8o-AN" executionInfo={"status": "ok", "timestamp": 1625722998809, "user_tz": -330, "elapsed": 539, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c9f34d26-fa4d-45ac-c979-4e78c705af65"
# Find the elbow in number of edits
plt.plot(all_histories.groupby(['userid','user']).userid.count().sort_values(ascending=False).values)
# plt.ylim([0,20000])
```

```python colab={"base_uri": "https://localhost:8080/"} id="-oCTGgfco98l" executionInfo={"status": "ok", "timestamp": 1625723040074, "user_tz": -330, "elapsed": 1140, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="88ba1a59-c3be-48c8-b82b-b375207d186a"
# What are the most popular pages (edited by the most users)
page_popularity = all_histories.drop_duplicates(subset=['title','user']).groupby('title').count().user.sort_values()

page_popularity.iloc[-10:].iloc[::-1]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="xvMvQp-8rXVT" executionInfo={"status": "ok", "timestamp": 1625723699252, "user_tz": -330, "elapsed": 2003, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1f260f62-34fd-463c-b012-30ac28005d4a"
oneuser = get_edit_history(user="Thornstrom",
                            latest_timestamp="2021-05-28T22:02:09Z",
                            earliest_timestamp="2020-05-28T22:02:09Z")

cols = ['userid', 'user', 'pageid', 'title', 'timestamp', 'sizediff']
oneuser = pd.DataFrame(oneuser).loc[:,cols]
oneuser
```

<!-- #region id="ElOMDp6frRo1" -->
### Data cleaning
<!-- #endregion -->

```python id="YCJA3SvHpa6q"
# Remove consecutive edits and summarize runs
def remove_consecutive_edits(df):
    c = dict(zip(df.columns, range(len(df.columns))))
    
    keyfunc = lambda x: (x[c['userid']],x[c['pageid']])
    first_and_last = lambda run: [run[0][c['userid']],
                                run[0][c['user']],
                                run[0][c['pageid']],
                                run[0][c['title']],
                                run[-1][c['timestamp']],
                                run[0][c['timestamp']],
                                sum([abs(r[c['sizediff']]) for r in run]),
                                len(run)]
    d  = df.values.tolist()
    return pd.DataFrame([first_and_last(list(g)) for k,g in itertools.groupby(d, key=keyfunc)], 
                        columns=['userid', 'user', 'pageid', 'title', 'first_timestamp', 'last_timestamp','sum_sizediff','consecutive_edits'])
                        
clean_histories = remove_consecutive_edits(all_histories)
```

```python colab={"base_uri": "https://localhost:8080/"} id="UJbKzOripa3n" executionInfo={"status": "ok", "timestamp": 1625723188945, "user_tz": -330, "elapsed": 1168, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="483e8120-f601-4a24-87a1-47fdd5a21b3d"
# Remove top N most popular pages
TOPN = 20
popularpages = all_histories.drop_duplicates(subset=['title','pageid','userid']).groupby(['title','pageid']).count().user.sort_values()[-TOPN:]

before_count = len(all_histories)

popular_pageids = popularpages.index.get_level_values(level='pageid').values
is_popular_page_edit = clean_histories.pageid.isin(popular_pageids)
clean_histories = clean_histories.loc[~is_popular_page_edit].copy()
all_histories = None
after_count = len(clean_histories)
print("%d edits (%.1f%%) were in top %d popular pages. Length after removing: %d" % (np.sum(is_popular_page_edit), 
                                                                                     100* np.sum(is_popular_page_edit)/before_count,
                                                                                     TOPN,
                                                                                     after_count)
     )
print("Number of unique page ids: {}".format(len(clean_histories.pageid.unique())))
```

```python colab={"base_uri": "https://localhost:8080/"} id="AXb35C3Dpay0" executionInfo={"status": "ok", "timestamp": 1625723250983, "user_tz": -330, "elapsed": 596, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bd9cb5e3-921c-4f54-e328-9afef49925ec"
# Remove users with too many or too few edits

MIN_EDITS = 5
MAX_EDITS = 10000

# Get user edit counts
all_user_edit_counts = clean_histories.groupby(['userid','user']).userid.count()

# Remove users with too few edits
keep_user = all_user_edit_counts.values >= MIN_EDITS

# Remove users with too many edits
keep_user = keep_user & (all_user_edit_counts.values <= MAX_EDITS)

# Remove users with "bot" in the name
is_bot = ['bot' in username.lower() for username in all_user_edit_counts.index.get_level_values(1).values]
keep_user = keep_user & ~np.array(is_bot)
print("Keep %d users out of %d (%.1f%%)" % (np.sum(keep_user), len(all_user_edit_counts), 100*float(np.sum(keep_user))/len(all_user_edit_counts)))

# Remove those users
userids_to_keep = all_user_edit_counts.index.get_level_values(0).values[keep_user]

clean_histories = clean_histories.loc[clean_histories.userid.isin(userids_to_keep)]

clean_histories = clean_histories.reset_index(drop=True)

print("Length after removing users: {}".format(len(clean_histories)))
```

```python id="Z_n_c4vSqDHa"
# Save cleaned histories
feather.write_feather(clean_histories, 'clean_histories_{}.feather'.format(suffix))
```

<!-- #region id="yqPcfycvqULl" -->
### Build lookup tables
<!-- #endregion -->

```python id="aSgHWfthqV7x"
# Page id to title and back
lookup = clean_histories.drop_duplicates(subset=['pageid']).loc[:,['pageid','title']]
p2t = dict(zip(lookup.pageid, lookup.title))
t2p = dict(zip(lookup.title, lookup.pageid))

# User id to name and back
lookup = clean_histories.drop_duplicates(subset=['userid']).loc[:,['userid','user']]
u2n = dict(zip(lookup.userid, lookup.user))
n2u = dict(zip(lookup.user, lookup.userid))
```

```python id="A3zHCF_pqX3G"
# Page id and userid to index in cooccurence matrix and back
pageids = np.sort(clean_histories.pageid.unique())
userids = np.sort(clean_histories.userid.unique())
 
p2i = {pageid:i for i, pageid in enumerate(pageids)}
u2i = {userid:i for i, userid in enumerate(userids)}


i2p = {v: k for k, v in p2i.items()}
i2u = {v: k for k, v in u2i.items()}
```

```python id="qS9Cgw6-qZKH"
# User name and page title to index and back
n2i = {k:u2i[v] for k, v in n2u.items() if v in u2i}
t2i = {k:p2i[v] for k, v in t2p.items() if v in p2i}

i2n = {v: k for k, v in n2i.items()}
i2t = {v: k for k, v in t2i.items()}
```

```python id="CZnMnMEvqZ6-"
save_pickle((p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t), '../lookup_tables_.pickle'.format(suffix))
save_pickle((userids, pageids), 'users_and_pages_{}.pickle'.format(suffix))
```

<!-- #region id="8xQmakRtZv43" -->
### Build test and training set
<!-- #endregion -->

```python id="Oo6K87vNZv44"
# Make a test set from the most recent edit by each user
histories_test = clean_histories.groupby(['userid','user'],as_index=False).first()
```

```python id="iQeN_IySZv45"
# Subtract it from the rest to make the training set
histories_train = dataframe_set_subtract(clean_histories, histories_test)
histories_train.reset_index(drop=True, inplace=True)
```

```python id="Mrabb-iUZv45"
# Make a dev set from the second most recent edit by each user
histories_dev = histories_train.groupby(['userid','user'],as_index=False).first()
# Subtract it from the rest to make the final training set
histories_train = dataframe_set_subtract(histories_train, histories_dev)
histories_train.reset_index(drop=True, inplace=True)
```

```python id="XpWyUraoZv45" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625723458381, "user_tz": -330, "elapsed": 482, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="871174eb-6d8f-4e2b-a24e-467190697970"
print("Length of test set: {}".format(len(histories_test)))
print("Length of dev set: {}".format(len(histories_dev)))
print("Length of training after removal of test: {}".format(len(histories_train)))
```

```python id="y4jCr5B7Zv46" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625723470244, "user_tz": -330, "elapsed": 642, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="39654ad0-9999-4eb9-be0d-744fa53f0bea"
print("Number of pages in training set: {}".format(len(histories_train.pageid.unique())))
print("Number of users in training set: {}".format(len(histories_train.userid.unique())))
print("Number of pages with > 1 user editing: {}".format(np.sum(histories_train.drop_duplicates(subset=['title','user']).groupby('title').count().user > 1)))
```

```python id="bD-CODScZv47"
feather.write_feather(histories_train, 'histories_train_{}.feather'.format(suffix))
feather.write_feather(histories_dev, 'histories_dev_{}.feather'.format(suffix))
feather.write_feather(histories_test, 'histories_test_{}.feather'.format(suffix))
```

```python id="mUKqOJWyZv47" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625723511389, "user_tz": -330, "elapsed": 692, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fc74fdac-b0af-4841-b9d2-f4e5e0de413b"
resurface_userids, discovery_userids = get_resurface_discovery(histories_train, histories_dev)

print("%d out of %d userids are resurfaced (%.1f%%)" % (len(resurface_userids), len(userids), 100*float(len(resurface_userids))/len(userids)))
print("%d out of %d userids are discovered (%.1f%%)" % (len(discovery_userids), len(userids), 100*float(len(discovery_userids))/len(userids)))
```

```python id="9ETiZsxbZv48"
wr.save_pickle((resurface_userids, discovery_userids), '../resurface_discovery_users_2021-05-28.pickle')
```

```python id="GeKU8oljrMWc"

```

<!-- #region id="EEw51nxpgpVI" -->
> Tip: To craft your evaluation, first think about what the recommender system is for

There is no generic metric or evaluation procedure that will work for every use case. Every decision about the offline evaluation depends on what you are using the recommender system for. Are you trying to DJ the ultimate 3 hour block of music for one person? Are you trying to get people to notice and click on a product in an ad? Are you trying to guess what a searcher wanted when they typed “green bank”? Are you trying to set a subscriber up with a date who’s a good match? That will form the basis for answering questions like: What is the metric of success? How exactly do you separate out the test set? How far down the list can the relevant stuff be placed before it stops being useful? How important is resurfacing items they’ve already shown an interest in?

In my case, I envision WikiRecs as a sort of co-pilot for editing Wikipedia. A personalized list that you can glance at to give you ideas for the next article you might want to edit. I also want it to flexibly and continuously adjust to the user’s changing interests. Therefore I chose to focus my evaluation on how well the learn-to-rank algorithms predict the next page the user really edited. Formulating the problem in terms of short term prediction makes it an excellent candidate for sequential recommendation systems, in which the sequence and timing of the user’s actions is explicitly modeled. 

Rather than predicting the next item, I could have taken a whole week or month of subsequent activity and predicted what was edited in that period, which would be a multilabel problem (and allow for additional metrics such as Precision@K), but this would require more data, add complexity, and shift the emphasis to more long-term predictions. I would take another approach again if my goal was to capture deeper, more permanent preferences of a user, that are expected to persist over the lifetime of their relationship with the app or tool. A quick menu of articles to edit probably shouldn’t be more than about 20 items, so I chose metrics that rewarded putting the relevant items in the top 20. The first, basic metric to look at then is Recall@20 (also known as hit rate, in cases like this with only one relevant item): how often did this algorithm place the actual next article to edited in the top 20?
<!-- #endregion -->

<!-- #region id="3ZF6Gqkvg1k6" -->
> Tip: Be aware of model and position bias

When you are building a recommender system to replace an existing solution, there is a problem of position bias (AKA presentation bias): items recommended by the current system have typically been placed at the tops of lists, or otherwise more prominently in front of users. This gives those items an advantage in offline evaluation, since there will be a tendency to favor models that also place them at the top, regardless of how customers would react to them in practice. Therefore simple evaluation metrics will be conservative towards the existing system, and the improvement will have to be large to beat the “historical recommender” (the ranked list that was really shown to the customer).

There are a number of techniques for fighting this bias, in particular inverse propensity scoring, which tries to model the unfair advantage items get at the top and then downweight actions on them proportionally. Another approach, though costly, is to have a “holdout lane” in production where personalized recommendations were not served, and use that to at least provide evaluation data — it may not provide enough data to train on. The ultimate evaluation data might be truly random lists of items presented to some portion of users, but even if a company could afford the likely huge drop in performance, it probably wouldn’t want the trust busting (see below) experience that would give!

Luckily in my case I didn’t have to worry about position bias much. Editing a Wikipedia page is a pretty high-intent action, so there is little problem with people idly editing articles just because they’re in front of them. The edits have nothing to do with my recommender system (since it doesn’t exist yet), so I don’t have to worry about previous-model bias (a form of presentation bias). There could be popularity bias, with people rushing to edit the hot topic of the day, but as we’ll see below it’s not very strong.
<!-- #endregion -->

<!-- #region id="ddJdkzFlhPbl" -->
> Tip: Take care with picking the test set

For most recommendation algorithms, reliable evaluation requires a portion of the data to be set aside and used only for that. I have framed my recommendation problem as a forecasting problem, so I divided my data into the most recent action and everything that came before it. The baseline algorithms in this post (such as “most recently edited”), are all unsupervised, so they look at the historical data as the “training data” and the next action as the test data. However, when in the future I am evaluating algorithms that are supervised, I will also need to set aside a random sample of users, their history and most recent edit, from the same time range. Make sure your training set doesn’t overlap your test set, and make two test sets, one being a dev set that can be used for tuning your hyper parameters rather than being used for the final evaluation.
<!-- #endregion -->

<!-- #region id="JxYh7wmahdM1" -->
> Tip: Use a metric that considers position

Recall@20, how often the top 20 recs contain the next page that was edited, is a reasonable, blunt metric for comparing models for my task. However, learn-to-rank algorithms can be compared at a more fine-grained level by considering whether they put the relevant items closer to the top of the sort. This can be inspected visually by simply plotting Recall @ 1 through N, which can reveal positional differences.
<!-- #endregion -->

```python id="WZMbUFSghasR"

```

<!-- #region id="wAe0oc2pZv5B" -->
# Build matrix for implicit collaborative filtering
<!-- #endregion -->

```python id="DCmUeSj8Zv5B"
# Get the user/page edit counts
for_implicit = histories_train.groupby(["userid","pageid"]).count().first_timestamp.reset_index().rename(columns={'first_timestamp':'edits'})
for_implicit.loc[:,'edits'] = for_implicit.edits.astype(np.int32)
```

```python id="J3yuocG7Zv5C"
row = np.array([p2i[p] for p in for_implicit.pageid.values])
col = np.array([u2i[u] for u in for_implicit.userid.values])

implicit_matrix_coo = coo_matrix((for_implicit.edits.values, (row, col)))


implicit_matrix = csc_matrix(implicit_matrix_coo)
```

```python id="I_l5JE-SZv5C"
save_pickle(implicit_matrix,'implicit_matrix_{}.pickle'.format(suffix))
```

<!-- #region id="OD1ASl9AZv5D" -->
### Test the matrix and indices
<!-- #endregion -->

```python id="s9zbhe4QZv5D" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1625723930794, "user_tz": -330, "elapsed": 634, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8b932313-b3c1-4b03-a4a4-c4e9b33f6f59"
# Crude item to item recs by looking for items edited by the same editors (count how many editors overlap)

veditors = np.flatnonzero(implicit_matrix[t2i["Barnard's Star b"],:].toarray())

indices =  np.flatnonzero(np.sum(implicit_matrix[:,veditors] > 0,axis=1))

totals = np.asarray(np.sum(implicit_matrix[:,veditors] > 0 ,axis=1)[indices])

sorted_order = np.argsort(totals.squeeze())

[i2t.get(i, "")  + " " + str(total[0]) for i,total in zip(indices[sorted_order],totals[sorted_order])][::-1][:10]
```

<!-- #region id="U684J8ySZv5F" -->
# Implicit recommendation
<!-- #endregion -->

```python id="9tI4GG1pZv5G"
bm25_matrix = bm25_weight(implicit_matrix, K1=100, B=0.25)
```

```python id="bNX-7iI1Zv5H"
num_factors =200
regularization = 0.01
os.environ["OPENBLAS_NUM_THREADS"] = "1"
model = implicit.als.AlternatingLeastSquares(
    factors=num_factors, regularization=regularization
)
model.fit(bm25_matrix)
```

```python id="Ul1nK1RbZv5J"
save_pickle(model,'als%d_bm25_model.pickle' % num_factors)
```

```python id="sn7ZzSsMZv5K"
model = load_pickle('als200_bm25_model_{}.pickle'.format(suffix))
```

```python id="g0YEDpPpZv5K" outputId="464c585c-c44d-4c42-a768-1613d41cda20"
results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]
```

```python id="gsNdABSmZv5L"
u = n2u["Rama"]
recommendations = model.recommend(u2i[u], bm25_matrix.tocsc(), N=1000, filter_already_liked_items=False)
[ ("*" if implicit_matrix[ind,u2i[u]]>0 else "") +
'%s %.4f' % (i2t[ind], score) + ' %d' % (implicit_matrix[ind,:]>0).sum()
 for ind, score in recommendations]
```

<!-- #region id="TC05uUZxZv5b" -->
## Grid search results
<!-- #endregion -->

```python id="WM-vAHOTZv5d"
grid_search_results = wr.load_pickle("implicit_grid_search.pickle")
```

```python id="20o_QtG_Zv5d" outputId="6bbf8662-3494-45f9-9fb7-041f8bfd6e54"
pd.DataFrame(grid_search_results)
```


```python id="WR9iGoIpZv5e" outputId="d8d7330e-62c9-406a-bafb-44e8964aa8a2"
pd.DataFrame([[i['num_factors'], i['regularization']] + list(i['metrics'].values()) for i in grid_search_results],
            columns = ['num_factors','regularization'] + list(grid_search_results[0]['metrics'].keys()))
```


```python id="hLmdkjT8Zv5f"
grid_search_results_bm25 = wr.load_pickle("implicit_grid_search_bm25.pickle")
```

```python id="I-gezdXHZv5f" outputId="9e1d308b-7051-4151-9cd6-48bf5e9ccf95"
pd.DataFrame([[i['num_factors'], i['regularization']] + list(i['metrics'].values()) for i in grid_search_results_bm25],
            columns = ['num_factors','regularization'] + list(grid_search_results_bm25[0]['metrics'].keys()))
```


<!-- #region id="musW541fZv5f" -->
# B25 Recommendation
<!-- #endregion -->

```python id="EmepuaMfZv5g" outputId="7aedd003-5def-4766-f5e6-e6ea2f374d51"
bm25_matrix = bm25_weight(implicit_matrix, K1=20, B=1)
bm25_matrix = bm25_matrix.tocsc()
sns.distplot(implicit_matrix[implicit_matrix.nonzero()],bins = np.arange(0,100,1),kde=False)

sns.distplot(bm25_matrix[bm25_matrix.nonzero()],bins = np.arange(0,100,1),kde=False)
```

```python id="x-1lMZ3KZv5g" colab={"referenced_widgets": ["789272e0be0841eab56c5e97832d6835"]} outputId="a97ff83b-97ae-4bae-f8b0-f20c1e5b5988"
K1 = 100
B = 0.25
model = BM25Recommender(K1, B)
model.fit(implicit_matrix)
```

```python id="C0ygHk0eZv5h"
wr.save_pickle(model, 'bm25_model_{}.pkl'.format(suffix))
```

```python id="DjeYF3yhZv5h" outputId="b4119c77-e7d5-49e1-a2a3-4cce00807eea"
results = model.similar_items(t2i['Mark Hamill'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]
```

```python id="aX6jGTMaZv5i"
a = ['Steven Universe 429.4746',
 'List of Steven Universe episodes 178.4544',
 'Demon Bear 128.7237',
 'Legion of Super Heroes (TV series) 128.7237',
 'The Amazing World of Gumball 126.3522',
 'Steven Universe Future 123.9198']
```

```python id="75dU7C-AZv5i" outputId="7af653cc-be3b-403c-f638-e1c3c608854b"
results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]
```

```python id="GFNZ_S0vZv5j" outputId="ecded46f-7aad-43c4-d025-120cb1728283"
results = model.similar_items(t2i['George Clooney'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]
```

```python id="QIxmxpUPZv5j" outputId="d62e3535-cd92-454b-f4b6-5e9319d642d6"
results = model.similar_items(t2i['Hamburger'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]
```

```python id="_x1bLobUZv5k"
u = n2u["Rama"]
recommendations = model.recommend(u2i[u], implicit_matrix.astype(np.float32), N=1000, filter_already_liked_items=True)
```

```python id="aWFnNlPuZv5k"
[ ("*" if implicit_matrix[ind,u2i[u]]>0 else "") +
'%s %.4f' % (i2t[ind], score) 
 for ind, score in recommendations]
```

```python id="CwaWOKSdZv5l" outputId="10aa5877-65e0-4e0c-81b4-7ca5c066ffd6"
plt.plot([ score for i,(ind, score) in enumerate(recommendations) if implicit_matrix[ind,u2i[u]]==0])
```

```python id="3fAd0VgRZv5l"
save_pickle(model, "b25_model.pickle")
```

```python id="CvvLYf0kZv5m"
model = load_pickle("b25_model.pickle")
```

<!-- #region id="IPFKPAShZv5m" -->
# Evaluate models
<!-- #endregion -->

<!-- #region id="kF6pDvOUZv5m" -->
## Item to item recommendation
<!-- #endregion -->

```python id="5eacRgpeZv5n" outputId="34b61a0b-faa4-425b-e554-67af17e4853a"
results = model.similar_items(t2i['Steven Universe'],20)
['%s %.4f' % (i2t[ind], score) for ind, score in results]
```

<!-- #region id="2iByFlteZv5n" -->
## User to item recommendations
<!-- #endregion -->

```python id="VNmdsfbGZv5n" outputId="e1e8c035-5d25-488d-88dc-f1e704ca7d5a"
# Check out a specific example

u = n2u["HyprMarc"]

wr.print_user_history(clean_histories, userid=u)
```

```python id="4ZXRhHnnZv5o"
u = n2u["HyprMarc"]
recommendations = model.recommend(u2i[u], implicit_matrix, N=100, filter_already_liked_items=False)
```

```python id="EFXuR_z9Zv5o"
[ ("*" if implicit_matrix[ind,u2i[u]]>0 else "") +
'%s %.4f' % (i2t[ind], score) 
 for ind, score in recommendations]
```

<!-- #region id="tYKQrUe0Zv5o" -->
# Visualize implicit embeddings
<!-- #endregion -->

```python id="3T9r6Y3oZv5p"
model = wr.load_pickle('als150_model.pickle')
```

```python id="oH90v2cKZv5p"
# Only plot the ones with over 3 entries
indices = np.squeeze(np.asarray(np.sum(implicit_matrix[nonzero,:],axis=1))) > 3

indices = nonzero[indices]
```

```python id="90iZGBmyZv5p" outputId="060392d6-dd67-4aea-9655-f73aabbc7af9"
len(indices)
```

```python id="aapuu6GUZv5p"
# Visualize  the collaborative filtering item vectors, embedding into 2D space with UMAP
# nonzero = np.flatnonzero(implicit_matrix.sum(axis=1))
# indices = nonzero[::100]
embedding = umap.UMAP().fit_transform(model.item_factors[indices,:])
```

```python id="RrnCOFTxZv5q" outputId="904a6ca3-abd8-414a-a94a-617acd3ec6b4"
plt.figure(figsize=(10,10))
plt.plot(embedding[:,0], embedding[:,1],'.')
# _ = plt.axis('square')
```

<!-- #region id="320AlPKbZv5q" -->
## Visualize actors in the embeddings space
<!-- #endregion -->

```python id="JVp2BGg-Zv5q"
edit_counts = np.squeeze(np.asarray(np.sum(implicit_matrix[indices,:],axis=1)))
log_edit_counts = np.log10(np.squeeze(np.asarray(np.sum(implicit_matrix[indices,:],axis=1))))

emb_df = pd.DataFrame({'dim1':embedding[:,0].squeeze(), 
                       'dim2':embedding[:,1].squeeze(),
                       'title':[i2t[i] for i in indices],
                       'edit_count':edit_counts,
                       'log_edit_count':log_edit_counts
                       })
```

```python id="AwnqVN9kZv5r" outputId="10010e96-0574-4d7c-8a08-c7511dc34c4c"
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

```python id="yY2JR57QZv5s"
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

<!-- #region id="PdcMikQmZv5s" -->
# Evaluate on test set
<!-- #endregion -->

```python id="VBc-GOq5Zv5s"
# Load the edit histories in the training set and the test set
histories_train = feather.read_feather('histories_train_{}.feather'.format(suffix))
histories_test = feather.read_feather('histories_test_{}.feather'.format(suffix))
histories_dev = feather.read_feather('histories_dev_{}.feather'.format(suffix))

implicit_matrix = load_pickle('implicit_matrix_{}.pickle'.format(suffix))
p2t, t2p, u2n, n2u, p2i, u2i, i2p, i2u, n2i, t2i, i2n, i2t = load_pickle('lookup_tables_{}.pickle'.format(suffix))

userids, pageids = load_pickle('users_and_pages_{}.pickle'.format(suffix))

resurface_userids, discovery_userids   = load_pickle('resurface_discovery_users_{}.pickle'.format(suffix))

results = {}
```

```python id="vMiO5TxQZv5t" outputId="7598b6ad-4154-42d0-e061-0357c847d5ff"
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

<!-- #region id="mXfsxgD7Zv5t" -->
## Most popular
<!-- #endregion -->

```python id="YJo0T3c-Zv5u" colab={"referenced_widgets": ["d2c71fccf9f24bf9ba26e3c1b50b531a"]} outputId="37dfe533-b5fe-462f-85eb-d2690737db2d"
%%time
K=20
rec_name = "Popularity"

prec = recommenders.PopularityRecommender(histories_train)
precs = prec.recommend_all(userids, K)
wr.save_pickle(precs, rec_name +"_recs.pickle")
```


```python id="kj2YOwQVZv5w" outputId="821e696b-9fb7-410a-aad2-ef1b04aa532a"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, precs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)

results[rec_name]
```


<!-- #region id="dgQXJF5hZv5x" -->
## Most recent
<!-- #endregion -->

```python id="opR4X1UtZv5y" colab={"referenced_widgets": ["d716ae7ee1874ee3a08b46a3a461e333"]} outputId="f4d90b9a-b11b-40ef-da15-f83168818abd"
%%time
# Most recent
K=20
rrec = recommenders.MostRecentRecommender(histories_train)
rrecs = rrec.recommend_all(userids, K, interactions=histories_train)
rec_name = "Recent"
wr.save_pickle(rrecs, rec_name +"_recs.pickle")
```

```python id="GadezT32Zv5z" outputId="7b210152-95f8-420b-afc3-f48771a3963d"
len(resurface_userids)
```

```python id="ABlm-8cyZv5z"
results ={}
```

```python id="9f_-UFFzZv5z" outputId="2da69d05-969f-41bd-fd45-900e24abe46d"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, rrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

<!-- #region id="aHDqzBI6Zv50" -->
## Most frequent
<!-- #endregion -->

```python id="m7Dcbr53Zv50" colab={"referenced_widgets": ["287b777493b948f2b0ce09f1ad79ea4d"]} outputId="5ceb09b6-22e1-4295-a15f-dd4f67bfe248"
%%time
# Sorted by frequency of edits
K=20
frec = recommenders.MostFrequentRecommender(histories_train)
frecs = frec.recommend_all(userids, K, interactions=histories_train)
rec_name = "Frequent"
wr.save_pickle(frecs, rec_name +"_recs.pickle")
```


```python id="18GPuEGOZv51" outputId="b8438727-bb20-4b99-8361-8ed961d747d1"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, frecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

<!-- #region id="9noCQYYlZv51" -->
## BM25
<!-- #endregion -->

```python id="olKWvnosZv51"
%%time
K=20
brec = recommenders.MyBM25Recommender(model, implicit_matrix)
```

```python id="AQIDvTLTZv52" colab={"referenced_widgets": ["d3955e88957c46e18fe203a3227fb31d"]} outputId="0b7c85b7-6be1-4132-ae27-c2a9f3361147"
brecs = brec.recommend_all(userids, K, u2i=u2i, n2i=n2i, i2p=i2p, filter_already_liked_items=False)
rec_name = "bm25"
wr.save_pickle(brecs, rec_name +"_recs.pickle")
```

```python id="cgWyxyHSZv52" outputId="ca819ec9-63c0-477a-864e-f6ac8d58b74e"
# filter_already_liked_items = False
results[rec_name] = wr.get_recs_metrics(
    histories_dev, brecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

```python id="gWu4QQ01Zv53"
# filter_already_liked_items = True
rec_name = "bm25_filtered"
brecs_filtered = brec.recommend_all(userids, K, u2i=u2i, n2i=n2i, i2p=i2p, filter_already_liked_items=True)
wr.save_pickle(brecs_filtered, rec_name +"_recs.pickle")
```


```python id="LeY8mCSsZv53" outputId="8bc773f9-f1b2-4f9a-a311-5408406349a5"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, recs['bm25_filtered'], K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

```python id="zqUcRq9RZv53" outputId="1f7b582e-5209-49c5-d244-26b95be34818"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, recs['bm25_filtered'], K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

<!-- #region id="KMFxi9jSZv54" -->
## ALS Implicit collaborative filtering
<!-- #endregion -->

```python id="A6uvTNzTZv54"
model_als = wr.load_pickle('als200_bm25_model_{}.pickle'.format(suffix))
```

```python id="PE48qQ28Zv54" colab={"referenced_widgets": ["14ff8e978a09477e91aca95e1f3b28f2"]} outputId="3ffdf3f7-90ae-4fb3-bab8-f916a15a1ca7"
%%time
rec_name = "als"
K=20
irec = recommenders.ImplicitCollaborativeRecommender(model_als, bm25_matrix.tocsc())
irecs = irec.recommend_all(userids, K, i2p=i2p, filter_already_liked_items=False)
wr.save_pickle(irecs, rec_name +"_recs.pickle")
```

```python id="CYHnFRHuZv55" colab={"referenced_widgets": ["79ee3142902143588e76b1a03e7b2d86"]} outputId="fa9a39e7-7820-440c-dbc1-54c37394f43b"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, irecs, K, discovery_userids, resurface_userids, bm25_matrix.tocsc(), i2p, u2i)
results[rec_name]
```

```python id="gM2aNlFsZv55"
rec_name = "als_filtered"
K=20
irec = recommenders.ImplicitCollaborativeRecommender(model_als, bm25_matrix.tocsc())
irecs_filtered = irec.recommend_all(userids, K, i2p=i2p, filter_already_liked_items=True)
results[rec_name] = wr.get_recs_metrics(
    histories_dev, irecs_filtered, K, discovery_userids, resurface_userids, bm25_matrix.tocsc(), i2p, u2i)
results[rec_name]
```

```python id="7YBK3MvAZv55"
wr.save_pickle(irecs_filtered, rec_name +"_recs.pickle")
```

```python id="CUKrdhJCZv56" outputId="09e172d0-e0ed-491b-bfe3-e52cc4d91439"
show(pd.DataFrame(results).T)
```

<!-- #region id="KqH3xh2qZv56" -->
## Jaccard
<!-- #endregion -->

```python id="zFPUNXdmZv56" colab={"referenced_widgets": ["8612548500614215b3221bdcf6ee204f", "dbca2b89d81647ff9c4a910a574d0d40"]} outputId="3f0f00d4-fab7-434e-e2c1-2545ab1846b5"
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

```python id="c1oOo2zfZv57"
wr.save_pickle(jrecs,"jaccard-1_recs.pickle")
```

```python id="va3JMRmAZv57" outputId="3d8b5cf6-55ae-4ba7-99a4-c9679640cd63"
rec_name = "Jaccard"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, jrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

```python id="vRtY7AqDZv58" outputId="19cbc652-e859-448a-c4c8-b5304cd28fa1"
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

```python id="AE8IMQ5OZv58" colab={"referenced_widgets": ["e8b374d98a2f401c9f48ff75e50b67e1"]} outputId="f681e22b-d6d0-4191-a523-6a40743e56cd"
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

```python id="K5Og2N-dZv59" outputId="99ad5919-f37c-4aa2-aac8-ae1f987f9585"
print("Recall @ %d: %.1f%%" % (K, 100*wr.recall(histories_test, jrecs, K)))
print("Prop resurfaced: %.1f%%" % (100*wr.prop_resurface(jrecs, K, implicit_matrix, i2p, u2i)))
print("Recall @ %d (discovery): %.1f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=discovery_userids)))
print("Recall @ %d (resurface): %.1f%%" % (K, 100*wr.recall(histories_test, jrecs, K, userid_subset=resurface_userids)))
```

<!-- #region id="yUxtXpveZv59" -->
## Interleaved
<!-- #endregion -->

```python id="kuUQJJXnZv5-" outputId="4d41ffa8-eff7-4bb1-c400-9737379a7666"
recs.keys()
```

```python id="DAUh-2kQZv5-" outputId="e98e27ac-ef8c-43fa-8879-acfa91ba0261"
# Interleaved jaccard and recent
K=20
rec_name = "Interleaved"
print(rec_name)
intrec = recommenders.InterleaveRecommender()
intrecs = intrec.recommend_all(K, [recs['Recent'], recs['bm25_filtered']])

wr.save_pickle(intrecs, rec_name +"_recs.pickle")
```

```python id="9cP_yk8TZv5_" outputId="0952801f-1de2-4f4a-e0f8-fa7f7d29a594"
results[rec_name] = wr.get_recs_metrics(
    histories_dev, intrecs, K, discovery_userids, resurface_userids, implicit_matrix, i2p, u2i)
results[rec_name]
```

<!-- #region id="yHDysar5Zv5_" -->
# Report on evaluations results
<!-- #endregion -->

<!-- #region id="6UEPDmdSZv5_" -->
## Hard coded metrics
<!-- #endregion -->

```python id="51N_eEsrZv6A"
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


<!-- #region id="mtGOtvmJZv6A" -->
## Table of results
<!-- #endregion -->

```python id="Wglg-nL7Zv6A" outputId="1ba8ce57-cea7-42a1-b7c3-340f5c572aa5"
results_df
```

<!-- #region id="j5V4bPHAZv6B" -->
### FIG Table for post
<!-- #endregion -->

```python id="NMdYHAjoZv6B"
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

```python id="YJmCRp1tZv6C" outputId="a96e4926-38dc-45fe-aae4-1c6678d19443"
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

<!-- #region id="caMEnw0cZv6C" -->
## Scatter plots (resurface vs discover)
<!-- #endregion -->

```python id="3maavYsNZv6C" outputId="f542c2ad-8e46-4fb6-b26e-8bceab311017"
fig = px.scatter(data_frame=results_df,
                 x='ndcg_discover',
                 y='ndcg_resurface',
                hover_name='index')
#                  hover_name='title',)
fig.show()
```

```python id="GT6PoNYqZv6D" outputId="2c856e98-6185-45d6-9de3-1689f422cc03"
fig = px.scatter(data_frame=results_df,
                 x='recall_discover',
                 y='recall_resurface',
                hover_name='index')
#                  hover_name='title',)
fig.show()
```

<!-- #region id="VTMIJ2xyZv6D" -->
### FIG Scatterplot for post
<!-- #endregion -->

```python id="xl5MrshfZv6E"
x = 2*[results_df.loc[results_df.Recommender == "Interleaved","Recall@20 resurface"].values[0]]
y = [0, results_df.loc[results_df.Recommender == "Interleaved","Recall@20 discovery"].values[0]]
```

```python id="Foozjym6Zv6E" outputId="fe0d46a2-b2fb-40ae-e8cd-3c6c86747171"
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

<!-- #region id="5NsAzhwPZv6E" -->
## Read recs in from files
<!-- #endregion -->

```python id="M12xU-o8Zv6F"
recommender_names = ['Popularity', 'Recent', 'Frequent', 'ALS', 'ALS_filtered', 'BM25', 'BM25_filtered', 'Interleaved']
```

```python id="fIua-8DLZv6F"
recs = {rname:wr.load_pickle(rname + "_recs.pickle") for rname in recommender_names}
```

<!-- #region id="DrzO07hWZv6F" -->
## Recall curves
<!-- #endregion -->

```python id="CQxVu3jkZv6G"
histories_dev = feather.read_feather('histories_dev_{}.feather'.format(suffix))
```

```python id="DenyFpLmZv6G" outputId="954aa2ef-098a-4830-ab22-eaa2ec806a83"
plt.figure(figsize=(15,10))
for rname in recommender_names:
    recall_curve = wr.recall_curve(histories_dev, recs[rname], 20)
#     print(recall_curve[-1])
    plt.plot(recall_curve,'.-')
plt.legend(recommender_names)
```

```python id="BUZ6DFIjZv6G" outputId="263ff0b1-420a-4ac2-9788-37037e41c36f"
plt.figure(figsize=(15,10))
for rname in recommender_names:
    recall_curve = wr.recall_curve(histories_dev, recs[rname], 20, discovery_userids)
    plt.plot(recall_curve,'.-')
plt.legend(recommender_names)
```

```python id="gWgERGRcZv6H" outputId="0e56fd00-e71f-48cc-ec34-aef6fbe17e43"
plt.figure(figsize=(15,10))
for rname in recommender_names:
    recall_curve = wr.recall_curve(histories_dev, recs[rname], 20, resurface_userids)
    plt.plot(recall_curve,'.-')
plt.legend(recommender_names)
```

<!-- #region id="KTeNsy-DZv6H" -->
### FIG Implicit vs BM25 figure
<!-- #endregion -->

```python id="SSqWc8yLZv6I" outputId="91fa7883-5fd0-4114-e8f5-9a58a6c41135"
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


<!-- #region id="l7p9yhvfZv6I" -->
# User recommendation comparison
<!-- #endregion -->

```python id="RD326W69Zv6J"
recs_subset = ["Recent","Frequent","Popularity","Implicit","bm25","interleaved"]
```

```python id="-n3AKAYiZv6J" outputId="91fafc9b-0685-4a91-f557-d6a03816dec4"
print("Next edit: " + histories_dev.loc[histories_dev.userid == userid].title.values[0])
```


<!-- #region id="lmki9MGOZv6K" -->
## FIG Rama table
<!-- #endregion -->

```python id="D_4W0wMyZv6K"
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

```python id="ASe4-b1XZv6L" outputId="cfd2cbe3-d285-4a9b-b5cd-3d47b11e70b1"
recs_subset = ["Recent","Frequent","Popularity","ALS","ALS_filtered","BM25","BM25_filtered"]

display_user_recs_comparison('Rama', recs, recs_subset, histories_train, histories_dev, N=10)
```

<!-- #region id="hZX9sboCZv6L" -->
## Other individuals tables
<!-- #endregion -->

```python id="-3M3Ef_EZv6L" outputId="f4755f42-12c9-4405-dfe8-9819e101dccd"
display_user_recs_comparison('Meow', recs, recs_subset, histories_train, histories_dev, N=10)
```

```python id="zmgN2nibZv6O" outputId="2010834a-8ba7-4583-ebf6-f346ba90efc4"
display_user_recs_comparison('KingArti', recs, recs_subset, histories_train, histories_dev, N=10)
```

```python id="zt61DkbSZv6P" outputId="e98e6cb1-3c81-44b1-da1c-cd1faab14f3b"
display_user_recs_comparison('Tulietto', recs, recs_subset, histories_train, histories_dev, N=10)
```

```python id="5-54hsajZv6P" outputId="a2898084-1477-45a3-acc4-d66a890b78fe"
display_user_recs_comparison('Thornstrom', recs, recs_subset, histories_train, histories_dev, N=10)
```

<!-- #region id="fKdhlfpjZv6Q" -->
## FIG Interleaved
<!-- #endregion -->

```python id="LYXrhOeeZv6Q" outputId="e76717f2-2747-46b2-d4e8-7ad89ae0ee17"
display_user_recs_comparison('Rama', recs,['Interleaved'], histories_train, histories_dev, N=10)
```

```python id="Y7LlQqbeZv6Q" outputId="5c39eb4d-34f8-4826-bdb9-592ec016fe4c"
display_user_recs_comparison('KingArti', recs,['Interleaved'], histories_train, histories_dev, N=10)
```

```python id="KPRJsDyyZv6R" outputId="36cf174f-87d9-4a59-8270-50d6f019ab82"
N = 20
display(pd.DataFrame({rec_name: [p2t[r] for r in recs[rec_name][n2u['HenryXVII']]][:N] for rec_name in recs_subset}))
```

```python id="f7EKl33NZv6R"
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

```python id="ixNX4VxXZv6R" colab={"referenced_widgets": ["df21373672484fc8a075c5acf4cf3e3b"]} outputId="3b3a8021-8e91-4a4f-c1f5-a444c3ee7c77"
irec_500 = recommenders.ImplicitCollaborativeRecommender(model, implicit_matrix)
irecs_poi = irec_500.recommend_all([n2u[user_name] for user_name in persons_of_interest], N, u2i=u2i, n2i=n2i, i2p=i2p)
```


<!-- #region id="r6CHnygFZv6S" -->
# Find interesting users
<!-- #endregion -->

```python id="Q7U9h5s6Zv6S"
edited_pages = clean_histories.drop_duplicates(subset=['title','user']).groupby('user').userid.count()

edited_pages = edited_pages[edited_pages > 50]
edited_pages = edited_pages[edited_pages < 300]
```


```python id="Zch7RoF2Zv6S" outputId="ea53f55e-5ca7-4806-d70a-b5fff3a9e11b"
clean_histories.columns
```

```python id="3J8bIjDFZv6S" outputId="e2d39922-b9c5-4632-fc78-a54dc15bd9b1"
display_user_recs_comparison("Rama", recs, recs_subset, histories_train, histories_dev, N=20)
```


```python id="affapbuVZv6T" outputId="4bf4dab0-e868-4de0-cf08-31ae00e6a5a8"
index = list(range(len(edited_pages)))
np.random.shuffle(index)

for i in index[:10]:
    user_name = edited_pages.index[i]
    print(user_name)
    display_user_recs_comparison(user_name, recs, recs_subset, histories_train, histories_dev, N=20)
    print("\n\n\n")
```

```python id="A4-QHy1uZv6U"
index = list(range(len(edited_pages)))
np.random.shuffle(index)

for i in index[:10]:
    print(edited_pages.index[i])
    display_user_recs_comparison
    wr.print_user_history(user=edited_pages.index[i],all_histories=clean_histories)
    print("\n\n\n")
```

```python id="b6xHBQ7yZv6V"
sns.distplot(edited_pages,kde=False,bins=np.arange(0,2000,20))
```

<!-- #region id="O4-tar8HZv6V" -->
# Repetition analysis
<!-- #endregion -->

```python id="8uvwvTHgZv6V"
import itertools
```

```python id="1P210EgeZv6V" outputId="b90538b8-61bf-4208-ea1a-ea00492ffe68"
clean_histories.head()
```

```python id="lPkQ-3FnZv6W" outputId="88a8621f-7b04-4f08-cb0a-e02f6b55363e"
clean_histories.iloc[:1000].values.tolist()
```

```python id="nik1EY2NZv6X" outputId="440936fa-f7e0-4de6-d013-e5fc882eb93e"
df = clean_histories
dict(zip(df.columns, range(len(df.columns))))
```

```python id="luoScXTDZv6X"
def identify_runs(df):
    d  = df.loc[:,['userid','pageid']].values.tolist()
    return [(k, len(list(g))) for k,g in itertools.groupby(d)]
```

```python id="o0mM-N6RZv6X" outputId="295d8135-d166-4395-cb50-dfc9b0458dc4"
%%time
runs = identify_runs(clean_histories)
```

```python id="cVtNW6R5Zv6Y" outputId="742a3494-69e5-4e8e-ad32-6f7af96ea677"
lens = np.array([r[1] for r in runs])

single_edits = np.sum(lens==1)
total_edits = len(clean_histories)

print("Percent of edits that are part of a run: %.1f%%" % (100*(1-(float(single_edits)/total_edits))))

print("Percent of edits that are repetitions: %.1f%%" % (100*(1-len(runs)/total_edits)))
```
