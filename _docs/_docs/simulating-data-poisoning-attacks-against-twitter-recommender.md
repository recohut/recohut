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
    language: python
    name: python3
---

<!-- #region id="B6CEBMzyx8k0" -->
# Simulating Data Poisoning Attacks against Twitter Recommender

This tutorial details experiments designed to simulate attacks against social network recommendation mechanisms, and evaluate their effectiveness. All experiments documented here follow the procedure outlined below:

1. Load a dataset of anonymized retweet interactions collected from actual Twitter data.
2. Train a collaborative filtering model on the loaded data.
3. Select a target account to be "amplified" such that it is recommended to a set of users who have interacted with a separate, high-profile user also in the dataset. We select 20 such users as a "control" set.
4. Implement recommendation logic based on cosine similarity of the vector representations of the trained model, and observe recommendations for the control set.
5. Select a set of "amplifier accounts" that have not interacted with either the target account or the high-profile account, and are not members of the control set.
6. For a number of different proposed sets of amplifier accounts and parameter choices, create a new dataset containing additional interactions between each selected amplifier account and both the target account and the high-profile account. In practise, this process involves appending two new rows per amplifier account - one adding a retweet count for the target account and another adding retweet count for the high-profile account.
7. Train a new model on the modified dataset.
8. Run both target-based and source-based recommendations for each member of the control group and record the number of times the target appeared in the top-n (3) recommendations.
9. Present and discuss the results.
<!-- #endregion -->

```python id="wNTGLPWfSpsv"
!pip install -U fastai
```

```python id="_VWhS9oRMyvh"
# Note this is broken for pytorch 1.7.x, so please use pytorch 1.6
# pip install torch==1.6.0 torchvision==0.7.0

from sklearn.metrics.pairwise import cosine_similarity

from fastai.tabular.all import *
from fastai.collab import *

import networkx as nx
import community
import community.community_louvain as community_louvain

import pandas as pd
import numpy as np
import json
import os

import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

random.seed(1)

# Helper functions
def save_ratings(df, fn):
    with open(fn, "w") as f:
        f.write("Source,Target,Weight\n")
        for item in zip(df["Source"], df["Target"], df["Weight"]):
            s, t, w = item
            f.write(str(s)+","+str(t)+","+str(w)+"\n")

def key_with_max_value(d):  
     v = list(d.values())
     k = list(d.keys())
     return k[v.index(max(v))], max(v)

def key_with_min_value(d):  
     v = list(d.values())
     k = list(d.keys())
     return k[v.index(min(v))], min(v)

def get_most_similar(uid, matrix, max_matches):
    if uid >= len(matrix):
        return None
    all_matches = matrix[uid]
    top_matches = np.flip(np.argsort(all_matches))
    match_rating = [[top_matches[i], all_matches[top_matches[i]]] for i in range(max_matches)]
    return match_rating

def print_similar_to_targets(samples, t_matrix):
    for n in samples:
        print("Target: " + tid_name[n] + " similar to:")
        matches = get_most_similar(n, t_matrix, t_max_matches)
        if matches == None:
            return
        for item in matches[:10]:
            tid, rating = item
            print(tid_name[tid] + " " + "%.4f"%rating)
        print()

def print_similar_to_sources(samples, s_matrix):
    for n in samples:
        print("User: " + sid_name[n] + " similar to:")
        matches = get_most_similar(n, s_matrix, s_max_matches)
        if matches == None:
            return
        for item in matches[:10]:
            sid, rating = item
            print(sid_name[sid] + " " + "%.4f"%rating)
        print()
        
def print_target_similarity(t1, t2, t_matrix):
    sim = t_matrix[t1][t2]
    print(tid_name[t1] + " similarity to " + tid_name[t2] + ": " + "%.4f"%sim)

def make_nx_graph(inter):
    mapping = []
    names = set()
    for source, targets in inter.items():
        names.add(source)
        for target, count in targets.items():
            mapping.append((source, target, count))
            names.add(target)
    g=nx.Graph()
    g.add_weighted_edges_from(mapping)
    return g, names

def get_median(var, source_list):
    vals = []
    for s in source_list:
        if s in var:
            vals.append(var[s])
    return np.median(vals)

def get_mean(var, source_list):
    vals = []
    for s in source_list:
        if s in var:
            vals.append(var[s])
    return np.mean(vals)

def get_jaccard_median(inter, source_list, target):
    g, names = make_nx_graph(inter)
    pairs = []
    for s in source_list:
        if s in names:
            pairs.append((s, target))
    preds = nx.jaccard_coefficient(g, pairs)
    vals = []
    for s, t, p in preds:
        vals.append(p)
    return np.median(vals)

def get_jaccard_mean(inter, source_list, target):
    g, names = make_nx_graph(inter)
    pairs = []
    for s in source_list:
        if s in names:
            pairs.append((s, target))
    preds = nx.jaccard_coefficient(g, pairs)
    vals = []
    for s, t, p in preds:
        vals.append(p)
    return np.mean(vals)

def get_communities(inter):
    g, names = make_nx_graph(inter)
    communities = community_louvain.best_partition(g)

    clusters = {}
    for node, mod in communities.items():
        if mod not in clusters:
            clusters[mod] = []
        clusters[mod].append(node)
    return clusters

def get_mean_distance(inter, target, source_list):
    g, names = make_nx_graph(inter)

    distance_vals = []
    for source in source_list:
        if source in names:
            length = nx.shortest_path_length(g, source=target, target=source)
            distance_vals.append(length)
    return np.mean(distance_vals)

# Return a new poisoned dataframe
def get_poisoned_dataset(ratings, amplifier_candidates, num_amplifiers, rating_val, save_path):
    ratings2 = pd.DataFrame(ratings)
    # For base set measurements
    if num_amplifiers < 1 or rating_val < 1:
        return ratings2
    new_data = []
    amplifiers = random.sample(amplifier_candidates, num_amplifiers)

    for uid in amplifiers:
        new_data.append([uid, target_tid, rating_val])
        new_data.append([uid, high_profile_tid, rating_val])
    new_ratings_df = pd.DataFrame(new_data, columns=['Source', 'Target', 'Weight'])
    ratings2 = ratings2.append(new_ratings_df, ignore_index=True)
    
    # Save poisoned dataset for further inspection or visualization in gephi
    interactions2 = {}
    for item in zip(ratings2['Source'], ratings2['Target'], ratings2['Weight']):
        s, t, r = item
        sid_label = sid_name[s]
        tid_label = tid_name[t]
        if sid_label not in interactions2:
            interactions2[sid_label] = Counter()
        interactions2[sid_label][tid_label] += r
    with open(save_path, "w") as f:
        f.write("Source,Target,Weight\n")
        for s, tw in interactions2.items():
            for t, w in tw.items():
                f.write(str(s)+","+str(t)+","+str(w)+"\n")
    return ratings2
```

<!-- #region id="Rz8dnm_JMyvz" -->
## Load and process dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="j5LCHxFPTWSo" executionInfo={"status": "ok", "timestamp": 1633267058793, "user_tz": -330, "elapsed": 814, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6480bb97-07ed-4087-de69-a5adcd9e3f91"
!wget -q --show-progress https://github.com/r0zetta/collaborative_filtering/raw/main/US2020/anonymized_interactions.csv
```

```python id="JbbLs3r1Myv5" colab={"base_uri": "https://localhost:8080/", "height": 238} executionInfo={"status": "ok", "timestamp": 1633267063334, "user_tz": -330, "elapsed": 514, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="866c3a00-ba3c-4624-e1f2-7d3f9ecb2306"
# Load and prepare raw data
raw = pd.read_csv("anonymized_interactions.csv")

# Source ids (accounts that retweeted)
sid_name = {}
name_sid = {}
sid = 0
for name in raw['Source']:
    if name not in name_sid:
        name_sid[name] = sid
        sid_name[sid] = name
        sid += 1

# Target ids (accounts that received retweets)
tid_name = {}
name_tid = {}
tid = 0
for name in raw['Target']:
    if name not in name_tid:
        name_tid[name] = tid
        tid_name[tid] = name
        tid += 1

print("Number of retweeters: " + str(len(name_sid)))
print("Number of retweeted: " + str(len(name_tid)))
# Assemble ratings dataframe used to train the model
ratings = pd.DataFrame()
ratings['Source'] = [name_sid[x] for x in raw['Source']]
ratings['Target'] = [name_tid[x] for x in raw['Target']]
ratings['Weight'] = raw['Weight']
ratings.head()
```

```python id="_cmL3XrnMyv7" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633267260109, "user_tz": -330, "elapsed": 6353, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7fd3682a-a58a-4ca2-efa4-788bf70953b3"
source_list = list(set(ratings['Source']))
target_list = list(set(ratings['Target']))
target_retweeted_by = {}
target_retweeted_count = {}
target_retweeters = Counter()
target_source_count = Counter()
source_retweeted = {}
source_retweets = Counter()
source_target_count = Counter()
interactions = {}
for item in zip(ratings['Source'], ratings['Target'], ratings['Weight']):
    s, t, r = item
    if sid_name[s] not in interactions:
        interactions[sid_name[s]] = Counter()
    interactions[sid_name[s]][tid_name[t]] += r
    source_retweets[s] += r
    if t not in target_retweeted_count:
        target_retweeted_count[t] = Counter()
    target_retweeted_count[t][s] = r
    if s not in source_retweeted:
        source_retweeted[s] = []
    if t not in source_retweeted[s]:
        source_retweeted[s].append(t)
        source_target_count[s] += 1
    if t not in target_retweeted_by:
        target_retweeted_by[t] = []
    if s not in target_retweeted_by[t]:
        target_retweeted_by[t].append(s)
        target_source_count[t] += 1
    target_retweeters[t] += 1
with open("labeled_ratings.csv", "w") as f:
    f.write("Source,Target,Weight\n")
    for s, tw in interactions.items():
        for t, w in tw.items():
            f.write(str(s)+","+str(t)+","+str(w)+"\n")
print("Number of sources: " + str(len(source_list)))
print("Number of targets: " + str(len(target_list)))
print("Total number of retweet interactions: " + str(sum(ratings['Weight'])))
print()
print("Targets with most retweets")
print("tid\tretweets")
for x, c in target_retweeters.most_common(10):
    print(tid_name[x] + "\t" + str(c))
print()
print("Targets with most unique sources retweeting them")
print("tid\tsources")
for x, c in target_source_count.most_common(10):
    print(tid_name[x] + "\t" + str(c))
print()
for x, c in target_retweeters.most_common(10):
    also_retweeted = Counter()
    for sid, tids in source_retweeted.items():
        if len(tids) > 1:
            if x in tids:
                for tid in tids:
                    if tid != x:
                        also_retweeted[tid] += 1
    msg = "Sources that retweeted " + tid_name[x]
    msg += " also retweeted " + str(len(also_retweeted)) + " other accounts."
    print(msg)
    for x, c in also_retweeted.most_common(10):
        print("Retweeted " + tid_name[x] + " " + str(c) + " times.")
    print("")

# People who retweeted x also retweeted y
print()
print("Sources that published the most retweets")
print("sid\tretweets")
for x, c in source_retweets.most_common(10):
    print(sid_name[x] + "\t" + str(c))
print()
print("Sources that retweeted the most unique targets")
print("sid\ttargets")
for x, c in source_target_count.most_common(10):
    print(sid_name[x] + "\t" + str(c))
    
communities = get_communities(interactions)
community_sids = {}
community_sizes = Counter()
for mod, names in communities.items():
    community_sizes[mod] = len(names)
print(len(communities))
print(community_sizes)
```

<!-- #region id="NwSzW2gPMyv-" -->
## Choose accounts for poisoning experiment
<!-- #endregion -->

```python id="GLBKbPk7Myv_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633267263795, "user_tz": -330, "elapsed": 516, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a30c92e3-5431-4d3a-a7a8-bff7354d7d50"
# Here a target to be boosted and a high-profile account that the target should be seen as similar are picked
# high_profile_tid was chosen from the original labeled dataset.
# It is a high-profile Twitter account that gets a lot of engagement
high_profile_tid = 191
high_profile_name = tid_name[high_profile_tid]
high_profile_sid = None
if high_profile_name in name_sid:
    high_profile_sid = name_sid[high_profile_name]
print("high_profile_tid: " + str(high_profile_tid) + " == " + tid_name[high_profile_tid])
print("high_profile_sid: " + str(high_profile_sid))
# The target account was selected based on a few criteria:
# - it is highly retweeted in the original dataset (top 10)
# - the original dataset contains plenty of accounts that haven't retweeted it and high_profile_tid
#   (thus enabling us to create a large number of amplifier candidates below)
target_tid = 4451
target_name = tid_name[target_tid]
target_sid = None
if target_name in name_sid:
    target_sid = name_sid[target_name]
print("target_tid: " + str(target_tid) + " == " + tid_name[target_tid])
print("target_sid: " + str(target_sid))
# Feel free to change these values for other experiments

# Pick a list of accounts that engaged with the high profile account in order to compare
# similarity values before and after poisoning
num_controls = 20
control_candidates = []
for sid, tids in source_retweeted.items():
    if len(tids) > 50:
        if high_profile_tid in tids:
            control_candidates.append(sid)
print("Candidates for control accounts: " + str(len(control_candidates)))
controls = random.sample(control_candidates, num_controls)
# For consistency's sake, here's a hard-coded list of control candidates 
# (selected by running the above code once)
# controls = [229, 6266, 340, 124, 25, 4000, 89, 4347, 1947, 20144, 14, 22, 107, 13426, 237, 708, 1560, 62, 9, 11]
print(controls)
```

```python id="sdNouUTkMywB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633267268746, "user_tz": -330, "elapsed": 656, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7947ebcf-13fd-42aa-9529-f5318c327075"
# Find accounts that engaged with both high_profile and target
retweeted_target = target_retweeted_by[target_tid]
print("Number of accounts that retweeted target:")
print(len(retweeted_target))
retweeted_high_profile = target_retweeted_by[high_profile_tid]
print("Number of accounts that retweeted high-profile:")
print(len(retweeted_high_profile))
retweeted_both = set(retweeted_target).intersection(set(retweeted_high_profile))
print("Number of accounts that retweeted both:")
print(len(retweeted_both))
```

```python id="P4Ofp_HhMywD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633267292937, "user_tz": -330, "elapsed": 21827, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e732aa18-72cb-42b0-f5d9-0d287dbbd3a2"
# Feature analysis of communities 

retweeted_target = target_retweeted_by[target_tid]
retweeted_target_count = target_retweeted_count[target_tid]
retweeted_high_profile = target_retweeted_by[high_profile_tid]
retweeted_high_profile_count = target_retweeted_count[high_profile_tid]

G, N = make_nx_graph(interactions)
pr = nx.pagerank(G)

num_amplifiers = 200
community_features = {}
for mod, names in communities.items():
    amplifier_candidates = []
    for name in names:
        if name in name_sid:
            sid = name_sid[name]
            if sid in source_retweeted:
                rtw = source_retweeted[sid]
                if high_profile_tid not in rtw and target_tid not in rtw:
                    amplifier_candidates.append(sid)
    if len(amplifier_candidates) < num_amplifiers:
        continue
    print("Mod: " + str(mod) + " size: " + str(len(names)))
    community_features[mod] = {}
    community_features[mod]["Community size"] = len(names)
    
    community_total_retweets = 0
    for n in names:
        if n in name_sid:
            community_total_retweets += source_retweets[name_sid[n]]
    community_features[mod]["Total retweets"] = community_total_retweets

    mpr = get_mean(pr, names)
    community_features[mod]['Mean pagerank * 10e5'] = mpr * 100000

    jaccard = get_jaccard_mean(interactions, names, target_name)
    mod_name = 'Mean Jaccard coefficient between accounts in community and target * 10e4'
    community_features[mod][mod_name] = jaccard * 10000

    sid_list = [name_sid[x] for x in names if x in name_sid]
    rtw_target_sids = set(retweeted_target).intersection(set(sid_list))
    community_features[mod]['Unique accounts in community that retweeted target'] = len(rtw_target_sids)
    rtw_target_count = 0
    for sid, count in retweeted_target_count.items():
        if sid in rtw_target_sids:
            rtw_target_count += count
    community_features[mod]['Total retweets of target'] = rtw_target_count
    rhps = set(retweeted_high_profile).intersection(set(sid_list))
    community_features[mod]['Unique accounts in community that retweeted high-profile'] = len(rhps)
    rhpc = 0
    for sid, count in retweeted_high_profile_count.items():
        if sid in rhps:
            rhpc += count
    community_features[mod]['Total retweets of high-profile'] = rhpc
    community_retweet_counts = [source_retweets[x] for x in sid_list]
    community_features[mod]['Mean retweets per account'] = np.mean(community_retweet_counts)
    community_features[mod]['Max retweet count'] = max(community_retweet_counts)
    
    controls_in_mod = set(sid_list).intersection(set(controls))
    community_features[mod]['Number of control accounts'] = len(controls_in_mod)
        
    rtw_controls_sids = 0
    rtw_controls_count = 0
    for sid in controls:
        if sid in target_retweeted_count:
            rcl = target_retweeted_count[sid]
            for s, c in rcl.items():
                if s in sid_list:
                    rtw_controls_sids += 1
                    rtw_controls_count += c
    community_features[mod]['Accounts in this community that retweeted control accounts'] = rtw_controls_sids
    community_features[mod]['Total control account retweets published by this community'] = rtw_controls_count
    
    target_mean_path_len = get_mean_distance(interactions, target_name, names)
    community_features[mod]['Mean path length between community nodes and target'] = target_mean_path_len

print(json.dumps(community_features, indent=4))
```

```python id="t1bVUMFXMywH" colab={"base_uri": "https://localhost:8080/", "height": 725} executionInfo={"status": "ok", "timestamp": 1633267292940, "user_tz": -330, "elapsed": 40, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="09aad8bb-d35e-402b-99b8-d04bac0cc70d"
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]
def highlight_min(s):
    is_min = s == s.min()
    return ['background-color: orange' if v else '' for v in is_min]
fe = pd.DataFrame(community_features)
fe.style.apply(highlight_max, axis=1).apply(highlight_min, axis=1).format("{:.2f}")
```

<!-- #region id="wZu1AVJ7MywL" -->
## Build model
<!-- #endregion -->

```python id="MO3i8sv6MywP"
# Given a ratings dataframe containing columns "Source", "Target", "Weight"
# train a collaborative filtering model and return the target and source weight embeds
def make_model_collab(ratings, epochs):
    min_rating = min(ratings["Weight"])
    max_rating = max(ratings["Weight"])
    print("Min rating: " + str(min_rating) + " Max rating: " + str(max_rating))
    dls = CollabDataLoaders.from_df(ratings, item_name='Target', bs=64)
    learn = collab_learner(dls, n_factors=50, y_range=(min_rating, max_rating))
    learn.fit_one_cycle(epochs)
    # Model weights
    target_w = learn.model.weight(dls.classes['Target'], is_item=True)
    source_w = learn.model.weight(dls.classes['Source'], is_item=False)
    return target_w, source_w

def make_model_nn(ratings, epochs):
    min_rating = min(ratings["Weight"])
    max_rating = max(ratings["Weight"])
    print("Min rating: " + str(min_rating) + " Max rating: " + str(max_rating))
    dls = CollabDataLoaders.from_df(ratings, item_name='Target', bs=64)
    learn = collab_learner(dls, use_nn=True, 
                           emb_szs={'userId': 50, 'movieId':50}, 
                           layers=[256, 128], y_range=(min_rating, max_rating))

    learn.fit_one_cycle(epochs)
    target_w = to_np(learn.model.embeds[1].weight[1:])
    source_w = to_np(learn.model.embeds[0].weight[1:])
    return target_w, source_w

def make_model(ratings, model_type, epochs):
    print("Model type: " + model_type)
    if model_type == "nn":
        return make_model_nn(ratings, epochs)
    else:
        return make_model_collab(ratings, epochs)

epochs = 5
model_type = "default"
```

```python id="sK_Iv-uOMywR" colab={"base_uri": "https://localhost:8080/", "height": 340} executionInfo={"status": "ok", "timestamp": 1633267376255, "user_tz": -330, "elapsed": 83348, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c80c2178-14c9-4199-a4ec-024becf0e18b"
# Train collab model on base dataset
target_w, source_w = make_model(ratings, model_type, epochs)

print(target_w.shape)
print(source_w.shape)

# Calculate cosine similarity matrix between all targets in the set
t_matrix = cosine_similarity(target_w)
                            
print(t_matrix.shape)
print()

# Calculate cosine similarity matrix between all sources in the set
s_matrix = cosine_similarity(source_w)

print(s_matrix.shape)
print()
```

<!-- #region id="bn6bsNXRMywT" -->
## Recommendations by target similarity
<!-- #endregion -->

```python id="Iu4J4O_2MywU" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633267376257, "user_tz": -330, "elapsed": 37, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2bca7ea8-16f3-4ac2-8c41-378eb8443602"
t_max_matches = 100 # top_n matches when doing target similarity

# Show closest matches to selected targets
samples = [target_tid, high_profile_tid]
print_similar_to_targets(samples, t_matrix)
print_target_similarity(target_tid, high_profile_tid, t_matrix)
```

```python id="MPFgg8aVMywV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633268540694, "user_tz": -330, "elapsed": 6128, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="86c8a846-84ab-4ade-c30f-b25f5b0d0432"
# graph tid similarity, to make a visualization
tid_inter = {}
threshold = 0.99
print(len(tid_name))
for tid, name in tid_name.items():
    matches = get_most_similar(tid, t_matrix, 50)
    if matches is not None:
        for item in matches:
            tid2, sim = item
            name2 = tid_name[tid2]
            if name != name2:
                if sim >= threshold:
                    if name not in tid_inter:
                        tid_inter[name] = Counter()
                    tid_inter[name][name2] = sim
print("Saving")
with open("tid_inter.csv", "w") as f:
    f.write("Source,Target,Weight\n")
    for source, targets in tid_inter.items():
        for target, weight in targets.items():
            f.write(str(source)+","+str(target)+","+str(weight)+"\n")
print("Done")
```

```python id="4r-pkpZNMywW"
# Build recommendations for source based on who they've retweeted
# For each target retweeted by a source, see if we have an entry in most_similar
# If we do, add each item to the recommended counter
# Assign the value to be the source's rating multiplied by the similarity score
# We'll also record what the user has already retweeted so we can recommend a target they haven't yet retweeted

# for target, num_retweets in get_source_retweets(source):
#    for similar, similarity in get_most_similar(target):
#        recommended[similar] += num_retweets * similarity

def get_user_recommendations_by_target(ratings, sid, t_matrix):
    s_ratings = ratings.loc[ratings['Source'] == sid]
    s_ratings = s_ratings.sort_values(by="Weight", ascending=False)
    s_r = list(zip(s_ratings['Target'], s_ratings['Weight']))
    recommended = Counter()
    seen = set()
    for item in s_r:
        tid, trating = item
        if tid > len(t_matrix):
            continue
        seen.add(tid)
        matches = get_most_similar(tid, t_matrix, t_max_matches)
        if matches != None:
            for entry in matches:
                t, r = entry
                recommended[t] += r * trating

    # Now we'll build a recomendations list that contains the highest scored items
    # calculated above that the user hasn't already rated
    seen_recommendations = Counter()
    not_seen_recommendations = Counter()
    for tid, score in recommended.most_common():
        if len(seen_recommendations) >= 10 and len(not_seen_recommendations) >= 10:
            break
        if tid not in seen:
            not_seen_recommendations[tid] = score
        else:
            seen_recommendations[tid] = score
    return seen_recommendations, not_seen_recommendations

def print_recommendations_by_target(sid, seen_recommendations, not_seen_recommendations):
    s_ratings = ratings.loc[ratings['Source'] == sid]
    s_ratings = s_ratings.sort_values(by="Weight", ascending=False)
    s_r = list(zip(s_ratings['Target'], s_ratings['Weight']))
    num_ratings = len(s_r)

    ind_rating = {}
    for item in s_r:
        ind, trating = item
        ind_rating[ind] = trating

    # Now let's print the output and see if it's sane
    print("User: " + sid_name[sid] + " retweeted " + str(num_ratings) + " different accounts.")
    print()
    top10 = []
    for item in s_r[:10]:
        tid, trating = item
        top10.append(tid)
        msg = "Retweeted by user: " + str(trating) + " times, total retweets: " + str(target_retweeters[tid]) 
        msg += "\t  " + tid_name[tid]
        print(msg)
    print()
    print("Recommended (seen):")
    for x, c in seen_recommendations.most_common(10):
        msg = "%.4f"%c + "\t(retweeted by user: " + str(ind_rating[x]) + " times,"
        msg += " total retweets: " + str(target_retweeters[x]) + ")" + "\t" + tid_name[x]
        if x == target_tid:
            msg += " [X]"
        if x in top10:
            msg += " [*]"
        print(msg)
    print()
    print("Recommended (not seen):")
    for x, c in not_seen_recommendations.most_common(10):
        msg = "%.4f"%c + "\t" + " (total retweets: " 
        msg += str(target_retweeters[x]) + ")\t" + tid_name[x]                  
        if x == target_tid:
            msg += " [X]"
        if x in top10:
            msg += " [*]"
        print(msg)
    print("=====================================================")
    print()

def print_user_recommendations_by_target(ratings, sid, t_matrix):
    seen, not_seen = get_user_recommendations_by_target(ratings, sid, t_matrix)
    print_recommendations_by_target(sid, seen, not_seen)
```

```python id="nXzd1VOwMywY" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633268564015, "user_tz": -330, "elapsed": 6876, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="42095b7a-b929-4d7c-aa43-0be47a8ca1e3"
# Display target-based recommendations for control set
for n in controls:
    print_user_recommendations_by_target(ratings, n, t_matrix)
```

<!-- #region id="qqw2JKSRMywZ" -->
## Recommendations by source similarity
<!-- #endregion -->

```python id="bzntztytMywb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633268564016, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d8a4a4a8-b867-4514-e46e-1e3c7becdb7d"
s_max_matches = 100 # top_n matches when doing source similarity

# Print 10 closest sources for control set
print_similar_to_sources(controls, s_matrix)
```

```python id="xolc3jWuMywc"
# Recommendations based on source similarity
# From the previously calculated source similarities, calculate recommendations thus:
# For each similar source, obtain their list of target ratings
# Record a counter for each target where we add a value: similarity * rating
# Once we have a ranked list of recommendations, choose the top items
# based on whether the original user has rated the target or not

# for similar_source, similarity in get_most_similar(source):
#     for target, num_retweets in get_source_retweets(similar_source):
#         recommended[target] += similarity * num_retweets


def get_user_recommendations_by_source(ratings, sid, s_matrix):
    s_ratings = ratings.loc[ratings['Source'] == sid]
    s_ratings = s_ratings.sort_values(by="Weight", ascending=False)
    s_r = list(zip(s_ratings['Target'], s_ratings['Weight']))
    seen = set()
    tid_rating = {}
    for item in s_r:
        tid, trating = item
        tid_rating[tid] = trating
        seen.add(tid)

    recommended = Counter()
    matches = get_most_similar(sid, s_matrix, s_max_matches)
    if matches != None:
        for item in matches:
            sid, similarity = item
            ur = ratings.loc[ratings['Source'] == sid]
            ur = list(zip(ur['Target'], ur['Weight']))
            for entry in ur:
                tid, mr = entry
                recommended[tid] += similarity * mr

    # Now we'll build a recomendations list that contains the highest scored items
    # calculated above that the user hasn't already rated
    seen_recommendations = Counter()
    not_seen_recommendations = Counter()
    for tid, score in recommended.most_common():
        if len(seen_recommendations) >= 10 and len(not_seen_recommendations) >= 10:
            break
        if tid not in seen:
            not_seen_recommendations[tid] = score
        else:
            seen_recommendations[tid] = score
    return seen_recommendations, not_seen_recommendations

def print_recommendations_by_source(sid, seen_recommendations, not_seen_recommendations):
    s_ratings = ratings.loc[ratings['Source'] == sid]
    s_ratings = s_ratings.sort_values(by="Weight", ascending=False)
    s_r = list(zip(s_ratings['Target'], s_ratings['Weight']))
    num_ratings = len(s_r)

    tid_rating = {}
    for item in s_r:
        tid, mrating = item
        tid_rating[tid] = mrating

    # Now let's print the output and see if it's sane
    print("User: " + sid_name[sid] + " retweeted " + str(num_ratings) + " different accounts.")
    print()
    top10 = []
    for item in s_r[:10]:
        tid, trating = item
        top10.append(tid)
        msg = "Retweeted by user: " + str(trating) + " times, total retweets: " + str(target_retweeters[tid]) 
        msg += "\t  " + tid_name[tid]
        print(msg)
    print()
    print("Recommended (seen):")
    for x, c in seen_recommendations.most_common(10):
        msg = "%.4f"%c
        msg += " (retweeted by user: " + str(tid_rating[x]) + " times,"
        msg += " total retweets: " + str(target_retweeters[x]) + ")" + "\t" 
        msg += tid_name[x]
        if x == target_tid:
            msg += " [X]"
        if x in top10:
            msg += " [*]"
        print(msg)
    print()
    print("Recommended (not seen):")
    for x, c in not_seen_recommendations.most_common(10):
        msg = "%.4f"%c 
        msg += " (total retweets: " + str(target_retweeters[x]) + ")\t"
        msg += tid_name[x]
        if x == target_tid:
            msg += " [X]"
        if x in top10:
            msg += " [*]"
        print(msg)
    print("=====================================================")
    print()

def print_user_recommendations_by_source(ratings, sid, s_matrix):
    seen, not_seen = get_user_recommendations_by_source(ratings, sid, s_matrix)
    print_recommendations_by_source(sid, seen, not_seen)
```

```python id="d-FfBs3yMywe" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633268574815, "user_tz": -330, "elapsed": 2751, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d043667f-7ab6-4c21-ee82-4225e0c1287f"
# Print source-based recommendations for control set
for n in controls:
    print_user_recommendations_by_source(ratings, n, s_matrix)
```

<!-- #region id="zvcfd-v5Mywf" -->
## Validation: count how many times target appeared in top_n recommendations for each source
<!-- #endregion -->

```python id="o2wfztRQMywf"
def validate_target_target(ratings, sid, t_matrix, target, top_n):
    ret = False
    seen, not_seen = get_user_recommendations_by_target(ratings, sid, t_matrix)
    top_tids = [x for x, c in not_seen.most_common(top_n)]
    if target in top_tids:
        ret = True
    return ret

def validate_target_source(ratings, sid, s_matrix, target, top_n):
    ret = False
    seen, not_seen = get_user_recommendations_by_source(ratings, sid, s_matrix)
    top_tids = [x for x, c in not_seen.most_common(top_n)]
    if target in top_tids:
        ret = True
    return ret
```

```python id="4IqUxE_jMywh" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633268587730, "user_tz": -330, "elapsed": 6833, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="cbe6ab4d-d9c4-470a-87f6-f53bb7e5dcc5"
ret = 0
top_n = 3
for sid in controls:
    found = validate_target_target(ratings, sid, t_matrix, target_tid, top_n)
    if found == True:
        ret += 1
print("Target was in top "+str(top_n)+" target-based recommendations for "+str(ret)+" users in control list.")
```

```python id="7ncbrwCwMywh" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1633268589092, "user_tz": -330, "elapsed": 1371, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0b2911ab-786e-49c6-c180-9741bfc9a549"
res = 0
top_n = 3
for sid in controls:
    found = validate_target_source(ratings, sid, s_matrix, target_tid, top_n)
    if found == True:
        res += 1
print("Target was in top "+str(top_n)+" source-based recommendations for "+str(res)+" users in control list.")
```

```python id="s4sY6KzYMywi"
# STOP
```

<!-- #region id="UN2iehxLMywi" -->
## Poisoning experiment 1 - randomly chosen amplifiers, variable amps, retweets
- with differing numbers of amplifiers and retweets:
    - repeat "iterations" times
        - create new poisoned dataframe based on supplied parameters
        - save csv for gephi visualization
        - train model
        - run source-based and target-based recommendations, see how often target appears in top_n recommendations
        - record all results to be graphed later
<!-- #endregion -->

```python id="W5QVwVcQMywj" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="f1e79e1b-65b5-4a9c-e29c-db64ea7da2e2"
# Run poisoning experiment 1
# Note that this cell can take a number of hours to run

# These are the experiments to run
# Each pair of numbers denotes [num_amplifiers, num_retweets]
# A poisoned copy of the dataset is generated as follows:
# 1. Make a copy of the original dataset
# 2. Randomly select num_amplifiers accounts from amplifier candidates
# 3. For each selected amplifier, add two rows to the copied dataset:
#    amplifier - target - num_retweets
#    amplifier - high_profile_user - num_retweets
# 4. Perform the rest of the experiment (train model, analyze recomendations)
experiments = [[0,0],
               [10,10], [10,20], [10,50], [10,100],
               [20,10], [20,20], [20,50], [20,100],
               [50,10], [50,20], [50,50], [50,100],
               [100,1], [100,5], [100,10], [100,20], [100,50],
               [200,1], [200,5], [200,10], [200,20], [200,50],
               [500,1], [500,5], [500,10], [500,20], [500,50],
               [1000,1], [1000,5], [1000,10], [1000,20], [1000,50],
               [2000,1], [2000,5], [2000,10], [2000,20], [2000,50],
               [4000,1], [4000,5], [4000,10], [4000,20], [4000,50]]

samples = [target_tid, high_profile_tid]
top_n = 3
iterations = 10
epochs = 5

# 1. Pick random accounts (not in control set) to do the boosting 
# that havent engaged with either high profile or target
amplifier_candidates = []
for sid, tids in source_retweeted.items():
    if len(tids) > 0:
        inter = set(tids).intersection(set(controls))
        if len(inter) == 0:
            if high_profile_tid not in tids and target_tid not in tids:
                amplifier_candidates.append(sid)
print("Number of random amplifier candidates: " + str(len(amplifier_candidates)))

# Loop through the experiment parameters
# For each set of parameters, perform the experiment iterations number of times
result_source = []
result_target = []
i = 1
save_dir = "US2020/exp1"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for item in experiments:
    amps, r = item
    for n in range(iterations):
        print()
        print("Experiment:" + str(i) + " amps:" + str(amps) + " r:" + str(r) + " take:" + str(n))
        i = i + 1
        save_path = save_dir + "/" + str(amps) + "_" + str(r) + "_" + str(n) + ".csv"
        new_ratings = get_poisoned_dataset(ratings, amplifier_candidates, amps, r, save_path)
        msg = "Base dataset length: " + str(len(ratings))
        msg += " Poisoned dataset length: " + str(len(new_ratings))
        print(msg)
        new_target_w, new_source_w = make_model(new_ratings, model_type, epochs)
        new_t_matrix = cosine_similarity(new_target_w)
        print_target_similarity(target_tid, high_profile_tid, new_t_matrix)
        new_s_matrix = cosine_similarity(new_source_w)
        ret = 0
        for sid in controls:
            found = validate_target_target(new_ratings, sid, new_t_matrix, target_tid, top_n)
            if found == True:
                ret += 1
        msg = "Target was in top " + str(top_n) 
        msg += " target recommendations for " + str(ret) 
        msg += " users in control list."
        print(msg)
        result_target.append([amps, r, n, ret])
        with open(save_dir + "/result_target.json", "w") as f:
            f.write(json.dumps(result_target, indent=4))
        res = 0
        for sid in controls:
            found = validate_target_source(new_ratings, sid, new_s_matrix, target_tid, top_n)
            if found == True:
                res += 1
        msg = "Target was in top " + str(top_n) 
        msg += " source recommendations for " + str(res) 
        msg += " users in control list."
        print(msg)
        result_source.append([amps, r, n, res])
        with open(save_dir + "/result_source.json", "w") as f:
            f.write(json.dumps(result_source, indent=4))
```

<!-- #region id="YOIZOah_Mywk" -->
## Display results as a plot
<!-- #endregion -->

```python id="AZplvJ4gMywk"
filename = "exp1/result_source.json"
title = "US2020 Experiment 1 - source-based recommendations"
results = []
with open(filename, "r") as f:
    results = json.loads(f.read())
results2 = []
order = []
for item in results:
    if len(item) == 4:
        a, r, t, v = item
        l = str(a) + "_" + str(r)
    else:
        m, t, v = item
        l = str(m)
    v = (v/20)*100
    if l not in order:
        order.append(l)
    results2.append([l, t, v])
df = pd.DataFrame(results2, columns=["params", "take", "val"])

plt.figure()
ax = None
fig = plt.figure(figsize=(20,8))
sns.set(style="whitegrid")
ax = sns.barplot(x="params", y="val", data=df, order=order, capsize=.2)
ax.set_title(title)
xlab = "Experiment parameters (num_accounts, num_retweets)"
ylab = "Percentage of control set that saw target account in top-3 recommendations"
plt.xlabel(xlab)
plt.ylabel(ylab)
for item in ax.get_xticklabels():
    item.set_rotation(45)
```

```python id="uNKKnFk7Mywl"
filename = "exp1/result_target.json"
title = "US2020 Experiment 1 - target-based recommendations"
results = []
with open(filename, "r") as f:
    results = json.loads(f.read())
results2 = []
order = []
for item in results:
    if len(item) == 4:
        a, r, t, v = item
        l = str(a) + "_" + str(r)
    else:
        m, t, v = item
        l = str(m)
    v = (v/20)*100
    if l not in order:
        order.append(l)
    results2.append([l, t, v])
df = pd.DataFrame(results2, columns=["params", "take", "val"])

plt.figure()
ax = None
fig = plt.figure(figsize=(20,8))
sns.set(style="whitegrid")
ax = sns.barplot(x="params", y="val", data=df, order=order, capsize=.2)
ax.set_title(title)
xlab = "Experiment parameters (num_accounts, num_retweets)"
ylab = "Percentage of control set that saw target account in top-3 recommendations"
plt.xlabel(xlab)
plt.ylabel(ylab)
for item in ax.get_xticklabels():
    item.set_rotation(45)
```

```python id="431krmZQMywm"
STOP
```

<!-- #region id="drwgDntmMywn" -->
## Poisoning experiment 2 - amplifiers chosen based on community
- with fixed number of amplifiers and retweets:
    - iterate through communities (discovered from louvain method)
    - if a community contains at least num_amplifiers, select a set of amplifiers randomly from the community
    - repeat "iterations" times
        - create new poisoned dataframe based on supplied parameters
        - train model
        - run source-based and target-based recommendations, see how often target appears in top_n recommendations
        - record all results to be graphed later
<!-- #endregion -->

```python id="HRb9OTY1Mywo"
# Run poisoning experiment 2

num_amplifiers = 200
num_retweets = 20

samples = [target_tid, high_profile_tid]
top_n = 3
iterations = 10
epochs = 5

result_source = []
result_target = []
i = 1
save_dir = "exp2"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

target_name = tid_name[target_tid]
high_profile_name = tid_name[high_profile_tid]
print("Community labels: " + ", ".join([str(x) for x in communities.keys()]))
for mod, names in communities.items():
    if target_name in names:
        print("target: " + target_name + " in community: " + str(mod) + " size: " + str(len(names)))
    if high_profile_name in names:
        print("high_profile: " + high_profile_name + " in community: " + str(mod) + " size: " + str(len(names)))

for mod, names in sorted(communities.items()):
    print("Community: " + str(mod) + " contains " + str(len(names)) + " names.")
    amplifier_candidates = []
    for name in names:
        if name in name_sid:
            sid = name_sid[name]
            if sid in source_retweeted:
                rtw = source_retweeted[sid]
                if high_profile_tid not in rtw and target_tid not in rtw:
                    amplifier_candidates.append(sid)
    if len(amplifier_candidates) < num_amplifiers:
        print("Skipping community: "+str(mod)+" (only found "+str(len(amplifier_candidates))+" candidates).")
        continue
    for n in range(iterations):
        print()
        print("Experiment:" + str(i) + " community:" + str(mod) + " take:" + str(n))
        i = i + 1
        save_path = save_dir + "/" + str(mod) + "_" + str(n) + ".csv"
        new_ratings = get_poisoned_dataset(ratings, amplifier_candidates, 
                                           num_amplifiers, num_retweets, save_path)
        msg = "Base dataset length: " + str(len(ratings))
        msg += " Poisoned dataset length: " + str(len(new_ratings))
        print(msg)
        new_target_w, new_source_w = make_model(new_ratings, model_type, epochs)
        new_t_matrix = cosine_similarity(new_target_w)
        print_target_similarity(target_tid, high_profile_tid, new_t_matrix)
        new_s_matrix = cosine_similarity(new_source_w)
        ret = 0
        for sid in controls:
            found = validate_target_target(new_ratings, sid, new_t_matrix, target_tid, top_n)
            if found == True:
                ret += 1
        msg = "Target was in top " + str(top_n) 
        msg += " target recommendations for " + str(ret) 
        msg += " users in control list."
        print(msg)
        result_target.append([mod, n, ret])
        with open(save_dir + "/result_target.json", "w") as f:
            f.write(json.dumps(result_target, indent=4))
        res = 0
        for sid in controls:
            found = validate_target_source(new_ratings, sid, new_s_matrix, target_tid, top_n)
            if found == True:
                res += 1
        msg = "Target was in top " + str(top_n) 
        msg += " source recommendations for " + str(res) 
        msg += " users in control list."
        print(msg)
        result_source.append([mod, n, res])
        with open(save_dir + "/result_source.json", "w") as f:
            f.write(json.dumps(result_source, indent=4))
```

```python id="KP2FJSJ8Mywt"
filename = "exp2/result_source.json"
title = "US2020 Experiment 2 - source-based recommendations"
results = []
with open(filename, "r") as f:
    results = json.loads(f.read())
results2 = []
order = []
for item in results:
    if len(item) == 4:
        a, r, t, v = item
        l = str(a) + "_" + str(r)
    else:
        m, t, v = item
        l = str(m)
    v = (v/20)*100
    if l not in order:
        order.append(l)
    results2.append([l, t, v])
df = pd.DataFrame(results2, columns=["params", "take", "val"])

plt.figure()
ax = None
fig = plt.figure(figsize=(20,8))
sns.set(style="whitegrid")
ax = sns.barplot(x="params", y="val", data=df, order=order, capsize=.2)
ax.set_title(title)
xlab = "Experiment parameters (community label)"
ylab = "Percentage of control set that saw target account in top-3 recommendations"
plt.xlabel(xlab)
plt.ylabel(ylab)
for item in ax.get_xticklabels():
    item.set_rotation(45)
```

```python id="dHjsXimdMywv"
filename = "exp2/result_target.json"
title = "US2020 Experiment 2 - target-based recommendations"
results = []
with open(filename, "r") as f:
    results = json.loads(f.read())
results2 = []
order = []
for item in results:
    if len(item) == 4:
        a, r, t, v = item
        l = str(a) + "_" + str(r)
    else:
        m, t, v = item
        l = str(m)
    v = (v/20)*100
    if l not in order:
        order.append(l)
    results2.append([l, t, v])
df = pd.DataFrame(results2, columns=["params", "take", "val"])

plt.figure()
ax = None
fig = plt.figure(figsize=(20,8))
sns.set(style="whitegrid")
ax = sns.barplot(x="params", y="val", data=df, order=order, capsize=.2)
ax.set_title(title)
xlab = "Experiment parameters (community label)"
ylab = "Percentage of control set that saw target account in top-3 recommendations"
plt.xlabel(xlab)
plt.ylabel(ylab)
for item in ax.get_xticklabels():
    item.set_rotation(45)
```

```python id="xW808hs4Myww"
STOP
```

<!-- #region id="3e46oPpjMywx" -->
## Poisoning experiment 3 - amplifiers chosen based on similarity to control accounts
- with varying number of amplifiers and retweets:
    - select a set of amplifiers that are similar to control accounts
    - repeat "iterations" times
        - create new poisoned dataframe based on supplied parameters
        - train model
        - run source-based and target-based recommendations, see how often target appears in top_n recommendations
        - record all results to be graphed later
<!-- #endregion -->

```python id="wunHOw4EMywy"
# Run poisoning experiment 3

experiments = [[0,0],
               [100,1], [100,5], [100,10], [100,20],
               [200,1], [200,5], [200,10], [200,20], 
               [500,1], [500,5], [500,10], [500,20], 
               [1000,1], [1000,5], [1000,10], [1000,20],
               [2000,1], [2000,5], [2000,10], [2000,20]]

samples = [target_tid, high_profile_tid]
top_n = 3
iterations = 10
epochs = 5

# 1. Pick accounts most similar to those in the control set
# that havent engaged with either high profile or target
# and aren't in the control group
amplifier_candidates = []
sims = set()
for sid in controls:
    sim = get_most_similar(sid, s_matrix, 250)
    for s, _ in sim:
        if s not in controls:
            if s in source_retweeted:
                rtw = source_retweeted[s]
                if high_profile_tid not in rtw and target_tid not in rtw:
                    sims.add(s)
amplifier_candidates = list(sims)
print("Number of amplifier candidates: " + str(len(amplifier_candidates)))
# Loop through the experiment parameters
# For each set of parameters, perform the experiment iterations number of times
result_source = []
result_target = []
i = 1
save_dir = "exp3"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for item in experiments:
    amps, r = item
    for n in range(iterations):
        print()
        print("Experiment:" + str(i) + " amps:" + str(amps) + " r:" + str(r) + " take:" + str(n))
        i = i + 1
        save_path = save_dir + "/" + str(amps) + "_" + str(r) + "_" + str(n) + ".csv"
        new_ratings = get_poisoned_dataset(ratings, amplifier_candidates, amps, r, save_path)
        msg = "Base dataset length: " + str(len(ratings))
        msg += " Poisoned dataset length: " + str(len(new_ratings))
        print(msg)
        new_target_w, new_source_w = make_model(new_ratings, model_type, epochs)
        new_t_matrix = cosine_similarity(new_target_w)
        print_target_similarity(target_tid, high_profile_tid, new_t_matrix)
        new_s_matrix = cosine_similarity(new_source_w)
        ret = 0
        for sid in controls:
            found = validate_target_target(new_ratings, sid, new_t_matrix, target_tid, top_n)
            if found == True:
                ret += 1
        msg = "Target was in top " + str(top_n) 
        msg += " target recommendations for " + str(ret) 
        msg += " users in control list."
        print(msg)
        result_target.append([amps, r, n, ret])
        with open(save_dir + "/result_target.json", "w") as f:
            f.write(json.dumps(result_target, indent=4))
        res = 0
        for sid in controls:
            found = validate_target_source(new_ratings, sid, new_s_matrix, target_tid, top_n)
            if found == True:
                res += 1
        msg = "Target was in top " + str(top_n) 
        msg += " source recommendations for " + str(res) 
        msg += " users in control list."
        print(msg)
        result_source.append([amps, r, n, res])
        with open(save_dir + "/result_source.json", "w") as f:
            f.write(json.dumps(result_source, indent=4))
```

```python id="XkgvUNHwMywz"
filename = "exp3/result_source.json"
title = "US2020 Experiment 3 - source-based recommendations"
results = []
with open(filename, "r") as f:
    results = json.loads(f.read())
results2 = []
order = []
for item in results:
    if len(item) == 4:
        a, r, t, v = item
        l = str(a) + "_" + str(r)
    else:
        m, t, v = item
        l = str(m)
    v = (v/20)*100
    if l not in order:
        order.append(l)
    results2.append([l, t, v])
df = pd.DataFrame(results2, columns=["params", "take", "val"])

plt.figure()
ax = None
fig = plt.figure(figsize=(20,8))
sns.set(style="whitegrid")
ax = sns.barplot(x="params", y="val", data=df, order=order, capsize=.2)
ax.set_title(title)
xlab = "Experiment parameters (num_accounts, num_retweets)"
ylab = "Percentage of control set that saw target account in top-3 recommendations"
plt.xlabel(xlab)
plt.ylabel(ylab)
for item in ax.get_xticklabels():
    item.set_rotation(45)
```

```python id="5Se-Uv-CMyw0"
filename = "exp3/result_target.json"
title = "US2020 Experiment 3 - target-based recommendations"
results = []
with open(filename, "r") as f:
    results = json.loads(f.read())
results2 = []
order = []
for item in results:
    if len(item) == 4:
        a, r, t, v = item
        l = str(a) + "_" + str(r)
    else:
        m, t, v = item
        l = str(m)
    v = (v/20)*100
    if l not in order:
        order.append(l)
    results2.append([l, t, v])
df = pd.DataFrame(results2, columns=["params", "take", "val"])

plt.figure()
ax = None
fig = plt.figure(figsize=(20,8))
sns.set(style="whitegrid")
ax = sns.barplot(x="params", y="val", data=df, order=order, capsize=.2)
ax.set_title(title)
xlab = "Experiment parameters (num_accounts, num_retweets)"
ylab = "Percentage of control set that saw target account in top-3 recommendations"
plt.xlabel(xlab)
plt.ylabel(ylab)
for item in ax.get_xticklabels():
    item.set_rotation(45)
```

```python id="L-AMXodIMyw0"
STOP
```
