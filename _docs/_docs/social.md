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

<!-- #region id="UR8sEDb_SEVh" -->
# Predicting Future Friendships from Social Network data

## Predicting future friendships from social network data

Welcome to FriendHook, Silicon Valley’s hottest new startup. FriendHook is a social networking app for college undergrads. To join, an undergrad must scan their college ID to prove their affiliation. After approval, undergrads can create a FriendHook profile, which lists their dorm name and scholastic interests. Once a profile is created, an undergrad can send *friend requests* to other students at their college. A student who receives a friend request can either approve or reject it. When a friend request is approved, the pair of students are officially *FriendHook friends*. Using their new digital connection, FriendHook friends can share photographs, collaborate on coursework, and keep each other up to date on the latest campus gossip.

The FriendHook app is a hit. It’s utilized on hundreds of college campuses worldwide. The user base is growing, and so is the company. You are FriendHook’s first data science hire! Your first challenging task will be to work on FriendHook’s friend recommendation algorithm.

Sometimes FriendHook users have trouble finding their real-life friends on the digital app. To facilitate more connections, the engineering team has implemented a simple friend-recommendation engine. Once a week, all users receive an email recommending a new friend who is not yet in their network. The users can ignore the email, or they can send a friend request. That request is then either accepted or rejected/ignored.

Currently, the recommendation engine follows a simple algorithm called the *friend-of-a-friend recommendation algorithm*. The algorithm works like this. Suppose we want to recommend a new friend for student A. We pick a random student B who is already friends with student A. We then pick a random student C who is friends with student B but not student A. Student C is then selected as the recommended friend for student A.
<!-- #endregion -->

<!-- #region id="rzhafgGjFh_8" -->
<!-- #endregion -->

<!-- #region id="FeF_omkZFjwc" -->
*The friend-of-a-friend recommendation algorithm in action. Mary has two friends: Fred and Bob. One of these friends (Bob) is randomly selected. Bob has two additional friends: Marty and Alice. Neither Alice nor Marty are friends with Mary. A friend of a friend (Marty) is randomly selected. Mary receives an email suggesting that she should send a friend request to Marty.*
<!-- #endregion -->

<!-- #region id="4l0MExh7Fmxx" -->
Essentially, the algorithm assumes that a friend of your friend is also likely to be your friend. This assumption is reasonable but also a bit simplistic. How well does this assumption hold? Nobody knows! However, as the company’s first data scientist, it’s your job to find out. You have been tasked with building a model that predicts student behavior in response to the recommendation algorithm.

Your task is to build a model that predicts user behavior based on user profiles and social network data. The model must generalize to other colleges and universities. This generalizability is very important—a model that cannot be utilized at other colleges is worthless to the product team. Consider, for example, a model that accurately predicts behavior in one or two of the dorms at the sampled university. In other words, it requires specific dorm names to make accurate predictions. Such a model is not useful because other universities will have different dormitory names. Ideally, the model should generalize to all dormitories across all universities worldwide.

Once you’ve built the generalized model, you should explore its inner workings. Your goal is to gain insights into how university life facilitates new FriendHook connections.

The project goals are ambitious but also very doable. You can complete them by carrying out the following tasks:

1. Load the three datasets pertaining to user behavior, user profiles, and the user friendship network. Explore each dataset, and clean it as required.
2. Build and evaluate a model that predicts user behavior based on user profiles and established friendship connections. You can optionally split this task into two subtasks: build a model using just the friendship network, and then add the profile information and test whether this improves the model’s performance.
3. Determine whether the model generalizes well to other universities.
4. Explore the inner workings of the model to gain better insights into student behavior.

Our data contains three files stored in a friendhook directory. These files are CSV tables and are named Profiles.csv, Observations.csv, and Friendships.csv. Let’s discuss each table individually.

**The Profiles table**

Profiles.csv contains profile information for all the students at the chosen university. This information is distributed across six columns: `Profile_ID`, `Sex`, `Relationship_ Status`, `Major`, `Dorm`, and `Year`. Maintaining student privacy is very important to the FriendHook team, so all the profile information has been carefully encrypted. FriendHook’s encryption algorithm takes in descriptive text and returns a unique, scrambled 12-character code known as a *hash code*. Suppose, for example, that a student lists their major as physics. The word *physics* is then scrambled and replaced with a hash code such as *b90a1221d2bc*. If another student lists their major as art history, a different hash code is returned (for example, *983a9b1dc2ef*). In this manner, we can check whether two students share the same major without necessarily knowing the identity of that major. All six profile columns have been encrypted as a precautionary measure. Let’s discuss the separate columns in detail:

- `Profile_ID`—A unique identifier used to track each student. The identifier can be linked to the user behaviors in the `Observations` table. It can also be linked to FriendHook connections in the `Friendships` table.
- `Sex`—This optional field describes the sex of a student as `Male` or `Female`. Students who don’t wish to specify a gender can leave the `Sex` field blank. Blank inputs are stored as empty values in the table.
- `Relationship_Status`—This optional field specifies the relationship status of the student. Each student has three relationship categories to choose from: `Single`, `In a Relationship`, or `It’s Complicated`. All students have a fourth option of leaving this field blank. Blank inputs are stored as empty values in the table.
- `Major`—The chosen area of study for the student, such as physics, history, economics, etc. This field is required to activate a FriendHook account. Students who have not yet picked their major can select `Undecided` from among the options.
- `Dorm`—The name of the dormitory where the student resides. This field is required to activate a FriendHook account. Students who reside in off-campus housing can select `Off-Campus Housing` from among the options.
- `Year`—The undergraduate student’s year. This field must be set to one of four options: `Freshman`, `Sophomore`, `Junior`, or `Senior`.

**The Observations table**

Observations.csv contains the observed user behavior in response to the emailed friend recommendation. It includes the following five fields:

- `Profile_ID`—The ID of the user who received a friend recommendation. The ID corresponds to the profile ID in the `Profiles` table.
- `Selected_Friend`—An existing friend of the user in the `Profile_ID` column.
- `Selected_Friend_of_Friend`—A randomly chosen friend of `Selected_Friend` who is not yet a friend of `Profile_ID`. This random friend of a friend is emailed as a friend recommendation for the user.
- `Friend_Requent_Sent`—A Boolean column that is `True` if a user sends a friend request to the suggested friend of a friend or `False` otherwise.
- `Friend_Request_Accepted`—A Boolean column that is `True` only if a user sends a friend request and that request is accepted.

This table stores all the observed user behaviors in response to the weekly recommendation email. Our goal is to predict the Boolean outputs of the final two table columns based on the profile and social networking data.

**The Friendships table**

Friendships.csv contains the FriendHook friendship network corresponding to the selected university. This network was used as input into the friend-of-a-friend recommendation algorithm. The `Friendships` table has just two columns: `Friend A` and `Friend B`. These columns contain profile IDs that map to the `Profile_ID` columns of the `Profiles` and `Observations` tables. Each row corresponds to a pair of FriendHook friends. For instance, the first row contains IDs `b8bc075e54b9` and `49194b3720b6`. From these IDs, we can infer that the associated students have an established FriendHook connection. Using the IDs, we can look up the profile of each student. The profiles then allow us to explore whether the friends share the same major or reside together in the same dorm.

To address the problem at hand, we need to know how to do the following:

- Analyze network data using Python
- Discover friendship clusters in social networks
- Train and evaluate supervised machine learning models
- Probe the inner workings of trained models to draw insights from our data

FriendHook is a popular social networking app designed for college campuses. Students can connect as friends in the FriendHook network. A recommendation engine emails users weekly with new friend suggestions based on their existing connections; students can ignore these recommendations, or they can send out friend requests. We have been provided with one week’s worth of data pertaining to friend recommendations and student responses. That data is stored in the friendhook/Observations.csv file. We’re provided with two additional files: friendhook/Profiles.csv and friendhook/Friendships.csv, containing user profile information and the friendship graph, respectively. The user profiles have been encrypted to protect student privacy. Our goal is to build a model that predicts user behavior in response to the friend recommendations. We will do so by following these steps:

1. Load the three datasets containing the observations, user profiles, and friendship connections.
2. Train and evaluate a supervised model that predicts behavior based on network features and profile features. We can optionally split this task into two subtasks: training a model using network features, and then adding profile features and evaluating the shift in model performance.
3. Check to ensure that the model generalizes well to other universities.
4. Explore the inner workings of our model to gain better insights into student behavior.
<!-- #endregion -->

```python id="5Qv4ngnqR5sf"
import warnings
warnings.filterwarnings('ignore')
```

```python colab={"base_uri": "https://localhost:8080/"} id="Z9S5jC3VSibR" executionInfo={"status": "ok", "timestamp": 1637511500127, "user_tz": -330, "elapsed": 1688, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f3473e09-508e-465f-852a-3e649822333a"
!wget -q --show-progress https://github.com/sparsh-ai/general-recsys/raw/T426474/Case_Study5/friendhook/Friendships.csv
!wget -q --show-progress https://github.com/sparsh-ai/general-recsys/raw/T426474/Case_Study5/friendhook/Observations.csv
!wget -q --show-progress https://github.com/sparsh-ai/general-recsys/raw/T426474/Case_Study5/friendhook/Profiles.csv
```

<!-- #region id="Rq1BGCEHR5sn" -->
## Exploring the Data

Let's separately explore the _Profiles_, _Observations_ and _Friendships_ tables.

### Examining the Profiles
We'll start by loading the _Profiles_ table into Pandas and summarizing the table's contents.

**Loading the Profiles table**
<!-- #endregion -->

```python id="lGpfo2Q1R5sq" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511510773, "user_tz": -330, "elapsed": 452, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="38efdd5e-26b5-4a32-d84e-27ded04a8d66"
import pandas as pd

def summarize_table(df):
    n_rows, n_columns = df.shape
    summary = df.describe()
    print(f"The table contains {n_rows} rows and {n_columns} columns.")
    print("Table Summary:\n")
    print(summary.to_string())

df_profile = pd.read_csv('Profiles.csv')
summarize_table(df_profile)
```

<!-- #region id="UGwQZxlNR5ss" -->
There is one place in the table summary where the numbers are off; the _Relationship Status_ column. Pandas has detected three _Relationship Status_ categories across 3631 of 4039 table rows. The remaining 400 or so rows are null. They don't contain any assigned relationship status. Lets count the total number of empty rows.

**Counting empty Relationship Status profiles**
<!-- #endregion -->

```python id="p9k-ODzhR5st" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511517033, "user_tz": -330, "elapsed": 421, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e120f3f6-2bc2-493b-c500-aa47f8c924ae"
is_null = df_profile.Relationship_Status.isnull()
num_null = df_profile[is_null].shape[0]
print(f"{num_null} profiles are missing the Relationship Status field.")
```

<!-- #region id="hZyWW76fR5su" -->
408 profiles do not contain a listed relationship status.  We can treat the lack of status as a fourth _unspecified_ relationship status category. Hence, we should assign these rows a category id. What id value should we choose? Before we answer the question, lets examine all unique ids within the _Relationship Status_ column.

**Checking unique Relationship Status values**
<!-- #endregion -->

```python id="Sdlzm8CRR5sv" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511519520, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c6bc7dcf-5cab-4beb-f2cc-b26bed9b70c1"
unique_ids = set(df_profile.Relationship_Status.values)
print(unique_ids)
```

<!-- #region id="zS5YSLPwR5sx" -->
The Scikit-Learn library is unable to process hash-codes or null values. It can only process numbers. Hence, we'll need to eventually convert the categories to numeric values. 

**Mapping Relationship Status values to numbers**
<!-- #endregion -->

```python id="5a7Y6WJvR5sy"
import numpy as np
category_map = {'9cea719429e9': 0, np.nan: 1, '188f9a32c360': 2, 
                'ac0b88e46e20': 3}
```

<!-- #region id="iluKRgqkR5s0" -->
Next, we'll replace the contents of the _Relationship Status_ column with the appropriate numeric values.

**Updating the Relationship Status column**
<!-- #endregion -->

```python id="9x0FZu3zR5s0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511558269, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2bd5bd0a-de9b-4f18-a912-c4f37bf4a326"
nums = [category_map[hash_code] 
        for hash_code in df_profile.Relationship_Status.values]
df_profile['Relationship_Status'] = nums
print(df_profile.Relationship_Status)
```

<!-- #region id="zStc1B59R5s1" -->
We've transformed _Relationship Status_ into a numeric variable. However, the remaining five columns in the table still contain hash-codes. Let's create a category mapping between hash-codes and numbers in each column. We'll track the category mappings in each column with a `col_to_mapping` dictionary. We'll also leverage the mappings in order to replace all hash-codes with numbers in `df_profile`.

**Replacing all Profile hash-codes with numeric values**
<!-- #endregion -->

```python id="yDm83aBZR5s1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511561919, "user_tz": -330, "elapsed": 433, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="110fa181-5c97-4182-bdef-6dadf77ce0dd"
col_to_mapping = {'Relationship_Status': category_map}

for column in df_profile.columns:
    if column in col_to_mapping:
        continue
        
    unique_ids = sorted(set(df_profile[column].values))
    category_map = {id_: i for i, id_ in enumerate(unique_ids)}
    col_to_mapping[column] = category_map
    nums = [category_map[hash_code] 
            for hash_code in df_profile[column].values]
    df_profile[column] = nums

head = df_profile.head()
print(head.to_string(index=False))
```

<!-- #region id="s-ToK8jUR5s3" -->
We've finished tweaking the `df_profile`. Now lets turn our attention to the table of experimental observations.

### Exploring the Experimental Observations
We'll start by loading the _Observations_ table into Pandas and summarizing the table's contents.

**Loading the Observations table**
<!-- #endregion -->

```python id="IKHOylu-R5s3" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511580047, "user_tz": -330, "elapsed": 425, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3205f7cc-6adb-483e-be43-1759cf47d799"
df_obs = pd.read_csv('Observations.csv')
summarize_table(df_obs)
```

<!-- #region id="SL0z-jZyR5s4" -->
The five table columns all consistantly show 4039 filled rows. There are no empty values in the table. This is good. However, the actual column names are hard to read. The names are very descriptive, but also very long. We should shorten some of the names in order to ease our cognitive load. 

**Renaming the observation columns**
<!-- #endregion -->

```python id="0QykAd_YR5s5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511582221, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="53fe4b1b-f161-4284-d5a5-57517a8430f2"
new_names = {'Selected_Friend': 'Friend', 
             'Selected_Friend_of_Friend': 'FoF',
             'Friend_Request_Sent': 'Sent',
             'Friend_Request_Accepted': 'Accepted'}
df_obs = df_obs.rename(columns=new_names)
summarize_table(df_obs)
```

<!-- #region id="Ou5Xqi_aR5s5" -->
Approximately 62% (2519) of the friend suggestions lead to friend request being sent. This is very promising; the friend-of-a-friend suggestions are quite effective. Furthermore, approximately 60% (2460) of sampled instances led to a friend request getting accepted. Hence, the sent friend requests are ignored or rejected just 2% (2519 - 2460 = 50) of the time. Of course, our numbers assume that there are no observations where _Sent_ is False and _Accepted_ is True. This scenario is not possible, because a friend-request cannot be accepted if it has not yet been sent. Still, as sanity check, lets test the integrity of the data by confirm that the scenario does not take place.

**Ensuring that Sent is `True` for all accepted requests**
<!-- #endregion -->

```python id="6lbK8y0-R5s6"
condition = (df_obs.Sent == False) & (df_obs.Accepted == True)
assert not df_obs[condition].shape[0]
```

<!-- #region id="p8brARl6R5s6" -->
Based on our observations, user behavior follows three possible scenarios. Hence, we can encode this categorical behavior by assigning numbers `0`, `1`, and `2` to the behavior patterns.

**Assigning classes of behavior to the user observations**
<!-- #endregion -->

```python id="Rm1IbPpDR5s7"
behaviors = []
for sent, accepted in df_obs[['Sent', 'Accepted']].values:
    behavior = 2 if (sent and not accepted) else int(sent) * int(accepted)
    behaviors.append(behavior)
df_obs['Behavior'] = behaviors
```

<!-- #region id="uEqNg7SqR5s7" -->
Additionally, we must transform the profile ids in the first three columns from hash-codes to numeric ids that are consistent with `df_profile.Profile_ID`. 

**Replacing all Observation hash-codes with numeric values**
<!-- #endregion -->

```python id="nnVtphvHR5s7" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511585199, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0c9abbe6-fe38-421c-839a-7ecced2909b4"
for col in ['Profile_ID', 'Friend', 'FoF']:
    nums = [col_to_mapping['Profile_ID'][hash_code]
            for hash_code in df_obs[col]]
    df_obs[col] = nums

head = df_obs.head()
print(head.to_string(index=False))
```

<!-- #region id="RKYhXx1qR5s8" -->
The `df_obs` now aligns with `df_profile`. Only a single data table remains unanalyzed. Lets proceed to explore the friendship linkages within the remaining _Friendships_ table.


### Exploring Friendships Linkage Table
We'll start by loading the _Friendships_ table into Pandas and summarizing the table's contents.

**Loading the Friendships table**
<!-- #endregion -->

```python id="6rfPDqBNR5s8" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511594631, "user_tz": -330, "elapsed": 581, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="680d7d2b-c617-457e-8368-5eed9a2709ae"
df_friends = pd.read_csv('Friendships.csv')
summarize_table(df_friends)
```

<!-- #region id="CQZ8eJ5wR5s9" -->
In order to carry out a more detailed analysis, we should load the frienship data into a NetworkX graph. 

**Loading the social graph into NetworkX**
<!-- #endregion -->

```python id="mGF4ZMpsR5s9" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511595734, "user_tz": -330, "elapsed": 620, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8a4b0ce4-457c-44cf-b442-8660b405842c"
import networkx as nx
G = nx.Graph()
for id1, id2 in df_friends.values:
    node1 = col_to_mapping['Profile_ID'][id1]
    node2 = col_to_mapping['Profile_ID'][id2]
    G.add_edge(node1, node2)
    
nodes = list(G.nodes)
num_nodes = len(nodes)
print(f"The social graph contains {num_nodes} nodes.")
```

<!-- #region id="0kLVaYBsR5s9" -->
Lets try to gain more insights into the graph structure by visualizing it with `nx.draw`. Please note the graph is rather large, so visulization might take 10-30 seconds to load.

**Visualizing the social graph**
<!-- #endregion -->

```python id="xmk3JhjQR5s9" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637511667288, "user_tz": -330, "elapsed": 69058, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b0f7ab28-f207-4190-bc50-02a56d76bda5"
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
nx.draw(G, node_size=5)
plt.show()
```

<!-- #region id="DA4vLks5R5s-" -->
Tightly clustered social groups are clearly visible within the network. Lets extract these groups using Markov clustering. 

**Finding social groups using Markov clustering**
<!-- #endregion -->

```python id="szZqOhZZTLzI"
!pip install -q markov_clustering
```

```python id="j2lf-aypR5s-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511675632, "user_tz": -330, "elapsed": 2648, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="47952b39-0748-45b1-d21a-d6a2e80d9cd3"
import markov_clustering as mc
matrix = nx.to_scipy_sparse_matrix(G)
result = mc.run_mcl(matrix)
clusters = mc.get_clusters(result) 
num_clusters = len(clusters)
print(f"{num_clusters} clusters were found in the social graph.")
```

<!-- #region id="Aaz8KRiLR5s-" -->
Ten clusters were found in the social graph. Lets visualize these clusters by coloring each node based on cluster id. To start, we'll need to iterate over `clusters` and assign a `cluster_id` attribute to every node.

**Assigning cluster attributes to nodes**
<!-- #endregion -->

```python id="jKBZmK0CR5s-"
for cluster_id, node_indices in enumerate(clusters):
    for i in node_indices:
        node = nodes[i]
        G.nodes[node]['cluster_id'] = cluster_id
```

<!-- #region id="aSXRgvFAR5s_" -->
Next, we'll color the nodes based on their attribute assignment.

**Coloring the nodes by cluster assignment**
<!-- #endregion -->

```python id="5ahEfgRFR5s_" colab={"base_uri": "https://localhost:8080/", "height": 319} executionInfo={"status": "ok", "timestamp": 1637511748316, "user_tz": -330, "elapsed": 69108, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6a083d3c-198b-4b25-910a-00c8e6fe660d"
np.random.seed(0)
colors = [G.nodes[n]['cluster_id'] for n in G.nodes]
nx.draw(G, node_size=5, node_color=colors, cmap=plt.cm.tab20)
plt.show()
```

<!-- #region id="ezbAq8OSR5tA" -->
The cluster colors clearly correspond to tight social groups. Our clustering thus has been effective. Hence, the assigned `cluster_id` attributes should prove useful during the model-building process. In this same manner, it might be useful to store all five profile features as attributes within the student nodes. 

**Assigning profile attributes to nodes**
<!-- #endregion -->

```python id="s5x-wrTzR5tA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511749092, "user_tz": -330, "elapsed": 783, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dcdfe7f4-7a28-4626-97e0-949bca1ba0f2"
attribute_names = df_profile.columns
for attributes in df_profile.values:
    profile_id = attributes[0]
    for name, att in zip(attribute_names[1:], attributes[1:]):
        G.nodes[profile_id][name] = att
        
first_node = nodes[0]
print(f"Attributes of node {first_node}:")
print(G.nodes[first_node])
```

<!-- #region id="hz2gpSS8R5tB" -->
We have finished exploring our input data. Now, we'll proceed to train a model that predicts user behavior. We'll start by constructing simple model that only utilizes network features.

## Training a Predictive Model Using Network Features

Our goal is to train a supervised ML model on our dataset, in order to predict user behavior. Currently, all possible classes of behavior are stored within  the _Behavior_ columns of `df_obs`. We'll assign our training class-label array to equal the `df_obs.Behavior` column.

**Assigning the class-label array `y`**
<!-- #endregion -->

```python id="RpMLjsHwR5tB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511749093, "user_tz": -330, "elapsed": 44, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c7b9d27d-30a4-496b-b193-76f2d70b6ea8"
y = df_obs.Behavior.values
print(y)
```

<!-- #region id="A1uWDrQHR5tC" -->
Now that we have class-labels, we'll need to create a feature-matrix `X`. Let's create an initial version of `X`, and populate it with some very basic features. The simplest question we can ask about a person within a social network is this; how many friends does the person have? Of course, that value equals edge-count associated with the person's node within the social graph. Lets make the edge-count our first feature in the feature-matrix. We'll iterate over all the rows in `df_obs` and assign an edge-count to each profile that's referenced within each row. 

**Creating a feature-matrix from edge counts**
<!-- #endregion -->

```python id="S5mqLCFyR5tD"
cols = ['Profile_ID', 'Friend', 'FoF']
features = {f'{col}_Edge_Count': [] for col in cols}
for node_ids in df_obs[cols].values:
    for node, feature_name in zip(node_ids, features.keys()):
        degree = G.degree(node)
        features[feature_name].append(degree)

df_features = pd.DataFrame(features)
X = df_features.values
```

<!-- #region id="g29SJDPtR5tD" -->
Lets quickly check the quality of the signal in the training data by training and testing a simple model. We have multiple possible models to choose from. One sensible choice is a decision tree classifier. Decision trees can handle non-linear decision boundaries and are easily interpretable. 

**Training and evaluating a Decision tree classifier**
<!-- #endregion -->

```python id="u8g5qI7qR5tD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511922904, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2d41bdd3-2e1e-4d54-904a-531788a64ed9"
np.random.seed(0)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def evaluate(X, y, model_type=DecisionTreeClassifier, **kwargs):
    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = model_type(**kwargs)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    f_measure = f1_score(pred, y_test, average='macro')
    return f_measure, clf

f_measure, clf = evaluate(X, y)
print(f"The f-measure is {f_measure:0.2f}")
```

<!-- #region id="U8UWaIfIR5tE" -->
Our f-measure is terrible! Clearly, friend-count by itself is not a sufficient signal for predicting user-behavior. Perhaps adding PageRank values to our training set will yield improved results? 

**Adding PageRank features**
<!-- #endregion -->

```python id="LueVuNuUR5tE" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511925750, "user_tz": -330, "elapsed": 1074, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0bb4aa49-a632-4b5d-f096-43e8219bfd0d"
np.random.seed(0)
node_to_pagerank = nx.pagerank(G)
features = {f'{col}_PageRank': [] for col in cols}
for node_ids in df_obs[cols].values:
    for node, feature_name in zip(node_ids, features.keys()):
        pagerank = node_to_pagerank[node]
        features[feature_name].append(pagerank)

def update_features(new_features):
    for feature_name, values in new_features.items():
        df_features[feature_name] = values
    return df_features.values

X = update_features(features)
f_measure, clf = evaluate(X, y)

print(f"The f-measure is {f_measure:0.2f}")
```

<!-- #region id="miyHKkVmR5tF" -->
The f-measure remains approximately the same. Basic centrality measures are insufficient. We need to expand `X` to include the social groups uncovered by Markov Clustering.  One approach is just consider the following binary question; are two people in the same social group? Yes or no? If they are, then perhaps they are more likely to eventually become friends on FriendHook. We can make this binary comparison between each pair of profile-ids within a single row of observations. 

**Adding social group features**
<!-- #endregion -->

```python id="Z5sHBEI7R5tF" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511926180, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ac73b828-4108-4aac-c1c4-2ab6dc25591b"
np.random.seed(0)
features = {f'Shared_Cluster_{e}': []
            for e in ['id_f', 'id_fof', 'f_fof']}

i = 0
for node_ids in df_obs[cols].values:
    c_id, c_f, c_fof = [G.nodes[n]['cluster_id'] 
                        for n in node_ids]
    features['Shared_Cluster_id_f'].append(int(c_id == c_f))
    features['Shared_Cluster_id_fof'].append(int(c_id == c_fof))
    features['Shared_Cluster_f_fof'].append(int(c_f == c_fof))
    
X = update_features(features)
f_measure, clf = evaluate(X, y)
print(f"The f-measure is {f_measure:0.2f}") 
```

<!-- #region id="Fbp2DmCJR5tG" -->
Our f-measure has noticeably improved, from 0.38 to 0.43. Perfomance is still poor. Nonetheless, the social group inclusion has led a slight enhacement of our model. How important are the new social group features relative the model's current perfomance? We can find check, using the `feature_importance_` attribute of our trained classifier.

**Ranking features by their importance score**
<!-- #endregion -->

```python id="8HzVCA25R5tG" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511927730, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ac0094ae-c96c-47e3-ddd5-b9be90dbbd5b"
def view_top_features(clf, feature_names):
    for i in np.argsort(clf.feature_importances_)[::-1]:
        feature_name = feature_names[i]
        importance = clf.feature_importances_[i]
        if not round(importance, 2):
            break
            
        print(f"{feature_name}: {importance:0.2f}")
feature_names = df_features.columns
view_top_features(clf, feature_names)
```

<!-- #region id="Tketm2ZTR5tH" -->
Social graph centrality plays some role in friendship determination. Of course, our model's performance is still poor, so we should be cautious with our inferences on how the features drive predictions.  What other graph-based features could we utilize? Perhaps the network cluster size can impact the predictions? We can find out. However, we should be cautious in our efforts to keep our model generalizable. Cluster size can inexplicably take the place of cluster-id, making the model very specific to the university. 

**Adding cluster-size features**
<!-- #endregion -->

```python id="23WZ_Zr5R5tH" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511937961, "user_tz": -330, "elapsed": 447, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="214acf3c-2834-4061-bf72-4673500cbd8c"
np.random.seed(0)
cluster_sizes = [len(cluster) for cluster in clusters]
features = {f'{col}_Cluster_Size': [] for col in cols}
for node_ids in df_obs[cols].values:
    for node, feature_name in zip(node_ids, features.keys()):
        c_id = G.nodes[node]['cluster_id']
        features[feature_name].append(cluster_sizes[c_id])
    
X = update_features(features)
f_measure, clf = evaluate(X, y)
print(f"The f-measure is {f_measure:0.2f}")
```

<!-- #region id="_8adpXgZR5tH" -->
The cluster did not improve the model. As a precaution, lets delete it from our feature set.

**Deleting cluster-size features**
<!-- #endregion -->

```python id="iKZyxJjIR5tI"
import re
def delete_features(df_features, regex=r'Cluster_Size'):
    
    df_features.drop(columns=[name for name in df_features.columns
                               if re.search(regex, name)], inplace=True)
    return df_features.values

X = delete_features(df_features)
```

<!-- #region id="SuB6EnwoR5tI" -->
The f-measure remains at 0.43. Perhaps we should try thinking outside the box. In what ways can social connections drive real world-behavior? Consider the following scenario. Suppose we analyze a student named Alex, whose node id in network `G` is `n`. Alex has 50 FriendHook friends. These are accessible through `G[n]`. We randomly sample two of the friends in `G[n]`. Their node ids are `a` and `b`. We then check if `a` and `b` are friends. They are! It seems that `a` is in `list(G[n])`. We then repeat this 100 times. In 95% percent of sampled instances, `a` is friend of `b`. Basically, there's a 95% likelihood that any pair of Alex's friends are also friends with each other. We'll refer to this probability as the **friend-sharing likelihood**.  Lets try incorporating this likelihood into our features. We'll start by computing the likehood for every node in `G`. 

**Computing friend-sharing likelihoods**
<!-- #endregion -->

```python id="6sddUPr3R5tI"
friend_sharing_likelihood = {}
for node in nodes:
    neighbors = list(G[node])
    friendship_count = 0
    total_possible = 0
    for i, node1 in enumerate(neighbors[:-1]):
        for node2 in neighbors[i + 1:]:
            if node1 in G[node2]:
                friendship_count += 1
                
            total_possible += 1
    
    prob = friendship_count / total_possible if total_possible else 0
    friend_sharing_likelihood[node] = prob
```

<!-- #region id="_pf4DSThR5tJ" -->
Next, we'll generate a friend-sharing likelihood feature for each of our three profile ids. After adding the features, we'll re-valuate the trained model's performance.

**Adding friend-sharing likelihood features**
<!-- #endregion -->

```python id="E0CaSuAkR5tJ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511959812, "user_tz": -330, "elapsed": 31, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e96b9192-271f-4fb2-a0bf-4305bfcaffd1"
np.random.seed(0)
features = {f'{col}_Friend_Sharing_Likelihood': [] for col in cols}
for node_ids in df_obs[cols].values:
    for node, feature_name in zip(node_ids, features.keys()):
        sharing_likelihood = friend_sharing_likelihood[node]
        features[feature_name].append(sharing_likelihood)

X = update_features(features)
f_measure, clf = evaluate(X, y)
print(f"The f-measure is {f_measure:0.2f}")
```

<!-- #region id="lzSCCbP-R5tK" -->
Performance has increased from 0.43 to 0.49! It's still not great, but its progressively getting better. How does the friend-sharing likelihood compare to other features in the model? Lets find out!

**Ranking features by their importance score**
<!-- #endregion -->

```python id="xQ0MxIKWR5tK" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511959813, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="93ceb01c-1fbe-414b-9e5f-274a0f13f7ef"
feature_names = df_features.columns
view_top_features(clf, feature_names)
```

<!-- #region id="IYdi9vJ7R5tK" -->
One of our new features ranks quite highly.Nonetheless, the model is incomplete. An f-measure of 0.49 is not acceptable. We need to incorporate features from the profiles stored within `df_profiles`.

## Adding Profile Features to the Model


Our aim is to incorporate the profile attributes of _Sex_, _Relationship_Status_, _Major_, _Dorm_ and _Year_ into our feature matrix. Based on our experience with the network data, there are three ways in which we can do this:
1. Exact Value Extraction.
    * We can store the exact value of the profile feature associated with each of the three profile-id columns in `df_obs`.
   
2. Equivalence Comparison
    * Given a profile attribute, we can carry-out a pairwise comparison of the attribute across all three profile-id columns in `df_obs`. For each comparison we would return a boolean feature demarcating whether the attribute is equal within the two columns.
   
3. Size:
    * Given a profile attribute, we can return the number of profiles that share that attribute.
  
    
Let's apply Exact Value Extraction to _Sex_, _Relationship_Status_ and _Year_. 

**Adding exact-value profile features**
<!-- #endregion -->

```python id="7Z1N3ya7R5tL" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511961540, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="34098105-1dde-4f33-ac2a-9330be9558cb"
attributes = ['Sex', 'Relationship_Status', 'Year']
for attribute in attributes:
    features = {f'{col}_{attribute}_Value': [] for col in cols}
    for node_ids in df_obs[cols].values:
        for node, feature_name in zip(node_ids, features.keys()):
            att_value = G.nodes[node][attribute]
            features[feature_name].append(att_value)
    
    X = update_features(features)
    
f_measure, clf = evaluate(X, y)
print(f"The f-measure is {f_measure:0.2f}") 
```

<!-- #region id="1DyXSFThR5tL" -->
Wow! The f-measure dramatically increased from 0.49 to 0.74! The profile features have provided a very valuable signal, but we can still do better. We need to incorporate information from the _Major_ and _Dorm_ attributes. Equivalence Comparison is an excellent way to do this.

**Adding equivalence-comparison profile features**
<!-- #endregion -->

```python id="P33DntlfR5tL" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511962707, "user_tz": -330, "elapsed": 700, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6cd41d91-8838-4457-9d67-332d44420ba0"
attributes = ['Major', 'Dorm']
for attribute in attributes:
    features = {f'Shared_{attribute}_{e}': []
            for e in ['id_f', 'id_fof', 'f_fof']}
    for node_ids in df_obs[cols].values:
        att_id, att_f, att_fof = [G.nodes[n][attribute] 
                                  for n in node_ids]
        features[f'Shared_{attribute}_id_f'].append(int(att_id == att_f))
        features[f'Shared_{attribute}_id_fof'].append(int(att_id == att_fof))
        features[f'Shared_{attribute}_f_fof'].append(int(att_f == att_fof))
        
    X = update_features(features)
        
f_measure, clf = evaluate(X, y)
print(f"The f-measure is {f_measure:0.2f}") 
```

<!-- #region id="gg7jh64TR5tM" -->
Incorporating the _Major_ and _Dorm_ attributes has improved model performance. Now lets consider adding _Major_ and _Dorm Size_ into the mix. We can count the number of students associated with each major / dorm, and include this count as one of our features. However, we need to be careful. As we previously discussed, our trained model can cheat by utilizing size as a substitute for a category id. 

**Adding size-related profile features**
<!-- #endregion -->

```python id="xeuLHwSPR5tM" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511963980, "user_tz": -330, "elapsed": 585, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="10d8f7a3-7386-4843-a0a5-0215e7c5a8fe"
from collections import Counter

for attribute in ['Major', 'Dorm']:
    counter = Counter(df_profile[attribute].values)
    att_to_size = {k: v 
                      for k, v in counter.items()}
    features = {f'{col}_{attribute}_Size': [] for col in cols}
    for node_ids in df_obs[cols].values:
        for node, feature_name in zip(node_ids, features.keys()):
            size = att_to_size[G.nodes[node][attribute]]
            features[feature_name].append(size)
    
    
    X = update_features(features)
    
f_measure, clf = evaluate(X, y)
print(f"The f-measure is {f_measure:0.2f}") 
```

<!-- #region id="HbdlsHoxR5tM" -->
Performance has increased from 0.82 to 0.85. The introduction of size has impacted our model. Lets dive deeper into that impact. We'll start by printing out the feature importance scores.

**Ranking features by their importance score**
<!-- #endregion -->

```python id="r-eKmdWtR5tM" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511963981, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3572c480-c5b2-4374-c0b4-04fc9387102b"
feature_names = df_features.columns.values
view_top_features(clf, feature_names)
```

<!-- #region id="mLJ29m_nR5tM" -->
The feature importance scores are dominated by two features; _FoF_Dorm_Size_ and _Shared_Cluster_id_fof_. The presence of _FoF_Dorm_Size_ is a bit concerning. As we've discussed, a single dorm dominates over 50% of the network data. Is our model simply memorizing that dorm based on its size? We can find out by actually visualizing a trained decision tree. 

**Displaying the top branches of the tree**
<!-- #endregion -->

```python id="45d4i0-qR5tN" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511964952, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8a6c0b62-cb32-4306-bf84-4423b6d1f9ec"
from sklearn import tree

clf_depth2 = DecisionTreeClassifier(max_depth=2)
clf_depth2.fit(X, y)
text_tree = tree.export_text(clf_depth2, feature_names=list(feature_names))
print(text_tree)
```

<!-- #region id="Pt_XS2LFR5tN" -->
According to the the tree, the most important signal is whether _FoF_Dorm_Size_ is less than 279. This begs the question, how many dorms contain at-least 279 students? 

**Checking dorms with at-least 279 students**
<!-- #endregion -->

```python id="IJYLfO6vR5tN" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511967128, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="92494656-88b6-4060-c6c9-f27e110a2b2d"
counter = Counter(df_profile.Dorm.values)
for dorm, count in counter.items():
    if count < 279:
        continue
        
    print(f"Dorm {dorm} holds {count} students.")
```

<!-- #region id="9gWCZH6sR5tN" -->
Just two of the 15 dorms contain more than 279 FriendHook-registered students. Essentially, our model relies on the two most-populous dorms to make its decisions. This might not generalize to other college campuses. For instance, consider a campus whose dormitories are smaller, and hold 200 students at the most. The model will completely fail to predict user-behavior in this instance. 

Perhaps we could try deleting the size-related features while also adjusting our choice of classifier. There is a slight chance that we'll achieve comparable performance without relying on dorm size. This is unlikely but is still worth trying. 

**Deleting all size-related features**
<!-- #endregion -->

```python id="zwO2A2hZR5tQ"
X_with_sizes = X.copy()
X = delete_features(df_features, regex=r'_Size')
```

<!-- #region id="eB27CUnCR5tQ" -->
## Optimizing Performance Across a Steady Set of Features

Will switching the model-type from a Decision tree to a Forest improve performance outcome? Lets find out.

**Training and evaluating a Random Forest classifier**
<!-- #endregion -->

```python id="cw7-0shbR5tQ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511976583, "user_tz": -330, "elapsed": 1449, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4e558a1c-f0d0-4fbe-fa91-4b5fd8212cf8"
np.random.seed(0)
from sklearn.ensemble import RandomForestClassifier
f_measure, clf = evaluate(X, y, model_type=RandomForestClassifier)
print(f"The f-measure is {f_measure:0.2f}") 
```

<!-- #region id="8AlbUwndR5tR" -->

Switching the type of model has not helped. Perhaps instead we can boost performance by optimizing on the hyperparameters? Within this book, we've focused on single Decision tree hyperparameter; max depth. Will limiting the depth improve our predictions? Lets quickly check using a simple grid search. 

**Optimizing max depth using grid search**
<!-- #endregion -->

```python id="7G7rA7LFR5tR" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511980785, "user_tz": -330, "elapsed": 4208, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d20f5aa8-8bcf-4412-89af-1b30a2f6c2fc"
from sklearn.model_selection import GridSearchCV
np.random.seed(0)

hyperparams = {'max_depth': list(range(1, 100)) + [None]}
clf_grid = GridSearchCV(DecisionTreeClassifier(), hyperparams, 
                        scoring='f1_macro', cv=2)
clf_grid.fit(X, y)
best_f = clf_grid.best_score_
best_depth = clf_grid.best_params_['max_depth']
print(f"A maximized f-measure of {best_f:.2f} is achieved when "
      f"max_depth equals {best_depth}") 
```

<!-- #region id="GqNB0WiMR5tR" -->
Setting `max_depth` to 5 improves the f-measure from from 0.82 to 0.84. This level of performance is comperable with our Dorm-size dependent model. Of course, we cannot make a fair comparison without first running a grid search on the size-inclusive `X_with_size` feature-matrix. Will optimizing on `X_with_size` yield an even better classifier? Lets find out.

**Applying grid search to size-dependent training data**
<!-- #endregion -->

```python id="ZE-wPb7NR5tR" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511985998, "user_tz": -330, "elapsed": 5217, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0b52b3de-fedf-49c6-a60b-7f2409643d4d"
np.random.seed(0)
clf_grid.fit(X_with_sizes, y)
best_f = clf_grid.best_score_
best_depth = clf_grid.best_params_['max_depth']
print(f"A maximized f-measure of {best_f:.2f} is achieved when "
      f"max_depth equals {best_depth}") 
```

<!-- #region id="JxspNBOWR5tS" -->
The grid search did not improve performance on `X_with_size`. Thus, we can conclude that with the right choice of max-depth, both the size-dependent and independent models perform with approximately equal quality. Consequently, we can train a generalizable, size-independent model without sacrificing perfomance. 

**Training a Decision tree with `max_depth` set to 5**
<!-- #endregion -->

```python id="ZWDYceBcR5tS" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511986000, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="773c8799-e098-4b3c-f027-676d05853521"
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X, y)
```

<!-- #region id="w1MMvAJBR5tS" -->
## Interpreting the Trained Model

Lets print our model's feature importance scores.

**Ranking features by their importance score**
<!-- #endregion -->

```python id="zf2YWB5HR5tS" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511991446, "user_tz": -330, "elapsed": 400, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9a942d24-ed85-4650-85f6-0d82c54cfb6c"
feature_names = df_features.columns
view_top_features(clf, feature_names)
```

<!-- #region id="IQ2x600eR5tT" -->
Only nine important features remain. Only three features have an importance score that's at or above 0.10. These are `Shared_Dorm_id_fof`, `Shared_Cluster_id_fof`, and `Shared_Major_id_fof`. Thus, the model is primarily driven by the following three questions:

1. Do the the user and the friend-of-friend share a dormitory? Yes or no?
2. Do the the user and the friend-of-friend share a social group? Yes or no?
3. Do the the user and the friend-of-friend share a major? Yes or no?

Intuitively, if the answers to all three questions are _Yes_, then the user and the friend-of-a-friend are more likely to connect on FriendHook. Lets test this intutition, by displaying the tree. 

**Displaying the top branches of the tree**
<!-- #endregion -->

```python id="EsbpNSHhR5tT" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511991984, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2d67aaf9-26e9-4319-c7d9-5f5bedee9790"
clf_depth3 = DecisionTreeClassifier(max_depth=3)
clf_depth3.fit(X, y)
text_tree = tree.export_text(clf_depth3, 
                             feature_names=list(feature_names))
print(text_tree)
```

<!-- #region id="xaLpaqqGR5tT" -->

The model's logic is very straightforward. Users who share social groups and living spaces or study-schedules are more likely to connect. There's nothing suprising about that. What is suprising is how the _Sex_ feature drives _Class 2_ label prediction. According to our tree, rejection is more likeley when:

1. The users share a dorm but are not in the same social group.
2. The request sender is of a certain specific sex.

Of course, we know that _Class 2_ labels are fairly sparse within our data. Perhaps the model's predictions are caused by random noise arising from the sparse sampling? Lets quickly check how well we predict rejection. 

**Evaluating a rejection classifier**
<!-- #endregion -->

```python id="UZdrRIOOR5tT" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511993082, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="27da7635-14a0-4bad-a6d9-ad50646372b0"
y_reject = y *(y == 2)
f_measure, clf_reject = evaluate(X, y_reject, max_depth=5)
print(f"The f-measure is {f_measure:0.2f}") 
```

<!-- #region id="UuGMqc8CR5tU" -->
Wow, the f-measure is actually very high! We can predict rejection very well, despite the sparsity of data. What features drive rejection? Lets check by printing the new feature importance scores.

**Ranking features by their importance score**
<!-- #endregion -->

```python id="yjMg-212R5tU" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511994583, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f086a958-3cff-4fd8-954d-b9b69a30b307"
view_top_features(clf_reject, feature_names)
```

<!-- #region id="Vvi2S_xRR5tU" -->
Interesting! Rejection is primarily driven by the user's _Sex_ and _Relationship_Status_ attributes. Lets visualize the trained tree to learn more.

**Displaying the rejection-predicting tree**
<!-- #endregion -->

```python id="iY3lwtWWR5tU" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637511995784, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="84153145-1545-4ddc-9f00-335d216675ca"
text_tree = tree.export_text(clf_reject, 
                             feature_names=list(feature_names))
print(text_tree)
```

<!-- #region id="c844Hq5RR5tV" -->
According to the tree, individuals with _Sex Category 1_ and _Relationship Status Category 3_ are sending friend-requests to people outside their social group. These friend requests are likely to get rejected. Of course, we don't know the identify of the categories that lead to rejection. However, as scientists, we still speculate. Given what we know about human nature, it wouldn't be suprising this behavior is driven by single men. Perhaps they are trying to connect with women outside their social circle, in order to get a date? If, so then the request will probably be rejected. Again, all this is speculation. However, this hypothesis is worth discussing with the product managers at FriendHook. If our hypothesis is correct, then certain changes should be introduced to the product. More steps could be taken to limit unwanted dating requests. Alternatively, new product changes could be added that make easier for single people to connect.
<!-- #endregion -->

<!-- #region id="lYhpcF49VW8Q" -->
In this case study, we have agonized over keeping our model generalizable. A model that does not generalize beyond the training set is worthless, even if the performance score seems high. Unfortunately, it’s hard to know whether a model can generalize until it’s tested on external data. But we can try to remain aware of hidden biases that won’t generalize well to other datasets. Failure to do so can yield serious consequences. Consider the following true story.

For many years, machine learning researchers have tried to automate the field of radiology. In radiology, trained doctors examine medical images (such as X-rays) to diagnose disease. This can be treated as a supervised learning problem in which the images are features and the diagnoses are class labels. By the year 2016, multiple radiology models were published in the scientific literature. Each published model was supposed to be highly accurate based on internal evaluation. That year, leading machine learning researchers publicly declared that “we should stop training radiologists” and that “radiologists should be worried about their jobs.” Four years later, the negative publicity had led to a worldwide radiologist shortage—medical students were reluctant to enter a field that seemed destined for full automation. But by 2020, the promise of automation had failed to materialize. Most of the published models performed very poorly on new data. Why? Well, it turns out that imaging outputs differ from hospital to hospital. Different hospitals use slightly different lighting and different settings on their imaging machines. Thus, a model trained at Hospital A could not generalize well to Hospital B. Despite their seemingly high performance scores, the models were not fit for generalized use. The machine learning researchers had been too optimistic; they failed to take into account the biases inherent in their data. These failures inadvertently led to a crisis in the medical community. A more thoughtful evaluation of generalizability could have prevented this from happening.
<!-- #endregion -->

<!-- #region id="Y7fRvG64F0YF" -->
## Summary

- Superior machine learning algorithms do not necessarily work in every situation. Our decision tree model outperformed our random forest model, even though random forests are considered superior in the literature. We should never blindly assume that a model will always work well in every possible scenario. Instead, we should intelligently calibrate our model choice based on the specifics of the problem.
- Proper feature selection is less of a science and more of an art. We cannot always know in advance which features will boost a model’s performance. However, the commonsense integration of diverse and interesting features into our model should eventually improve prediction quality.
- We should pay careful attention to the features we feed into our model. Otherwise, the model may not generalize to other datasets.
- Proper hyperparameter optimization can sometimes significantly boost a model’s performance.
- Occasionally, it seems like nothing is working and our data is simply insufficient. However, with grit and perseverance, we can eventually yield meaningful resources. Remember, a good data scientist should never give up until they have exhausted every possible avenue of analysis.
<!-- #endregion -->

<!-- #region id="a9khCVuLUnzd" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Mq6Q1HzjUnzd" executionInfo={"status": "ok", "timestamp": 1637512032091, "user_tz": -330, "elapsed": 3739, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="579d9d3d-a4d7-498f-c8b6-280a436d464f"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="SLUgXbkjUnze" -->
---
<!-- #endregion -->
