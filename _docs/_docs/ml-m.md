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

<!-- #region id="WlLz323iuT4n" -->
# Random, Item Popularity, and Global Effects Recommender Models on ML-10m Dataset
<!-- #endregion -->

<!-- #region id="CxiWmRiFzT2X" -->
## Setup
<!-- #endregion -->

<!-- #region id="fQ64dOFO0fJe" -->
### Git
<!-- #endregion -->

```python id="BaXpEKLdXNPa"
import os
project_name = "general-recsys"; branch = "T434220"; account = "sparsh-ai"
project_path = os.path.join('/content', branch)

if not os.path.exists(project_path):
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    !mkdir "{project_path}"
    %cd "{project_path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout -b "{branch}"
else:
    %cd "{project_path}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="MZvPHRyMXdlS" executionInfo={"status": "ok", "timestamp": 1636807859469, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f0df3d88-9317-4192-db52-a406a836f35a"
%cd /content
```

```python id="2eRcpGL6XfDs"
!cd /content/main && git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="BXJY8c9d4Xi5" -->
### Installations
<!-- #endregion -->

<!-- #region id="GB_yDppW3_Yt" -->
### Imports
<!-- #endregion -->

```python id="vrEmNkAAsQlM"
from tqdm.notebook import tqdm
import sys
import os
import logging
from os import path as osp
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sps

import bz2
import pickle
import _pickle as cPickle

import matplotlib.pyplot as plt
```

<!-- #region id="NyxCtlrJ3_Ta" -->
### Params
<!-- #endregion -->

```python id="MXBwnUCD3_RD"
class Args:
    datapath_bronze = '/content'
    datapath_silver = '/content'

    URM_train = None
    URM_test = None

    top_k = 5

args = Args()
```

<!-- #region id="Q40X4lHf4JHw" -->
### Logger
<!-- #endregion -->

```python id="cibwpV5L4JFb"
logging.basicConfig(stream=sys.stdout,
                    level = logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] : %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger('Logger')
```

<!-- #region id="2M0-cN2ZzWE-" -->
## Modules
<!-- #endregion -->

<!-- #region id="qY9Y0q2sz1MS" -->
### Utils
<!-- #endregion -->

```python id="n0KPXS9jT9lx"
def save_pickle(data, path):
 with bz2.BZ2File(path + '.pbz2', 'w') as f: 
    cPickle.dump(data, f)

def load_pickle(path):
    data = bz2.BZ2File(path+'.pbz2', 'rb')
    data = cPickle.load(data)
    return data
```

<!-- #region id="X2u3zmqMXJhs" -->
### Dataset
<!-- #endregion -->

```python id="AGTVUdmtwWgZ"
def download_dataset():
    # If file exists, skip the download
    data_file_name = args.datapath_bronze + "movielens_10m.zip"

    # If directory does not exist, create
    if not os.path.exists(args.datapath_bronze):
        os.makedirs(args.datapath_bronze)

    if not os.path.exists(data_file_name):
        url_path = 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'
        logger.info('Download ml-10m.zip file from {}'.format(url_path))
        urlretrieve(url_path, data_file_name)

    dataFile = zipfile.ZipFile(args.datapath_bronze + "movielens_10m.zip")
    args.ratings_path = dataFile.extract("ml-10M100K/ratings.dat", path=args.datapath_bronze)
    args.tags_path = dataFile.extract("ml-10M100K/tags.dat", path=args.datapath_bronze)
    logger.info('Dataset downloaded in {}'.format(osp.join(args.datapath_bronze,'ml-10M100K')))
```

```python id="vk93jRMwtEWP"
def preprocess_dataset():
    URM_all_dataframe = pd.read_csv(filepath_or_buffer=args.ratings_path, 
                                    sep="::", 
                                    header=None, 
                                    dtype={0:int, 1:int, 2:float, 3:int},
                                    engine='python')

    URM_all_dataframe.columns = ["UserID", "ItemID", "Interaction", "Timestamp"]
    print(URM_all_dataframe.head())

    print("The number of interactions is {}".format(len(URM_all_dataframe)))
    print("We can use this data to create a sparse matrix, notice that we have red UserID and ItemID as int")
    print("This is not always possible if the IDs are alphanumeric")
    print("Now we can extract the list of unique user id and item id and display some statistics")

    userID_unique = URM_all_dataframe["UserID"].unique()
    itemID_unique = URM_all_dataframe["ItemID"].unique()

    n_users = len(userID_unique)
    n_items = len(itemID_unique)
    n_interactions = len(URM_all_dataframe)

    print("Number of items\t {}, Number of users\t {}".format(n_items, n_users))
    print("Max ID items\t {}, Max Id users\t {}\n".format(max(itemID_unique), max(userID_unique)))

    print("See that the max ID of items and users is higher than the number of unique values -> empty profiles")
    print("We should remove empty indices, to do so we create a new mapping")

    user_original_ID_to_index_dict = {}

    for user_id in userID_unique:
        user_original_ID_to_index_dict[user_id] = len(user_original_ID_to_index_dict)
    item_original_ID_to_index_dict = {}

    for item_id in itemID_unique:
        item_original_ID_to_index_dict[item_id] = len(item_original_ID_to_index_dict)
    original_item_ID = 292

    print("New index for item {} is {}".format(original_item_ID, item_original_ID_to_index_dict[original_item_ID]))

    print("We now replace the IDs in the dataframe and we are ready to use the data")

    URM_all_dataframe["UserID"] = [user_original_ID_to_index_dict[user_original] for user_original in
                                        URM_all_dataframe["UserID"].values]

    URM_all_dataframe["ItemID"] = [item_original_ID_to_index_dict[item_original] for item_original in 
                                        URM_all_dataframe["ItemID"].values]
    print(URM_all_dataframe.head())

    userID_unique = URM_all_dataframe["UserID"].unique()
    itemID_unique = URM_all_dataframe["ItemID"].unique()

    n_users = len(userID_unique)
    n_items = len(itemID_unique)
    n_interactions = len(URM_all_dataframe)

    print("Number of items\t {}, Number of users\t {}".format(n_items, n_users))
    print("Max ID items\t {}, Max Id users\t {}\n".format(max(itemID_unique), max(userID_unique)))
    print("Average interactions per user {:.2f}".format(n_interactions/n_users))
    print("Average interactions per item {:.2f}\n".format(n_interactions/n_items))

    print("Sparsity {:.2f} %".format((1-float(n_interactions)/(n_items*n_users))*100))

    print("To store the data we use a sparse matrix. We build it as a COO matrix and then change its format")
    print("The COO constructor expects (data, (row, column))")

    URM_all = sps.coo_matrix((URM_all_dataframe["Interaction"].values, 
                            (URM_all_dataframe["UserID"].values, URM_all_dataframe["ItemID"].values)))

    print(URM_all)

    URM_all.tocsr()

    print("We compute the item popularity as the number of interaction in each column")
    print("We can use the properties of sparse matrices in CSC format")

    item_popularity = np.ediff1d(URM_all.tocsc().indptr)
    item_popularity = np.sort(item_popularity)

    ten_percent = int(n_items/10)

    print("Average per-item interactions over the whole dataset {:.2f}".
        format(item_popularity.mean()))

    print("Average per-item interactions for the top 10% popular items {:.2f}".
        format(item_popularity[-ten_percent:].mean()))

    print("Average per-item interactions for the least 10% popular items {:.2f}".
        format(item_popularity[:ten_percent].mean()))

    print("Average per-item interactions for the median 10% popular items {:.2f}".
        format(item_popularity[int(n_items*0.45):int(n_items*0.55)].mean()))

    print("Number of items with zero interactions {}".
        format(np.sum(item_popularity==0)))

    print("We compute the user activity (profile length) as the number of interaction in each row")
    print("We can use the properties of sparse matrices in CSR format")

    user_activity = np.ediff1d(URM_all.tocsr().indptr)
    user_activity = np.sort(user_activity)

    print("The splitting of the data is very important to ensure your algorithm is evaluated in a realistic scenario by using test it has never seen")

    train_test_split = 0.80
    n_interactions = URM_all.nnz
    train_mask = np.random.choice([True,False], n_interactions, p=[train_test_split, 1-train_test_split])
    URM_train = sps.csr_matrix((URM_all.data[train_mask],
                                (URM_all.row[train_mask], URM_all.col[train_mask])))
    test_mask = np.logical_not(train_mask)
    URM_test = sps.csr_matrix((URM_all.data[test_mask],
                                (URM_all.row[test_mask], URM_all.col[test_mask])))

    save_pickle(URM_train,osp.join(args.datapath_silver,'URM_train'))
    save_pickle(URM_test,osp.join(args.datapath_silver,'URM_test'))


    ICM_dataframe = pd.read_csv(filepath_or_buffer=args.tags_path, 
                                sep="::", 
                                header=None, 
                                dtype={0:int, 1:int, 2:str, 3:int},
                                engine='python')

    ICM_dataframe.columns = ["UserID", "ItemID", "FeatureID", "Timestamp"]
    print(ICM_dataframe.head())

    print("We can see that most users and items have no data associated to them")

    n_features = len(ICM_dataframe["FeatureID"].unique())
    print ("Number of tags\t {}, Number of item-tag tuples {}".format(n_features, len(ICM_dataframe)))

    print("We now build the sparse ICM matrices")
    print("The tags are strings, we should traslate them into numbers so we can use them as indices in the ICM.")
    print("We should also ensure that the item and user indices we use in ICM and URM are consistent. To do so we use the same mapper, first we populate it with the URM and then we add the new ids that appear only in the ICM")

    user_original_ID_to_index_dict = {}

    for user_id in URM_all_dataframe["UserID"].unique():
        user_original_ID_to_index_dict[user_id] = len(user_original_ID_to_index_dict)  

    print("Unique user_id in the URM are {}".format(len(user_original_ID_to_index_dict)))
        
    for user_id in ICM_dataframe["UserID"].unique():
        if user_id not in user_original_ID_to_index_dict:
            user_original_ID_to_index_dict[user_id] = len(user_original_ID_to_index_dict)
            
    print("Unique user_id in the URM and ICM are {}".format(len(user_original_ID_to_index_dict)))

    item_original_ID_to_index_dict = {}

    for item_id in URM_all_dataframe["ItemID"].unique():
        item_original_ID_to_index_dict[item_id] = len(item_original_ID_to_index_dict)

    print("Unique item_id in the URM are {}".format(len(item_original_ID_to_index_dict)))
        
    for item_id in ICM_dataframe["ItemID"].unique():
        if item_id not in item_original_ID_to_index_dict:
            item_original_ID_to_index_dict[item_id] = len(item_original_ID_to_index_dict)
            
    print("Unique item_id in the URM and ICM are {}".format(len(item_original_ID_to_index_dict)))

    feature_original_ID_to_index_dict = {}

    for feature_id in ICM_dataframe["FeatureID"].unique():
        feature_original_ID_to_index_dict[feature_id] = len(feature_original_ID_to_index_dict)

    print("Unique feature_id in the URM are {}".format(len(feature_original_ID_to_index_dict)))

    original_feature_ID = "star wars"
    print("New index for feature '{}' is {}".format(original_feature_ID, feature_original_ID_to_index_dict[original_feature_ID]))

    print("We can now build the ICM using the new indices")

    URM_all_dataframe["UserID"] = [user_original_ID_to_index_dict[user_original] for user_original in
                                        URM_all_dataframe["UserID"].values]

    URM_all_dataframe["ItemID"] = [item_original_ID_to_index_dict[item_original] for item_original in 
                                        URM_all_dataframe["ItemID"].values]

    ICM_dataframe["UserID"] = [user_original_ID_to_index_dict[user_original] for user_original in
                                        ICM_dataframe["UserID"].values]

    ICM_dataframe["ItemID"] = [item_original_ID_to_index_dict[item_original] for item_original in 
                                        ICM_dataframe["ItemID"].values]

    ICM_dataframe["FeatureID"] = [feature_original_ID_to_index_dict[feature_original] for feature_original in 
                                        ICM_dataframe["FeatureID"].values]

    ICM_all = sps.csr_matrix((np.ones(len(ICM_dataframe["ItemID"].values)), 
                            (ICM_dataframe["ItemID"].values, ICM_dataframe["FeatureID"].values)),
                            shape = (n_items, n_features))

    ICM_all.data = np.ones_like(ICM_all.data)

    ICM_all = sps.csr_matrix(ICM_all)
    features_per_item = np.ediff1d(ICM_all.indptr)

    ICM_all = sps.csc_matrix(ICM_all)
    items_per_feature = np.ediff1d(ICM_all.indptr)

    ICM_all = sps.csr_matrix(ICM_all)

    print(features_per_item.shape)
    print(items_per_feature.shape)

    features_per_item = np.sort(features_per_item)
    items_per_feature = np.sort(items_per_feature)

    save_pickle(ICM_all,osp.join(args.datapath_silver,'ICM_all'))
```

```python id="cJ4YtoWVXw5Z"
def load_processed_datasets():
    args.URM_train = load_pickle(osp.join(args.datapath_silver,'URM_train'))
    args.URM_test = load_pickle(osp.join(args.datapath_silver,'URM_test'))
    args.ICM_all = load_pickle(osp.join(args.datapath_silver,'ICM_all'))
```

<!-- #region id="_F4vRpFCzYsf" -->
### Metrics
<!-- #endregion -->

```python id="DB4nqBeKXNE9"
def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    return precision_score
```

```python id="UU4AFIvJXN3p"
def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    return recall_score
```

```python id="43FaRa8-XPeU"
def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return map_score
```

<!-- #region id="Sh_hYaMIbTav" -->
### Evaluation
<!-- #endregion -->

```python id="yFOx8syNbTYc"
def evaluate_algorithm(recommender_object):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    num_eval = 0

    for user_id in range(args.URM_test.shape[0]):
        relevant_items = args.URM_test.indices[args.URM_test.indptr[user_id]:args.URM_test.indptr[user_id+1]]
        
        if len(relevant_items)>0:
            recommended_items = recommender_object.recommend(user_id, at=args.top_k)
            num_eval+=1
            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_MAP += MAP(recommended_items, relevant_items)
            
    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval
    
    print("Recommender results are: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP)) 
```

<!-- #region id="3DYlyMq1XWiL" -->
### Models
<!-- #endregion -->

```python id="p3whP2SwXWfx"
class RandomRecommender(object):
    """In a random recommend we don't have anything to learn from the data"""
    def fit(self, URM_train):
        self.n_items = URM_train.shape[1]
    
    def recommend(self, user_id, at=5):
        recommended_items = np.random.choice(self.n_items, at)
        return recommended_items
```

```python id="zvzmJlT6b83p"
class TopPopRecommender(object):
    """We recommend to all users the most popular items, 
    that is those with the highest number of interactions
    In this case our model is the item popularity
    """
    def fit(self, URM_train):
        self.URM_train = URM_train
        item_popularity = np.ediff1d(URM_train.tocsc().indptr)
        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popular_items = np.argsort(item_popularity)
        self.popular_items = np.flip(self.popular_items, axis = 0)
    
    def recommend(self, user_id, at=5):
        if args.remove_seen:
            """remove items already seen by the user. We can either remove them from the
            recommended item list or we can set them to a score so low that it will cause 
            them to end at the very bottom of all the available items"""
            seen_items = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id+1]]
            unseen_items_mask = np.in1d(self.popular_items, seen_items,
                                        assume_unique=True, invert = True)
            unseen_items = self.popular_items[unseen_items_mask]
            recommended_items = unseen_items[0:at]
        else:
            recommended_items = self.popular_items[0:at]
        return recommended_items
```

```python id="_X1m6ArKj4Ou"
class GlobalEffectsRecommender(object):
    """We recommend to all users the highest rated items.
    First we compute the average of all ratings, or global average
    We subtract the bias to all ratings
    Then we compute the average rating for each item, or itemBias
    We cannot use the mean function because it would include also the zero values,
    which we want to exclude since they mean "missing data"
    The mean should be computed only on existing ratings. Unfortunately this 
    requires to do several operations to change the data classes so the code 
    is quite long. Also remember to exclude items with no ratings to avoid a 
    division by zero.
    And the average rating for each user, or userBias
    Now we can sort the items by their itemBias and use the same recommendation 
    principle as in TopPop.
    """
    def fit(self, URM_train):
        self.URM_train = URM_train
        globalAverage = np.mean(URM_train.data)
        URM_train_unbiased = URM_train.copy()
        URM_train_unbiased.data -= globalAverage
        # User Bias
        user_mean_rating = URM_train_unbiased.mean(axis=1)
        user_mean_rating = np.array(user_mean_rating).squeeze()
        
        # In order to apply the user bias we have to change the rating value 
        # in the URM_train_unbiased inner data structures
        # If we were to write:
        # URM_train_unbiased[user_id].data -= user_mean_rating[user_id]
        # we would change the value of a new matrix with no effect on the original data structure
        for user_id in range(len(user_mean_rating)):
            start_position = URM_train_unbiased.indptr[user_id]
            end_position = URM_train_unbiased.indptr[user_id+1]
            URM_train_unbiased.data[start_position:end_position] -= user_mean_rating[user_id]
        # Item Bias
        item_mean_rating = URM_train_unbiased.mean(axis=0)
        item_mean_rating = np.array(item_mean_rating).squeeze()

        self.bestRatedItems = np.argsort(item_mean_rating)
        self.bestRatedItems = np.flip(self.bestRatedItems, axis = 0)

    def recommend(self, user_id, at=5, remove_seen=True):
        if remove_seen:
            seen_items = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id+1]]
            unseen_items_mask = np.in1d(self.bestRatedItems, seen_items,
                                        assume_unique=True, invert = True)
            unseen_items = self.bestRatedItems[unseen_items_mask]
            recommended_items = unseen_items[0:at]
        else:
            recommended_items = self.bestRatedItems[0:at]
        return recommended_items
```

```python id="2T4hsKM9Xnxa"
def train_model():
    if args.URM_train is None:
        load_processed_datasets()

    if args.model_name in ['Random','ItemPopularity','GlobalEffects']:
        model_dict = {
            'Random':RandomRecommender(),
            'ItemPopularity':TopPopRecommender(),
            'GlobalEffects':GlobalEffectsRecommender(),
        }
    
        model = model_dict[args.model_name]
        model.fit(args.URM_train)

        for user_id in range(10):
            print('As per {} model, User {} would prefer these {} items: {}'
            .format(args.model_name,
                    user_id,
                    args.top_k,
                    model.recommend(user_id, at=args.top_k)
            ))
        evaluate_algorithm(model)
```

<!-- #region id="bDpG9ILPXWd3" -->
## Jobs
<!-- #endregion -->

```python id="2y8mdDjds6dr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636807914124, "user_tz": -330, "elapsed": 2506, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d6d1a602-63d2-4b8d-ed8c-c028f8c9e1df"
logger.info('JOB START: DOWNLOAD_RAW_DATASET')
download_dataset()
logger.info('JOB END: DOWNLOAD_RAW_DATASET')
```

```python id="3ig3tPpB2Fx-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636808002136, "user_tz": -330, "elapsed": 81836, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3d1fe2b1-9690-4b19-fb56-81d4368d555e"
logger.info('JOB START: PREPROCESSING_MOVIELENS_10M')
preprocess_dataset()
logger.info('JOB END: PREPROCESSING_MOVIELENS_10M')
```

```python colab={"base_uri": "https://localhost:8080/"} id="IHUfmN7yWsWe" executionInfo={"status": "ok", "timestamp": 1636809680687, "user_tz": -330, "elapsed": 10431, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0e35945a-b097-45c5-e55e-1e387e869299"
logger.info('JOB START: TRAINING_RANDOM_RECOMMENDER_MODEL')
args.model_name = 'Random'
train_model()
logger.info('JOB END: TRAINING_RANDOM_RECOMMENDER_MODEL')
```

```python colab={"base_uri": "https://localhost:8080/"} id="OizdBFOLdYZb" executionInfo={"status": "ok", "timestamp": 1636809974793, "user_tz": -330, "elapsed": 9347, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="396bb042-9166-44c6-a1f9-ea7cf332589b"
logger.info('JOB START: ITEM_POP_RECOMMENDER_MODEL')
args.model_name = 'ItemPopularity'
args.remove_seen = False
train_model()
logger.info('JOB END: ITEM_POP_RECOMMENDER_MODEL')
```

```python colab={"base_uri": "https://localhost:8080/"} id="1i-hLBOTeUi2" executionInfo={"status": "ok", "timestamp": 1636810062749, "user_tz": -330, "elapsed": 55117, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bb01c9fb-983d-41fe-9cb4-7acc74853a9a"
logger.info('JOB START: ITEM_POP_WITH_FILTER_RECOMMENDER_MODEL')
args.model_name = 'ItemPopularity'
args.remove_seen = True
train_model()
logger.info('JOB END: ITEM_POP_WITH_FILTER_RECOMMENDER_MODEL')
```

```python colab={"base_uri": "https://localhost:8080/"} id="Taj8oNKcef2O" executionInfo={"status": "ok", "timestamp": 1636811920145, "user_tz": -330, "elapsed": 50742, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b1543dfd-b0f1-486c-dca9-9940ad18a4f8"
logger.info('JOB START: GLOBAL_EFFECTS_RECOMMENDER_MODEL')
args.model_name = 'GlobalEffects'
args.remove_seen = True
train_model()
logger.info('JOB END: GLOBAL_EFFECTS_RECOMMENDER_MODEL')
```
