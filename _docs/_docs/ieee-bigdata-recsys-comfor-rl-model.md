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

<!-- #region id="3GofkbEjwQCI" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="BaXpEKLdXNPa" executionInfo={"elapsed": 663, "status": "ok", "timestamp": 1636271220486, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="2aacf9ae-cc79-43f0-f3f9-41ae04c75e16"
import os
project_name = "ieee21cup-recsys"; branch = "main"; account = "sparsh-ai"
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

```python colab={"base_uri": "https://localhost:8080/"} id="MZvPHRyMXdlS" executionInfo={"elapsed": 967, "status": "ok", "timestamp": 1636271225175, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="9ace871e-5585-4062-e21a-70e4d38d537e"
%cd /content
```

```python id="2eRcpGL6XfDs"
!cd /content/main && git add . && git commit -m 'commit' && git push origin main
```

```python id="DctyNOSdx-7h"
!pip install -q wget
```

```python id="vrEmNkAAsQlM"
import io
import copy
import sys
import wget
import os
import logging
import pandas as pd
from os import path as osp
import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path

import multiprocessing as mp
import functools
from sklearn.preprocessing import MinMaxScaler

import bz2
import pickle
import _pickle as cPickle

import matplotlib.pyplot as plt
%matplotlib inline
```

```python id="M4swQxyAsQnj"
class Args:

    # Paths
    datapath_bronze = '/content/main/data/bronze'
    datapath_silver = '/content/main/data/silver/T445041'
    datapath_gold = '/content/main/data/gold/T445041'

    filename_trainset = 'train.csv'
    filename_iteminfo = 'item_info.csv'
    filename_track1_testset = 'track1_testset.csv'

    data_sep = ' '

    N_ITEMS = 380
    N_USER_PORTRAITS = 10
    N_THREADS = mp.cpu_count() - 1


args = Args()
```

```python id="wIDRSKqOtEdb"
logging.basicConfig(stream=sys.stdout,
                    level = logging.INFO,
                    format='%(asctime)s [%(levelname)s] : %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger('IEEE21 Logger')
```

<!-- #region id="N1bmqnvQv27E" -->
## Utilities
<!-- #endregion -->

```python id="tH7lmOJbAOIf"
def save_pickle(data, title):
 with bz2.BZ2File(title, 'w') as f: 
    cPickle.dump(data, f)

def load_pickle(path):
    data = bz2.BZ2File(path, 'rb')
    data = cPickle.load(data)
    return data
```

```python id="AGTVUdmtwWgZ"
def download_dataset():
    # create bronze folder if not exist
    Path(args.datapath_bronze).mkdir(parents=True, exist_ok=True)
    # also creating silver and gold folder for later use
    Path(args.datapath_silver).mkdir(parents=True, exist_ok=True)
    Path(args.datapath_gold).mkdir(parents=True, exist_ok=True)
    # for each of the file, download if not exist
    datasets = ['train.parquet.snappy', 'item_info.parquet.snappy',
                'track1_testset.parquet.snappy', 'track2_testset.parquet.snappy']
    for filename in datasets:
        file_savepath = osp.join(args.datapath_bronze,filename)
        if not osp.exists(file_savepath):
            logger.info('Downloading {}'.format(filename))
            wget.download(url='https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/{}'.format(filename),
                          out=file_savepath)
        else:
            logger.info('{} file already exists, skipping!'.format(filename))
```

```python id="vk93jRMwtEWP"
def parquet_to_csv(path):
    savepath = osp.join(str(Path(path).parent),str(Path(path).name).split('.')[0]+'.csv')
    pd.read_parquet(path).to_csv(savepath, index=False, sep=args.data_sep)
```

```python id="_F4vRpFCzYsf"
def convert_dataset():
    # for each of the file, convert into csv, if csv not exist
    datasets = ['train.parquet.snappy', 'item_info.parquet.snappy',
                'track1_testset.parquet.snappy', 'track2_testset.parquet.snappy']
    datasets = {x:str(Path(x).name).split('.')[0]+'.csv' for x in datasets}
    for sfilename, tfilename in datasets.items():
        file_loadpath = osp.join(args.datapath_bronze,sfilename)
        file_savepath = osp.join(args.datapath_bronze,tfilename)
        if not osp.exists(file_savepath):
            logger.info('Converting {} to {}'.format(sfilename, tfilename))
            parquet_to_csv(file_loadpath)
        else:
            logger.info('{} file already exists, skipping!'.format(tfilename))
```

<!-- #region id="3usPrS7pKdo5" -->
```
Script to prepare data objects for training and testing
   Usage: from DataPrep import getUserFeaturesTrainSet, getPurchasedItemsTrainSet, getUserFeaturesTestSet
   1. getUserFeaturesTrainSet():
        return: DataFrame with N_ITEMS+N_USER_PORTRAITS columns
            first N_ITEMS cols: one hot encoding of clicked items
            last N_USER_PORTRAITS cols: normalized user portraits to [0,1] range
        DataFrame shape: (260087, 380+10)
   2. getPurchasedItemsTrainSet():
        return: a list, each element is a list of purchased itemIDs by a user
            each element i of the list corresponds to a user in row i of getUserFeaturesTrainSet()
        list length: 260087
   3. getUserFeaturesTestSet():
        return: DataFrame with N_ITEMS+N_USER_PORTRAITS columns
            first N_ITEMS cols: one hot encoding of clicked items
            last N_USER_PORTRAITS cols: normalized user portraits to [0,1] range
   4. getClusterLabels():
      return: (model, labels)
            model : model for testset prediction
            labels: numpy array of labels of clusters from the trainset
```
<!-- #endregion -->

```python id="I5iL4t4TM5C9"
def parseUserFeaturesOneLine(inputArray):
    """
    Kernel function
    Return: list of length args.N_ITEMS + args.N_USER_PORTRAITS 
    Input:
        inputArray: an array as a row of trainset or testset raw data
    ASSUMPTIONS:
        user_click_history is on column index  1 of inputArray
        user_portrait is on column index 2 of inputArray
    """
    CLICKHIST_INDEX = 1
    PORTRAIT_INDEX = 2
    output = [0]*(args.N_ITEMS + args.N_USER_PORTRAITS)
    # parse click history, assuming 
    clickSeries = inputArray[CLICKHIST_INDEX].split(',')
    clickedItems = [item.partition(':')[0] for item in clickSeries]
    # add clicked items to output
    for itemID in clickedItems:
        if int(itemID)<=0 or int(itemID)>=args.N_ITEMS:  # ignore if itemID invalid
            continue
        colIndex = int(itemID) - 1  # index of clicked item on an element of outputSharedList
        output[colIndex] = 1
    # parse user portraits
    portraits = inputArray[PORTRAIT_INDEX].split(',')
    if len(portraits)!=args.N_USER_PORTRAITS:
        raise Exception("row "+rowIndex+" of data set does not have the expected number of portrait features")
    # add portrait features to output
    for i in range(args.N_USER_PORTRAITS):
        colIndex = args.N_ITEMS + i  # index of feature on an element of outputSharedList
        output[colIndex] = int(portraits[i])
    return output
```

```python id="cTygb9y5Kdmc"
def prepareUserFeaturesTrainSet():
    """
    save to UserFeaturesTrainSet.pkl
        data frame with N_ITEMS+N_USER_PORTRAITS columns
        first N_ITEMS cols: one hot encoding of clicked items
        last N_USER_PORTRAITS cols: normalized user portraits
    Data source: trainset.csv
    """
    readfilepath = osp.join(args.datapath_bronze,args.filename_trainset)
    outfilepath = osp.join(args.datapath_silver,'UserFeaturesTrainSet.pkl')

    if not osp.exists(outfilepath):
        # read data to pd dataframe
        logger.info('reading raw data file ...')
        rawTrainSet = pd.read_csv(readfilepath, sep=args.data_sep)
        # create output frame
        colNames = ['clickedItem'+str(i+1) for i in range(args.N_ITEMS)] + ['userPortrait'+str(i+1) for i in range(args.N_USER_PORTRAITS)]
        output = pd.DataFrame(data = np.zeros(shape = (rawTrainSet.shape[0], args.N_ITEMS+args.N_USER_PORTRAITS)), columns = colNames)
        # parse each line in parallel
        # first objects in shared memory for input and output
        logger.info('creating shared memory objects ...')
        inputList = rawTrainSet.values.tolist()  # for memory efficiency
        p = mp.Pool(args.N_THREADS)
        logger.info('multiprocessing ... ')
        outputList = p.map(parseUserFeaturesOneLine, inputList)
        # convert outputSharedList back to DataFrame
        logger.info('convert to DataFrame ...')
        output = pd.DataFrame(data = outputList, columns = colNames)

        import gc; gc.collect()

        # normalize the portraits columns
        for i in range(args.N_USER_PORTRAITS):
            colName = 'userPortrait' + str(i+1)
            scaler = MinMaxScaler()
            output[colName] = scaler.fit_transform(output[colName].values.reshape(-1,1))
        # save to pickle file
        output.to_pickle(outfilepath)
        logger.info('Saved processed file at {}'.format(outfilepath))
    else:
        logger.info('{} Processed data already exists, skipping!'.format(outfilepath))
```

```python colab={"base_uri": "https://localhost:8080/"} id="NAu5MxaOU8Ze" executionInfo={"elapsed": 86158, "status": "ok", "timestamp": 1636271401213, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="5472f3fe-fbdc-4bec-f5db-22a88c94631e"
prepareUserFeaturesTrainSet()
```

```python id="56YPgpz4KdkL"
def preparePurchasedItemsTrainSet():
    """
    save to PurchasedItemsTrainSet.pkl
    Data source: trainset.csv
    """
    readfilepath = osp.join(args.datapath_bronze,args.filename_trainset)
    outfilepath = osp.join(args.datapath_silver,'PurchasedItemsTrainSet.pkl')

    if not osp.exists(outfilepath):
        # read data to pd dataframe
        logger.info('reading raw data file ...')
        rawTrainSet = pd.read_csv(readfilepath, sep=args.data_sep)
        output = []
        logger.info('processing ...')
        for i in tqdm(range(rawTrainSet.shape[0])):
            # parse each line
            exposedItems = rawTrainSet.exposed_items[i]
            labels = rawTrainSet.labels[i]
            exposedItems = exposedItems.split(',')
            labels = labels.split(',')
            purchasedItems = []
            for j in range(len(labels)):
                if int(labels[j])==1:
                    # item is purchased, append it to the purchasedItems list
                    purchasedItems.append(int(exposedItems[j]))

            import gc; gc.collect()

            # append the list of this row to output
            output.append(purchasedItems)
        # save to pickle file
        save_pickle(output, outfilepath)
        logger.info('Saved processed file at {}'.format(outfilepath))
    else:
        logger.info('{} Processed data already exists, skipping!'.format(outfilepath))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["c869695ad6a145b28d6940503dbba252", "9cf661a2b1264855aca52f042fae12d3", "b56438dab4e442a9bb21fac14ee32e4e", "4810d1c20ec54a15aaacdeb8f56e2478", "ee3ff1c5c1e848f39657742b0f5e0a18", "04e53ec292c94fe0b7e35ce5a8838b01", "5bd3f62468854d29b2a6c0c01c6e9edb", "32b52cab9c0149bf927e9f54794164cc", "45a71bf5bda641aeaca69f252401e758", "ccbed971f11a45ed8020060b2412b5ed", "5fb96f9cd5ba4fdd9a187743460e50d7"]} id="VcuA07bJWzap" outputId="1bb4ca57-16e2-46c3-dc01-3f0569643141"
preparePurchasedItemsTrainSet()
```

```python id="nQ5kzkryOonD"
def prepareUserFeaturesTestSet():
    """
    save to PurchasedItemsTestSet.pkl
    write content: userIDs, UserFeaturesTestSet
        userIDs: array of user ids
        UserFeaturesTestSet: data frame with N_ITEMS+N_USER_PORTRAITS columns
    Data source: track1_testset.csv
    """
    readfilepath = osp.join(args.datapath_bronze,args.filename_track1_testset)
    outfilepath = osp.join(args.datapath_silver,'PurchasedItemsTestSet.pkl')

    if not osp.exists(outfilepath):
        # read data to pd dataframe
        logger.info('reading raw data file ...')
        rawTestSet = pd.read_csv(readfilepath)
        # create output frame
        colNames = ['clickedItem'+str(i+1) for i in range(args.N_ITEMS)] + ['userPortrait'+str(i+1) for i in range(args.N_USER_PORTRAITS)]
        output = pd.DataFrame(data = np.zeros(shape = (rawTestSet.shape[0], args.N_ITEMS+args.N_USER_PORTRAITS)), columns = colNames)
        # parse each line in parallel
        # first objects in shared memory for input and output
        print('creating shared memory objects ... ')
        inputList = rawTestSet.values.tolist()  # for memory efficiency
        p = mp.Pool(args.N_THREADS)
        print('multiprocessing ... ')
        outputList = p.map(parseUserFeaturesOneLine, inputList)
        # convert outputSharedList back to DataFrame
        print('convert to DataFrame ...')
        output = pd.DataFrame(data = outputList, columns = colNames)
        # normalize the portraits columns
        for i in range(args.N_USER_PORTRAITS):
            colName = 'userPortrait' + str(i+1)
            scaler = MinMaxScaler()
            output[colName] = scaler.fit_transform(output[colName].values.reshape(-1,1))
        # create userIDs array
        userIDs = rawTestSet['user_id'].tolist()
        # save to pickle file
        save_pickle((userIDs,output), outfilepath)
        logger.info('Saved processed file at {}'.format(outfilepath))
    else:
        logger.info('{} Processed data already exists, skipping!'.format(outfilepath))
```

```python id="0desQ4yISXDX"
def getUserFeaturesTrainSet():
    savefilepath = osp.join(args.datapath_silver,'UserFeaturesTrainSet.pkl')
    return load_pickle(savefilepath)


def getPurchasedItemsTrainSet():
    savefilepath = osp.join(args.datapath_silver,'PurchasedItemsTrainSet.pkl')
    return load_pickle(savefilepath)


def getUserFeaturesTestSet():
    savefilepath = osp.join(args.datapath_silver,'UserFeaturesTestSet.pkl')
    return load_pickle(savefilepath)
```

```python id="DQMyFaoQUSiv"
def getExposedItemsTrainSet():
    """return list of exposed items in trainset and whether they are purchased
    (exposedItems, purchaseLabels)
    both are list of list
    """
    readfilepath = osp.join(args.datapath_bronze,args.filename_trainset)
    rawTrainSet = pd.read_csv(readfilepath)
    exposedItems = rawTrainSet.exposed_items
    purchaseLabels = rawTrainSet.labels
    exposedItems_out = []
    purchaseLabels_out = []
    for i in range(len(exposedItems)):
        items = exposedItems[i]
        labels = purchaseLabels[i]
        items = [int(x) for x in items.split(',')]
        labels = [int(x) for x in labels.split(',')]
        exposedItems_out.append(items)
        purchaseLabels_out.append(labels)
    return (exposedItems_out, purchaseLabels_out)
```

```python id="ewFHPoEmUsGG"
def getItemPrice():
    """return: array of item prices"""
    readfilepath = osp.join(args.datapath_bronze,args.filename_iteminfo)
    itemInfo = pd.read_csv(readfilepath)
    itemInfo = itemInfo.sort_values(by = 'item_id')
    itemPrice = itemInfo.price
    return itemPrice
```

```python id="ePwzVpKuSXBC"
def splitTrainSet(percentageTrain = 0.8):
    readfilepath = osp.join(args.datapath_bronze,args.filename_trainset)
    outfilepath = osp.join(args.datapath_silver,'splitTrainSet.pkl')

    if not osp.exists(outfilepath):
        # read raw data
        userFeatures = getUserFeaturesTrainSet()
        rawTrainSet = pd.read_csv(readfilepath)
        purchaseLabels1 = rawTrainSet.labels
        recItems1 = rawTrainSet.exposed_items
        N = len(purchaseLabels1)
        # create permutation index
        permutedIndex = np.random.permutation(N)
        trainIndex = permutedIndex[:int(N*percentageTrain)]
        testIndex = permutedIndex[int(N*percentageTrain):]
        # split user features
        userFeaturesTrain = userFeatures.iloc[trainIndex]
        userFeaturesTest = userFeatures.iloc[testIndex]
        # convert recItems to integer
        recItems = []
        for i, s in enumerate(recItems1):
        # loop thru samples
            recItems.append([int(x) for x in s.split(',')])
        recItems = np.array(recItems)
        # convert purchaseLabels to integer
        purchaseLabels = []
        for i, s in enumerate(purchaseLabels1):
        # loop thru samples
            purchaseLabels.append([int(x) for x in s.split(',')])
        purchaseLabels = np.array(purchaseLabels)
        # split recItems
        recItemsTrain = recItems[trainIndex]
        recItemsTest = recItems[testIndex]
        # split purchaseLabels
        purchaseLabelTrain = purchaseLabels[trainIndex]
        purchaseLabelTest = purchaseLabels[testIndex]
        # saving pickle
        save_pickle((userFeaturesTrain, recItemsTrain, purchaseLabelTrain, userFeaturesTest, recItemsTest, purchaseLabelTest), outfilepath)
        logger.info('Saved processed file at {}'.format(outfilepath))
    else:
        logger.info('{} Processed data already exists, skipping!'.format(outfilepath))
```

```python id="CaYdx9KRSW-y"
class Metrics:
    def __init__(self, recommended_testset, purchaseLabels_testset, itemPrice):
        """ recommended_testset: list
            purchaseLabels_testset: list
            itemPrice: list
        """
        self.rec = recommended_testset
        self.labels = purchaseLabels_testset
        self.price = itemPrice
        
    def calculate_metrics1(self, recommendedItems):
        """
        recommendedItems: list of length equal to recommended_testset
        metrics calculated by summing total rewards of purchased items, no punishment
        """
        score = 0
        for i in range(len(recommendedItems)):
        # loop each sample in data
            predItems = recommendedItems[i]
            givenItems = self.rec[i]
            labels = self.labels[i]
            purchaseAND = [givenItems[i] for i in range(9) if labels[i]==1]
            for item in predItems:
            # loop each items in the sample
                if item in purchaseAND:
                    score = score + self.price[item-1]
        return score
```

```python id="BM2s0auSacpX"
################################################################
# Exploring Collaborative Filtering based on KNN
################################################################
# 1. Use User data with clicked items and user_portraits
# 2. train KNN algorithm
# 3. for a test observaion, find K nearest neighbors
# 4. find the most common items from the neighbors to recommend
# 4. Use cross-validation to calibrate K

from sklearn.neighbors import NearestNeighbors

class KNNModel:
    def __init__(self, TrainData, purchaseData, K_neighbors):
        """
        train KNN model on TrainData
        purchaseData: list of length len(TrainData), each element is a list of purchased itemID
        K_neighbors: KNN parameter
        """
        self.model = NearestNeighbors(n_neighbors = K_neighbors)
        self.model.fit(TrainData)
        self.purchaseData = purchaseData
        self.K_neighbors = K_neighbors
    def predict(self, newPoint):
        """
        newPoint should have the same columns as TrainData, any number of row
        first find the nearest neighbors
        then count the frequency of their purchased items
        return: list with length = nrow of newPoint
            each element of list is a list of length 9
        """
        neighborDist, neighborIDs = self.model.kneighbors(newPoint)
        output = []
        # calculate score of purchased items with dictionary
        itemScore = {}
        for rowID in range(len(neighborIDs)):
            for i in range(self.K_neighbors):
                uID = neighborIDs[rowID][i]
                dist = neighborDist[rowID][i]
                if dist==0:
                    dist = 1e-7
                itemList = self.purchaseData[uID]
                for itemID in itemList:
                    if itemID not in itemScore.keys():
                        itemScore[itemID] = 1/dist
                    else:
                        itemScore[itemID] = itemScore[itemID] + 1/dist
            # find 9 items with highest frequency
            # first sort the dict by decreasing value
            sortedDict = {k: v for k, v in sorted(itemScore.items(), key=lambda item: item[1], reverse = True)}
            finalItems = list(sortedDict.keys())[:9]
            output.append(finalItems)
        return output
```

```python id="NL4KP4XrasQn"
def knn_training_and_prediction():
    # load processed datasets
    TrainSet = getUserFeaturesTrainSet()
    PurchasedItems = getPurchasedItemsTrainSet()
    # initiate knn model object
    model = KNNModel(TrainSet, PurchasedItems, 50)
    # get test set
    userIDs, TestSet = getUserFeaturesTestSet()
    # make prediction
    recommendedItems = model.predict(TestSet)
    # format data according to submission format and write to file
    outFile = '/tf/shared/track2_output.csv'
    f = open(outFile, "w")
    f.write('id,itemids')
    for i in range(len(userIDs)):
        f.write('\n')
        itemList = recommendedItems[i]
        itemString = ' '.join([str(j) for j in itemList])
        outString = str(userIDs[i]) + ',' + itemString
        f.write(outString)
```

<!-- #region id="igLLZV6gGu-v" -->
---
<!-- #endregion -->

<!-- #region id="koFQxtgos6gE" -->
## Jobs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2y8mdDjds6dr" executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1636267384268, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="c4db2615-7617-4509-ae6a-3bbb88430e98"
logger.info('JOB START: DOWNLOAD_RAW_DATASET')
download_dataset()
logger.info('JOB END: DOWNLOAD_RAW_DATASET')
```

```python colab={"base_uri": "https://localhost:8080/"} id="3ig3tPpB2Fx-" executionInfo={"elapsed": 14250, "status": "ok", "timestamp": 1636267418097, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="bbc27a15-6f9c-4e5a-f227-e6cfd9363c6f"
logger.info('JOB START: DATASET_CONVERSION_PARQUET_TO_CSV')
convert_dataset()
logger.info('JOB END: DATASET_CONVERSION_PARQUET_TO_CSV')
```

```python id="toXUL9pVItp_"

```
