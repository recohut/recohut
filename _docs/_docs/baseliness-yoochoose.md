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

<!-- #region id="RD5Z0MFEGDxN" -->
# Training AR, SR, S-POP, and VSKNN models on Yoochoose dataset
<!-- #endregion -->

<!-- #region id="_a5mQwhZADCf" -->
## Yoochoose Dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3BJTggv4AfLp" executionInfo={"status": "ok", "timestamp": 1641013225139, "user_tz": -330, "elapsed": 6815, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="93f01e5b-3d89-4695-b424-c8b838d08296"
!pip install -qq recohut==0.0.9
```

```python id="3Z0xeHBh_9bc"
from recohut.utils.common_utils import download_url
```

```python colab={"base_uri": "https://localhost:8080/", "height": 70} id="IIcwSWSa1j7S" executionInfo={"status": "ok", "timestamp": 1641013297948, "user_tz": -330, "elapsed": 2381, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ae220435-0079-4d5c-b11c-0bdd7f6107bf"
data_root = '/content/data'
download_url('https://github.com/RecoHut-Datasets/yoochoose/raw/v4/yoochoose_train.txt', data_root)
download_url('https://github.com/RecoHut-Datasets/yoochoose/raw/v4/yoochoose_valid.txt', data_root)
```

<!-- #region id="IIqP4BuO_jXp" -->
## Association Rules (AR)

Simple Association Rules (AR) are a simplified version of the association rule mining technique [Agrawal et al. 1993] with a maximum rule size of two. The method is designed to capture the frequency of two co-occurring events, e.g., “Customers who bought . . . also bought”. Algorithmically, the rules and their corresponding importance are “learned” by counting how often the items i and j occurred together in a session of any user. Let a session s be a chronologically ordered tuple of item click events s = ($s_1$,$s_2$,$s_3$, . . . ,$s_m$) and $S_p$ the set of all past sessions. Given a user’s current session s with $s_{|s|}$ being the last item interaction in s, we can define the score for a recommendable item i as follows, where the indicator function $1_{EQ}(a,b)$ is 1 in case a and b refer to the same item and 0 otherwise.

$$score_{AR}(i,s) = \dfrac{1}{\sum_{p \in S_p}\sum_{x=1}^{|p|}1_{EQ}(s_{|s|},p_x)\cdot(|p|-1)}\sum_{p \in s_p}\sum_{x=1}^{|p|}\sum_{y=1}^{|p|}1_{EQ}(s_{|s|},p_x)\cdot1_{EQ}(i,p_y)$$

In the above equation, the sums at the right-hand side represent the counting scheme. The term at the left-hand side normalizes the score by the number of total rule occurrences originating from the current item $s_{|s|}$. A list of recommendations returned by the ar method then contains the items with the highest scores in descending order. No minimum support or confidence thresholds are applied.

> References
- [https://arxiv.org/pdf/1803.09587.pdf](https://arxiv.org/pdf/1803.09587.pdf)
- [http://www.rakesh.agrawal-family.com/papers/sigmod93assoc.pdf](http://www.rakesh.agrawal-family.com/papers/sigmod93assoc.pdf)
- [https://github.com/mmaher22/iCV-SBR/tree/master/Source Codes/AR%26SR_Python](https://github.com/mmaher22/iCV-SBR/tree/master/Source%20Codes/AR%26SR_Python)
<!-- #endregion -->

```python id="DPCu6bnQ1j7R"
import numpy as np
import pandas as pd
import collections as col
```

```python id="ItJSYYDU9o7z"
class AssosiationRules: 
    '''
    AssosiationRules(pruning=10, session_key='SessionId', item_keys=['ItemId'])
    Parameters
    --------
    pruning : int
        Prune the results per item to a list of the top N co-occurrences. (Default value: 10)
    session_key : string
        The data frame key for the session identifier. (Default value: SessionId)
    item_keys : string
        The data frame list of keys for the item identifier as first item in list 
        and features keys next. (Default value: [ItemId])
    '''
    def __init__( self, pruning=10, session_key='SessionID', item_keys=['ItemID'] ):
        self.pruning = pruning
        self.session_key = session_key
        self.item_keys = item_keys
        self.items_features = {}
        self.predict_for_item_ids = []
        
    def fit( self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. 
            It has one column for session IDs, one for item IDs and many for the
            item features if exist.
            It must have a header. Column names are arbitrary, but must 
            correspond to the ones you set during the initialization of the 
            network (session_key, item_keys).
        '''
        cur_session = -1
        last_items = []
        all_rules = []
        indices_item = []
        for i in self.item_keys:
            all_rules.append(dict())
            indices_item.append( data.columns.get_loc(i) )

        data.sort_values(self.session_key, inplace=True)
        index_session = data.columns.get_loc(self.session_key)
        
        #Create Dictionary of items and their features
        for row in data.itertuples( index=False ):
            item_id = row[indices_item[0]]
            if not item_id in self.items_features.keys() :
                self.items_features[item_id] = []
                for i in indices_item:
                    self.items_features[item_id].append(row[i])
              
        for i in range(len(self.item_keys)):
            rules = all_rules[i]
            index_item = indices_item[i]
            for row in data.itertuples( index=False ):
                session_id, item_id = row[index_session], row[index_item]
                if session_id != cur_session:
                    cur_session = session_id
                    last_items = []
                else: 
                    for item_id2 in last_items:                
                        if not item_id in rules :
                            rules[item_id] = dict()                
                        if not item_id2 in rules :
                            rules[item_id2] = dict()                
                        if not item_id in rules[item_id2]:
                            rules[item_id2][item_id] = 0           
                        if not item_id2 in rules[item_id]:
                            rules[item_id][item_id2] = 0
                        
                        rules[item_id][item_id2] += 1
                        rules[item_id2][item_id] += 1
                        
                last_items.append(item_id)
                
            if self.pruning > 0:
                rules = self.prune(rules) 
                
            all_rules[i] = rules
        self.all_rules = all_rules
        self.predict_for_item_ids = list(self.all_rules[0].keys())
    def predict_next(self, session_items, k = 20):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_items : List
            Items IDs in current session.
        k : Integer
            How many items to recommend
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. 
            Indexed by the item IDs.
        
        '''
        all_len = len(self.predict_for_item_ids)
        input_item_id = session_items[-1]
        preds = np.zeros( all_len ) 
             
        if input_item_id in self.all_rules[0].keys():
            for k_ind in range(all_len):
                key = self.predict_for_item_ids[k_ind]
                if key in session_items:
                    continue
                try:
                    preds[ k_ind ] += self.all_rules[0][input_item_id][key]
                except:
                    pass
                for i in range(1, len(self.all_rules)):
                    input_item_feature = self.items_features[input_item_id][i]
                    key_feature = self.items_features[key][i]
                    try:
                        preds[ k_ind ] += self.all_rules[i][input_item_feature][key_feature]
                    except:
                        pass
        
        series = pd.Series(data=preds, index=self.predict_for_item_ids)
        series = series / series.max()
        
        return series.nlargest(k).index.values
    
    def prune(self, rules): 
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
        Parameters
            --------
            rules : dict of dicts
                The rules mined from the training data
        '''
        for k1 in rules:
            tmp = rules[k1]
            if self.pruning < 1:
                keep = len(tmp) - int( len(tmp) * self.pruning )
            elif self.pruning >= 1:
                keep = self.pruning
            counter = col.Counter( tmp )
            rules[k1] = dict()
            for k2, v in counter.most_common( keep ):
                rules[k1][k2] = v
        return rules
```

```python id="ZMoCIzvw9mhZ"
import os
import time
import argparse
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/"} id="4JK1xoL59WBm" executionInfo={"status": "ok", "timestamp": 1641017921756, "user_tz": -330, "elapsed": 159820, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1ef77be3-cce8-471a-c8fc-7779d1931371"
parser = argparse.ArgumentParser()
parser.add_argument('--prune', type=int, default=10, help="Association Rules Pruning Parameter")
parser.add_argument('--K', type=int, default=20, help="K items to be used in Recall@K and MRR@K")
parser.add_argument('--itemid', default='sid', type=str)
parser.add_argument('--sessionid', default='uid', type=str)
parser.add_argument('--item_feats', default='', type=str, 
                    help="Names of Columns containing items features separated by #")
parser.add_argument('--valid_data', default='yoochoose_valid.txt', type=str)
parser.add_argument('--train_data', default='yoochoose_train.txt', type=str)
parser.add_argument('--data_folder', default=data_root, type=str)

# Get the arguments
args = parser.parse_args([])
train_data = os.path.join(args.data_folder, args.train_data)
x_train = pd.read_csv(train_data)
valid_data = os.path.join(args.data_folder, args.valid_data)
x_valid = pd.read_csv(valid_data)
x_valid.sort_values(args.sessionid, inplace=True)

items_feats = [args.itemid]
ffeats = args.item_feats.strip().split("#")
if ffeats[0] != '':
    items_feats.extend(ffeats)

print('Finished Reading Data.')
# Fitting AR Model
print('Start Model Fitting...')
t1 = time.time()
model = AssosiationRules(session_key = args.sessionid, item_keys = items_feats, pruning=args.prune)
model.fit(x_train)
t2 = time.time()
print('End Model Fitting with total time =', t2 - t1)

print('Start Predictions...')
# Test Set Evaluation
test_size = 0.0
hit = 0.0
MRR = 0.0
cur_length = 0
cur_session = -1
last_items = []
t1 = time.time()
index_item = x_valid.columns.get_loc(args.itemid)
index_session = x_valid.columns.get_loc(args.sessionid)
train_items = model.items_features.keys()
counter = 0
for row in x_valid.itertuples(index=False):
    counter += 1
    if counter % 5000 == 0:
        print('Finished Prediction for ', counter, 'items.')
    session_id, item_id = row[index_session], row[index_item]
    if session_id != cur_session:
        cur_session = session_id
        last_items = []
        cur_length = 0
    
    if not item_id in last_items and item_id in train_items:
        if len(last_items) > cur_length: #make prediction
            cur_length += 1
            test_size += 1
            # Predict the most similar items to items
            predictions = model.predict_next(last_items, k = args.K)
            #print('preds:', predictions)
            # Evaluation
            rank = 0
            for predicted_item in predictions:
                rank += 1
                if predicted_item == item_id:
                    hit += 1.0
                    MRR += 1/rank
                    break
        
        last_items.append(item_id)
t2 = time.time()
print('Recall: {}'.format(hit / test_size))
print ('\nMRR: {}'.format(MRR / test_size))
print('End Model Predictions with total time =', t2 - t1)
```

<!-- #region id="kJWaToepAU5h" -->
## Sequential Rules

The SR method as proposed in [Kamehkhosh et al. 2017] is a variation of MC and AR. It also takes the order of actions into account, but in a less restrictive manner. In contrast to the MC method, we create a rule when an item q appeared after an item p in a session even when other events happened between p and q. When assigning weights to the rules, we consider the number of elements appearing between p and q in the session. Specifically, we use the weight function $w_{SR}(x)$ = 1/(x), where x corresponds to the number of steps between the two items. Given the current session s, the sr method calculates the score for the target item i as follows:

$$score_{SR}(i,s) = \dfrac{1}{\sum_{p \in S_p}\sum_{x=2}^{|p|}1_{EQ}(s_{|s|},p_x)\cdot x}\sum_{p \in s_p}\sum_{x=2}^{|p|}\sum_{y=1}^{x-1}1_{EQ}(s_{|s|},p_y)\cdot1_{EQ}(i,p_x)\cdot w_{SR}(x-y)$$

In contrast to the equation for AR, the third inner sum only considers indices of previous item view events for each session p. In addition, the weighting function $w_{SR}(x)$ is added. Again, we normalize the absolute score by the total number of rule occurrences for the current item $s_{|s|}$.

> References
- [https://arxiv.org/pdf/1803.09587.pdf](https://arxiv.org/pdf/1803.09587.pdf)
- [https://github.com/mmaher22/iCV-SBR/tree/master/Source Codes/AR%26SR_Python](https://github.com/mmaher22/iCV-SBR/tree/master/Source%20Codes/AR%26SR_Python)
<!-- #endregion -->

```python id="4M2bpBGKAU3U"
import numpy as np
import pandas as pd
from math import log10
import collections as col
```

```python id="Kd0SJmn9AU1j"
class SequentialRules: 
    '''
    SequentialRules(steps = 10, weighting='div', pruning=20.0, session_key='SessionId', item_keys=['ItemId'])
        
    Parameters
    --------
    pruning : int
        Prune the results per item to a list of the top N co-occurrences. (Default value: 10)
    session_key : string
        The data frame key for the session identifier. (Default value: SessionId)
    item_keys : string
        The data frame list of keys for the item identifier as first item in list 
        and features keys next. (Default value: [ItemID])    
    steps : int
        Number of steps to walk back from the currently viewed item. (Default value: 10)
    weighting : string
        Weighting function for the previous items (linear, same, div, log, qudratic). (Default value: div)
    pruning : int
        Prune the results per item to a list of the top N sequential co-occurrences. (Default value: 20). 
    '''
    
    def __init__( self, steps = 10, weighting='div', pruning=20, 
                 session_key='SessionID', item_keys=['ItemId']):
        self.steps = steps
        self.pruning = pruning
        self.weighting = weighting
        self.session_key = session_key
        self.item_keys = item_keys
        self.items_features = {}
        self.predict_for_item_ids = []
        self.session = -1
        self.session_items = []
            
    def fit( self, train):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. 
            It has one column for session IDs, one for item IDs and many for the
            item features if exist.
            It must have a header. Column names are arbitrary, but must 
            correspond to the ones you set during the initialization of the 
            network (session_key, item_keys).
        '''
        cur_session = -1
        last_items = []
        all_rules = []
        indices_item = []
        for i in self.item_keys:
            all_rules.append(dict())
            indices_item.append( train.columns.get_loc(i) )
            
        train.sort_values(self.session_key, inplace=True)
        index_session = train.columns.get_loc(self.session_key)
        
        #Create Dictionary of items and their features
        for row in train.itertuples( index=False ):
            item_id = row[indices_item[0]]
            if not item_id in self.items_features.keys() :
                self.items_features[item_id] = []
                for i in indices_item:
                    self.items_features[item_id].append(row[i])
        
        for i in range(len(self.item_keys)):
            rules = all_rules[i]
            index_item = indices_item[i] #which feature of the items to work on
            for row in train.itertuples( index=False ):
                session_id, item_id = row[index_session], row[index_item]
                if session_id != cur_session:
                    cur_session = session_id
                    last_items = []
                else: 
                    for j in range( 1, self.steps+1 if len(last_items) >= self.steps else len(last_items)+1 ):
                        prev_item = last_items[-j]   
                        if not prev_item in rules :
                            rules[prev_item] = dict()        
                        if not item_id in rules[prev_item]:
                            rules[prev_item][item_id] = 0
                        
                        rules[prev_item][item_id] += getattr(self, self.weighting)( j )
                        
                last_items.append(item_id)
                
            if self.pruning > 0 :
                rules = self.prune( rules )
            
            all_rules[i] = rules
        
        self.all_rules = all_rules
        self.predict_for_item_ids = list(self.all_rules[0].keys())
    
    def linear(self, i):
        return 1 - (0.1*i) if i <= 100 else 0
    
    def same(self, i):
        return 1
    
    def div(self, i):
        return 1/i
    
    def log(self, i):
        return 1/(log10(i+1.7))
    
    def quadratic(self, i):
        return 1/(i*i)
    
    def predict_next(self, session_items, k = 20):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_items : List
            Items IDs in current session.
        k : Integer
            How many items to recommend
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. 
            Indexed by the item IDs.
        
        '''
        all_len = len(self.predict_for_item_ids)
        input_item_id = session_items[-1]
        preds = np.zeros( all_len ) 
             
        if input_item_id in self.all_rules[0].keys():
            for k_ind in range(all_len):
                key = self.predict_for_item_ids[k_ind]
                if key in session_items:
                    continue
                try:
                    preds[ k_ind ] += self.all_rules[0][input_item_id][key]
                except:
                    pass
                for i in range(1, len(self.all_rules)):
                    input_item_feature = self.items_features[input_item_id][i]
                    key_feature = self.items_features[key][i]
                    try:
                        preds[ k_ind ] += self.all_rules[i][input_item_feature][key_feature]
                    except:
                        pass
        
        series = pd.Series(data=preds, index=self.predict_for_item_ids)
        series = series / series.max()
        
        return series.nlargest(k).index.values
    
    def prune(self, rules): 
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
        Parameters
            --------
            rules : dict of dicts
                The rules mined from the training data
        '''
        for k1 in rules:
            tmp = rules[k1]
            if self.pruning < 1:
                keep = len(tmp) - int( len(tmp) * self.pruning )
            elif self.pruning >= 1:
                keep = self.pruning
            counter = col.Counter( tmp )
            rules[k1] = dict()
            for k2, v in counter.most_common( keep ):
                rules[k1][k2] = v
        return rules
```

```python id="-iV8WJ2RAUzU"
import os
import time
import argparse
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/"} id="vcjwSWY5AcQV" executionInfo={"status": "ok", "timestamp": 1641013547897, "user_tz": -330, "elapsed": 146710, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3c0dfba2-b8dd-41cb-cb02-3e62cfbcf1cb"
parser = argparse.ArgumentParser()
parser.add_argument('--prune', type=int, default=0, help="Association Rules Pruning Parameter")
parser.add_argument('--K', type=int, default=20, help="K items to be used in Recall@K and MRR@K")
parser.add_argument('--steps', type=int, default=10, help="Max Number of steps to walk back from the currently viewed item")
parser.add_argument('--weighting', type=str, default='div', help="Weighting function for the previous items (linear, same, div, log, qudratic)")
parser.add_argument('--itemid', default='sid', type=str)
parser.add_argument('--sessionid', default='uid', type=str)
parser.add_argument('--item_feats', default='', type=str, 
                    help="Names of Columns containing items features separated by #")
parser.add_argument('--valid_data', default='yoochoose_valid.txt', type=str)
parser.add_argument('--train_data', default='yoochoose_train.txt', type=str)
parser.add_argument('--data_folder', default=data_root, type=str)

# Get the arguments
args = parser.parse_args([])
train_data = os.path.join(args.data_folder, args.train_data)
x_train = pd.read_csv(train_data)
valid_data = os.path.join(args.data_folder, args.valid_data)
x_valid = pd.read_csv(valid_data)
x_valid.sort_values(args.sessionid, inplace=True)

items_feats = [args.itemid]
ffeats = args.item_feats.strip().split("#")
if ffeats[0] != '':
    items_feats.extend(ffeats)

print('Finished Reading Data \nStart Model Fitting...')
# Fitting AR Model
t1 = time.time()
model = SequentialRules(session_key = args.sessionid, item_keys = items_feats, 
                        pruning=args.prune, steps=args.steps, weighting=args.weighting)
model.fit(x_train)
t2 = time.time()
print('End Model Fitting with total time =', t2 - t1, '\n Start Predictions...')

# Test Set Evaluation
test_size = 0.0
hit = 0.0
MRR = 0.0
cur_length = 0
cur_session = -1
last_items = []
t1 = time.time()
index_item = x_valid.columns.get_loc(args.itemid)
index_session = x_valid.columns.get_loc(args.sessionid)
train_items = model.items_features.keys()
counter = 0
for row in x_valid.itertuples( index=False ):
    counter += 1
    if counter % 5000 == 0:
        print('Finished Prediction for ', counter, 'items.')
    session_id, item_id = row[index_session], row[index_item]
    if session_id != cur_session:
        cur_session = session_id
        last_items = []
        cur_length = 0
    
    if not item_id in last_items and item_id in train_items:
        if len(last_items) > cur_length: #make prediction
            cur_length += 1
            test_size += 1
            # Predict the most similar items to items
            predictions = model.predict_next(last_items, k = args.K)
            #print('preds:', predictions)
            # Evaluation
            rank = 0
            for predicted_item in predictions:
                rank += 1
                if predicted_item == item_id:
                    hit += 1.0
                    MRR += 1/rank
                    break
        
        last_items.append(item_id)
t2 = time.time()
print('Recall: {}'.format(hit / test_size))
print ('\nMRR: {}'.format(MRR / test_size))
print('End Model Predictions with total time =', t2 - t1)
```

<!-- #region id="pxzrxEm-BSZt" -->
## S-Pop

Session popularity predictor that gives higher scores to items with higher number of occurrences in the session. Ties are broken up by adding the popularity score of the item.

The score is given by $r_{s,i} = supp_{s,i} + \frac{supp_i}{(1+supp_i)}$.

> References
- [https://github.com/mmaher22/iCV-SBR/tree/master/Source Codes/S-POP_Python](https://github.com/mmaher22/iCV-SBR/tree/master/Source%20Codes/S-POP_Python)
<!-- #endregion -->

```python id="FnAGmanYCG6r"
import numpy as np
import pandas as pd
```

```python id="MCqdZrvLCHPb"
class SessionPop:
    '''
    SessionPop(top_n=100, item_key='ItemId', support_by_key=None)
    Session popularity predictor that gives higher scores to items with higher number of occurrences in the session. 
    Ties are broken up by adding the popularity score of the item.
    The score is given by:
    .. math::
        r_{s,i} = supp_{s,i} + \\frac{supp_i}{(1+supp_i)}
    Parameters
    --------
    top_n : int
        Only give back non-zero scores to the top N ranking items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    item_key : string
        The header of the item IDs in the training data. (Default value: 'ItemId')
    '''    
    def __init__(self, top_n = 1000, session_key = 'SessionId', item_key = 'ItemId'):
        self.top_n = top_n
        self.item_key = item_key
        self.session_id = session_key
        
    def fit(self, data):
        '''
        Trains the predictor.
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. 
            It has one column for session IDs, one for item IDs.
        '''
        self.items = data[self.item_key].unique()
        grp = data.groupby(self.item_key)
        self.pop_list = grp.size()
        self.pop_list = self.pop_list / (self.pop_list + 1)
        self.pop_list.sort_values(ascending=False, inplace=True)
        self.pop_list = self.pop_list.head(self.top_n)
        self.prev_session_id = -1
         
    def predict_next(self, last_items, k):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
        Parameters
        --------
        last_items : list of items clicked in current session
        k : number of items to recommend and evaluate based on it
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        '''
        pers = {}
        for i in last_items:
            pers[i] = pers[i] + 1 if i in pers.keys() else  1
        
        preds = np.zeros(len(self.items))
        mask = np.in1d(self.items, self.pop_list.index)
        ser = pd.Series(pers)
        preds[mask] = self.pop_list[self.items[mask]]
        
        mask = np.in1d(self.items, ser.index)
        preds[mask] += ser[self.items[mask]]
        
        series = pd.Series(data=preds, index=self.items)
        series = series / series.max()    
        return series.nlargest(k).index.values
```

```python id="jENyoADWCJ1C"
import os
import time
import argparse
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/"} id="CuDUFxAzCMOu" executionInfo={"status": "ok", "timestamp": 1641013705386, "user_tz": -330, "elapsed": 35868, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d23b66f7-d48f-43e9-d9c6-33a0d95ae315"
parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=20, help="K items to be used in Recall@K and MRR@K")
parser.add_argument('--topn', type=int, default=100, help="Number of top items to return non zero scores for them (most popular)")
parser.add_argument('--itemid', default='sid', type=str)
parser.add_argument('--sessionid', default='uid', type=str)
parser.add_argument('--valid_data', default='yoochoose_valid.txt', type=str)
parser.add_argument('--train_data', default='yoochoose_train.txt', type=str)
parser.add_argument('--data_folder', default=data_root, type=str)

# Get the arguments
args = parser.parse_args([])
train_data = os.path.join(args.data_folder, args.train_data)
x_train = pd.read_csv(train_data)
valid_data = os.path.join(args.data_folder, args.valid_data)
x_valid = pd.read_csv(valid_data)
x_valid.sort_values(args.sessionid, inplace=True)

print('Finished Reading Data \nStart Model Fitting...')
# Fitting AR Model
t1 = time.time()
model = SessionPop(top_n = args.topn, session_key = args.sessionid, item_key = args.itemid)
model.fit(x_train)
t2 = time.time()
print('End Model Fitting with total time =', t2 - t1, '\n Start Predictions...')

# Test Set Evaluation
test_size = 0.0
hit = 0.0
MRR = 0.0
cur_length = 0
cur_session = -1
last_items = []
t1 = time.time()
index_item = x_valid.columns.get_loc(args.itemid)
index_session = x_valid.columns.get_loc(args.sessionid)
train_items = model.items
counter = 0
for row in x_valid.itertuples( index=False ):
    counter += 1
    if counter % 5000 == 0:
        print('Finished Prediction for ', counter, 'items.')
    session_id, item_id = row[index_session], row[index_item]
    if session_id != cur_session:
        cur_session = session_id
        last_items = []
        cur_length = 0
    
    if item_id in train_items:
        if len(last_items) > cur_length: #make prediction
            cur_length += 1
            test_size += 1
            # Predict the most similar items to items
            predictions = model.predict_next(last_items, k = args.K)
            # Evaluation
            rank = 0
            for predicted_item in predictions:
                rank += 1
                if predicted_item == item_id:
                    hit += 1.0
                    MRR += 1/rank
                    break
        
        last_items.append(item_id)
t2 = time.time()
print('Recall: {}'.format(hit / test_size))
print ('\nMRR: {}'.format(MRR / test_size))
print('End Model Predictions with total time =', t2 - t1)
```

<!-- #region id="E6JBbdouCWiY" -->
## VSKNN
<!-- #endregion -->

```python id="NHpMDGCZCrDF"
from operator import itemgetter
from math import sqrt
import time
import numpy as np
import pandas as pd
from math import log10
```

```python id="aTe878vVC2WW"
class VMContextKNN:
    '''
    VMContextKNN( k, sample_size=1000, similarity='cosine', weighting='div', weighting_score='div_score', session_key = 'SessionId', item_key= 'ItemId')
    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 200)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 2000)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: cosine)
    weighting : string
        Decay function to determine the importance/weight of individual actions in the current session (linear, same, div, log, quadratic). (default: div)
    weighting_score : string
        Decay function to lower the score of candidate items from a neighboring sessions that were selected by less recently clicked items in the current session. (linear, same, div, log, quadratic). (default: div_score)
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    '''
    def __init__( self, k=200, sample_size=0, similarity='cosine', weighting='div', weighting_score='div_score', session_key = 'SessionId', item_key= 'ItemId'):
       
        self.k = k
        self.sample_size = sample_size
        self.weighting = weighting
        self.weighting_score = weighting_score
        self.similarity = similarity
        self.session_key = session_key
        self.item_key = item_key
        
        #updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.session_time = dict()
        self.min_time = -1
        
        self.sim_time = 0
        
    def fit(self, train, items=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        self.items_ids = list(train[self.item_key].unique())
        train[self.item_key] = train[self.item_key].astype('category')
        self.new_old = dict(enumerate(train[self.item_key].cat.categories))
        self.old_new = {y:x for x,y in self.new_old.items()}
        train[[self.item_key]] = train[[self.item_key]].apply(lambda x: x.cat.codes)
        
        self.freqs = dict(train[self.item_key].value_counts())
        
        self.num_items = train[self.item_key].max()
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        
        session = -1
        session_items = set()
        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session : session_items})
                session = row[index_session]
                session_items = set()
            session_items.add(row[index_item])
            
            # cache sessions involving an item
            map_is = self.item_session_map.get( row[index_item] )
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item] : map_is})
            map_is.add(row[index_session])
            
        # Add the last tuple    
        self.session_item_map.update({session : session_items})
        self.predict_for_item_ids = list(range(1, self.num_items+1))
        
        
    def predict_next(self, session_items, k):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_items : List
            Items IDs in current session.
        k : Integer
            How many items to recommend
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. 
            Indexed by the item IDs.
        '''
            
        all_len = len(self.predict_for_item_ids)
        input_item_id = session_items[-1]
        neighbors = self.find_neighbors(input_item_id, session_items)
        scores = self.score_items(neighbors, session_items)
        
        # Create things in the format ..
        preds = np.zeros(all_len)
        scores_keys = list(scores.keys())
        for i in range(all_len):
            if i+1 in scores_keys:
                preds[i] = scores[i+1]
                
        series = pd.Series(data = preds, index = self.predict_for_item_ids)
        series = series / series.max()
        return series.nlargest(k).index.values
    
    def items_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_item_map.get(session);
    
    def vec_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_vec_map.get(session);
    
    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.item_session_map.get( item_id ) if item_id in self.item_session_map else set()
        
        
    def most_recent_sessions( self, sessions, number ):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get( session )
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))
            
        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        #print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add( element[0] )
        #print 'returning sample of size ', len(sample)
        return sample
        
        
    def possible_neighbor_sessions(self, input_item_id):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears. 
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        
        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id )
               
        if self.sample_size == 0: #use all session as possible neighbors
            return self.relevant_sessions

        else: #sample some sessions
            if len(self.relevant_sessions) > self.sample_size:    
                return self.relevant_sessions[-self.sample_size:]
            else: 
                return self.relevant_sessions
                        
    def calc_similarity(self, session_items, sessions):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id,similarity)           
        '''
        pos_map = {}
        length = len(session_items)
        
        count = 1
        for item in session_items:
            if self.weighting is not None: 
                pos_map[item] = getattr(self, self.weighting)(count, length)
                count += 1
            else:
                pos_map[item] = 1
        #print('POS MAP: ', pos_map, session_items)
        items = set(session_items)
        neighbors = []
        for session in sessions: 
            n_items = self.items_for_session(session)
            similarity = self.vec(items, n_items, pos_map)        
            if similarity > 0:
                neighbors.append((session, similarity))
        return neighbors

    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def find_neighbors( self, input_item_id, session_items):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        session_items: list of item ids in current session
        input_item_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        #print('SESSION ITEMS1:', session_items)
        possible_neighbors = self.possible_neighbor_sessions(input_item_id)
        possible_neighbors = self.calc_similarity(session_items, possible_neighbors)
        
        possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1] )
        possible_neighbors = possible_neighbors[:self.k]
        
        return possible_neighbors
    
            
    def score_items(self, neighbors, current_session):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session( session[0] )
            step = 1
            
            for item in reversed( current_session ):
                if item in items:
                    decay = getattr(self, self.weighting_score)(step)
                    break
                step += 1
                                    
            for item in items:
                old_score = scores.get( item )
                similarity = session[1]
                
                if old_score is None:
                    scores.update({item : ( similarity * decay ) })
                else: 
                    new_score = old_score + ( similarity * decay )
                    scores.update({item : new_score})
                    
        return scores
    
    
    def linear_score(self, i):
        return 1 - (0.1*i) if i <= 100 else 0
    
    def same_score(self, i):
        return 1
    
    def div_score(self, i):
        return 1/i
    
    def log_score(self, i):
        return 1/(log10(i+1.7))
    
    def quadratic_score(self, i):
        return 1/(i*i)
    
    def linear(self, i, length):
        return 1 - (0.1*(length-i)) if i <= 10 else 0
    
    def same(self, i, length):
        return 1
    
    def div(self, i, length):
        return i/length
    
    def log(self, i, length):
        return 1/(log10((length-i)+1.7))
    
    def quadratic(self, i, length):
        return (i/length)**2


    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second )
        res = intersection / union
        
        self.sim_time += (time.clock() - sc)
        
        return res 
    
    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result
    
    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / ( la + lb -li )

        return result
    
    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = len(first&second)
        b = len(first)
        c = len(second)
        
        result = (2 * a) / ((2 * a) + b + c)

        return result
    
    def vec(self, first, second, map):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = first & second
        sum = 0
        for i in a:
            sum += map[i]
        
        result = sum / len(map)

        return result    
```

```python id="1-q4woM5C7fO"
import gc
import os
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/"} id="RX4xhYdbC-He" executionInfo={"status": "ok", "timestamp": 1641017117647, "user_tz": -330, "elapsed": 2828265, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a7647ecf-ba8b-4fd5-a1f1-012d3b3f1be6"
parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=20, help="K items to be used in Recall@K and MRR@K")
parser.add_argument('--neighbors', type=int, default=200, help="K neighbors to be used in KNN")
parser.add_argument('--sample', type=int, default=0, help="Max Number of steps to walk back from the currently viewed item")
parser.add_argument('--weight_score', type=str, default='div_score', help="Decay function to lower the score of candidate items from a neighboring sessions that were selected by less recently clicked items in the current session. (linear, same, div, log, quadratic)_score")
parser.add_argument('--weighting', type=str, default='div', help="Decay function to determine the importance/weight of individual actions in the current session(linear, same, div, log, qudratic)")
parser.add_argument('--similarity', type=str, default='cosine', help="String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: cosine)")
parser.add_argument('--itemid', default='sid', type=str)
parser.add_argument('--sessionid', default='uid', type=str)
parser.add_argument('--valid_data', default='yoochoose_valid.txt', type=str)
parser.add_argument('--train_data', default='yoochoose_train.txt', type=str)
parser.add_argument('--data_folder', default=data_root, type=str)
args = parser.parse_args([])

# Get the arguments
train_data = os.path.join(args.data_folder, args.train_data)
x_train = pd.read_csv(train_data)
x_train.sort_values(args.sessionid, inplace=True)
distinct_train = x_train[args.itemid].nunique()

valid_data = os.path.join(args.data_folder, args.valid_data)
x_valid = pd.read_csv(valid_data)
x_valid.sort_values(args.sessionid, inplace=True)

print('Finished Reading Data \nStart Model Fitting...')
# Fitting Model
t1 = time.time()
model = VMContextKNN(k = args.neighbors, sample_size = args.sample, similarity = args.similarity, 
					 weighting = args.weighting, weighting_score = args.weight_score,
					 session_key = args.sessionid, item_key = args.itemid)
model.fit(x_train)
#memory_task.kill()
train_time = time.time() - t1
print('End Model Fitting\n Start Predictions...')

# Test Set Evaluation
test_size = 0.0
hit = [0.0]
MRR = [0.0]
cov = [[]]
pop = [[]]
Ks = [args.K]
cur_length = 0
cur_session = -1
last_items = []
t1 = time.time()
index_item = x_valid.columns.get_loc(args.itemid)
index_session = x_valid.columns.get_loc(args.sessionid)
train_items = model.items_ids
counter = 0
for row in x_valid.itertuples( index=False ):
	counter += 1
	if counter % 5000 == 0:
		print('Finished Prediction for ', counter, 'items.')
	session_id, item_id = row[index_session], row[index_item]
	if session_id != cur_session:
		cur_session = session_id
		last_items = []
		cur_length = 0
	
	if not item_id in last_items and item_id in train_items:
		#print(item_id, item_id in train_items)
		item_id = model.old_new[item_id]
		if len(last_items) > cur_length: #make prediction
			cur_length += 1
			test_size += 1
			# Predict the most similar items to items
			for k in range(len(Ks)):
				predictions = model.predict_next(last_items, k = Ks[k])
				# Evaluation
				rank = 0
				for predicted_item in predictions:
					if predicted_item not in cov[k]:
						cov[k].append(predicted_item)
					pop[k].append(model.freqs[predicted_item])
					rank += 1
					if predicted_item == item_id:
						hit[k] += 1.0
						MRR[k] += 1/rank
						break
		
		last_items.append(item_id)
  
#memory_task.kill()
hit[:] = [x / test_size for x in hit]
MRR[:] = [x / test_size for x in MRR]
cov[:] = [len(x) / distinct_train for x in cov]
maxi = max(model.freqs.values())
pop[:] = [np.mean(x) / maxi for x in pop]
test_time = (time.time() - t1)
print('Recall:', hit)
print ('\nMRR:', MRR)
print ('\nCoverage:', cov)
print ('\nPopularity:', pop)
print ('\ntrain_time:', train_time)
print ('\ntest_time:', test_time)
print('End Model Predictions')
```

```python colab={"base_uri": "https://localhost:8080/"} id="V9iB90SjDKCe" executionInfo={"status": "ok", "timestamp": 1641017530493, "user_tz": -330, "elapsed": 3722, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ac58a675-a8eb-41dc-9a68-9542c70347a7"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```
