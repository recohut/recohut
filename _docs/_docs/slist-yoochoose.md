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

<!-- #region id="TthNcibSQiab" -->
# SLIST on Yoochoose Preprocessed Sample Dataset
<!-- #endregion -->

<!-- #region id="YxLc0jRFV3S-" -->
## Executive summary

| | |
| --- | --- |
| Prblm Stmnt | The goal of session-based recommendation is to predict the next item(s) a user would likely choose to consume, given a sequence of previously consumed items in a session. Formally, we build a session-based model M(𝑠) that takes a session ⁍ for ⁍ as input and returns a list of top-𝑁 candidate items to be consumed as the next one ⁍. |
| Solution | Firstly, we devise two linear models focusing on different properties of sessions: (i) Session-aware Linear Item Similarity (SLIS) model aims at better handling session consistency, and (ii) Session-aware Linear Item Transition (SLIT) model focuses more on sequential dependency. With both SLIS and SLIT, we relax the constraint to incorporate repeated items and introduce a weighting scheme to take the timeliness of sessions into account. Combining these two types of models, we then suggest a unified model, namely Session-aware Item Similarity/Transition (SLIST) model, which is a generalized solution to holistically cover various properties of sessions. |
| Dataset | Yoochoose |
| Preprocessing | We discard the sessions having only one interaction and items appearing less than five times following the convention. We hold-out the sessions from the last 𝑁-days for test purposes and used the last 𝑁 days in the training set for the validation set. To evaluate session-based recommender models, we adopt the iterative revealing scheme, which iteratively exposes the item of a session to the model. Each item in the session is sequentially appended to the input of the model. Therefore, this scheme is useful for reflecting the sequential user behavior throughout a session |
| Metrics | HR, MRR, Coverage, Popularity |
| Models | SLIST
| Cluster | Python 3.x |
| Tags | LinearRecommender, SessionBasedRecommender |
<!-- #endregion -->

<!-- #region id="UYrwO2hmWLjr" -->
## Process flow

![](https://github.com/RecoHut-Stanzas/S181315/raw/main/images/process_flow_prototype_1.svg)
<!-- #endregion -->

<!-- #region id="B_x3nAE9QZ_7" -->
## Setup
<!-- #endregion -->

<!-- #region id="I1mMP5dnQbML" -->
### Imports
<!-- #endregion -->

```python id="BZTJO2V8E6IJ"
import os.path
import numpy as np
import pandas as pd
from _datetime import datetime, timezone, timedelta

from tqdm import tqdm
import collections as col
import scipy
import os
import pickle

from scipy import sparse
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix, csc_matrix, vstack
from sklearn.preprocessing import normalize
```

<!-- #region id="PP2SgjqSQhxB" -->
## Dataset
<!-- #endregion -->

<!-- #region id="5SpMxtzFQcXy" -->
### Load data

Preprocessed Yoochoose clicks 100k
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="0iiHncDp6gvs" executionInfo={"status": "ok", "timestamp": 1639118974794, "user_tz": -330, "elapsed": 2453, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a6a1504e-a5bd-4c45-e1e2-7671e4535311"
!mkdir -p prepared
!wget -O prepared/events_test.txt -q --show-progress https://github.com/RecoHut-Stanzas/S181315/raw/main/data/rsc15/prepared/yoochoose-clicks-100k_test.txt
!wget -O prepared/events_train_full.txt -q --show-progress https://github.com/RecoHut-Stanzas/S181315/raw/main/data/rsc15/prepared/yoochoose-clicks-100k_train_full.txt
!wget -O prepared/events_train_tr.txt -q --show-progress https://github.com/RecoHut-Stanzas/S181315/raw/main/data/rsc15/prepared/yoochoose-clicks-100k_train_tr.txt
!wget -O prepared/events_train_valid.txt -q --show-progress https://github.com/RecoHut-Stanzas/S181315/raw/main/data/rsc15/prepared/yoochoose-clicks-100k_train_valid.txt
```

```python id="tkzyj627E9F2"
def load_data_session( path, file, sessions_train=None, sessions_test=None, slice_num=None, train_eval=False ):
    '''
    Loads a tuple of training and test set with the given parameters. 
    Parameters
    --------
    path : string
        Base path to look in for the prepared data files
    file : string
        Prefix of  the dataset you want to use.
        "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt
    rows_train : int or None
        Number of rows to load from the training set file. 
        This option will automatically filter the test set to only retain items included in the training set.  
    rows_test : int or None
        Number of rows to load from the test set file. 
    slice_num : 
        Adds a slice index to the constructed file_path
        yoochoose-clicks-full_train_full.0.txt
    density : float
        Percentage of the sessions to randomly retain from the original data (0-1). 
        The result is cached for the execution of multiple experiments. 
    Returns
    --------
    out : tuple of pandas.DataFrame
        (train, test)
    
    '''
    
    print('START load data') 
    import time
    st = time.time()
    sc = time.perf_counter()
    
    split = ''
    if( slice_num != None and isinstance(slice_num, int ) ):
        split = '.'+str(slice_num)
    
    train_appendix = '_train_full'
    test_appendix = '_test'
    if train_eval:
        train_appendix = '_train_tr'
        test_appendix = '_train_valid'
            
    train = pd.read_csv(path + file + train_appendix +split+'.txt', sep='\t' )
    test = pd.read_csv(path + file + test_appendix +split+'.txt', sep='\t' )
        
    if( sessions_train != None ):
        keep = train.sort_values('Time', ascending=False).SessionId.unique()[:(sessions_train-1)]
        train = train[ np.in1d( train.SessionId, keep ) ]
        test = test[np.in1d(test.ItemId, train.ItemId)]
    
    if( sessions_test != None ):
        keep = test.SessionId.unique()[:(sessions_test)]
        test = test[ np.in1d( test.SessionId, keep ) ]
    
    session_lengths = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, session_lengths[ session_lengths>1 ].index)]
    
    #output
    data_start = datetime.fromtimestamp( train.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( train.Time.max(), timezone.utc )
    
    print('Loaded train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format( len(train), train.SessionId.nunique(), train.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    data_start = datetime.fromtimestamp( test.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( test.Time.max(), timezone.utc )
    
    print('Loaded test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format( len(test), test.SessionId.nunique(), test.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    check_data(train, test)
    
    print( 'END load data ', (time.perf_counter()-sc), 'c / ', (time.time()-st), 's' ) 
    
    return (train, test)
```

```python id="SioAVZHXFCP1"
def check_data( train, test ):
    
    if 'ItemId' in train.columns and 'SessionId' in train.columns:
    
        new_in_test = set( test.ItemId.unique() ) - set( train.ItemId.unique() )
        if len( new_in_test ) > 0:
            print( 'WAAAAAARRRNIIIIING: new items in test set' )
            
        session_min_train = train.groupby( 'SessionId' ).size().min()
        if session_min_train == 0:
            print( 'WAAAAAARRRNIIIIING: session length 1 in train set' )
            
        session_min_test = test.groupby( 'SessionId' ).size().min()
        if session_min_test == 0:
            print( 'WAAAAAARRRNIIIIING: session length 1 in train set' )
          
    else: 
        print( 'data check not possible due to individual column names' )
```

```python id="D-KD8eRwFQFj"
def evaluate_sessions(pr, metrics, test_data, train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time'):
    '''
    Evaluates the baselines wrt. recommendation accuracy measured by recall@N and MRR@N. Has no batch evaluation capabilities. Breaks up ties.
    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    metrics : list
        A list of metric classes providing the proper methods
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    Returns
    --------
    out :  list of tuples
        (metric_name, value)
    '''

    actions = len(test_data)
    sessions = len(test_data[session_key].unique())
    count = 0
    print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')

    import time
    sc = time.perf_counter()
    st = time.time()

    time_sum = 0
    time_sum_clock = 0
    time_count = 0

    for m in metrics:
        m.reset()

    test_data.sort_values([session_key, time_key], inplace=True)
    items_to_predict = train_data[item_key].unique()
    prev_iid, prev_sid = -1, -1
    pos = 0

    for i in tqdm(range(len(test_data))):

        # if count % 1000 == 0:
        #     print(f'eval process: {count} of  {actions} actions: {(count / actions * 100.0):.2f} % in {(time.time()-st):.2f} s')

        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]
        ts = test_data[time_key].values[i]
        if prev_sid != sid:
            prev_sid = sid
            pos = 0
        else:
            if items is not None:
                if np.in1d(iid, items):
                    items_to_predict = items
                else:
                    items_to_predict = np.hstack(([iid], items))

            crs = time.perf_counter()
            trs = time.time()

            for m in metrics:
                if hasattr(m, 'start_predict'):
                    m.start_predict(pr)

            preds = pr.predict_next(sid, prev_iid, items_to_predict, timestamp=ts)

            for m in metrics:
                if hasattr(m, 'stop_predict'):
                    m.stop_predict(pr)

            preds[np.isnan(preds)] = 0
#             preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
            preds.sort_values(ascending=False, inplace=True)

            time_sum_clock += time.perf_counter()-crs
            time_sum += time.time()-trs
            time_count += 1

            for m in metrics:
                if hasattr(m, 'add'):
                    m.add(preds, iid, for_item=prev_iid, session=sid, position=pos)

            pos += 1

        prev_iid = iid

        count += 1

    print('\nEND evaluation in ', (time.perf_counter()-sc), 'c / ', (time.time()-st), 's')
    print('    avg rt ', (time_sum/time_count), 's / ', (time_sum_clock/time_count), 'c')
    print('    time count ', (time_count), 'count/', (time_sum), ' sum')

    res = []
    for m in metrics:
        if type(m).__name__ == 'Time_usage_testing':
            res.append(m.result_second(time_sum_clock/time_count))
            res.append(m.result_cpu(time_sum_clock / time_count))
        else:
            res.append(m.result())

    return res
```

<!-- #region id="QalAVbeRRMCr" -->
## Metrics
<!-- #endregion -->

```python id="fUVIGWZz2xPu"
class MRR: 
    '''
    MRR( length=20 )
    Used to iteratively calculate the average mean reciprocal rank for a result list with the defined length. 
    Parameters
    -----------
    length : int
        MRR@length
    '''
    def __init__(self, length=20):
        self.length = length;
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.pos=0
        
        self.test_popbin = {}
        self.pos_popbin = {}
        
        self.test_position = {}
        self.pos_position = {}
    
    def skip(self, for_item = 0, session = -1 ):
        pass
        
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None ):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        res = result[:self.length]
        
        self.test += 1
        
        if pop_bin is not None:
            if pop_bin not in self.test_popbin:
                self.test_popbin[pop_bin] = 0
                self.pos_popbin[pop_bin] = 0
            self.test_popbin[pop_bin] += 1
        
        if position is not None:
            if position not in self.test_position:
                self.test_position[position] = 0
                self.pos_position[position] = 0
            self.test_position[position] += 1
        
        if next_item in res.index:
            rank = res.index.get_loc( next_item )+1
            self.pos += ( 1.0/rank )
            
            if pop_bin is not None:
                self.pos_popbin[pop_bin] += ( 1.0/rank )
            
            if position is not None:
                self.pos_position[position] += ( 1.0/rank )
                   
        
        
    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.
        
        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i=0
        for part, series in result.iteritems(): 
            result.sort_values( part, ascending=False, inplace=True )
            self.add( series, next_item[i] )
            i += 1
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("MRR@" + str(self.length) + ": "), (self.pos/self.test), self.result_pop_bin(), self.result_position()
    
    def result_pop_bin(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Bin: ;'
        for key in self.test_popbin:
            csv += str(key) + ';'
        csv += '\nPrecision@' + str(self.length) + ': ;'
        for key in self.test_popbin:
            csv += str( self.pos_popbin[key] / self.test_popbin[key] ) + ';'
            
        return csv
    
    def result_position(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Pos: ;'
        for key in self.test_position:
            csv += str(key) + ';'
        csv += '\nPrecision@' + str(self.length) + ': ;'
        for key in self.test_position:
            csv += str( self.pos_position[key] / self.test_position[key] ) + ';'
            
        return csv
```

```python id="NL1uEgE6FUg1"
class HitRate: 
    '''
    MRR( length=20 )
    Used to iteratively calculate the average hit rate for a result list with the defined length. 
    Parameters
    -----------
    length : int
        HitRate@length
    '''
    
    def __init__(self, length=20):
        self.length = length;
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.hit=0
        
        self.test_popbin = {}
        self.hit_popbin = {}
        
        self.test_position = {}
        self.hit_position = {}
        
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        
        self.test += 1
         
        if pop_bin is not None:
            if pop_bin not in self.test_popbin:
                self.test_popbin[pop_bin] = 0
                self.hit_popbin[pop_bin] = 0
            self.test_popbin[pop_bin] += 1
        
        if position is not None:
            if position not in self.test_position:
                self.test_position[position] = 0
                self.hit_position[position] = 0
            self.test_position[position] += 1
                
        if next_item in result[:self.length].index:
            self.hit += 1
            
            if pop_bin is not None:
                self.hit_popbin[pop_bin] += 1
            
            if position is not None:
                self.hit_position[position] += 1
            
        
        
    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.
        
        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i=0
        for part, series in result.iteritems(): 
            result.sort_values( part, ascending=False, inplace=True )
            self.add( series, next_item[i] )
            i += 1
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("HitRate@" + str(self.length) + ": "), (self.hit/self.test), self.result_pop_bin(), self.result_position()

    
    def result_pop_bin(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Bin: ;'
        for key in self.test_popbin:
            csv += str(key) + ';'
        csv += '\nHitRate@' + str(self.length) + ': ;'
        for key in self.test_popbin:
            csv += str( self.hit_popbin[key] / self.test_popbin[key] ) + ';'
            
        return csv
    
    def result_position(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        csv = ''
        csv += 'Pos: ;'
        for key in self.test_position:
            csv += str(key) + ';'
        csv += '\nHitRate@' + str(self.length) + ': ;'
        for key in self.test_position:
            csv += str( self.hit_position[key] / self.test_position[key] ) + ';'
            
        return csv
```

```python id="QDlDFjyE28eB"
class Coverage:
    '''
    Coverage( length=20 )
    Used to iteratively calculate the coverage of an algorithm regarding the item space. 
    Parameters
    -----------
    length : int
        Coverage@length
    '''
    
    item_key = 'ItemId'
    
    def __init__(self, length=20):
        self.num_items = 0
        self.length = length
        self.time = 0;
        
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''        
        self.coverage_set = set()
        self.items = set(train[self.item_key].unique()) # keep track of full item list
        self.num_items = len( train[self.item_key].unique() )
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.coverage_set = set()
        return
    
    def skip(self, for_item = 0, session = -1 ):
        pass
    
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        recs = result[:self.length]
        items = recs.index.unique()
        self.coverage_set.update( items )
        self.items.update( items ) # update items
        self.num_items = len( self.items )
        
    def add_multiple(self, result, next_items, for_item=0, session=0, position=None):   
        self.add(result, next_items[0], for_item, session)
        
    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.
        
        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i=0
        for part, series in result.iteritems(): 
            result.sort_values( part, ascending=False, inplace=True )
            self.add( series, next_item[i] )
            i += 1
        
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("Coverage@" + str(self.length) + ": "), ( len(self.coverage_set) / self.num_items )
```

```python id="_cvJCNlb3BUx"
class Popularity:
    '''
    Popularity( length=20 )
    Used to iteratively calculate the average overall popularity of an algorithm's recommendations. 
    Parameters
    -----------
    length : int
        Coverage@length
    '''
    
    session_key = 'SessionId'
    item_key    = 'ItemId'
    
    def __init__(self, length=20):
        self.length = length;
        self.sum = 0
        self.tests = 0
    
    def init(self, train):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        self.train_actions = len( train.index )
        #group the data by the itemIds
        grp = train.groupby(self.item_key)
        #count the occurence of every itemid in the trainingdataset
        self.pop_scores = grp.size()
        #sort it according to the  score
        self.pop_scores.sort_values(ascending=False, inplace=True)
        #normalize
        self.pop_scores = self.pop_scores / self.pop_scores[:1].values[0]
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.tests=0;
        self.sum=0
     
    def skip(self, for_item = 0, session = -1 ):
        pass 
        
    def add(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        #only keep the k- first predictions
        recs = result[:self.length]
        #take the unique values out of those top scorers
        items = recs.index.unique()
                
        self.sum += ( self.pop_scores[ items ].sum() / len( items ) )
        self.tests += 1
    
    def add_multiple(self, result, next_items, for_item=0, session=0, position=None):   
        self.add(result, next_items[0], for_item, session)
    
    def add_batch(self, result, next_item):
        '''
        Update the metric with a result set and the correct next item.
        
        Parameters
        --------
        result: pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        next_item: Array of correct next items
        '''
        i=0
        for part, series in result.iteritems(): 
            result.sort_values( part, ascending=False, inplace=True )
            self.add( series, next_item[i] )
            i += 1
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("Popularity@" + str( self.length ) + ": "), ( self.sum / self.tests )
```

<!-- #region id="F5cC3PEPRPpE" -->
## SLIST model
<!-- #endregion -->

```python id="FnPMwD7_32BW"
class SLIST:
    '''
    SLIST(reg=10)
    Parameters
    --------
    Will be added
    --------
    '''

    # Must need
    def __init__(self, reg=10, alpha=0.5, session_weight=-1, train_weight=-1, predict_weight=-1,
                 direction='part', normalize='l1', epsilon=10.0, session_key='SessionId', item_key='ItemId',
                 verbose=False):
        self.reg = reg
        self.normalize = normalize
        self.epsilon = epsilon
        self.alpha = alpha
        self.direction = direction 
        self.train_weight = float(train_weight)
        self.predict_weight = float(predict_weight)
        self.session_weight = session_weight*24*3600

        self.session_key = session_key
        self.item_key = item_key

        # updated while recommending
        self.session = -1
        self.session_items = []
        
        self.verbose = verbose

    # Must need
    def fit(self, data, test=None):
        '''
        Trains the predictor.
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        # make new session ids(1 ~ #sessions)
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        self.sessionidmap = pd.Series(data=np.arange(self.n_sessions), index=sessionids)
        data = pd.merge(data, pd.DataFrame({self.session_key: sessionids, 'SessionIdx': self.sessionidmap[sessionids].values}), on=self.session_key, how='inner')

        # make new item ids(1 ~ #items)
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}), on=self.item_key, how='inner')

        # ||X - XB||
        input1, target1, row_weight1 = self.make_train_matrix(data, weight_by='SLIS')
        # ||Y - ZB||
        input2, target2, row_weight2 = self.make_train_matrix(data, weight_by='SLIT')
        # alpha * ||X - XB|| + (1-alpha) * ||Y - ZB||
        input1.data = np.sqrt(self.alpha) * input1.data
        target1.data = np.sqrt(self.alpha) * target1.data
        input2.data = np.sqrt(1-self.alpha) * input2.data
        target2.data = np.sqrt(1-self.alpha) * target2.data

        input_matrix = vstack([input1, input2])
        target_matrix = vstack([target1, target2])
        w2 = row_weight1 + row_weight2  # list

        # P = (X^T * X + λI)^−1 = (G + λI)^−1
        # (A+B)^-1 = A^-1 - A^-1 * B * (A+B)^-1
        # P =  G
        W2 = sparse.diags(w2, dtype=np.float32)
        G = input_matrix.transpose().dot(W2).dot(input_matrix).toarray()
        if self.verbose:
            print(f"G is made. Sparsity:{(1 - np.count_nonzero(G)/(self.n_items**2))*100}%")

        P = np.linalg.inv(G + np.identity(self.n_items, dtype=np.float32) * self.reg)
        if self.verbose:
            print("P is made")
        del G

        if self.alpha == 1:
            C = -P @ (input_matrix.transpose().dot(W2).dot(input_matrix-target_matrix).toarray())

            mu = np.zeros(self.n_items)
            mu += self.reg
            mu_nonzero_idx = np.where(1 - np.diag(P)*self.reg + np.diag(C) >= self.epsilon)
            mu[mu_nonzero_idx] = (np.diag(1 - self.epsilon + C) / np.diag(P))[mu_nonzero_idx]

            # B = I - Pλ + C
            self.enc_w = np.identity(self.n_items, dtype=np.float32) - P @ np.diag(mu) + C
            if self.verbose:
                print("weight matrix is made")
        else:
            self.enc_w = P @ input_matrix.transpose().dot(W2).dot(target_matrix).toarray()


    def make_train_matrix(self, data, weight_by='SLIT'):
        input_row = []
        target_row = []
        input_col = []
        target_col = []
        input_data = []
        target_data = []

        maxtime = data.Time.max()
        w2 = []
        sessionlengthmap = data['SessionIdx'].value_counts(sort=False)
        rowid = -1
        
        directory = os.path.dirname('./data_ckpt/')
        if not os.path.exists(directory):
            os.makedirs(directory)

        if weight_by == 'SLIT':
            if os.path.exists(f'./data_ckpt/{self.n_sessions}_{self.n_items}_{self.direction}_SLIT.p'):
                with open(f'./data_ckpt/{self.n_sessions}_{self.n_items}_{self.direction}_SLIT.p','rb') as f:
                    input_row, input_col, input_data, target_row, target_col, target_data, w2 = pickle.load(f)
            else:
                for sid, session in tqdm(data.groupby(['SessionIdx']), desc=weight_by):
                    slen = sessionlengthmap[sid]
                    # sessionitems = session['ItemIdx'].tolist() # sorted by itemid
                    sessionitems = session.sort_values(['Time'])['ItemIdx'].tolist()  # sorted by time
                    slen = len(sessionitems)
                    if slen <= 1:
                        continue
                    stime = session['Time'].max()
                    w2 += [stime-maxtime] * (slen-1)
                    for t in range(slen-1):
                        rowid += 1
                        # input matrix
                        if self.direction == 'part':
                            input_row += [rowid] * (t+1)
                            input_col += sessionitems[:t+1]
                            for s in range(t+1):
                                input_data.append(-abs(t-s))
                            target_row += [rowid] * (slen - (t+1))
                            target_col += sessionitems[t+1:]
                            for s in range(t+1, slen):
                                target_data.append(-abs((t+1)-s))
                        elif self.direction == 'all':
                            input_row += [rowid] * slen
                            input_col += sessionitems
                            for s in range(slen):
                                input_data.append(-abs(t-s))
                            target_row += [rowid] * slen
                            target_col += sessionitems
                            for s in range(slen):
                                target_data.append(-abs((t+1)-s))
                        elif self.direction == 'sr':
                            input_row += [rowid]
                            input_col += [sessionitems[t]]
                            input_data.append(0)
                            target_row += [rowid] * (slen - (t+1))
                            target_col += sessionitems[t+1:]
                            for s in range(t+1, slen):
                                target_data.append(-abs((t+1)-s))
                        else:
                            raise ("You have to choose right 'direction'!")
                with open(f'./data_ckpt/{self.n_sessions}_{self.n_items}_{self.direction}_SLIT.p','wb') as f:
                    pickle.dump([input_row, input_col, input_data, target_row, target_col, target_data, w2], f, protocol=4)
            input_data = list(np.exp(np.array(input_data) / self.train_weight))
            target_data = list(np.exp(np.array(target_data) / self.train_weight))
        elif weight_by == 'SLIS':
            if os.path.exists(f'./data_ckpt/{self.n_sessions}_{self.n_items}_SLIS.p'):
                with open(f'./data_ckpt/{self.n_sessions}_{self.n_items}_SLIS.p','rb') as f:
                    input_row, input_col, input_data, target_row, target_col, target_data, w2 = pickle.load(f)
            else:
                for sid, session in tqdm(data.groupby(['SessionIdx']), desc=weight_by):
                    rowid += 1
                    slen = sessionlengthmap[sid]
                    sessionitems = session['ItemIdx'].tolist()
                    stime = session['Time'].max()
                    w2.append(stime-maxtime)
                    input_row += [rowid] * slen
                    input_col += sessionitems

                target_row = input_row
                target_col = input_col
                input_data = np.ones_like(input_row)
                target_data = np.ones_like(target_row)
                
                with open(f'./data_ckpt/{self.n_sessions}_{self.n_items}_SLIS.p','wb') as f:
                    pickle.dump([input_row, input_col, input_data, target_row, target_col, target_data, w2], f, protocol=4)
        else:
            raise ("You have to choose right 'weight_by'!")

        # Use train_weight or not
        input_data = input_data if self.train_weight > 0 else list(np.ones_like(input_data))
        target_data = target_data if self.train_weight > 0 else list(np.ones_like(target_data))

        # Use session_weight or not
        w2 = list(np.exp(np.array(w2) / self.session_weight))
        w2 = w2 if self.session_weight > 0 else list(np.ones_like(w2))

        # Make sparse_matrix
        input_matrix = csr_matrix((input_data, (input_row, input_col)), shape=(max(input_row)+1, self.n_items), dtype=np.float32)
        target_matrix = csr_matrix((target_data, (target_row, target_col)), shape=input_matrix.shape, dtype=np.float32)
        if self.verbose:
            print(f"[{weight_by}]sparse matrix {input_matrix.shape} is made.  Sparsity:{(1 - input_matrix.count_nonzero()/(self.n_items*input_matrix.shape[0]))*100}%")


        if weight_by == 'SLIT':
            pass
        elif weight_by == 'SLIS':
            # Value of repeated items --> 1
            input_matrix.data = np.ones_like(input_matrix.data)
            target_matrix.data = np.ones_like(target_matrix.data)

        # Normalization
        if self.normalize == 'l1':
            input_matrix = normalize(input_matrix, 'l1')
        elif self.normalize == 'l2':
            input_matrix = normalize(input_matrix, 'l2')
        else:
            pass

        return input_matrix, target_matrix, w2

    # 필수

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        '''
        # new session
        if session_id != self.session:
            self.session_items = []
            self.session = session_id
            self.session_times = []

        if type == 'view':
            if input_item_id in self.itemidmap.index:
                self.session_items.append(input_item_id)
                self.session_times.append(timestamp)

        # item id transfomration
        session_items_new_id = self.itemidmap[self.session_items].values
        predict_for_item_ids_new_id = self.itemidmap[predict_for_item_ids].values
        
        if session_items_new_id.shape[0] == 0:
            skip = True

        if skip:
            return pd.Series(data=0, index=predict_for_item_ids)

        W_test = np.ones_like(self.session_items, dtype=np.float32)
        W_test = self.enc_w[session_items_new_id[-1], session_items_new_id]
        for i in range(len(W_test)):
            W_test[i] = np.exp(-abs(i+1-len(W_test))/self.predict_weight)

        W_test = W_test if self.predict_weight > 0 else np.ones_like(W_test)
        W_test = W_test.reshape(-1, 1)

        # [session_items, num_items]
        preds = self.enc_w[session_items_new_id] * W_test
        # [num_items]
        preds = np.sum(preds, axis=0)
        preds = preds[predict_for_item_ids_new_id]

        series = pd.Series(data=preds, index=predict_for_item_ids)

        series = series / series.max()
        
        # remove current item from series of prediction
        # series.drop(labels=[input_item_id])
        
        return series

    # 필수
    def clear(self):
        self.enc_w = {}
```

<!-- #region id="qtKntJplRUu9" -->
## Main
<!-- #endregion -->

```python id="yOMQysGb3Iwp"
'''
FILE PARAMETERS
'''
PATH_PROCESSED = './prepared/'
FILE = 'events'
```

```python id="RYIZjliz3LUh"
'''
MODEL HYPERPARAMETER TUNING
'''
alpha = 0.2 #[0.2, 0.4, 0.6, 0.8] 
direction = 'all' # sr / part / all
reg = 10
train_weight = 1 #0.5 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
predict_weight = 1 #4 #[0.125, 0.25, 0.5, 1, 2, 4, 8]
session_weight = 1 #256 #[1, 2, 4, 8, 16, 32, 64, 128, 256]
```

```python colab={"base_uri": "https://localhost:8080/"} id="4a4mvCbi3TAy" outputId="0d221f82-da35-42aa-e0b1-91fa3f8051cd" executionInfo={"status": "ok", "timestamp": 1639119017665, "user_tz": -330, "elapsed": 17591, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
# Training
train, val = load_data_session(PATH_PROCESSED, FILE, train_eval=True)
model = SLIST(alpha=alpha, direction=direction, reg=reg, train_weight=train_weight, 
              predict_weight=predict_weight, session_weight=session_weight)
model.fit(train, val)

mrr = MRR(length=100)
hr = HitRate()
pop = Popularity()
pop.init(train)
cov = Coverage()
cov.init(train)
```

```python colab={"base_uri": "https://localhost:8080/"} id="n94vXWR67O36" executionInfo={"status": "ok", "timestamp": 1639119130459, "user_tz": -330, "elapsed": 62776, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5d2f7bbe-e29f-46f5-9929-16a28702ef9d"
result = evaluate_sessions(model, [mrr, hr, pop, cov], val, train)
```

```python id="SpsUr_403z0U" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1639119133944, "user_tz": -330, "elapsed": 555, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b51ed462-d70f-41df-d262-3bc6927f15ff"
result
```

<!-- #region id="ogypTsIWHEyR" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HY5KNqOJHEyT" executionInfo={"status": "ok", "timestamp": 1639119146044, "user_tz": -330, "elapsed": 3786, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ed8e9ba0-c8c1-4072-8e22-f40f8efb340e"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="3RXL1ys5HEyU" -->
---
<!-- #endregion -->

<!-- #region id="E8qwaCPdHEyU" -->
**END**
<!-- #endregion -->
