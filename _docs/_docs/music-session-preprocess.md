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

<!-- #region id="KR2HTM55eBXn" -->
# Preprocessing Music Session Dataset
<!-- #endregion -->

```python id="-HJaBLCJPGow"
import numpy as np
import pandas as pd
from datetime import timezone, datetime, timedelta
import random
import time
import sys
```

```python id="DQdMXuTlEERB"
'''
preprocessing method ["info","org","days_test","slice"]
    info: just load and show info
    org: from gru4rec (last day => test set)
    days_test: adapted from gru4rec (last N days => test set)
    slice: new (create multiple train-test-combinations with a sliding window approach  
'''
# METHOD = "slice"
METHOD = input('Preprocessing method (info/org/days_test/slice):') or 'slice'
assert(METHOD in 'info/org/days_test/slice'.split('/')), 'Invalid Preprocessing method.'

'''
data config (all methods) // change dataset here
'''
#30music/nowplaying/aotm
# PATH = './30music/raw/' 
# PATH_PROCESSED = './30music/slices/'
DATASET_CODE = input('Dataset (30music/nowplaying/aotm):') or '30music'
assert(DATASET_CODE in '30music/nowplaying/aotm'.split('/')), 'Invalid dataset.'

PATH = './{}/raw/'.format(DATASET_CODE)
PATH_PROCESSED = './{}/slices/'.format(DATASET_CODE)
_filenames = {'30music':'30music-200ks','nowplaying':'nowplaying','aotm':'playlists-aotm'}
FILE = _filenames[DATASET_CODE]

'''
filtering config (all methods)
'''
#filtering config (all methods)
MIN_SESSION_LENGTH = 5
MIN_ITEM_SUPPORT = 2

'''
days test default config
'''
DAYS_FOR_TEST = 4

'''
slicing default config
'''
NUM_SLICES = 5 #offset in days from the first date in the data set
DAYS_OFFSET = 0 #number of days the training start date is shifted after creating one slice
DAYS_SHIFT = 60
#each slice consists of...
DAYS_TRAIN = 90
DAYS_TEST = 5
```

```python id="nlavuPjUD_Zm"
if DATASET_CODE=='30music':
    !wget -q --show-progress https://github.com/RecoHut-Datasets/30music/raw/v1/30music.zip
    !unzip 30music.zip
elif DATASET_CODE=='nowplaying':
    !wget -q --show-progress https://github.com/RecoHut-Datasets/nowplaying/raw/v2/nowplaying.zip
    !unzip nowplaying.zip
elif DATASET_CODE=='aotm':
    !wget -q --show-progress https://github.com/RecoHut-Datasets/aotm/raw/v1/aotm.zip
    !unzip aotm.zip
```

```python id="AXRx3TyJZsUr"
def load_data( file ) : 
    
    #load csv
    data = pd.read_csv( file+'.csv', sep='\t' )
    
    #data.sort_values( by=['Time'], inplace=True )
    #data['SessionId'] = data.groupby( [data.SessionId] ).grouper.group_info[0]
    
    data.sort_values( by=['SessionId','Time'], inplace=True )
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data
```

```python id="R4OafNnCAyKR"
def filter_data( data, min_item_support, min_session_length ) : 
    
    #filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>1 ].index)]
    
    #filter item support
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[ item_supports>= min_item_support ].index)]
    
    #filter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths>= min_session_length ].index)]
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data;
```

```python id="OxRuh9HUA-gV"
def split_data_org( data, output_file ) :
    
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_test = session_max_times[session_max_times >= tmax-86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)
    
    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_valid = session_max_times[session_max_times >= tmax-86400].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength>=2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
    train_tr.to_csv( output_file + '_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
    valid.to_csv( output_file + '_train_valid.txt', sep='\t', index=False)
```

```python id="_aTk3ux_BCX-"
def split_data( data, output_file, days_test ) :
    
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    test_from = data_end - timedelta( days_test )
    
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[ session_max_times < test_from.timestamp() ].index
    session_test = session_max_times[ session_max_times >= test_from.timestamp() ].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(output_file + '_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(output_file + '_test.txt', sep='\t', index=False)
```

```python id="p6OAJHphBE4N"
def slice_data( data, output_file, num_slices, days_offset, days_shift, days_train, days_test ): 
    
    for slice_id in range( 0, num_slices ) :
        split_data_slice( data, output_file, slice_id, days_offset+(slice_id*days_shift), days_train, days_test )
```

```python id="Lgya17MmBHWJ"
def split_data_slice( data, output_file, slice_id, days_offset, days_train, days_test ) :
    
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(), data_end.isoformat() ) )
    
    
    start = datetime.fromtimestamp( data.Time.min(), timezone.utc ) + timedelta( days_offset ) 
    middle =  start + timedelta( days_train )
    end =  middle + timedelta( days_test )
    
    #prefilter the timespan
    session_max_times = data.groupby('SessionId').Time.max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(data.SessionId, greater_start.intersection( lower_end ))]
    
    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {}'.
          format( slice_id, len(data_filtered), data_filtered.SessionId.nunique(), data_filtered.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat(), end.date().isoformat() ) )
    
    #split to train and test
    session_max_times = data_filtered.groupby('SessionId').Time.max()
    sessions_train = session_max_times[session_max_times < middle.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle.timestamp()].index
    
    train = data[np.in1d(data.SessionId, sessions_train)]
    
    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format( slice_id, len(train), train.SessionId.nunique(), train.ItemId.nunique(), start.date().isoformat(), middle.date().isoformat() ) )
    
    train.to_csv(output_file + '_train_full.'+str(slice_id)+'.txt', sep='\t', index=False)
    
    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    
    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format( slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle.date().isoformat(), end.date().isoformat() ) )
    
    test.to_csv(output_file + '_test.'+str(slice_id)+'.txt', sep='\t', index=False)
```

```python id="sku1JVzKChMH"
#preprocessing from original gru4rec
def preprocess_org( path=PATH, file=FILE, path_proc=PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
#    data = load_data( path+file )
    #for listening logs
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data_org( data, path_proc+file )
```

```python id="LZN_OM5xCp6q"
#preprocessing adapted from original gru4rec
def preprocess_days_test( path=PATH, file=FILE, path_proc=PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST ):
    
#    data = load_data( path+file )
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data( data, path_proc+file, days_test )
```

```python id="FjoIiqitCsOP"
#preprocessing to create data slices with a sliding window
def preprocess_slices( path=PATH, file=FILE, path_proc=PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                       num_slices = NUM_SLICES, days_offset = DAYS_OFFSET, days_shift = DAYS_SHIFT, days_train = DAYS_TRAIN, days_test=DAYS_TEST ):
    
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    slice_data( data, path_proc+file, num_slices, days_offset, days_shift, days_train, days_test )
```

```python id="ShHQDbECCtFa"
#just load and show info
def preprocess_info( path=PATH, file=FILE, path_proc=PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )     
```

```python colab={"base_uri": "https://localhost:8080/"} id="WJQVmNwlBNy9" executionInfo={"status": "ok", "timestamp": 1638632213241, "user_tz": -330, "elapsed": 18991, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fe9ba43b-3e5a-473e-8091-593beb1d3b37"
if __name__ == '__main__':
    '''
    Run the preprocessing configured above.
    '''
    
    print( "START preprocessing ", METHOD )
    sc, st = time.time(), time.time()
    
    if METHOD == "info":
        preprocess_info( PATH, FILE, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH )
    
    elif METHOD == "org":
        preprocess_org( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH )
        
    elif METHOD == "days_test":
        preprocess_days_test( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, DAYS_FOR_TEST )
    
    elif METHOD == "slice":
        preprocess_slices( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, NUM_SLICES, DAYS_OFFSET, DAYS_SHIFT, DAYS_TRAIN, DAYS_TEST )
    else: 
        print( "Invalid method ", METHOD )
        
    print( "END preproccessing ", (time.time() - sc), "c ", (time.time() - st), "s" )
```

<!-- #region id="g-iyxuibDWgT" -->
---
<!-- #endregion -->

```python id="xgr27_gWDWgV"
# !apt-get -qq install tree
# !rm -r sample_data
```

```python colab={"base_uri": "https://localhost:8080/"} id="9ivY8TNnDWgW" executionInfo={"status": "ok", "timestamp": 1638631606270, "user_tz": -330, "elapsed": 519, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7a68c7f1-980b-4a95-c9d8-7fab91d9efcd"
# !tree -h --du .
```

```python colab={"base_uri": "https://localhost:8080/"} id="bKMlo1SfDWgX" executionInfo={"status": "ok", "timestamp": 1638631625923, "user_tz": -330, "elapsed": 3698, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0ff5e171-0e98-4bef-e126-db78765a1f78"
# !pip install -q watermark
# %reload_ext watermark
# %watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="GWg5v7TbDWgY" -->
---
<!-- #endregion -->

<!-- #region id="yPjQ9oQtDWgZ" -->
**END**
<!-- #endregion -->
