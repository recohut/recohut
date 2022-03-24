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

<!-- #region id="0-C5ulLAspbz" -->
# Predictability limits in session-based next item recommendation on Retailrocket data
<!-- #endregion -->

<!-- #region id="IYkBeq6Rr6jJ" -->
Estimate the predictability limits due to randomness and due to algorithm design in some methods of session-based recommendation.
<!-- #endregion -->

<!-- #region id="KR2HTM55eBXn" -->
## Preprocessing RetailRocket Session Dataset
<!-- #endregion -->

```python id="-HJaBLCJPGow"
import numpy as np
import pandas as pd
from datetime import timezone, datetime, timedelta
import time
```

```python id="DQdMXuTlEERB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1639127826324, "user_tz": -330, "elapsed": 5817, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="69200791-4f08-407d-9edc-7110fae79257"
'''
preprocessing method ["info","org","org_min_date","days_test","slice","buys"]
    info: just load and show info
    org: from gru4rec (last day => test set)
    org_min_date: from gru4rec (last day => test set) but from a minimal date onwards
    days_test: adapted from gru4rec (last N days => test set)
    slice: new (create multiple train-test-combinations with a sliding window approach  
    buys: load buys and safe file to prepared
'''
# METHOD = "slice"
METHOD = input('Preprocessing method (info/org/org_min_date/days_test/slice/buys):') or 'slice'
assert(METHOD in 'info/org/org_min_date/days_test/slice/buys'.split('/')), 'Invalid Preprocessing method.'

'''
data config (all methods)
'''
PATH = './retailrocket/'
PATH_PROCESSED = './retailrocket/slices/'
FILE = 'events'

'''
org_min_date config
'''
MIN_DATE = '2015-09-02'

'''
filtering config (all methods)
'''
SESSION_LENGTH = 30 * 60 #30 minutes
MIN_SESSION_LENGTH = 2
MIN_ITEM_SUPPORT = 5
MIN_DATE = '2014-04-01'

'''
days test default config
'''
DAYS_TEST = 2

'''
slicing default config
'''
NUM_SLICES = 5 #offset in days from the first date in the data set
DAYS_OFFSET = 0 #number of days the training start date is shifted after creating one slice
DAYS_SHIFT = 27
#each slice consists of...
DAYS_TRAIN = 25
DAYS_TEST = 2
```

```python id="nlavuPjUD_Zm" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1639127836976, "user_tz": -330, "elapsed": 2695, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8dff44d5-6049-47da-e3b3-bad76059eecf"
!wget -q --show-progress https://github.com/RecoHut-Datasets/retail_rocket/raw/v2/retailrocket.zip
!unzip retailrocket.zip
!mkdir retailrocket/slices
```

```python id="psEwAk-wbgBn"
#preprocessing from original gru4rec
def preprocess_org( path=PATH, file=FILE, path_proc=PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data_org( data, path_proc+file )

#preprocessing from original gru4rec but from a certain point in time
def preprocess_org_min_date( path=PATH, file=FILE, path_proc=PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, min_date=MIN_DATE ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    data = filter_min_date( data, min_date )
    split_data_org( data, path_proc+file )

#preprocessing adapted from original gru4rec
def preprocess_days_test( path=PATH, file=FILE, path_proc=PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    split_data( data, path_proc+file, days_test )

#preprocessing from original gru4rec but from a certain point in time
def preprocess_days_test_min_date( path=PATH, file=FILE, path_proc=PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH, days_test=DAYS_TEST, min_date=MIN_DATE ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    data = filter_min_date( data, min_date )
    split_data( data, path_proc+file, days_test )

#preprocessing to create data slices with a sliding window
def preprocess_slices( path=PATH, file=FILE, path_proc=PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH,
                       num_slices = NUM_SLICES, days_offset = DAYS_OFFSET, days_shift = DAYS_SHIFT, days_train = DAYS_TRAIN, days_test=DAYS_TEST ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    slice_data( data, path_proc+file, num_slices, days_offset, days_shift, days_train, days_test )
    
#just load and show info
def preprocess_info( path=PATH, file=FILE, path_proc=PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    
def preprocess_save( path=PATH, file=FILE, path_proc=PATH_PROCESSED, min_item_support=MIN_ITEM_SUPPORT, min_session_length=MIN_SESSION_LENGTH ):
    
    data, buys = load_data( path+file )
    data = filter_data( data, min_item_support, min_session_length )
    data.to_csv(path_proc + file + '_preprocessed.txt', sep='\t', index=False)
    
#preprocessing to create a file with buy actions
def preprocess_buys( path=PATH, file=FILE, path_proc=PATH_PROCESSED ): 
    data, buys = load_data( path+file )
    store_buys(buys, path_proc+file)
    
def load_data( file ) : 
    
    #load csv
    data = pd.read_csv( file+'.csv', sep=',', header=0, usecols=[0,1,2,3], dtype={0:np.int64, 1:np.int32, 2:str, 3:np.int32})
    #specify header names
    data.columns = ['Time','UserId','Type','ItemId']
    data['Time'] = (data.Time / 1000).astype( int )
    
    data.sort_values( ['UserId','Time'], ascending=True, inplace=True )
    
    #sessionize    
    data['TimeTmp'] = pd.to_datetime(data.Time, unit='s')
    
    data.sort_values( ['UserId','TimeTmp'], ascending=True, inplace=True )
#     users = data.groupby('UserId')
    
    data['TimeShift'] = data['TimeTmp'].shift(1)
    data['TimeDiff'] = (data['TimeTmp'] - data['TimeShift']).dt.total_seconds().abs()
    data['SessionIdTmp'] = (data['TimeDiff'] > SESSION_LENGTH).astype( int )
    data['SessionId'] = data['SessionIdTmp'].cumsum( skipna=False )
    del data['SessionIdTmp'], data['TimeShift'], data['TimeDiff']
    
    
    data.sort_values( ['SessionId','Time'], ascending=True, inplace=True )
    
    cart = data[data.Type == 'addtocart']
    data = data[data.Type == 'view']
    del data['Type']
    
    print(data)
    
    #output
    
    print( data.Time.min() )
    print( data.Time.max() )
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    del data['TimeTmp']
    
    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data, cart;


def filter_data( data, min_item_support, min_session_length ) : 
    
    #y?
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

def filter_min_date( data, min_date='2014-04-01' ) :
    
    min_datetime = datetime.strptime(min_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
    
    #filter
    session_max_times = data.groupby('SessionId').Time.max()
    session_keep = session_max_times[ session_max_times > min_datetime.timestamp() ].index
    
    data = data[ np.in1d(data.SessionId, session_keep) ]
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    return data;



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
    
    
    
def slice_data( data, output_file, num_slices, days_offset, days_shift, days_train, days_test ): 
    
    for slice_id in range( 0, num_slices ) :
        split_data_slice( data, output_file, slice_id, days_offset+(slice_id*days_shift), days_train, days_test )

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


def store_buys( buys, target ):
    buys.to_csv( target + '_buys.txt', sep='\t', index=False )
```

```python colab={"base_uri": "https://localhost:8080/"} id="WJQVmNwlBNy9" executionInfo={"status": "ok", "timestamp": 1639127858592, "user_tz": -330, "elapsed": 17550, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="84800b16-e522-45a2-919a-b41db43971cc"
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
     
    elif METHOD == "org_min_date":
        preprocess_org_min_date( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MIN_DATE )
        
    elif METHOD == "day_test":
        preprocess_days_test( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, DAYS_TEST )
        
    elif METHOD == "day_test_min_date":
        preprocess_days_test_min_date( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, DAYS_TEST, MIN_DATE )
    
    elif METHOD == "slice":
        preprocess_slices( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, NUM_SLICES, DAYS_OFFSET, DAYS_SHIFT, DAYS_TRAIN, DAYS_TEST )
        
    elif METHOD == "buys":
        preprocess_buys( PATH, FILE, PATH_PROCESSED )
        
    elif METHOD == "save":
        preprocess_save( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH )
        
    else: 
        print( "Invalid method ", METHOD )
        
    print( "END preproccessing ", (time.time() - sc), "c ", (time.time() - st), "s" )
```

<!-- #region id="UQa-FNXIohQF" -->
## Limit using entropy rate estimation
<!-- #endregion -->

<!-- #region id="HUHpOrO-ohNY" -->
### Convert the session data into a sequence format file
<!-- #endregion -->

```python id="IrQHjidKohHi"
import time
import os.path
import numpy as np
import pandas as pd
from _datetime import timezone, datetime


def load_data( path, file, rows_train=None, rows_test=None, slice_num=None, density=1, train_eval=False ):
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
    st = time.time()
    sc = time.time()
    
    split = ''
    if( slice_num != None and isinstance(slice_num, int ) ):
        split = '.'+str(slice_num)
    
    train_appendix = '_train_full'
    test_appendix = '_test'
    if train_eval:
        train_appendix = '_train_tr'
        test_appendix = '_train_valid'
    
    density_appendix = ''
    if( density < 1 ): #create sample
        
        if not os.path.isfile( path + file + train_appendix + split + '.txt.'+str( density ) ) :
            
            train = pd.read_csv(path + file + train_appendix + split + '.txt', sep='\t', dtype={'ItemId':np.int64})
            test = pd.read_csv(path + file + test_appendix + split + '.txt', sep='\t', dtype={'ItemId':np.int64} )
            
            sessions = train.SessionId.unique() 
            drop_n = round( len(sessions) - (len(sessions) * density) )
            drop_sessions = np.random.choice(sessions, drop_n, replace=False)
            train = train[ ~train.SessionId.isin( drop_sessions ) ]
            train.to_csv( path + file + train_appendix +split+'.txt.'+str(density), sep='\t', index=False )
            
            sessions = test.SessionId.unique() 
            drop_n = round( len(sessions) - (len(sessions) * density) )
            drop_sessions = np.random.choice(sessions, drop_n, replace=False)
            test = test[ ~test.SessionId.isin( drop_sessions ) ]
            test = test[np.in1d(test.ItemId, train.ItemId)]
            test.to_csv( path + file + test_appendix +split+'.txt.'+str(density), sep='\t', index=False )
    
        density_appendix = '.'+str(density)
            
    if( rows_train == None ):
        train = pd.read_csv(path + file + train_appendix +split+'.txt'+density_appendix, sep='\t', dtype={'ItemId':np.int64})
    else:
        train = pd.read_csv(path + file + train_appendix +split+'.txt'+density_appendix, sep='\t', dtype={'ItemId':np.int64}, nrows=rows_train)
        session_lengths = train.groupby('SessionId').size()
        train = train[np.in1d(train.SessionId, session_lengths[ session_lengths>1 ].index)]     
    
    if( rows_test == None ):
        test = pd.read_csv(path + file + test_appendix +split+'.txt'+density_appendix, sep='\t', dtype={'ItemId':np.int64} )
    else :
        test = pd.read_csv(path + file + test_appendix +split+'.txt'+density_appendix, sep='\t', dtype={'ItemId':np.int64}, nrows=rows_test )
        session_lengths = test.groupby('SessionId').size()
        test = test[np.in1d(test.SessionId, session_lengths[ session_lengths>1 ].index)]
    
#     rows_train = 10000
#     train = train.tail(10000)
        
    if( rows_train != None ):
        test = test[np.in1d(test.ItemId, train.ItemId)]
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
    
    print( 'END load data ', (time.time()-sc), 'c / ', (time.time()-st), 's' )
    
    return (train, test)


def load_buys( path, file ):
    '''
    Load all buy events from the youchoose file, retains events fitting in the given test set and merges both data sets into one
    Parameters
    --------
    path : string
        Base path to look in for the prepared data files
    file : string
        Prefix of  the dataset you want to use.
        "yoochoose-clicks-full" loads yoochoose-clicks-full_train_full.txt and yoochoose-clicks-full_test.txt
        
    Returns
    --------
    out : pandas.DataFrame
        test with buys
    
    '''
    
    print('START load buys') 
    st = time.time()
    sc = time.time()
        
    #load csv
    buys = pd.read_csv(path + file + '.txt', sep='\t', dtype={'ItemId':np.int64})
        
    print( 'END load buys ', (time.time()-sc), 'c / ', (time.time()-st), 's' )
    
    return buys
```

```python id="eFg1Ml3YohE0"
def dump_sequence(data_path, file_prefix, out_fn, density=1, slic=0):
    """
        Convert training/testing slices into a sequence format
        suitable for entropy rate estimation
    """

    train, test = load_data(data_path, file_prefix,
        rows_train=None, rows_test=None, density=density,
        slice_num=slic)

    # append all
    all_data = train.append(test)

    # sort by sequence, then timestamp
    groupby = all_data.groupby("SessionId")
    with open(out_fn, "w") as f:
        for session_id, session in groupby:
            item_ids = [item_id for
                item_id in session.sort_values("Time")["ItemId"]]
            for item_id in item_ids:
                f.write("{}\n".format(item_id))
            f.write("-1\n")
```

```python colab={"base_uri": "https://localhost:8080/"} id="svYQ5fPdpU4h" executionInfo={"status": "ok", "timestamp": 1639128270265, "user_tz": -330, "elapsed": 29672, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="10cf5f56-e767-4e6a-b8d4-1fdca2543664"
data_path = './retailrocket/slices/'
file_prefix = 'events'
output_file = './retailrocket/seq/s0.txt'
d = 1 # downsample the input data (0.1 - use only 10%% of input)
s = 0 # slice number, 0-4

!mkdir ./retailrocket/seq
dump_sequence(data_path, file_prefix, output_file, d, s)
```

<!-- #region id="sKYOt42XqFPS" -->
### Entropy rate estimation
<!-- #endregion -->

```python id="Iv7VGm-_qFMY"
import numpy as np
from collections import defaultdict

def calc_entropy2(in_fn):
    """
        Entropy rate estimation for a sequence
        input: file with each sequence element (integer) on its own row
    """
    with open(in_fn) as f:
        events = [int(l.strip()) for l in f]

    # calculate Lempel-Ziv estimate of entropy
    lambda_sum = 0
    seq1 = set()                # single item sequences
    seq2 = set()                # two-item sequences
    seq3 = defaultdict(list)    # three-item sequences index

    n = len(events)
    print(in_fn, n)
    timestep = int(n / 10) + 1
    for i in range(n):
        k_max = 0
        # single item
        if events[i] in seq1:
            k_max = 1
            # two items
            if i + 1 < n and tuple(events[i:i+2]) in seq2:
                k_max = 2
                # three or more
                if i + 2 < n:
                    for subseq_start in seq3[tuple(events[i:i+3])]:
                        k = 3
                        while subseq_start + k < i and i + k < n:
                            if events[subseq_start + k] != events[i + k]:
                                break
                            k += 1
                        k_max = max(k, k_max)

        lambda_sum += (k_max + 1) # as in Xu, et al. (2019)
        #print(i, ev, k_max)

        # update index
        seq1.add(events[i])
        if i > 0:
            seq2.add(tuple(events[i-1:i+1]))
            if i > 1:
                 seq3[tuple(events[i-2:i+1])].append(i - 2)

        if i % timestep == 0 and i > 0:
            print(i, "done")

    S = (n / lambda_sum) * np.log2(n)
    print("S:", S)
    print("m (for \Pi^max equation):", len(seq1))
```

```python colab={"base_uri": "https://localhost:8080/"} id="RjW1htoSqOjZ" executionInfo={"status": "ok", "timestamp": 1639128345079, "user_tz": -330, "elapsed": 1715, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e27aafbd-ff61-435f-8f1e-1330c9e9d584"
input_file = './retailrocket/seq/s0.txt'

calc_entropy2(input_file)
```

<!-- #region id="XTIJYG-wqFFM" -->
The predictability limit can be computed using the entropy rate estimate S and the unique event count m.
<!-- #endregion -->

<!-- #region id="Pe9rmiXDqFAt" -->
## Limit for some algorithms
<!-- #endregion -->

<!-- #region id="m_VZDP4JqxXI" -->
Calculate co-occurrence of the item to predict (in a recommendation accuracy test) and the current item (given to the recommender as an input) in the training data.
<!-- #endregion -->

```python id="4NWJY9Avq9gJ"
def test_all(data_path, file_prefix, density=1, slic=[0]):
    all_stats = defaultdict(int)
    for i in slic:
        train, test = load_data(data_path, file_prefix,
            rows_train=None, rows_test=None, density=density,
            slice_num=i)
        
        s, i2s = load_sessions(train)
        print(data_path, file_prefix, i)

        stats = test_reachability(s, i2s, test)
        for k, v in stats.items():
            all_stats[k] += v
    for k, v in all_stats.items():
        print(k, v)
```

```python id="0uj-l1gArBvG"
def test_reachability(sessions, item2session, data, max_span=10):
    """Item co-occurrence in sessions"""
    stats = {"r_cnt" : 0,
        "cnt_next" : 0,
        "cnt_fwd10" : 0,
        "cnt_anywhere" : 0,
        "cnt_anywhere_sess" : 0}

    groupby = data.groupby("SessionId")
    for session_id, session in groupby:
        item_ids = [item_id for
            item_id in session.sort_values("Time")["ItemId"]]

        l = len(item_ids)
        for i in range(l - 1):
            # step 1: calculate relative to current item
            # MC cnt_next
            # SR, windowed NB cnt_fwd10
            # AR cnt_anywhere
            item_id = item_ids[i]
            target_id = item_ids[i + 1]

            next_found = 0
            fwd10_found = 0
            any_found = 0
            sess_found = 0
            seen_sessions = set()

            # loop through all sessions
            for train_sess_id in item2session[item_id]:
                seen_sessions.add(train_sess_id)
                train_sess = sessions[train_sess_id]
                last_item = None
                for i, train_item in enumerate(train_sess):
                    if train_item == target_id:
                        any_found = 1
                        sess_found = 1
                        if last_item == item_id:
                            next_found = 1
                            fwd10_found = 1
                            break
                        elif not fwd10_found and i > 1 and item_id in train_sess[max(0, i - max_span):i - 1]:
                            fwd10_found = 1
                    last_item = train_item

                if next_found:
                    break
                # otherwise need to keep searching other sessions

            # step 2: search using the remainder of the items seen so far
            # NB cnt_anywhere_sess
            if not sess_found:
                sess_so_far = set(item_ids[:i])
                for item_id in sess_so_far:
                    for train_sess_id in item2session[item_id]:
                        if train_sess_id in seen_sessions:
                            continue
                        seen_sessions.add(train_sess_id)

                        train_sess = sessions[train_sess_id]
                        last_item = None
                        for i, train_item in enumerate(train_sess):
                            if train_item == target_id:
                                sess_found = 1
                                break

            # summarize results
            stats["r_cnt"] += 1
            stats["cnt_next"] += next_found
            stats["cnt_fwd10"] += fwd10_found
            stats["cnt_anywhere"] += any_found
            stats["cnt_anywhere_sess"] += sess_found

    return stats
```

```python id="69fghpYJrDsS"
def test_forward_backward(sessions, item2session, data):
    """Statistics of whether the item to predict occurs
    before or after the current item (when co-occurring in a session)
    """
    stats = {"f_cnt" : 0,
        "cnt_bwd" : 0,
        "cnt_fwd" : 0,
        "cnt_both" : 0}

    groupby = data.groupby("SessionId")
    for session_id, session in groupby:
        item_ids = [item_id for
            item_id in session.sort_values("Time")["ItemId"]]

        l = len(item_ids)
        for i in range(l - 1):
            item_id = item_ids[i]
            target_id = item_ids[i + 1]
            if item_id == target_id:
                continue

            common_sessions = set(item2session[item_id]).intersection(
                set(item2session[target_id]))

            bwd = 0
            fwd = 0
            both = 0

            # loop through all sessions
            for train_sess_id in common_sessions:
                train_sess = sessions[train_sess_id]
                item_pos = []
                target_pos = []
                for i in range(len(train_sess)):
                    if train_sess[i] == item_id:
                        item_pos.append(i)
                    elif train_sess[i] == target_id:
                        target_pos.append(i)

                b = f = 0
                if min(target_pos) < max(item_pos):
                    b = 1
                if min(item_pos) < max(target_pos):
                    f = 1
                bwd += b
                fwd += f
                if b == f:
                    both += 1

            # summarize results
            stats["f_cnt"] += len(common_sessions)
            stats["cnt_bwd"] += bwd
            stats["cnt_fwd"] += fwd
            stats["cnt_both"] += both

    return stats
```

```python id="P0nzL4xgrE_d"
def test_out_edges(sessions, item2session):
    """Count outgoing edges in an item-to-item graph
       (edge is one item following another in a session)
    """
    stats = {"e_cnt" : 0,
        "cnt_u20" : 0,
        "cnt_u10" : 0,
        "cnt_u05" : 0}

    out_cnt = defaultdict(set)
    for session_id, item_ids in sessions.items():

        last_item_id = None
        for item_id in item_ids:
            if last_item_id is not None:
                out_cnt[last_item_id].add(item_id)
            last_item_id = item_id

    for item_id, out_edges in out_cnt.items():
        stats["e_cnt"] += 1
        l = len(out_edges)
        if l <= 20:
            stats["cnt_u20"] += 1
            if l <= 10:
                stats["cnt_u10"] += 1
                if l <= 5:
                    stats["cnt_u05"] += 1

    return stats
```

```python id="8if8VdzErGAa"
def load_sessions(data):
    """Build a dictionary of sessions and a lookup map for
    finding which sessions an item belongs to
    """
    sessions = defaultdict(list)
    item2session = defaultdict(list)

    groupby = data.groupby("SessionId")
    for session_id, session in groupby:
        item_ids = [item_id for
            item_id in session.sort_values("Time")["ItemId"]]
        sessions[session_id] = item_ids

        for item_id in item_ids:
            item2session[item_id].append(session_id)

    return sessions, item2session
```

```python colab={"base_uri": "https://localhost:8080/"} id="QA8t130QrHCe" executionInfo={"status": "ok", "timestamp": 1639128792812, "user_tz": -330, "elapsed": 143166, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="17bf1ff2-3285-4f72-b5c5-1840d3be7f2f"
d = 1 # downsample the input data (0.1 - use only 10%% of input)
data_path = './retailrocket/slices/'
file_prefix = 'events'

test_all(data_path, file_prefix, d, [0,1,2,3,4])
```

<!-- #region id="Cf5ltvjjqxQP" -->
`"r_cnt"` in results is the total number of test cases examined.

Interpreting the results:

| Key        | Item to predict appears | Applies to algorithm  |
| ------------- | ------------- | ----- |
| cnt_next      | next to current item | MC, SF-SKNN |
| cnt_fwd10     | among 10 items after current item | SR |
| cnt_anywhere  | anywhere in session | AR, IKNN |
| cnt_anywhere_sess | in session with any current session item | \*SKNN |
<!-- #endregion -->

<!-- #region id="g-iyxuibDWgT" -->
---
<!-- #endregion -->

```python id="xgr27_gWDWgV"
# !apt-get -qq install tree
# !rm -r sample_data
```

```python colab={"base_uri": "https://localhost:8080/"} id="9ivY8TNnDWgW" executionInfo={"status": "ok", "timestamp": 1638638389489, "user_tz": -330, "elapsed": 431, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="88120d85-2201-4e66-ec9a-6827e870d1ca"
# !tree -h --du .
```

```python colab={"base_uri": "https://localhost:8080/"} id="bKMlo1SfDWgX" executionInfo={"status": "ok", "timestamp": 1638638417182, "user_tz": -330, "elapsed": 3625, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7ee406f5-fcd8-472e-944c-e8029b80553a"
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
