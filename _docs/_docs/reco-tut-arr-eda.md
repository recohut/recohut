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

```python id="-UOOzCs9ukul" executionInfo={"status": "ok", "timestamp": 1627893771333, "user_tz": -330, "elapsed": 908, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
project_name = "reco-tut-arr"; branch = "main"; account = "sparsh-ai"
```

```python colab={"base_uri": "https://localhost:8080/"} id="PYvHGli8ukum" executionInfo={"status": "ok", "timestamp": 1627893775113, "user_tz": -330, "elapsed": 2891, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="51b83a8a-2aba-4c20-b571-690a004f18ca"
!cp /content/drive/MyDrive/mykeys.py /content
import mykeys
!rm /content/mykeys.py
path = "/content/" + project_name; 
!mkdir "{path}"
%cd "{path}"
import sys; sys.path.append(path)
!git config --global user.email "arr@recohut.com"
!git config --global user.name  "reco-tut-arr"
!git init
!git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
!git pull origin "{branch}"
!git checkout main
```

```python id="XWXfp83ZDPbJ"
!pip install autoviz
```

```python id="J6GnSXizHC08" executionInfo={"status": "ok", "timestamp": 1627894816568, "user_tz": -330, "elapsed": 581, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
import numpy as np
import pandas as pd

# from scipy import stats # statistical library
# from statsmodels.stats.weightstats import ztest # statistical library for hypothesis testing
# import plotly.graph_objs as go # interactive plotting library
# import pandas_profiling # library for automatic EDA
# from autoviz.AutoViz_Class import AutoViz_Class
# from IPython.display import display # display from IPython.display
# from itertools import cycle # function used for cycling over values

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)
%matplotlib inline
```

```python colab={"base_uri": "https://localhost:8080/"} id="Z12sTqxTDvMB" executionInfo={"status": "ok", "timestamp": 1627894310995, "user_tz": -330, "elapsed": 1469, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9f1b3ab8-d325-48f9-cb39-09f630120f9e"
df = dict()

for dirname, _, filenames in os.walk('./data/bronze'):
    for filename in filenames:
        if filename.endswith('.parquet.gz'):
            name = filename.split('.')[0]
            df[name] = pd.read_parquet(os.path.join(dirname, filename))
        print(os.path.join(dirname, filename))
```

<!-- #region id="3GUO_RYmMD5T" -->
| filename | description |
| -------- | ----------- |
| test_customers.csv | customer id’s in the test set |
| test_locations.csv | latitude and longitude for the different locations of each customer |
| train_locations.csv | customer id’s in the test set |
| train_customers.csv | latitude and longitude for the different locations of each customer |
| orders.csv | orders that the customers train_customers.csv from made |
| vendors.csv | vendors that customers can order from |
<!-- #endregion -->

<!-- #region id="zHU1zCzTMk32" -->
### Train Customers
Information on the customers in the training set. 
- 'akeed_customer_id': Unique customer ID, used in train_locations and train_orders
- 'gender': Customer gender
- 'dob': Birth Year (if entered)
- 'status' and 'verified': Account status
- 'language': Chosen language
- 'Created_at' and 'updated_at': dates when account was created/updated

### Train Locations
Each customer orders from one or more locations. Each is assigned a location number. 
- 'customer_id': The unique customer ID 
- 'location_number': Location number (most customers have one or two)
- 'location_type': Home, Work, Other or NA
- 'Latitude' and 'longitude': Not true latitude and longitude - locations have been masked, but nearby locations remain nearby in the new reference frame and can thus be used for clustering. However, not all locations are useful due to GPS errors and missing data - you may want to treat outliers separately.

### Train Orders
This is a record of all orders made by customers in the train set from the vendors. Each order contains:
- 'akeed_order_id': The order ID used internally - can be ignored
- 'customer_id': The customer making the order, used to link with customer info
- 'item_count': how many items were in the order
- 'grand_total': total cost
- Payment related columns: 'payment_mode', 'Promo_code', 'vendor_discount_amount', 'Promo_code_discount_percentage'
- Vendor related columns:  'is_favorite', 'is_rated', 'vendor_rating', 'driver_rating'
- Order details:  'deliverydistance', 'preparationtime',  'delivery_time', 'order_accepted_time', 'driver_accepted_time', 'ready_for_pickup_time', 'picked_up_time', 'delivered_time', 'delivery_date','created_at'
- 'vendor_id': the unique ID of the vendor
- 'LOCATION_NUMBER': The location number specifies which of the customers locations the delivery was made to
- 'LOCATION_TYPE': same as location type in the train_locations table
- 'CID X LOC_NUM X VENDOR': Customer ID, location number and Vendor number

### Vendors
Contains info on the different vendors. Important columns are:
- 'id': The vendor ID used for the competition
- 'latitude' and 'longitude' : masked the same way as the customer locations
- 'vendor_tag_name': Tags describing the vendor

> Note: Test Customers and Test Locations follow the same format as train.

> Note: Other columns are mostly self-explanatory.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 343} id="uRdAH95GIxA9" executionInfo={"status": "ok", "timestamp": 1627894349065, "user_tz": -330, "elapsed": 498, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8a59917e-47ef-4c40-abee-50f63d973a3c"
df['orders'].head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="8zeqHNF1Ixr-" executionInfo={"status": "ok", "timestamp": 1627894360464, "user_tz": -330, "elapsed": 732, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c6e6832a-c932-4f1e-e95d-17fc6d444e55"
df['orders'].info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 317} id="uJM--o8nJbY5" executionInfo={"status": "ok", "timestamp": 1627894378188, "user_tz": -330, "elapsed": 662, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8962fc78-611d-47d2-dd0a-d6bb6d8347d3"
df['orders'].describe()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 244} id="SPhUBPhzKCo5" executionInfo={"status": "ok", "timestamp": 1627894383287, "user_tz": -330, "elapsed": 1184, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f99655f0-d377-449e-bdc5-6707acd8512f"
df['orders'].describe(include=['O'])
```

```python id="DnUhXx9RLGVk" executionInfo={"status": "ok", "timestamp": 1627895085918, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow, figsize=None):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    if not figsize:
        figsize = (6 * nGraphPerRow, 8 * nGraphRow)
    plt.figure(num = None, figsize = figsize, dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
```

```python id="MW1tvrwAFdpZ" executionInfo={"status": "ok", "timestamp": 1627895318170, "user_tz": -330, "elapsed": 433, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix', fontsize=15)
    plt.show()
```

```python id="ryP3v3LYFfdr" executionInfo={"status": "ok", "timestamp": 1627894716197, "user_tz": -330, "elapsed": 449, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="5qzM_PxMFhJI" executionInfo={"status": "ok", "timestamp": 1627895171826, "user_tz": -330, "elapsed": 4273, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4483fba6-3c29-4629-f9df-5719dfe06c59"
plotPerColumnDistribution(df['orders'], 10, 5, (20,6))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="36kQL-eGFs2w" executionInfo={"status": "ok", "timestamp": 1627895324960, "user_tz": -330, "elapsed": 1167, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="57d20ebd-8f86-4331-fbca-5d077385c913"
plotCorrelationMatrix(df['orders'], 8)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="m9qSt1yMHaLs" executionInfo={"status": "ok", "timestamp": 1627895464260, "user_tz": -330, "elapsed": 63399, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b2ed11c-a9b7-4805-dac8-9e53abcc7b14"
plotScatterMatrix(df['orders'], 20, 10)
```

```python id="eb3FHfKEIIND" executionInfo={"status": "ok", "timestamp": 1627896062537, "user_tz": -330, "elapsed": 468, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
## Reduce memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
```

```python colab={"base_uri": "https://localhost:8080/"} id="UVddrSJdKp2L" executionInfo={"status": "ok", "timestamp": 1627896088906, "user_tz": -330, "elapsed": 950, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6675ed3b-3a25-43c0-d48d-ba380ce0644d"
orders_df = reduce_mem_usage(df['orders'])
```
