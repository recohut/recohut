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

<!-- #region id="cYbYigdtlpEZ" -->
# Recommend Vendors to Agencies
> Recommend vendors to local agencies in Washington DC using cosine similarity on interaction matrix

- toc: true
- badges: true
- comments: true
- categories: [Reference, CosineSimilarity]
- image:
<!-- #endregion -->

<!-- #region id="yrueKUUpmCJ0" -->
## Setup
<!-- #endregion -->

```python id="KFWUqZdrhXBw" executionInfo={"status": "ok", "timestamp": 1626979361922, "user_tz": -330, "elapsed": 1324, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import decomposition
import math

from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import f1_score as f1
```

```python id="As0BbOv7hXB0"
TOP_N_VALS = 20
```

<!-- #region id="HctoTGYhhXB0" -->
## EDA
<!-- #endregion -->

```python id="b10VDiwkhXB1"
data = pd.read_csv('Purchase_Card_Transactions.csv')
```

```python id="fbJVzyo1hXB1" outputId="9dbbd9e0-9ff3-4a03-dfd9-55274e121498"
data.head()
```

```python id="22l-IBlMhXB2" outputId="f00392ee-6d86-466b-b880-2e2ed38df325"
data.shape
```

```python id="9jLneQz_hXB3" outputId="57fb7d47-40c8-4613-eca8-d727c3de5ae9"
agencies = dict(data.AGENCY.value_counts()[:TOP_N_VALS])
plt.bar(agencies.keys(), height=agencies.values())
plt.xticks(rotation='vertical')
plt.show()
```

```python id="4IoCqkVchXB4" outputId="d8a04eff-3209-432f-a87b-9f2a23bca6fc"
vendors = dict(data.VENDOR_NAME.value_counts()[:TOP_N_VALS])
plt.bar(vendors.keys(), height=vendors.values())
plt.xticks(rotation='vertical')
plt.show()
```

```python id="YQKF1-mWhXB5" outputId="b99f8ab0-caee-43e5-d5bb-d12a0b33d139"
mcc = dict(data.MCC_DESCRIPTION.value_counts()[:TOP_N_VALS])
plt.bar(mcc.keys(), height=mcc.values())
plt.xticks(rotation='vertical')
plt.show()
```

```python id="ish8nlFVhXB5" outputId="cce0c73e-720f-4881-9d4d-3c66ab24c6d2"
num_categories = {}
category_cols = ['AGENCY', 'VENDOR_NAME', 'VENDOR_STATE_PROVINCE', 'MCC_DESCRIPTION']
for col in category_cols:
    num_categories[col] = data[col].nunique()
plt.bar(num_categories.keys(), height=num_categories.values())
plt.xticks(rotation='vertical')
plt.show()
```

```python id="79DNjrb_hXB6" outputId="7ee8f651-231a-4a4c-a112-d449a4f1b525"
plt.hist(data.TRANSACTION_AMOUNT, range = (0, 10000))
```

```python id="STYtMnZFhXB9" outputId="ddcef8ab-d857-4152-face-10ac7f3d058d"
plt.hist(data.TRANSACTION_AMOUNT, range = (0, 1000))
```

```python id="Wx60krI_hXB_" outputId="50d59892-aaf7-48d4-a4eb-4e845a42bcbc"
data[data.TRANSACTION_AMOUNT < 0]
```

```python id="4b7HDA8lhXCA" outputId="149a436f-3aeb-4d8e-d4f0-eae7b2780fdd"
data[data.TRANSACTION_AMOUNT > 100000]
```

```python id="LuQpv5XwhXCB" outputId="64197541-9e2d-4773-9416-a7988f0ef826"
set(data.AGENCY.unique()).intersection(data.VENDOR_NAME.unique())
```

```python id="2cN6Srm8hXCB" outputId="f282d488-dbae-4993-e767-f64cc749a213"
#hide
data.MCC_DESCRIPTION.unique()
```

<!-- #region id="16xu7FRZhXCC" -->
## Preprocessing
<!-- #endregion -->

```python id="8xLaYzoFhXCD" outputId="5a80454f-e27e-4873-feb2-c45b233718a7"
data[data.VENDOR_STATE_PROVINCE.isna()]
```

```python id="pXFb5iL7hXCE" outputId="666b24f6-fd57-4a74-b43c-37c1bef3ad59"
data[data.AGENCY.isna()]
```

```python id="j2DKHx-uhXCF" outputId="c6d6c820-69bb-4d09-bde0-0a1b1a386d5a"
data[data.VENDOR_NAME.isna()]
```

```python id="wSUyQPyQhXCF" outputId="55c17245-bacc-461a-a3bb-38dd2bcd1718"
data[data.TRANSACTION_AMOUNT.isna()]
```

```python id="Ws0cyTqihXCG"
cleaned_data = data[data.VENDOR_NAME.notna()]
cleaned_data = cleaned_data[data.TRANSACTION_AMOUNT >= 0]
```

```python id="6BNkwBoBhXCG"
agencies = sorted(cleaned_data.AGENCY.unique())
vendors = sorted(cleaned_data.VENDOR_NAME.unique())
industries = sorted(cleaned_data.MCC_DESCRIPTION.unique())
agency_ids = {}
vendor_ids = {}
industry_ids = {}
for i in range(len(agencies)):
    agency_ids[agencies[i]] = i
for i in range(len(vendors)):
    vendor_ids[vendors[i]] = i
for i in range(len(industries)):
    industry_ids[industries[i]] = i
```

```python id="QWKx2TtVhXCG"
cleaned_data['AGENCY_ID'] = [agency_ids[agency] for agency in cleaned_data.AGENCY]
cleaned_data['VENDOR_ID'] = [vendor_ids[vendor] for vendor in cleaned_data.VENDOR_NAME]
cleaned_data['INDUSTRY_ID'] = [industry_ids[industry] for industry in cleaned_data.MCC_DESCRIPTION]
```

```python id="mS_8Gi9UhXCH"
cleaned_data.to_csv('cleaned_data.csv')
```

```python id="Z8akeDMKhXCH"
cleaned_data = cleaned_data[["AGENCY_ID", "TRANSACTION_AMOUNT", "VENDOR_ID"]]
```

```python id="2Mc3zfRVhXCH" outputId="c3029516-6a1c-4511-d1e1-a41e6f2fb0f0"
cleaned_data.head()
```

```python id="mjUvXc6zhXCI" outputId="c3f282f4-4a84-425a-9371-db60b5d7fd04"
cleaned_data.shape
```

```python id="MkgH3NRthXCI"
def convert_to_matrix(df, rows, cols):
    matrix = [[0 for i in range(len(cols))] for j in range(len(rows))]
    for idx in df.index:
        matrix[idx[0]][idx[1]] = df.loc[idx].TRANSACTION_AMOUNT
    return matrix 
```

<!-- #region id="FyDoJa0ahXCI" -->
**Average Transactions Matrix**
<!-- #endregion -->

```python id="rR4ba-8fhXCJ"
means = cleaned_data.groupby(['AGENCY_ID', 'VENDOR_ID']).mean()
means_matrix = convert_to_matrix(means, agency_ids.values(), vendor_ids.values())
means_df = pd.DataFrame(means_matrix)
means_df.to_csv('means.csv')
```

<!-- #region id="mRRtfTG4hXCJ" -->
**Raw Number of Transactions Matrix**
<!-- #endregion -->

```python id="akESSAtUhXCJ"
counts = cleaned_data.groupby(['AGENCY_ID', 'VENDOR_ID']).count()
counts_matrix = convert_to_matrix(counts, agency_ids.values(), vendor_ids.values())
counts_df = pd.DataFrame(counts_matrix)
counts_df.to_csv('counts.csv')
```

<!-- #region id="g-kTBhrThXCJ" -->
**Sum of Transactions Matrix**
<!-- #endregion -->

```python id="2J75h32VhXCK"
sums = cleaned_data.groupby(['AGENCY_ID', 'VENDOR_ID']).sum()
sums_matrix = convert_to_matrix(sums, agency_ids.values(), vendor_ids.values())
sums_df = pd.DataFrame(sums_matrix)
sums_df.to_csv('sums.csv')
```

<!-- #region id="ppYwXJ58lLA8" -->
## Load cleaned data
<!-- #endregion -->

```python id="zGvZPNQohXl3"
data = pd.read_csv('cleaned_data.csv')
data.drop('Unnamed: 0', axis='columns', inplace=True)
data.sort_values(by=['TRANSACTION_DATE'], inplace=True)
```

<!-- #region id="JpJjqmi0hXl3" -->
## Train/test split
<!-- #endregion -->

```python id="hfKYAo6ehXl4"
train = data[:math.floor(.8*len(data))]
test = data[math.floor(.8*len(data)):]
```

```python id="c_Uy2VuhhXl4" outputId="4081812c-bd95-4e69-9e44-8233c2638139"
print(sorted(train.TRANSACTION_DATE.unique())[:3], sorted(train.TRANSACTION_DATE.unique())[-3:])
```

```python id="Jn8YYB75hXl6" outputId="84b365ae-7370-47b5-8fdc-51b8c9eea9ad"
print(sorted(test.TRANSACTION_DATE.unique())[:3], sorted(test.TRANSACTION_DATE.unique())[-3:])
```

<!-- #region id="KrNJS5s6kGVl" -->
<!-- #endregion -->

<!-- #region id="6w2QeHJYhXl7" -->
## Convert to matrix
<!-- #endregion -->

```python id="YFNkO8zhhXl7"
agencies = sorted(data.AGENCY.unique())
vendors = sorted(data.VENDOR_NAME.unique())
agency_ids = {}
vendor_ids = {}
for i in range(len(agencies)):
    agency_ids[agencies[i]] = i
for i in range(len(vendors)):
    vendor_ids[vendors[i]] = i
```

```python id="m_K9pmJ0hXl8"
train = train[["AGENCY_ID", "TRANSACTION_AMOUNT", "VENDOR_ID"]]
test = test[["AGENCY_ID", "TRANSACTION_AMOUNT", "VENDOR_ID"]]
```

```python id="khJKmUR7hXl8"
def convert_to_matrix(df, rows, cols):
    matrix = [[0 for i in range(len(cols))] for j in range(len(rows))]
    for idx in df.index:
        matrix[idx[0]][idx[1]] = df.loc[idx].TRANSACTION_AMOUNT
    return matrix 
```

<!-- #region id="algUAgZWjUpW" -->
<!-- #endregion -->

```python id="OS6wNFn9hXl9"
counts = train.groupby(['AGENCY_ID', 'VENDOR_ID']).count()
train_counts = pd.DataFrame(convert_to_matrix(counts, agency_ids.values(), vendor_ids.values()))
```

```python id="OVzy80RKhXl9"
counts = test.groupby(['AGENCY_ID', 'VENDOR_ID']).count()
test_counts = pd.DataFrame(convert_to_matrix(counts, agency_ids.values(), vendor_ids.values()))
```

<!-- #region id="IqdFXnNBhXl-" -->
## Making recommendations
<!-- #endregion -->

<!-- #region id="eblKKy5FkSnB" -->
<!-- #endregion -->

```python id="jIZnmj5IhXl-"
cos_counts = cosine_similarity(train_counts)
```

```python id="bouKGvYOhXl-" outputId="54250b4e-c4f5-4d8c-c35a-54dd34f95cc4"
pd.DataFrame(cos_counts)
```

```python id="AJ8TcbZ8hXl_"
def make_recommendations(transactions, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(transactions) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
```

```python id="B_GgfL1PhXmA"
np_train_counts = train_counts.to_numpy()
```

```python id="t9D80nvdhXmA"
recommendations = make_recommendations(np_train_counts, cos_counts)
```

```python id="0f_zyv2mhXmB"
for i in range(len(recommendations)): 
    for j in range(len(recommendations[i])): 
        recommendations[i][j]/=4
```

```python id="1vdWy4fqhXmB"
def get_mae(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mae(pred, actual)
```

```python id="rX6ZQIHdhXmB"
np_test_counts = test_counts.to_numpy()
```

```python id="DEusRyxLhXmB"
nonzero_mae = get_mae(recommendations, np_test_counts)
full_mae = mae(recommendations, np_test_counts)
```

```python id="zSoooSAFhXmC" outputId="296d125c-af90-41a9-fc3a-ee9fd98b5c52"
full_mae
```

<!-- #region id="pmJ7hWhQkcQy" -->
## References
1. https://github.com/vinaytummarakota/datathon-2021 `code`
2. https://docs.google.com/presentation/d/1zQbqXUQDL26hPcwlVOvDa6cKGqFDV_LaBGqTJNh3JqE/edit#slide=id.gbb5bc7b637_0_290 `ppt`
<!-- #endregion -->
