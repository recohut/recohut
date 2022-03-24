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

<!-- #region id="Qp9CyeCjAP6I" -->
# Recsys'20 Feature Engineering Tutorial Part 2
> RecSys'20 tutorial on feature engineering on a large retail dataset part 2

- toc: true
- badges: true
- comments: true
- categories: [features, recsys, cudf, retail, bigdata]
- image: 
<!-- #endregion -->

<!-- #region id="qU7bKl-xHQi3" -->
### Install RAPIDS
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="A477KsIzC8yR" outputId="ca302d26-3fe9-40bc-9819-331bbf2298d7"
# Check Python Version
!python --version

# Check Ubuntu Version
!lsb_release -a

# Check CUDA/cuDNN Version
!nvcc -V && which nvcc

# Check GPU
!nvidia-smi
```

```python colab={"base_uri": "https://localhost:8080/"} id="mMqV93ivDa4q" outputId="ab4bbfe3-8395-47df-fe31-3ddb9ca6335c"
# This get the RAPIDS-Colab install files and test check your GPU.  Run this and the next cell only.
# Please read the output of this cell.  If your Colab Instance is not RAPIDS compatible, it will warn you and give you remediation steps.
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!python rapidsai-csp-utils/colab/env-check.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="wOzXb8sdE69d" outputId="1c7dcd93-c41b-42ff-e186-fee489f2dd17"
# This will update the Colab environment and restart the kernel.  Don't run the next cell until you see the session crash.
!bash rapidsai-csp-utils/colab/update_gcc.sh
import os
os._exit(00)
```

```python colab={"base_uri": "https://localhost:8080/"} id="uajvUZkpEqSe" outputId="83ab714b-bed6-4a73-fdae-20c2c27a4281"
# This will install CondaColab.  This will restart your kernel one last time.  Run this cell by itself and only run the next cell once you see the session crash.
import condacolab
condacolab.install()
```

```python colab={"base_uri": "https://localhost:8080/"} id="HckAjLPjENGz" outputId="c0d682b2-728e-447a-d70d-2dee9441580c"
# you can now run the rest of the cells as normal
import condacolab
condacolab.check()
```

```python colab={"base_uri": "https://localhost:8080/"} id="7_Ip6zuYG3u-" outputId="866e9dc3-59dd-42e7-b864-8e143a79a13c"
# Installing RAPIDS is now 'python rapidsai-csp-utils/colab/install_rapids.py <release> <packages>'
# The <release> options are 'stable' and 'nightly'.  Leaving it blank or adding any other words will default to stable.
# The <packages> option are default blank or 'core'.  By default, we install RAPIDSAI and BlazingSQL.  The 'core' option will install only RAPIDSAI and not include BlazingSQL, 
!python rapidsai-csp-utils/colab/install_rapids.py stable
```

```python id="op3d43QrB4bT"
import IPython

import pandas as pd
import cudf
import numpy as np
import cupy
import matplotlib.pyplot as plt
```

<!-- #region id="Q8E5moaL3pb6" -->
### Data load
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pN-Foz-x2-uO" outputId="0c8ef673-f8b1-4837-ed9b-6c91a9f5393f"
!cp /content/drive/MyDrive/Recommendation/data_silver_l2.zip /content
!unzip /content/data_silver_l2.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="t5g5KwR66gTi" outputId="84a9b173-3e12-476e-ee3c-23314f00b062"
df_train = cudf.read_parquet('/content/train.parquet')
df_valid = cudf.read_parquet('/content/valid.parquet')
df_test = cudf.read_parquet('/content/test.parquet')

df_train.isna().sum()
```

```python id="hyHiAN-FIc7f"
_temp = df_train['category_code'].str.split(".", n=3, expand=True).fillna('NA')
_temp.columns = ['cat_{}'.format(x) for x in _temp.columns]
df_train.drop('category_code', axis=1, inplace=True)
df_train = df_train.join(_temp)

_temp = df_valid['category_code'].str.split(".", n=3, expand=True).fillna('NA')
_temp.columns = ['cat_{}'.format(x) for x in _temp.columns]
df_valid.drop('category_code', axis=1, inplace=True)
df_valid = df_valid.join(_temp)

_temp = df_test['category_code'].str.split(".", n=3, expand=True).fillna('NA')
_temp.columns = ['cat_{}'.format(x) for x in _temp.columns]
df_test.drop('category_code', axis=1, inplace=True)
df_test = df_test.join(_temp)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 479} id="X0tlszhuOLfx" outputId="4eadc6e2-1321-460e-d3fd-4878b7c0da79"
df_train.head()
```

<!-- #region id="QDBrto66IJX7" -->
### Missing value imputation
<!-- #endregion -->

<!-- #region id="UIPbHCJMIRdL" -->
Categorical Features: Imputing categorical features is easy - a unique category value (e.g. "UNKNOWN") can be imputed

Important: Before imputing the missing values, it is beneficial to create a indicator column, which indicate if the a value was imputed or not. There is maybe a underlying pattern for the missing values and models can learn the pattern.
<!-- #endregion -->

```python id="t_69z61dCA27"
cols = ['brand', 'user_session', 'cat_0', 'cat_1', 'cat_2', 'cat_3']

for col in cols:
    df_train['NA_' + col] = df_train[col].isna().astype(np.int8)
    df_train[col].fillna('UNKNOWN', inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="sQAc_Sf6Otqt" outputId="47e3ea40-0635-4a83-8343-44fc421a96e2"
df_train.isna().sum()
```

<!-- #region id="QkmIIh2JPFUV" -->
Numerical Features: Imputing median for the numerical value (per group)
Imputing mean for numercial value (per group)
In some cases, we may know what value should be used as the default value (e.g. 0 for historical data or the max)


Important: For the same reason as in the categorical case, it is important to add a indicator column that the datapoint was imputed.

In our case, we do not have missing values in the numerical column price. Therefore, we artificially inject nans and then compare the difference.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="OwoNFjhHOu-X" outputId="25f4c3bc-ef90-41c9-9655-dcbad5a45cfc"
np.random.seed(42)
df_train.loc[np.random.random(df_train.shape[0])<0.01, 'price'] = None
df_train['price'].isna().mean()
```

<!-- #region id="78GxvWHFPL58" -->
We calculate the median per cat_2 and merge it to the dataset.
<!-- #endregion -->

```python id="WInk0_goPLgf"
df_median = df_train[['cat_2', 'price']].groupby('cat_2').median().reset_index()
df_median.columns = ['cat_2', 'price_median_per_cat2']
df_train = df_train.merge(df_median, how='left', on='cat_2')
```

<!-- #region id="QAmEbcYtPQB-" -->
We create an indicator column, when price was not available and then overwrite the missing values with the median.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 479} id="vPtjViuHPQVB" outputId="e0f2decb-0c8f-4acb-b691-2d1ba0aa6644"
df_train['NA_price'] = df_train[col].isna().astype(np.int8)
df_train.loc[df_train['price'].isna(), 'price'] = df_train.loc[df_train['price'].isna(), 'price_median_per_cat2']
df_train.drop('price_median_per_cat2', axis=1, inplace=True)
df_train.head(5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="gvol2f7nPlqM" outputId="f825b59f-bdb5-4030-9c36-4f70df320f9e"
df_train['price'].isna().mean()
```

<!-- #region id="us4AeY7RPxsK" -->
Predicting missing values: In [Improving Deep Learning For Airbnb Search](https://arxiv.org/abs/2002.05515), the authors propose to use a DNN for missing user engagement features of new items (listenings). New items have no historical user engagements, such as # of views, # of bookings, etc.. In the paper, they train a DNN based on the meta information, such as price, location and predict the user engagements feature. This could be interpreted in what are the expected user engagement.

Instead of the hand-crafted default values for missing user engagement, the authors replaced the missing values with the prediction of the DNN and showed that it reduced the error by 43% (offline test) and improved the overall bookings by 0.38% (online A/B test).
<!-- #endregion -->

<!-- #region id="za7BuNoIQJnb" -->
### Feature cross - basics
<!-- #endregion -->

<!-- #region id="pfRasuDrQXpc" -->
*Combining Categories (CC)* is a simple, powerful technique, but often undervalued. We will use this strategy in other feature engineering techniques, as well, and will introduce its value in a simple example.

In some datasets, categories by itself provide no information to predict the target. But if we combine multiple categories, together, then we can indentify patterns.

For example, we have the following categories:
- Weekday
- Hour of the day

Each of them independently has no significant pattern in the dataset. If we combine them with Weekday_HourOfTheDay, then we can observe some strong behavior for certainn times on the weekend

Decision Trees determine the split in the dataset on single features. If each categorical feature by itself does not provide the information gain, then Decision Trees cannot find a good split. If we provide a combined categorical feature, the Decision Tree can easier split the dataset.
<!-- #endregion -->

<!-- #region id="8qiWXy3PQfD9" -->
Combining categories, also called Cross Column or Cross Product, is used in the Wide Deep Architecture by Google and is implemented in Tensorflow.
<!-- #endregion -->

```python id="9dsdm-n4Pugm"
f1 = [0]*45 + [1]*45 + [2]*10 + [0]*5 + [1]*5 + [2]*90 + [0]*5 + [1]*5 + [2]*90 + [0]*45 + [1]*45 + [2]*10
f2 = [0]*45 + [0]*45 + [0]*10 + [1]*5 + [1]*5 + [1]*90 + [0]*5 + [0]*5 + [0]*90 + [1]*45 + [1]*45 + [1]*10
t = [1]*45 + [1]*45 + [1]*10 + [1]*5 + [1]*5 + [1]*90 + [0]*5 + [0]*5 + [0]*90 + [0]*45 + [0]*45 + [0]*10

data = cudf.DataFrame({
    'f1': f1,
    'f2': f2,
})

for i in range(3,5):
    data['f' + str(i)] = np.random.choice(list(range(3)), data.shape[0])

data['target'] = t
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="lhygn9-oQpdX" outputId="cd52d452-d4fb-45cd-8b00-201b62673552"
data.head()
```

<!-- #region id="mjj5QZL2QwnL" -->
We take a look on the features f1 and f2. Each of the feature provides no information gain as each category has a 0.5 probability for the target.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="8BC9QcUtQqq7" outputId="18541ac9-db8c-4d61-d70b-1733eb6dc802"
data.groupby('f1').target.agg(['mean', 'count'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 142} id="Bh800BKIQ32b" outputId="f67372be-c9c8-41cd-948e-b1a613b9bf4b"
data.groupby('f2').target.agg(['mean', 'count'])
```

<!-- #region id="J9GSOS7MQ-hQ" -->
If we analyze the features f1 and f2 together, we can observe a significant pattern in the target variable.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 266} id="e_6G8b4UQ7QL" outputId="9b75a01e-11ec-40bd-c0fb-e822389f2a66"
data.groupby(['f1', 'f2']).target.agg(['mean', 'count'])
```

<!-- #region id="bthWPaDzRBAq" -->
Next, we train a simple Decision Tree to show how combining categories will support the decision boundaries.
<!-- #endregion -->

```python id="JL1C6dk1Q_5u"
df = data.to_pandas()
```

```python id="cPyGtFO4RGic"
import pydotplus
import sklearn.tree as tree
from IPython.display import Image
```

```python id="ce-PCOHURJFh"
def get_hotn_features(df):
    out = []
    for col in df.columns:
        if col != 'target':
            out.append(pd.get_dummies(df[col], prefix=col))
    return(pd.concat(out, axis=1))

def viz_tree(df, lf):
    dt_feature_names = list(get_hotn_features(df).columns)
    dt_target_names = 'target'
    tree.export_graphviz(lf, out_file='tree.dot', 
                         feature_names=dt_feature_names, class_names=dt_target_names,
                         filled=True)  
    graph = pydotplus.graph_from_dot_file('tree.dot')
    return(graph.create_png())
```

<!-- #region id="JGhu2ehHRP9Z" -->
First, we train it without the combined categories f1 and f2. We can see, that the Decision Trees creates the split on the random features f3 and f4. The leaves have only a small information gain (e.g. 98 negative vs. 82 positive).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 436} id="GIEZNXnXRNg0" outputId="fc2ac105-b1de-49bc-88f2-097127af5127"
lf = tree.DecisionTreeClassifier(max_depth=2)
lf.fit(get_hotn_features(df), df[['target']])
Image(viz_tree(df, lf))
```

<!-- #region id="SwMHGlbIRZoM" -->
Now, we combine the categories f1 and f2 as an additional feature. We can see that the Decision Tree uses that feature first and that the splits have a high information gain. For example, 190 negative vs. 110 positives.
<!-- #endregion -->

```python id="eU7Kb8dGRVfI"
df['f1_f2'] = df['f1'].astype(str) + df['f2'].astype(str)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 436} id="dinz7i5IRb0w" outputId="5bd7ce04-40f0-4aca-d66a-c4135fd6b6e0"
lf.fit(get_hotn_features(df), df[['target']])
Image(viz_tree(df, lf))
```

<!-- #region id="U2h0VZsGRsF9" -->
This simple technique will be used in combination with other feature engineering techniques.

We may have the idea - that is great, let's combine all categories into one feature. Unfortunately, this is not that easy. We want to balance the number of categories used, the number of observations in resulting category values and the information gain:

The more categories we combine, we will identify more underlying patterns - but combining more categories together reduces the number of observation per categoy in the resulting features
Higher number of observation in the resulting category shows a strong pattern and it is more generalizable
High information gain supports our model, but only if it is generalizable
The extreme example is that we combine all features f1, f2, f3 and f4 together. But the observation per category (count) is very small (4-20)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="fFXBqCfgRdDP" outputId="41e893b2-dba8-46df-9158-e02834c806a7"
df.groupby([x for x in df.columns if 'target' not in x and 'f1_f2' not in x]).target.agg(['mean', 'count']).head(10)
```

<!-- #region id="TPYLLytNRwg_" -->
Best practices:

- Combining low cardinal categories is a good start. For example, the dataset size is 100M rows and there are multiple categories with a caridnality (# of unique values) of 10-50, then combining them should not result in low observation count
- Exploratory Data Analysis (EDA) is faster than training a model. Analyzing the information value for different combination of categorical features (on a sample) is really fast.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 173} id="t6052-UpRuH_" outputId="7e90e5db-2c86-41d6-9f3f-8b72f866c18a"
# Example of getting the cardinality for categories:
df.astype(str).describe()
```

<!-- #region id="SBL8pzpESUYh" -->
### Apply feature cross
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9DHLWCDyR-gN" outputId="7aaa4b82-7063-4cea-8d12-f6ca933d5bc0"
def explore_cat(df, cats):
    df_agg = df_train[cats + ['target']].groupby(cats).agg(['mean', 'count']).reset_index()
    df_agg.columns = cats + ['mean', 'count']
    print(df_agg.sort_values('count', ascending=False).head(20))
    
cats = ['product_id', 'user_id']  
explore_cat(df_train, cats)
```

```python colab={"base_uri": "https://localhost:8080/"} id="qjNt2wV5S69R" outputId="34fcbf5b-2e9a-466b-d7fb-8be95b1d9571"
cats = ['ts_weekday', 'ts_hour']  
explore_cat(df_train, cats)
```

```python colab={"base_uri": "https://localhost:8080/"} id="kbBeaeqUTB1i" outputId="283f3bf7-9876-42d6-aa98-6f47e5309907"
cats = ['cat_2', 'brand']  
explore_cat(df_train, cats)
```

<!-- #region id="DTOYOnEqTJlB" -->
Hypothesis:
- Some user will always buy the same one-way products e.g. cleaning supplies, food
- Behavior changes on weekday+hour - e.g. during the week, users will not stay up late as they work next day
- Category and brand are both powerful features, but the combination can be more important. E.g. do people buy apple smartphones or accessories?
<!-- #endregion -->

<!-- #region id="MKDvvtSuTXra" -->
### Categorify
<!-- #endregion -->

<!-- #region id="waLGDJLtUhSL" -->
*Categorifying* is required for using categorical features in deep learning models with Embedding layers. An Embedding layer encodes the category into a hidden latent vector with a smaller dimension.

Categorical features can be from datatype String or Integer. The Embedding layer requires that categorical features are continoues, positive Integers from 0 to |C| (number of unique category values).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="FeqVSEiDTGVk" outputId="f4f3b70b-f3f7-42c6-8d2f-580ba27b0921"
df_train['product_id'].unique()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Jx2LLSySU8wT" outputId="ba2e1a81-4312-412f-81cb-64b7036cf280"
# Using factorize creates continous integers from a categorical feature
codes, uniques = df_train['product_id'].factorize()
codes
```

```python colab={"base_uri": "https://localhost:8080/"} id="kcIH8T63VFQh" outputId="90aeb5cf-8d72-4775-8c98-8b2a374550c3"
uniques
```

<!-- #region id="nrn9IC53VVfd" -->
Another important reason to Categorify categorical features is to reduce the size of the dataset. Often categorical features are of the datatype String and sometimes, they are hashed to protect the user / dataset privacy.
<!-- #endregion -->

```python id="BZ23FZXZVMTR"
import hashlib
from sys import getsizeof
```

<!-- #region id="cHm4qcMlVaDn" -->
For example, we can hash the Integer 0 to a md5 hash
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="5pFDddngVaad" outputId="7a789865-5343-416a-d60a-c94c21a23e75"
hashlib.md5(b'0').hexdigest()
```

<!-- #region id="K36IY_V8Vfjh" -->
We can hash the full product_id column
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="606WOQsOVcGZ" outputId="2b19edb4-a551-4c77-9ab6-f3ac80fae71a"
hashSeries = df_train['product_id'].to_pandas().apply(lambda x: hashlib.md5(bytes(str(x), encoding='utf-8')).hexdigest())
hashSeries
```

```python id="BZ1BhmIxVoop"
codes, uniques = hashSeries.factorize()
```

```python colab={"base_uri": "https://localhost:8080/"} id="mz5YxGczVpAZ" outputId="e0ab482c-c6e0-4fd3-d081-47c0a4916352"
print("product id column size is reduced from {} to {}. We need only {:.2f}% of the original DataSeries memory."\
      .format(getsizeof(hashSeries), getsizeof(pd.DataFrame(codes)[0]),
              getsizeof(hashSeries)/getsizeof(pd.DataFrame(codes)[0])))
```

<!-- #region id="nY4nhADxWn3x" -->
Finally, we can prevent overfitting for low frequency categories. Categories with low frequency can be grouped together to an new category called 'other'. In the previous exercise we learned that it is powerful to combine categorical features together to create a new feature. However, combining categories increases the cardinality of the new feature and the number of observations per category will decrease. Therefore, we can apply a treshhold to group all categories with lower frequency count to the the new category.

In addition, categories, which occurs in the validation dataset and do not occur in the training dataset, should be mapped to the 'other' category as well.

We use in our example the category Ids 0 or 1 for a placeholder for the low frequency and unkown category. Then our function is independent of the cardinality of the categorical feature and we do not keep records of the cardinality to know the low frequency/unkown category.

In our dataset, we see that multiple product_ids occur only once in the training dataset. Our model would overfit to these low frequency categories.
<!-- #endregion -->

```python id="zp1NjDeGXOkP"
cat = 'product_id'
```

```python id="m-ih8rxNWIJA"
freq = df_train[cat].value_counts()
freq = freq.reset_index()
freq.columns = [cat, 'count']
freq = freq.reset_index()
freq.columns = [cat + '_Categorify', cat, 'count']
freq_filtered = freq[freq['count']>5]
freq_filtered[cat + '_Categorify'] = freq_filtered[cat + '_Categorify']+1
freq_filtered = freq_filtered.drop('count', axis=1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="8OaPTcoeXuwq" outputId="491690d5-027a-4f0c-e5a0-2fa6a933d8e1"
freq_filtered.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="JkRkDzebXwmA" outputId="7421d74c-5c97-42c4-9f0d-a5965ad9a4ff"
freq_filtered.shape
```

```python id="0x-TrwUNXuMS"
df_train = df_train.merge(freq_filtered, how='left', on=cat) #giving memory error
df_train[cat + '_Categorify'] = df_train[cat + '_Categorify'].fillna(0)
df_train['product_id_Categorify'].min(), df_train['product_id_Categorify'].max(), df_train['product_id_Categorify'].drop_duplicates().shape
```

<!-- #region id="LZtDwDfBXIkc" -->
We need to apply the categorify to our validation and test sets.
<!-- #endregion -->

```python id="smrWtZrpW-B4"
df_valid = df_valid.merge(freq_filtered, how='left', on=cat)
df_valid[cat + '_Categorify'] = df_valid[cat + '_Categorify'].fillna(0)

df_test = df_test.merge(freq_filtered, how='left', on=cat)
df_test[cat + '_Categorify'] = df_test[cat + '_Categorify'].fillna(0)
```

<!-- #region id="uCjz3xVcYMWc" -->
- Categorify is important to enable deep learning models to use categorical features
- Categorify can significantly reduce the dataset size by tranforming categorical features from String datatypes to Integer datatypes
- Categorify can prevent overfitting by grouping categories with low frequency into one category together
<!-- #endregion -->

<!-- #region id="i2ZRFKcxYVta" -->
Categorify the category features brand, Apply a frequency treshhold of minimum 20, Map low frequency categories to the id=0, and Map unkown categories to the id=1 in the validation and test set
<!-- #endregion -->

```python id="NXoDUHB8YJ_K"
cat = 'brand'

freq = df_train[cat].value_counts()
freq = freq.reset_index()
freq.columns = [cat, 'count']
freq = freq.reset_index()
freq.columns = [cat + '_Categorify', cat, 'count']
freq[cat + '_Categorify'] = freq[cat + '_Categorify']+2
freq.loc[freq['count']<20, cat + '_Categorify'] = 0

freq = freq.drop('count', axis=1)

# df_train = df_train.merge(freq, how='left', on=cat)
# df_train[cat + '_Categorify'] = df_train[cat + '_Categorify'].fillna(1)

df_valid = df_valid.merge(freq, how='left', on=cat)
df_valid[cat + '_Categorify'] = df_valid[cat + '_Categorify'].fillna(1)

df_test = df_test.merge(freq, how='left', on=cat)
df_test[cat + '_Categorify'] = df_test[cat + '_Categorify'].fillna(1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 479} id="lcJxM2GfYX34" outputId="bc7edff6-8741-4a99-cde7-5a6096f1a5d5"
df_test.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 317} id="A74KgrDtZHR2" outputId="be060a4f-d880-49f5-e9a5-c8471e976974"
df_test.describe()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 190} id="Cf7PQrylZKwi" outputId="f58498ff-2fd3-4dc3-bcad-ac23928eef21"
df_test.describe(include=['O'])
```

<!-- #region id="hJqluc2iZE7-" -->
### Target encoding
<!-- #endregion -->

<!-- #region id="iPsFk4igZahK" -->
*Target Encoding (TE)* calculates the statistics from a target variable grouped by the unique values of one or more categorical features.

For example in a binary classification problem, it calculates the probability that the target is true for each category value - a simple mean.
<!-- #endregion -->

```python id="GCQpUgWWY4_Y"
df_train['brand'] = df_train['brand'].fillna('UNKNOWN')
df_valid['brand'] = df_valid['brand'].fillna('UNKNOWN')
df_test['brand'] = df_test['brand'].fillna('UNKNOWN')
df_train['cat_2'] = df_train['cat_2'].fillna('UNKNOWN')
df_valid['cat_2'] = df_valid['cat_2'].fillna('UNKNOWN')
df_test['cat_2'] = df_test['cat_2'].fillna('UNKNOWN')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="mZfmRvnnZW8G" outputId="ea7cd1bd-b9a2-4988-93d6-b54d7c30c1a6"
cat = 'brand'

te = df_train[[cat, 'target']].groupby(cat).mean()
te
```

<!-- #region id="0bgc9xsIkX6X" -->
if you get MemoryError: std::bad_alloc: CUDA error at: /usr/local/include/rmm/mr/device/cuda_memory_resource.hpp:69: cudaErrorMemoryAllocation out of memory error, restart the session and run data loading blocks. It will free-up the GPU memory.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 479} id="5MYoD3UHZlOA" outputId="bfe8d1f7-9916-4469-e940-cfb22121f3ac"
te = te.reset_index()
te.columns = [cat, 'TE_' + cat]



# using small sample of 1 million records
df_train.sample(1000000, random_state=42).merge(te, how='left', on=cat).head()
```

<!-- #region id="ZV0vk6mRaY0B" -->
Similarly, we can apply Target Encoding to a group of categorical features.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="DyxQRS8YZre4" outputId="ccd71f07-3b97-41c9-9903-63f19b6d4d5d"
te = df_train[['brand', 'cat_2', 'target']].groupby(['brand', 'cat_2']).mean()
te
```

```python colab={"base_uri": "https://localhost:8080/", "height": 479} id="C_anqp6YarTz" outputId="d6e060c0-e973-47e9-ea86-8d02c749a094"
te = te.reset_index()
te.columns = ['brand', 'cat_2', 'TE_brand_cat_2']
df_train.sample(1000000, random_state=42).merge(te, how='left', left_on=['brand', 'cat_2'], right_on=['brand', 'cat_2']).head()
```

<!-- #region id="YXtaei12bF9a" -->
Target Encoding creates a new features, which can be used by the model for training. The advantage of Target Encoding is, that it process the categorical features and makes them easier accessible to the model during training and validation.

Tree-based model requires to create a split for each categorical value (depending on the exact model). Target Encoding saves to create many splits for the model. In particular, when applying Target Encoding to multiple columns, it reduces significantly the number of splits. The model can directly operate on the probablities/averages and creates a split based on them.
Another advantage is, that some boosted-tree libraries, such as XGBoost, cannot handle categorical features. The library requires to hot-n encode them. Categorical features with large cardinality (e.g. >100) are inefficient to store as hot-n.

Deep learning models often apply Embedding Layers to categorical features. Embedding layer can overfit quickly and categorical values with low frequencies have ony a few gradient descent updates and can memorize the training data.
<!-- #endregion -->

<!-- #region id="JP0XAhPPdZEZ" -->
### Smoothing
<!-- #endregion -->

<!-- #region id="6qqG22jYdbLG" -->
The introduced Target Encoding is a good first step, but it lacks to generalize well and it will tend to overfit, as well.

Let's take a look on Target Encoding with the observation count:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 481} id="QdT6B0sGa3Tq" outputId="763a17b8-381c-4b68-eea8-7fcf415d53f4"
df_train[[cat, 'target']].groupby(cat).agg(['mean', 'count'])
```

```python id="yubqkhlfdwLt"
dd = df_train[[cat, 'target']].groupby(cat).agg(['mean', 'count']).reset_index()['target']['count']
```

```python colab={"base_uri": "https://localhost:8080/", "height": 282} id="rRfuWxF2d1Pp" outputId="761982a6-235c-4119-86a2-e94da3e4ca86"
plt.bar(dd.groupby('count').count().index.to_array(), dd.groupby('count').count().to_array())
plt.xlim(0,50)
```

<!-- #region id="PMPIP9sQd-YN" -->
We can observe, that the observation count for some categories are 1. This means, that we have only one data point to calculate the average and Target Encoding overfits to these values. Therefore, we need to adjust the calculation:

- if the number of observation is high, we want to use the mean of this category value
- if the number of observation is low, we want to use the global mean
<!-- #endregion -->

<!-- #region id="an5ucQCieVn1" -->
<!-- #endregion -->

```python id="LQpiJd-0d2JV"
feat = ['brand', 'cat_2']
w = 20

mean_global = df_train.target.mean()
te = df_train.groupby(feat)['target'].agg(['mean','count']).reset_index()
te['TE_brand_cat_2'] = ((te['mean']*te['count'])+(mean_global*w))/(te['count']+w)

df_train = df_train.sample(1e6, random_state=42).merge(te, on=feat, how='left')
df_valid = df_valid.merge( te, on=feat, how='left' )
df_test = df_test.merge( te, on=feat, how='left' )
df_valid['TE_brand_cat_2'] = df_valid['TE_brand_cat_2'].fillna(mean_global)
df_test['TE_brand_cat_2'] = df_test['TE_brand_cat_2'].fillna(mean_global)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 479} id="_T9pX78Ie35z" outputId="fc5734da-96cd-461b-de3b-cb2a53c13693"
df_test.head()
```

<!-- #region id="YC8lrt4ZgImh" -->
### Improve TargetEncoding with out-of-fold
We can still improve our Target Encoding function. We can even make it more generalizable, if we apply an out of fold calculation.

In our current definition, we use the full training dataset to Target Encode the training dataset and validation/test dataset. Therefore, we will likely overfit slightly on our training dataset, because we use the information from it to encode the categorical values. A better strategy is to use out of fold:

- use the full training dataset to encode the validation/test dataset
- split the training dataset in k-folds and encode the i-th fold by using all folds except of the i-th one

<!-- #endregion -->

<!-- #region id="phLfIffvgOPL" -->
The following figure visualize the strategy for k=5:
<!-- #endregion -->

<!-- #region id="PMRJLWD0gPyp" -->
<!-- #endregion -->

```python id="5zYPc7TsfS_4"
def target_encode(train, valid, col, target, kfold=5, smooth=20):
    """
        train:  train dataset
        valid:  validation dataset
        col:   column which will be encoded (in the example RESOURCE)
        target: target column which will be used to calculate the statistic
    """
    
    # We assume that the train dataset is shuffled
    train['kfold'] = ((train.index) % kfold)
    # We keep the original order as cudf merge will not preserve the original order
    train['org_sorting'] = cupy.arange(len(train), dtype="int32")
    # We create the output column, we fill with 0
    col_name = '_'.join(col)
    train['TE_' + col_name] = 0.
    for i in range(kfold):
        ###################################
        # filter for out of fold
        # calculate the mean/counts per group category
        # calculate the global mean for the oof
        # calculate the smoothed TE
        # merge it to the original dataframe
        ###################################
        
        df_tmp = train[train['kfold']!=i]
        mn = df_tmp[target].mean()
        df_tmp = df_tmp[col + [target]].groupby(col).agg(['mean', 'count']).reset_index()
        df_tmp.columns = col + ['mean', 'count']
        df_tmp['TE_tmp'] = ((df_tmp['mean']*df_tmp['count'])+(mn*smooth)) / (df_tmp['count']+smooth)
        df_tmp_m = train[col + ['kfold', 'org_sorting', 'TE_' + col_name]].merge(df_tmp, how='left', left_on=col, right_on=col).sort_values('org_sorting')
        df_tmp_m.loc[df_tmp_m['kfold']==i, 'TE_' + col_name] = df_tmp_m.loc[df_tmp_m['kfold']==i, 'TE_tmp']
        train['TE_' + col_name] = df_tmp_m['TE_' + col_name].fillna(mn).values

    
    ###################################
    # calculate the mean/counts per group for the full training dataset
    # calculate the global mean
    # calculate the smoothed TE
    # merge it to the original dataframe
    # drop all temp columns
    ###################################    
    
    df_tmp = train[col + [target]].groupby(col).agg(['mean', 'count']).reset_index()
    mn = train[target].mean()
    df_tmp.columns = col + ['mean', 'count']
    df_tmp['TE_tmp'] = ((df_tmp['mean']*df_tmp['count'])+(mn*smooth)) / (df_tmp['count']+smooth)
    valid['org_sorting'] = cupy.arange(len(valid), dtype="int32")
    df_tmp_m = valid[col + ['org_sorting']].merge(df_tmp, how='left', left_on=col, right_on=col).sort_values('org_sorting')
    valid['TE_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    
    valid = valid.drop('org_sorting', axis=1)
    train = train.drop('kfold', axis=1)
    train = train.drop('org_sorting', axis=1)
    return(train, valid)
```

<!-- #region id="jHmuQFGvhGiB" -->
- Target Encoding calculates statistics of a target column given one or more categorical features
- Target Encoding smooths the statistics as a weighted average of the category value and the global statistic
- Target Encoding uses a out-of-fold strategy to prevent overfitting to the training dataset.

We can see the advantage of using Target Encoding as a feature engineering step. A tree-based model or a neural network learns the average probability for the category value. However, neither model is designed to prevent overfitting.
<!-- #endregion -->

<!-- #region id="HeZ0ZTvvhQCy" -->
### Count encoding
<!-- #endregion -->

<!-- #region id="NknHxLqphTFD" -->
*Count Encoding (CE)* calculates the frequency from one or more categorical features given the training dataset.

For example we can consider Count Encoding as the populiarity of an item or activity of an user.

Count Encoding creates a new feature, which can be used by the model for training. It groups categorical values based on the frequency together.

For example,

users, which have only 1 interaction in the datasets, are encoded with 1. Instead of having 1 datapoint per user, now, the model can learn a behavior pattern of these users at once.
products, which have many interactions in the datasets, are encoded with a high number. The model can learn to see them as top sellers and treat them, accordingly.

The advantage of Count Encoding is that the category values are grouped together based on behavior. Particularly in cases with only a few observation, a decision tree is not able to create a split and neural networks have only a few gradient descent updates for these values.
<!-- #endregion -->

```python id="SKu-1bfbhMvE"
col = 'user_id'

df_train['org_sorting'] = cupy.arange(len(df_train), dtype="int32")
    
train_tmp = df_train[col].value_counts().reset_index()
train_tmp.columns = [col,  'CE_' + col]
df_tmp = df_train[[col, 'org_sorting']].merge(train_tmp, how='left', left_on=col, right_on=col).sort_values('org_sorting')
df_train['CE_' + col] = df_tmp['CE_' + col].fillna(0).values
df_train = df_train.drop('org_sorting', axis=1)
        
df_valid['org_sorting'] = cupy.arange(len(df_valid), dtype="int32")
df_tmp = df_valid[[col, 'org_sorting']].merge(train_tmp, how='left', left_on=col, right_on=col).sort_values('org_sorting')
df_valid['CE_' + col] = df_tmp['CE_' + col].fillna(0).values
df_valid = df_valid.drop('org_sorting', axis=1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 462} id="ngtjgCvWhogv" outputId="543a20d3-b5bc-49c1-c35e-407da625ceba"
df_train.head()
```

<!-- #region id="RhXeuH1RiQvL" -->
### Binning
<!-- #endregion -->

<!-- #region id="b8uDE63_iXIX" -->
*Binning* maps multiple ordinal categorical or numerical features into groups. It is mainly applied to numerical features:

- prevent overfitting by grouping values together
- enables us to add some expert knowledge into the model
- most simple case: binary flags, e.g. features is greater than 0

Examples:
- binning weekdays into weekday and weekend
- binning hours into morning, early afternoon, late afternoon, evening and night
- binning age into child, adlult and retired

We can take a look on the hour of the day. We can see multiple patterns:
- 0-3 Night: Low purchase probability
- 4-7 Early morning: Mid purchase probability
- 8-14 Morning/Lunch: Higher purchase probability
- 15-20 Afternoon: Low purchase probability
- 21-23: Evening: High purchase probability
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 421} id="qL0lTpQrh37u" outputId="bbe50f28-aad8-4cf1-a776-9d147b8ca394"
df_train[['ts_hour', 'target']].groupby('ts_hour').agg(['count', 'mean']).head(10)
```

```python id="DzzQxbkMihe_"
hour = list(range(0,24))
hour_bin = [0]*4 + [1]*4 + [2]*7 + [3]*6 + [4]*3

data = cudf.DataFrame({
    'hour': hour,
    'hour_bin': hour_bin,
})
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="4LJ1wIaliiDc" outputId="961eca33-6e50-4ce6-c763-c7b55f09dc95"
data.head(10)
```

```python id="AFtrw11uijYX"
df_train = df_train.merge(data, how='left', right_on='hour', left_on='ts_hour')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 266} id="nz0TmY86ilxl" outputId="a1a404f7-c050-4995-aa14-d7afd12c88f8"
df_train[['hour_bin', 'target']].groupby('hour_bin').agg(['count', 'mean'])
```

<!-- #region id="dLqG0WN0iqLA" -->
Binning the numerical features reduces the cardinality (# of unique values). Therefore, a model can easier learn the relationship to the target variables, as there are more observation per category. In addition, binning prevents overfitting.

Another reason to apply binning is to standardize numeric variables per category group. The datasets provides information about the product category (cat_1) and price information.

For example, the headphones and smartphones have a different price distribution.

- We can probably buy good headphones between 100−200
- For a good smartphone, prices are probably in the range of 400−1200

Therefore, the buying behavior should be different depending on the price per category (what is a good deal).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="T_Ubdzpyim8V" outputId="da8bfd50-e06d-434d-a9c2-cc11f50a1890"
plt.hist(df_train[df_train['cat_2']=='headphone'].price.to_pandas(), bins=50)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="2wmQeEzgixMO" outputId="56208ec7-3eb2-4a4c-92c7-111ea51fdbe3"
plt.hist(df_train[df_train['cat_1']=='smartphone'].price.to_pandas(), bins=50)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="woQ9ar7ajFZf" outputId="f4fe3bc1-0e51-4522-e330-cae378e16727"
print('Headphones mean price: ' + str(df_train[df_train['cat_2']=='headphone'].price.mean()) + ' median price: ' + str(df_train[df_train['cat_2']=='headphone'].price.median()))
print('Smartphones mean price: ' + str(df_train[df_train['cat_1']=='smartphone'].price.mean()) + ' median price: ' + str(df_train[df_train['cat_1']=='smartphone'].price.median()))
```

<!-- #region id="Efn5mxlSjLMP" -->
Based on the category tree, we want to bin the prices as a combination of cat_0, cat_1 and cat_2.
<!-- #endregion -->

```python id="oBxzuI4qjIzw"
df_train['cat_012'] = df_train['cat_0'].astype(str) + '_' + df_train['cat_1'].astype(str) + '_' + df_train['cat_2'].astype(str)
q_list = [0.1, 0.25, 0.5, 0.75, 0.9]
```

<!-- #region id="4WYhw7uYjRwO" -->
We calculate the quantiles per category group and then merge the quantile to the original dataframe.
<!-- #endregion -->

```python id="xG8D7-RWjRWn"
for q_value in q_list:
    q = df_train[['cat_012', 'price']].groupby(['cat_012']).quantile(q_value)
    q = q.reset_index()
    q.columns = ['cat_012', 'price' + str(q_value)]
    df_train = df_train.merge(q, how='left', on='cat_012')
```

<!-- #region id="nGCecXWkjW1a" -->
Afterwards, we loop through the columns and update the price_bin depending, if the price is between quantiles.
<!-- #endregion -->

```python id="SYvAf9NMjVus"
df_train['price_bin'] = -1

for i, q_value in enumerate(q_list):
    if i == 0:
        df_train.loc[df_train['price']<=df_train['price' + str(q_value)], 'price_bin'] = i
    else:
        df_train.loc[(df_train['price']>df_train['price' + str(q_list[i-1])]) & (df_train['price']<=df_train['price' + str(q_value)]), 'price_bin'] = i
        
df_train.loc[df_train['price']>df_train['price' + str(q_value)], 'price_bin'] = i+1
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="1bDMXe1KjmaU" outputId="439d4e05-bbb6-470a-cea6-0d4d8b8a2eba"
df_train[df_train['price_bin']==3][['price', 'price0.1', 'price0.25', 'price0.5', 'price0.75', 'price0.9', 'price_bin']].drop_duplicates()
```

```python id="nyv_CgM7jm7h"
df_train = df_train.drop(['price' + str(x) for x in q_list], axis=1)
```

<!-- #region id="LyRh_1htj-nK" -->
We can see the pattern, that products in a lower quantile 0-10% and 10-25% have lower purchase probabilities.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="StNkrwttjsAN" outputId="7d2d7210-7d47-406f-b31d-5bacaf21fd7b"
df_train[['price_bin', 'target']].groupby('price_bin').agg(['count', 'mean'])
```

<!-- #region id="6P-gaoYxkHQV" -->
Now, let's take a look on ts_weekday.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 328} id="dmJh_uoEkD8q" outputId="322a3e99-ca90-428b-f03e-8637b45cbd81"
df_train[['ts_weekday', 'target']].groupby('ts_weekday').agg(['count', 'mean'])
```

```python id="tsBOrZVnkMNu"
weekday = list(range(0,7))
weekday_bin = [0, 1, 1, 2, 2, 2, 0]

data = cudf.DataFrame({
    'weekday': weekday,
    'weekday_bin': weekday_bin,
})
```

```python id="59KnnYmtkNPl"
df_train = df_train.merge(data, how='left', right_on='weekday', left_on='ts_weekday')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="dcwtUdsHkNrb" outputId="c18506dc-8f69-4335-a8c3-e9f2d53c77f8"
df_train[['weekday_bin', 'target']].groupby('weekday_bin').agg(['count', 'mean'])
```

<!-- #region id="RMtFcJAEk18K" -->
It is maybe counterintuitive:
* the highest days are Sunday and Monday - a hypothesis could be that people shop on Sunday evening and the first day of the week
* the lowest days are Thur-Sat - a hypothesis could be that Thu/Fri is end of 
 week and people are finishing up their work and have no time to do online shopping
* Saturday is maybe a day go outside
<!-- #endregion -->

<!-- #region id="zIskg9s5lc1O" -->
### Normalization
<!-- #endregion -->

<!-- #region id="Wv-anGkSmCpm" -->
*Normalization* is required to enable neural networks to leverage numerical features. Tree-based models do not require normalization as they define the split independent of the scale of a feature. Without normalization, neural networks are difficult to train. The image visualizes the loss surface and the gradient updates for non-normalized input (left) and normalized input (right).

We will first generate some numerical features with the feature engineering that we also covered in previous steps.

The reason is that different numerical features have different scales. When we combine the features in a hidden layer, the different scales make it more difficult to extract patterns from it.

Normalization Techniques
After we outline the importance for normalizing the numerical input feature, we will discuss different strategy to achieve a normal distributed input feature:
- Normalization with mean/std
- Log-based normalization
- Scale to 0-1
- Gauss Rank (separate notebook)
- Power transfomer
<!-- #endregion -->

```python id="wt2SL_yCleu4"
def target_encode(train, valid, col, target, kfold=5, smooth=20, gpu=True):
    """
        train:  train dataset
        valid:  validation dataset
        col:   column which will be encoded (in the example RESOURCE)
        target: target column which will be used to calculate the statistic
    """
    
    # We assume that the train dataset is shuffled
    train['kfold'] = ((train.index) % kfold)
    # We keep the original order as cudf merge will not preserve the original order
    if gpu:
        train['org_sorting'] = cupy.arange(len(train), dtype="int32")
    else:
        train['org_sorting'] = np.arange(len(train), dtype="int32")
    # We create the output column, we fill with 0
    col_name = '_'.join(col)
    train['TE_' + col_name] = 0.
    for i in range(kfold):
        ###################################
        # filter for out of fold
        # calculate the mean/counts per group category
        # calculate the global mean for the oof
        # calculate the smoothed TE
        # merge it to the original dataframe
        ###################################
        
        df_tmp = train[train['kfold']!=i]
        mn = df_tmp[target].mean()
        df_tmp = df_tmp[col + [target]].groupby(col).agg(['mean', 'count']).reset_index()
        df_tmp.columns = col + ['mean', 'count']
        df_tmp['TE_tmp'] = ((df_tmp['mean']*df_tmp['count'])+(mn*smooth)) / (df_tmp['count']+smooth)
        df_tmp_m = train[col + ['kfold', 'org_sorting', 'TE_' + col_name]].merge(df_tmp, how='left', left_on=col, right_on=col).sort_values('org_sorting')
        df_tmp_m.loc[df_tmp_m['kfold']==i, 'TE_' + col_name] = df_tmp_m.loc[df_tmp_m['kfold']==i, 'TE_tmp']
        train['TE_' + col_name] = df_tmp_m['TE_' + col_name].fillna(mn).values

    
    ###################################
    # calculate the mean/counts per group for the full training dataset
    # calculate the global mean
    # calculate the smoothed TE
    # merge it to the original dataframe
    # drop all temp columns
    ###################################    
    
    df_tmp = train[col + [target]].groupby(col).agg(['mean', 'count']).reset_index()
    mn = train[target].mean()
    df_tmp.columns = col + ['mean', 'count']
    df_tmp['TE_tmp'] = ((df_tmp['mean']*df_tmp['count'])+(mn*smooth)) / (df_tmp['count']+smooth)
    if gpu:
        valid['org_sorting'] = cupy.arange(len(valid), dtype="int32")
    else:
        valid['org_sorting'] = np.arange(len(valid), dtype="int32")
    df_tmp_m = valid[col + ['org_sorting']].merge(df_tmp, how='left', left_on=col, right_on=col).sort_values('org_sorting')
    valid['TE_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    
    valid = valid.drop('org_sorting', axis=1)
    train = train.drop('kfold', axis=1)
    train = train.drop('org_sorting', axis=1)
    return(train, valid)
```

```python id="Zp9NfIP8lmTj"
cats = [['cat_0'], ['cat_1'], ['cat_2'], ['cat_0', 'cat_1', 'cat_2'], ['ts_hour'], ['ts_weekday'], ['ts_weekday', 'ts_hour', 'cat_2', 'brand']]

for cat in cats:
    df_train, df_valid = target_encode(df_train, df_valid, cat, 'target')
```

```python id="eSGViLTTl0Xn"
def count_encode(train, valid, col, gpu=True):
    """
        train:  train dataset
        valid:  validation dataset
        col:    column which will be count encoded (in the example RESOURCE)
    """
    # We keep the original order as cudf merge will not preserve the original order
    if gpu:
        train['org_sorting'] = cupy.arange(len(train), dtype="int32")
    else:
        train['org_sorting'] = np.arange(len(train), dtype="int32")
    
    train_tmp = train[col].value_counts().reset_index()
    train_tmp.columns = [col,  'CE_' + col]
    df_tmp = train[[col, 'org_sorting']].merge(train_tmp, how='left', left_on=col, right_on=col).sort_values('org_sorting')
    train['CE_' + col] = df_tmp['CE_' + col].fillna(0).values
        
    if gpu:
        valid['org_sorting'] = cupy.arange(len(valid), dtype="int32")
    else:
        valid['org_sorting'] = np.arange(len(valid), dtype="int32")
    df_tmp = valid[[col, 'org_sorting']].merge(train_tmp, how='left', left_on=col, right_on=col).sort_values('org_sorting')
    valid['CE_' + col] = df_tmp['CE_' + col].fillna(0).values
    
    valid = valid.drop('org_sorting', axis=1)
    train = train.drop('org_sorting', axis=1)
    return(train, valid)
```

```python id="MgxwaFoClver"
cats = ['brand', 'user_id', 'product_id', 'cat_0', 'cat_1', 'cat_2']

for cat in cats:
    df_train, df_valid = count_encode(df_train, df_valid, cat, gpu=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 479} id="-jv0pn3ul7tz" outputId="0e74c0dc-d01d-456e-d1ea-941cccf95ff3"
df_train.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="CB7mJ12Vl9Py" outputId="dba7eff5-a98c-4d95-bfa2-a1c4ed00ff4e"
df_train.columns
```

<!-- #region id="kx_SfKiame1U" -->
Let's normalize the features: price, TE_ts_weekday_ts_hour_cat_2_brand, CE_cat_2
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 244} id="NC7zcfXjl_h1" outputId="4b2f8589-4b52-4fe0-9122-9199c5b66292"
cat = 'price'

X = df_train[cat]
X_norm = (X-X.mean())/X.std()

X_log = np.log(X.to_pandas()+1)
X_log_norm = (X_log-X_log.mean())/X_log.std()

X_minmax = ((X-X.min())/(X.max()-X.min()))

fig, axs = plt.subplots(1, 4, figsize=(16,3))
axs[0].hist(X.sample(frac=0.01).to_pandas(), bins=50)
axs[0].set_title('Histogram non-normalised')
axs[1].hist(X_norm.sample(frac=0.01).to_pandas(), bins=50)
axs[1].set_title('Histogram normalised')
axs[2].hist(X_log_norm.sample(frac=0.01), bins=50)
axs[2].set_title('Histogram log-normalised')
axs[3].hist(X_minmax.sample(frac=0.01).to_pandas(), bins=50)
axs[3].set_title('Histogram minmax')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 244} id="lPNIGID3midD" outputId="f34c5949-5fab-442c-f59b-ac9112b4ebff"
cat = 'TE_ts_weekday_ts_hour_cat_2_brand'

X = df_train[cat]
X_norm = (X-X.mean())/X.std()

X_log = np.log(X.to_pandas()+1)
X_log_norm = (X_log-X_log.mean())/X_log.std()

X_minmax = ((X-X.min())/(X.max()-X.min()))

fig, axs = plt.subplots(1, 4, figsize=(16,3))
axs[0].hist(X.sample(frac=0.01).to_pandas(), bins=50)
axs[0].set_title('Histogram non-normalised')
axs[1].hist(X_norm.sample(frac=0.01).to_pandas(), bins=50)
axs[1].set_title('Histogram normalised')
axs[2].hist(X_log_norm.sample(frac=0.01), bins=50)
axs[2].set_title('Histogram log-normalised')
axs[3].hist(X_minmax.sample(frac=0.01).to_pandas(), bins=50)
axs[3].set_title('Histogram minmax')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 257} id="fwLMIRhRmkHK" outputId="3bfe0f54-691e-4aca-8ceb-04b80efcd8d2"
cat = 'CE_cat_2'

X = df_train[cat]
X_norm = (X-X.mean())/X.std()

X_log = np.log(X.to_pandas()+1)
X_log_norm = (X_log-X_log.mean())/X_log.std()

X_minmax = ((X-X.min())/(X.max()-X.min()))

fig, axs = plt.subplots(1, 4, figsize=(16,3))
axs[0].hist(X.sample(frac=0.01).to_pandas(), bins=50)
axs[0].set_title('Histogram non-normalised')
axs[1].hist(X_norm.sample(frac=0.01).to_pandas(), bins=50)
axs[1].set_title('Histogram normalised')
axs[2].hist(X_log_norm.sample(frac=0.01), bins=50)
axs[2].set_title('Histogram log-normalised')
axs[3].hist(X_minmax.sample(frac=0.01).to_pandas(), bins=50)
axs[3].set_title('Histogram minmax')
```

<!-- #region id="vSLsbqUdmv6B" -->
### Gauss rank
<!-- #endregion -->

<!-- #region id="GkhtEw2YmxU1" -->
Gauss Rank* transforms any arbitrary distribution to a Gaussian normal distribution by

1. Compute the rank (or sort the values ascending)
2. Scale the values linearly from -1 to +1
3. Apply the erfinv function
<!-- #endregion -->

<!-- #region id="kz1FZrPJm-75" -->
<!-- #endregion -->

```python id="sYEsdKcDmlsh"
import cupy as cp
from cupyx.scipy.special import erfinv
import cudf as gd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erfinv as sp_erfinv
```

```python id="LpBRtTgXnCJm"
def gaussrank_cpu(data, epsilon = 1e-6):
    r_cpu = data.argsort().argsort()
    r_cpu = (r_cpu/r_cpu.max()-0.5)*2 # scale to (-1,1)
    r_cpu = np.clip(r_cpu,-1+epsilon,1-epsilon)
    r_cpu = sp_erfinv(r_cpu)
    return(r_cpu)

def gaussrank_gpu(data, epsilon = 1e-6):
    r_gpu = data.argsort().argsort()
    r_gpu = (r_gpu/r_gpu.max()-0.5)*2 # scale to (-1,1)
    r_gpu = cp.clip(r_gpu,-1+epsilon,1-epsilon)
    r_gpu = erfinv(r_gpu)
    return(r_gpu)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 244} id="6aqQQHEdnFNd" outputId="86b16f95-d656-415d-9825-87828d8fb372"
fig, axs = plt.subplots(1, 2, figsize=(16,3))
col = 'CE_product_id'
data_sample = df_train[col].sample(frac=0.01)
axs[0].hist(data_sample.to_pandas().values, bins=50)
axs[1].hist(cp.asnumpy(gaussrank_gpu(df_train[col].values)), bins=50)
axs[0].set_title('Histogram non-normalized')
axs[1].set_title('Histogram Gauss Rank')
```

<!-- #region id="gCTSBzX6nIsH" -->
Let's normalize the features price, TE_ts_weekday_ts_hour_cat_2_brand and CE_cat_2 with GaussRank, and plot the non-normalized and normalized values
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 566} id="e2FSjqRGnFcz" outputId="bae46377-5960-4b93-ee70-f81ea41b108e"
fig, axs = plt.subplots(3, 2, figsize=(16,9))
for i, col in enumerate(['price', 'TE_ts_weekday_ts_hour_cat_2_brand', 'CE_cat_2']):
    data_sample = df_train[col].sample(frac=0.01)
    axs[i, 0].hist(data_sample.to_pandas(), bins=50)
    axs[i, 1].hist(cp.asnumpy(gaussrank_gpu(data_sample.values)), bins=50)
    if i==0:
        axs[i, 0].set_title('Histogram non-normalized')
        axs[i, 1].set_title('Histogram Gauss Rank')
```

<!-- #region id="Qtu_4jsQnbvI" -->
### Timeseries historical events
<!-- #endregion -->

<!-- #region id="cSU7fJsFnijN" -->
Many real-world recommendation systems contain time information. The system normally logs events with a timestamp. Tree-based or deep learning based models usually only uses the information from the datapoint itself for the prediction and they have difficulties to capture relationships over multiple datapoints.

Let's take a look at a simple example. Let's assume we have the interaction events of an itemid, userid and action with the timestamp.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 979} id="AuX2geBynPi2" outputId="90fdf567-8793-4c43-ef6e-a9ab29c6bed5"
itemid = [1000001]*10 + [1000002]*5 + [1000001]*5 + [1000002]*5 + [1000001]*1 + [1000002]*1 + [1000001]*2 + [1000002]*2
itemid += [1000001]*3 + [1000002]*2 + [1000001]*1 + [1000002]*1 + [1000001]*6 + [1000002]*3 + [1000001]*2 + [1000002]*2
userid = np.random.choice(list(range(10000)), len(itemid))
action = np.random.choice(list(range(2)), len(itemid), p=[0.2, 0.8])
timestamp = [pd.to_datetime('2020-01-01')]*15
timestamp += [pd.to_datetime('2020-01-02')]*10
timestamp += [pd.to_datetime('2020-01-03')]*2
timestamp += [pd.to_datetime('2020-01-04')]*4
timestamp += [pd.to_datetime('2020-01-05')]*5
timestamp += [pd.to_datetime('2020-01-07')]*2
timestamp += [pd.to_datetime('2020-01-08')]*9
timestamp += [pd.to_datetime('2020-01-09')]*4

data = pd.DataFrame({
    'itemid': itemid,
    'userid': userid,
    'action': action,
    'timestamp': timestamp
})

data = cudf.from_pandas(data)

data[data['itemid']==1000001]
```

<!-- #region id="rjAoJEpxnv5m" -->
We can extract many interesting features based on the history, such as

- the sum number of actions of the last day, last 3 days or last 7 days
- the average number of actions of the last day, last 3 days or last 7 days
- the average probability of the last day, last 3 days or last 7 days
etc.

In general, these operations are called window function and uses .rolling() function. For each row, the function looks at a window (# of rows around it) and apply a certain function to it.

Current, our data is on a userid and itemid level. First, we need to aggregate it on the level, we want to apply the window function.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 576} id="WucjI8GbnvJn" outputId="73526f99-9231-487e-b20a-cf90d956baec"
data_window = data[['itemid', 'timestamp', 'action']].groupby(['itemid', 'timestamp']).agg(['count', 'sum']).reset_index()
data_window.columns = ['itemid', 'timestamp', 'count', 'sum']
data_window.index = data_window['timestamp']

data_window
```

<!-- #region id="xsGGUP_5o2fJ" -->
We are interested how many positive interaction an item had on the previous day. Next, we want to groupby our dataframe by itemid. Then we apply the rolling function for two days (2D).

Note: To use the rolling function with days, the dataframe index has to by a timestamp.

We can see that every row contains the sum of the row value + the previous row value. For example, itemid=1000001 for data 2020-01-02 counts 15 observations and sums 12 positive interactions.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 576} id="AwGNFUhJo2Gd" outputId="ac728dfe-a325-40db-a935-992c151fdc5f"
offset = '3D'

data_window_roll = data_window[['itemid', 'count', 'sum']].groupby(['itemid']).rolling(offset).sum().drop('itemid', axis=1)
data_window_roll
```

<!-- #region id="Vn9kzhDvpPa-" -->
If we take a look on the calculations, we see that the .rolling() inclues the value from the current row, as well. This could be a kind of data leakage. Therefore, we shift the values by one row.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 545} id="eXwEtz8ipO4L" outputId="2ebcdc4e-12b1-4372-8efa-9ee5835f1bdb"
data_window_roll = data_window_roll.reset_index()
data_window_roll.columns = ['itemid', 'timestamp', 'count_' + offset, 'sum_' + offset]
data_window_roll[['count_' + offset, 'sum_' + offset]] = data_window_roll[['count_' + offset, 'sum_' + offset]].shift(1)
data_window_roll.loc[data_window_roll['itemid']!=data_window_roll['itemid'].shift(1), ['count_' + offset, 'sum_' + offset]] = 0
data_window_roll['avg_' + offset] = data_window_roll['sum_' + offset]/data_window_roll['count_' + offset]
data_window_roll
```

<!-- #region id="FI_TTI-FpVMr" -->
After we calculated the aggregated values and applied the window function, we want to merge it to our original dataframe.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="dOcBt21xpS4Y" outputId="5ee54312-2983-403e-800c-b17ee1113835"
data = data.merge(data_window_roll, how='left', on=['itemid', 'timestamp'])
data
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="YglWt4khpXxJ" outputId="dcd13868-52e8-4291-faa2-f2ef8b2158f8"
offset = '7D'

data_window_roll = data_window[['itemid', 'count', 'sum']].groupby(['itemid']).rolling(offset).sum().drop('itemid', axis=1)
data_window_roll = data_window_roll.reset_index()
data_window_roll.columns = ['itemid', 'timestamp', 'count_' + offset, 'sum_' + offset]
data_window_roll[['count_' + offset, 'sum_' + offset]] = data_window_roll[['count_' + offset, 'sum_' + offset]].shift(1)
data_window_roll.loc[data_window_roll['itemid']!=data_window_roll['itemid'].shift(1), ['count_' + offset, 'sum_' + offset]] = 0
data_window_roll['avg_' + offset] = data_window_roll['sum_' + offset]/data_window_roll['count_' + offset]
data = data.merge(data_window_roll, how='left', on=['itemid', 'timestamp'])
data
```

<!-- #region id="6tnEI33Lpim5" -->
Let's get the # of purchases per product in the 7 days before.
<!-- #endregion -->

```python id="4XjcqI8hp4TO"
# cuDF does not support date32, right now. We use pandas to transform the timestamp in only date values.
df_train['date'] = cudf.from_pandas(pd.to_datetime(df_train['timestamp'].to_pandas()).dt.date)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 479} id="1ugoVj5apa5X" outputId="ed0b4b07-590c-4778-d185-e45215bf80a1"
offset = '7D'

data_window = df_train[['product_id', 'date', 'target']].groupby(['product_id', 'date']).agg(['count', 'sum']).reset_index()
data_window.columns = ['product_id', 'date', 'count', 'sum']
data_window.index = data_window['date']

data_window_roll = data_window[['product_id', 'count', 'sum']].groupby(['product_id']).rolling(offset).sum().drop('product_id', axis=1)
data_window_roll = data_window_roll.reset_index()
data_window_roll.columns = ['product_id', 'date', 'count_' + offset, 'sum_' + offset]
data_window_roll[['count_' + offset, 'sum_' + offset]] = data_window_roll[['count_' + offset, 'sum_' + offset]].shift(1)
data_window_roll.loc[data_window_roll['product_id']!=data_window_roll['product_id'].shift(1), ['count_' + offset, 'sum_' + offset]] = 0
data_window_roll['avg_' + offset] = data_window_roll['sum_' + offset]/data_window_roll['count_' + offset]
data = df_train.merge(data_window_roll, how='left', on=['product_id', 'date'])
data.head()
```

<!-- #region id="Eh07Qws4qJTi" -->
### Differences
<!-- #endregion -->

<!-- #region id="q-V8QzAFqN6R" -->
Another category of powerful features is to calculate the differences to previous datapoints based on a timestamp. For example, we can calculate if the price changed of a product and how much the price change was.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="EUuNlaMvp1bK" outputId="dbdd1b06-3fc8-4fe4-b17c-98be615755a6"
itemid = [1000001]*10 + [1000002]*5 + [1000001]*5 + [1000002]*5 + [1000001]*1 + [1000002]*1 + [1000001]*2 + [1000002]*2
itemid += [1000001]*3 + [1000002]*2 + [1000001]*1 + [1000002]*1 + [1000001]*6 + [1000002]*3 + [1000001]*2 + [1000002]*2
userid = np.random.choice(list(range(10000)), len(itemid))
action = np.random.choice(list(range(2)), len(itemid), p=[0.2, 0.8])

price = [100.00]*10 + [25.00]*5 + [100.00]*5 + [30.00]*5 + [125.00]*1 + [30.00]*1 + [125.00]*2 + [30.00]*2
price += [110.00]*3 + [30.00]*2 + [110.00]*1 + [20.00]*1 + [90.00]*6 + [20.00]*3 + [90.00]*2 + [20.00]*2

timestamp = [pd.to_datetime('2020-01-01')]*15
timestamp += [pd.to_datetime('2020-01-02')]*10
timestamp += [pd.to_datetime('2020-01-03')]*2
timestamp += [pd.to_datetime('2020-01-04')]*4
timestamp += [pd.to_datetime('2020-01-05')]*5
timestamp += [pd.to_datetime('2020-01-07')]*2
timestamp += [pd.to_datetime('2020-01-08')]*9
timestamp += [pd.to_datetime('2020-01-09')]*4

data = pd.DataFrame({
    'itemid': itemid,
    'userid': userid,
    'price': price,
    'action': action,
    'timestamp': timestamp
})

data = cudf.from_pandas(data)

data[data['itemid']==1000001].head(10)
```

<!-- #region id="Zf768ni5qTFf" -->
Tree-based or deep learning based models have difficulties processing these relationships on their own. Providing the models with these features can significantly improve the performance.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="5RF5YmcvqQLT" outputId="315fee83-c501-4dbb-ec93-757ea94c607a"
offset = 1
data_shift = data[['itemid', 'timestamp', 'price']].groupby(['itemid', 'timestamp']).mean().reset_index()
data_shift.columns = ['itemid', 'timestamp', 'mean']
data_shift['mean_' + str(offset)] = data_shift['mean'].shift(1)
data_shift.loc[data_shift['itemid']!=data_shift['itemid'].shift(1), 'mean_' + str(offset)] = None
data_shift['diff_' + str(offset)] = data_shift['mean'] - data_shift['mean_' + str(offset)]
data_shift.head(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="PXhtcPZHqWZo" outputId="1f7c7daa-4133-4ca5-ee74-86137ff9fa95"
data_shift.columns = ['itemid', 'timestamp', 'c1', 'c2', 'price_diff_1']
data_shift.drop(['c1', 'c2'], inplace=True, axis=1)
data_shift.head(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="3D-LeWrSqYl8" outputId="36c1e398-f275-476e-acac-19688805bc2c"
data = data.merge(data_shift, how='left', on=['itemid', 'timestamp'])
data.head()
```

<!-- #region id="t9heMbn5qk38" -->
We can combine techniques of TimeSeries data and chain them together. For example, we can calculate the # of purchases per item and then compare the previous week with a the week, 2, 3 or 5 weeks ago. We can recognize patterns over time.
<!-- #endregion -->

<!-- #region id="z07Fkvh0qrHk" -->
Let's get the price difference of the previous price to the current price per item
<!-- #endregion -->

```python id="ZdaJ2Lz-q_Wl"
# cuDF does not support date32, right now. We use pandas to transform the timestamp in only date values.
df_train['date'] = cudf.from_pandas(pd.to_datetime(df_train['timestamp'].to_pandas()).dt.date)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 479} id="6xArlhsGqlIG" outputId="b73949cb-0e32-4008-e162-af1222204c15"
offset = 1
data_shift = df_train[['product_id', 'date', 'price']].groupby(['product_id', 'date']).mean().reset_index()
data_shift.columns = ['product_id', 'date', 'mean']
data_shift['mean_' + str(offset)] = data_shift['mean'].shift(1)
data_shift.loc[data_shift['product_id']!=data_shift['product_id'].shift(1), 'mean_' + str(offset)] = None
data_shift['diff_' + str(offset)] = data_shift['mean'] - data_shift['mean_' + str(offset)]
data_shift.columns = ['product_id', 'date', 'c1', 'c2', 'price_diff_1']
data_shift.drop(['c1', 'c2'], inplace=True, axis=1)
df_train = df_train.merge(data_shift, how='left', on=['product_id', 'date'])
df_train.head()
```
