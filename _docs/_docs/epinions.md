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

<!-- #region id="pej35mf5F4Rj" -->
# Epinions Data Preprocessing
> Loading and transformation of epinions user item interaction dataset

- toc: true
- badges: true
- comments: true
- categories: [data processing]
<!-- #endregion -->

```python id="ErXSaQ9RGWV_"
!pip install lenskit
```

```python colab={"base_uri": "https://localhost:8080/"} id="hTa3ROMZGeYW" outputId="a1962f18-7bdf-49b5-8581-8948ddf4cf72"
!wget -q --show-progress https://github.com/RecoHut-Datasets/epinions/raw/v1/trust_data.txt
```

```python id="aOkn7m2pGod4"
import pandas as pd
import lenskit.crossfold as xf
import numpy as np
import json
```

```python id="id3lZcQXGyKY" colab={"base_uri": "https://localhost:8080/"} outputId="4afd4a84-9086-4292-85f9-576e5b8c5bef"
ratings = pd.read_csv('trust_data.txt', header=None, index_col=None, sep=' ')
ratings.dropna(axis=1, how='all', inplace=True) 
columns = ['user', 'item', 'rating']
ratings.columns = columns
print(ratings.head())
```

```python colab={"base_uri": "https://localhost:8080/"} id="lnHJJIXLG7jd" outputId="eac78cb0-e56e-4678-bdfc-ed80b339fc5c"
n_user = len(pd.unique(ratings.user))
n_item = len(pd.unique(ratings.item))

print("Num_of_users: {}\nNum_of_items: {}".format(n_user, n_item))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="qWH3j4SNPTTI" outputId="cf883704-3777-4ceb-e260-1b79bee9a14d"
ratings.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="LsdLwUbUIcrV" outputId="0f667bfd-f457-4429-9014-c0f298bf8df5"
df_25 = ratings[ratings.user.isin(ratings.user.value_counts()[ratings.user.value_counts() >= 25].index)]
df_25 = df_25.reset_index(drop=True)
print("\033[4mCount after only keeping users with at least 25 relevant interactions\033[0m")
print("Num_of_users: {}\nNum_of_items: {}\nTotal_interactions: {}".format(len(pd.unique(df_25.user)), len(pd.unique(df_25.item)), len(df_25)))
```

```python colab={"base_uri": "https://localhost:8080/"} id="qASWgDMHPFeK" outputId="5535886d-5faf-49cf-bad2-c1d17a88adbf"
print(df_25.head())
```

```python id="vIIkD0GBJMfE"
def get_unique_id(data_pd: pd.DataFrame, column: str) -> (dict, pd.DataFrame):
	"""
	clear the ids
	:param data_pd: pd.DataFrame 
	:param column: specified col
	:return: dict: {value: id}
	"""
	new_column = '{}_id'.format(column)
	assert new_column not in data_pd.columns
	temp = data_pd.loc[:, [column]].drop_duplicates().reset_index(drop=True)
	temp[new_column] = temp.index
	temp.index = temp[column]
	del temp[column]
	# data_pd.merge()
	data_pd = pd.merge(left=data_pd,
		right=temp,
		left_on=column,
		right_index=True,
		how='left')

	return temp[new_column].to_dict(), data_pd
```

```python id="FtaemIZSJf-2" colab={"base_uri": "https://localhost:8080/"} outputId="42fb51df-ee68-4ad7-db81-4b8ac5e8add2"
_, df_25 = get_unique_id(df_25, 'user')
_, df_25 = get_unique_id(df_25, 'item')
print(df_25.head())
```

```python colab={"base_uri": "https://localhost:8080/"} id="-HPTdD6dJg0C" outputId="e814e61d-9e09-4039-d462-008b0adea60a"
n_user = df_25.user_id.drop_duplicates().size
n_item = df_25.item_id.drop_duplicates().size
print(n_user, n_item)
```

```python id="QAIOH1exJgxf"
import os

dataset_meta_info = {'dataset_size': len(df_25),
                     'user_size': n_user,
                     'item_size': n_item
                     }
with open(os.path.join('dataset_meta_info.json'), 'w') as f:
	json.dump(dataset_meta_info, f) 
```

```python colab={"base_uri": "https://localhost:8080/"} id="QhpdSqNCJgu9" outputId="15cd2ccf-167b-40cc-f41d-5c4fc7acb13a"
seeds = [1, 777, 1992, 2003, 2020]

for j in range(len(seeds)):
	for i, tp in enumerate(xf.partition_users(df_25, partitions=1, method=xf.SampleN(20), rng_spec=seeds[j])):
		save_path = '.'
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		train = tp.test
		test = tp.train

		train.to_csv(os.path.join(save_path, 'train.csv'))
		test.to_csv(os.path.join(save_path, 'test.csv'))
		print(len(tp.train))
		print(len(tp.test))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="JkWsWld0LArn" outputId="f0dd0cae-f48d-451f-89a1-6f901f1e10e8"
train_df = pd.read_csv('train.csv', index_col=0)
train_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="7TS-BvKDLAoh" outputId="354d9c3c-f00d-40f5-a5d5-2274e7ac53cd"
train_df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="iEe13D6ALAlS" outputId="ecb1b33e-450b-4da6-b4fa-be874b706ca7"
train_df.describe().T
```

<!-- #region id="BIyZZlffKTpy" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="x1bfRShBKU-l" outputId="5c570644-2391-4113-cc8f-b1bf13fc91c2"
!ls -al .
```

```python colab={"base_uri": "https://localhost:8080/"} id="uFongZU4KTp2" outputId="efcf3245-3866-4fd8-82ae-fc40b0273419"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="O1HWMD6TKTp4" -->
---
<!-- #endregion -->

<!-- #region id="t5RatZHVKTp5" -->
**END**
<!-- #endregion -->
