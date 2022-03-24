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
    language: python
    name: python3
---

<!-- #region id="djWxyJxqo7Yv" -->
# LightFM Hybrid Model on ML-10m using RePlay Library
<!-- #endregion -->

```python id="yMD0Oyucog24"
K=10
SEED=1234
```

<!-- #region id="j0zLOCnRog25" -->
The notebook contains an example of LightFM model usage and dataset preprocessing with RePlay, including:
1. Data loading
2. Features preprocessing with pyspark
3. Hyperparameters tuning and building LightFM model based on interaction matrix
4. Model evaluation
5. Hyperparameters tuning and building LightFM model based on interaction matrix and features
<!-- #endregion -->

<!-- #region id="2fTaEkFVog28" -->
# 1) Data loading
<!-- #endregion -->

<!-- #region id="uoOgyQdnog29" -->
We will use MovieLens 10m dataset from rs_datasets package, which contains a list of recommendations datasets.
<!-- #endregion -->

```python id="xkbuNKKfog2-" outputId="7ff491ba-89ca-4a6a-fe16-f71501eba0bb"
from rs_datasets import MovieLens

data = MovieLens("10m")
data.info()
```

<!-- #region id="fdKPjVmKog3D" -->
### Convert interaction log to RePlay format
<!-- #endregion -->

```python id="oo67aIRyog3E"
from replay.data_preparator import DataPreparator

log = DataPreparator().transform(
    data=data.ratings,
    columns_names={
        "user_id": "user_id",
        "item_id": "item_id",
        "relevance": "rating",
        "timestamp": "timestamp"
    }
)
```

<!-- #region id="AaTp8jtWog3F" -->
### Data split
<!-- #endregion -->

```python id="m0AONToVog3F"
from replay.splitters import UserSplitter

user_random_splitter = UserSplitter(
    item_test_size=K,
    user_test_size=500,
    drop_cold_items=True,
    drop_cold_users=True,
    shuffle=True,
    seed=SEED
)
```

```python id="wX_-9LUIog3G" outputId="1bc10ac7-5421-4b70-d024-68d18847fd9b"
train, test = user_random_splitter.split(log)
train.count(), test.count()
```

```python id="-OGyzPJAog3H" outputId="7a5bca01-700d-4cfc-bd63-5caf261e1f90"
train_opt, val_opt = user_random_splitter.split(train)
train_opt.count(), val_opt.count()
```

<!-- #region id="Et1n8udXog3H" -->
# 2) Features preprocessing with pyspark
<!-- #endregion -->

<!-- #region id="niY9Ouz-og3H" -->
### Convert features to RePlay format
<!-- #endregion -->

```python id="4H5lfFqzog3I" outputId="190f4ed4-2c48-456a-b560-0d087596c349"
%%time
item_features = DataPreparator().transform(
    data=data.items,
    columns_names={
        "item_id": "item_id"
    }
)
```

```python id="vGhYTfiCog3K" outputId="2b5db740-9d79-4523-9f7f-e42176027249"
item_features.show(2)
```

<!-- #region id="o13r0hegog3L" -->
#### Year
<!-- #endregion -->

```python id="aZBi95GWog3M"
from pyspark.sql import functions as sf
from pyspark.sql.types import IntegerType
```

```python id="Mo1fffGTog3M" outputId="86c3956d-b1af-4a91-d334-17a86ba3cf86"
year = item_features.withColumn('year', sf.substring(sf.col('title'), -5, 4).astype(IntegerType())).select('item_id', 'year')
year.show(2)
```

<!-- #region id="ldcCIuFYog3N" -->
#### Genres
<!-- #endregion -->

```python id="NEYxry8eog3N"
from replay.session_handler import State
from pyspark.sql.functions import split

genres = (
    State().session.createDataFrame(data.items[["item_id", "genres"]])
    .select(
        "item_id",
        split("genres", "\|").alias("genres")
    )
)
```

```python id="_rqd3gNCog3N" outputId="c420f13e-e9cd-4b52-affe-2f518534ae3d"
genres.show()
```

```python id="ZlJ9qLtMog3O"
from pyspark.sql.functions import explode

genres_list = (
    genres.select(explode("genres").alias("genre"))
    .distinct().filter('genre <> "(no genres listed)"')
    .toPandas()["genre"].tolist()
)
```

```python id="LBrZjIALog3O" outputId="dba637b0-0567-4030-e6a4-6f1458791fbb"
genres_list
```

```python id="e2KNudV9og3P" outputId="39cdc682-5cb0-42ac-f7e1-ee1ccbe2ae5a"
from pyspark.sql.functions import col, lit, array_contains
from pyspark.sql.types import IntegerType

item_features = genres
for genre in genres_list:
    item_features = item_features.withColumn(
        genre,
        array_contains(col("genres"), genre).astype(IntegerType())
    )
item_features = item_features.drop("genres").cache()
item_features.count()
```

```python id="BVY6fC5Qog3Q" outputId="bdb3442a-baed-4491-e298-599f8bcef06d"
item_features.show(2)
```

```python id="PrwMze0Zog3R" outputId="50b7f8db-0f4a-4f19-c4ce-c4c4b8587a4f"
item_features = item_features.join(year, on='item_id', how='inner')
item_features.cache()
item_features.count()
```

<!-- #region id="20m9DF2Hog3S" -->
# 3) Hyperparameters tuning and building LightFM model based on interaction matrix
<!-- #endregion -->

```python id="V6-Wyvw-og3S"
from replay.models import LightFMWrap

# fixing warp loss as it usually shows better performance
model = LightFMWrap(random_state=SEED, loss='warp')
```

<!-- #region id="U53GGTkkog3T" -->
To see the default model search space, get ``_search_space`` attribute
<!-- #endregion -->

```python id="j2tzMNmpog3T" outputId="ebdc1804-967d-4fc8-d4cc-e94ef5841558"
model._search_space
```

<!-- #region id="lHtm3eAIog3T" -->
We will fix loss function and optimize a number of components only
<!-- #endregion -->

```python id="h4eJjUkVog3U"
best_params = model.optimize(train=train_opt, test=val_opt, param_grid={'no_components':[5, 128]}, budget=10)
```

```python id="4pKJH6ANog3U" outputId="1188b67b-a188-4558-df5b-26f065c57598"
best_params
```

```python id="PbLVUmW6og3U"
model = LightFMWrap(random_state=SEED, loss='warp', **best_params)
```

```python id="hiTd01Ozog3V" outputId="75782a2b-4abf-4b47-fa00-191924b7f400"
%%time
model.fit(train)
```

```python id="bWOYxQYDog3W" outputId="2b9096de-b7d5-4638-eb19-0be218b10bb9"
%%time
recs = model.predict(
    k=K,
    users=test.select('user_id').distinct(),
    log=train,
    filter_seen_items=True
)
```

<!-- #region id="h1QObNfHog3W" -->
# 4) Model evaluation
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} id="1OxHoZhCog3X"
from replay.metrics import HitRate, NDCG, MAP
from replay.experiment import Experiment

metrics = Experiment(test, {NDCG(): K,
                            MAP() : K,
                            HitRate(): [1, K]})

```

```python id="HhsdE_baog3X" outputId="2b4093b3-19ac-4701-860e-fdcc12c1cb04"
%%time
metrics.add_result("LightFM_no_features", recs)
metrics.results
```

<!-- #region id="MRVy42vgog3Y" -->
# 5) Hyperparameters tuning and building LightFM model based on interaction matrix and features
<!-- #endregion -->

```python id="BN7_UPxvog3Y"
from replay.models import LightFMWrap

model_feat = LightFMWrap(random_state=SEED, loss='warp')
```

```python id="LWzRRxj7og3Y"
best_params = model_feat.optimize(train=train_opt, test=val_opt, param_grid={'no_components':[5, 256]}, budget=10, item_features=item_features)
```

```python id="c5wvg8vGog3Y" outputId="82bc8121-ce16-4468-e966-6f491fa3c0d0"
best_params
```

<!-- #region id="qG43T_kzog3Z" -->
#### best_params
<!-- #endregion -->

```python id="anpUfmbHog3Z"
model_feat = LightFMWrap(random_state=SEED, **best_params, loss='warp')
```

```python id="ccnTHdu6og3Z" outputId="ed098a35-c7c1-4b49-ce0a-a6c9a0f058bc"
%%time
model_feat.fit(train, item_features=item_features)
```

```python id="I2xaS4kKog3Z" outputId="06bbc92d-33f6-48ac-fda9-68872515516b"
%%time
recs = model_feat.predict(
    k=K,
    users=test.select('user_id').distinct(),
    log=train,
    filter_seen_items=True,
    item_features=item_features
)
```

```python id="pRpWsPYvog3Z" outputId="19e0e4f2-7a5e-4712-e84d-109da676c3f9"
metrics.add_result("LightFM_item_features", recs)
metrics.results
```

<!-- #region id="6yxfsfG4og3a" -->
In our experiment item features did not improve quality, but in general features might be helpful especially for making recommendations for cold users.

<!-- #endregion -->

```python id="eEagr18Wog3a"

```
