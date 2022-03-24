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

<!-- #region id="QBW8NyVnSuK6" -->
# RecSys Evaluation Metrics using RePlay Library
<!-- #endregion -->

<!-- #region id="YGqJzbSIP9vL" -->
**Links**

- https://sberbank-ai-lab.github.io/RePlay/pages/modules/metrics.html
<!-- #endregion -->

```python id="MhhXDw00Ol_H"
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop3.2.tgz
!tar xf spark-3.0.0-bin-hadoop3.2.tgz
!pip install -q findspark
!pip install -q pyspark
```

```python id="-YJwm-W8PB0i"
!pip install replay-rec #v0.6.1
```

```python id="4Ctev1swR4tR"
!pip install ipytest
```

```python id="A6zNhUM9SASe"
import ipytest
ipytest.autoconfig()
```

```python id="Ro_aHv8kOhuB"
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.0.0-bin-hadoop3.2"

import findspark
findspark.init()
```

```python id="gYKEylAPQrfA"
import os
import re

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pyspark.ml.linalg import DenseVector
from pyspark.sql import DataFrame

from replay.metrics import *
from replay.distributions import item_distribution
from replay.metrics.base_metric import sorter
from replay.constants import REC_SCHEMA, LOG_SCHEMA
from replay.session_handler import get_spark_session
```

```python id="AARhI1KISamZ"
import warnings
warnings.filterwarnings('ignore')
```

```python id="SZ_0j36kQpMQ"
def assertDictAlmostEqual(d1: Dict, d2: Dict) -> None:
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        assert_allclose(d1[key], d2[key])


@pytest.fixture
def spark():
    return get_spark_session(1, 1)


@pytest.fixture
def log2(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", datetime(2019, 9, 12), 3.0],
            ["user1", "item5", datetime(2019, 9, 13), 2.0],
            ["user1", "item2", datetime(2019, 9, 17), 1.0],
            ["user2", "item6", datetime(2019, 9, 14), 4.0],
            ["user2", "item1", datetime(2019, 9, 15), 3.0],
            ["user3", "item2", datetime(2019, 9, 15), 3.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", datetime(2019, 8, 22), 4.0],
            ["user1", "item3", datetime(2019, 8, 23), 3.0],
            ["user1", "item2", datetime(2019, 8, 27), 2.0],
            ["user2", "item4", datetime(2019, 8, 24), 3.0],
            ["user2", "item1", datetime(2019, 8, 25), 4.0],
            ["user3", "item2", datetime(2019, 8, 26), 5.0],
            ["user3", "item1", datetime(2019, 8, 26), 5.0],
            ["user3", "item3", datetime(2019, 8, 26), 3.0],
            ["user4", "item2", datetime(2019, 8, 26), 5.0],
            ["user4", "item1", datetime(2019, 8, 26), 5.0],
            ["user4", "item1", datetime(2019, 8, 26), 1.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def long_log_with_features(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            ["u1", "i1", date, 1.0],
            ["u1", "i4", datetime(2019, 1, 5), 3.0],
            ["u1", "i2", date, 2.0],
            ["u1", "i5", date, 4.0],
            ["u2", "i1", date, 1.0],
            ["u2", "i3", datetime(2018, 1, 1), 2.0],
            ["u2", "i7", datetime(2019, 1, 1), 4.0],
            ["u2", "i8", datetime(2020, 1, 1), 4.0],
            ["u3", "i9", date, 3.0],
            ["u3", "i2", date, 2.0],
            ["u3", "i6", datetime(2020, 3, 1), 1.0],
            ["u3", "i7", date, 5.0],
        ],
        schema=["user_id", "item_id", "timestamp", "relevance"],
    )


@pytest.fixture
def short_log_with_features(spark):
    date = datetime(2021, 1, 1)
    return spark.createDataFrame(
        data=[
            ["u1", "i3", date, 1.0],
            ["u1", "i7", datetime(2019, 1, 5), 3.0],
            ["u2", "i2", date, 1.0],
            ["u2", "i10", datetime(2018, 1, 1), 2.0],
            ["u3", "i8", date, 3.0],
            ["u3", "i1", date, 2.0],
            ["u4", "i7", date, 5.0],
        ],
        schema=["user_id", "item_id", "timestamp", "relevance"],
    )


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame(
        [("u1", 20.0, -3.0, "M"), ("u2", 30.0, 4.0, "F")]
    ).toDF("user_id", "age", "mood", "gender")


@pytest.fixture
def item_features(spark):
    return spark.createDataFrame(
        [
            ("i1", 4.0, "cat", "black"),
            ("i2", 10.0, "dog", "green"),
            ("i3", 7.0, "mouse", "yellow"),
            ("i4", -1.0, "cat", "yellow"),
            ("i5", 11.0, "dog", "white"),
            ("i6", 0.0, "mouse", "yellow"),
        ]
    ).toDF("item_id", "iq", "class", "color")


def unify_dataframe(data_frame: DataFrame):
    pandas_df = data_frame.toPandas()
    columns_to_sort_by: List[str] = []

    if len(pandas_df) == 0:
        columns_to_sort_by = pandas_df.columns
    else:
        for column in pandas_df.columns:
            if not type(pandas_df[column][0]) in {
                DenseVector,
                list,
                np.ndarray,
            }:
                columns_to_sort_by.append(column)

    return (
        pandas_df[sorted(data_frame.columns)]
        .sort_values(by=sorted(columns_to_sort_by))
        .reset_index(drop=True)
    )


def sparkDataFrameEqual(df1: DataFrame, df2: DataFrame):
    return pd.testing.assert_frame_equal(
        unify_dataframe(df1), unify_dataframe(df2), check_like=True
    )


def sparkDataFrameNotEqual(df1: DataFrame, df2: DataFrame):
    try:
        sparkDataFrameEqual(df1, df2)
    except AssertionError:
        pass
    else:
        raise AssertionError("spark dataframes are equal")


def del_files_by_pattern(directory: str, pattern: str) -> None:
    """
    Deletes files by pattern
    """
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            os.remove(os.path.join(directory, filename))


def find_file_by_pattern(directory: str, pattern: str) -> Optional[str]:
    """
    Returns path to first found file, if exists
    """
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            return os.path.join(directory, filename)
    return None
```

```python colab={"base_uri": "https://localhost:8080/"} id="hjbJkbenN85D" executionInfo={"status": "ok", "timestamp": 1636102199250, "user_tz": -330, "elapsed": 25125, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eebf5dea-e1c6-4c17-914a-a3a456cfd19d"
%%ipytest

@pytest.fixture
def one_user():
    df = pd.DataFrame({"user_id": [1], "item_id": [1], "relevance": [1]})
    return df


@pytest.fixture
def two_users():
    df = pd.DataFrame(
        {"user_id": [1, 2], "item_id": [1, 2], "relevance": [1, 1]}
    )
    return df


@pytest.fixture
def recs(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", 3.0],
            ["user1", "item2", 2.0],
            ["user1", "item3", 1.0],
            ["user2", "item1", 3.0],
            ["user2", "item2", 4.0],
            ["user2", "item5", 1.0],
            ["user3", "item1", 5.0],
            ["user3", "item3", 1.0],
            ["user3", "item4", 2.0],
        ],
        schema=REC_SCHEMA,
    )


@pytest.fixture
def recs2(spark):
    return spark.createDataFrame(
        data=[["user1", "item4", 4.0], ["user1", "item5", 5.0]],
        schema=REC_SCHEMA,
    )


@pytest.fixture
def empty_recs(spark):
    return spark.createDataFrame(
        data=[],
        schema=REC_SCHEMA,
    )


@pytest.fixture
def true(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", datetime(2019, 9, 12), 3.0],
            ["user1", "item5", datetime(2019, 9, 13), 2.0],
            ["user1", "item2", datetime(2019, 9, 17), 1.0],
            ["user2", "item6", datetime(2019, 9, 14), 4.0],
            ["user2", "item1", datetime(2019, 9, 15), 3.0],
            ["user3", "item2", datetime(2019, 9, 15), 3.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def quality_metrics():
    return [NDCG(), HitRate(), Precision(), Recall(), MAP(), MRR(), RocAuc()]


@pytest.fixture
def duplicate_recs(spark):
    return spark.createDataFrame(
        data=[
            ["user1", "item1", 3.0],
            ["user1", "item2", 2.0],
            ["user1", "item3", 1.0],
            ["user1", "item1", 3.0],
            ["user2", "item1", 3.0],
            ["user2", "item2", 4.0],
            ["user2", "item5", 1.0],
            ["user2", "item2", 2.0],
            ["user3", "item1", 5.0],
            ["user3", "item3", 1.0],
            ["user3", "item4", 2.0],
        ],
        schema=REC_SCHEMA,
    )


def test_test_is_bigger(quality_metrics, one_user, two_users):
    for metric in quality_metrics:
        assert metric(one_user, two_users, 1) == 0.5, str(metric)


def test_pred_is_bigger(quality_metrics, one_user, two_users):
    for metric in quality_metrics:
        assert metric(two_users, one_user, 1) == 1.0, str(metric)


def test_hit_rate_at_k(recs, true):
    assertDictAlmostEqual(
        HitRate()(recs, true, [3, 1]),
        {3: 2 / 3, 1: 1 / 3},
    )


def test_user_dist(log, recs, true):
    vals = HitRate().user_distribution(log, recs, true, 1)["value"].to_list()
    assert_allclose(vals, [0.0, 0.5])


def test_item_dist(log, recs):
    assert_allclose(
        item_distribution(log, recs, 1)["rec_count"].to_list(),
        [0, 0, 1, 2],
    )


def test_ndcg_at_k(recs, true):
    pred = [300, 200, 100]
    k_set = [1, 2, 3]
    user_id = 1
    ground_truth = [200, 400]
    ndcg_value = 1 / np.log2(3) / (1 / np.log2(2) + 1 / np.log2(3))
    assert (
        NDCG()._get_metric_value_by_user_all_k(
            k_set, user_id, pred, ground_truth
        )
        == [(1, 0, 1), (1, ndcg_value, 2), (1, ndcg_value, 3)],
    )
    assertDictAlmostEqual(
        NDCG()(recs, true, [1, 3]),
        {
            1: 1 / 3,
            3: 1
            / 3
            * (
                1
                / (1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4))
                * (1 / np.log2(2) + 1 / np.log2(3))
                + 1 / (1 / np.log2(2) + 1 / np.log2(3)) * (1 / np.log2(3))
            ),
        },
    )


def test_precision_at_k(recs, true):
    assertDictAlmostEqual(
        Precision()(recs, true, [1, 2, 3]),
        {3: 1 / 3, 1: 1 / 3, 2: 1 / 2},
    )


def test_map_at_k(recs, true):
    assertDictAlmostEqual(
        MAP()(recs, true, [1, 3]),
        {3: 11 / 36, 1: 1 / 3},
    )


def test_recall_at_k(recs, true):
    assertDictAlmostEqual(
        Recall()(recs, true, [1, 3]),
        {3: (1 / 2 + 2 / 3) / 3, 1: 1 / 9},
    )


def test_surprisal_at_k(true, recs, recs2):
    assertDictAlmostEqual(Surprisal(true)(recs2, [1, 2]), {1: 1.0, 2: 1.0})

    assert_allclose(
        Surprisal(true)(recs, 3),
        5 * (1 - 1 / np.log2(3)) / 9 + 4 / 9,
    )


def test_unexpectedness_at_k(true, recs, recs2):
    assert Unexpectedness._get_metric_value_by_user(2, (), (2, 3)) == 0
    assert Unexpectedness._get_metric_value_by_user(2, (1, 2), (1,)) == 0.5


def test_coverage(true, recs, empty_recs):
    coverage = Coverage(recs.union(true.drop("timestamp")))
    assertDictAlmostEqual(
        coverage(recs, [1, 3, 5]),
        {1: 0.3333333333333333, 3: 0.8333333333333334, 5: 0.8333333333333334},
    )
    assertDictAlmostEqual(
        coverage(empty_recs, [1, 3, 5]),
        {1: 0.0, 3: 0.0, 5: 0.0},
    )


def test_bad_coverage(true, recs):
    assert_allclose(Coverage(true)(recs, 3), 1.25)


def test_empty_recs(quality_metrics):
    for metric in quality_metrics:
        assert_allclose(
            metric._get_metric_value_by_user(
                k=4, pred=[], ground_truth=[2, 4]
            ),
            0,
            err_msg=str(metric),
        )


def test_bad_recs(quality_metrics):
    for metric in quality_metrics:
        assert_allclose(
            metric._get_metric_value_by_user(
                k=4, pred=[1, 3], ground_truth=[2, 4]
            ),
            0,
            err_msg=str(metric),
        )


def test_not_full_recs(quality_metrics):
    for metric in quality_metrics:
        assert_allclose(
            metric._get_metric_value_by_user(
                k=4, pred=[4, 1, 2], ground_truth=[2, 4]
            ),
            metric._get_metric_value_by_user(
                k=3, pred=[4, 1, 2], ground_truth=[2, 4]
            ),
            err_msg=str(metric),
        )


def test_duplicate_recs(quality_metrics, duplicate_recs, recs, true):
    for metric in quality_metrics:
        assert_allclose(
            metric(k=4, recommendations=duplicate_recs, ground_truth=true),
            metric(k=4, recommendations=recs, ground_truth=true),
            err_msg=str(metric),
        )


def test_sorter():
    result = sorter(((1, 2), (2, 3), (3, 2)))
    assert result == [2, 3]


def test_sorter_index():
    result = sorter([(1, 2, 3), (2, 3, 4), (3, 3, 5)], index=2)
    assert result == [5, 3]
```
