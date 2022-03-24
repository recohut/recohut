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

<!-- #region id="VGKjc2Bda-Qn" -->
# Trivago Popular Destination Recommender
> Using Recsys 2019 challange Trivago travel dataset to build popularity based model and recommending popular most clicked destinations to the users

- toc: true
- badges: true
- comments: true
- categories: [Trivago, Popularity, Travel]
- image:
<!-- #endregion -->

<!-- #region id="l2Tn8P3LWLII" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="69ynq4AbT8sc" outputId="f48b1fc5-f427-4e06-e418-58a60b6a9ec0"
!pip install -q git+https://github.com/sparsh-ai/recochef.git
```

```python id="yfD8l7n3VqUS"
import math
import pandas as pd
import numpy as np

from recochef.datasets.trivago import Trivago
```

<!-- #region id="u0CbvZnWWMo4" -->
## Data loading
<!-- #endregion -->

```python id="tdC6SWFyWHg3"
trivago = Trivago()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="gMSx11nvV0ua" outputId="e369a779-3a89-482e-a3c9-12e4ed6af22d"
df_train = trivago.load_train()
df_train.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="R6swP2ZlWOgO" outputId="c09e3785-23ad-4add-e8b6-efb6bde76c5b"
df_test = trivago.load_test()
df_test.head()
```

<!-- #region id="TDZvN5GHb4HT" -->
## Utilities
<!-- #endregion -->

```python id="LnRG_Du0XBoQ"
GR_COLS = ["USERID", "SESSIONID", "TIMESTAMP", "STEP"]
```

```python id="f8yIXz1tXE3V"
def get_submission_target(df):
    """Identify target rows with missing click outs."""

    mask = df["REFERENCE"].isnull() & (df["EVENTTYPE"] == "clickout item")
    df_out = df[mask]

    return df_out
```

```python id="jWY4ucZqYv-E"
def get_popularity(df):
    """Get number of clicks that each item received in the df."""

    mask = df["EVENTTYPE"] == "clickout item"
    df_clicks = df[mask]
    df_item_clicks = (
        df_clicks
        .groupby("REFERENCE")
        .size()
        .reset_index(name="NCLICKS")
        .transform(lambda x: x.astype(int))
    )

    return df_item_clicks
```

```python id="9ycY6ZibZB6G"
def string_to_array(s):
    """Convert pipe separated string to array."""

    if isinstance(s, str):
        out = s.split("|")
    elif math.isnan(s):
        out = []
    else:
        raise ValueError("Value must be either string of nan")
    return out
```

```python id="x2NZTyj-ZJO4"
def explode(df_in, col_expl):
    """Explode column col_expl of array type into multiple rows."""

    df = df_in.copy()
    df.loc[:, col_expl] = df[col_expl].apply(string_to_array)

    df_out = pd.DataFrame(
        {col: np.repeat(df[col].values,
                        df[col_expl].str.len())
         for col in df.columns.drop(col_expl)}
    )

    df_out.loc[:, col_expl] = np.concatenate(df[col_expl].values)
    df_out.loc[:, col_expl] = df_out[col_expl].apply(int)

    return df_out
```

```python id="SKS19F0jZPp8"
def group_concat(df, gr_cols, col_concat):
    """Concatenate multiple rows into one."""

    df_out = (
        df
        .groupby(gr_cols)[col_concat]
        .apply(lambda x: ' '.join(x))
        .to_frame()
        .reset_index()
    )

    return df_out
```

```python id="ZfhxnNMaVv-g"
def calc_recommendation(df_expl, df_pop):
    """Calculate recommendations based on popularity of items.
    The final data frame will have an impression list sorted according to the number of clicks per item in a reference data frame.
    :param df_expl: Data frame with exploded impression list
    :param df_pop: Data frame with items and number of clicks
    :return: Data frame with sorted impression list according to popularity in df_pop
    """

    df_expl_clicks = (
        df_expl[GR_COLS + ["IMPRESSIONS"]]
        .merge(df_pop,
               left_on="IMPRESSIONS",
               right_on="REFERENCE",
               how="left")
    )

    df_out = (
        df_expl_clicks
        .assign(IMPRESSIONS=lambda x: x["IMPRESSIONS"].apply(str))
        .sort_values(GR_COLS + ["NCLICKS"],
                     ascending=[True, True, True, True, False])
    )

    df_out = group_concat(df_out, GR_COLS, "IMPRESSIONS")
    df_out.rename(columns={'IMPRESSIONS': 'ITEM_RECOMMENDATIONS'}, inplace=True)

    return df_out
```

<!-- #region id="ZH51BRUebt_Y" -->
## Getting popular items
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 376} id="Pe_JDBqvZwnz" outputId="774761bb-2050-438a-8465-b17f5bea1c96"
print("Get popular items...")
df_popular = get_popularity(df_train)
df_popular.sort_values(by='NCLICKS', ascending=False).head(10)
```

<!-- #region id="WY1Bvf_Xbw7z" -->
## Identify target users
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 634} id="d0RqGTj3ZwjM" outputId="4b182348-bd76-4d64-ab0b-21e09ba9f264"
print("Identify target rows...")
df_target = get_submission_target(df_test)
df_target.head(10)
```

<!-- #region id="nYh_qJj7bqyM" -->
## Recommendations
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 376} id="E28J3TWkZ6QP" outputId="23e128c6-8986-4c43-8696-b504d6304ba9"
print("Get recommendations...")
df_expl = explode(df_target, "IMPRESSIONS")
df_out = calc_recommendation(df_expl, df_popular)
df_out.head(10)
```
