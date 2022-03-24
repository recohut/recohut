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

# Booking.com trip recommendation part 2 - EDA
> Booking.com challenge on trip recommendation part 2

- toc: true
- badges: true
- comments: true
- categories: [travel, eda]
- image: 

```python id="YrHhkJNbghNP"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
%matplotlib inline

sns.set_theme(style="ticks")
```

```python id="LR03pKu4hTyH"
!wget https://github.com/sparsh-ai/reco-data/raw/master/BookingChallenge.zip
!unzip BookingChallenge.zip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="2EluESXAl189" outputId="023c490f-09a9-4ec3-a89f-2565cd5d400c"
df = pd.read_csv('train_set.csv',
                 dtype={"user_id": str, "city_id": str, 
                        'affiliate_id': str, 'utrip_id': str},
                 date_parser=['checkin', 'checkout'])

df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="J0gNMVvzl16Y" outputId="a8413122-40b2-4342-c754-99be3c1907f5"
df['checkin']  = pd.to_datetime(df['checkin'])
df['checkout'] = pd.to_datetime(df['checkout'])
df['duration'] = (df['checkout'] - df['checkin']).dt.days

df_group_checkin = df.groupby('checkin').agg({'user_id': 'count', 'duration': 'mean'})
df_group_checkin['duration_7d'] = df_group_checkin['duration'].rolling(window=7).mean()

df_group_checkin.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 454} id="GSvgSg69l135" outputId="478c343e-8e91-409e-b59e-aa7b0a282f5e"
g = sns.relplot(data=df_group_checkin, x="checkin", y="user_id", kind="line", height=6)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 454} id="adaeCKKrl10h" outputId="cdd81eeb-faad-452c-ad91-cc7bba3e8095"
g = sns.relplot(data=df_group_checkin, x="checkin", y="duration_7d", kind="line", height=6)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="EWA3iHxtl1wv" outputId="58fd2dcc-07bb-47a1-df9b-6bc4365b5c09"
df = df.iloc[0:1000]

df['checkin_month']      = df.checkin.dt.month
df['checkin_dayofweek']  = df.checkin.dt.dayofweek
df['checkin_quarter']    = df.checkin.dt.quarter
df['checkin_is_weekend'] = df.checkin.apply(lambda x : 1 if x.day_name in ['Saturday','Sunday'] else 0)

df['checkin_str']  = df['checkin'].astype(str)
df['checkout_str']  = df['checkout'].astype(str)
df['step'] = 1
df['step']= df.groupby(['utrip_id']).step.cumsum()

df.head()
```

```python id="jL5yY_78oH7a"
def count_hotel(hotel_country):
    return len(list(np.unique(hotel_country)))

def list_without_last(itens):
    return list(itens[:-1])

def list_without_last_and_pad(pad=5, dtype=int):
    def add_pad(items): 
        arr = list_without_last(items)
        arr = list(([dtype(0)] * (pad - len(arr[-pad:])) + arr[-pad:])) 
        return arr
    return add_pad
```

```python colab={"base_uri": "https://localhost:8080/", "height": 510} id="hia7HbV_oSAM" outputId="6ff28c71-90c1-4e6b-8c7b-a79178137ded"
df_trip = df.sort_values(['checkin']).groupby(['utrip_id']).agg(
    user_id=('user_id', 'first'),
    count_unique_city=('city_id', count_hotel),
    trip_size=('checkin', len),
    start_trip=('checkin', 'first'),
    checkin_list=('checkin_str', list_without_last_and_pad(5, str)),
    checkout_list=('checkout', list_without_last_and_pad(5)),
    duration_list=('duration', list_without_last_and_pad(5, int)),
    city_id_list=('city_id', list_without_last_and_pad(5, str)),
    device_class_list=('device_class', list_without_last_and_pad(5, str)),
    affiliate_id_list=('affiliate_id', list_without_last_and_pad(5, str)),
    booker_country_list=('booker_country', list_without_last_and_pad(5, str)),
    hotel_country_list=('hotel_country', list_without_last_and_pad(5, str)),
    step_list=('step', list_without_last_and_pad(5, int)),
    last_city_id=('city_id', 'last')
)

df_trip['end_trip']  = df_trip['checkout_list'].apply(lambda x: x[-1] if len(x) > 1 else None)
df_trip = df_trip.loc[df_trip['end_trip']!=0,:]
df_trip['end_trip'] = pd.to_datetime(df_trip['end_trip'])
df_trip['duration']  = (df_trip['end_trip'] - df_trip['start_trip']).dt.days

df_trip.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="JEZ2Jo_5oVQM" outputId="67fc7ed4-4676-420d-818a-9f1f483fd635"
df_city_group = df.groupby('city_id').agg({'checkin': 'count', 'duration': 'mean'})\
                    .sort_values('checkin',ascending=False)

df_city_month_group = df.groupby(['city_id', 'checkin_month']).agg({'checkin': 'count', 'duration': 'mean'})\
                    .reset_index().sort_values(['city_id', 'checkin_month', 'checkin'],ascending=False)

city_idx = list(df_city_group.index)[:50]

df_plot  = df_city_month_group[df_city_month_group.city_id.isin(city_idx)]

grid     = sns.FacetGrid(df_plot, col="city_id", hue="city_id", palette="tab20c",
                     col_wrap=5, height=2.5)
grid.map(plt.plot, "checkin_month", "checkin", marker="o")
grid.fig.tight_layout(w_pad=1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 775} id="4OwSTDsOovT4" outputId="5a390b26-6fb9-427c-c764-12bb96f005c4"
def count_hotel(hotel_country):
    return len(list(np.unique(hotel_country)))

def join_city(city):
    return "_".join(list(city))

df_utrip_id_group = df.groupby('utrip_id').agg({'checkin': 'count', 
                                                'duration': ['mean', 'sum'], 
                                                'hotel_country': count_hotel,
                                                'city_id': join_city})
df_utrip_id_group.columns = ["_".join(pair) for pair in df_utrip_id_group.columns]
df_utrip_id_group = df_utrip_id_group.sort_values('checkin_count', ascending=False)

df_utrip_id_group['multiply_country'] = (df_utrip_id_group['hotel_country_count_hotel'] > 1).astype(int)

sns.pairplot(df_utrip_id_group, hue="multiply_country")
```

```python id="JhEyT3vVqei-"

```
