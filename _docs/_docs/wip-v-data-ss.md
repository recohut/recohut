---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: 'Python 3.8.5 64-bit (''venv'': venv)'
    name: python_defaultSpec_1600126418091
---

```python
path = '.'
```

```python
import os
import numpy as np
import pandas as pd

import pickle
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import category_encoders as ce
```

```python tags=[]
orders_df = pd.read_csv(os.path.join(path,'orders.csv'), header=0, dtype={'order_id': 'str', 'clientId': 'str'})
orders_df.info()
```

```python tags=[]
events_df = pd.read_csv(os.path.join(path,'events.csv'), header=0, dtype={'order_id': 'str', 'clientId': 'str'}, parse_dates=['@timestamp'])
# events_df = events_df.head(100000)
events_df.info()
```

```python
def preprocess_orders(ao):
  ao = ao.drop_duplicates(subset=['order_id'], keep='last')
  ao['order_id'] = ao['order_id'].astype('str')
  ao['class'] = ao['class'].astype('str')
  ao['age'] = ao['age'].astype('float').fillna(ao['age'].mean()).round(2)
  return ao
```

```python
orders_dfx = preprocess_orders(orders_df)
orders_dfx.head()
```

```python
def preprocess_events(df):

    TIMEOUT = 300
    action_vals = ['add_to_cart', 'begin_checkout', 'checkout_progress', 'conversion', 'purchase', 'remove_from_cart', 'view_item', 'view_item_list']
    df = df.loc[df.hitType == 'event']
    df = df.loc[df.eventAction.isin(action_vals) ]

    df_cols = ['@timestamp', 'order_id', 'eventAction', 'value', 'headers.http_user_agent', 'items.quantity', 'items.category', 'items.price']

    df = df[df_cols]

    # sort by timestamp
    df['@timestamp'] = pd.to_datetime(df['@timestamp'])
    df = df.sort_values(by='@timestamp', ascending=False)

    df['order_id'] = df['order_id'].fillna('undefined').astype('str')
    df['eventAction'] = df['eventAction'].fillna('#')
    df['items.category'] = df['items.category'].fillna('undefined')
    df['items.quantity'] = df['items.quantity'].fillna(0).astype('int')
    df['items.price'] = df['items.price'].fillna(0).astype('float')
    df['value'] = df['value'].fillna(0).astype('int')
    df['eventCount'] = 1
    df['conversions'] = df['eventAction'].apply(lambda x: 1 if x == 'conversion' else 0 )

    # aggregate events
    from ua_parser import user_agent_parser

    agg_df = df.groupby(['order_id'])['items.quantity', 'items.price', 'eventCount', 'conversions'].sum()
    first_df = df.groupby(['order_id']).first()
    first_df = first_df[['headers.http_user_agent']]
    first_df['browser'] = first_df['headers.http_user_agent'].apply(lambda x: user_agent_parser.ParseUserAgent(x).get('family', 'unknown'))
    first_df['os'] = first_df['headers.http_user_agent'].apply(lambda x: user_agent_parser.ParseOS(x).get('family', 'unknown'))
    first_df['device'] = first_df['headers.http_user_agent'].apply(lambda x: user_agent_parser.ParseDevice(x).get('brand', 'unknown'))
    first_df['device'] = first_df['device'].fillna('unknown').astype('str')

    value_df = df.loc[df['eventAction'] == 'conversion'].groupby('order_id').first()['value']

    agg_df = pd.merge(agg_df, first_df, on='order_id')
    agg_df = pd.merge(agg_df, value_df, on='order_id')
    agg_df.drop(columns=['headers.http_user_agent', 'items.price'], inplace=True)

    # time dynamic variables
    activity_df = df.copy()
    activity_df = activity_df.sort_values(by='@timestamp', ascending=False)
    activity_df['@timestamp'] = pd.to_datetime(activity_df['@timestamp'])
    activity_df['hourofday'] = activity_df['@timestamp'].dt.hour
    activity_df['dayofmonth'] = activity_df['@timestamp'].dt.day
    activity_df['weekofyear'] = activity_df['@timestamp'].dt.week
    #   activity_df['monthofyear'] = activity_df['@timestamp'].dt.month
    # time delta
    activity_df['timedelta'] = (activity_df.groupby('order_id')['@timestamp'].transform('max')-activity_df.groupby('order_id')['@timestamp'].transform('min'))
    activity_df['timedelta'] = pd.to_timedelta(activity_df['timedelta'], unit='h').dt.components['hours']

    activity_df['time_diff'] = activity_df.groupby('order_id')['@timestamp'].diff(periods=-1)
    activity_df['time_spent'] = activity_df['time_diff']/np.timedelta64(1,'s')
    activity_df['time_spent'] = activity_df['time_spent'].mask(activity_df['time_spent'].gt(TIMEOUT), TIMEOUT)
    activity_df['time_spent'] =activity_df['time_spent'].fillna(0).astype('int')

    activity_df = activity_df[['order_id', 'timedelta', 'hourofday', 'dayofmonth', 'weekofyear', 'time_spent', 'eventAction', 'items.category', 'items.price']]

    return agg_df, activity_df
```

```python
agg_df, activity_df = preprocess_events(events_df)
```

```python
agg_df.head()
```

```python
agg_df['device'].value_counts()
```

```python
agg_df['browser'].value_counts()
```

```python
agg_df['os'].value_counts()
```

```python
activity_df.head()
```

```python
ts_size = 50
```

```python
activity_df = activity_df.groupby('order_id').head(ts_size).set_index('order_id')
```

```python tags=[]
activity_df.head()
```

```python
def attr_encoder(df):
    scaler_value = StandardScaler()
    le_browser = preprocessing.LabelEncoder()
    le_os = preprocessing.LabelEncoder()
    le_device = preprocessing.LabelEncoder()

    le_browser.fit(df['browser'])
    pickle.dump(le_browser, open('v3_3_le_browser.pkl', 'wb'))
    le_os.fit(df['os'])
    pickle.dump(le_os, open('v3_3_le_os.pkl', 'wb'))
    le_device.fit(df['device'])
    pickle.dump(le_device, open('v3_3_le_device.pkl', 'wb'))
    
    s_cols = ['items.quantity', 'value', 'eventCount']
    scaler_value.fit(df[s_cols])
    pickle.dump(scaler_value, open('v3_3_scaler_value.pkl', 'wb'))

    df['browser'] = le_browser.transform(df['browser'])
    df['os'] = le_os.transform(df['os'])
    df['device'] = le_device.transform(df['device'])
    df[s_cols] = scaler_value.transform(df[s_cols])
    df['value'] = df['value'].round(2)

    return df
```

```python
agg_dfx = attr_encoder(agg_df)
agg_dfx.head()
```

```python
def ts_encoder(df):
    df = pd.get_dummies(df, columns = ['eventAction'], drop_first=True)
    scaler_price = StandardScaler()
    # le_action = preprocessing.LabelEncoder()
    # le_itemcat = preprocessing.LabelEncoder()
    # le_action.fit(df['eventAction'])
    # le_itemcat.fit(df['items.category'])
    ts_cols = ['items.price', 'time_spent']
    scaler_price.fit(df[ts_cols])
    pickle.dump(scaler_price, open('v3_3_scaler_price.pkl', 'wb'))
    # df['eventAction'] = le_action.transform(df['eventAction'])
    # df['items.category'] = le_itemcat.transform(df['items.category'])
    df[ts_cols] = scaler_price.transform(df[ts_cols])
    df['items.price'] = df['items.price'].round(2)

    return df
```

```python
activity_df = ts_encoder(activity_df)
activity_df.head()
```

```python
activity_df.dtypes
```

```python
timedelta_df = activity_df.groupby('order_id')['timedelta'].apply(list)
timedelta_df = timedelta_df.apply(lambda x: np.pad(x, (0,ts_size-len(x)), 'constant'))
timedelta_df.head()
```

```python
time_spent_df = activity_df.groupby('order_id')['time_spent'].apply(list)
time_spent_df = time_spent_df.apply(lambda x: np.pad(x, (0,ts_size-len(x)), 'constant'))
time_spent_df.head()
```

```python
eventAction_begin_checkout_df = activity_df.groupby('order_id')['eventAction_begin_checkout'].apply(list)
eventAction_begin_checkout_df = eventAction_begin_checkout_df.apply(lambda x: np.pad(x, (0,ts_size-len(x)), 'constant'))
eventAction_checkout_progress_df = activity_df.groupby('order_id')['eventAction_checkout_progress'].apply(list)
eventAction_checkout_progress_df = eventAction_checkout_progress_df.apply(lambda x: np.pad(x, (0,ts_size-len(x)), 'constant'))
eventAction_conversion_df = activity_df.groupby('order_id')['eventAction_conversion'].apply(list)
eventAction_conversion_df = eventAction_conversion_df.apply(lambda x: np.pad(x, (0,ts_size-len(x)), 'constant'))
eventAction_purchase_df = activity_df.groupby('order_id')['eventAction_purchase'].apply(list)
eventAction_purchase_df = eventAction_purchase_df.apply(lambda x: np.pad(x, (0,ts_size-len(x)), 'constant'))
eventAction_remove_from_cart_df = activity_df.groupby('order_id')['eventAction_remove_from_cart'].apply(list)
eventAction_remove_from_cart_df = eventAction_remove_from_cart_df.apply(lambda x: np.pad(x, (0,ts_size-len(x)), 'constant'))
eventAction_view_item_df = activity_df.groupby('order_id')['eventAction_view_item'].apply(list)
eventAction_view_item_df = eventAction_view_item_df.apply(lambda x: np.pad(x, (0,ts_size-len(x)), 'constant'))
eventAction_view_item_list_df = activity_df.groupby('order_id')['eventAction_view_item_list'].apply(list)
eventAction_view_item_list_df = eventAction_view_item_list_df.apply(lambda x: np.pad(x, (0,ts_size-len(x)), 'constant'))
```

```python
# itemcat_df = activity_df.groupby('order_id')['items.category'].apply(list)
# itemcat_df = itemcat_df.apply(lambda x: np.pad(x, (0,ts_size-len(x)), 'constant'))
# itemcat_df.head()
```

```python
itemprice_df = activity_df.groupby('order_id')['items.price'].apply(list)
itemprice_df = itemprice_df.apply(lambda x: np.pad(x, (0,ts_size-len(x)), 'constant'))
itemprice_df.head()
```

```python
activity_dfx = pd.concat([timedelta_df, time_spent_df, eventAction_begin_checkout_df, eventAction_checkout_progress_df, eventAction_conversion_df, eventAction_purchase_df, eventAction_remove_from_cart_df, eventAction_view_item_df, eventAction_view_item_list_df, itemprice_df], axis=1)
activity_dfx.head()
```

```python tags=[]
activity_dfx.info()
```

```python
agg_dfx= agg_dfx.merge(orders_dfx, on='order_id')
agg_dfx.head()
```

```python
dfx = agg_dfx.merge(activity_dfx, on='order_id')
dfx.columns
```

```python
dfx_cols = ['value', 'items.quantity', 'eventCount',
       'conversions', 'browser', 'os', 'device', 'class',
       'age', 'timedelta', 'time_spent', 'eventAction_begin_checkout',
       'eventAction_checkout_progress', 'eventAction_conversion',
       'eventAction_purchase', 'eventAction_remove_from_cart',
       'eventAction_view_item', 'eventAction_view_item_list', 'items.price']
dfx = dfx[dfx_cols]
dfx.head()
```

```python
dfx.to_pickle('dfx.pkl')
```

```python

```
