# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/utils/utils.filters.ipynb (unless otherwise specified).

__all__ = ['filter_by_time', 'filter_top_k']

# Cell
import pandas as pd
import numpy as np
import datetime
import time
import calendar
from collections import Counter

# Internal Cell
def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = int(sourcedate.year + month / 12)
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)

# Cell
def filter_by_time(df, last_months, ts_col='timestamp'):
    max_ts = df[ts_col].max().timestamp()
    lastdate = datetime.datetime.fromtimestamp(max_ts)
    firstdate = pd.Timestamp(add_months(lastdate, -last_months))
    # filter out older interactions
    df = df[df[ts_col] >= firstdate]
    return df

# Cell
def filter_top_k(df: pd.DataFrame,
                 topk: int = 0,
                 user_col: str = 'userid',
                 item_col: str = 'itemid',
                 sess_col: str = 'sessid',
                 ts_col: str = 'timestamp',
                 ):
    c = Counter(list(df[item_col]))

    if topk > 1:
        keeper = set([x[0] for x in c.most_common(topk)])
        df = df[df[item_col].isin(keeper)]

    # group by session id
    groups = df.groupby(sess_col)

    # convert item ids to string, then aggregate them to lists
    aggregated = groups[item_col].agg(sequence = lambda x: list(map(str, x)))
    init_ts = groups[ts_col].min()
    users = groups[user_col].min()  # it's just fast, min doesn't actually make sense

    result = aggregated.join(init_ts).join(users)
    result.reset_index(inplace=True)
    return result