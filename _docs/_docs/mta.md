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

# Multi-Touch Attribution

<!-- #region id="P_iqWoiE9mt7" -->
## Rule-based MTA
<!-- #endregion -->

```python id="_o4Wy87d9hXa"
import pandas as pd
from itertools import chain, tee, combinations
from functools import reduce, wraps
from operator import mul
from collections import defaultdict, Counter
import random
import time
import numpy as np
import copy
import json
import os

import arrow

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

```python id="_rQFADJE9hXe" outputId="8c8a692f-c003-4877-cd22-ef06f8661217"
#Preprocessing
 
df_raw = pd.read_csv('dataset1.csv')
df = df_raw.copy()

#converting touchpoints to journeys for each customer
df = df.groupby('customer_id')[['channel','conversion']].agg(lambda x : x.sum() if x.dtype=='int64' else ' > '.join(x))

df['path_no'] = df.groupby('customer_id')['conversion'].cumsum(skipna=True)#.unstack()

#selecting only first conversion journeys
df = df[(df['path_no']==0) | (df['path_no']==1)]
df.drop('path_no', axis=1, inplace=True)

df['non_conversion'] = 0
df.loc[df['conversion']==0,'non_conversion']=1

df = df.groupby('channel').sum().sort_values(by='conversion', ascending=False).reset_index()
df.head()
```

```python id="h33ejXI69hXj"
def normalize_dict(d):
    """
    returns a value-normalized version of dictionary d
    """
    sum_all_values = sum(d.values())

    for _ in d:
        d[_] = round(d[_]/sum_all_values, 6)

    return d
```

```python id="9Q-PTpYl9hXm"
NULL = '(null)'
START = '(start)'
CONV = '(conversion)'
channels = df_raw.channel.unique().tolist()
channels_ext = [START] + channels + [CONV, NULL]
c2i = {c: i for i, c in enumerate(channels_ext)}
i2c = {i: c for c, i in c2i.items()}

removal_effects = defaultdict(float)

data = df.copy()

sep = '>'
data['channel'] = data['channel'].apply(lambda _: [ch.strip() for ch in _.split(sep.strip())])
```

<!-- #region id="st-Js3dH9hXo" -->
## Rule-based Modeling
<!-- #endregion -->

```python id="_FL6COkg9hXp"
def pairs(lst):
    it1, it2 = tee(lst)
    next(it2, None)
    return zip(it1, it2)
```

```python id="6k3BCZpI9hXs" outputId="2be07e01-78b3-4c89-a01a-243fb6f34b1a"
def first_touch(normalize=True):
    first_touch = defaultdict(int)
    for c in channels:
        # total conversions for all paths where the first channel was c
        first_touch[c] = data.loc[data['channel'].apply(lambda _: _[0] == c), 'conversion'].sum()

    if normalize:
        first_touch = normalize_dict(first_touch)

    return first_touch

first_touch()
```

```python id="VRaQzQwJ9hXv" outputId="bf536ec2-890b-43a4-8ecd-913b37468ff2"
def last_touch(normalize=True):

    last_touch = defaultdict(int)

    for c in channels:

        # total conversions for all paths where the last channel was c
        last_touch[c] = data.loc[data['channel'].apply(lambda _: _[-1] == c), 'conversion'].sum()

    if normalize:
        last_touch = normalize_dict(last_touch)

    return last_touch

last_touch()
```

```python id="9qypnsw59hXz" outputId="eba4b08b-9ca2-4a64-ce71-86c3522c5486"
def linear(share='same', normalize=True):

    if share not in 'same proportional'.split():
        raise ValueError('share parameter must be either *same* or *proportional*!')

    linear = defaultdict(float)

    for row in data.itertuples():
        if row.conversion:
            if share == 'same':
                n = len(set(row.channel))    # number of unique channels visited during the journey
                s = row.conversion/n    # each channel is getting an equal share of conversions
                for c in set(row.channel):
                    linear[c] += s

            elif share == 'proportional':
                c_counts = Counter(row.channel)  # count how many times channels appear on this path
                tot_appearances = sum(c_counts.values())
                c_shares = defaultdict(float)
                for c in c_counts:
                    c_shares[c] = c_counts[c]/tot_appearances
                for c in set(row.channel):

                    linear[c] += row.conversion*c_shares[c]

    if normalize:
        linear = normalize_dict(linear)

    return linear

linear()
```

```python id="tPawSNUE9hX1" outputId="4e0d3a23-b990-4e4e-8f8b-a15232e3d2a2"
def time_decay(count_direction='left', normalize=True):

    """
    time decay - the closer to conversion was exposure to a channel, the more credit this channel gets

    this can work differently depending how you get timing sorted. 

    example: a > b > c > b > a > c > (conversion)

    we can count timing backwards: c the latest, then a, then b (lowest credit) and done. Or we could count left to right, i.e.
    a first (lowest credit), then b, then c. 

    """

    time_decay = defaultdict(float)

    if count_direction not in 'left right'.split():
        raise ValueError('argument count_direction must be *left* or *right*!')

    for row in data.itertuples():

        if row.conversion:

            channels_by_exp_time = []

            _ = row.channel if count_direction == 'left' else row.channel[::-1]

            for c in _:
                if c not in channels_by_exp_time:
                    channels_by_exp_time.append(c)

            if count_direction == 'right':
                channels_by_exp_time = channels_by_exp_time[::-1]

            # first channel gets 1, second 2, etc.

            score_unit = 1./sum(range(1, len(channels_by_exp_time) + 1))

            for i, c in enumerate(channels_by_exp_time, 1):
                time_decay[c] += i*score_unit*row.conversion

    if normalize:
        time_decay = normalize_dict(time_decay)

    return time_decay

time_decay()
```

<!-- #region id="QSMl7Wpi9rOI" -->
## Markov Chain MTA
<!-- #endregion -->

```python id="xFIRBeXz9_WR"
import pandas as pd
from itertools import chain, tee, combinations
from functools import reduce, wraps
from operator import mul
from collections import defaultdict, Counter
import random
import time
import numpy as np
import copy
import json
import os

import arrow

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

```python id="wCTyGrvX9_WW"
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

```python id="aDWjtb2h9_Wa" outputId="edb271c3-83fe-4c8c-95be-952d1eca3ef8"
df_raw = pd.read_csv('dataset1.csv')
df_raw.head()
```

```python id="LEA4yCAm9_Wf" outputId="405b4f71-28d0-4961-c2a3-16ad9ad777fd"
df_raw.info()
```

```python id="OqSpIv7Z9_Wi"
df = df_raw.copy()
```

<!-- #region id="kShKuZQ69_Wl" -->
## Preprocessing
<!-- #endregion -->

```python id="c3_f8QiD9_Wm" outputId="d598795f-b75f-4855-97b9-e559a91adf6a"
df['path_no'] = df.groupby('customer_id')['conversion'].cumsum(skipna=True)#.unstack()
df.head()
```

```python id="Zd-dX1cH9_Wp" outputId="4f468e29-abbe-45c9-f916-46f810e7825c"
#selecting only first conversion journeys
df = df[(df['path_no']==0) | (df['path_no']==1)]
df.drop('path_no', axis=1, inplace=True)
df.head()
```

```python id="16SP0Q_v9_Wu" outputId="2b1648f2-49c1-4c80-b1b8-f2cd6430f262"
#converting touchpoints to journeys for each customer
df = df.groupby('customer_id')[['channel','conversion']].agg(lambda x : x.sum() if x.dtype=='int64' else ' > '.join(x))
df.head()
```

```python id="Y84rJLdc9_Wx" outputId="b7f85d00-68ad-454b-a585-ade9b6f1af4c"
df['non_conversion'] = 0
df.loc[df['conversion']==0,'non_conversion']=1
df.head()
```

```python id="unW1PSda9_Wz" outputId="628ad996-3a8b-44e2-c7ea-48eccb8ba966"
df_uniquePath = df.groupby('channel').sum().sort_values(by='conversion', ascending=False).reset_index()
df_uniquePath.head()
```

<!-- #region id="4HT5eVQj9_W2" -->
## EDA
<!-- #endregion -->

```python id="XnTGTSt-9_W3" outputId="24c3e3ca-f6b3-4458-99a0-19185e350200"
df_raw.conversion = df_raw.conversion.astype('object')
df_raw.describe()
```

```python id="IjisLAFs9_W6" outputId="f2dff6c4-732e-4089-9ab6-1e8c961e779b"
sns.set(rc={'figure.figsize':(13,4)})
sns.countplot(df_raw['channel'])
```

```python id="8n56Yg0I9_W-" outputId="812f5459-3bb2-496e-f398-6f6f47bd5e76"
df.info()
```

```python id="yHPv7GZ79_XA" outputId="ca5a0500-d873-418d-dbb5-e28bf2a96e7e"
sns.set(rc={'figure.figsize':(7,4)})
sns.countplot('conversion', data=df)
df.conversion.value_counts(dropna=False)
```

```python id="fStOS5uq9_XD"
def normalize_dict(d):
    """
    returns a value-normalized version of dictionary d
    """
    sum_all_values = sum(d.values())

    for _ in d:
        d[_] = round(d[_]/sum_all_values, 6)

    return d
```

<!-- #region id="hTapiF-L9_XF" -->
## Markov Modeling
<!-- #endregion -->

```python id="oPPwAsYB9_XG"
NULL = '(null)'
START = '(start)'
CONV = '(conversion)'
channels = df_raw.channel.unique().tolist()
channels_ext = [START] + channels + [CONV, NULL]
c2i = {c: i for i, c in enumerate(channels_ext)}
i2c = {i: c for c, i in c2i.items()}

removal_effects = defaultdict(float)

data = df.copy()

sep = '>'
data['channel'] = data['channel'].apply(lambda _: [ch.strip() for ch in _.split(sep.strip())])
```

```python id="grsAV3It9_XJ"
def pairs(lst):
    it1, it2 = tee(lst)
    next(it2, None)
    return zip(it1, it2)
```

```python id="TgqjYM_K9_XN"
def count_pairs():

    """
    count how many times channel pairs appear on all recorded customer journey paths
    """

    c = defaultdict(int)

    for row in data.itertuples():

        for ch_pair in pairs([START] + row.channel):
            c[ch_pair] += (row.conversion + row.non_conversion)

        c[(row.channel[-1], NULL)] += row.non_conversion
        c[(row.channel[-1], CONV)] += row.conversion

    return c
```

```python id="xnVZNAb59_XR"
def ordered_tuple(self, t):

    """
    return tuple t ordered 
    """

    if not isinstance(t, tuple):
        raise TypeError(f'provided value {t} is not tuple!')

    if all([len(t) == 1, t[0] in '(start) (null) (conversion)'.split()]):
        raise Exception(f'wrong transition {t}!')

    if (len(t) > 1) and (t[-1] == self.START): 
        raise Exception(f'wrong transition {t}!')

    if (len(t) > 1) and (t[0] == self.START):
        return (t[0],) + tuple(sorted(list(t[1:])))

    if (len(t) > 1) and (t[-1] in '(null) (conversion)'.split()):
        return tuple(sorted(list(t[:-1]))) + (t[-1],)

    return tuple(sorted(list(t)))
```

```python id="bMJIN0W49_XT"
def trans_matrix():

        """
        calculate transition matrix which will actually be a dictionary mapping 
        a pair (a, b) to the probability of moving from a to b, e.g. T[(a, b)] = 0.5
        """

        tr = defaultdict(float)

        outs = defaultdict(int)

        # here pairs are unordered
        pair_counts = count_pairs()

        for pair in pair_counts:

            outs[pair[0]] += pair_counts[pair]

        for pair in pair_counts:

            tr[pair] = pair_counts[pair]/outs[pair[0]]

        return tr
```

```python id="eZ_t_K1f9_XW"
def simulate_path(trans_mat, drop_channel=None, n=int(1e6)):

    """
    generate n random user journeys and see where these users end up - converted or not;
    drop_channel is a channel to exclude from journeys if specified
    """

    outcome_counts = defaultdict(int)

    idx0 = c2i[START]
    null_idx = c2i[NULL]
    conv_idx = c2i[CONV]

    drop_idx = c2i[drop_channel] if drop_channel else null_idx

    for _ in range(n):

        stop_flag = None

        while not stop_flag:

            probs = [trans_mat.get((i2c[idx0], i2c[i]), 0) for i in range(len(channels_ext))]

            # index of the channel where user goes next
            idx1 = np.random.choice([c2i[c] for c in channels_ext], p=probs, replace=False)

            if idx1 == conv_idx:
                outcome_counts[CONV] += 1
                stop_flag = True
            elif idx1 in {null_idx, drop_idx}:
                outcome_counts[NULL] += 1
                stop_flag = True
            else:
                idx0 = idx1

    return outcome_counts
```

```python id="oEUKm0069_XY"
def prob_convert(trans_mat, drop='None'):
    _d = data[data['channel'].apply(lambda x: drop not in x) & (data['conversion'] > 0)]

    p = 0

    for row in _d.itertuples():

        pr_this_path = []

        for t in pairs([START] + row.channel + [CONV]):
            pr_this_path.append(trans_mat.get(t, 0))

        p += reduce(mul, pr_this_path)

    return p
```

```python id="K5m4PJey9_Xa"
def markov(sim=False, normalize=False):

    markov = defaultdict(float)

    # calculate the transition matrix
    tr = trans_matrix()

    if not sim:
        p_conv = prob_convert(trans_mat=tr)
        for c in channels:
            markov[c] = (p_conv - prob_convert(trans_mat=tr, drop=c))/p_conv
    else:
        outcomes = defaultdict(lambda: defaultdict(float))
        # get conversion counts when all chennels are in place
        outcomes['full'] = simulate_path(trans_mat=tr, drop_channel=None)
        for c in channels:
            outcomes[c] = simulate_path(trans_mat=tr, drop_channel=c)
            # removal effect for channel c
            markov[c] = (outcomes['full'][CONV] - outcomes[c][CONV])/outcomes['full'][CONV]

    if normalize:
        markov = normalize_dict(markov)
    
    return markov
```

```python id="IGyifWbv9_Xd" outputId="27fdfef5-76af-4c4f-cf6f-80ca4e0287b0"
markov(normalize=True)
```

<!-- #region id="mnPL8iIV9_Xi" -->
## Survival Model MTA
<!-- #endregion -->

```python id="GsUkLAnA-WHV"
import pandas as pd
from itertools import chain, tee, combinations
from functools import reduce, wraps
from operator import mul
from collections import defaultdict, Counter
import random
import time
import numpy as np
import copy
import json
import os

import arrow

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

```python id="zrsMbVqU-WHa" outputId="558c37b7-3fc4-4878-f3c1-a8a42e15e4c9"
#Preprocessing
 
df_raw = pd.read_csv('dataset1.csv')
df = df_raw.copy()

#converting touchpoints to journeys for each customer
df = df.groupby('customer_id')[['channel','conversion']].agg(lambda x : x.sum() if x.dtype=='int64' else ' > '.join(x))

df['path_no'] = df.groupby('customer_id')['conversion'].cumsum(skipna=True)#.unstack()

#selecting only first conversion journeys
df = df[(df['path_no']==0) | (df['path_no']==1)]
df.drop('path_no', axis=1, inplace=True)

df['non_conversion'] = 0
df.loc[df['conversion']==0,'non_conversion']=1

df.head()
```

```python id="_xOufYoR-WHe"
channels = df_raw.channel.unique().tolist()
data = df.copy()
sep = '>'
data['channel'] = data['channel'].apply(lambda _: [ch.strip() for ch in _.split(sep.strip())])
```

```python id="-marYc8A-WHh" outputId="a5819ff1-2c26-4bfc-8ddc-6b705d05ddea"
# Add exposure time in dataset for Additive Hazard time modeling

def add_exposure_times(dt=None):

    """
    generate synthetic exposure times; if dt is specified, the exposures will be dt=1 sec away from one another, otherwise
    we'll generate time spans randomly

    - the times are of the form 2018-11-26T03:54:26.532091+00:00
    """

    ts = []    # this will be a list of time instant lists one per path 

    if not dt:

        _t0 = arrow.utcnow()

        data['channel']\
        .apply(lambda lst: ts.append(sep.join([r.format('YYYY-MM-DD') 
                                for r in arrow.Arrow.range('day', _t0, _t0.shift(days=+(len(lst) - 1)))])))

    data['exposure_times'] = ts
    data['exposure_times'] = data['exposure_times'].apply(lambda _: [ch.strip() for ch in _.split(sep.strip())])
    return data


data = add_exposure_times()
data.head()
```

```python id="iz_tyN3B-WHk"
def normalize_dict(d):
    """
    returns a value-normalized version of dictionary d
    """
    sum_all_values = sum(d.values())

    for _ in d:
        d[_] = round(d[_]/sum_all_values, 6)

    return d
```

<!-- #region id="5fooa7kb-WHn" -->
## Additive Hazard Modeling
<!-- #endregion -->

```python id="yqFZEKgr-WHo"
def pi(path, exposure_times, conv_flag, beta_by_channel, omega_by_channel):

    """

    calculate contribution of channel i to conversion of journey (user) u - (p_i^u) in the paper

     - path is a list of states that includes (start) but EXCLUDES (null) or (conversion)
     - exposure_times is list of exposure times

    """

    p = {c: 0 for c in path}    # contributions by channel

    # all contributions are zero if no conversion
    if not conv_flag:
        return p

    dts = [(arrow.get(exposure_times[-1]) - arrow.get(t)).days for t in exposure_times]

    _ = defaultdict(float)

    for c, dt in zip(path, dts):
        _[c] += beta_by_channel[c]*omega_by_channel[c]*np.exp(-omega_by_channel[c]*dt)

    for c in _:
        p[c] = _[c]/sum(_.values())

    return p
```

```python id="7hz0AMq2-WHr"
def update_coefs(beta, omega):

    """
    return updated beta and omega
    """

    delta = 1e-3

    beta_num = defaultdict(float)
    beta_den = defaultdict(float)
    omega_den = defaultdict(float)

    for u, row in enumerate(data.itertuples()):

        p = pi(row.channel, row.exposure_times, row.conversion, beta, omega)

        r = copy.deepcopy(row.channel)

        dts = [(arrow.get(row.exposure_times[-1]) - arrow.get(t)).seconds for t in row.exposure_times]

        while r:

            # pick channels starting from the last one
            c = r.pop()
            dt = dts.pop()

            beta_den[c] += (1.0 - np.exp(-omega[c]*dt))
            omega_den[c] += (p[c]*dt + beta[c]*dt*np.exp(-omega[c]*dt))

            beta_num[c] += p[c]

    # now that we gone through every user, update coefficients for every channel

    beta0 = copy.deepcopy(beta)
    omega0 = copy.deepcopy(omega)

    df = []

    for c in channels:

        beta_num[c] = (beta_num[c] > 1e-6)*beta_num[c]
        beta_den[c] = (beta_den[c] > 1e-6)*beta_den[c]
        omega_den[c] = max(omega_den[c], 1e-6)

        if beta_den[c]:
            beta[c] = beta_num[c]/beta_den[c]

        omega[c] = beta_num[c]/omega_den[c]

        df.append(abs(beta[c] - beta0[c]) < delta)
        df.append(abs(omega[c] - omega0[c]) < delta)

    return (beta, omega, sum(df))
```

```python id="6CJaAnU7-WHu"
def additive_hazard(epochs=20, normalize=True):

    """
    additive hazard model as in Multi-Touch Attribution in On-line Advertising with Survival Theory
    """

    beta = {c: random.uniform(0.001,1) for c in channels}
    omega = {c: random.uniform(0.001,1) for c in channels}

    for _ in range(epochs):

        beta, omega, h = update_coefs(beta, omega)

        if h == 2*len(channels):
            print(f'converged after {_ + 1} iterations')
            break

    # time window: take the max time instant across all journeys that converged

    additive_hazard = defaultdict(float)

    for u, row in enumerate(data.itertuples()):

        p = pi(row.channel, row.exposure_times, row.conversion, beta, omega)

        for c in p:
            additive_hazard[c] += p[c]

    if normalize:
        additive_hazard = normalize_dict(additive_hazard)
    
    return additive_hazard
```

```python id="ok9faAbi-WHx" outputId="2de374a6-0ec0-48d5-f5a5-50b993bac075"
result = additive_hazard()
```

```python id="E3U5-Q2P-WH0" outputId="31fb0544-2cad-457a-b15d-0ee253f4ff58"
result
```

```python id="STmlcF7T-WH3"
pd.DataFrame(list(result.items()), columns=['channel','attribution']).to_csv('additiveHazardMTA.csv')
```

<!-- #region id="xfCFRErf-bSM" -->
## Attribution Model Comparison Part 1/2
<!-- #endregion -->

```python id="17SP5ckx-kWi"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```

```python id="_LK-wrPq-kWo" outputId="617f9d51-5e39-40b2-bebe-bc66de829406"
path = 'attribution/'
data1 = pd.read_csv(path+'RuleAndMarkovMTA.csv', index_col=0)
data1.columns = ['channel','first_touch','last_touch','linear_touch','markov_model']
data1
```

```python id="Kpb1kPVd-kWu" outputId="f5ceddfd-ae94-4c37-f6c0-24ad1dd1cf54"
data2 = pd.read_csv(path+'additiveHazardMTA.csv', index_col=0)
data2.columns = ['channel','additive_hazard']
data2
```

```python id="GODeXMTa-kWx" outputId="ac5c257a-3662-4588-8f05-30502855930b"
df = pd.merge(data1, data2, on=['channel'], how='inner').set_index('channel')
df
```

```python id="mXyi7cD3-kW1" outputId="c833d8c4-6b09-4e44-fc63-f7eea8d26494"
for col in df.columns:
    df[col] = df[col]/df[col].sum()
df.reset_index(inplace=True)
df
```

```python id="a4uoRkLp-kW4" outputId="0f642d04-d264-4bfc-bf58-54afb89042b7"
f, axes = plt.subplots(3, 2, figsize=(24,12))

sns.barplot(x='channel', y='first_touch', data=df, ax=axes[0,0], palette="Blues")
sns.barplot(x='channel', y='last_touch', data=df, ax=axes[0,1], palette="Greens")
sns.barplot(x='channel', y='linear_touch', data=df, ax=axes[1,0], palette="BuGn_r")
sns.barplot(x='channel', y='markov_model', data=df, ax=axes[1,1], palette="Purples")
sns.barplot(x='channel', y='additive_hazard', data=df, ax=axes[2,0], palette="Purples")
f.delaxes(axes[2,1])
plt.show()
```

```python id="_wIx8SgL-kW8" outputId="9abed288-d087-4125-c559-fe857a5453a9"
df.plot(colormap='Paired', x="channel", y=["first_touch", "last_touch", "linear_touch", "markov_model", "additive_hazard"], kind="bar", figsize=(12,5))
```

<!-- #region id="GUJ-g_nQ-pdO" -->
## Criteo dataset analysis EDA
<!-- #endregion -->

```python id="NfRV99TN-xK_" outputId="e9472716-5864-494d-d5fe-79a44f40606a"
%pylab inline
import pandas as pd
plt.style.use('ggplot')
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import gc
```

```python id="BhDyB8YC-xLG" outputId="f8596eea-6184-4477-eed0-be1807b7883d"
DATA_FILE='../input/criteo_attribution_dataset.tsv/criteo_attribution_dataset.tsv'
df = pd.read_csv(DATA_FILE, sep='\t')
df.head()
```

```python id="tF6qHN2H-xLK" outputId="d167e43b-75bc-4c12-8ca0-d710ccbe7631"
df = df[['timestamp', 'uid', 'campaign', 'conversion', 'conversion_id', 'click', 'cat1', 'cat2', 'cat3', 'cat4',
       'cat5', 'cat6', 'cat7', 'cat8', 'cat9']]
df.head()
gc.collect()
```

```python id="GEL8wrFq-xLN" outputId="6e362da6-d7a2-4579-ba14-198b2b7f87ec"
df.info()
```

```python id="Gi0XiCKj-xLQ" outputId="4d4abc13-55e3-478e-c724-63e7f2332803"
ax = df.day.hist(bins=len(df.day.unique()))
ax.set_xlabel("time (in days)")
ax.set_ylabel("frequency")
plt.show()
```

```python id="V9lNakVu-xLU" outputId="16f529a0-a508-491c-92cb-fc61210a431e"
ax = df.day.hist(bins=len(df.day.unique()), alpha=0.5, label='Impressions')
ax = df.loc[df['click']==1].day.hist(bins=len(df.day.unique()), alpha=0.7, label='Clicks')
ax = df.loc[df['conversion']==1].day.hist(bins=len(df.day.unique()), alpha=1, label='Conversions')
ax.legend(loc='upper right')
ax.set_xlabel("time (in days)")
ax.set_ylabel("frequency")
plt.show()
```

```python _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" id="TUiMpjTq-xLX" outputId="5080bf49-62c5-4e2f-a8fd-eebd824f4c95"
df['day'] = np.floor(df.timestamp / 86400.).astype(int)
```

```python id="Rn-02g3x-xLb"
# df['gap_click_sale'] = -1
# df.loc[df.conversion == 1, 'gap_click_sale'] = df.conversion_timestamp - df.timestamp
```

```python id="4F7XreeA-xLf" outputId="088393d9-eb94-4b2b-b477-dd84c0606ae9"
df.set_index(['uid','conversion_id'], inplace=True)
df.reset_index(drop=True, inplace=True)
sns.countplot(df['conversion'])
gc.collect()
```

```python id="qyElyvZw-xLj" outputId="b62754d9-cb7a-4bf4-811d-f14de14c44fb"
sns.countplot('click', hue='conversion', data=df)
```

```python id="0CWiCvFF-xLm" outputId="9046028b-6446-4099-b06e-c37e88c331d1"
df.head()
```

```python id="dIhW0BKG-xLq" outputId="2544c09f-7a08-4350-91a8-c5394ba31e0e"
df.columns
```

```python id="pd68DTdk-xLu" outputId="b866cad4-c5cc-48cc-e2da-a95593d22651"
gc.collect()

# df['timestamp']= df['timestamp'].astype('category')
df['campaign']= df['campaign'].astype('category')
df['conversion']= df['conversion'].astype('bool')
df['click']= df['click'].astype('bool')

gc.collect()
```

```python id="HepMq8Nd-xLx" outputId="fc72efbb-0a8f-43bb-dfeb-ac1176f78e07"
df['cat1']= df['cat1'].astype('category')
df['cat2']= df['cat2'].astype('category')
df['cat3']= df['cat3'].astype('category')
df['cat4']= df['cat4'].astype('category')
df['cat5']= df['cat5'].astype('category')
df['cat6']= df['cat6'].astype('category')
df['cat7']= df['cat7'].astype('category')
df['cat8']= df['cat8'].astype('category')
df['cat9']= df['cat9'].astype('category')

gc.collect()

```

```python id="OSUmpuA5-xL5"
# df.pop('day')
# df['timestamp']= df['timestamp'].astype('int')
```

```python id="Vqx9aoFO-xL8" outputId="5a1089c5-31bc-486a-f639-574dd5e3af6f"
df.describe(exclude=['int'])
```

```python id="IWLujd4E-xL_" outputId="a5ca3070-58c4-4ced-ffc1-477ae12b4352"
sns.kdeplot(df['timestamp'])
```

<!-- #region id="CYJtrI4t-86b" -->
## MTA models - LogisticRegression, AdditiveHazard, Attention RNN and Dual-attention RNN
<!-- #endregion -->

```python id="Lg_dJ8r_Iyuv"
import _pickle as pkl
import os
import time
from datetime import timedelta
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import *
import sys
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from model_config import config
from wrapped_loadCriteo import loadrnnattention
from wrapped_loadCriteo import loadLRF
from wrapped_loadCriteo import loaddualattention
```

<!-- #region id="lI0kTNe1KsTo" -->
## Logistic
<!-- #endregion -->

```python id="C7ZDpxwpKsof"
class LR_f_criteo():
    def __init__(self, path, learning_rate=0.1, epochs=10000):
        self.graph = tf.Graph()

        self._path = path
        self._save_path, self._logs_path = None, None
        self.cross_entropy, self.train_step, self.prediction = None, None, None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.features = 5867
        self.classes = 2

        with self.graph.as_default():
            self._define_sparse_inputs()
            self._build_sparse_graph()
            self.saver = tf.train.Saver()
            self.global_initializer = tf.global_variables_initializer()
            self.local_initializer = tf.local_variables_initializer()
        self._initialize_session()

    def _define_inputs(self):
        self.X = tf.placeholder(tf.float32, [None, self.features])
        self.Y = tf.placeholder(tf.float32, [None, self.classes])

    def _define_sparse_inputs(self):
        self.X = tf.sparse_placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32, [None, self.classes])

    def _initialize_session(self, set_logs=False):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(self.global_initializer)
        self.sess.run(self.local_initializer)

    def _build_graph(self):
        W = tf.Variable(tf.random_normal([self.features, self.classes]))
        B = tf.Variable(tf.random_normal([self.classes]))

        pY = tf.sigmoid(tf.matmul(self.X, W) + B)
        pY = tf.nn.softmax(pY)

        cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pY, labels=self.Y))
        self.cross_entropy = cost_func
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(cost_func)
        self.train_step = opt
        self.prediction = pY

    def _build_sparse_graph(self):
        W = tf.Variable(tf.random_normal([self.features, self.classes]))
        B = tf.Variable(tf.random_normal([self.classes]))
        X = tf.sparse_tensor_to_dense(self.X)

        pY = tf.sigmoid(tf.matmul(X, W) + B)
        cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pY, labels=self.Y))
        pY = tf.nn.softmax(pY)
        self.cross_entropy = cost_func
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(cost_func)
        self.train_step = opt
        self.prediction = pY
        self.W = W

    def train_one_epoch(self):
        total_time = 0
        total_loss = []
        pred = []
        label = []
        trainfile = open('train_usr.yzx.txt', 'rb')
        while True:
            train_X, train_Y = loadLRF(500, 20, 12, trainfile)
            feed_dict = {
                self.X: train_X,
                self.Y: train_Y
            }
            fetches = [self.train_step, self.cross_entropy, self.prediction]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, prediction = result
            total_loss.append(loss)
            pred += prediction.tolist()
            label += train_Y
            if len(train_Y) < 500:
                break
        pred = np.array(pred)
        auc = roc_auc_score(np.argmax(label, 1), pred[:, 1])
        print("Training AUC = " + "{:.4f}".format(auc))
        mean_loss = np.mean(total_loss)
        print("Training Loss = " + "{:.4f}".format(mean_loss))
        _pY = np.argmax(pred, 1)
        Y = np.argmax(label, 1)
        precision = precision_score(Y, _pY)
        recall = recall_score(Y, _pY)
        F1 = f1_score(Y, _pY)
        accuracy = accuracy_score(Y, _pY)
        return mean_loss, auc, accuracy, precision, recall, F1

    def train_all_epochs(self):
        total_start_time = time.time()
        losses = []
        for epoch in range(self.epochs):
            print("Training...")
            result = self.train_one_epoch()
            loss = result[0]
            losses.append(loss)
            self.test(epoch)
            
            if epoch > 3:
                if losses[-1] >= losses[-2] and losses[-2] >= losses[-3]:
                    break

    def test(self, epoch):
        total_loss = []
        pred = []
        label = []
        file = open('test_usr.yzx.txt', 'rb')
        while True:
            train_X, train_Y = loadLRF(500, 20, 12, file)
            feed_dict = {
                self.X: train_X,
                self.Y: train_Y
            }
            fetches = [self.train_step, self.cross_entropy, self.prediction]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, prediction = result
            total_loss.append(loss)
            pred += prediction.tolist()
            label += train_Y
            if len(train_Y) < 500:
                break
        pred = np.array(pred)
        auc = roc_auc_score(np.argmax(label, 1), pred[:, 1])
        loglikelihood = -log_loss(np.argmax(label, 1), pred[:, 1])
        print("Test AUC = " + "{:.4f}".format(auc))
        mean_loss = np.mean(total_loss)
        print("Test Loss = " + "{:.4f}".format(mean_loss))
        _pY = np.argmax(pred, 1)
        Y = np.argmax(label, 1)
        precision = precision_score(Y, _pY)
        recall = recall_score(Y, _pY)
        F1 = f1_score(Y, _pY)
        accuracy = accuracy_score(Y, _pY)
        result = mean_loss, auc, loglikelihood, accuracy, precision, recall, F1

    def attr(self):
        file = open('test_usr.yzx.txt', 'rb')
        channelfile = open('i2c_converted.pkl', 'rb')
        channel_set = pkl.load(channelfile)
        outfile = open('lr_f.txt', 'w')
        train_X, train_Y = loadLRF(1, 20, 12, file)
        feed_dict = {
            self.X: train_X,
            self.Y: train_Y
        }
        W = self.sess.run(self.W, feed_dict)
        for key in channel_set:
            outfile.write(channel_set[key] + '\t' + str(W[int(key)][1]) + '\n')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 442} id="bH8IIOK6KweL" outputId="16545290-d8e1-4e43-963b-2e3d8391c809"
if __name__ == '__main__':
    learning_rate = 0.01
    epochs = 5
    model = LR_f_criteo("../Model/LR", learning_rate=learning_rate, epochs=epochs)
    model.train_all_epochs()
#     model.test(0)
    model.attr()
```

<!-- #region id="zNftDpLBLPO-" -->
## Survival Additive Hazard
<!-- #endregion -->

```python id="ooZSr-BJLRqc"
import loadCriteo

def optmize(data, num_of_channel, beta, omega):
    # Preparation

    for item in data:
        item['p'] = np.zeros(len(item['action']))

    # update_P
    for item in data:
        X_u = item['isconversion']
        if (X_u == 0):
            continue
        T_u = item['conversionTime']
        l = len(item['action'])
        sum_p = 0.
        for i in range(l):
            a_i = item['action'][i][0]
            t_i = item['action'][i][1]
            item['p'][i] = beta[a_i] * omega[a_i] * np.exp(-omega[a_i] * (T_u - t_i))
            sum_p += item['p'][i]
        for i in range(l):
            item['p'][i] = item['p'][i] / sum_p

    # update_Beta
    fz = np.zeros(num_of_channel)
    fmb = np.zeros(num_of_channel)
    for item in data:
        X_u = item['isconversion']
        T_u = item['conversionTime']
        l = len(item['action'])
        for i in range(l):
            a_i = item['action'][i][0]
            t_i = item['action'][i][1]
            fmb[a_i] += 1 - np.exp(-omega[a_i] * (T_u - t_i))
            if (X_u == 1):
                fz[a_i] += item['p'][i]

    # update_ogema
    fz = np.zeros(num_of_channel)
    fm = np.zeros(num_of_channel)
    for item in data:
        X_u = item['isconversion']
        T_u = item['conversionTime']
        l = len(item['action'])
        for i in range(l):
            a_i = item['action'][i][0]
            t_i = item['action'][i][1]
            fm[a_i] += item['p'][i] * (T_u - t_i) + beta[a_i] * (T_u - t_i) * np.exp(-omega[a_i] * (T_u - t_i))
            if (X_u == 1):
                fz[a_i] += item['p'][i]

    for k in range(num_of_channel):
        if (fm[k] > 0.):
            if fm[k] < 1e-10:
                fm[k] = 1e-5
            omega[k] = fz[k] / fm[k]

    for k in range(num_of_channel):
        if (fmb[k] > 0.):
            beta[k] = fz[k] / fmb[k]
    return beta, omega


def attr(beta, omega, num_of_channel, data):
    for item in data:
        item['p'] = np.zeros(len(item['action']), dtype=np.float64)

    for item in data:
        X_u = item['isconversion']
        if (X_u == 0):
            continue
        T_u = item['conversionTime']
        l = len(item['action'])
        sum_p = 0.
        for i in range(l):
            a_i = item['action'][i][0]
            t_i = item['action'][i][1]
            item['p'][i] = beta[a_i] * omega[a_i] * np.exp(-omega[a_i] * (T_u - t_i))
            sum_p += item['p'][i]
        for i in range(l):
            item['p'][i] = item['p'][i] / sum_p
    global ChannelList
    ChannelSet = pkl.load(open('i2c_converted.pkl', 'rb'))
    channel_value = {}
    channel_time = {}
    for item in data:
        X_u = item['isconversion']
        if (X_u == 0):
            continue
        l = len(item['action'])
        for i in range(l):
            a_i = item['action'][i][0]
            channel = ChannelList[a_i]
            if channel in channel_value:
                channel_value[channel] += item['p'][i]
                channel_time[channel] += 1
            else:
                channel_value[channel] = item['p'][i]
                channel_time[channel] = 1
    outfile = open('survival.txt', 'w')
    for channel in channel_value:
        outfile.write(ChannelSet[str(channel)] + '\t' + str(channel_value[channel] / channel_time[channel]) + '\n')


def vertical_attr(beta, omega, num_of_channel, data, lenth):
    for item in data:
        item['p'] = np.zeros(len(item['action']), dtype=np.float64)

    for item in data:
        X_u = item['isconversion']
        if (X_u == 0):
            continue
        T_u = item['conversionTime']
        l = len(item['action'])
        sum_p = 0.
        for i in range(l):
            a_i = item['action'][i][0]
            t_i = item['action'][i][1]
            item['p'][i] = beta[a_i] * omega[a_i] * np.exp(-omega[a_i] * (T_u - t_i))
            sum_p += item['p'][i]
        for i in range(l):
            item['p'][i] = item['p'][i] / sum_p
    v = [0.] * lenth
    for item in data:
        X_u = item['isconversion']
        if (X_u == 0):
            continue
        l = len(item['action'])
        if l != lenth:
            continue
        for i in range(l):
            v[i] += item['p'][i]
    print(v)


def test(beta, omega, num_of_channel, test_data):
    y = []
    pred = []
    for item in test_data:
        y.append(item['isconversion'])
        ans = 1.
        T_u = item['conversionTime']
        Diff_Set = {}
        for tmp in item['action']:
            a_i = tmp[0]
            if a_i not in Diff_Set:
                Diff_Set[a_i] = 1
            t_i = tmp[1]
            # print T_u - t_i
            pred_now = np.exp(-beta[a_i] * (1 - np.exp(-omega[a_i] * (T_u - t_i + 0.1))))
            ans *= pred_now
        pred.append((1. - ans) * (0.95 ** len(Diff_Set)))

    auc = roc_auc_score(y, pred)
    loglikelihood = -log_loss(y, pred)
    print("Testing Auc= " + "{:.6f}".format(auc))
    print("Testing loglikelihood " + "{:.6f}".format(loglikelihood))
    return auc


num_of_epoches = 20

train_path = 'train_usr.yzx.txt'
test_path = 'test_usr.yzx.txt'
traindata_size = loadCriteo.count(train_path)
testdata_size = loadCriteo.count(test_path)

ChannelSet = {}
ChannelList = []


def loadCriteoData(datasize, fin):
    total_data = []
    for i in range(datasize):
        tmpseq = {}
        try:
            (tmp_seq_len, tmp_label) = [int(_) for _ in fin.readline().split()]
            tmpseq['isconversion'] = tmp_label
            tmp_action = []
            for _ in range(tmp_seq_len):
                tmpline = fin.readline().split()
                tmp_campaign, tmp_time = int(tmpline[2]), float(tmpline[0]) * 31.
                global ChannelSet, ChannelList
                if tmp_campaign not in ChannelSet:
                    ChannelSet[tmp_campaign] = len(ChannelList)
                    ChannelList.append(tmp_campaign)
                tmp_action.append((ChannelSet[tmp_campaign], tmp_time))
                if _ == tmp_seq_len - 1:
                    tmpseq['conversionTime'] = tmp_time
            tmpseq['action'] = tmp_action
        except:
            continue
        total_data.append(tmpseq)
    print(len(total_data))
    return total_data
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1612} id="r3-oRAbvLas6" outputId="4b5e009f-8052-4acd-c61c-66849f502c96"
def main():
    start_time = time.time()
    with open(train_path) as f_train:
        train_data = loadCriteoData(traindata_size, f_train)
    with open(test_path) as f_test:
        test_data = loadCriteoData(testdata_size, f_test)
    finish_time = time.time()
    print("load dataset finished, {} seconds took".format(finish_time - start_time))
    num_of_channel = len(ChannelList)
    beta = np.random.uniform(0, 1, num_of_channel)
    omega = np.random.uniform(0, 1, num_of_channel)
    auc = []
    beta_list = []
    omega_list = []
    for epoch in range(num_of_epoches):
        beta, omega = optmize(train_data, num_of_channel, beta, omega)
        print('Test on Epoch %d' % epoch)
        auc.append(test(beta, omega, num_of_channel, test_data))
        beta_list.append(beta)
        omega_list.append(omega)
        if epoch > 10 and auc[-1] < auc[-2] < auc[-3]:
            break
    sns.kdeplot(beta_list[-1])
    plt.show()
    sns.kdeplot(omega_list[-1])
    plt.show()
    attr(beta_list[-1],omega_list[-1],671,train_data)
      
if __name__ == "__main__":
    main()
```

<!-- #region id="qumqfqRxJ9UE" -->
## ARNN
<!-- #endregion -->

```python id="YQwelgZUJPva"
class RnnWithattention(object):
    def __init__(self, path, train_dataset, test_dataset, config):
        self.graph = tf.Graph()

        self._path = path
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self._save_path, self._logs_path = None, None
        self.config = config
        self.batches_step = 0
        self.cross_entropy, self.train_step, self.prediction = None, None, None
        self.data_shape = config.data_shape
        self.label_shape = config.label_shape
        self.n_classes = config.n_classes
        self.attention = None
        with self.graph.as_default():
            self._define_inputs()
            self._build_graph()
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self._initialize_session()

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        label_shape = [None]
       
        self.input = tf.placeholder(tf.float32, shape=[None, self.config.seq_max_len, 12], name='input')
        self.labels = tf.placeholder(tf.float32, shape=label_shape, name='labels')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.seqlen = tf.placeholder(tf.int32, shape=[None], name='seqlen')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    def _initialize_session(self, set_logs=False):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(self.initializer)

    def _build_graph(self):
        x = self.input
        batchsize = tf.shape(x)[0]
        embedding_matrix = tf.Variable(
            tf.random_normal([self.config.max_features, self.config.embedding_output], stddev=0.1))
        x1, x2 = tf.split(x, [2, 10], 2)
        x2 = tf.to_int32(x2)
        x2 = tf.nn.embedding_lookup(embedding_matrix, x2)
        x2 = tf.reshape(x2, [-1, self.config.seq_max_len, 10 * self.config.embedding_output])
        x = tf.concat((x1, x2), axis=2)
        weights = {
            'out': tf.Variable(tf.random_normal([self.config.n_hidden])),
            'attention_h': tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden])),
            'attention_x': tf.Variable(tf.random_normal([self.config.n_input, self.config.n_hidden])),
            'v_a': tf.Variable(tf.random_normal([self.config.n_hidden]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal(shape=[self.config.batch_size], dtype=tf.float32))
        }

        index = tf.range(0, self.config.batch_size) * self.config.seq_max_len + (self.seqlen - 1)
        x_last = tf.gather(tf.reshape(x, [-1, self.config.n_input]), index)
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.config.n_input])
        x = tf.split(axis=0, num_or_size_splits=self.config.seq_max_len, value=x)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob,
                                                  output_keep_prob=self.keep_prob)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=self.seqlen)

        # attention
        e = []
        Ux = tf.matmul(x_last, weights['attention_x'])
        for output in outputs:
            e_ = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(output, weights['attention_h']) + Ux), weights['v_a']),
                               reduction_indices=1)
            e.append(e_)
        e = tf.stack(e)
        a = tf.nn.softmax(e, dim=0)
        a = tf.split(a, self.config.seq_max_len, 0)
        c = tf.zeros([self.config.batch_size, self.config.n_hidden])
        for i in range(self.config.seq_max_len):
            c = c + tf.multiply(outputs[i], tf.transpose(a[i]))
        cvr = tf.reduce_sum(tf.multiply(c, weights['out']), axis=1) + biases['out']
        cvr = tf.nn.dropout(cvr, keep_prob=self.keep_prob)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=cvr))
        cvr = tf.nn.sigmoid(cvr)
        for v in tf.trainable_variables():
            loss += self.config.miu * tf.nn.l2_loss(v)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        gvs, v = zip(*optimizer.compute_gradients(loss))
        gvs, _ = tf.clip_by_global_norm(gvs, 5.0)
        gvs = zip(gvs, v)

        self.cross_entropy = loss
        self.train_step = optimizer.apply_gradients(gvs)
        self.prediction = cvr
        self.attention = a

    def train_one_epoch(self, batch_size, learning_rate):
        total_loss = []
        cvr_pred = []
        cvr_label = []
        infile = open(self.train_dataset, 'rb')
        while True:
            batch = loadrnnattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            train_data, train_compaign_data, train_label, train_seqlen = batch
            if len(train_label) != batch_size:
                break
            feed_dict = {
                self.input: train_data,
                self.labels: train_label,
                self.seqlen: train_seqlen,
                self.learning_rate: learning_rate,
                self.is_training: True,
                self.keep_prob: 0.5
            }

            result = self.sess.run([self.train_step, self.cross_entropy, self.prediction], feed_dict=feed_dict)
            _, loss, cvr = result
            total_loss.append(loss)
            cvr_pred += cvr.tolist()
            cvr_label += train_label
        auc_cov = roc_auc_score(cvr_label, cvr_pred)
        print("conversion_AUC = " + "{:.4f}".format(auc_cov))
        mean_loss = np.mean(total_loss)
        print("Loss = " + "{:.4f}".format(mean_loss))
        return mean_loss, auc_cov

    def train_all_epochs(self, start_epoch=1):
        n_epoches = self.config.n_epochs
        learning_rate = self.config.learning_rate
        batch_size = self.config.batch_size

        for epoch in range(start_epoch, n_epoches + 1):
            print('\n', '-' * 30, 'Train epoch: %d' % epoch, '-' * 30, '\n')
            start_time = time.time()

            print("Training...")
            result = self.train_one_epoch(batch_size, learning_rate)

    def train_until_cov(self):
        learning_rate = self.config.learning_rate
        batch_size = self.config.batch_size

        epoch = 1
        losses = []
        n_epochs = self.config.n_epochs

        while True:
            print('-' * 30, 'Train epoch: %d' % epoch, '-' * 30)
           
            print("Training...")
            # if epoch > 50 or (epoch > 3 and clk_losses[-1] < clk_losses[-2] < clk_losses[-3]):
            #     fetch = [self.train_step, self.clk_loss, self.cov_loss, self.cross_entropy, self.click_prediction,
            #              self.conversion_prediction]
            result = self.train_one_epoch(batch_size, learning_rate)
            
            self.test(epoch)
            loss = result[0]
            losses.append(loss)
            if epoch > 10 and losses[-1] > losses[-2] > losses[-3]:
                break
            epoch += 1
            
    def test(self, epoch):
        batch_size = self.config.batch_size
        total_loss = []
        cvr_pred = []
        cvr_label = []
        infile = open(self.test_dataset, 'rb')
        while True:
            batch = loadrnnattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            test_data, test_compaign_data, test_label, test_seqlen = batch
            if len(test_label) != batch_size:
                break
            feed_dict = {
                self.input: test_data,
                self.labels: test_label,
                self.seqlen: test_seqlen,
                self.is_training: False,
                self.keep_prob: 1
            }
            fetches = [self.cross_entropy, self.prediction]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            loss, cvr = result
            total_loss.append(loss)
            cvr_pred += cvr.tolist()
            cvr_label += test_label
        auc_cov = roc_auc_score(cvr_label, cvr_pred)
        log = log_loss(cvr_label, cvr_pred)
        print("loglikelihood = " + "{:.4f}".format(log))
        print("conversion_AUC = " + "{:.4f}".format(auc_cov))
        mean_loss = np.mean(total_loss)
        print("Loss = " + "{:.4f}".format(mean_loss))
        return auc_cov

    def attr(self):
        filec = open('i2c_converted.pkl', 'rb')
        Channel_Set = pkl.load(filec)
        Channel_value = {}
        Channel_time = {}
        infile = open(self.test_dataset, 'rb')
        outfile = open('rnn_withattention.txt', 'w')
        while True:
            batch = loadrnnattention(self.config.batch_size, self.config.seq_max_len, self.config.feature_number,
                                     infile)
            test_data, test_compaign_data, test_label, test_seqlen = batch
            if len(test_label) != self.config.batch_size:
                break
            feed_dict = {
                self.input: test_data,
                self.labels: test_label,
                self.seqlen: test_seqlen,
                self.is_training: False,
                self.keep_prob: 1
            }
            fetches = self.attention
            attention = self.sess.run(fetches, feed_dict=feed_dict)
            for i in range(self.config.batch_size):
                if test_label[i] != 0:
                    for j in range(test_seqlen[i]):
                        # if click_label[i][j] == 1:
                        index = Channel_Set[str(test_data[i][j][2])]
                        v = attention[j][0][i]
                        if index in Channel_value:
                            Channel_value[index] += v
                            Channel_time[index] += 1
                        else:
                            Channel_value[index] = v
                            Channel_time[index] = 1

        for key in Channel_value:
            outfile.write(key + '\t' + str(Channel_value[key] / Channel_time[key]) + '\n')

    def vertical_attr(self, lenth):
        batch_size = self.config.batch_size
        value = [0.] * lenth
        infile = open(self.test_dataset, 'rb')
        while True:
            batch = loadrnnattention(self.config.batch_size, self.config.seq_max_len, self.config.feature_number,
                                     infile)
            test_data, test_compaign_data, test_label, test_seqlen = batch
            if len(test_label) != self.config.batch_size:
                break
            feed_dict = {
                self.input: test_data,
                self.labels: test_label,
                self.seqlen: test_seqlen,
                self.is_training: False,
                self.keep_prob: 1
            }
            fetches = self.attention
            attention = self.sess.run(fetches, feed_dict=feed_dict)
            for i in range(batch_size):
                if test_seqlen[i] == lenth and test_label[i] == 1:
                    for j in range(test_seqlen[i]):
                        # if click_label[i][j] == 1:
                        v = attention[j][0][i]
                        value[j] += v

        print(value)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 10417} id="bWNlvu7NJpco" outputId="f17785ca-ed2d-4913-c5c8-86e146ff7edb"
if __name__ == '__main__':
    traindata = 'train_usr.yzx.txt'
    testdata = 'test_usr.yzx.txt'
    learning_rate = 0.005
    batch_size = 512
    mu = 1e-6

    C = config(max_features=5897, learning_rate=learning_rate, batch_size=batch_size, feature_number=12,
               seq_max_len=20, n_input=2,
               embedding_output=256, n_hidden=512, n_classes=2, n_epochs=10, isseq=True, miu=mu)
    path = '../Model/ARNN'
    model = RnnWithattention(path, traindata, testdata, C)
    model.train_until_cov()
    model.test(0)
    model.attr()
```

<!-- #region id="l6djdRIDKF5j" -->
## DARNN
<!-- #endregion -->

```python id="eO5oGZDcJ0g4"
class DualAttention(object):
    def __init__(self, path, train_dataset, test_dataset, config):
        self.graph = tf.Graph()

        self._path = path
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self._save_path, self._logs_path = None, None
        self.config = config
        self.batches_step = 0
        self.cross_entropy, self.train_step, self.prediction = None, None, None
        self.data_shape = config.data_shape
        self.label_shape = config.label_shape
        self.n_classes = config.n_classes
        self.attention = None
        with self.graph.as_default():
            self._define_inputs()
            self._build_graph()
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self._initialize_session()

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = '%s/checkpoint' % self._path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, 'model.ckpt')
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = '%s/logs' % self._path
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            self._logs_path = logs_path
        return self._logs_path

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        label_shape = [None]
        self.input = tf.placeholder(
            tf.float32,
            shape=[None, self.config.seq_max_len, 12]
        )
        self.click_label = tf.placeholder(
            tf.float32,
            shape=[None, self.config.seq_max_len, self.config.n_classes]
        )
        self.labels = tf.placeholder(
            tf.float32,
            shape=label_shape,
            name='labels'
        )
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate'
        )
        self.seqlen = tf.placeholder(
            tf.int32,
            shape=[None],
            name='seqlen'
        )
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    def _initialize_session(self, set_logs=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.sess.run(self.initializer)
        if set_logs:
            logswriter = tf.summary.FileWriter
            self.summary_writer = logswriter(self.logs_path, graph=self.graph)

    def _build_graph(self):
        x = self.input
        batchsize = tf.shape(x)[0]
        embedding_matrix = tf.Variable(
            tf.random_normal([self.config.max_features, self.config.embedding_output], stddev=0.1))
        x1, x2 = tf.split(x, [2, 10], 2)
        x2 = tf.to_int32(x2)
        x2 = tf.nn.embedding_lookup(embedding_matrix, x2)
        x2 = tf.reshape(x2, [-1, self.config.seq_max_len, 10 * self.config.embedding_output])
        x = tf.concat((x1, x2), axis=2)
        # Define Variables
        W = tf.Variable(tf.random_normal([self.config.n_classes, self.config.n_hidden], stddev=0.1), name='W')
        U = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='U')
        C = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='C')
        U_a = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='U_a')
        # W_a = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='W_a')
        v_a = tf.Variable(tf.random_normal([self.config.n_hidden], stddev=0.1), name='v_a')
        W_z = tf.Variable(tf.random_normal([self.config.n_classes, self.config.n_hidden], stddev=0.1), name='W_z')
        U_z = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='U_z')
        C_z = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='C_z')
        W_r = tf.Variable(tf.random_normal([self.config.n_classes, self.config.n_hidden], stddev=0.1), name='W_r')
        U_r = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='U_r')
        C_r = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='C_r')
        W_s = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='W_s')
        W_o = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_classes], stddev=0.1), name='W_o')
        b_o = tf.Variable(tf.random_normal([self.config.n_classes], stddev=0.1), name='b_o')
        W_h = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='W_h')
        U_s = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='U_s')
        W_C = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_hidden], stddev=0.1), name='W_C')
        W_x1 = tf.Variable(tf.random_normal([self.config.n_input, self.config.n_hidden], stddev=0.1), name='W_x1')
        W_x2 = tf.Variable(tf.random_normal([self.config.n_input, self.config.n_hidden], stddev=0.1), name='W_x2')
        W_x3 = tf.Variable(tf.random_normal([self.config.n_input, self.config.n_hidden], stddev=0.1), name='W_x3')
        v_a2 = tf.Variable(tf.random_normal([self.config.n_hidden], stddev=0.1), name='v_a2')
        v_a3 = tf.Variable(tf.random_normal([self.config.n_hidden], stddev=0.1), name='v_a3')
        W_c = tf.Variable(tf.random_normal([self.config.n_hidden], stddev=0.1), name='W_c')
        b_c = tf.Variable(tf.random_normal([self.config.batch_size]))

        index = tf.range(0, batchsize) * self.config.seq_max_len + (self.seqlen - 1)
        x_last = tf.gather(params=tf.reshape(x, [-1, self.config.n_input]), indices=index)
        x = tf.transpose(x, [1, 0, 2])
        # x = tf.reshape(x,[-1, self.config.n_input])
        # x = tf.split(axis=0, num_or_size_splits=self.config.seq_max_len, value=x)
        y = tf.transpose(self.click_label, [1, 0, 2])
        y = tf.reshape(y, [-1, self.config.n_classes])
        y = tf.split(value=y, num_or_size_splits=self.config.seq_max_len, axis=0)
        # encoder
        gru_cell = tf.contrib.rnn.GRUCell(self.config.n_hidden)
        gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=self.keep_prob,
                                                 output_keep_prob=self.keep_prob)
        states_h, last_h = tf.nn.dynamic_rnn(gru_cell, x, self.seqlen, dtype=tf.float32, time_major=True)
        states_h = tf.reshape(states_h, [-1, self.config.n_hidden])
        states_h = tf.split(states_h, self.config.seq_max_len, 0)

        Uhs = []
        for state_h in states_h:
            Uh = tf.matmul(state_h, U_a)
            Uhs.append(Uh)

        # decoder
        state_s = tf.tanh(tf.matmul(states_h[-1], W_s))
        # s0 =  tanh(Ws * h_last)
        states_s = [state_s]
        outputs = []
        output = tf.zeros(shape=[batchsize, self.config.n_classes])
        for i in range(self.config.seq_max_len):
            # e = []
            # for Uh in Uhs:
            # 	e_ = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(states_s[i], W_a) + Uh), v_a), reduction_indices=1)
            # 	e.append(e_)
            # e = tf.stack(e)
            # # (seq_max_len, batch_size)
            # a1 = tf.nn.softmax(e, dim=0)
            # a1 = tf.split(a1, self.config.seq_max_len, 0)
            # c = tf.zeros([batchsize, self.config.n_hidden])
            # for j in range(self.config.seq_max_len):
            # 	c = c + tf.multiply(states_h[j], tf.transpose(a1[j]))
            c = states_h[-1]
            if self.is_training == True:
                last_output = y[i]
            else:
                last_output = tf.nn.softmax(output)
            r = tf.sigmoid(tf.matmul(last_output, W_r) + tf.matmul(states_s[i], U_r) + tf.matmul(c, C_r))
            z = tf.sigmoid(tf.matmul(last_output, W_z) + tf.matmul(states_s[i], U_z) + tf.matmul(c, C_z))
            s_hat = tf.tanh(tf.matmul(last_output, W) + tf.matmul(tf.multiply(r, states_s[i]), U) + tf.matmul(c, C))
            state_s = tf.multiply(tf.subtract(1.0, z), states_s[i]) + tf.multiply(z, s_hat)
            states_s.append(state_s)
            state_s = tf.nn.dropout(state_s, self.keep_prob)
            output = tf.matmul(state_s, W_o) + b_o
            outputs.append(output)

        e2 = []
        e3 = []
        Ux = tf.matmul(x_last, W_x1)
        Ux2 = tf.matmul(x_last, W_x2)
        for i in range(self.config.seq_max_len):
            e2_ = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(states_h[i], W_h) + Ux), v_a2), reduction_indices=1)
            e2.append(e2_)
            e3_ = tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(states_s[i], U_s) + Ux2), v_a3), reduction_indices=1)
            e3.append(e3_)
        e2 = tf.stack(e2)
        e3 = tf.stack(e3)
        a2 = tf.nn.softmax(e2, dim=0)
        a3 = tf.nn.softmax(e3, dim=0)
        a2 = tf.split(a2, self.config.seq_max_len, 0)
        a3 = tf.split(a3, self.config.seq_max_len, 0)
        c2 = tf.zeros([batchsize, self.config.n_hidden])
        c3 = tf.zeros([batchsize, self.config.n_hidden])
        for i in range(self.config.seq_max_len):
            c2 = c2 + tf.multiply(states_h[i], tf.transpose(a2[i]))
            c3 = c3 + tf.multiply(states_s[i], tf.transpose(a3[i]))
        e4 = []
        Ux3 = tf.matmul(x_last, W_x3)
        e4.append(tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(c2, W_C) + Ux3), v_a), reduction_indices=1))
        e4.append(tf.reduce_sum(tf.multiply(tf.tanh(tf.matmul(c3, W_C) + Ux3), v_a), reduction_indices=1))
        e4 = tf.stack(e4)
        a4 = tf.split(tf.nn.softmax(e4, dim=0), 2, 0)
        C = tf.multiply(c2, tf.transpose(a4[0])) + tf.multiply(c3, tf.transpose(a4[1]))
        # self.attention = C
        cvr = tf.reduce_sum(tf.multiply(C, W_c), axis=1) + b_c
        cvr = tf.nn.dropout(cvr, keep_prob=self.keep_prob)
        conversion_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=cvr))
        cvr = tf.nn.sigmoid(cvr)
        mask = tf.sequence_mask(self.seqlen, self.config.seq_max_len)
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        # (batchsize, max_seq_len, n_classes)
        loss_click = tf.nn.softmax_cross_entropy_with_logits(labels=self.click_label, logits=outputs)
        loss_click = tf.boolean_mask(loss_click, mask)
        loss_click = tf.reduce_mean(loss_click)
        click_pred = tf.nn.softmax(outputs)
        loss = loss_click + conversion_loss
        for v in tf.trainable_variables():
            loss += self.config.miu * tf.nn.l2_loss(v)
            loss_click += self.config.miu * tf.nn.l2_loss(v)
            conversion_loss += self.config.miu * tf.nn.l2_loss(v)

        global_step = tf.Variable(0, trainable=False)
        start_learningrate = self.config.learning_rate
        cov_learning_rate = tf.train.exponential_decay(start_learningrate, global_step, 50000, 0.96)
        clk_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        optimizer = tf.train.AdamOptimizer(learning_rate=cov_learning_rate)
        gvs_clk, v_clk = zip(*clk_optimizer.compute_gradients(loss_click))
        gvs_clk, _ = tf.clip_by_global_norm(gvs_clk, 5.0)
        gvs_clk = zip(gvs_clk, v_clk)
        gvs_cov, v_cov = zip(*optimizer.compute_gradients(conversion_loss))
        gvs_cov, _ = tf.clip_by_global_norm(gvs_cov, 5.0)
        gvs_cov = zip(gvs_cov, v_cov)
        gvs, v = zip(*optimizer.compute_gradients(loss))
        gvs, _ = tf.clip_by_global_norm(gvs, 5.0)
        gvs = zip(gvs, v)

        self.clk_train_step = clk_optimizer.apply_gradients(gvs_clk)
        self.cov_train_step = optimizer.apply_gradients(gvs_cov, global_step=global_step)
        self.train_step = optimizer.apply_gradients(gvs, global_step=global_step)
        self.click_prediction = click_pred
        self.conversion_prediction = cvr
        self.cross_entropy = loss
        self.clk_loss = loss_click
        self.cov_loss = conversion_loss
        self.click_attention = a3
        self.attention = a4
        self.impression_attention = a2

    def train_one_epoch(self, batch_size, learning_rate, fetches):
        total_loss = []
        total_clk_loss = []
        total_cov_loss = []
        clk_pred = []
        clk_label = []
        cvr_pred = []
        cvr_label = []
        infile = open(self.train_dataset, 'rb')
        while True:
            batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            train_data, train_compaign_data, click_label, train_label, train_seqlen = batch
            if len(train_label) != batch_size:
                break
            feed_dict = {
                self.input: train_data,
                self.click_label: click_label,
                self.labels: train_label,
                self.seqlen: train_seqlen,
                self.learning_rate: learning_rate,
                self.is_training: True,
                self.keep_prob: 0.5
            }
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, clk_loss, cov_loss, loss, clk, cvr = result
            total_loss.append(loss)
            total_clk_loss.append(clk_loss)
            total_cov_loss.append(cov_loss)
            clk = np.reshape(clk, (-1, 2)).tolist()
            click_label = np.reshape(click_label, (-1, 2)).tolist()
            clk_pred += clk
            clk_label += click_label
            cvr_pred += cvr.tolist()
            cvr_label += train_label
        clk_pred = np.array(clk_pred)
        auc_clk = roc_auc_score(np.argmax(clk_label, 1), clk_pred[:, 1])
        auc_cov = roc_auc_score(cvr_label, cvr_pred)
        print("click_AUC = " + "{:.4f}".format(auc_clk))
        print("conversion_AUC = " + "{:.4f}".format(auc_cov))
        mean_loss = np.mean(total_loss)
        print("Loss = " + "{:.4f}".format(mean_loss))
        mean_clk_loss = np.mean(total_clk_loss)
        mean_cov_loss = np.mean(total_cov_loss)
        print("Clk Loss = " + "{:.4f}".format(mean_clk_loss))
        print("Cov_Loss = " + "{:.4f}".format(mean_cov_loss))
        return mean_loss, mean_cov_loss, mean_clk_loss, auc_clk, auc_cov

    def train_all_epochs(self, start_epoch=1):
        n_epoches = self.config.n_epochs
        learning_rate = self.config.learning_rate
        batch_size = self.config.batch_size

        total_start_time = time.time()
        for epoch in range(start_epoch, n_epoches + 1):
            print('\n', '-' * 30, 'Train epoch: %d' % epoch, '-' * 30, '\n')
            start_time = time.time()

            print("Training...")
            result = self.train_one_epoch(batch_size, learning_rate,
                                          [self.clk_train_step, self.clk_loss, self.cov_loss, self.cross_entropy,
                                           self.click_prediction,
                                           self.conversion_prediction])
            self.log(epoch, result, prefix='train')
            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epoches - epoch) * time_per_epoch)
            print('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))
            ))

        self.save_model()

        total_training_time = time.time() - total_start_time
        print('\nTotal training time: %s' % str(timedelta(seconds=total_training_time)))

    def train_until_cov(self):
        learning_rate = self.config.learning_rate
        batch_size = self.config.batch_size

        total_start_time = time.time()
        epoch = 1
        losses = []
        clk_losses = []
        n_epochs = self.config.n_epochs
        fetch = [self.clk_train_step, self.clk_loss, self.cov_loss, self.cross_entropy, self.click_prediction,
                 self.conversion_prediction]
        flag = 0
        while True:
            print('-' * 30, 'Train epoch: %d' % epoch, '-' * 30)
            start_time = time.time()

            print("Training...")
            if flag == 0 and (epoch > 10 or (epoch > 3 and clk_losses[-1] < clk_losses[-2] < clk_losses[-3])):
                flag = epoch
                fetch = [self.train_step, self.clk_loss, self.cov_loss, self.cross_entropy, self.click_prediction,
                         self.conversion_prediction]
            result = self.train_one_epoch(batch_size, learning_rate, fetch)
            self.log(epoch, result, prefix='train')

            loss = self.test(epoch)
            time_per_epoch = time.time() - start_time
            losses.append(loss[0])
            clk_losses.append(loss[1])
            if flag != 0 and (epoch > flag + 3 and losses[-1] < losses[-2] < losses[-3]):
                self.save_model()
                break
            print('Time per epoch: %s' % (
                str(timedelta(seconds=time_per_epoch))
            ))
            epoch += 1
            self.save_model()

        total_training_time = time.time() - total_start_time
        print('\nTotal training time: %s' % str(timedelta(seconds=total_training_time)))

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def test(self, epoch):
        batch_size = self.config.batch_size
        total_loss = []
        clk_pred = []
        clk_label = []
        cvr_pred = []
        total_clk_loss = []
        total_cov_loss = []
        cvr_label = []
        infile = open(self.test_dataset, 'rb')
        while True:
            batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            test_data, test_compaign_data, click_label, test_label, test_seqlen = batch
            if len(test_label) != batch_size:
                break
            feed_dict = {
                self.input: test_data,
                self.click_label: click_label,
                self.labels: test_label,
                self.seqlen: test_seqlen,
                self.is_training: False,
                self.keep_prob: 1
            }
            fetches = [self.clk_loss, self.cov_loss, self.cross_entropy, self.click_prediction,
                       self.conversion_prediction]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            clk_loss, cov_loss, loss, clk, cvr = result
            total_loss.append(loss)
            total_clk_loss.append(clk_loss)
            total_cov_loss.append(cov_loss)
            clk = np.reshape(clk, (-1, 2)).tolist()
            click_label = np.reshape(click_label, (-1, 2)).tolist()
            clk_pred += clk
            clk_label += click_label
            cvr_pred += cvr.tolist()
            cvr_label += test_label
        clk_pred = np.array(clk_pred)
        auc_clk = roc_auc_score(np.argmax(clk_label, 1), clk_pred[:, 1])
        auc_cov = roc_auc_score(cvr_label, cvr_pred)
        loglikelyhood = -log_loss(cvr_label, cvr_pred)
        print("click_AUC = " + "{:.4f}".format(auc_clk))
        print("conversion_AUC = " + "{:.4f}".format(auc_cov))
        print("loglikelyhood = " + "{:.4f}".format(loglikelyhood))
        mean_loss = np.mean(total_loss)
        print("Loss = " + "{:.4f}".format(mean_loss))
        mean_clk_loss = np.mean(total_clk_loss)
        mean_cov_loss = np.mean(total_cov_loss)
        print("Clk Loss = " + "{:.4f}".format(mean_clk_loss))
        print("Cov_Loss = " + "{:.4f}".format(mean_cov_loss))
        self.log(epoch, [mean_loss, mean_cov_loss, mean_clk_loss, auc_clk, auc_cov], 'test')
        return auc_cov, auc_clk

    def log(self, epoch, result, prefix):
        s = prefix + '\t' + str(epoch)
        for i in result:
            s += ('\t' + str(i))
        fout = open("%s/%s_%s_%s_%s" % (
            self.logs_path, str(self.config.learning_rate), str(self.config.batch_size), str(self.config.n_hidden),
            str(self.config.miu)),
                    'a')
        fout.write(s + '\n')

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception:
            raise IOError('Failed to load model from save path: %s' % self.save_path)
        print('Successfully load model from save path: %s' % self.save_path)

    def attr(self):
        filec = open('i2c_converted.pkl', 'rb')
        Channel_Set = pkl.load(filec)
        Channel_value = {}
        Channel_time = {}
        batch_size = 128
        infile = open(self.test_dataset, 'rb')
        outfile = open('trainlambda.txt', 'w')
        while True:
            batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            test_data, test_compaign_data, click_label, test_label, test_seqlen = batch
            if len(test_label) != batch_size:
                break
            feed_dict = {
                self.input: test_data,
                self.labels: test_label,
                self.seqlen: test_seqlen,
                self.is_training: False,
                self.click_label: click_label,
                self.keep_prob: 1
            }
            fetches = [self.attention, self.click_attention, self.impression_attention]
            lamb, click_attention, imp_attention = self.sess.run(fetches, feed_dict=feed_dict)
            # print(len(lamb), len(lamb[0]), len(lamb[0][0]))
            # print(len(click_attention), len(click_attention[0]), len(click_attention[0][0]))
            # print(imp_attention.shape)
            click_label = np.array(click_label)
            click_label = click_label[:, :, 1]
            for i in range(batch_size):
                if test_label[i] != 0:
                    for j in range(test_seqlen[i]):
                        # if click_label[i][j] == 1:
                        index = Channel_Set[str(test_data[i][j][2])]
                        v = click_attention[j][0][i] * lamb[0][0][i] + imp_attention[j][0][i] * lamb[1][0][i]
                        if index in Channel_value:
                            Channel_value[index] += v
                            Channel_time[index] += 1
                        else:
                            Channel_value[index] = v
                            Channel_time[index] = 1
        # infile = open(self.train_dataset, 'rb')
        # while True:
        #     batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
        #     test_data, test_compaign_data, click_label, test_label, test_seqlen = batch
        #     if len(test_label) != batch_size:
        #         break
        #     feed_dict = {
        #         self.input: test_data,
        #         self.labels: test_label,
        #         self.seqlen: test_seqlen,
        #         self.is_training: False,
        #         self.click_label: click_label,
        #         self.keep_prob: 1
        #     }
        #     fetches = self.attention
        #     attention = self.sess.run(fetches, feed_dict=feed_dict)
        #     click_label = np.array(click_label)
        #     click_label = click_label[:, :, 1]
        #     for i in range(batch_size):
        #         for j in range(test_seqlen[i]):
        #             # if click_label[i][j] == 1:
        #             index = Channel_Set[str(test_data[i][j][2])]
        #             v = attention[j][0][i]
        #             if Channel_value.has_key(index):
        #                 Channel_value[index] += v
        #                 Channel_time[index] += 1
        #             else:
        #                 Channel_value[index] = v
        #                 Channel_time[index] = 1

        for key in Channel_value:
            outfile.write(key + '\t' + str(Channel_value[key]) + '\n')

    # for (i, j, k) in zip(Channel_List, Channel_value, Channel_time):
    #     if k != 0:
    #         print(i + '\t' + str(j / k))
    #     else:
    #         print(i + '\t0')
    def cal_lamba(self):
        filec = open('i2c_converted.pkl', 'rb')
        Channel_Set = pkl.load(filec)
        Channel_value = {}
        Channel_time = {}
        batch_size = 128
        infile = open(self.test_dataset, 'rb')
        outfile = open('./dis_lambda.pkl', 'wb')
        l = []
        while True:
            batch = loaddualattention(batch_size, self.config.seq_max_len, self.config.feature_number, infile)
            test_data, test_compaign_data, click_label, test_label, test_seqlen = batch
            if len(test_label) != batch_size:
                break
            feed_dict = {
                self.input: test_data,
                self.labels: test_label,
                self.seqlen: test_seqlen,
                self.is_training: False,
                self.click_label: click_label,
                self.keep_prob: 1
            }
            fetches = self.attention
            lamb = self.sess.run(fetches, feed_dict=feed_dict)
            # print(len(lamb), len(lamb[0]), len(lamb[0][0]))
            # print(len(click_attention), len(click_attention[0]), len(click_attention[0][0]))
            # print(imp_attention.shape)
            click_label = np.array(click_label)
            click_label = click_label[:, :, 1]
            for i in range(batch_size):
                for j in range(test_seqlen[i]):
                    v = lamb[0][0][i]
                    l.append(v)
        pkl.dump(l, outfile)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1951} id="fIswwHVKKRzW" outputId="addfedde-b1cd-4469-b5e6-16ccc8a1c2f3"
if __name__ == '__main__':
    traindata = 'train_usr.yzx.txt'
    testdata = 'test_usr.yzx.txt'
    learning_rate = 0.001
    batch_size = 256
    mu = 1e-6

    C = config(max_features=5897, learning_rate=learning_rate, batch_size=batch_size, feature_number=12,
               seq_max_len=20, n_input=2,
               embedding_output=256, n_hidden=512, n_classes=2, n_epochs=50, isseq=True, miu=mu)
    path = '../Model/DARNN'
    model = DualAttention(path, traindata, testdata, C)
    model.train_until_cov()
    model.test(0)
    model.attr()
```

<!-- #region id="J1GKCY3b_X0g" -->
## Attribution Model Comparison Part 2/2
<!-- #endregion -->

```python id="x24qfshk_o1P"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```

```python id="X0e1eZeH_o1U" outputId="a5bbad4f-7628-4739-90c3-2663a1447977"
path = 'attribution/'
data_lr = pd.read_csv(path+'LR.txt', sep="\t", header=None); data_lr.columns = ['channel','attribution LR']
data_ah = pd.read_csv(path+'AH.txt', sep="\t", header=None); data_ah.columns = ['channel','attribution AH']
data_arnn = pd.read_csv(path+'ARNN.txt', sep="\t", header=None); data_arnn.columns = ['channel','attribution ARNN']
data_darnn = pd.read_csv(path+'DARNN.txt', sep="\t", header=None); data_darnn.columns = ['channel','attribution DARNN']

data_darnn.head()
```

```python id="Ty8-ZvgK_o1Y" outputId="8839c51a-6c22-4b70-ad15-f42c3ba69194"
print('Shape of Logistic Regression:', data_lr.shape)
print('Shape of Additive Hazard:', data_ah.shape)
print('Shape of Attention RNN:', data_arnn.shape)
print('Shape of Dual Attention RNN:', data_darnn.shape)
```

```python id="W9364yJH_o1c" outputId="57414aba-3fd4-4430-9ef4-b519f474503f"
df = pd.merge(data_lr, data_ah, on=['channel'], how='inner')
df = pd.merge(df, data_arnn, on=['channel'], how='inner')
df = pd.merge(df, data_darnn, on=['channel'], how='inner')
df.shape
```

```python id="AletcBOV_o1f" outputId="8f6e0a9a-a237-4357-c925-377124a63f64"
df.head(10)
```

```python id="lfrTJx-7_o1i"
df.drop(df[df['channel']=='other'].index, axis=0, inplace=True)
```

```python id="orWgI4We_o1m"
df['channel'] = pd.cut(df['channel'].astype('int'), 10, labels=['A', 'B', 'C','D','E','F','G','H','I','J'])
```

```python id="k6Nz9K90_o1p" outputId="c0244821-2536-47f2-bd8f-e12779b968d7"
df.head()
```

```python id="oIQAz7xJ_o1s" outputId="a8ebaa6b-75a6-4c0c-fcfe-182490b24496"
df = df.groupby(by='channel').sum()
df
```

```python id="Z44D-5ln_o1v" outputId="6988def3-22de-4f1f-9563-8644c9e07d89"
for col in df.columns:
    df[col] = df[col]/df[col].sum()
df.reset_index(inplace=True)
df
```

```python id="dlWsB9Wr_o1y" outputId="3dd35e89-0967-49cd-c3fb-a96160bc5def"
f, axes = plt.subplots(2, 2, figsize=(12,8))

sns.barplot(x='channel', y='attribution LR', data=df, ax=axes[0,0], palette="Blues")
sns.barplot(x='channel', y='attribution AH', data=df, ax=axes[0,1], palette="Greens")
sns.barplot(x='channel', y='attribution ARNN', data=df, ax=axes[1,0], palette="BuGn_r")
sns.barplot(x='channel', y='attribution DARNN', data=df, ax=axes[1,1], palette="Purples")

plt.show()
```

<!-- #region id="TkK_BGXR_yYe" -->
---
<!-- #endregion -->
