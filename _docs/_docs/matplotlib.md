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

<!-- #region id="6OX43mmyhfbD" -->


### What You’ll Learn 
- how to set the amount of time each slide will take to finish 
- how to include code snippets 
- how to hyperlink items 
- how to include images 
- other stuff
<!-- #endregion -->

```python id="2mSCSV_BprjK" executionInfo={"status": "ok", "timestamp": 1637854542376, "user_tz": -330, "elapsed": 36, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import matplotlib as mpl
import matplotlib.pyplot as plt
```

<!-- #region id="8-bdkZQYq4iP" -->
<button>
  [Download Data](https://vega.github.io/vega-datasets/data/movies.json)
</button>
<!-- #endregion -->

```python id="j98eKD-QrYGJ" executionInfo={"status": "ok", "timestamp": 1637854542383, "user_tz": -330, "elapsed": 41, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
plt.style.use('ggplot')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="BBiEf1BErYDZ" executionInfo={"status": "ok", "timestamp": 1637854542387, "user_tz": -330, "elapsed": 44, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bde6e33e-565c-45af-efdb-1f8140db3e5e"
import numpy as np
x = np.linspace(0, 10, 100)

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--');
```

<!-- #region id="aqcO-uvkqesZ" -->
<aside class="positive">
Matplotlib provides all the essential plotting features
</aside>
<!-- #endregion -->

<!-- #region id="aVkyJ-m4hokv" -->
<!-- ------------------------ -->
## Setting Duration
Duration: 2
<!-- #endregion -->

<!-- #region id="0YiZssXzhwJW" -->
To indicate how long each slide will take to go through, set the `Duration` under each Heading 2 (i.e. `##`) to an integer. 
The integers refer to minutes. If you set `Duration: 4` then a particular slide will take 4 minutes to complete. 

The total time will automatically be calculated for you and will be displayed on the codelab once you create it.
<!-- #endregion -->

<!-- #region id="B1MpAsYYqqJX" -->
<aside class="negative">
Matplotlib is not ideal for interactive plotting.
</aside>
<!-- #endregion -->

```python id="ULFZpQWWpsjB" executionInfo={"status": "ok", "timestamp": 1637854542390, "user_tz": -330, "elapsed": 37, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def hist_and_lines():
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(10))
    ax[1].legend(['a', 'b', 'c'], loc='lower left')
```

```python id="DAF2EMihr63s" colab={"base_uri": "https://localhost:8080/", "height": 269} executionInfo={"status": "ok", "timestamp": 1637854543937, "user_tz": -330, "elapsed": 1583, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b21919ad-2518-45e8-80c4-1472b9723e37"
hist_and_lines()
```

```python id="pMFWlyBMvM-K" executionInfo={"status": "ok", "timestamp": 1637854608015, "user_tz": -330, "elapsed": 2057, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
import matplotlib
import seaborn as sns
import functools
  
sns.set(font_scale=1.3)
sns.set_style("darkgrid", {"axes.facecolor": ".95"})

import matplotlib.pyplot as plt

def format_plot(x=None, y=None): 
  # plt.grid(False)
  ax = plt.gca()
  if x is not None:
    plt.xlabel(x, fontsize=20)
  if y is not None:
    plt.ylabel(y, fontsize=20)
  
def finalize_plot(shape=(1, 1)):
  plt.gcf().set_size_inches(
    shape[0] * 1.5 * plt.gcf().get_size_inches()[1], 
    shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
  plt.tight_layout()

legend = functools.partial(plt.legend, fontsize=10)


def plot_fn(train, test, *fs):
  train_xs, train_ys = train

  plt.plot(train_xs, train_ys, 'ro', markersize=10, label='train')
  
  if test != None:
    test_xs, test_ys = test
    plt.plot(test_xs, test_ys, 'k--', linewidth=3, label='$f(x)$')

    for f in fs:
      plt.plot(test_xs, f(test_xs), '-', linewidth=3)

  plt.xlim([-np.pi, np.pi])
  plt.ylim([-1.5, 1.5])

  format_plot('$x$', '$f$')
```

```python id="geu7fyUAvZTF" executionInfo={"status": "ok", "timestamp": 1637855022459, "user_tz": -330, "elapsed": 739, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import numpy as np

key = 10
train_points = 5
test_points = 50
noise_scale = 1e-1

target_fn = lambda x: np.sin(x)

train_xs = np.random.uniform(size=(train_points, 1), low=-np.pi, high=np.pi)
train_ys = target_fn(train_xs)
train = (train_xs, train_ys)

test_xs = np.linspace(-np.pi, np.pi, test_points)
test_xs = np.reshape(test_xs, (test_points, 1))
test_ys = target_fn(test_xs)
test = (test_xs, test_ys)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 341} id="7l-iAfU0vw2G" executionInfo={"status": "ok", "timestamp": 1637855043741, "user_tz": -330, "elapsed": 2923, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a930bc06-c0ed-4e9f-8652-a3c9e08ed6f0"
plot_fn(train, test)
legend(loc='upper left')
finalize_plot((0.85, 0.6))
```

```python id="hqd4kQPkvxDi"

```
