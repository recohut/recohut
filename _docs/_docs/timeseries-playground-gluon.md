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

```python id="0_jOa2U5-ymg" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 496} outputId="971af8db-7567-4715-c018-5b127d89f231" executionInfo={"status": "ok", "timestamp": 1586688381158, "user_tz": -330, "elapsed": 5270, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!pip install --upgrade mxnet==1.6 gluonts
```

<!-- #region id="MK277d79_sLq" colab_type="text" -->
Example: In this example we will use the volume of tweets mentioning the AMZN ticker symbol.
<!-- #endregion -->

```python id="z9s2jwHO_jkr" colab_type="code" colab={}
import pandas as pd
url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
df = pd.read_csv(url, header=0, index_col=0)
```

```python id="FcrRJkaE_2RG" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 262} outputId="bd15251e-13cf-4efe-ee4a-1efaacb426b9" executionInfo={"status": "ok", "timestamp": 1586688489383, "user_tz": -330, "elapsed": 5764, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import matplotlib.pyplot as plt
df[:100].plot(linewidth=2)
plt.xticks([])
plt.grid(which='both')
plt.show()
```

<!-- #region id="olBxyzp6Agqv" colab_type="text" -->
We can now prepare a training dataset for our model to train on. Datasets in GluonTS are essentially iterable collections of dictionaries: each dictionary represents a time series with possibly associated features. For this example, we only have one entry, specified by the "start" field which is the timestamp of the first datapoint, and the "target" field containing time series data. For training, we will use data up to midnight on April 5th, 2015.
<!-- #endregion -->

```python id="8oNywgx1_4dT" colab_type="code" colab={}
from gluonts.dataset.common import ListDataset
training_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-05 00:00:00"]}],
    freq = "5min"
)
```

<!-- #region id="w4wqCQHeA0pW" colab_type="text" -->
A forecasting model in GluonTS is a predictor object. One way of obtaining predictors is by training a correspondent estimator. Instantiating an estimator requires specifying the frequency of the time series that it will handle, as well as the number of time steps to predict. In our example we're using 5 minutes data, so freq="5min", and we will train a model to predict the next hour, so prediction_length=12. We also specify some minimal training options.
<!-- #endregion -->

```python id="2ZqBunIT__dl" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 850} outputId="420c5015-f6c5-43e9-cedf-31da294e0826" executionInfo={"status": "ok", "timestamp": 1586688707874, "user_tz": -330, "elapsed": 58978, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

estimator = DeepAREstimator(freq="5min", prediction_length=12, trainer=Trainer(epochs=10))
predictor = estimator.train(training_data=training_data)
```

<!-- #region id="zCkecXMeBIOh" colab_type="text" -->
We're now ready to make predictions: we will forecast the hour following the midnight on April 15th, 2015.
<!-- #endregion -->

```python id="geeBJ2JvA2Z5" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 277} outputId="d7e15b09-976e-458a-a24d-d43d8fe6c1fd" executionInfo={"status": "ok", "timestamp": 1586688844786, "user_tz": -330, "elapsed": 1307, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
test_data = ListDataset(
    [{"start": df.index[0], "target": df.value[:"2015-04-15 00:00:00"]}],
    freq = "5min"
)

from gluonts.dataset.util import to_pandas

for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
    to_pandas(test_entry)[-60:].plot(linewidth=2)
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
plt.grid(which='both')
```

<!-- #region id="1vKb8sMFBrMg" colab_type="text" -->
Note that the forecast is displayed in terms of a probability distribution: the shaded areas represent the 50% and 90% prediction intervals, respectively, centered around the median (dark green line).
<!-- #endregion -->

```python id="YymFSEZWBl6P" colab_type="code" colab={}

```
