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

<!-- #region id="O64DaeQl0L-l" -->
# Scikit Multiflow
<!-- #endregion -->

<!-- #region id="w5_IJ4T10NsI" -->
## Setup
<!-- #endregion -->

```python id="329eLJ3nPyqX"
!pip install -U scikit-multiflow
```

<!-- #region id="qP2t-QeJPzI7" -->
## Train and test a stream classification model in scikit-multiflow
<!-- #endregion -->

<!-- #region id="KE2jwGs3P5C3" -->
In this example, we will use a data stream to train a HoeffdingTreeClassifier and will measure its performance using prequential evaluation.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="EYMMbT92P5ey" executionInfo={"status": "ok", "timestamp": 1635245968195, "user_tz": -330, "elapsed": 21801, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="393fe3fd-0012-4096-b4b9-9609ddc1b0d4"
from skmultiflow.data import WaveformGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.evaluation import EvaluatePrequential

# 1. Create a stream
stream = WaveformGenerator()

# 2. Instantiate the HoeffdingTreeClassifier
ht = HoeffdingTreeClassifier()

# 3. Setup the evaluator
evaluator = EvaluatePrequential(show_plot=False,
                                pretrain_size=200,
                                max_samples=20000)

# 4. Run evaluation
evaluator.evaluate(stream=stream, model=ht)
```

<!-- #region id="gtY1m239QhNs" -->
![](https://scikit-multiflow.readthedocs.io/en/stable/_images/example_classifier_plot.gif)
<!-- #endregion -->

<!-- #region id="NRky6JyjSyn3" -->
## Adaptive Sliding Window (ADWIN) for concept-drift detection

ADWIN adjusts the mean values of the objects and keeps those below a threshold level (epsilon). If the mean values significantly deviate from a threshold, it deletes the corresponding old part. It is adaptive to the changing data. For instance, if the change is taking place the window size will shrink automatically, else if the data is stationary the window size will grow to improve the accuracy.

The intuition behind using ADWIN is to keep statistics from a window of variable size while detecting concept drift. By using the scikit-multiflow library I simulated a distorted data stream with a normal distribution.

The code below is used for catching the concept drift in the normal distribution (with a mean of 0 and a standard deviation of 0.25). I changed the stream values with the indices between 1000 and 2000 with a different normal distribution (with a mean of 1 and a standard deviation of 0.5). Hence, I expected a width change (decrease) between the stream values 1000 till 2000 and an increase in width till the end of the stream.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hYndnezGP9xG" executionInfo={"status": "ok", "timestamp": 1635246587774, "user_tz": -330, "elapsed": 1349, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9eb3b7ac-355a-46f6-8058-8579c58b6ede"
import numpy as np

from skmultiflow.drift_detection.adwin import ADWIN

adwin = ADWIN(delta=0.0002)
SEED = np.random.seed(42)

# Simulating a data stream as a normal distribution of 1's and 0's
mu, sigma = 0, 0.25  # mean and standard deviation
data_stream = np.random.normal(mu, sigma, 4000)

# Changing the data concept from index 1000 to 2000
mu_broken, sigma_broken = 1, 0.5
data_stream[1000:2000] = np.random.normal(mu_broken, sigma_broken, 1000)

width_vs_variance = []

# Adding stream elements to ADWIN and verifying if drift occurred
for idx in range(4000):

    adwin.add_element(data_stream[idx])

    if adwin.detected_change():
        print(f"Change in index {idx} for stream value {data_stream[idx]}")

    width_vs_variance.append((adwin.width, adwin.variance, idx))
```
