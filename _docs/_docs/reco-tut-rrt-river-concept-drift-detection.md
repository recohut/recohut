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

```python id="fbhn_BlE4KbY" executionInfo={"status": "ok", "timestamp": 1630502330857, "user_tz": -330, "elapsed": 2864, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-rrt"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python id="H9fA0_x76d7z"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="lRQIrpkDkN1P"
!pip install -U -q dvc dvc[gdrive]
```

```python id="pwaewDV7k6VC"
!dvc pull
```

```python id="jZm3c_YnloUJ"
!dvc commit && dvc push
```

```python id="S4knsUJY6d76"
!git status
```

```python id="EEiOf79m6d77"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="oxCWLlkjbU25" -->
---
<!-- #endregion -->

<!-- #region pycharm={"name": "#%% md\n"} id="J8jA0rSdIH9-" -->
In the context of data streams, it is assumed that data can change over time. The change in the relationship between the data (features) and the target to learn is known as **Concept Drift**. As examples we can mention, the electricity demand across the year, the stock market, and the likelihood of a new movie to be successful. Let's consider the movie example: Two movies can have similar features such as popular actors/directors, storyline, production budget, marketing campaigns, etc. yet it is not certain that both will be similarly successful. What the target audience *considers* worth watching (and their money) is constantly changing and production companies must adapt accordingly to avoid "box office flops".

## Impact of drift on learning

Concept drift can have a significant impact on predictive performance if not handled properly. Most batch learning models will fail in the presence of concept drift as they are essentially trained on different data. On the other hand, stream learning methods continuously update themselves and adapt to new concepts. Furthermore, drift-aware methods use change detection methods (a.k.a. drift detectors) to trigger *mitigation mechanisms* if a change in performance is detected.

## Detecting concept drift

Multiple drift detection methods have been proposed. The goal of a drift detector is to signal an alarm in the presence of drift. A good drift detector maximizes the number of true positives while keeping the number of false positives to a minimum. It must also be resource-wise efficient to work in the context of infinite data streams.

For this example, we will generate a synthetic data stream by concatenating 3 distributions of 1000 samples each:

- $dist_a$: $\mu=0.8$, $\sigma=0.05$
- $dist_b$: $\mu=0.4$, $\sigma=0.02$
- $dist_c$: $\mu=0.6$, $\sigma=0.1$.
<!-- #endregion -->

<!-- #region id="PasR1CuXJxNv" -->
## Setup
<!-- #endregion -->

```python id="It-vcW2kIjMr"
!pip install river
!pip install -U numpy
```

```python id="sw24c-Z0JpdE" executionInfo={"status": "ok", "timestamp": 1630502334216, "user_tz": -330, "elapsed": 1422, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from river import drift
```

```python colab={"base_uri": "https://localhost:8080/"} id="anFBvNsfJ3_d" executionInfo={"status": "ok", "timestamp": 1627308690355, "user_tz": -330, "elapsed": 3514, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e5a3ed25-ec3b-4681-ba34-6c5bee6fcfaf"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv -u -t -d
```

<!-- #region id="9N2Uyi-fJzbA" -->
## Synthetic data generation
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"} id="wyb4HmcXIH-H" colab={"base_uri": "https://localhost:8080/", "height": 225} executionInfo={"status": "ok", "timestamp": 1630502344492, "user_tz": -330, "elapsed": 1508, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6fa3a69b-e218-4630-d985-6ce87cfcc613"
# Generate data for 3 distributions
random_state = np.random.RandomState(seed=42)
dist_a = random_state.normal(0.8, 0.05, 1000)
dist_b = random_state.normal(0.4, 0.02, 1000)
dist_c = random_state.normal(0.6, 0.1, 1000)

# Concatenate data to simulate a data stream with 2 drifts
stream = np.concatenate((dist_a, dist_b, dist_c))

# Auxiliary function to plot the data
def plot_data(dist_a, dist_b, dist_c, drifts=None):
    fig = plt.figure(figsize=(7,3), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])
    ax1.grid()
    ax1.plot(stream, label='Stream')
    ax2.grid(axis='y')
    ax2.hist(dist_a, label=r'$dist_a$')
    ax2.hist(dist_b, label=r'$dist_b$')
    ax2.hist(dist_c, label=r'$dist_c$')
    if drifts is not None:
        for drift_detected in drifts:
            ax1.axvline(drift_detected, color='red')
    plt.show()

plot_data(dist_a, dist_b, dist_c)
```

<!-- #region pycharm={"name": "#%% md\n"} id="rgqOMx4-IH-L" -->
### Drift detection test

We will use the ADaptive WINdowing (`ADWIN`) drift detection method. Remember that the goal is to indicate that drift has occurred after samples **1000** and **2000** in the synthetic data stream.
<!-- #endregion -->

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"} id="nx526ymrIH-M" colab={"base_uri": "https://localhost:8080/", "height": 259} executionInfo={"status": "ok", "timestamp": 1630502346064, "user_tz": -330, "elapsed": 1076, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1ecba743-5eea-4d14-d493-c2105aac5088"
drift_detector = drift.ADWIN()
drifts = []

for i, val in enumerate(stream):
    drift_detector.update(val)   # Data is processed one sample at a time
    if drift_detector.change_detected:
        # The drift detector indicates after each sample if there is a drift in the data
        print(f'Change detected at index {i}')
        drifts.append(i)
        drift_detector.reset()   # As a best practice, we reset the detector

plot_data(dist_a, dist_b, dist_c, drifts)
```

<!-- #region pycharm={"name": "#%% md\n"} id="MmIYq220IH-P" -->
We see that `ADWIN` successfully indicates the presence of drift (red vertical lines) close to the begining of a new data distribution.


---
We conclude this example with some remarks regarding concept drift detectors and their usage:

- In practice, drift detectors provide stream learning methods with robustness against concept drift. Drift detectors monitor the model usually through a performance metric.
- Drift detectors work on univariate data. This is why they are used to monitor a model's performance and not the data itself. Remember that concept drift is defined as a change in the relationship between data and the target to learn (in supervised learning).
- Drift detectors define their expectations regarding input data. It is important to know these expectations to feed a given drift detector with the correct data.

<!-- #endregion -->
