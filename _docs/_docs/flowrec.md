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

<!-- #region id="CJ5n0eERj6Cj" -->
# FlowRec
<!-- #endregion -->

```python id="sXjyoPc8Vgs9"
!pip install scikit-multiflow
```

```python colab={"base_uri": "https://localhost:8080/"} id="FkfolI9XV8cL" executionInfo={"status": "ok", "timestamp": 1635251043928, "user_tz": -330, "elapsed": 1749, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eb879d8f-7f71-40ad-da0e-b3620f794265"
!wget -q --show-progress https://github.com/paraschakis/flowrec/raw/master/data/trivago_1M100K.csv
# !wget -q --show-progress https://github.com/paraschakis/flowrec/raw/master/data/yoochoose_clicks_1M100K.csv
# !wget -q --show-progress https://github.com/paraschakis/flowrec/raw/master/data/clef_1M100K.csv
```

```python id="fGDRN9iTyYc9"
# !git clone https://github.com/paraschakis/flowrec.git
# %cd flowrec
# import sys
# sys.path.append('.')
```

```python id="j9EOWUSxVzhc"
from skmultiflow.data import FileStream
```

```python id="ih0NS5qWiTl_"
from evaluation.evaluate_prequential import EvaluatePrequential
from recommendation.random import RandomClassifier
from recommendation.popular import PopularClassifier
from recommendation.co_events import CoEventsClassifier
from recommendation.seq_events import SeqEventsClassifier
from recommendation.ht_wrapper import HTWrapper
from recommendation.beer import BeerEnsemble
from recommendation.sknn import SKNNClassifier
```

```python id="uAwvJLclWAwp"
# Create stream
stream = FileStream("trivago_1M100K.csv")
stream.prepare_for_use()
```

```python colab={"base_uri": "https://localhost:8080/"} id="4hpX4FGIVZnV" executionInfo={"status": "ok", "timestamp": 1635248452439, "user_tz": -330, "elapsed": 326524, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3af09acb-5e5d-42a4-9fcb-6ea309566bcf"
# Instantiate recommenders
random = RandomClassifier()
ht = HTWrapper(weight_mc=5, weight_inv=0.90)
sknn = SKNNClassifier(k=100, sample_size=1000, sample_recent=True,
                      similarity='cosine', sliding_window=True)

popular = PopularClassifier(sliding_window=True)
ar = CoEventsClassifier(sliding_window=False)
sr = SeqEventsClassifier(sliding_window=False)
mc = SeqEventsClassifier(steps_back=1, sliding_window=False)
beer = BeerEnsemble(cf_components=[ar, sr, mc, popular, sknn])

evaluator = EvaluatePrequential(session_column_index=0,
                                time_column_index=1,
                                rec_size=10,
                                allow_reminders=True,
                                allow_repeated=True,
                                show_plot=False,
                                n_wait=100,
                                n_keep=500,
                                n_skip=1000,
                                pretrain_size=0,
                                max_samples=10000,
                                metrics=['recall', 'mrr', 'running_time'])

# Run prequential evaluation
evaluator.evaluate(stream=stream, model=[ar, sr, mc, popular, random, sknn, beer, ht],
                   model_names=['AR', 'SR', 'MC', 'POP', 'RAND', 'S-KNN', 'BEER[TS]', 'HT'])
```
