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

<!-- #region id="YABk0Xkgz7lU" -->
# Black-box Attack API
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="z_vBaj_fSeVe" executionInfo={"status": "ok", "timestamp": 1632132416776, "user_tz": -330, "elapsed": 2264, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fe0366db-99ad-4880-eebf-53af82f1f21e"
!git clone https://github.com/Yueeeeeeee/RecSys-Extraction-Attack.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="d4Kf7L9-Sq6N" executionInfo={"status": "ok", "timestamp": 1632138125205, "user_tz": -330, "elapsed": 686, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="168c38f5-560a-48d9-aede-8f3f010bcfc0"
%cd RecSys-Extraction-Attack/
```

```python id="JwWXvit6SiBW"
!apt-get install libarchive-dev
!pip install faiss-cpu --no-cache
!apt-get install libomp-dev
!pip install wget
!pip install libarchive
```

```python id="HKqZtdWQToqy"
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)
```

<!-- #region id="BuFwoWToSqCM" -->
## Black-Box Model Training
<!-- #endregion -->

<!-- #region id="MpMd1Kp4XEeQ" -->
**NARM model trained on ML-1M dataset.**
<!-- #endregion -->

<!-- #region id="1G4X86lVWnNx" -->
Given a user sequence 𝒙 with length 𝑇 , we use $𝒙_{[:𝑇−2]}$ as training data and use the last two items for validation and testing respectively. We use hyper-parameters from grid-search. Additionally, all models are trained using Adam optimizer with weight decay 0.01, learning rate 0.001, batch size 128 and 100 linear warmup steps, allowed sequence length as 200.
<!-- #endregion -->

<!-- #region id="iQWpnX7wXhTz" -->
We accelerate evaluation by uniformly sampling 100 negative items for each user. Then we rank them with the positive item and report the average performance on these 101 testing items. Our Evaluation focuses on two aspects:
- Ranking Performance: We to use truncated Recall@K that is equivalent to Hit Rate (HR@K) in our evaluation, and Normalized Discounted Cumulative Gain (NDCG@K) to measure the ranking quality.
- Agreement Measure: We define Agreement@K (Agr@K) to evaluate the output similarity between the black-box model and our extracted white-box model.
<!-- #endregion -->

<!-- #region id="_AdxgqLUYAAe" -->
Official results:
<!-- #endregion -->

<!-- #region id="FLqr32cH0GNL" -->
<img src='_images/T697871_1.png'>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9rk0JMgMTCkO" executionInfo={"status": "ok", "timestamp": 1631989596151, "user_tz": -330, "elapsed": 14037225, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f71e6f5d-3259-45b0-9a4a-e98e00ab9fad"
!python train.py
```

```python id="jWvx3H9FKcMm"
# !zip -r bb_model_narm_ml1m.zip ./experiments
# !cp bb_model_narm_ml1m.zip /content/drive/MyDrive/TempData
# !ls /content/drive/MyDrive/TempData
```

<!-- #region id="zk5t6lYBKVUB" -->
## White-Box Model Distillation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="F0FUWkMyrOFC" executionInfo={"status": "ok", "timestamp": 1632132568336, "user_tz": -330, "elapsed": 2484, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e36d57ee-2741-48c5-eed0-44d1fafbb4fc"
!cp /content/drive/MyDrive/TempData/bb_model_narm_ml1m.zip .
!unzip bb_model_narm_ml1m.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="1NQyI0sHroLU" outputId="cbdeb93c-8e11-4454-dad4-70556ca8f427"
!python distill.py
```

```python id="_1-tTQXUrx58"
!zip -r wb_model_narm_ml1m.zip ./experiments
!cp wb_model_narm_ml1m.zip /content/drive/MyDrive/TempData
!ls /content/drive/MyDrive/TempData
```

<!-- #region id="MH7EcEarA2-K" -->
## Attack
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="YGQ1SzDnA86M" executionInfo={"status": "ok", "timestamp": 1632141032559, "user_tz": -330, "elapsed": 2615663, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ad1b361e-ed45-495c-9b37-71c32a53c2cf"
!python attack.py
```

```python id="MZvUX-jgBQ6P"
!zip -r wb_model_narm_ml1m.zip ./experiments
!cp wb_model_narm_ml1m.zip /content/drive/MyDrive/TempData
!ls /content/drive/MyDrive/TempData
```

<!-- #region id="4QH7REySMW4B" -->
## Retrain
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="8T_KFLL6MZUc" outputId="64e3fde0-b372-4ba5-9b94-3ddc9b0fdb61"
!python retrain.py
```
