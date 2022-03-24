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

<!-- #region id="j_iezICyf5Mw" -->
# Neural Interactive Collaborative Filtering
<!-- #endregion -->

```python id="X1j5UFHN0da9"
!pip install ipdb
```

```python colab={"base_uri": "https://localhost:8080/"} id="vdCkSXqAzkJ6" executionInfo={"elapsed": 822, "status": "ok", "timestamp": 1634836205473, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="9d662b24-fe49-42e8-a155-06c36190dbea"
!git clone https://github.com/guyulongcs/SIGIR2020_NICF.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="Z5RoYOnq0pJV" executionInfo={"elapsed": 646, "status": "ok", "timestamp": 1634836206116, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="f4c028e6-661d-4856-aebf-71ae42bada87"
%cd SIGIR2020_NICF/NICF_code
```

```python colab={"base_uri": "https://localhost:8080/"} id="Z3wu_uqMv6vT" executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1634836449818, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="27d41e80-5a65-430d-e6ea-0caa7fad5ee3"
!head data/data/env.dat
```

```python colab={"base_uri": "https://localhost:8080/"} id="USDF36ZN0mHu" executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1634836206117, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="7dfb109a-b77e-417d-e664-96bf4cd61332"
%tensorflow_version 1.x
```

```python colab={"base_uri": "https://localhost:8080/"} id="qOyGG3QizpVa" executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1634836349342, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="731c140d-6da1-4a1d-f07f-616677311957"
%%writefile run.sh

python ./launch.py -data_path ./data/data/ -environment env -T 40 -ST [5,10,20,40] -agent Train -FA FA -latent_factor 50 \
-learning_rate 0.001 -training_epoch 10 -seed 145 -gpu_no 0 -inner_epoch 50 -rnn_layer 2 -gamma 0.8 -batch 50 -restore_model False
```

```python colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="OPJKaTAVzpRz" executionInfo={"elapsed": 74579, "status": "ok", "timestamp": 1634836424365, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} outputId="0d4bc991-21c0-4781-96ff-a1a5223c0f49"
#collapse-hide
!sh run.sh
```
