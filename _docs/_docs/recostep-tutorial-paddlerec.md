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

```python id="0DPubQi1z8WL"
# !pip install -q paddlepaddle
# !git clone https://github.com/PaddlePaddle/PaddleRec/
# !pip install -U redis pyyaml grpcio-tools
# !pip install -U https://paddle-serving.bj.bcebos.com/whl/paddle_serving_server-0.0.0-py2-none-any.whl https://paddle-serving.bj.bcebos.com/whl/paddle_serving_client-0.0.0-cp27-none-any.whl https://paddle-serving.bj.bcebos.com/whl/paddle_serving_app-0.0.0-py2-none-any.whl
```

```python colab={"base_uri": "https://localhost:8080/"} id="_DXAdDfgqVkY" executionInfo={"status": "ok", "timestamp": 1623005425723, "user_tz": -330, "elapsed": 611, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="18c3a018-5247-402e-9ec3-41985ff76254"
%cd PaddleRec
```

<!-- #region id="VfNl2Aw7o90z" -->
### Quick start
We take the dnn algorithm as an example to get start of PaddleRec, and we take 100 pieces of training data from Criteo Dataset
<!-- #endregion -->

```python id="fukEpj-1ozTV"
!python -u tools/trainer.py -m models/rank/dnn/config.yaml # Training with dygraph model
```

```python id="0AQvvBcEpHU6"
!python -u tools/static_trainer.py -m models/rank/dnn/config.yaml #  Training with static model
```

<!-- #region id="tudaXMwJqZ2y" -->
### redis/milvus service started
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="G67wSgzdpQCc" executionInfo={"status": "ok", "timestamp": 1623005861680, "user_tz": -330, "elapsed": 403013, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f7faf407-84eb-4dac-80d7-3f392f764acc"
!wget http://download.redis.io/releases/redis-stable.tar.gz --no-check-certificate
!tar -xf redis-stable.tar.gz && cd redis-stable/src && make && ./redis-server &
```

```python id="kWoP2BTZqe1e"

```
