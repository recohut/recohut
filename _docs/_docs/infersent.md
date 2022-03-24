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

<!-- #region colab_type="text" id="TL2OSAa1bGqO" -->
# InferSent Primer
<!-- #endregion -->

<!-- #region colab_type="text" id="rOhysGNNY-Yw" -->
## Installation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" executionInfo={"elapsed": 106657, "status": "ok", "timestamp": 1586694900825, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="8bDoWRm9D0Xb" outputId="6e604384-fe3e-4036-c22f-077ca0d720da"
# !mkdir GloVe
# !curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
# !unzip GloVe/glove.840B.300d.zip -d GloVe/
!mkdir fastText
!curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
!unzip fastText/crawl-300d-2M.vec.zip -d fastText/
```

```python colab={} colab_type="code" id="1MmzJVUUXM9G"
import os
from os.path import exists, join, basename, splitext

git_repo_url = 'https://github.com/facebookresearch/InferSent.git'
project_name = splitext(basename(git_repo_url))[0]
if not exists(project_name):
  # clone and install dependencies
  !git clone -q {git_repo_url}
  !pip install -q youtube-dl cython gdown
  !pip install -q -U PyYAML
  !apt-get install -y -q libyaml-dev
  !cd {project_name} && python setup.py build develop --user
  
import sys
sys.path.append(project_name)
import time
import matplotlib
import matplotlib.pylab as plt
plt.rcParams["axes.grid"] = False

from IPython.display import YouTubeVideo
```

```python colab={"base_uri": "https://localhost:8080/", "height": 119} colab_type="code" executionInfo={"elapsed": 18352, "status": "ok", "timestamp": 1586694472969, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="Cj3EMfzOW8mO" outputId="227b5174-9ef1-4dfb-bd25-a24ec51baf45"
!mkdir encoder
!curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
!curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
```

```python colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" executionInfo={"elapsed": 1071, "status": "ok", "timestamp": 1586694684429, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="A6T_yMb2V8a2" outputId="2c16793c-680e-4018-81da-bc21517c94cd"
%load_ext autoreload
%autoreload 2

import torch
import numpy as np
from random import randint
```

```python colab={"base_uri": "https://localhost:8080/", "height": 68} colab_type="code" executionInfo={"elapsed": 3219, "status": "ok", "timestamp": 1586695043242, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="mvLi0gHJZOcq" outputId="a5a12888-37ce-47fc-b76e-1f3623677b06"
import nltk
nltk.download('punkt')
```

<!-- #region colab_type="text" id="oskk01P6Wz2a" -->
Download our InferSent models (V1 trained with GloVe, V2 trained with fastText).

Note that infersent1 is trained with GloVe (which have been trained on text preprocessed with the PTB tokenizer) and infersent2 is trained with fastText (which have been trained on text preprocessed with the MOSES tokenizer). The latter also removes the padding of zeros with max-pooling which was inconvenient when embedding sentences outside of their batches.
<!-- #endregion -->

<!-- #region colab_type="text" id="6rVTH5bdY60R" -->
## Load Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 64273, "status": "ok", "timestamp": 1586694906420, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="lofmxTcmWx7l" outputId="ee1a69ac-a3b3-43c6-ec08-baddf400a0e8"
# Load model
from models import InferSent
model_version = 2
MODEL_PATH = "encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))
```

```python colab={} colab_type="code" id="jiYk9uIiXHzA"
# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model
```

```python colab={} colab_type="code" id="njZjqfXuYene"
# Set word vector path for the model
# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 7597, "status": "ok", "timestamp": 1586694917586, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="sj74RpHoYjrG" outputId="7e25c8df-ddd7-4612-ce48-9505a60b23db"
# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)
```

<!-- #region colab_type="text" id="ZKeLFyTKYyUJ" -->
## Load Sentences
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 3264, "status": "ok", "timestamp": 1586695106797, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="DkdFzME2ZdU7" outputId="32df2a01-6aed-4550-8b4d-53dab73dac7d"
!wget https://raw.githubusercontent.com/facebookresearch/InferSent/master/samples.txt
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 1223, "status": "ok", "timestamp": 1586695113381, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="weS7cb5fYu_g" outputId="12a35d42-20ff-487d-a31f-e0351e5562ae"
# Load some sentences
sentences = []
with open('samples.txt') as f:
    for line in f:
        sentences.append(line.strip())
print(len(sentences))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" executionInfo={"elapsed": 6728, "status": "ok", "timestamp": 1586695133452, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="pCzHF-ECZgUj" outputId="fff8bf66-ad59-488a-f625-58931cfc2d21"
sentences[:5]
```

<!-- #region colab_type="text" id="v1B-a46vZnde" -->
## Encode sentences
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 68} colab_type="code" executionInfo={"elapsed": 214017, "status": "ok", "timestamp": 1586695429596, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="EG6eTWV2Zj59" outputId="5d117966-ac39-4082-b0ef-2bd8afe84c58"
embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
print('nb sentences encoded : {0}'.format(len(embeddings)))
```

<!-- #region colab_type="text" id="csBhvj2SaGwS" -->
## Visualization
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 141120, "status": "ok", "timestamp": 1586695429597, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ukMNeE_LZ5mJ" outputId="a539b876-9ead-4c32-8bc1-fec0d9792fb0"
np.linalg.norm(model.encode(['the cat eats.']))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 16701, "status": "ok", "timestamp": 1586695476777, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="0ZXgi6t4aLZT" outputId="c02ea3ee-5ab5-4991-928e-2a0e7ec99fca"
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

cosine(model.encode(['the cat eats.'])[0], model.encode(['the cat drinks.'])[0])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 305} colab_type="code" executionInfo={"elapsed": 2803, "status": "ok", "timestamp": 1586695515860, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="uhMOSUArajp3" outputId="fa3e1664-494f-4c7f-8b5a-162696f8f38f"
idx = randint(0, len(sentences))
_, _ = model.visualize(sentences[idx])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 307} colab_type="code" executionInfo={"elapsed": 4280, "status": "ok", "timestamp": 1586695520114, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="kMorOAqZamjB" outputId="dfcc0efa-6621-429c-d45f-f947e19d4ecf"
my_sent = 'The cat is drinking milk.'
_, _ = model.visualize(my_sent)
```

<!-- #region colab_type="text" id="Y6mbAIACbKIv" -->
## Squad Insersent Q&A
<!-- #endregion -->

<!-- #region colab_type="text" id="5Lco5f7wcWtG" -->
https://github.com/aswalin/SQuAD
<!-- #endregion -->
