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

```python id="2f33VIJdAxBC" colab_type="code" cellView="form" outputId="76968d4f-dd9f-4626-c2cd-2c752ed6873f" executionInfo={"status": "ok", "timestamp": 1586202211008, "user_tz": -330, "elapsed": 71505, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} colab={"base_uri": "https://localhost:8080/", "height": 1000}
#@title Install DeepSpeech
import os
from os.path import exists, join, basename, splitext

if not exists('deepspeech-0.6.1-models'):
  !apt-get install -qq sox
  !pip install -q deepspeech-gpu==0.6.1 youtube-dl
  !wget https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/deepspeech-0.6.1-models.tar.gz
  !tar xvfz deepspeech-0.6.1-models.tar.gz
  
from IPython.display import YouTubeVideo
```

<!-- #region id="afTz2NXpBL10" colab_type="text" -->
## Transcribe YouTube Video
We are going to make speech recognition on the following youtube video:
<!-- #endregion -->

```python id="l3DzEpg3A7w1" colab_type="code" outputId="b0dc10a2-664b-49f2-83a8-7b3fb1d86ffd" executionInfo={"status": "ok", "timestamp": 1586202298037, "user_tz": -330, "elapsed": 1224, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} colab={"base_uri": "https://localhost:8080/", "height": 321}
YOUTUBE_ID = 'DpXxdSr1FWs'


YouTubeVideo(YOUTUBE_ID)
```

<!-- #region id="1rzDIKtxBo0F" colab_type="text" -->
Download the above video, convert to a WAV file and do speech recognition:
<!-- #endregion -->

```python id="undj2pG2BkGQ" colab_type="code" outputId="2ae2f0f5-aab8-4d06-b283-76611a8dde23" executionInfo={"status": "ok", "timestamp": 1586202589448, "user_tz": -330, "elapsed": 223492, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} colab={"base_uri": "https://localhost:8080/", "height": 360}
!rm -rf *.wav
!youtube-dl --extract-audio --audio-format wav --output "test.%(ext)s" https://www.youtube.com/watch\?v\={YOUTUBE_ID}
!deepspeech --model deepspeech-0.6.1-models/output_graph.pbmm --lm deepspeech-0.6.1-models/lm.binary --trie deepspeech-0.6.1-models/trie --audio test.wav
```

```python id="jhUrDKODB0-6" colab_type="code" colab={}

```
