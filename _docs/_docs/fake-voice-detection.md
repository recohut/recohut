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

<!-- #region id="SqVfP_FPlQMv" -->
# Fake Voice Detection
<!-- #endregion -->

<!-- #region id="LV2Y4ySWPs4_" -->
## Temporal Convolutional Model
<!-- #endregion -->

<!-- #region id="ou14LVTFDBrJ" -->
### Environment Setup Process
<!-- #endregion -->

```python id="ge7gt1_tn7gJ" cellView="both"
# Installation
!git clone https://github.com/dessa-oss/fake-voice-detection.git
%cd /content/fake-voice-detection/
!pip install -r '/content/fake-voice-detection/code/requirements.txt'
```

```python id="S2R-uHraaLj1"
# Load data
!wget https://asv-audio-data-atlas.s3.amazonaws.com/realtalk.zip
!unzip realtalk.zip
```

```python id="spn3-qpa3EHX" outputId="4c5b02e3-8402-4027-d23e-2a0cd8e8db58" executionInfo={"status": "ok", "timestamp": 1586872444811, "user_tz": -330, "elapsed": 116809, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}} colab={"base_uri": "https://localhost:8080/", "height": 75}
# Checking files
from glob import glob
fake_audio = glob('realtalk/fake/*.wav')
real_audio = glob('realtalk/real/*.wav')

import IPython
IPython.display.Audio(fake_audio[0])
```

```python id="7FqKtpBO4exE"
# Load pretrained model
%cd /content/fake-voice-detection/code
!mkdir fitted_objects
%cd fitted_objects
!wget https://asv-audio-data-atlas.s3.amazonaws.com/saved_model_240_8_32_0.05_1_50_0_0.0001_100_156_2_True_True_fitted_objects.h5
```

```python id="zH6LXoLaczjJ"
# # Loading testing audio file
%cd /content/fake-voice-detection/data
!mkdir inference_data
%cd inference_data
!mkdir unlabeled
!cp /content/fake-voice-detection/realtalk/fake/JREa631-0030.wav /content/fake-voice-detection/data/inference_data/unlabeled
!cp /content/realtalk/fake/JREa631-0030.wav /content/fake-voice-detection/data/inference_data/unlabeled
%cd /content/fake-voice-detection/code/
```

```python id="8xGsQQ6WDP_e"
import os
import numpy as np
import subprocess
import sys
sys.path.append('/content/fake-voice-detection/code')
from sklearn.metrics import f1_score, accuracy_score
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

```python id="Xj5YIt1MdOkZ"
def detect_deepfake(file_path='/content/fake-voice-detection/data/inference_data', mode='unlabeled'):
  data_dir = file_path
  mode = mode  # real, fake, or unlabeled
  pretrained_model_name = 'saved_model_240_8_32_0.05_1_50_0_0.0001_100_156_2_True_True_fitted_objects.h5'
  print(f"Loading inference data from {os.path.join(data_dir,mode)}")
  print(f"Loading pretrained model {pretrained_model_name}")

  # preprocess the files
  processed_data = preprocess_from_ray_parallel_inference(data_dir, mode, use_parallel=True)
  processed_data = sorted(processed_data, key = lambda x: len(x[0]))

  # Load trained model
  discriminator = Discriminator_Model(load_pretrained=True, saved_model_name=pretrained_model_name, real_test_mode=False)

  # Do inference
  if mode == 'unlabeled':
      # Visualize the preprocessed data
      plot_spectrogram(processed_data[0], path=os.path.join(file_path,'visualize_inference_spectrogram.png'))

      real_proba = discriminator.predict_labels(processed_data, 
                                                raw_prob=True, 
                                                batch_size=20)[0][0]
  else:
      features = [x[0] for x in processed_data]
      labels = [x[1] for x in processed_data]
      preds = discriminator.predict_labels(features, threshold=0.5, batch_size=1)
      print(f"Accuracy on data set: {accuracy_score(labels, preds)}")

      all_filenames = sorted(os.listdir(os.path.join(data_dir, mode)))

      if mode == 'real':
          # True Positive Examples
          ind_tp = np.equal((preds + labels).astype(int), 2)
          ind_tp = np.argwhere(ind_tp == True).reshape(-1, )
          tp_filenames = [all_filenames[i] for i in ind_tp]
          print(f'correctly predicted filenames: {tp_filenames}')

          # False Negative Examples
          ind_fn = np.greater(labels, preds)
          ind_fn = np.argwhere(ind_fn == True).reshape(-1, )
          fn_filenames = [all_filenames[i] for i in ind_fn]
          print(f'real clips classified as fake: {fn_filenames}')
      elif mode == 'fake':
          # True Negative Examples
          ind_tn = np.equal((preds + labels).astype(int), 0)
          ind_tn = np.argwhere(ind_tn == True).reshape(-1, )
          tn_filenames = [all_filenames[i] for i in ind_tn]
          print(f'correctly predicted filenames: {tn_filenames}')

          # False Positive Examples
          ind_fp = np.greater(preds, labels)
          ind_fp = np.argwhere(ind_fp == True).reshape(-1, )
          fp_filenames = [all_filenames[i] for i in ind_fp]
          print(f'fake clips classified as real: {fp_filenames}')

  return real_proba
```

<!-- #region id="006rraojFgc0" -->
### Detect DeepFake
<!-- #endregion -->

```python id="cIk8G9OpFcjl" outputId="9c4d5ee2-a82b-4305-812e-786d41200e5c" executionInfo={"status": "ok", "timestamp": 1586874598570, "user_tz": -330, "elapsed": 28968, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}} colab={"base_uri": "https://localhost:8080/", "height": 105}
# Step 1: Set the location where your files are stored
my_file_path = '/content/fake-voice-detection/data/inference_data/'

# Step 2: Run the model
real_probability = detect_deepfake(my_file_path)
```

```python id="k4XKd-kKFykm" outputId="1a365dbe-524d-4d91-a3a5-7d93e3daad9c" executionInfo={"status": "ok", "timestamp": 1586874600285, "user_tz": -330, "elapsed": 707, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
# Step 3: Get the result
print("The probability of the clip being real is: {:.2%}".format(real_probability))
```

<!-- #region id="1TfRlSTFlkHc" -->
## GMM-UGB and CVAE model
<!-- #endregion -->

```python id="5AD5At6ztury" outputId="53e3c6aa-34ee-4eb1-817c-7b77bceb159a" executionInfo={"status": "ok", "timestamp": 1586275387858, "user_tz": -330, "elapsed": 4017, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
# Installation
%tensorflow_version 1.x
!git clone https://github.com/kstoneriv3/Fake-Voice-Detection.git
!pip install --user -r requirements.txt
```

```python id="9uPv2e9QXi34" outputId="f2a40fc8-5ffc-4084-89c6-a18b733dd57e" executionInfo={"status": "ok", "timestamp": 1586275175907, "user_tz": -330, "elapsed": 812, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
%cd Fake-Voice-Detection/
```

```python id="0nn4Pni-s7C9" outputId="c9f8a519-0db9-4f31-b2a3-fe52aba1f1ce" executionInfo={"status": "ok", "timestamp": 1586275311496, "user_tz": -330, "elapsed": 132064, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}} colab={"base_uri": "https://localhost:8080/", "height": 136}
# Download and unzip datasets and pretrained models
!python ./src/download.py
```

```python id="IU3vUYX_UEXb" outputId="f2c6e166-c116-4e0b-c76d-fe406157cc60" executionInfo={"status": "ok", "timestamp": 1586275353672, "user_tz": -330, "elapsed": 17738, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}} colab={"base_uri": "https://localhost:8080/", "height": 51}
# Split the raw speech
!python ./src/split_normalize_raw_speech.py
```

```python id="BJUS46BedgEM"
!zip /content/Fake-Voice-Detection/data
```

```python id="0zEwhV2RVD2G"
# CycleGAN Voice Conversion

# Train the Voice Conversion Model
# !python ./src/conversion/train.py --model_dir='./model/conversion/pretrained'

# Convert the source speaker's voice
!python ./src/conversion/convert.py --model_dir='./model/conversion/pretrained'
```

```python id="KhPbGOGjVJRI"
# GMM-UBG verification

# Train the GMM based verification system and Plot the scores
!python ./src/verification_gmm/train_and_plot.py
```

```python id="JzF0S4j6VVk-"
# compute AUC converted samples of every 50 epoch
!python ./src/verification_gmm/compute_auc.py
```

```python id="xhm9woF9WIXd"
# Convolutional VAE

# Train the CVAE based verification system and Plot the scores
!python ./src/verification_cvae/cvae_verification.py
```

<!-- #region id="FtG-ohLXlwqi" -->
## Audio encoding based detection
<!-- #endregion -->

```python id="tgfjinJbeUOI"
# Installation
!pip install resemblyzer
!git clone https://github.com/resemble-ai/Resemblyzer.git
import sys
%cd Resemblyzer
```

```python id="b-mxRD6vj8lV" outputId="1fdacb60-fd1f-4294-9078-42322b13c23e" executionInfo={"status": "ok", "timestamp": 1586351841138, "user_tz": -330, "elapsed": 124443, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}} colab={"base_uri": "https://localhost:8080/", "height": 544}
# Similarity Matching
from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
from tqdm import tqdm
import numpy as np


# Load data
!pip install -q youtube-dl
YOUTUBE_ID = ['cQ54GDm1eL0']
for index,uid in enumerate(YOUTUBE_ID):
  !youtube-dl --extract-audio --audio-format wav --output "downloaded.%(ext)s" https://www.youtube.com/watch\?v\={uid}
  !ffmpeg -loglevel panic -y -i downloaded.wav -acodec pcm_s16le -ac 1 -ar 16000 obamaFake_{index}.wav

## Load and preprocess the audio
data_dir = Path("audio_data", "donald_trump")
wav_fpaths = list(data_dir.glob("**/*.mp3"))
wavs = [preprocess_wav(wav_fpath) for wav_fpath in \
        tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit=" utterances")]

## Compute the embeddings
encoder = VoiceEncoder()
embeds = np.array([encoder.embed_utterance(wav) for wav in wavs])
speakers = np.array([fpath.parent.name for fpath in wav_fpaths])
names = np.array([fpath.stem for fpath in wav_fpaths])

# Take 6 real embeddings at random, and leave the 6 others for testing
gt_indices = np.random.choice(*np.where(speakers == "real"), 6, replace=False) 
mask = np.zeros(len(embeds), dtype=np.bool)
mask[gt_indices] = True
gt_embeds = embeds[mask]
gt_names = names[mask]
gt_speakers = speakers[mask]
embeds, speakers, names = embeds[~mask], speakers[~mask], names[~mask]

## Compare all embeddings against the ground truth embeddings, and compute the average similarities.
scores = (gt_embeds @ embeds.T).mean(axis=0)

# Order the scores by decreasing order
sort = np.argsort(scores)[::-1]
scores, names, speakers = scores[sort], names[sort], speakers[sort]

## Plot the scores
fig, _ = plt.subplots(figsize=(6, 6))
indices = np.arange(len(scores))
plt.axhline(0.84, ls="dashed", label="Prediction threshold", c="black")
plt.bar(indices[speakers == "real"], scores[speakers == "real"], color="green", label="Real")
plt.bar(indices[speakers == "fake"], scores[speakers == "fake"], color="red", label="Fake")
plt.legend()
plt.xticks(indices, names, rotation="vertical", fontsize=8)
plt.xlabel("Youtube video IDs")
plt.ylim(0.7, 1)
plt.ylabel("Similarity to ground truth")
fig.subplots_adjust(bottom=0.25)
plt.show()
```

```python id="n2-cb_GywbZc" outputId="b14f0b2a-db67-4258-c1ea-2e89078dfba8" executionInfo={"status": "ok", "timestamp": 1586351972675, "user_tz": -330, "elapsed": 255965, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}} colab={"base_uri": "https://localhost:8080/", "height": 411}
# Embedding Visualization
from sklearn.linear_model import LogisticRegression 
from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
from tqdm import tqdm
import numpy as np

## Gather a single utterance per speaker
data_dir = Path("audio_data", "librispeech_train-clean-100")
wav_fpaths = list(data_dir.glob("*.flac"))
speakers = [fpath.stem.split("-")[0] for fpath in wav_fpaths]
wavs = [preprocess_wav(wav_fpath) for wav_fpath in \
        tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit=" utterances")]

# Get the sex of each speaker from the metadata file
with data_dir.joinpath("SPEAKERS.TXT").open("r") as f:
    sexes = dict(l.replace(" ", "").split("|")[:2] for l in f if not l.startswith(";"))
markers = ["x" if sexes[speaker] == "M" else "o" for speaker in speakers]
colors = ["black"] * len(speakers)


## Compute the embeddings
encoder = VoiceEncoder()
utterance_embeds = np.array(list(map(encoder.embed_utterance, wavs)))


## Project the embeddings in 2D space. 
_, ax = plt.subplots(figsize=(6, 6))
# Passing min_dist=1 to UMAP will make it so the projections don't necessarily need to fit in 
# clusters, so that you can have a better idea of what the manifold really looks like. 
projs = plot_projections(utterance_embeds, speakers, ax, colors, markers, False,
                         min_dist=1)
ax.set_title("Embeddings for %d speakers" % (len(speakers)))
ax.scatter([], [], marker="x", c="black", label="Male speaker")
ax.scatter([], [], marker="o", c="black", label="Female speaker")

# Separate the data by the sex
classifier = LogisticRegression(solver="lbfgs")
classifier.fit(projs, markers)
x = np.linspace(*ax.get_xlim(), num=200)
y = -(classifier.coef_[0, 0] * x + classifier.intercept_) / classifier.coef_[0, 1]
mask = (y > ax.get_ylim()[0]) & (y < ax.get_ylim()[1])
ax.plot(x[mask], y[mask], label="Decision boundary")

ax.legend()
plt.show()
```

```python id="mOLDz0sWySc1" outputId="368429a0-d6b3-489a-8094-b64e3caea8f1" executionInfo={"status": "ok", "timestamp": 1586351973524, "user_tz": -330, "elapsed": 256801, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}} colab={"base_uri": "https://localhost:8080/", "height": 408}
# Print Embeddings
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

fpath = Path("/content/Resemblyzer/obamaFake_0.wav")
wav = preprocess_wav(fpath)

encoder = VoiceEncoder()
embed = encoder.embed_utterance(wav)
np.set_printoptions(precision=3, suppress=True)
print(embed)
```

```python id="nBMFIZWezGWF" outputId="bdcaf642-2d7d-4c5d-c3a9-ccfcef1367bb" executionInfo={"status": "ok", "timestamp": 1586351988701, "user_tz": -330, "elapsed": 271965, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}} colab={"base_uri": "https://localhost:8080/", "height": 187}
# Comparing Voice Similarity

# Load data - Obama and Modi voice
!pip install -q youtube-dl
YOUTUBE_ID = ['y7hddyiR47k','avzJdPkRX_4']
!youtube-dl --extract-audio --audio-format wav --output "downloaded.%(ext)s" https://www.youtube.com/watch\?v\={YOUTUBE_ID[0]}
!ffmpeg -loglevel panic -y -i downloaded.wav -acodec pcm_s16le -ac 1 -ar 16000 obama.wav
!youtube-dl --extract-audio --audio-format wav --output "downloaded.%(ext)s" https://www.youtube.com/watch\?v\={YOUTUBE_ID[1]}
!ffmpeg -loglevel panic -y -i downloaded.wav -acodec pcm_s16le -ac 1 -ar 16000 modi.wav
```

```python id="dnzfETap0GcM" outputId="98c75dfe-73a6-402f-d8a3-1d7b960ad65b" executionInfo={"status": "ok", "timestamp": 1586351988702, "user_tz": -330, "elapsed": 271954, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}} colab={"base_uri": "https://localhost:8080/", "height": 34}
from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

encoder = VoiceEncoder()
```

```python id="8eBel02N0X0R" outputId="f536105f-2b6b-40b5-e614-e3eb9ea5304b" executionInfo={"status": "ok", "timestamp": 1586351994120, "user_tz": -330, "elapsed": 277338, "user": {"displayName": "sparsh agarwal", "photoUrl": "", "userId": "00322518567794762549"}} colab={"base_uri": "https://localhost:8080/", "height": 51}
fpath = Path('/content/Resemblyzer/obama.wav')
wav1 = preprocess_wav(fpath)
fpath = Path('/content/Resemblyzer/modi.wav')
wav2 = preprocess_wav(fpath)

encoder = VoiceEncoder()
embeds_a = encoder.embed_utterance(wav1)
embeds_b = encoder.embed_utterance(wav2)
print("Shape of embeddings: %s" % str(embeds_a.shape))

utt_sim_matrix = np.inner(embeds_a, embeds_b)
```
