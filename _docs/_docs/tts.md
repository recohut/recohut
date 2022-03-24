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

# Text to Speech
> Generating synthetic voice from the given text using open-source TTS models.

- toc: false
- badges: true
- comments: true
- categories: [tts, audio]

```python colab={} colab_type="code" id="EXiuKtGYe1wl"
TEXT = "hello homosapien, I am a synthetic voice created by Sparsh"
```

<!-- #region colab_type="text" id="JtT8cQjmeTqW" -->
## Tacotron2 + WaveGlow
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" executionInfo={"elapsed": 22315, "status": "ok", "timestamp": 1587669507724, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="XQdE5JIUk3r5" outputId="59a2a454-24a9-4e8f-990a-07a5dabfc473"
%tensorflow_version 1.x
import os
from os.path import exists, join, basename, splitext

git_repo_url = 'https://github.com/NVIDIA/tacotron2.git'
project_name = splitext(basename(git_repo_url))[0]
if not exists(project_name):
  # clone and install
  !git clone -q --recursive {git_repo_url}
  !cd {project_name}/waveglow && git checkout 9168aea
  !pip install -q librosa unidecode
  
import sys
sys.path.append(join(project_name, 'waveglow/'))
sys.path.append(project_name)
import time
import matplotlib
import matplotlib.pylab as plt
plt.rcParams["axes.grid"] = False
```

```python colab={"base_uri": "https://localhost:8080/", "height": 221} colab_type="code" executionInfo={"elapsed": 62285, "status": "ok", "timestamp": 1587669547701, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="jXLnnH9zeZb6" outputId="d2d86d74-1c8e-407f-e0db-83cda780a259"
def download_from_google_drive(file_id, file_name):
  # download a file from the Google Drive link
  !rm -f ./cookie
  !curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id={file_id}" > /dev/null
  confirm_text = !awk '/download/ {print $NF}' ./cookie
  confirm_text = confirm_text[0]
  !curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm={confirm_text}&id={file_id}" -o {file_name}

tacotron2_pretrained_model = 'tacotron2_statedict.pt'
if not exists(tacotron2_pretrained_model):
  # download the Tacotron2 pretrained model
  download_from_google_drive('1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA', tacotron2_pretrained_model)
waveglow_pretrained_model = 'waveglow_old.pt'
if not exists(waveglow_pretrained_model):
  # download the Waveglow pretrained model  
  download_from_google_drive('1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx', waveglow_pretrained_model)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} colab_type="code" executionInfo={"elapsed": 90000, "status": "ok", "timestamp": 1587669576163, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="sGzgsTsgea06" outputId="0785c82d-9e34-4f5f-c828-cabc72379477"
import IPython.display as ipd
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
from denoiser import Denoiser

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none', cmap='viridis')

torch.set_grad_enabled(False)
        
# initialize Tacotron2 with the pretrained model
hparams = create_hparams()
hparams.sampling_rate = 22050
model = Tacotron2(hparams)
model.load_state_dict(torch.load(tacotron2_pretrained_model)['state_dict'])
_ = model.cuda().eval()#.half()

# initialize Waveglow with the pretrained model
# waveglow = torch.load(waveglow_pretrained_model)['model']
# WORKAROUND for: https://github.com/NVIDIA/tacotron2/issues/182
import json
from glow import WaveGlow
waveglow_config = json.load(open('%s/waveglow/config.json' % project_name))['waveglow_config']
waveglow = WaveGlow(**waveglow_config)
waveglow.load_state_dict(torch.load(waveglow_pretrained_model)['model'].state_dict())
_ = waveglow.cuda().eval()#.half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} colab_type="code" executionInfo={"elapsed": 4560, "status": "ok", "timestamp": 1587669602232, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="okc_vkk0ecoS" outputId="c4669ea0-4ac8-4d4f-dde7-5533c33aafe0"
sequence = np.array(text_to_sequence(TEXT, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
sequence = sequence.cuda()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
plot_data((mel_outputs.data.cpu().numpy()[0],
           mel_outputs_postnet.data.cpu().numpy()[0],
           alignments.data.cpu().numpy()[0].T))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 75} colab_type="code" executionInfo={"elapsed": 3072, "status": "ok", "timestamp": 1587669606497, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="EWCfCJ1Ne32W" outputId="e8f9bfa4-4f8a-4f48-bb6c-da5157bd64e9"
audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 75} colab_type="code" executionInfo={"elapsed": 3006, "status": "ok", "timestamp": 1587669612413, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ZmuZsnqFe5Qe" outputId="00151118-d719-4328-971e-65b9fdafc623"
# remove waveglow bias
audio_denoised = denoiser(audio, strength=0.01)[:, 0]
ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)
```

<!-- #region colab_type="text" id="I3VZTK8Wh_Xx" -->
## ESPnet real time E2E-TTS demonstration

This notebook provides a demonstration of the realtime E2E-TTS using ESPnet-TTS and ParallelWaveGAN (+ MelGAN).

- ESPnet: https://github.com/espnet/espnet
- ParallelWaveGAN: https://github.com/kan-bayashi/ParallelWaveGAN
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 187} colab_type="code" executionInfo={"elapsed": 31114, "status": "ok", "timestamp": 1587670986808, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="0cUJji_HhT4b" outputId="6c527994-93e3-4877-ad88-17cb5ef1b998"
# install minimal components
!pip install -q parallel_wavegan PyYaml unidecode ConfigArgparse g2p_en nltk
!git clone -q https://github.com/espnet/espnet.git
!cd espnet && git fetch && git checkout -b v.0.6.1 1e8b6ce88d57b53d1b60cbb3f306652468b0ab63
```

```python colab={"base_uri": "https://localhost:8080/", "height": 561} colab_type="code" executionInfo={"elapsed": 42135, "status": "ok", "timestamp": 1587670998066, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ztWZjy_XOPZR" outputId="4a81572d-0455-4e41-a489-c1eb78774fad"
# download pretrained model
import os
if not os.path.exists("downloads/en/transformer"):
    !./espnet/utils/download_from_google_drive.sh \
        https://drive.google.com/open?id=1z8KSOWVBjK-_Ws4RxVN4NTx-Buy03-7c downloads/en/transformer tar.gz

# set path
trans_type = "phn"
dict_path = "downloads/en/transformer/data/lang_1phn/phn_train_no_dev_units.txt"
model_path = "downloads/en/transformer/exp/phn_train_no_dev_pytorch_train_pytorch_transformer.v3.single/results/model.last1.avg.best"

print("sucessfully finished download.")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 428} colab_type="code" executionInfo={"elapsed": 47258, "status": "ok", "timestamp": 1587671003450, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="nQDFNuQ2dK-M" outputId="d7085533-24ea-4cc9-f499-efebab6414bd"
# download pretrained model
import os
if not os.path.exists("downloads/en/parallel_wavegan"):
    !./espnet/utils/download_from_google_drive.sh \
        https://drive.google.com/open?id=1Grn7X9wD35UcDJ5F7chwdTqTa4U7DeVB downloads/en/parallel_wavegan tar.gz

# set path
vocoder_path = "downloads/en/parallel_wavegan/ljspeech.parallel_wavegan.v2/checkpoint-400000steps.pkl"
vocoder_conf = "downloads/en/parallel_wavegan/ljspeech.parallel_wavegan.v2/config.yml"

print("Sucessfully finished download.")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 187} colab_type="code" executionInfo={"elapsed": 61212, "status": "ok", "timestamp": 1587671017940, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="i8JXOfRfqMFN" outputId="115517a8-d1fc-4dab-a3fe-e5950e1af87b"
# add path
import sys
sys.path.append("espnet/egs/ljspeech/tts1/local")
sys.path.append("espnet")

# define device
import torch
device = torch.device("cuda")

# define E2E-TTS model
from argparse import Namespace
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.utils.dynamic_import import dynamic_import
idim, odim, train_args = get_model_conf(model_path)
model_class = dynamic_import(train_args.model_module)
model = model_class(idim, odim, train_args)
torch_load(model_path, model)
model = model.eval().to(device)
inference_args = Namespace(**{
    "threshold": 0.5,"minlenratio": 0.0, "maxlenratio": 10.0,
    "use_attention_constraint": True,
    "backward_window": 1, "forward_window":3,
    })

# define neural vocoder
import yaml
import parallel_wavegan.models
with open(vocoder_conf) as f:
    config = yaml.load(f, Loader=yaml.Loader)
vocoder_class = config.get("generator_type", "ParallelWaveGANGenerator")
vocoder = getattr(parallel_wavegan.models, vocoder_class)(**config["generator_params"])
vocoder.load_state_dict(torch.load(vocoder_path, map_location="cpu")["model"]["generator"])
vocoder.remove_weight_norm()
vocoder = vocoder.eval().to(device)

# define text frontend
from text.cleaners import custom_english_cleaners
from g2p_en import G2p
with open(dict_path) as f:
    lines = f.readlines()
lines = [line.replace("\n", "").split(" ") for line in lines]
char_to_id = {c: int(i) for c, i in lines}
g2p = G2p()
def frontend(text):
    """Clean text and then convert to id sequence."""
    text = custom_english_cleaners(text)
    
    if trans_type == "phn":
        text = filter(lambda s: s != " ", g2p(text))
        text = " ".join(text)
        print(f"Cleaned text: {text}")
        charseq = text.split(" ")
    else:
        print(f"Cleaned text: {text}")
        charseq = list(text)
    idseq = []
    for c in charseq:
        if c.isspace():
            idseq += [char_to_id["<space>"]]
        elif c not in char_to_id.keys():
            idseq += [char_to_id["<unk>"]]
        else:
            idseq += [char_to_id[c]]
    idseq += [idim - 1]  # <eos>
    return torch.LongTensor(idseq).view(-1).to(device)

import nltk
nltk.download('punkt')
print("Now ready to synthesize!")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 143} colab_type="code" executionInfo={"elapsed": 103982, "status": "ok", "timestamp": 1587671061890, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="9gGRzrjyudWF" outputId="5b54332c-e25b-4923-af7d-d44c8074dbf9"
import time
print("Input your favorite sentence in English!")
input_text = input()
pad_fn = torch.nn.ReplicationPad1d(
    config["generator_params"].get("aux_context_window", 0))
use_noise_input = vocoder_class == "ParallelWaveGANGenerator"
with torch.no_grad():
    start = time.time()
    x = frontend(input_text)
    c, _, _ = model.inference(x, inference_args)
    c = pad_fn(c.unsqueeze(0).transpose(2, 1)).to(device)
    xx = (c,)
    if use_noise_input:
        z_size = (1, 1, (c.size(2) - sum(pad_fn.padding)) * config["hop_size"])
        z = torch.randn(z_size).to(device)
        xx = (z,) + xx
    y = vocoder(*xx).view(-1)
rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
print(f"RTF = {rtf:5f}")

from IPython.display import display, Audio
display(Audio(y.view(-1).cpu().numpy(), rate=config["sampling_rate"]))
```
