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

```python colab={"base_uri": "https://localhost:8080/"} id="6WN-7_HbSE7j" executionInfo={"status": "ok", "timestamp": 1608621014307, "user_tz": -330, "elapsed": 2262, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a3351780-6250-421b-d921-7b1ef7207541"
!git clone https://github.com/clovaai/deep-text-recognition-benchmark
%cd deep-text-recognition-benchmark
```

```python id="qd_yosLzVPTZ"
models = {
    'None-ResNet-None-CTC.pth': 'https://drive.google.com/open?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
    'None-VGG-BiLSTM-CTC.pth': 'https://drive.google.com/open?id=1GGC2IRYEMQviZhqQpbtpeTgHO_IXWetG',
    'None-VGG-None-CTC.pth': 'https://drive.google.com/open?id=1FS3aZevvLiGF1PFBm5SkwvVcgI6hJWL9',
    'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth': 'https://drive.google.com/open?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY',
    'TPS-ResNet-BiLSTM-Attn.pth': 'https://drive.google.com/open?id=1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9',
    'TPS-ResNet-BiLSTM-CTC.pth': 'https://drive.google.com/open?id=1FocnxQzFBIjDT2F9BkNUiLdo1cC3eaO0',
}
```

```python colab={"base_uri": "https://localhost:8080/"} id="ouoXFnONSJcK" executionInfo={"status": "ok", "timestamp": 1608621099410, "user_tz": -330, "elapsed": 5165, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d2cc1143-8aa9-4587-b2f0-17a51c4055af"
!gdown --id 1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9
```

```python colab={"base_uri": "https://localhost:8080/"} id="dyhHQAz8SfUl" executionInfo={"status": "ok", "timestamp": 1608621324151, "user_tz": -330, "elapsed": 7577, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="30a400d9-05c4-469c-842b-98428083991d"
!python demo.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder demo_image/ --saved_model TPS-ResNet-BiLSTM-Attn.pth
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="RhbRnNHWU6tA" executionInfo={"status": "ok", "timestamp": 1608621744253, "user_tz": -330, "elapsed": 6960, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2a534ff0-82e0-4125-84e1-49635f4cddb8"
output = !CUDA_VISIBLE_DEVICES=0 python3 demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder demo_image/ \
--saved_model TPS-ResNet-BiLSTM-Attn.pth

from IPython.core.display import display, HTML
from PIL import Image
import base64
import io
import pandas as pd

data = pd.DataFrame()
for ind, row in enumerate(output[output.index('image_path               \tpredicted_labels         \tconfidence score')+2:]):
  row = row.split('\t')
  filename = row[0].strip()
  label = row[1].strip()
  conf = row[2].strip()
  img = Image.open(filename)
  img_buffer = io.BytesIO()
  img.save(img_buffer, format="PNG")
  imgStr = base64.b64encode(img_buffer.getvalue()).decode("utf-8") 

  data.loc[ind, 'img'] = '<img src="data:image/png;base64,{0:s}">'.format(imgStr)
  data.loc[ind, 'id'] = filename
  data.loc[ind, 'label'] = label
  data.loc[ind, 'conf'] = conf

html_all = data.to_html(escape=False)
display(HTML(html_all))
```

```python colab={"base_uri": "https://localhost:8080/"} id="A7E6eEIGTMVx" executionInfo={"status": "ok", "timestamp": 1608621583252, "user_tz": -330, "elapsed": 3099, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="840cb42a-ecdb-4c56-af29-31ddc4d3e1c1"
!mkdir -p ./test_image
!wget -O ./test_image/img1.jpg "https://fastly.4sqi.net/img/general/200x200/7440763_ZYmeiNDw296JU-cYjrbeMI4go4kcmVKe-8kk51i0ZQQ.jpg"
!wget -O ./test_image/img2.jpg "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQVrKs07miTOuC86Rj3eFe7t0dC346hTrmq6A&usqp=CAU"
```

```python colab={"base_uri": "https://localhost:8080/"} id="DZZcGq-qUV8N" executionInfo={"status": "ok", "timestamp": 1608621605720, "user_tz": -330, "elapsed": 3969, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8ca57faf-eba3-48c9-ce3a-00c0a6624b61"
!python demo.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder test_image/ --saved_model TPS-ResNet-BiLSTM-Attn.pth
```

```python id="_3V9ThSLUhdP"

```
