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

<!-- #region id="BwcrQnsO3vJD" -->
# OpenPifPaf Pose Estimation
<!-- #endregion -->

```python id="AjX2DiLAfRMh"
!pip install --upgrade openpifpaf==0.10.1

import io
import numpy as np
import openpifpaf
import PIL
import requests
import torch

print(openpifpaf.__version__)
print(torch.__version__)
```

```python id="of95kiw6fXBG" colab={"base_uri": "https://localhost:8080/", "height": 561} outputId="c6771b94-27a3-4638-92d9-3af55cdf1025" executionInfo={"status": "ok", "timestamp": 1589045619982, "user_tz": -330, "elapsed": 23902, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
image_response = requests.get('https://i.pinimg.com/originals/8e/30/a6/8e30a6c50bcf6c3dd8fc0ed025df48f4.png')
pil_im = PIL.Image.open(io.BytesIO(image_response.content)).convert('RGB')
im = np.asarray(pil_im)
with openpifpaf.show.image_canvas(im) as ax:
  pass
```

```python id="BEsjD1jQffga" colab={"base_uri": "https://localhost:8080/", "height": 375, "referenced_widgets": ["243856e0bf024e24829bc726297d5586", "e346503124674cc3ac865e72f6081d2f", "2a68e8feb94e48999b2e664eec4fcd46", "5ab57ce3e06941a6a8f5044a754abc1e", "885d5e1e28c042bebed2e5eaa523ce3d", "2f88f763d0c94eaf845a9ba6dcc3e57c", "07c1c22e742d46da950e396966574150", "36a4d998aaa84516b58e8b8b45b53927"]} outputId="9c3c4ebe-a00e-4b01-8e80-ce5c7fd76514" executionInfo={"status": "ok", "timestamp": 1589045625539, "user_tz": -330, "elapsed": 29434, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
net_cpu, _ = openpifpaf.network.factory(checkpoint='resnet101')
```

```python id="6dvRDJCEfqWP"
net = net_cpu.cuda()
decode = openpifpaf.decoder.factory_decode(net, 
                                           seed_threshold=0.5)
processor = openpifpaf.decoder.Processor(net, decode, 
                                         instance_threshold=0.2,
                                         keypoint_threshold=0.3)
```

```python id="k3NBKKfIfz_u" colab={"base_uri": "https://localhost:8080/", "height": 561} outputId="61a1ff40-ea27-49f3-cf71-b3d5460f7fb5" executionInfo={"status": "ok", "timestamp": 1589045672783, "user_tz": -330, "elapsed": 3482, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
data = openpifpaf.datasets.PilImageList([pil_im])
loader = torch.utils.data.DataLoader(data, batch_size=1, pin_memory=True)

keypoint_painter = openpifpaf.show.KeypointPainter(color_connections=True, linewidth=6)

for images_batch, _, __ in loader:
  images_batch = images_batch.cuda()
  fields_batch = processor.fields(images_batch)
  predictions = processor.annotations(fields_batch[0])
  
  with openpifpaf.show.image_canvas(im) as ax:
    keypoint_painter.annotations(ax, predictions)
```
