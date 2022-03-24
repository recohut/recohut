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

<!-- #region id="L0HphU76Nfbe" -->
### Download the dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2Ym3k-6FJ4lI" executionInfo={"status": "ok", "timestamp": 1629876535125, "user_tz": -330, "elapsed": 11300, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="678c41f9-9f31-49d7-8a7d-f8fd3c9f8702"
![[ ! -f google_api.py ]] && wget -q --show-progress https://raw.githubusercontent.com/Hernan4444/MyScripts/master/google_drive/google_api.py
from google_api import download_file_without_authenticate
import os
 
if not os.path.exists('wikimedia_recsys.zip'): 
    download_file_without_authenticate('1rXRT4Pa1opD_3koIQ2uvImjBlACcdC6R', 'wikimedia_recsys.zip')
```

```python id="ZatGleOoNg_G" executionInfo={"status": "ok", "timestamp": 1629876584414, "user_tz": -330, "elapsed": 24628, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
![[ ! -d wikimedia_recsys ]] && unzip -q wikimedia_recsys.zip
```

```python id="OvRVTH1zNpy0" executionInfo={"status": "ok", "timestamp": 1629876597213, "user_tz": -330, "elapsed": 453, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
path_to_data = 'wikimedia_recsys/images'
```

```python colab={"base_uri": "https://localhost:8080/"} id="WXmj2k6-Ny1G" executionInfo={"status": "ok", "timestamp": 1629876607290, "user_tz": -330, "elapsed": 422, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0cdda45e-831f-48df-82ee-768e086ce052"
!ls wikimedia_recsys/images | wc -l
```

```python id="EQm_XtZoN1Tw" executionInfo={"status": "ok", "timestamp": 1629876631289, "user_tz": -330, "elapsed": 5014, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from collections import defaultdict
import math
from pathlib import Path
import random
import numpy as np
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.io import read_image
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid
import PIL
from PIL import Image, ImageFile
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
```

<!-- #region id="s4UaFyuxOERG" -->
## Read the data
<!-- #endregion -->

```python id="x3mUCOUtN6A4" executionInfo={"status": "ok", "timestamp": 1629876675895, "user_tz": -330, "elapsed": 414, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Needed for some images in the Wikimedia dataset, 
# They are too large for PIL

Image.MAX_IMAGE_PIXELS = 3_000_000_000
# Some images are "broken" in Wikimedia dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["8f4e8a1f59c64ceb87c02923e5ff2062", "5510b9672cdf4086b571dcbb3a467424", "cd5b60a512f54f879738b454e22f9a3a", "0111934b521843f7bcf295d03af64b3f", "3680c16cfdf44593989238266fb4c100", "2258894e0c904a3daaa705a3e9363185", "10a98bf262c440439853c2301d310ce7", "cb27c9d014ee407185166ff43dbb9283", "c4d2b163aaa544f58a81ce0d58c0b5c1", "a4b1087d48f44197a97d5c2fa64e916e", "7f5f18379cc6492d8565902cff45d7cd", "2e1a202666aa42cc98c11a6205b86cd8", "d610d6df5627416da02e6eed0d55ce7e"]} id="ZiHATAO_OGDq" executionInfo={"status": "ok", "timestamp": 1629876792375, "user_tz": -330, "elapsed": 10683, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="04b280d5-b8e2-4884-9bef-866b3da2f1ba"
# Create an array data[] to store images and images' paths

file_types = ['jpg', 'jpeg', 'png']

data = defaultdict(list)
for file_type in file_types:
    for path in tqdm(Path(path_to_data).glob(f'*.{file_type}')):
        try:
            image = Image.open(str(path)).convert("RGB")
            data['image_path'].append(path)
            data['images'].append(image)
        except RuntimeError:
            pass
```

```python id="59oheVEZOXvA" executionInfo={"status": "ok", "timestamp": 1629876870595, "user_tz": -330, "elapsed": 445, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def display_images(elements, columns=5):
    for label, items in elements.items():
        n_rows = ((len(items) - 1) // columns + 1)

        fig = plt.figure(figsize=(columns * 2, 2.7 * n_rows))
        plt.title(f"{label.title()} (n={len(items)})\n", fontdict={'fontweight': 'bold'})
        plt.axis("off")
        for i, item in enumerate(items, start=1):
            image_title = item.get('title', "")
            image = item['image']
            image_subtitle = item.get('subtitle', "")

            image_height = image.height
            subtext_loc_x = 0
            subtext_loc_y = image_height + 30

            ax = fig.add_subplot(n_rows, columns, i)
            ax.patch.set_edgecolor("black")
            ax.patch.set_linewidth("5")
            ax.set_title(image_title, color="black")
            ax.text(subtext_loc_x, subtext_loc_y, image_subtitle)

            plt.xticks([])
            plt.yticks([])
            plt.imshow(image)

to_std_size = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 502} id="QzMWpYD3O1jz" executionInfo={"status": "ok", "timestamp": 1629876907268, "user_tz": -330, "elapsed": 3372, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cfb97059-bbb6-4ebc-a60e-aca139db9589"
n_items = 15
columns = 5

title = "Random Wikimedia Dataset Images\n"

selected_indices = random.choices(list(range(len(data['images']))), k=n_items)
# items = [(idx, to_std_size(data['images'][idx])) for idx in sorted(selected_indices)]
items_to_display = [{
    'title': f'idx={idx}',
    'image': to_std_size(data['images'][idx])
    } for idx in sorted(selected_indices)]

display_images({"Random Wikimedia Dataset Images": items_to_display}, columns=columns)
```

```python id="2REpiY4TO9zu" executionInfo={"status": "ok", "timestamp": 1629877018157, "user_tz": -330, "elapsed": 563, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
class FeatureExtractor(nn.Module):
    def __init__(self, base_model, output_layer, requires_grad=True):
        super().__init__()
        self.output_layer = output_layer
        self.pretrained = base_model(pretrained=True)
        self.children_list = []
        for n, c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == self.output_layer:
                break

        self.net = nn.Sequential(*self.children_list)
        for param in self.net.parameters():
            param.requires_grad = requires_grad

        self.pretrained = None
        
    def forward(self,x):
        x = self.net(x)
        return x
```

```python id="ZN1Bck6BPZkt" executionInfo={"status": "ok", "timestamp": 1629877027409, "user_tz": -330, "elapsed": 431, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# Mean and std deviation from images in the Imagenet dataset
mean_pixel = [0.485, 0.456, 0.406]
std_pixel = [0.229, 0.224, 0.225]

normalize_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_pixel, std=std_pixel)
])
```

<!-- #region id="CHDzwwstRd8T" -->

So far we have our images' visual features. What's next? We can use these features to identify similar images. The following method will allow us:

*   To find the *k* most similar images to an input image *query*: **topk_similar**(query, knowledge_base, k=10)


We will use this function for the already instanced feature extractor (AlexNet), but we will also use it for the other methods (VGG, ResNet, NASNet)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kYyMVKHEPb3o" executionInfo={"status": "ok", "timestamp": 1629877072608, "user_tz": -330, "elapsed": 476, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c002a518-f9aa-438f-d365-5a3de2f16fc9"
# Load the pretrained model with this simple code!

models.alexnet()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 152, "referenced_widgets": ["da6ff256c81f449db88d5034bd7ced64", "5f43878d8187468b8143b48152caf44a", "12675056814a41eb9418167b2749e473", "eac6c963da954f638d555c395d766499", "34483aa50f9a4822ac44bb01bfe76233", "1769e865fc634967b6898bbe6f2a9e24", "6a34cedb2d334b8da78acf8fb682f0ab", "378d704d865c4a9e92cf115446b4a7c3", "4f2221efa923444bbc44d844fcfe28b9", "4523fd2af2ac4ec9a614e9f88757ba34", "d5e8cc979adc4b3d99915a1a8a65c13b", "8e6360370f1a43b3ad41316ebee491a2", "5eb2fd52e7e84d65aa8a997866e03926", "83b2dfa16b354cd5970626c767096aba", "340e9dce9d524c75a2b83f2d01e8f255", "15dffbc9f8554b3fb14f10ad3e14b612", "c0a93964e42240caa5b13b88b0c694a3", "3599ff5be36e48a6bfa791ae3b358fe6", "33b7e4f5c69a4729ae4ea6d4758f0404", "115ce694aa124c1598cd1c52e5e91096", "798f2ce577ca4e4699365f97dbb781dd", "6b1a34eade024813b0ed5a49a695152e"]} id="LCdRiy_1Pm0j" executionInfo={"status": "ok", "timestamp": 1629877309580, "user_tz": -330, "elapsed": 193795, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="559c8251-9cea-43f9-8cd7-3ed5d41c1340"
# this block downloads a checkpoint of AlexNet from pytorch

model = FeatureExtractor(models.alexnet, 'avgpool', requires_grad=False)
# model = model.cuda()

# we use model.eval() to prevent "training" mode
model.eval()
for image in tqdm(data['images']):
    normalized_image = normalize_image(image).unsqueeze(dim=0)
    # normalized_image = normalized_image.cuda()
    features = model(normalized_image)
    features = F.adaptive_avg_pool2d(features, output_size=1) if len(features.shape) > 2 else features
    features = features.cpu()
    data['alexnet_features'].append(features)

data['alexnet_features'] = torch.cat(data['alexnet_features'])
data['alexnet_features'] = data['alexnet_features'].squeeze()

model = normalized_image = features = None
```

```python id="ZtXFZLtjPxfC" executionInfo={"status": "ok", "timestamp": 1629877309583, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def topk_similar(query, knowledge_base, k=10):
    k = k + 1 # Remove self similarity
    distances = 1 - F.cosine_similarity(knowledge_base, query)
    topk = distances.topk(k, largest=False)
    indices = topk.indices.tolist()
    distances = topk.values.tolist()

    if len(indices) > 0:
        # Remove query item
        closest_distance = distances[0]
        indices = indices[1:] if math.isclose(closest_distance, 0.0) else indices
        distances = distances[1:] if math.isclose(closest_distance, 0.0) else distances

    return indices, distances
```

```python colab={"base_uri": "https://localhost:8080/", "height": 547} id="ZYwTpTazQN7v" executionInfo={"status": "ok", "timestamp": 1629877415803, "user_tz": -330, "elapsed": 3968, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a932f3c7-22f0-4e95-92ec-d1f6913c45d7"
#pick_randomly = True
#idx = random.randint(0, len(data['image_path'])) if pick_randomly else idx


idx = 2460

features_id = 'alexnet_features'
features = data[features_id][idx, :].unsqueeze(0)

image = to_std_size(data['images'][idx])

closest_indices, closest_distances = topk_similar( features, data[features_id], k=10 )
closest_images = [{
    'title': f'idx={idx_}',
    'image': to_std_size(data['images'][idx_]),
    'subtitle': f'similarity={distance:.4f}'
    } for idx_, distance in zip(closest_indices, closest_distances)]

display_images({
    'Query Image': [{
        'title': f'idx={idx}',
        'image': image
        }],
    f'Closest Images using AlexNet features': closest_images
}, columns=5)
```

```python id="X0SVCtQgQ50U"
models.vgg16()
```

<!-- #region id="j08f3x4GXtR-" -->
Let's try this code with other Deep CNNs! VGG_16, ResNet_18 and NASNet. 
<!-- #endregion -->

<!-- #region id="P7jD1Jta99pg" -->
The VGG architecture was able to train deeper networks by using smaller filter (notice differences in kernel size with Alexnet)

> Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556

<!-- #endregion -->

```python id="i_VnewzFZ3Lh" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1618329831441, "user_tz": 240, "elapsed": 3885, "user": {"displayName": "DENIS PARRA SANTANDER", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjmVafZPZWJqYZbC-4vPKLLZpO3NQinpXv04DYS=s64", "userId": "18074780456384962897"}} outputId="43766692-03c9-4754-d5bb-64b710afc170"
models.vgg16()
```

```python id="P4iuzTc8UKrs" colab={"base_uri": "https://localhost:8080/", "height": 152, "referenced_widgets": ["1fd4fa20db254e61ba6550f90b6c9b94", "f2c8cdcf1c864261a1b391b0ce733a18", "0e312d3f2bf64d1b96b7b0a907527379", "b0760039b75a4f9d9a6f67cf9231dbfc", "0532a4caa6f8468593d2a3f1f392f63c", "9ae3aae772d04a09985c792bb3955426", "5175d0345c4c4e259a25aa105f2478e1", "d5a9170334c94679bb56dbf56c2b0c01", "2d371ff6bcc94e34b5d0e56d3b7a2983", "f98dfacc0b95419198e61f3b5334e6db", "c904bc6758374b818c83cd740b181a19", "fc91b269bcf0465c9028760cfba18450", "9e555e54f14945568ae08e24c769dd6d", "9e5383cd16924ace8aee201c280b2ef4", "e248afc541c04bd1ab9f0ed14d206011", "5e4fc4b8d1614f6181a6fc8383c2c9a4"]} executionInfo={"status": "ok", "timestamp": 1618329953614, "user_tz": 240, "elapsed": 118787, "user": {"displayName": "DENIS PARRA SANTANDER", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjmVafZPZWJqYZbC-4vPKLLZpO3NQinpXv04DYS=s64", "userId": "18074780456384962897"}} outputId="590c7c5e-f1ea-4875-9a04-3867f909e265"
# this block downloads a checkpoint of VGG-16 from pytorch

model = FeatureExtractor(models.vgg16, 'avgpool', requires_grad=False)
model = model.cuda()
    
model.eval()
for image in tqdm(data['images']):
    normalized_image = normalize_image(image).unsqueeze(dim=0)
    normalized_image = normalized_image.cuda()
    features = model(normalized_image)
    features = F.adaptive_avg_pool2d(features, output_size=1) if len(features.shape) > 2 else features
    features = features.cpu()
    data['vgg16_features'].append(features)

data['vgg16_features'] = torch.cat(data['vgg16_features'])
data['vgg16_features'] = data['vgg16_features'].squeeze()

model = normalized_image = features = None
```

<!-- #region id="tVAnROiUAZZH" -->
ResNet introduces residual connections, what reduces number of parameters and allows to increase network depth

> He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ptdZsMzuVYiN" executionInfo={"status": "ok", "timestamp": 1618330357783, "user_tz": 240, "elapsed": 1310, "user": {"displayName": "DENIS PARRA SANTANDER", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjmVafZPZWJqYZbC-4vPKLLZpO3NQinpXv04DYS=s64", "userId": "18074780456384962897"}} outputId="4b0f1ec6-ea41-4279-e688-437e19ac07be"
models.resnet18()
```

```python id="RzN9-cTGUKgU" colab={"base_uri": "https://localhost:8080/", "height": 152, "referenced_widgets": ["fa32cdbe7408490db8794c699539101b", "76143776c48a40839105e5dfa4cd1f53", "a17851bc889545299c71b5396e338696", "e25b98aaa9e745bd8d2b9944d920e950", "af320f63f3a040b3a1f3b1c8f961967a", "9aeef1f8d08a42d0b6cb1d73620d38fc", "55c302ab3e52493daeda872b155a318b", "64ae054ec8f14160aef671fbd9573856", "84e321b313ab49399036cd2e874eb8a6", "5098a6ddd3d146e186e3370e53740f6d", "55105a72c2cf458bb1701ae1d96298b2", "5b24864aba95496ea1fb92f7be7900ba", "c747adccb6ba43a9b408a024ae1e5fc2", "18b8d8e1ca544609be2abb73a6384ef6", "80dae4f737c842fc9bff8c662830067f", "ac52932069c046ca87f76a35a24f1ff8"]} executionInfo={"status": "ok", "timestamp": 1618330596866, "user_tz": 240, "elapsed": 67575, "user": {"displayName": "DENIS PARRA SANTANDER", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjmVafZPZWJqYZbC-4vPKLLZpO3NQinpXv04DYS=s64", "userId": "18074780456384962897"}} outputId="b8ba7331-9f4a-4ffd-f49a-df132c067695"
# this block downloads a checkpoint of ResNet-18 from pytorch

model = FeatureExtractor(models.resnet18, 'avgpool', requires_grad=False)
model = model.cuda()    

model.eval()
for image in tqdm(data['images']):
    normalized_image = normalize_image(image).unsqueeze(dim=0)
    normalized_image = normalized_image.cuda()
    features = model(normalized_image)
    features = F.adaptive_avg_pool2d(features, output_size=1) if len(features.shape) > 2 else features # Reduce feature cube to 1-d
    features = features.cpu()
    data['resnet18_features'].append(features)

data['resnet18_features'] = torch.cat(data['resnet18_features'])
data['resnet18_features'] = data['resnet18_features'].squeeze()

model = normalized_image = features = None
```

<!-- #region id="Cot4B1UCBKVH" -->
In NASNet (Network Architecture Search Networks) the global architure is fixed but blocks or cells are not predefined by authors. Instead, they are searched by reinforcement learning search.

> Zoph, B., Vasudevan, V., Shlens, J., & Le, Q. V. (2018). Learning transferable architectures for scalable image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 8697-8710).

MNASNet is the version available in Pytorch, where "M" is for Mobile


> Tan, M., Chen, B., Pang, R., Vasudevan, V., Sandler, M., Howard, A., & Le, Q. V. (2019). Mnasnet: Platform-aware neural architecture search for mobile. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2820-2828).


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LeQrEm-8wJKD" executionInfo={"status": "ok", "timestamp": 1618330840272, "user_tz": 240, "elapsed": 1808, "user": {"displayName": "DENIS PARRA SANTANDER", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjmVafZPZWJqYZbC-4vPKLLZpO3NQinpXv04DYS=s64", "userId": "18074780456384962897"}} outputId="b8978641-30f8-4135-c8c5-d8764e52a987"
models.mnasnet1_0()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 152, "referenced_widgets": ["c6c8f3093517405aaeb5fabb7df73783", "06b64203639a4d8b85b1e31e61eebc66", "0c275e1143c6480cb74c4e56d8aa1682", "916a449868f646499f789b24ff3ca00f", "22e9f6c6a5d749aab26bf1ddaeb49e58", "30d055802011433ebd76a9d1b84fcbbc", "f3e7516d6871477e8cbcc51081d561db", "bd5746503478427cb2b82812ed011201", "0781075dd9f242bcb044db11eac25d8c", "7ab715162c5c458394ff4164ab26d802", "e1fc58aabf3240b3bb27744884c42bb3", "68cb86d646ae42249e74a39e48f07250", "eed7631240f1475e9128d49bc9ca3508", "e77c5d0df7504c5d884d50cc4404827c", "de2ee6affe234e319c8e0464cfe0b08c", "0546a7b0747246efbdceeef18c87a5e9"]} id="dn8cMtaOwJKV" executionInfo={"status": "ok", "timestamp": 1618330917294, "user_tz": 240, "elapsed": 48092, "user": {"displayName": "DENIS PARRA SANTANDER", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjmVafZPZWJqYZbC-4vPKLLZpO3NQinpXv04DYS=s64", "userId": "18074780456384962897"}} outputId="b8d28aad-a8c6-4b11-f21b-b1b7545de0cc"
# this block downloads a checkpoint of NASNet from pytorch

model = FeatureExtractor(models.mnasnet1_0, 'layers', requires_grad=False)
model = model.cuda()
    
model.eval()
for image in tqdm(data['images']):
    normalized_image = normalize_image(image).unsqueeze(dim=0)
    normalized_image = normalized_image.cuda()
    features = model(normalized_image)
    features = F.adaptive_avg_pool2d(features, output_size=1) if len(features.shape) > 2 else features
    features = features.cpu()
    data['mnasnet_features'].append(features)

data['mnasnet_features'] = torch.cat(data['mnasnet_features'])
data['mnasnet_features'] = data['mnasnet_features'].squeeze()

model = normalized_image = features = None
```

<!-- #region id="rjCnpg0ORzXN" -->

So far we have our images' visual features. What's next? We can use these features to identify similar images. The following method will allow us:

*   To find the *k* most similar images to an input image *query*: **topk_similar**(query, knowledge_base, k=10)


We will use this function for the already instanced feature extractor (AlexNet), but we will also use it for the other methods (VGG, ResNet, NASNet)
<!-- #endregion -->

```python id="ZV1Cw2dgguMB" cellView="form"
idx =  596#@param {type:"integer"}
pick_randomly = False #@param {type:"boolean"}
features_to_compare = "resnet18" #@param ["", "alexnet", "vgg16", "resnet18", "mnasnet"]
k_closest = 10 #@param {type:"integer"}
```

```python colab={"base_uri": "https://localhost:8080/", "height": 547} id="veTC35YRa65A" executionInfo={"status": "ok", "timestamp": 1618331155754, "user_tz": 240, "elapsed": 3546, "user": {"displayName": "DENIS PARRA SANTANDER", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjmVafZPZWJqYZbC-4vPKLLZpO3NQinpXv04DYS=s64", "userId": "18074780456384962897"}} outputId="ef9638a2-bbd8-4d66-b399-3bd7c6d2dea7"
idx = random.randint(0, len(data['image_path'])) if pick_randomly else idx

features_to_compare = random.choice(
    ["alexnet", "vgg16", "resnet18", "mnasnet"]) if not features_to_compare else features_to_compare
features_id = f'{features_to_compare}_features'
features = data[features_id][idx, :].unsqueeze(0)

image = to_std_size(data['images'][idx])

closest_indices, closest_distances = topk_similar(features, data[features_id], k=k_closest)
closest_images = [{
    'title': f'idx={idx_}',
    'image': to_std_size(data['images'][idx_]),
    'subtitle': f'similarity={distance:.4f}'
    } for idx_, distance in zip(closest_indices, closest_distances)]

display_images({
    'Query Image': [{
        'title': f'idx={idx}',
        'image': image
        }],
    f'Closest Images using {features_to_compare} features': closest_images
}, columns=6)
```
