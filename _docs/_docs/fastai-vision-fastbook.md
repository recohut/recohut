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

<!-- #region id="BHuwyXsDwxh1" -->
# FastAI Vision from Fastbook
<!-- #endregion -->

```python id="O_l6mog-NErp"
!git clone https://github.com/fastai/fastbook.git
%cd fastbook
!pip install -r requirements.txt
```

```python id="wyfH-SjyZL3q"
from utils import *
from fastai2.vision.widgets import *
```

```python id="aeVYjKSUraQW"
from fastai2.vision.all import *
```

```python id="LW0uWvsWNZlT"
path = untar_data(URLs.PETS)/'images'
dbunch = ImageDataLoaders.from_name_func(path, get_image_files(path), valid_pct=0.2, 
                                       label_func=lambda x: x[0].isupper(),
                                       item_tfms = Resize(224))
learn = cnn_learner(dbunch, resnet34, metrics=error_rate)
learn.fine_tune(2)
```

```python id="akAkDefMSrk2" outputId="7932b808-7c5a-45f5-b557-c27a3c1be9ff" executionInfo={"status": "ok", "timestamp": 1591223418892, "user_tz": -330, "elapsed": 1206, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} colab={"base_uri": "https://localhost:8080/", "height": 52}
img = PILImage.create('/content/dog_test.jpg')
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
```

```python id="qhVFq_ikbG3c"
path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    codes = np.loadtxt(path/'codes.txt', dtype=str)
)

learn = unet_learner(dls, resnet34)
learn.fine_tune(8)
```

```python id="B1C7yOdNb-Ck" outputId="32bd8503-3245-4276-948e-05eaf0cf8765" executionInfo={"status": "ok", "timestamp": 1591259390992, "user_tz": -330, "elapsed": 4345, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} colab={"base_uri": "https://localhost:8080/", "height": 216}
learn.show_results(max_n=6, figsize=(4,3))
```

```python id="WVFKJPpBreqC"
from fastai2.text.all import *
```

```python id="BykkkD1FckNf"
dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)
```

```python id="RPFV4B9Wda7a"
learn.predict("This was a time pass and not good")
```

<!-- #region id="pECg_9Sqp3Uu" -->
[How to determine if cattle are bulls, steers, cows or heifers](https://www.farmanddairy.com/top-stories/how-to-determine-if-cattle-are-bulls-steers-cows-or-heifers/274534.html)
<!-- #endregion -->

```python id="MvvAHvGApru1" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="bac8a340-713c-4bda-8ea6-ab582d33f593" executionInfo={"status": "ok", "timestamp": 1591289810567, "user_tz": -330, "elapsed": 1839, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# bing image search api
key = '519a49fbf31549b2adf45c926f85b679'
results = search_images_bing(key, 'bull cattle')
ims = results.attrgot('content_url')
len(ims)
```

```python id="LwZbKZBQtiTT" colab={"base_uri": "https://localhost:8080/", "height": 145} outputId="2d7bbf00-8bda-4550-f9a3-f0e0c6853b43" executionInfo={"status": "ok", "timestamp": 1591289812309, "user_tz": -330, "elapsed": 1113, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
dest = 'images/bullx.jpg'
download_url(ims[0], dest)
im = Image.open(dest)
im.to_thumb(128,128)
```

```python id="1bSyjnoctiRF" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="e816de8b-7cda-4cd2-811e-3f52692b227e" executionInfo={"status": "ok", "timestamp": 1591289881374, "user_tz": -330, "elapsed": 68330, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
cattle_types = 'bull','steer','cow','heifer'
path = Path('cattles')

# download the images
if not path.exists():
    path.mkdir()
    for o in cattle_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} cattle')
        download_images(dest, urls=results.attrgot('content_url'))

# verify the images
fns = get_image_files(path)
failed = verify_images(fns)
failed.map(Path.unlink)
```

```python id="1DtqjPla0a4A"
# from data to dataloaders
cattles = DataBlock(blocks=(ImageBlock, CategoryBlock),
                    get_items=get_image_files,
                    splitter=RandomSplitter(valid_pct=0.2, seed=42),
                    get_y=parent_label,
                    item_tfms=Resize(128))

dls = cattles.dataloaders(path)
```

```python id="HcF_Thtp8jyK" colab={"base_uri": "https://localhost:8080/", "height": 193} outputId="31ac18c4-3fe7-4189-dfc8-6de5a041ee57" executionInfo={"status": "ok", "timestamp": 1591289894325, "user_tz": -330, "elapsed": 74919, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
dls.valid.show_batch(max_n=4, nrows=1)
```

```python id="kv1dWCE-9w12" colab={"base_uri": "https://localhost:8080/", "height": 193} outputId="243996fe-a5e0-4aed-fa14-8e05efbf0687" executionInfo={"status": "ok", "timestamp": 1591289896987, "user_tz": -330, "elapsed": 77361, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
cattles = cattles.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = cattles.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
```

```python id="dQvO8Uql-fG8" colab={"base_uri": "https://localhost:8080/", "height": 193} outputId="ba63b7cd-534c-4088-ee8c-1e4eae1459c6" executionInfo={"status": "ok", "timestamp": 1591289899881, "user_tz": -330, "elapsed": 79760, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
cattles = cattles.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = cattles.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
```

```python id="xyGXYNWI_yHT" colab={"base_uri": "https://localhost:8080/", "height": 371} outputId="611d66aa-44d7-4ff1-ebf1-8d3ffc853de0" executionInfo={"status": "ok", "timestamp": 1591289903185, "user_tz": -330, "elapsed": 82884, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# data augmentation
cattles = cattles.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = cattles.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)
```

<!-- #region id="HP6iNm5xzuls" -->
We don't have a lot of data for our problem (150 pictures of each sort of cattle at most), so to train our model, we'll use RandomResizedCrop with an image size of 224 px, which is fairly standard for image classification, and default aug_transforms:
<!-- #endregion -->

```python id="XX8uvF0ny6dI"
cattles = cattles.new(item_tfms=RandomResizedCrop(224, min_scale=0.5),
                      batch_tfms=aug_transforms())
dls = cattles.dataloaders(path)
```

```python id="QT3g1tCC0Djy" colab={"base_uri": "https://localhost:8080/", "height": 320, "referenced_widgets": ["d1497a792e754d858fc402d28995eaab", "82bd840ff5e74ef099354bd33f18a8ca", "93e6882d4bc14ce6b35a4a068fc8bde9", "a6707036c2814c0488ba2ca771e7b778", "1e2f75588fe649c9a8a3744b638f3e60", "0e681431d1884412b28496b359753fa3", "b683a033fdae40608669e270f9932bc7", "293940340f76488787166069fe50705a"]} outputId="0990a3d6-5d78-48df-b2a3-a0cd2ff91a2d" executionInfo={"status": "ok", "timestamp": 1591290000340, "user_tz": -330, "elapsed": 177549, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
learn = cnn_learner(dls, resnet50, metrics=error_rate)
learn.fine_tune(4)
```

```python id="qxl8TGeW1u6h" colab={"base_uri": "https://localhost:8080/", "height": 310} outputId="59a79e80-0ac9-4115-8d99-0d7697a91798" executionInfo={"status": "ok", "timestamp": 1591290003927, "user_tz": -330, "elapsed": 180770, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```

```python id="Yh51ZWGv2OdQ" colab={"base_uri": "https://localhost:8080/", "height": 189} outputId="687f8a1e-9be5-4c97-bc6d-cee6538f9a61" executionInfo={"status": "ok", "timestamp": 1591290005557, "user_tz": -330, "elapsed": 179927, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
interp.plot_top_losses(5, nrows=1)
```
