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

```python id="6jJDNZiz_d2F" executionInfo={"status": "ok", "timestamp": 1629003047982, "user_tz": -330, "elapsed": 1838, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-nlpt"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZLL5UuDj_ls6" executionInfo={"status": "ok", "timestamp": 1629003049667, "user_tz": -330, "elapsed": 1718, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="67ee6386-9da6-4063-877b-20d69c71f467"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="X_hW4F3V_ltA"
!git status
```

```python id="gK8rOIlb_ltB"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="KSJjtHmhJeV8" -->
---
<!-- #endregion -->

<!-- #region id="E_YmCpgIJexK" -->
To build our emotion detector we’ll use a great dataset from an article that explored how emotions are represented in English Twitter messages. Unlike most sentiment analysis datasets that involve just “positive” and “negative” polarities, this dataset contains six basic emotions: anger, disgust, fear, joy, sadness, and surprise. Given a tweet, our task will be to train a model that can classify it into one of these emotions!
<!-- #endregion -->

```python id="jqqJRhlRPrxa"
!pip install datasets
!pip install transformers
```

```python id="LidQl24sQBCi" executionInfo={"status": "ok", "timestamp": 1629007324587, "user_tz": -330, "elapsed": 6990, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from datasets import list_datasets, load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings('ignore')
```

```python colab={"base_uri": "https://localhost:8080/"} id="0E9IAbRqP6hR" executionInfo={"status": "ok", "timestamp": 1629005044602, "user_tz": -330, "elapsed": 472, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="229aa854-eeac-465c-b7a9-efafbdc3af37"
datasets = list_datasets()
print("There are {} datasets currently available on the Hub.".format(len(datasets)))
print("The first 10 are: {}".format(', '.join(datasets[:10])))
```

```python colab={"base_uri": "https://localhost:8080/"} id="M27ZWfd9QCMl" executionInfo={"status": "ok", "timestamp": 1629005146051, "user_tz": -330, "elapsed": 1129, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a89c065a-21dd-4769-a9c4-96d4193985e6"
metadata = list_datasets(with_details=True)[datasets.index("emotion")]
# Show dataset description
print("Description:", metadata.description, "\n")
# Show first 8 lines of the citation string
print("Citation:", "\n".join(metadata.citation.split("\n")[:8]))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 486, "referenced_widgets": ["d9d489a111f5441a84b41eadf4f4e7eb", "12e0d7cc3d7a408ea2b4952c87412e13", "7038d8a432114910a2d065f23e9b019e", "bc384ac8ae694ceb87c5738ce5683e8e", "6d91cb535929408e8d31d7e0c8b18b2e", "ef6518b4c73c4ed99f81b95a0e6b0d81", "53a3b4b7b87a410fb2d9e12f492fa1fb", "09d0a55c84cf41afbdb402353c047186", "b13ea5a52b9f4ff8990a29a4be42d5db", "1ac9adfd8ed6473dab93617ff0a4d1d5", "52ab1b7d64614ac89079825c0c140b39", "991ce5fa00664ce6a12425c149676e26", "edf4e003e4ab4acb85be3a28d8fa57ac", "da4d5e4a6edb40b3a5c5e042f9c7347f", "9d7c235dea304eb9b3b44ca1d74f7eec", "09bf88e1380b4a5d94def5c29e45ebfc", "4c998c14c9a0495685c27a7a843b61bf", "f9e410482d39470599344ddc43cb0e44", "a38fc6e63a4746c7989f08a235ab215c", "ef0f80bc5bc14154a3e7734cea4fca73", "99676081a8a94069b96c79d8a4525614", "29bad101fafe47eab0a7e5828c39b1b4", "235b6c60b6044bcb8d948e60b2a28cb7", "e5fa1a08662b4a8b8b430c771f061559", "b066a84bb3c94327aa7da273870ac699", "920bd16bb53d48e590e20b2160e03e7d", "8319e4a21f38498686fd1b24fc17e951", "450a2e1cd43144c58ce86c544ea72b7f", "15fbcf6438ff4899bc3457f4f02790d5", "74cd1a0b60ff4c639d3bf7216e7d2490", "715d84edb8de4633b0190212a609c75f", "10a6eb56120a49149d29c08a889c0ac6", "4df9b1a3256e4fed8a12bda8158749db", "7dd64155f8934b759f2d216e70e0d3f9", "09f345b042234a4dbbeedae0f5dbf616", "8024d8e88bbd42daa40bc2f2737199ca", "5b00b8930d8f401f8e08771d3633e765", "8c2735fd79414077b9d53b8536592865", "6765f41f63fe4073b052773cef06f743", "be7f7400334040e29e4e97f438371831", "5e1281068c0146c1b4248c6800d13a81", "d0910db851ec43339cdf2e7491115b80", "6d6fbc69620f41198289ba34b1dee507", "d08ee2ecd4d849f1b0556759c58c9eb7", "aec40ce1df0b466fa87f92f616f00626", "8f5fb30572164d7eb503c998a597719b", "83c213e38c384a36801fbf5b3e0ef5cb", "f837791fc5684d3c9f5ea1c7cddcd948", "d1ae62c266614cb8b610e668162380c5", "c942c6cb740040a29f041a85764deeea", "cd94ff7706c647f2898077d488c12f75", "79bb83f9ba0a42868391029641a3d610", "cb209eaf64ed42e5a50c1c73537919e3", "4de12ab4b0e24b2bbc06fbee66ecfb97", "7725f6a5db404cd1b21f8604d6147b72", "0eec97f9114442438fcf820c5de72606", "bd8be6d9393f469d91c264127a0a7501", "b3238df67c91437188f33f4d50170bf3", "580d3b42f0e44b64b0340ac847ec378d", "664bc85eb3b04cb8b6bc260cb66a1292", "1b54e8fb06cd4bb68b455d46cac2b3c8", "ec3db2b2be1b4f3898397a91ba4f95c0", "41be6969c9b844afbca3551ce9764385", "ffc8b09543f54e26943684cd33a73ad6", "dabc8e7e3d3740579babbd8d159d15be", "bace110e99074843ac6b0c1e57d72526", "37bffe8d7ca14cf39df10dc94478d144", "890a9c8fd3ad42fb89f5bc81a1600edd", "5ec8a7b53cd64bf5b1d592b13a96f1ce", "b0402e35ec2c4e0180b1ae02d4e52c5b", "1ba9e6ab7f8343a58563671624eaab91", "fa7df142d1cc4d1f9ae0c3380377dc2a", "0393ea1b780746b89d1ef00a0cd3fd51", "b68e19e32a0349b6ac2c33d4396eaced", "18ff6ab28334462b824bb214e7824c5e", "1c02ab244ff14d1595bd2838718e30a4", "e63e0a4483d044019ec83d32f8aa3d95", "a8fe0667d86b47b8a34f7bc15dcbe639", "d3c2f7d9cb704eb7b27645789c830bff", "0b5efddbea994cd488a8fb7887458b43", "9e4bbc6025b946a7ba73eda56db85875", "987e72861e6e44cc95b1e415477983fc", "c739963c24504b8884dad06f5601bf67", "c0336f1ceb94456d879e5a415bc4cc73", "c44d59e08ee54a55aee986c46e9bada5", "664c17b8a3d640b9baf7ad23397a3288", "1bf0dc5108a745aaa4445b176cd8d6c4", "5e6cfba89887435280a2d64ccea9a534"]} id="drlOe84ARdtp" executionInfo={"status": "ok", "timestamp": 1629006326144, "user_tz": -330, "elapsed": 8279, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a7ab99ac-0238-446d-d61b-18d7c61567a2"
emotions = load_dataset("emotion")
emotions
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="OpCim_4FV8UY" executionInfo={"status": "ok", "timestamp": 1629006450067, "user_tz": -330, "elapsed": 686, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="efc58e37-0656-4daa-a766-0226653bd683"
emotions.set_format(type="pandas")
df = emotions["train"][:]
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="U-iE5-7LWcfZ" executionInfo={"status": "ok", "timestamp": 1629006567563, "user_tz": -330, "elapsed": 484, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="84134ae3-b015-4c97-ba8e-cbf55f81c98a"
def label_int2str(row, split):
    return emotions[split].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2str, split="train")
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 281} id="NjdXyK7nW5NL" executionInfo={"status": "ok", "timestamp": 1629006677239, "user_tz": -330, "elapsed": 747, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ecfb726a-b7e7-4751-b450-d3b86f24efc1"
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Category Counts");
```

```python colab={"base_uri": "https://localhost:8080/", "height": 280} id="55Xx6vykXKCp" executionInfo={"status": "ok", "timestamp": 1629006867276, "user_tz": -330, "elapsed": 601, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0273e567-3398-43bf-a569-9ec6c498722d"
df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by='label_name', grid=False, showfliers=False,
           color='black', )
plt.suptitle("")
plt.xlabel("");
```

```python colab={"base_uri": "https://localhost:8080/", "height": 145, "referenced_widgets": ["a46794cc1b0d40f4a1833a3983c6a6af", "e04e125a861f4fe285c63f7cabbc5978", "32730142dfe54ca29f19ef38f081f353", "c2ef471372b2493496add6f57441640c", "07ccf3792b864665b43641be60fca7d2", "5cf618efc1a1428e81f2fe7c1ba080bc", "2bae3bb701944542859b88649312f367", "70b89448493f424fb2360c756663d662", "e4f46467e8424febaa1c85a42fcc3d42", "4db518761dbb4722a11731f6fda78979", "96cd2c32587f46d1bec46d9534163481", "1fe2d97718c64ba094fbd8cc64f4a116", "e16a252487e044d896cbccaacd775f80", "d93e0a8455c744eb8be9c7b8ff9e5d1e", "c5c1218e79074cdab86de983e4c07e16", "313e9e4e8a8d4e46bf7fb92230ee10d6", "0973c856cbd94f4b86e2099d9c9d30de", "49b716ad047347e2b0af701402f9fe9e", "042f7ddc9e30434a8437b646f6b09e84", "4050d775e91043ea8a4d39ac65a82b46", "7e8512eb1b0f45aebbbcfeb543be1116", "6deb2350942c4140af0b82ec8e3f930f", "ef420e9348934eb39d275b33661cf3ea", "dad5ee9bad184f29b88fcc09019ebf41", "aa3b58dba5ff4cdd9a86744f7540b409", "a0f00b581e184eea81bc5ebb04de3672", "6cf565b89301435db223d7699399d60d", "a31520f9ef234da68295876741c1a1b0", "4c3025aa56744b6ba9d01d5b1e55adb8", "c63467250a1f4852b26d637b79b6db40", "48af44d01ce4493893a77dcfb8c41412", "3abd43ed4d824a7e8941eeffd0393a76", "22de36edee084b9e9848a3b97196d5a3", "4068c53ac55a4bea8b8d810e9fe2ce04", "4f45f5a0a9f247149c4d3ae31515a9f0", "3a231a936e0d400c957180f6110b8631", "85cb3403013d41039ae3f02a06a83c74", "f2dc43f4b984420198af62527de8c537", "b1239a4689bc4f1582c5c4ab97756b57", "86cb98596423411f832ee93e614c1fd3", "734cee46cacb48ad9e6d834d5e645705", "cbf0dcf51432473ea0bec88d9dfb3f89", "25b34dce4be744bc9e7ce5738a2642cf", "3fb1a2e0c562448199d406ef44b94bdd"]} id="57kdB0WaXfIo" executionInfo={"status": "ok", "timestamp": 1629007334542, "user_tz": -330, "elapsed": 2503, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b701eda4-b1d3-4be9-840b-bba967001d9d"
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="CSQxUMUKZz_8" executionInfo={"status": "ok", "timestamp": 1629007356636, "user_tz": -330, "elapsed": 438, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7bbb65bc-213d-4220-d772-0290e8103f3f"
encoded_str = tokenizer.encode("this is a complicatedtest")
encoded_str
```

```python colab={"base_uri": "https://localhost:8080/"} id="6xb85xMZZ52U" executionInfo={"status": "ok", "timestamp": 1629007367984, "user_tz": -330, "elapsed": 430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4e33a866-d776-429e-fe55-9a1b27e98ff0"
for token in encoded_str:
    print(token, tokenizer.decode([token]))
```

```python id="egRW4keaZ8px"

```
