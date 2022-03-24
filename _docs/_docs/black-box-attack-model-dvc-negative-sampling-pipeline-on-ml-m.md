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

```python id="ehMex-aqyDNF"
import os
project_name = "recobase"; branch = "US567625"; account = "recohut"
project_path = os.path.join('/content', branch)

if not os.path.exists(project_path):
    !pip install -U -q dvc dvc[gdrive]
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    !mkdir "{project_path}"
    %cd "{project_path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout -b "{branch}"
else:
    %cd "{project_path}"
```

```python id="V-iFgtQizQfH"
!git status
```

```python id="ZfU-gZ86yDNN"
!git add .
!git commit -m 'commit'
!git push origin "{branch}"
```

```python id="001B38bSC_KY"
!dvc pull data/bronze/ml-1m/ratings.dat
```

```python colab={"base_uri": "https://localhost:8080/"} id="vT9-NFmqD-MT" executionInfo={"status": "ok", "timestamp": 1631367223700, "user_tz": -330, "elapsed": 14263, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="94f0625e-3c9c-431f-f5ba-d43a110761f9"
!dvc repro
```

```python colab={"base_uri": "https://localhost:8080/"} id="1oYU6Se6EQAx" executionInfo={"status": "ok", "timestamp": 1631370790222, "user_tz": -330, "elapsed": 439, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d45dac4e-8638-4fc4-e37d-bf4dda4b9892"
%%writefile ./src/negative_sampling.py
from abc import *
from pathlib import Path
import pickle
import os
from tqdm import trange
import numpy as np
from collections import Counter


class AbstractNegativeSampler(metaclass=ABCMeta):
    def __init__(self, train, val, test, user_count, item_count, sample_size, seed, flag, save_folder):
        self.train = train
        self.val = val
        self.test = test
        self.user_count = user_count
        self.item_count = item_count
        self.sample_size = sample_size
        self.seed = seed
        self.flag = flag
        self.save_path = save_path

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def generate_negative_samples(self):
        pass

    def get_negative_samples(self):
        savefile_path = self._get_save_path()
        print("Negative samples don't exist. Generating.")
        seen_samples, negative_samples = self.generate_negative_samples()
        with savefile_path.open('wb') as f:
            pickle.dump([seen_samples, negative_samples], f)
        return seen_samples, negative_samples

    def _get_save_path(self):
        folder = Path(self.save_path)
        if not folder.is_dir():
            folder.mkdir(parents=True)
        # filename = '{}-sample_size{}-seed{}-{}.pkl'.format(
        #     self.code(), self.sample_size, self.seed, self.flag)
        filename = 'negative_samples_{}.pkl'.format(self.flag)
        return folder.joinpath(filename)


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        num_samples = 2 * self.user_count * self.sample_size
        all_samples = np.random.choice(self.item_count, num_samples) + 1

        seen_samples = {}
        negative_samples = {}
        print('Sampling negative items randomly...')
        j = 0
        for i in trange(self.user_count):
            user = i + 1
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])
            seen_samples[user] = seen

            samples = []
            while len(samples) < self.sample_size:
                item = all_samples[j % num_samples]
                j += 1
                if item in seen or item in samples:
                    continue
                samples.append(item)
            negative_samples[user] = samples

        return seen_samples, negative_samples


class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        popularity = self.items_by_popularity()
        items = list(popularity.keys())
        total = 0
        for i in range(len(items)):
            total += popularity[items[i]]
        for i in range(len(items)):
            popularity[items[i]] /= total
        probs = list(popularity.values())
        num_samples = 2 * self.user_count * self.sample_size
        all_samples = np.random.choice(items, num_samples, p=probs)

        seen_samples = {}
        negative_samples = {}
        print('Sampling negative items by popularity...')
        j = 0
        for i in trange(self.user_count):
            user = i + 1
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])
            seen_samples[user] = seen

            samples = []
            while len(samples) < self.sample_size:
                item = all_samples[j % num_samples]
                j += 1
                if item in seen or item in samples:
                    continue
                samples.append(item)
            negative_samples[user] = samples

        return seen_samples, negative_samples

    def items_by_popularity(self):
        popularity = Counter()
        self.users = sorted(self.train.keys())
        for user in self.users:
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])

        popularity = dict(popularity)
        popularity = {k: v for k, v in sorted(popularity.items(), key=lambda item: item[1], reverse=True)}
        return popularity


NEGATIVE_SAMPLERS = {
    PopularNegativeSampler.code(): PopularNegativeSampler,
    RandomNegativeSampler.code(): RandomNegativeSampler,
}

def negative_sampler_factory(code, train, val, test, 
                             user_count, item_count,
                             sample_size, seed, flag,
                             save_path):
    negative_sampler = NEGATIVE_SAMPLERS[code]
    return negative_sampler(train, val, test, user_count,
                            item_count, sample_size, seed, 
                            flag, save_path)


if __name__ == '__main__':
    PREP_DATASET_ROOT_FOLDER = 'data/silver'
    FEATURES_ROOT_FOLDER = 'data/gold'
    source_filepath = Path(os.path.join(PREP_DATASET_ROOT_FOLDER, 'ml-1m/dataset.pkl'))
    dataset = pickle.load(source_filepath.open('rb'))
    code = 'random'
    train = dataset['train']
    val = dataset['val']
    test = dataset['test']
    umap = dataset['umap']
    smap = dataset['smap']
    user_count = len(umap)
    item_count = len(smap)
    sample_size = 100
    seed = 0
    flag = 'val'
    save_path = os.path.join(FEATURES_ROOT_FOLDER, 'ml-1m', 'negative_samples')
    negative_sampler = negative_sampler_factory(code, train, val, test, 
                             user_count, item_count,
                             sample_size, seed, flag,
                             save_path)
    _, _ = negative_sampler.get_negative_samples()
```

```python id="4GSf9WmayqgG"
!python ./src/negative_sampling.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="kMkIAJCN4i63" executionInfo={"status": "ok", "timestamp": 1631370818446, "user_tz": -330, "elapsed": 6478, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="235c0c7b-6ea2-475e-ee69-f020887ef335"
!dvc run -n negative_sampling \
          -d src/negative_sampling.py -d data/silver/ml-1m/dataset.pkl \
          -o data/gold/ml-1m/negative_samples \
          python src/negative_sampling.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="pM9lssEo541j" executionInfo={"status": "ok", "timestamp": 1631370829701, "user_tz": -330, "elapsed": 440, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bce59356-5211-4eda-c51b-1731fe771ff4"
!git status -u
```

```python colab={"base_uri": "https://localhost:8080/"} id="pmIx9Y1F5_TD" executionInfo={"status": "ok", "timestamp": 1631364668470, "user_tz": -330, "elapsed": 4427, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7b4fa21c-945e-4cc3-de15-1af95d065867"
!dvc status
```

```python colab={"base_uri": "https://localhost:8080/"} id="tYuj8l5M6URQ" executionInfo={"status": "ok", "timestamp": 1631370897896, "user_tz": -330, "elapsed": 48152, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="96d58604-e82f-4c1c-b465-7125a7363650"
!dvc commit
!dvc push
```

```python colab={"base_uri": "https://localhost:8080/"} id="prWgCZcT6W2_" executionInfo={"status": "ok", "timestamp": 1631370903430, "user_tz": -330, "elapsed": 1544, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3208808d-33d5-4afc-f928-8f11ede46a3b"
!git add .
!git commit -m 'commit'
!git push origin "{branch}"
```

```python id="wqnAUiFQ6nO3"

```
