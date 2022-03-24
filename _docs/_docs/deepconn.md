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

<!-- #region id="2VZ8fr0qMt5t" -->
# DeepCoNN on Amazon Music Instruments in PyTorch
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="L60wRz3KLutF" executionInfo={"status": "ok", "timestamp": 1641537515379, "user_tz": -330, "elapsed": 1220, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2e592bdd-5451-4999-9b69-86d8fefaeaad"
!wget -q --show-progress https://github.com/RecoHut-Datasets/amazon_music_instruments/raw/v1/Musical_Instruments_5.json
```

```python colab={"base_uri": "https://localhost:8080/"} id="pgm_ixhCRjdt" executionInfo={"status": "ok", "timestamp": 1641537766384, "user_tz": -330, "elapsed": 1239, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="219fb486-f23d-4f65-88ca-bc72002fb508"
!wget -q --show-progress https://github.com/huangjunheng/recommendation_model/raw/master/DeepCoNN/data/embedding_data/stopwords.txt
!wget -q --show-progress https://github.com/huangjunheng/recommendation_model/raw/master/DeepCoNN/data/embedding_data/punctuations.txt
```

```python colab={"base_uri": "https://localhost:8080/"} id="vcYWhWrlR0Na" executionInfo={"status": "ok", "timestamp": 1641537933311, "user_tz": -330, "elapsed": 95967, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0ca55c5a-3290-4965-e73f-dfb1904343bf"
!wget -q --show-progress -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
!gzip -d GoogleNews-vectors-negative300.bin.gz
```

<!-- #region id="_tqBBPOQQySy" -->
## Preprocessing
<!-- #endregion -->

```python id="vt6fSpGqQ2La"
import numpy as np
import pandas as pd
import torch
import nltk
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import pickle
from gensim.models.keyedvectors import Word2VecKeyedVectors
```

```python colab={"base_uri": "https://localhost:8080/"} id="2LL5FMAARKw6" executionInfo={"status": "ok", "timestamp": 1641538034166, "user_tz": -330, "elapsed": 1545, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d2529bc2-b793-4cf7-b3cb-ad70bd0285a8"
nltk.download('wordnet')
```

```python id="fhuePKLCQ32f"
path = '/content/'
PAD_WORD = '<pad>'
PAD_WORD_ID = 3000000
WORD_EMBEDDINF_SIZE = 300
```

```python colab={"base_uri": "https://localhost:8080/"} id="vSPtZeoJQ6cz" outputId="3bb94082-1307-4326-a4c8-cbc483f6339e"
def process_raw_data(in_path, out_path):
    df = pd.read_json(in_path, lines=True)
    df = df[['reviewerID', 'asin', 'reviewText', 'overall']]
    df.columns = ['userID', 'itemID', 'review', 'rating']

    df['userID'] = df.groupby(df['userID']).ngroup()
    df['itemID'] = df.groupby(df['itemID']).ngroup()

    with open('stopwords.txt') as f:
        stop_words = set(f.read().splitlines())

    with open('punctuations.txt') as f:
        punctuations = set(f.read().splitlines())

    def clean_review(review):
        lemmatizer = nltk.WordNetLemmatizer()
        review = review.lower()
        for p in punctuations:
            review = review.replace(p, ' ')
        tokens = review.split()
        tokens = [word for word in tokens if word not in stop_words]
        # 词形归并 词干提取
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    df['review'] = df['review'].apply(clean_review)
    df.to_json(out_path, orient='records', lines=True)


def get_word_vec():
    # 加载预训练词嵌入模型
    in_path = 'GoogleNews-vectors-negative300.bin'
    out_path = path + 'embedding_weight.pt'
    word_vec = KeyedVectors.load_word2vec_format(in_path, binary=True)
    word_vec.add([PAD_WORD], np.zeros([1, 300]))

    # 保存预训练模型为tensor格式， 以便于后续训练
    weight = torch.Tensor(word_vec.vectors)
    torch.save(weight, out_path)
    return word_vec


def load_embedding_weights(in_path=path + 'embedding_weight.pt'):
    return torch.load(in_path)


def get_reviews_in_idx(data, word_vec):
    def review2wid(review):
        wids = []
        for word in review.split():
            if word in word_vec:
                wid = word_vec.vocab[word].index
            else:
                wid = word_vec.vocab[PAD_WORD].index
            wids.append(wid)
        return wids

    data['review'] = data['review'].apply(review2wid)
    review_by_user = dict(list(data[['itemID', 'review']].groupby(data['userID'])))
    review_by_item = dict(list(data[['userID', 'review']].groupby(data['itemID'])))
    return review_by_user, review_by_item


def get_max_review_length(data, percentile=0.85):
    review_lengths = data['review'].apply(lambda review: len(review.split()))
    max_length = int(review_lengths.quantile(percentile, interpolation='lower'))
    return max_length


def get_max_review_count(data, percentile=0.85):
    review_count_user = data['review'].groupby(data['userID']).count()
    review_count_user = int(review_count_user.quantile(percentile, interpolation='lower'))

    review_count_item = data['review'].groupby(data['itemID']).count()
    review_count_item = int(review_count_item.quantile(percentile, interpolation='lower'))

    return max(review_count_user, review_count_item)


def get_max_user_id(data):
    return max(data['userID'])


def get_max_item_id(data):
    return max(data['itemID'])


def save_review_dict(data, word_vec, data_type):
    user_review, item_review = get_reviews_in_idx(data, word_vec)
    pickle.dump(user_review, open(path + 'user_review_word_idx_{}.p'.format(data_type), 'wb'))
    pickle.dump(item_review, open(path + 'item_review_word_idx_{}.p'.format(data_type), 'wb'))


def get_review_dict(data_type):
    user_review = pickle.load(open(path + 'user_review_word_idx_{}.p'.format(data_type), 'rb'))
    item_review = pickle.load(open(path + 'item_review_word_idx_{}.p'.format(data_type), 'rb'))
    return user_review, item_review


def main():
    process_raw_data(path + 'Musical_Instruments_5.json', path + 'reviews.json')
    df = pd.read_json(path + 'reviews.json', lines=True)
    train, test = train_test_split(df, test_size=0.2, random_state=3)
    train, dev = train_test_split(train, test_size=0.2, random_state=4)
    known_data = pd.concat([train, dev], axis=0)
    all_data = pd.concat([train, dev, test], axis=0)

    print('max review length is {}'.format(get_max_review_length(all_data)))
    print('max review count is {}'.format(get_max_review_count(all_data)))
    print('max user id is {}'.format(get_max_user_id(all_data)))
    print('max item id is {}'.format(get_max_item_id(all_data)))

    word_vec = get_word_vec()

    save_review_dict(known_data, word_vec, 'train')
    save_review_dict(all_data, word_vec, 'test')


if __name__ == '__main__':
    main()
```
