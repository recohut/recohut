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

<!-- #region id="OX59ktWF29bu" -->
# Adversarial Training (Regularization) on a Recommender System
<!-- #endregion -->

<!-- #region id="bce1rkKF2-Ii" -->
Adversarial training (also adversarial regularization) is a defense strategy against adversarial perturbations. The main intuition is to increase the robustness of a recommender system on minimal adversarial perturbation of model parameters by adding further training iterations that takes into account the application of such perturbations.
<!-- #endregion -->

<!-- #region id="AWUR7Onl3Dfb" -->
In this notebook, we become familiar with the usefulness of the adversarial regularization by:
- Training classical model-based recommender (BPR-MF) on a small part of Movielens-1M
- Attacking the learned model with FGSM-like Adversarial Perturbations
- Adversarially Training the model-based recommender (BPR-MF) with Adversarial Personalized Ranking (APR)
- Attacking the robustified model
<!-- #endregion -->

<!-- #region id="zNLxxtYj35Ry" -->
### Imports
<!-- #endregion -->

```python id="vx6mrrnv-_g_"
!pip install -q timethis
```

```python id="mzYu3R8B37SC"
import numpy as np
import tensorflow as tf
from abc import ABC
from timethis import timethis
```

<!-- #region id="Xf9-n3ii3yaj" -->
### Global settings
<!-- #endregion -->

```python id="QmgqS8C730q6"
np.random.seed(42)
```

<!-- #region id="UaY2poEh3mMX" -->
### Utils
<!-- #endregion -->

```python id="YwPuM5DJ3n_9"
import time
from functools import wraps

# A simple decorator
def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        print(end-start)
        return r
    return wrapper
```

<!-- #region id="5ZUmfAuI3JSF" -->
### Evaluator
<!-- #endregion -->

```python id="FjUioN5J3Xw3"
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import sys
import math
from time import time


_feed_dict = None
_dataset = None
_model = None
_sess = None
_K = None


def _init_eval_model(data):
    global _dataset
    _dataset = data

    pool = Pool(cpu_count() - 1)
    feed_dicts = pool.map(_evaluate_input, range(_dataset.num_users))
    pool.close()
    pool.join()

    return feed_dicts


def _evaluate_input(user):
    # generate items_list
    try:
        test_item = _dataset.test[user][1]
        item_input = set(range(_dataset.num_items)) - set(_dataset.train_list[user])
        if test_item in item_input:
            item_input.remove(test_item)
        item_input = list(item_input)
        item_input.append(test_item)
        user_input = np.full(len(item_input), user, dtype='int32')[:, None]
        item_input = np.array(item_input)[:, None]
        return user_input, item_input
    except:
        print('******' + user)
        return 0, 0


def _eval_by_user(user):
    # get predictions of data in testing set
    user_input, item_input = _feed_dicts[user]
    predictions, _, _ = _model.get_inference(user_input, item_input)

    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict.numpy() >= pos_predict.numpy()).sum()

    # calculate from HR@1 to HR@100, and from NDCG@1 to NDCG@100, AUC
    hr, ndcg, auc = [], [], []
    for k in range(1, _K + 1):
        hr.append(position < k)
        ndcg.append(math.log(2) / math.log(position + 2) if position < k else 0)
        auc.append(
            1 - (position / len(neg_predict)))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]

    return hr, ndcg, auc


class Evaluator:
    def __init__(self, model, data, k):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self.data = data
        self.k = k
        self.eval_feed_dicts = _init_eval_model(data)
        self.model = model

    def eval(self, epoch=0, results={}, epoch_text='', start_time=0):
        """
        Runtime Evaluation of Accuracy Performance (top-k)
        :return:
        """
        global _model
        global _K
        global _dataset
        global _feed_dicts
        _dataset = self.data
        _model = self.model
        _K = self.k
        _feed_dicts = self.eval_feed_dicts

        res = []
        for user in range(self.model.data.num_users):
            res.append(_eval_by_user(user))

        hr, ndcg, auc = (np.array(res).mean(axis=0)).tolist()
        print("%s %.3f Performance@%d \tHR: %.4f\tnDCG: %.4f\tAUC: %.4f" % (
            epoch_text, time() - start_time, _K, hr[_K - 1], ndcg[_K - 1], auc[_K - 1]))

        if len(epoch_text) != '':
            results[epoch] = {'hr': hr, 'ndcg': ndcg, 'auc': auc[0]}

    def store_recommendation(self, attack_name=""):
        """
        Store recommendation list (top-k) in order to be used for the ranksys framework (sisinflab)
        attack_name: The name for the attack stored file
        :return:
        """
        results = self.model.get_full_inference().numpy()
        with open('{0}{1}_best{2}_top{3}_rec.tsv'.format(self.model.path_output_rec_result,
                                                         attack_name + self.model.path_output_rec_result.split('/')[
                                                             -2],
                                                         self.model.best,
                                                         self.k),
                  'w') as out:
            for u in range(results.shape[0]):
                results[u][self.data.train_list[u]] = -np.inf
                top_k_id = results[u].argsort()[-self.k:][::-1]
                top_k_score = results[u][top_k_id]
                for i, value in enumerate(top_k_id):
                    out.write(str(u) + '\t' + str(value) + '\t' + str(top_k_score[i]) + '\n')

    def evaluate(self):
        """
        Runtime Evaluation of Accuracy Performance (top-k)
        """
        global _model
        global _K
        global _dataset
        global _feed_dicts
        _dataset = self.data
        _model = self.model
        _K = self.k
        _feed_dicts = self.eval_feed_dicts

        res = []
        for user in range(self.model.data.num_users):
            res.append(_eval_by_user(user))

        hr, ndcg, auc = (np.array(res).mean(axis=0)).tolist()
        print("Performance@%d\n\tHR: %.4f\tnDCG: %.4f\tAUC: %.4f" % (
            _K, hr[_K - 1], ndcg[_K - 1], auc[_K - 1]))

        return hr[_K - 1], ndcg[_K - 1], auc[_K - 1]
```

<!-- #region id="l3x1_EGH3-FN" -->
## Data
<!-- #endregion -->

<!-- #region id="NgQf4Uvs4Gjk" -->
### Dataloader
<!-- #endregion -->

```python id="A5101miN4J8I"
import scipy.sparse as sp
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import pandas as pd
from scipy.sparse import dok_matrix
from time import time


_user_input = None
_item_input_pos = None
_batch_size = None
_index = None
_model = None
_train = None
_test = None
_num_items = None


def _get_train_batch(i):
    """
    Generation of a batch in multiprocessing
    :param i: index to control the batch generayion
    :return:
    """
    user_batch, item_pos_batch, item_neg_batch = [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_pos_batch.append(_item_input_pos[_index[idx]])
        j = np.random.randint(_num_items)
        while j in _train[_user_input[_index[idx]]]:
            j = np.random.randint(_num_items)
        item_neg_batch.append(j)
    return np.array(user_batch)[:, None], np.array(item_pos_batch)[:, None], np.array(item_neg_batch)[:, None]


class DataLoader(object):
    """
    Load train and test dataset
    """

    def __init__(self, path_train_data, path_test_data):
        """
        Constructor of DataLoader
        :param path_train_data: relative path for train file
        :param path_test_data: relative path for test file
        """
        self.num_users, self.num_items = self.get_length(path_train_data, path_test_data)
        self.load_train_file(path_train_data)
        self.load_train_file_as_list(path_train_data)
        self.load_test_file(path_test_data)
        self._user_input, self._item_input_pos = self.sampling()
        print('{0} - Loaded'.format(path_train_data))
        print('{0} - Loaded'.format(path_test_data))

    def get_length(self, train_name, test_name):
        train = pd.read_csv(train_name, sep='\t', header=None)
        test = pd.read_csv(test_name, sep='\t', header=None)
        try:
            train.columns = ['user', 'item', 'r', 't']
            test.columns = ['user', 'item', 'r', 't']
            data = train.copy()
            data = data.append(test, ignore_index=True)
        except:
            train.columns = ['user', 'item', 'r']
            test.columns = ['user', 'item', 'r']
            data = train.copy()
            data = data.append(test, ignore_index=True)

        return data['user'].nunique(), data['item'].nunique()

    def load_train_file(self, filename):
        """
        Read /data/dataset_name/train file and Return the matrix.
        """
        # Get number of users and items
        # self.num_users, self.num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                # self.num_users = max(self.num_users, u)
                # self.num_items = max(self.num_items, i)
                line = f.readline()

        # Construct URM
        self.train = sp.dok_matrix((self.num_users + 1, self.num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    self.train[user, item] = 1.0
                line = f.readline()

        # self.num_users = self.train.shape[0]
        # self.num_items = self.train.shape[1]

    def load_train_file_as_list(self, filename):
        # Get number of users and items
        u_ = 0
        self.train_list, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    self.train_list.append(items)
                    items = []
                    u_ += 1
                index += 1
                items.append(i)
                line = f.readline()
        self.train_list.append(items)

    def load_test_file(self, filename):
        self.test = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                self.test.append([user, item])
                line = f.readline()

    def sampling(self):
        _user_input, _item_input_pos = [], []
        for (u, i) in self.train.keys():
            # positive instance
            _user_input.append(u)
            _item_input_pos.append(i)
        return _user_input, _item_input_pos

    def shuffle(self, batch_size=512):
        """
        Shuffle dataset to create batch with batch size
        Variable are global to be faster!
        :param batch_size: default 512
        :return: set of all generated random batches
        """
        global _user_input
        global _item_input_pos
        global _batch_size
        global _index
        global _model
        global _train
        global _num_items

        _user_input, _item_input_pos = self._user_input, self._item_input_pos
        _batch_size = batch_size
        _index = list(range(len(_user_input)))
        _train = self.train_list
        _num_items = self.num_items

        np.random.shuffle(_index)
        _num_batches = len(_user_input) // _batch_size
        pool = Pool(cpu_count())
        res = pool.map(_get_train_batch, range(_num_batches))
        pool.close()
        pool.join()

        user_input = [r[0] for r in res]
        item_input_pos = [r[1] for r in res]
        item_input_neg = [r[2] for r in res]
        return user_input, item_input_pos, item_input_neg
```

<!-- #region id="1D9J4pA44iDp" -->
### Download
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yacgk5lF4EaX" executionInfo={"status": "ok", "timestamp": 1633177854810, "user_tz": -330, "elapsed": 972, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="12246c2d-48e7-4676-c26b-e51984b5f64c"
!wget -q --show-progress https://github.com/sisinflab/HandsOn-ECIR2021/raw/master/data/movielens-500/trainingset.tsv
!wget -q --show-progress https://github.com/sisinflab/HandsOn-ECIR2021/raw/master/data/movielens-500/testset.tsv
```

<!-- #region id="2SKLoD2v3-y4" -->
First, we will load a short version of Movielens 1M dataset, which has been pre-processed and stored as a TSV file with the following structure: user_id, item_id, rating, timestamp. We have already divided the dataset in training and test sets using the leave-one-out evaluation protocol. We have used a small version with 500 users to reduce the computation time. To execute with the full dataset, you can change 'movielens-500' with 'movielens'.
<!-- #endregion -->

<!-- #region id="o5wsntg265S_" -->
### Load
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="BZ0aZ2b-4gNj" executionInfo={"status": "ok", "timestamp": 1633177857251, "user_tz": -330, "elapsed": 1613, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8cacb0b5-cc56-4a86-e006-2d69f325ee32"
data = DataLoader(path_train_data='trainingset.tsv', path_test_data='testset.tsv')

print('\nStatistics:\nNumber of Users: {0}\nNumber of Items: {1}\nTraining User-Item Ratings: {2}'.format(data.num_users, data.num_items, len(data.train)))
```

<!-- #region id="9CE7Zo7-62FY" -->
## Model
<!-- #endregion -->

<!-- #region id="1_scRTiS4qP7" -->
### Define the model
<!-- #endregion -->

<!-- #region id="-0hXnqt06Y7p" -->
We will define a new Tensorflow 2 model class to define the model (BPR-MF). For a matter of simplicity we have also implemented the adversarial attack and defense strategies,, that will be used in the later sections.
<!-- #endregion -->

```python id="mnAmNEmQ6hIz"
class RecommenderModel(tf.keras.Model, ABC):
    """
    This class represents a recommender model.
    You can load a pretrained model by specifying its ckpt path and use it for training/testing purposes.
    """

    def __init__(self, data, path_output_rec_result, path_output_rec_weight, rec):
        super(RecommenderModel, self).__init__()
        self.rec = rec
        self.data = data
        self.num_items = data.num_items
        self.num_users = data.num_users
        self.path_output_rec_result = path_output_rec_result
        self.path_output_rec_weight = path_output_rec_weight

    def train(self):
        pass
```

```python id="u9BG_LpS6s5U"
TOPK = 100 # Top-K
```

```python id="0cr8QzL16tF5"
class BPRMF(RecommenderModel):
    def __init__(self, data_loader, path_output_rec_result, path_output_rec_weight):
        super(BPRMF, self).__init__(data_loader, path_output_rec_result, path_output_rec_weight, 'bprmf')
        self.embedding_size = 64
        self.learning_rate = 0.05
        self.reg = 0
        self.epochs = 5
        self.batch_size = 512
        self.verbose = 1
        self.evaluator = Evaluator(self, data, TOPK)

        self.initialize_model_parameters()
        self.initialize_perturbations()
        self.initialize_optimizer()

    def initialize_model_parameters(self):
        """
            Initialize Model Parameters
        """
        self.embedding_P = tf.Variable(tf.random.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01))  # (users, embedding_size)
        self.embedding_Q = tf.Variable(tf.random.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01))  # (items, embedding_size)
        self.h = tf.constant(1.0, tf.float32, [self.embedding_size, 1])

    def initialize_optimizer(self):
        """
            Optimizer
        """
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)

    def initialize_perturbations(self):
        """
            Set delta variables useful to store delta perturbations,
        """
        self.delta_P = tf.Variable(tf.zeros(shape=[self.num_users, self.embedding_size]), trainable=False)
        self.delta_Q = tf.Variable(tf.zeros(shape=[self.num_items, self.embedding_size]), trainable=False)

    def get_inference(self, user_input, item_input_pos):
        """
            Generate Prediction Matrix with respect to passed users and items identifiers
        """
        self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P + self.delta_P, user_input), 1)
        self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q + self.delta_Q, item_input_pos), 1)

        return tf.matmul(self.embedding_p * self.embedding_q,self.h), self.embedding_p, self.embedding_q  # (b, embedding_size) * (embedding_size, 1)

    def get_full_inference(self):
        """
            Get Full Predictions useful for Full Store of Predictions
        """
        return tf.matmul(self.embedding_P + self.delta_P, tf.transpose(self.embedding_Q + self.delta_Q))

    @timethis
    def _train_step(self, batches):
        """
            Apply a Single Training Step (across all the batches in the dataset).
        """
        user_input, item_input_pos, item_input_neg = batches

        for batch_idx in range(len(user_input)):
            with tf.GradientTape() as t:
                t.watch([self.embedding_P, self.embedding_Q])

                # Model Inference
                self.output_pos, embed_p_pos, embed_q_pos = self.get_inference(user_input[batch_idx],
                                                                               item_input_pos[batch_idx])
                self.output_neg, embed_p_neg, embed_q_neg = self.get_inference(user_input[batch_idx],
                                                                               item_input_neg[batch_idx])
                self.result = tf.clip_by_value(self.output_pos - self.output_neg, -80.0, 1e8)

                self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

                # Regularization Component
                self.reg_loss = self.reg * tf.reduce_mean(tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))

                # Loss Function
                self.loss_opt = self.loss + self.reg_loss

            gradients = t.gradient(self.loss_opt, [self.embedding_P, self.embedding_Q])
            self.optimizer.apply_gradients(zip(gradients, [self.embedding_P, self.embedding_Q]))
    
    @timethis
    def train(self):
        for epoch in range(self.epochs):
            batches = self.data.shuffle(self.batch_size)
            self._train_step(batches)
            print('Epoch {0}/{1}'.format(epoch+1, self.epochs))

    @timethis
    def _adversarial_train_step(self, batches, epsilon):
        """
            Apply a Single Training Step (across all the batches in the dataset).
        """
        user_input, item_input_pos, item_input_neg = batches
        adv_reg = 1

        for batch_idx in range(len(user_input)):
            with tf.GradientTape() as t:
                t.watch([self.embedding_P, self.embedding_Q])

                # Model Inference
                self.output_pos, embed_p_pos, embed_q_pos = self.get_inference(user_input[batch_idx],
                                                                               item_input_pos[batch_idx])
                self.output_neg, embed_p_neg, embed_q_neg = self.get_inference(user_input[batch_idx],
                                                                               item_input_neg[batch_idx])
                self.result = tf.clip_by_value(self.output_pos - self.output_neg, -80.0, 1e8)

                self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

                # Regularization Component
                self.reg_loss = self.reg * tf.reduce_mean(tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))

                # Adversarial Regularization Component
                ##  Execute the Adversarial Attack on the Current Model (Perturb Model Parameters)
                self.execute_adversarial_attack(epsilon)
                ##  Inference on the Adversarial Perturbed Model
                self.output_pos_adver, _, _ = self.get_inference(user_input[batch_idx], item_input_pos[batch_idx])
                self.output_neg_adver, _, _ = self.get_inference(user_input[batch_idx], item_input_neg[batch_idx])

                self.result_adver = tf.clip_by_value(self.output_pos_adver - self.output_neg_adver, -80.0, 1e8)
                self.loss_adver = tf.reduce_sum(tf.nn.softplus(-self.result_adver))

                # Loss Function
                self.adversarial_regularizer = adv_reg * self.loss_adver # AMF = Adversarial Matrix Factorization
                self.bprmf_loss = self.loss + self.reg_loss

                self.amf_loss = self.bprmf_loss + self.adversarial_regularizer

            gradients = t.gradient(self.amf_loss, [self.embedding_P, self.embedding_Q])
            self.optimizer.apply_gradients(zip(gradients, [self.embedding_P, self.embedding_Q]))

        self.initialize_perturbations()


    @timethis
    def adversarial_train(self, adversarial_epochs, epsilon):
        for epoch in range(adversarial_epochs):
            batches = self.data.shuffle(self.batch_size)
            self._adversarial_train_step(batches, epsilon)
            print('Epoch {0}/{1}'.format(self.epochs+epoch+1, self.epochs+adversarial_epochs))

    def execute_adversarial_attack(self, epsilon):
        user_input, item_input_pos, item_input_neg = self.data.shuffle(len(self.data._user_input))
        self.initialize_perturbations()

        with tf.GradientTape() as tape_adv:
            tape_adv.watch([self.embedding_P, self.embedding_Q])
            # Evaluate Current Model Inference
            output_pos, embed_p_pos, embed_q_pos = self.get_inference(user_input[0],
                                                                      item_input_pos[0])
            output_neg, embed_p_neg, embed_q_neg = self.get_inference(user_input[0],
                                                                      item_input_neg[0])
            result = tf.clip_by_value(output_pos - output_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))
            loss += self.reg * tf.reduce_mean(
                tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))
        # Evaluate the Gradient
        grad_P, grad_Q = tape_adv.gradient(loss, [self.embedding_P, self.embedding_Q])
        grad_P, grad_Q = tf.stop_gradient(grad_P), tf.stop_gradient(grad_Q)

        # Use the Gradient to Build the Adversarial Perturbations (https://doi.org/10.1145/3209978.3209981)
        self.delta_P = tf.nn.l2_normalize(grad_P, 1) * epsilon
        self.delta_Q = tf.nn.l2_normalize(grad_Q, 1) * epsilon
```

<!-- #region id="k8gvf2v46wCD" -->
## Initialize and Train The Model
<!-- #endregion -->

```python id="PJg3jn7h7Grs"
!mkdir -p rec_result rec_weights
```

```python colab={"base_uri": "https://localhost:8080/"} id="HyKeZ_8n7CHH" executionInfo={"status": "ok", "timestamp": 1633178123010, "user_tz": -330, "elapsed": 15956, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="92e55739-b1b2-4630-ed15-ac52fe2cd013"
recommender_model = BPRMF(data, 'rec_result/', 'rec_weights/')
recommender_model.train()
```

<!-- #region id="gIYX8PnS7JuW" -->
## Evaluate The Model
<!-- #endregion -->

<!-- #region id="XFWdk-ol8EwX" -->
The evaluation is computed on TOPK recommendation lists (default K = 100).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4qy6ACNxAPB9" executionInfo={"status": "ok", "timestamp": 1633178535451, "user_tz": -330, "elapsed": 2172, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="99d4a990-7ce4-4f55-b1f0-0c6636bcdbe0"
before_adv_hr, before_adv_ndcg, before_adv_auc = recommender_model.evaluator.evaluate()
```

<!-- #region id="EH4cOZeZBprW" -->
## Adversarial Attack Against The Model
We can attack the model with adversarial perturbation and measure the performance after the attack. Epsilon is the perturbation budget.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="YRpVSN56By9G" executionInfo={"status": "ok", "timestamp": 1633178576310, "user_tz": -330, "elapsed": 1442, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="07ca572d-ee39-4f0c-e5bf-a0550e1353a5"
epsilon = 0.5
print('Running the Attack with Epsilon = {0}'.format(epsilon))
recommender_model.execute_adversarial_attack(epsilon=epsilon)
print('The model has been Adversarially Perturbed.')
```

<!-- #region id="KqsPksbpB1yQ" -->
## Evaluate the Effects of the Adversarial Attack
We will now evaluate the performance of the attacked model.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="P5oI0_wWB9ft" executionInfo={"status": "ok", "timestamp": 1633178610580, "user_tz": -330, "elapsed": 2156, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d8abbeb9-1aea-4c2e-c41f-41123991072d"
after_adv_hr, after_adv_ndcg, after_adv_auc = recommender_model.evaluator.evaluate()

print('HR decreases by %.2f%%' % ((1-after_adv_hr/before_adv_hr)*100))
print('nDCG decreases by %.2f%%' % ((1-after_adv_ndcg/before_adv_ndcg)*100))
print('AUC decreases by %.2f%%' % ((1-after_adv_auc/before_adv_auc)*100))
```

<!-- #region id="d0QyTKufB953" -->
## Implement The Adversarial Training/Regularization
We have identified the clear performance degradation of the recommender under adversarial attack. Now, we can test whether the adversarial regularization will make the model more robust.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4hJLvP6jCQ--" executionInfo={"status": "ok", "timestamp": 1633178897616, "user_tz": -330, "elapsed": 189048, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="3d6c782b-4b98-47c0-a50c-259990856cb6"
recommender_model.adversarial_train(adversarial_epochs=1, epsilon=epsilon)
```

<!-- #region id="mst7p8IFCVji" -->
## Evaluated The Adversarially Defended Model before the Attack
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="N2kEcR6ZCZ2M" executionInfo={"status": "ok", "timestamp": 1633178957304, "user_tz": -330, "elapsed": 2236, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bcf622e1-55f4-4e2a-96c1-a3c51ab39084"
before_adv_hr, before_adv_ndcg, before_adv_auc = recommender_model.evaluator.evaluate()
```

<!-- #region id="nCLmbcDmCYP-" -->
## Adversarial Attack Against The Defended Model
<!-- #endregion -->

```python id="RcPRL_1aCi-Q"
recommender_model.execute_adversarial_attack(epsilon=epsilon)
```

<!-- #region id="OleMekZjCiBb" -->
##Evaluate the Effects of the Adversarial Attack against the Defended Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="TlxieDX-Cke3" executionInfo={"status": "ok", "timestamp": 1633178967878, "user_tz": -330, "elapsed": 2568, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ad7b6166-566f-486b-a2f6-064da29fc946"
after_adv_hr, after_adv_ndcg, after_adv_auc = recommender_model.evaluator.evaluate()

print('HR decreases by %.2f%%' % ((1-after_adv_hr/before_adv_hr)*100))
print('nDCG decreases by %.2f%%' % ((1-after_adv_ndcg/before_adv_ndcg)*100))
print('AUC decreases by %.2f%%' % ((1-after_adv_auc/before_adv_auc)*100))
```

<!-- #region id="L-mWToBfCMq8" -->
## Consequences
At this point, we have seen that the adversarial training has been effective in reducing the effectiveness of the FGSM-based adversarial attack against the recommender model. Furthermore, we have also identified another important consequences of the adversarial regularization. If we compare the performance of the model before and after the attack we can identify that there has been a performance improvement. For this reason, several recent works have implemented this robustification technique as an additional model component to increase the accuracy power of the recommender model.
<!-- #endregion -->

<!-- #region id="FVxushStCH_O" -->
## References
- Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme: BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009: 452-461
- Xiangnan He, Zhankui He, Xiaoyu Du, Tat-Seng Chua: Adversarial Personalized Ranking for Recommendation. SIGIR 2018: 355-364
- https://github.com/merrafelice/HandsOn-RecSys2020
<!-- #endregion -->
