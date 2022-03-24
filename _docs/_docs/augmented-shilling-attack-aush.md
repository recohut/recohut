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

```python colab={"base_uri": "https://localhost:8080/"} id="v2XUbj7yX7v2" executionInfo={"status": "ok", "timestamp": 1633100494236, "user_tz": -330, "elapsed": 825, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8b4095ca-70a2-4384-964f-90291abf667c"
%tensorflow_version 1.x
```

```python colab={"base_uri": "https://localhost:8080/"} id="pdpN0eIdYyT5" executionInfo={"status": "ok", "timestamp": 1633100740620, "user_tz": -330, "elapsed": 27620, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d2a75b8d-6abd-4711-bfd4-e3b39fc4a86d"
!pip install -q surprise
```

```python id="f0PJLz1SXem7"
%%writefile example.sh
for target_id in 5 395 181 565 254 601 623 619 64 558
do
	for rec_model_name in IAUtoRec UAUtoRec NNMF NMF_25
	do
		python main_eval_attack.py --dataset filmTrust --rec_model_name $rec_model_name --attack_method G0 --target_id $target_id --attack_num 50 --filler_num 36 >> filmTrust_result_G0
		#nohup python main_gan_attack_baseline.py --dataset filmTrust --target_id 5 --attack_num 50 --filler_num 36 --loss 0 >> G0_log 2>&1 &
	done
done
```

```python id="Dt2Y8carX2NZ" executionInfo={"status": "ok", "timestamp": 1633101111117, "user_tz": -330, "elapsed": 678, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import time
import numpy as np
import scipy
import math
import os
import shutil
import pandas as pd
from scipy.sparse import csr_matrix
from six.moves import xrange
import random
import copy
import itertools
import gzip
import sys, argparse
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.framework import ops

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)
```

```python id="_txSkOEiaKYc" executionInfo={"status": "ok", "timestamp": 1633101083278, "user_tz": -330, "elapsed": 746, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter
```

```python id="vGifuYTUZu_r"
class load_data():

    def __init__(self, path_train, path_test,
                 header=None, sep='\t', threshold=4, print_log=True):
        self.path_train = path_train
        self.path_test = path_test
        self.header = header if header is not None else ['user_id', 'item_id', 'rating']
        self.sep = sep
        self.threshold = threshold
        self.print_log = print_log

        self._main_load()

    def _main_load(self):
        # load data 得到用户总数，item总数，dataframe格式的train,test,train_without vali,validate
        self._load_file()
        #
        # dataframe to matrix
        self.train_matrix, self.train_matrix_implicit = self._data_to_matrix(self.train_data)
        self.test_matrix, self.test_matrix_implicit = self._data_to_matrix(self.test_data)

    def _load_file(self):
        if self.print_log:
            print("load train/test data\t:\n", self.path_train)
        self.train_data = pd.read_csv(self.path_train, sep=self.sep, names=self.header, engine='python').loc[:,
                          ['user_id', 'item_id', 'rating']]
        self.test_data = pd.read_csv(self.path_test, sep=self.sep, names=self.header, engine='python').loc[:,
                         ['user_id', 'item_id', 'rating']]
        # 不能保证每个item都有在训练集里出现
        self.n_users = len(set(self.test_data.user_id.unique()) | set(self.train_data.user_id.unique()))
        self.n_items = len(set(self.test_data.item_id.unique()) | set(self.train_data.item_id.unique()))

        if self.print_log:
            print("Number of users:", self.n_users, ",Number of items:", self.n_items, flush=True)
            print("Train size:", self.train_data.shape[0], ",Test size:", self.test_data.shape[0], flush=True)

    def _data_to_matrix(self, data_frame):
        row, col, rating, implicit_rating = [], [], [], []
        for line in data_frame.itertuples():
            uid, iid, r = list(line)[1:]
            implicit_r = 1 if r >= self.threshold else 0

            row.append(uid)
            col.append(iid)
            rating.append(r)
            implicit_rating.append(implicit_r)

        matrix = csr_matrix((rating, (row, col)), shape=(self.n_users, self.n_items))
        matrix_implicit = csr_matrix((implicit_rating, (row, col)), shape=(self.n_users, self.n_items))
        return matrix, matrix_implicit

    def get_global_mean_std(self):
        return self.train_matrix.data.mean(), self.train_matrix.data.std()

    def get_all_mean_std(self):
        flag = 1
        for v in ['global_mean', 'global_std', 'item_means', 'item_stds']:
            if not hasattr(self, v):
                flag = 0
                break
        if flag == 0:
            global_mean, global_std = self.get_global_mean_std()
            item_means, item_stds = [global_mean] * self.n_items, [global_std] * self.n_items
            train_matrix_t = self.train_matrix.transpose()
            for iid in range(self.n_items):
                item_vec = train_matrix_t.getrow(iid).toarray()[0]
                ratings = item_vec[np.nonzero(item_vec)]
                if len(ratings) > 0:
                    item_means[iid], item_stds[iid] = ratings.mean(), ratings.std()
            self.global_mean, self.global_std, self.item_means, self.item_stds \
                = global_mean, global_std, item_means, item_stds
        return self.global_mean, self.global_std, self.item_means, self.item_stds

    def get_item_pop(self):
        # item_pops = [0] * self.n_items
        # train_matrix_t = self.train_matrix.transpose()
        # for iid in range(self.n_items):
        #     item_vec = train_matrix_t.getrow(iid).toarray()[0]
        #     item_pops[iid] = len(np.nonzero(item_vec)[0])
        item_pops_dict = dict(self.train_data.groupby('item_id').size())
        item_pops = [0] * self.n_items
        for iid in item_pops_dict.keys():
            item_pops[iid] = item_pops_dict[iid]
        return item_pops

    def get_user_nonrated_items(self):
        non_rated_indicator = self.train_matrix.toarray()
        non_rated_indicator[non_rated_indicator > 0] = 1
        non_rated_indicator = 1 - non_rated_indicator
        user_norated_items = {}
        for uid in range(self.n_users):
            user_norated_items[uid] = list(non_rated_indicator[uid].nonzero()[0])
        return user_norated_items

    def get_item_nonrated_users(self, item_id):
        item_vec = np.squeeze(self.train_matrix[:, item_id].toarray())
        # item_vec = self.train_matrix.toarray().transpose()[item_id]
        item_vec[item_vec > 0] = 1
        non_rated_indicator = 1 - item_vec
        return list(non_rated_indicator.nonzero()[0])
```

```python id="V7I7rQQOZvC5"
def load_attack_info(seletced_item_path, target_user_path):
    attack_info = {}
    with open(seletced_item_path, "r") as fin:
        for line in fin:
            line = line.strip("\n").split("\t")
            target_item, selected_items = int(line[0]), list(map(int, line[1].split(",")))
            attack_info[target_item] = [selected_items]
    with open(target_user_path, "r") as fin:
        for line in fin:
            line = line.strip("\n").split("\t")
            target_item, target_users = int(line[0]), list(map(int, line[1].split(",")))
            attack_info[target_item].append(target_users)
    return attack_info


def attacked_file_writer(clean_path, attacked_path, fake_profiles, n_users_ori):
    data_to_write = ""
    i = 0
    for fake_profile in fake_profiles:
        injected_iid = fake_profile.nonzero()[0]
        injected_rating = fake_profile[injected_iid]
        data_to_write += ('\n'.join(
            map(lambda x: '\t'.join(map(str, [n_users_ori + i] + list(x))), zip(injected_iid, injected_rating))) + '\n')
        i += 1
    if os.path.exists(attacked_path): os.remove(attacked_path)
    shutil.copyfile(clean_path, attacked_path)
    with open(attacked_path, 'a+')as fout:
        fout.write(data_to_write)


def target_prediction_writer(predictions, hit_ratios, dst_path):
    # uid - rating - HR
    data_to_write = []
    for uid in range(len(predictions)):
        data_to_write.append('\t'.join(map(str, [uid, predictions[uid]] + hit_ratios[uid])))
    with open(dst_path, 'w')as fout:
        fout.write('\n'.join(data_to_write))
```

```python id="YTNUTKooikPi"
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def data_preprocess(data_set, gz_path):
    data = getDF(gz_path)[['reviewerID', 'asin', 'overall']]
    data.columns = ['uid', 'iid', 'rating']
    # 数据统计
    uids, iids = data.uid.unique(), data.iid.unique()
    n_uids, n_iids, n_ratings = len(uids), len(iids), data.shape[0]
    print('用户数:', n_uids, '\t物品数:', n_iids, '\t评分数:', n_ratings, '\t Sparsity :', n_ratings / (n_iids * n_uids))
    print('用户平均评分数:', n_ratings / n_uids)
    # id转换
    uid_update = dict(zip(uids, range(n_uids)))
    iid_update = dict(zip(iids, range(n_iids)))

    data.uid = data.uid.apply(lambda x: uid_update[x])
    data.iid = data.iid.apply(lambda x: iid_update[x])
    # 数据集划分
    train_idxs, test_idxs = train_test_split(list(range(n_ratings)), test_size=0.1)
    # 结果保存
    train_data = data.iloc[train_idxs]
    test_data = data.iloc[test_idxs]
    path_train = "../data/data/" + data_set + "_train.dat"
    path_test = "../data/data/" + data_set + "_test.dat"
    train_data.to_csv(path_train, index=False, header=None, sep='\t')
    test_data.to_csv(path_test, index=False, header=None, sep='\t')
    np.save("../data/data/" + data_set + "_id_update", [uid_update, iid_update])


def exp_select(data_set, target_items, selected_num, target_user_num):
    path_test = "../data/data/" + data_set + "_test.dat"
    path_train = "../data/data/" + data_set + "_train.dat"
    dataset_class = load_data(path_train=path_train, path_test=path_test,
                              header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=True)
    # 物品流行度
    item_pops = dataset_class.get_item_pop()
    # 物品id按照流行度降序排列
    # TODO 可以参考按照这个选择selected item，比如[item_pops[items_sorted[i*len(items_sorted)//20]] for i in range(5)]
    items_sorted = np.array(item_pops).argsort()[::-1]
    """1.bandwagon攻击方法，每个目标的selcted都是全局热门top3"""
    bandwagon_selected = items_sorted[:selected_num]
    print('bandwagon_selected:', bandwagon_selected)

    """2.segment攻击方法，每个目标的selcted都是全局热门topN中随机组合"""
    threshold = dataset_class.test_data.rating.mean()
    threshold = threshold if threshold < 3 else 3.0
    print('高分阈值:', threshold)
    selected_candidates = items_sorted[:20]
    # 排列组合
    selected_candidates = list(itertools.combinations(selected_candidates, selected_num))

    result = {}
    target_items = [j for i in range(2, 10) for j in
                    items_sorted[i * len(items_sorted) // 10:(i * len(items_sorted) // 10) + 2]][::-1]
    target_items = list(
        np.random.choice([i for i in range(len(item_pops)) if item_pops[i] == 3], 4, replace=False)) + target_items
    print('target_items:', target_items)
    print('评分数:', [item_pops[i] for i in target_items])
    for target in target_items:
        target_rated = set(dataset_class.train_data[dataset_class.train_data.item_id == target].user_id.values)
        data_tmp = dataset_class.train_data[~dataset_class.train_data.user_id.isin(target_rated)].copy()
        data_tmp = data_tmp[data_tmp.rating >= threshold]
        np.random.shuffle(selected_candidates)
        # 目标用户硬约束，要求对每个selected都有评分
        for selected_items in selected_candidates:
            target_users = data_tmp[data_tmp.item_id.isin(selected_items)].groupby(
                'user_id').size()
            # 对selected_items都有评分
            if target_users[(target_users == selected_num)].shape[0] >= target_user_num:
                target_users = sorted(target_users[(target_users == selected_num)].index)
                result[target] = [sorted(selected_items), target_users]
                print('target:', target, '硬约束')
                break
        # 硬约束找不到则执行，软约束，部分有评分
        if target not in result:
            for selected_items in selected_candidates:
                # 对selected_items有评分
                target_users = data_tmp[data_tmp.item_id.isin(selected_items)].groupby(
                    'user_id').size()
                target_users = sorted(dict(target_users).items(), key=lambda x: x[1], reverse=True)
                min = target_users[target_user_num][1]
                target_users = [i[0] for i in target_users[:target_user_num] if i[1] > selected_num // 2]
                if len(target_users) >= target_user_num:
                    result[target] = [sorted(selected_items), sorted(target_users)]
                    print('target:', target, '软约束,最少评selected数目：', min)
                    break
        # 无目标用户
        if target not in result:
            print('target:', target, '无目标用户')
            a = 1
    """3.存result"""
    key = list(result.keys())
    selected_items = [','.join(map(str, result[k][0])) for k in key]
    target_users = [','.join(map(str, result[k][1])) for k in key]
    selected_items = pd.DataFrame(dict(zip(['id', 'selected_items'], [key, selected_items])))
    target_users = pd.DataFrame(dict(zip(['id', 'target_users'], [key, target_users])))
    selected_items.to_csv("../data/data/" + data_set + '_selected_items', index=False, header=None, sep='\t')
    target_users.to_csv("../data/data/" + data_set + '_target_users', index=False, header=None, sep='\t')


if __name__ == '__main__':
    data_set = 'office'
    gz_path = 'C:\\Users\\ariaschen\\Downloads\\reviews_Office_Products_5.json.gz'

    """step1:数据统计+格式转换"""
    data_preprocess(data_set, gz_path)
    """# step2:选攻击目标,以及每个目标的selected items和目标用户"""
    target_items = None
    # selselected_num和target_user_num是为每个攻击目标选择多少个selected_items和多少个目标用户，默认为3和50
    # 但可能会遇到宣布不够个数的情况，处理办法（1）换攻击目标（2）参数调小,比如这个我改为selected_num=2, target_user_num=30
    exp_select(data_set, target_items, selected_num=2, target_user_num=30)
```

```python cellView="form" id="x2OBX_scYRsg"
#@markdown class NNMF()
class NNMF():
    def __init__(self, sess, dataset_class, num_factor_1=100, num_factor_2=10, hidden_dimension=50,
                 learning_rate=0.001, reg_rate=0.01, epoch=500, batch_size=256,
                 show_time=False, T=5, display_step=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.dataset_class = dataset_class
        self.num_user = dataset_class.n_users
        self.num_item = dataset_class.n_items
        self.dataset_class.test_matrix_dok = self.dataset_class.test_matrix.todok()

        self.num_factor_1 = num_factor_1
        self.num_factor_2 = num_factor_2
        self.hidden_dimension = hidden_dimension
        self.show_time = show_time
        self.T = T
        self.display_step = display_step
        print("NNMF.")

        self.dataset_class_train_matrix_coo = self.dataset_class.train_matrix.tocoo()
        self.user = self.dataset_class_train_matrix_coo.row.reshape(-1)
        self.item = self.dataset_class_train_matrix_coo.col.reshape(-1)
        self.rating = self.dataset_class_train_matrix_coo.data

        self._build_network()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _build_network(self):
        print("num_factor_1=%d, num_factor_2=%d, hidden_dimension=%d" % (
            self.num_factor_1, self.num_factor_2, self.hidden_dimension))

        # model dependent arguments
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder("float", [None], 'rating')
        # latent feature vectors
        P = tf.Variable(tf.random_normal([self.num_user, self.num_factor_1], stddev=0.01))
        Q = tf.Variable(tf.random_normal([self.num_item, self.num_factor_1], stddev=0.01))
        # latent feature matrix(K=1?)
        U = tf.Variable(tf.random_normal([self.num_user, self.num_factor_2], stddev=0.01))
        V = tf.Variable(tf.random_normal([self.num_item, self.num_factor_2], stddev=0.01))

        input = tf.concat(values=[tf.nn.embedding_lookup(P, self.user_id),
                                  tf.nn.embedding_lookup(Q, self.item_id),
                                  tf.multiply(tf.nn.embedding_lookup(U, self.user_id),
                                              tf.nn.embedding_lookup(V, self.item_id))
                                  ], axis=1)
        #
        # tf1->tf2
        # regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_rate)
        regularizer = tf.keras.regularizers.l2(self.reg_rate)
        layer_1 = tf.layers.dense(inputs=input, units=2 * self.num_factor_1 + self.num_factor_2,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer, activation=tf.sigmoid,
                                  kernel_regularizer=regularizer)
        layer_2 = tf.layers.dense(inputs=layer_1, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        layer_3 = tf.layers.dense(inputs=layer_2, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        layer_4 = tf.layers.dense(inputs=layer_3, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        output = tf.layers.dense(inputs=layer_4, units=1, activation=None,
                                 bias_initializer=tf.random_normal_initializer,
                                 kernel_initializer=tf.random_normal_initializer,
                                 kernel_regularizer=regularizer)
        self.pred_rating = tf.reshape(output, [-1])
        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_rating)) \
                    + tf.losses.get_regularization_loss() + self.reg_rate * (
                            tf.norm(U) + tf.norm(V) + tf.norm(P) + tf.norm(Q))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):
        self.num_training = len(self.rating)
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(self.user[idxs])
        item_random = list(self.item[idxs])
        rating_random = list(self.rating[idxs])
        # train
        for i in range(total_batch):
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.user_id: batch_user,
                                                                            self.item_id: batch_item,
                                                                            self.y: batch_rating
                                                                            })
        return loss

    def test(self, test_data):
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict([u], [i])[0]
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        rmse = np.sqrt(error / len(test_set))
        mae = error_mae / len(test_set)
        return rmse, mae

    def execute(self):
        loss_prev = float("inf")
        for epoch in range(self.epochs):
            loss_cur = self.train()
            if epoch % self.T == 0:
                print("epoch:\t", epoch, "\tloss:\t", loss_cur)
            if abs(loss_cur - loss_prev) < math.exp(-5):
                break
            loss_prev = loss_cur
        rmse, mae = self.test(self.dataset_class.test_matrix_dok)
        print("training done\tRMSE : ", rmse, "\tMAE : ", mae)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore(self, path):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def predict(self, user_id, item_id):
        if type(item_id) != list:
            item_id = [item_id]
        if type(user_id) != list:
            user_id = [user_id] * len(item_id)
        return self.sess.run([self.pred_rating], feed_dict={self.user_id: user_id, self.item_id: item_id})[0]
```

```python cellView="form" id="s8wYiC7MYHG8"
#@markdown class IAutoRec()
class IAutoRec():
    def __init__(self, sess, dataset_class, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=500,
                 hidden_neuron=500, verbose=False, T=5, display_step=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.hidden_neuron = hidden_neuron
        self.sess = sess
        self.dataset_class = dataset_class
        self.num_user = dataset_class.n_users
        self.num_item = dataset_class.n_items
        self.dataset_class.test_matrix_dok = self.dataset_class.test_matrix.todok()
        self.verbose = verbose
        self.T = T
        self.display_step = display_step

        self.train_data = self.dataset_class.train_matrix.toarray()
        self.train_data_mask = scipy.sign(self.train_data)

        print("IAutoRec.",end=' ')
        self._build_network()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _build_network(self):
        # placeholder
        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.keep_rate_net = tf.placeholder(tf.float32)
        self.keep_rate_input = tf.placeholder(tf.float32)
        # Variable
        V = tf.Variable(tf.random_normal([self.hidden_neuron, self.num_user], stddev=0.01))
        W = tf.Variable(tf.random_normal([self.num_user, self.hidden_neuron], stddev=0.01))
        mu = tf.Variable(tf.random_normal([self.hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_user], stddev=0.01))
        layer_1 = tf.nn.dropout(tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix)),
                                self.keep_rate_net)
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
                            tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):
        self.num_training = self.num_item
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        loss = float('inf')
        for i in range(total_batch):
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={
                                        self.rating_matrix: self.dataset_class.train_matrix[:, batch_set_idx].toarray(),
                                        self.rating_matrix_mask: scipy.sign(
                                            self.dataset_class.train_matrix[:, batch_set_idx].toarray()),
                                        self.keep_rate_net: 1
                                    })  # 0.95
        return loss

    def test(self, test_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.keep_rate_net: 1})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.reconstruction[u, i]  # self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        rmse = np.sqrt(error / len(test_set))
        mae = error_mae / len(test_set)
        return rmse, mae

    def execute(self):
        loss_prev = float("inf")
        for epoch in range(self.epochs):
            loss_cur = self.train()
            # if epoch % self.T == 0:
                # print("epoch:\t", epoch, "\tloss:\t", loss_cur)
            if abs(loss_cur - loss_prev) < math.exp(-5):
                break
            loss_prev = loss_cur
        rmse, mae = self.test(self.dataset_class.test_matrix_dok)
        print("training done\tRMSE : ", rmse, "\tMAE : ", mae)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore(self, path):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def predict(self, user_id, item_id):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.keep_rate_net: 1})
        return self.reconstruction[user_id, item_id]
        # if not hasattr(self, 'reconstruction_all'):
        #     self.reconstruction_all = self.sess.run(self.layer_2,
        #                                             feed_dict={self.rating_matrix: self.train_data,
        #                                                        self.rating_matrix_mask: self.train_data_mask,
        #                                                        self.keep_rate_net: 1})
        # return self.reconstruction_all[user_id, item_id]
```

```python cellView="form" id="d4JUb7ctYeFn"
#@markdown class UAutoRec()
class UAutoRec():
    def __init__(self, sess, dataset_class, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=200,
                 hidden_neuron=500, verbose=False, T=5, display_step=1000, layer=1):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.hidden_neuron = hidden_neuron
        self.sess = sess
        self.dataset_class = dataset_class
        self.num_user = dataset_class.n_users
        self.num_item = dataset_class.n_items
        self.dataset_class.test_matrix_dok = self.dataset_class.test_matrix.todok()
        self.verbose = verbose
        self.T = T
        self.display_step = display_step
        print("UAutoRec.")
        # 评分矩阵是IAutoRec的转置
        self.train_data = self.dataset_class.train_matrix.toarray().transpose()
        self.train_data_mask = scipy.sign(self.train_data)

        self.layer = layer

        self._build_network()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _build_network(self):
        # placeholder
        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        if self.layer == 1:
            # Variable
            V = tf.Variable(tf.random_normal([self.hidden_neuron, self.num_item], stddev=0.01))
            W = tf.Variable(tf.random_normal([self.num_item, self.hidden_neuron], stddev=0.01))

            mu = tf.Variable(tf.random_normal([self.hidden_neuron], stddev=0.01))
            b = tf.Variable(tf.random_normal([self.num_item], stddev=0.01))
            layer_1 = tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix))
            self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
            Loss_norm = tf.square(tf.norm(W)) + tf.square(tf.norm(V))
        elif self.layer == 3:
            V_1 = tf.Variable(tf.random_normal([self.hidden_neuron, self.num_item], stddev=0.01))
            V_2 = tf.Variable(tf.random_normal([self.hidden_neuron // 2, self.hidden_neuron], stddev=0.01))
            V_3 = tf.Variable(tf.random_normal([self.hidden_neuron, self.hidden_neuron // 2], stddev=0.01))
            W = tf.Variable(tf.random_normal([self.num_item, self.hidden_neuron], stddev=0.01))
            mu_1 = tf.Variable(tf.random_normal([self.hidden_neuron], stddev=0.01))
            mu_2 = tf.Variable(tf.random_normal([self.hidden_neuron // 2], stddev=0.01))
            mu_3 = tf.Variable(tf.random_normal([self.hidden_neuron], stddev=0.01))
            b = tf.Variable(tf.random_normal([self.num_item], stddev=0.01))
            #
            layer_1 = tf.sigmoid(tf.matmul(V_1, self.rating_matrix) + tf.expand_dims(mu_1, 1))
            layer_2 = tf.sigmoid(tf.matmul(V_2, layer_1) + tf.expand_dims(mu_2, 1))
            layer_3 = tf.sigmoid(tf.matmul(V_3, layer_2) + tf.expand_dims(mu_3, 1))
            self.layer_2 = tf.matmul(W, layer_3) + tf.expand_dims(b, 1)
            Loss_norm = tf.square(tf.norm(W)) + tf.square(tf.norm(V_1)) + tf.square(tf.norm(V_3)) + tf.square(
                tf.norm(V_3))
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2),
                                self.rating_matrix_mask)))) + self.reg_rate + Loss_norm

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):
        self.num_training = self.num_user
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        for i in range(total_batch):
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                               self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx]
                                               })
        return loss

    def test(self, test_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask:
                                                                         self.train_data_mask})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        rmse = np.sqrt(error / len(test_set))
        mae = error_mae / len(test_set)
        return rmse, mae

    def execute(self):
        loss_prev = float("inf")
        for epoch in range(self.epochs):
            loss_cur = self.train()
            if epoch % self.T == 0:
                print("epoch:\t", epoch, "\tloss:\t", loss_cur)
            if abs(loss_cur - loss_prev) < math.exp(-5):
                break
            loss_prev = loss_cur
        rmse, mae = self.test(self.dataset_class.test_matrix_dok)
        print("training done\tRMSE : ", rmse, "\tMAE : ", mae)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore(self, path):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def predict(self, user_id, item_id):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask:
                                                                         self.train_data_mask})
        return self.reconstruction[item_id, user_id]
```

```python cellView="form" id="RlSrs12GaeTS"
#@markdown DCGAN
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


# kernel_size = 5 * 5
def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        try:
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
        except ValueError as err:
            msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
            err.args = err.args + (msg,)
            raise
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def gen_random(size):
    # z - N(0,100)
    return np.random.normal(0, 100, size=size)


class DCGAN(object):
    def __init__(self, sess, dataset_class,batch_size=64, height=29, width=58, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, max_to_keep=1):
        self.sess = sess
        self.dataset_class = dataset_class
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.max_to_keep = max_to_keep

        self.build_model()

    def build_model(self):
        self.inputs = tf.placeholder(tf.float32,
                                     [self.batch_size, self.height, self.width, 1],
                                     name='real_images')
        inputs = self.inputs
        # 生成器
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.G = self.generator(self.z)
        # 判别器 - real&fake
        self.D, self.D_logits = self.discriminator(inputs, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # 损失函数
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        #
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        train_idxs = list(range(self.dataset_class.train_matrix.shape[0]))
        for epoch in xrange(config.epoch):
            np.random.shuffle(train_idxs)
            for i in range(len(train_idxs) // self.batch_size):
                cur_idxs = train_idxs[i * self.batch_size:(i + 1) * self.batch_size]
                batch_inputs = self.dataset_class.train_matrix[cur_idxs].toarray()
                # transform range&shape
                batch_inputs = (batch_inputs - 2.5) / 2.5
                batch_inputs = np.reshape(batch_inputs, [self.batch_size, self.height, self.width, 1])
                # batch_inputs = np.random.random_sample([self.batch_size, self.height, self.width, 1])
                batch_z = gen_random(size=[config.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _ = self.sess.run(d_optim, feed_dict={self.inputs: batch_inputs, self.z: batch_z})

                # Update G network
                _ = self.sess.run(g_optim, feed_dict={self.z: batch_z})

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_inputs})
                errG = self.g_loss.eval({self.z: batch_z})

                print("Epoch:[%2d/%2d]d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, config.epoch, errD_fake + errD_real, errG))

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            # 论文中给的判别器结构:[conv+BN+LeakyRelu[64,128,256,512]]+[FC]+[sigmoid]
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.height, self.width
            # CONV stride=2
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # FC of 2*4*512&ReLU&BN
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)
            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            # four transposed CONV of [256,128,64] &ReLU&BN&kernel_size = 5 * 5
            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))
            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))
            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            # transposed CONV of [1] &tanh
            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, 1], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)
```

```python id="ozxoV-hYYkNu"
def get_model_network(sess, model_name, dataset_class):
    model = None
    if model_name == "IAutoRec":
        model = IAutoRec(sess, dataset_class)
    elif model_name == "UAutoRec":
        model = UAutoRec(sess, dataset_class)
    elif model_name == "NNMF":
        model = NNMF(sess, dataset_class)
    return model


def get_top_n(model, n):
    top_n = {}
    user_nonrated_items = model.dataset_class.get_user_nonrated_items()
    for uid in range(model.num_user):
        items = user_nonrated_items[uid]
        ratings = model.predict([uid] * len(items), items)
        item_rating = list(zip(items, ratings))
        item_rating.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [x[0] for x in item_rating[:n]]
    return top_n


def pred_for_target(model, target_id):
    target_predictions = model.predict(list(range(model.num_user)), [target_id] * model.num_user)

    top_n = get_top_n(model, n=50)
    hit_ratios = {}
    for uid in top_n:
        hit_ratios[uid] = [1 if target_id in top_n[uid][:i] else 0 for i in [1, 3, 5, 10, 20, 50]]
    return target_predictions, hit_ratios


def rec_trainer(model_name, dataset_class, target_id, is_train, model_path):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:

        rec_model = get_model_network(sess, model_name, dataset_class)
        if is_train:
            print('--> start train recommendation model...')
            rec_model.execute()
            rec_model.save(model_path)
        else:
            rec_model.restore(model_path)
        print('--> start pred for each user...')
        predictions, hit_ratios = pred_for_target(rec_model, target_id)
    return predictions, hit_ratios
```

```python cellView="form" id="3iPMFbT_YkKI"
#@markdown surprise trainer
# import os
# from surprise import Dataset, Reader, accuracy
# from surprise import SVD, SVDpp, NMF, KNNBasic, KNNWithMeans, KNNWithZScore
# from surprise.model_selection import PredefinedKFold
# from collections import defaultdict


# def get_top_n(predictions, n=50):
#     # First map the predictions to each user.
#     top_n = defaultdict(list)
#     for uid, iid, true_r, est, _ in predictions:
#         top_n[uid].append((iid, est))
#     # Then sort the predictions for each user and retrieve the k highest ones.
#     for uid, user_ratings in top_n.items():
#         user_ratings.sort(key=lambda x: x[1], reverse=True)
#         top_n[uid] = user_ratings[:n]
#     return top_n


# def get_model(model_name):
#     algo = None
#     if 'KNN' in model_name:
#         model_name = model_name.split('_')
#         knn_model_name = model_name[0]
#         user_based = False if len(model_name) > 1 and model_name[1] == 'I' else True
#         dis_method = 'msd' if len(model_name) < 3 else model_name[2]
#         k = 20 if len(model_name) < 4 else int(model_name[3])
#         sim_options = {'user_based': user_based, 'name': dis_method}
#         if knn_model_name == 'KNNBasic':
#             algo = KNNBasic(sim_options=sim_options, k=k)
#         elif knn_model_name == 'KNNWithMeans':
#             algo = KNNWithMeans(sim_options=sim_options, k=k)
#         elif knn_model_name == 'KNNWithZScore':
#             algo = KNNWithZScore(sim_options=sim_options, k=k)
#     elif 'SVDpp' in model_name or 'SVD' in model_name or 'NMF' in model_name:
#         model_name = model_name.split('_')
#         n_factors = 25 if len(model_name) == 1 else int(model_name[1])
#         if model_name[0] == 'SVDpp':
#             algo = SVDpp(n_factors=n_factors)
#         elif model_name[0] == 'SVD':
#             algo = SVD(n_factors=n_factors)
#         elif model_name[0] == 'NMF':
#             algo = NMF(n_factors=n_factors)
#     return algo


# def get_model_old(model_name):
#     algo = None
#     if model_name == 'KNNBasic_U':
#         sim_options = {'user_based': True}
#         algo = KNNBasic(sim_options=sim_options, k=20)
#     elif model_name == 'KNNBasic_I':
#         sim_options = {'user_based': False}
#         algo = KNNBasic(sim_options=sim_options, k=20)
#         # algo = KNNBasic()
#     elif model_name == 'KNNWithMeans_I':
#         algo = KNNWithMeans(sim_options={'user_based': False}, k=20)
#     elif model_name == 'KNNWithMeans_U':
#         algo = KNNWithMeans(sim_options={'user_based': True}, k=20)
#     elif model_name == 'KNNWithZScore_I':
#         algo = KNNWithZScore(sim_options={'user_based': False}, k=20)
#     elif model_name == 'KNNWithZScore_U':
#         algo = KNNWithZScore(sim_options={'user_based': True}, k=20)
#     elif model_name == 'SVDpp':
#         algo = SVDpp()
#     elif model_name == 'SVD':
#         algo = SVD()
#     elif model_name == 'NMF':
#         algo = NMF()
#     elif 'NMF_' in model_name:
#         n_factors = int(model_name.split("_")[1])
#         algo = NMF(n_factors=n_factors)
#     elif 'SVDpp_' in model_name:
#         n_factors = int(model_name.split("_")[1])
#         algo = SVDpp(n_factors=n_factors)
#     elif 'SVD_' in model_name:
#         n_factors = int(model_name.split("_")[1])
#         algo = SVD(n_factors=n_factors)
#     elif 'KNNBasic_U_' in model_name:
#         k = int(model_name.split("_")[-1])
#         sim_options = {'user_based': True}
#         algo = KNNBasic(sim_options=sim_options, k=k)
#     elif 'KNNBasic_I_' in model_name:
#         k = int(model_name.split("_")[-1])
#         sim_options = {'user_based': False}
#         algo = KNNBasic(sim_options=sim_options, k=k)
#     return algo


# def basic_rec(model_name, train_path, test_path, target_id):
#     # build data
#     # TODO check float and min_r
#     reader = Reader(line_format='user item rating', sep='\t', rating_scale=(1, 5))
#     data = Dataset.load_from_folds([(train_path, test_path)], reader=reader)
#     trainset, testset = None, None
#     pkf = PredefinedKFold()
#     for trainset_, testset_ in pkf.split(data):
#         trainset, testset = trainset_, testset_

#     # train model
#     rec_algo = get_model(model_name)
#     rec_algo.fit(trainset)
#     # eval
#     preds = rec_algo.test(testset)
#     rmse = accuracy.rmse(preds, verbose=True)

#     # predor target
#     fn_pred = lambda uid: rec_algo.predict(str(uid), str(target_id), r_ui=0).est
#     target_predictions = list(map(fn_pred, range(trainset.n_users)))

#     # topn
#     testset = trainset.build_anti_testset()
#     predictions = rec_algo.test(testset)
#     top_n = get_top_n(predictions, n=50)

#     hit_ratios = {}
#     for uid, user_ratings in top_n.items():
#         topN = [int(iid) for (iid, _) in user_ratings]
#         hits = [1 if target_id in topN[:i] else 0 for i in [1, 3, 5, 10, 20, 50]]
#         hit_ratios[int(uid)] = hits
#     return target_predictions, hit_ratios
```

```python cellView="form" id="ovapnDmQYkIK"
#@markdown class BaselineAttack
class BaselineAttack:

    def __init__(self, attack_num, filler_num, n_items, target_id,
                 global_mean, global_std, item_means, item_stds, r_max, r_min, fixed_filler_indicator=None):
        #
        self.attack_num = attack_num
        self.filler_num = filler_num
        self.n_items = n_items
        self.target_id = target_id
        self.global_mean = global_mean
        self.global_std = global_std
        self.item_means = item_means
        self.item_stds = item_stds
        self.r_max = r_max
        self.r_min = r_min
        # 固定sample的filler
        self.fixed_filler_indicator = fixed_filler_indicator

    def RandomAttack(self):
        filler_candis = list(set(range(self.n_items)) - {self.target_id})
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # target
        fake_profiles[:, self.target_id] = self.r_max
        # fillers
        for i in range(self.attack_num):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:
                # 读已有的sample结果
                fillers = np.where(np.array(self.fixed_filler_indicator[i])== 1)[0]
            ratings = np.random.normal(loc=self.global_mean, scale=self.global_std, size=self.filler_num)
            for f_id, r in zip(fillers, ratings):
                fake_profiles[i][f_id] = max(math.exp(-5), min(self.r_max, r))
        return fake_profiles

    def BandwagonAttack(self, selected_ids):
        filler_candis = list(set(range(self.n_items)) - set([self.target_id] + selected_ids))
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # target & selected patch
        fake_profiles[:, [self.target_id] + selected_ids] = self.r_max
        # fillers
        for i in range(self.attack_num):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:
                # 读已有的sample结果
                fillers = np.where(np.array(self.fixed_filler_indicator[i])== 1)[0]
            ratings = np.random.normal(loc=self.global_mean, scale=self.global_std, size=self.filler_num)
            for f_id, r in zip(fillers, ratings):
                fake_profiles[i][f_id] = max(math.exp(-5), min(self.r_max, r))
        return fake_profiles

    def AverageAttack(self):
        filler_candis = list(set(range(self.n_items)) - {self.target_id})
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # target
        fake_profiles[:, self.target_id] = self.r_max
        # fillers
        fn_normal = lambda iid: np.random.normal(loc=self.item_means[iid], scale=self.item_stds[iid], size=1)[0]
        for i in range(self.attack_num):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:
                # 读已有的sample结果
                fillers = np.where(np.array(self.fixed_filler_indicator[i])== 1)[0]
            ratings = map(fn_normal, fillers)
            for f_id, r in zip(fillers, ratings):
                fake_profiles[i][f_id] = max(math.exp(-5), min(self.r_max, r))
        return fake_profiles

    def SegmentAttack(self, selected_ids):
        filler_candis = list(set(range(self.n_items)) - set([self.target_id] + selected_ids))
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # target & selected patch
        fake_profiles[:, [self.target_id] + selected_ids] = self.r_max
        # fillers
        for i in range(self.attack_num):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:
                # 读已有的sample结果
                fillers = np.where(np.array(self.fixed_filler_indicator[i])== 1)[0]
            fake_profiles[i][fillers] = self.r_min
        return fake_profiles
```

```python cellView="form" id="pvBb5S53Zi4s"
#@markdown class GAN_Attacker
class GAN_Attacker:
    def __init__(self):
        print("GAN Attack model")

    def DIS(self, input, inputDim, h, activation, hiddenLayers, _reuse=False):
        # input->hidden
        y, _, W, b = self.FullyConnectedLayer(input, inputDim, h, activation, "dis", 0, reuse=_reuse)

        # stacked hidden layers
        for layer in range(hiddenLayers - 1):
            y, _, W, b = self.FullyConnectedLayer(y, h, h, activation, "dis", layer + 1, reuse=_reuse)

        # hidden -> output
        y, _, W, b = self.FullyConnectedLayer(y, h, 1, "none", "dis", hiddenLayers + 1, reuse=_reuse)

        return y

    def GEN(self, input, num_item, h, outputDim, activation, decay, name="gen", _reuse=False):
        """
        input   :   sparse filler vectors
        output  :   reconstructed selected vector
        """
        # input+thnh
        # input_tanh = tf.nn.tanh(input)

        # input->hidden

        y, L2norm, W, b = self.FullyConnectedLayer(input, num_item, h // decay, activation, name, 0, reuse=_reuse)

        # stacked hidden layers
        h = h // decay
        layer = 0
        # for layer in range(hiddenLayers - 1):
        while True:
            y, this_L2, W, b = self.FullyConnectedLayer(y, h, h // decay, activation, name, layer + 1, reuse=_reuse)
            L2norm = L2norm + this_L2
            layer += 1
            if h // decay > outputDim:
                h = h // decay
            else:
                break
        # hidden -> output
        y, this_L2, W, b = self.FullyConnectedLayer(y, h // decay, outputDim, "none", name, layer + 1, reuse=_reuse)
        L2norm = L2norm + this_L2
        y = tf.nn.sigmoid(y) * 5
        return y, L2norm

    def FullyConnectedLayer(self, input, inputDim, outputDim, activation, model, layer, reuse=False):
        scale1 = math.sqrt(6 / (inputDim + outputDim))

        wName = model + "_W" + str(layer)
        bName = model + "_B" + str(layer)

        with tf.variable_scope(model) as scope:

            if reuse == True:
                scope.reuse_variables()

            W = tf.get_variable(wName, [inputDim, outputDim],
                                initializer=tf.random_uniform_initializer(-scale1, scale1))
            b = tf.get_variable(bName, [outputDim], initializer=tf.random_uniform_initializer(-0.01, 0.01))

            y = tf.matmul(input, W) + b

            L2norm = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

            if activation == "none":
                y = tf.identity(y, name="output")
                return y, L2norm, W, b

            elif activation == "sigmoid":
                return tf.nn.sigmoid(y), L2norm, W, b

            elif activation == "tanh":
                return tf.nn.tanh(y), L2norm, W, b
            elif activation == "relu":
                return tf.nn.relu(y), L2norm, W, b
```

```python cellView="form" id="LKVytFWgYkGc"
#@markdown class CopyGanAttacker
class CopyGanAttacker:
    def __init__(self, dataset_class, target_id, filler_num, attack_num, filler_method):
        # data set info
        self.dataset_class = dataset_class
        self.num_user = dataset_class.n_users
        self.num_item = dataset_class.n_items
        self.rating_matrix = dataset_class.train_matrix.toarray()  # tf.constant()

        # attack info
        self.target_id = target_id
        self.filler_num = filler_num
        self.attack_num = attack_num
        self.filler_method = filler_method

    def build_model(self):
        # define place_holder
        # self.user_vector = tf.placeholder(tf.int32, [None, self.num_item])
        # self.item_vector = tf.placeholder(tf.int32, [None, self.num_item])
        self.sampled_template = tf.placeholder(tf.int32, [self.args.batch_size, self.num_item])
        self.batch_filler_index = tf.placeholder(tf.int32, [None, self.args.batch_size])
        # user/item embedding
        # c = tf.constant(c)
        user_embedding = self.towerMlp(self.rating_matrix, self.num_item, self.args.embedding_dim)
        item_embedding = self.towerMlp(self.rating_matrix.transpose(), self.num_user, self.args.embedding_dim)

        """
        copy net  
        p_copy(j)=sigmoid (w x j’s item embedding + w x u’s user embedding + b)"""
        with tf.name_scope("copyNet"):
            w1 = tf.get_variable('w1', [self.args.embedding_dim, self.num_item])
            p1 = tf.matmul(tf.nn.embedding_lookup(user_embedding, self.batch_filler_index), w1)  # batch*item_num
            w2 = tf.get_variable('w2', [self.args.embedding_dim, 1])
            p2 = tf.matmul(item_embedding, w2)  # item_num*1
            b = tf.get_variable('b', [self.item_num])
            copy_prob = tf.nn.sigmoid(p1 + p2 + b)  # batch*item_num
        """
        generate net
        p_gen(j=r)
        """
        with tf.name_scope("genNet"):
            gen_probabilitiy_list = []
            for i in range(5):
                with tf.name_scope("s_%d" % i):
                    w1 = tf.get_variable('w1', [self.args.embedding_dim, self.num_item])
                    p1 = tf.matmul(tf.nn.embedding_lookup(user_embedding, self.batch_filler_index),
                                   w1)  # batch*item_num
                    w2 = tf.get_variable('w2', [self.args.embedding_dim, 1])
                    p2 = tf.matmul(item_embedding, w2)  # item_num*1
                    b = tf.get_variable('b', [self.item_num])
                    gen_probability = p1 + p2 + b
                    gen_probabilitiy_list.append(tf.expand_dims(gen_probability, 2))  # batch*item_num*1
            gen_rating_distri = tf.nn.softmax(tf.concat(gen_probabilitiy_list, axis=2))  # batch*item_num*5
        """
        Rating
        rating p(r) = p_copy(j) x p_copy(j=r) + (1-p_copy(j)) x p_gen(j=r)
        """
        copy_rating_distri = tf.reshape(tf.expand_dims(tf.one_hot(self.sampled_template, 5), 3),
                                        [self.args.batch_size, -1, 5])
        rating_distri = copy_prob * copy_rating_distri + (1 - copy_prob) * gen_rating_distri  # batch*item_num*5
        rating_value = tf.tile(tf.constant([[[1., 2., 3., 4., 5.]]]), [self.args.batch_size, self.num_item, 1])
        fake_profiles = tf.reduce_sum(rating_distri * rating_value, 2)

        """
        loss function
        """
        with tf.name_scope("Discriminator"):
            D_real = self.towerMlp(self.sampled_template, self.num_item, 1)
            D_fake = self.towerMlp(fake_profiles, self.num_item, 1)

        """
        loss function
        """
        with tf.name_scope("loss_D"):
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)),
                name="loss_real")
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)),
                name="loss_fake")
            loss_D = d_loss_real + d_loss_fake
        with tf.name_scope("loss_G"):
            # reconstruction loss
            loss_rec = tf.reduce_mean(tf.square(fake_profiles - self.sampled_template))
            # adversial loss
            loss_adv = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
            loss_G = loss_rec + loss_adv

    def towerMlp(self, input, inputDim, outputDim):
        dim, x = inputDim // 2, input
        while dim > outputDim:
            layer = tf.layers.dense(
                inputs=x,
                units=dim,
                kernel_initializer=tf.random_normal_initializer,
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
            dim, x = dim // 2, layer
        output = tf.layers.dense(
            inputs=x,
            units=outputDim,
            kernel_initializer=tf.random_normal_initializer,
            activation=tf.nn.sigmoid,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        return output
```

```python cellView="form" id="IlNqogtTYkEi"
#@markdown class Train_G_Attacker
class Train_G_Attacker:
    def __init__(self, dataset_class, params_D, params_G, target_id, selected_id_list,
                 filler_num, attack_num, filler_method, loss_setting):
        # TODO:init refine
        # data set info
        self.dataset_class = dataset_class
        self.num_user = dataset_class.n_users
        self.num_item = dataset_class.n_items

        # attack info
        self.target_id = target_id
        self.selected_id_list = selected_id_list
        self.selected_num = len(self.selected_id_list)
        self.filler_num = filler_num
        self.attack_num = attack_num
        self.filler_method = filler_method
        self.loss_setting = loss_setting

        # model params
        self.totalEpochs = 150
        self.ZR_ratio = 0.5
        # G
        if params_G is None:
            # MLP structure
            self.hiddenDim_G = 400
            # optimize params
            self.reg_G = 0.0001
            self.lr_G = 0.01
            self.opt_G = 'adam'
            self.step_G = 1
            self.batchSize_G = 128 * 2
            self.batchNum_G = 10
            # self.G_loss_weights = [1, 1, 1, 1]
            self.G_loss_weights = [1, 1, 1]
            self.decay_g = 3
        else:
            self.hiddenDim_G, self.hiddenLayer_G, self.scale, \
            self.reg_G, self.lr_G, self.opt_G, self.step_G, self.batchSize_G, self.batchNum_G, self.G_loss_weights = params_G

        # if params_D is None:
        #     # MLP structure
        #     self.hiddenDim_D = 150
        #     self.hiddenLayer_D = 3
        #     # optimize params
        #     self.reg_D = 1e-05
        #     self.lr_D = 0.0001
        #     self.opt_D = 'adam'
        #     self.step_D = 1
        #     self.batchSize_D = 64
        # else:
        #     self.hiddenDim_D, self.hiddenLayer_D, \
        #     self.reg_D, self.lr_D, self.opt_D, self.step_D, self.batchSize_D = params_D
        #
        self.log_dir = '_'.join(
            list(map(str, [self.loss_setting] + self.G_loss_weights + [self.step_G, self.ZR_ratio, str(target_id)])))

    def train_gan(self):
        for epoch in range(self.totalEpochs):
            self.epoch = epoch
            with open(self.log_path, "a+") as fout:
                fout.write("epoch:" + str(epoch) + "\n")
                fout.flush()

            # for epoch_D in range(self.step_D):
            #     self.epoch_D = epoch_D
            #     loss_D, a, b = self.train_D()
            #     print('D', epoch_D, ':', round(loss_D, 5), a, end="")
            #     print(b[0])
            #     with open(self.log_path, "a+") as fout:
            #         log_tmp = 'D' + str(epoch_D) + ':' + str(round(loss_D, 5)) + str(a) + str(b[0])
            #         fout.write(log_tmp + "\n")
            #         fout.flush()

            for epoch_G in range(self.step_G):
                self.epoch_G = epoch_G
                loss_G, loss_G_array, g_out_seed, log_info = self.train_G()
                with open(self.log_path, "a+") as fout:
                    log_tmp = 'G' + str(epoch_G) + ':' + str(round(loss_G, 5)) \
                              + str(loss_G_array) + str(g_out_seed) + str(log_info)
                    fout.write(log_tmp + "\n")
                    fout.flush()
                print('G', epoch_G, ':', round(loss_G, 5), loss_G_array, g_out_seed, log_info)

    def execute(self, is_train, model_path, final_attack_setting):
        self.log_path = 'logs/' + self.log_dir + '/' + "training_log.log"
        # if os.path.exists('logs/' + self.log_dir):
        #     print("\n\n\nexist!!\n\n\n")
        #     return

        with tf.Graph().as_default():
            self._data_preparation()
            self._build_graph()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            if is_train == 0:
                self.restore(model_path)
            else:
                self.wirter = tf.summary.FileWriter('logs/' + self.log_dir + '/', self.sess.graph)
                self.train_gan()
                self.save(model_path)
            # 生成攻击文件
            fake_profiles, real_profiles_, filler_indicator_ \
                = self.fake_profiles_generator(final_attack_setting)
            return fake_profiles, real_profiles_, filler_indicator_

    def fake_profiles_generator(self, final_attack_setting):
        fake_num, real_vector, filler_indicator = final_attack_setting

        # input filler
        if real_vector is None or filler_indicator is None:
            batchList = self.batchList.copy()
            while fake_num > len(batchList):
                batchList += batchList
            random.shuffle(batchList)
            sampled_index = batchList[:fake_num]
            real_vector = self.dataset_class.train_matrix[sampled_index].toarray()
            filler_indicator = self.filler_sampler(sampled_index)

        # output fake profiles
        fake_profiles = self.sess.run(self.fakeData, feed_dict={self.G_input: real_vector,
                                                                self.filler_dims: filler_indicator})
        return fake_profiles, real_vector, filler_indicator

    def _build_graph(self):
        self.filler_dims = tf.placeholder(tf.float32, [None, self.num_item])  # filler = 1, otherwise 0
        self.selected_dims = tf.squeeze(
            tf.reduce_sum(tf.one_hot([self.selected_id_list], self.num_item, dtype=tf.float32), 1))

        self.models = GAN_Attacker()
        # G
        with tf.name_scope("Generator"):
            self.G_input = tf.placeholder(tf.float32, [None, self.num_item], name="G_input")
            self.rating_matrix_mask = tf.placeholder(tf.float32, [None, self.num_item])  # rated = 1, otherwise 0
            self.G_output, self.G_L2norm = self.models.GEN(self.G_input * self.filler_dims, self.num_item,
                                                           self.hiddenDim_G, self.selected_num, 'sigmoid',
                                                           decay=self.decay_g, name="gen")

        with tf.name_scope("Fake_Data"):
            selected_patch = None
            for i in range(self.selected_num):
                one_hot = tf.one_hot(self.selected_id_list[i], self.num_item, dtype=tf.float32)
                mask = tf.boolean_mask(self.G_output, tf.one_hot(i, self.selected_num, dtype=tf.int32), axis=1)
                if i == 0:
                    selected_patch = one_hot * mask
                else:
                    selected_patch += one_hot * mask
            self.fakeData = selected_patch + self.target_patch + self.G_input * self.filler_dims
        # # D
        # with tf.name_scope("Discriminator"):
        #     self.realData_ = tf.placeholder(tf.float32, shape=[None, self.num_item], name="real_data")
        #     self.filler_dims_D = tf.placeholder(tf.float32, [None, self.num_item])  # filler = 1, otherwise 0
        #     self.realData = self.realData_ * (self.filler_dims_D + self.selected_dims)
        #
        #     self.D_real = self.models.DIS(self.realData * self.target_mask, self.num_item * 1, self.hiddenDim_D,
        #                                   'sigmoid', self.hiddenLayer_D)
        #
        #     self.D_fake = self.models.DIS(self.fakeData * self.target_mask, self.num_item * 1, self.hiddenDim_D,
        #                                   'sigmoid', self.hiddenLayer_D, _reuse=True)

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        # self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')

        # define loss & optimizer for G
        with tf.name_scope("loss_G"):
            # self.g_loss_gan = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
            self.g_loss_reconstruct_seed = tf.reduce_mean(
                tf.reduce_sum(tf.square(self.fakeData - self.G_input) * self.rating_matrix_mask * self.selected_dims,
                              1, keepdims=True))
            self.g_loss_list = [self.g_loss_reconstruct_seed]

            if self.loss_setting == 1:
                self.g_loss_seed = tf.reduce_mean(
                    tf.reduce_mean(tf.square(self.G_output - 5.0), 1, keepdims=True))
                self.g_loss_list.append(self.g_loss_seed)
            self.g_loss_l2 = self.reg_G * self.G_L2norm
            self.g_loss_list.append(self.g_loss_l2)
            # self.g_loss_list = [self.g_loss_gan, self.g_loss_seed,
            #                     self.g_loss_reconstruct_seed, self.g_loss_l2]
            # self.g_loss_list = [self.g_loss_seed, self.g_loss_reconstruct_seed, self.g_loss_l2]
            self.g_loss = sum(self.g_loss_list[i] * self.G_loss_weights[i] for i in range(len(self.g_loss_list)))

        # tensorboard summary
        self.add_loss_summary(type='G')

        with tf.name_scope("optimizer_G"):
            if self.opt_G == 'sgd':
                self.trainer_G = tf.train.GradientDescentOptimizer(self.lr_G).minimize(self.g_loss,
                                                                                       var_list=self.g_vars,
                                                                                       name="GradientDescent_G")
            elif self.opt_G == 'adam':
                self.trainer_G = tf.train.AdamOptimizer(self.lr_G).minimize(self.g_loss, var_list=self.g_vars,
                                                                            name="Adam_G")

        # define loss & optimizer for D

        # with tf.name_scope("loss_D"):
        #     d_loss_real = tf.reduce_mean(
        #         tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real)),
        #         name="loss_real")
        #     d_loss_fake = tf.reduce_mean(
        #         tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)),
        #         name="loss_fake")
        #     D_L2norm = 0
        #     for pr in self.d_vars:
        #         D_L2norm += tf.nn.l2_loss(pr)
        #     self.d_loss = d_loss_real + d_loss_fake + self.reg_D * D_L2norm
        #     self.d_loss_real, self.d_loss_fake, self.D_L2norm = d_loss_real, d_loss_fake, D_L2norm
        # with tf.name_scope("optimizer_D"):
        #     if self.opt_D == 'sgd':
        #         self.trainer_D = tf.train.GradientDescentOptimizer(self.lr_D).minimize(self.d_loss,
        #                                                                                var_list=self.d_vars,
        #                                                                                name="GradientDescent_D")
        #     elif self.opt_D == 'adam':
        #         self.trainer_D = tf.train.AdamOptimizer(self.lr_D).minimize(self.d_loss, var_list=self.d_vars,
        #                                                                     name="Adam_D")

    def _data_preparation(self):
        self.target_patch = tf.one_hot(self.target_id, self.num_item, dtype=tf.float32) * 5
        self.target_mask = 1 - tf.one_hot(self.target_id, self.num_item, dtype=tf.float32)
        # prepare train data
        self.train_matrix = self.dataset_class.train_matrix.toarray().astype(np.float32)
        self.train_mask = self.train_matrix.copy()
        self.train_mask[self.train_mask > 0] = 1
        self.filler_candi_set = set(range(self.num_item)) - set(self.selected_id_list + [self.target_id])
        self.filler_candi_list = list(self.filler_candi_set)

        # sample filler>filler num
        self.batchList = []
        for i in range(self.num_user):
            set_rated = set(self.train_mask[i].nonzero()[0])
            if len(self.filler_candi_set & set_rated) < self.filler_num: continue
            self.batchList.append(i)

        # 没有在train set对target item评分的用户，用来算all user的pred shift
        self.non_rated_users = self.dataset_class.get_item_nonrated_users(self.target_id)
        # item pop/avg
        self.item_pop = np.array(self.dataset_class.get_item_pop())
        _, _, self.item_avg, _ = self.dataset_class.get_all_mean_std()
        self.item_avg = np.array(self.item_avg)

        # big cap
        if self.filler_method == 3:
            print("\n==\n==\n修改路径！！\n==\n")
            attack_info_path = ["../data/data/filmTrust_selected_items", "../data/data/filmTrust_selected_items"]
            attack_info = load_attack_info(*attack_info_path)
            target_users = attack_info[self.target_id][1]
            uid_values = self.dataset_class.train_data.user_id.values
            idxs = [idx for idx in range(len(uid_values)) if uid_values[idx] in target_users]
            iid_values = self.dataset_class.train_data.loc[idxs, 'item_id']
            iid_values = iid_values.tolist()
            from collections import Counter
            iid_values = Counter(iid_values)
            self.item_big_cap = np.array([iid_values.get(iid, 0.5) for iid in range(self.num_item)])

    def train_G(self):
        t1 = time.time()
        random.seed(int(t1))
        random.shuffle(self.batchList)
        #
        batch_real_vector = None
        batch_run_res = None
        #
        total_loss_g = 0
        # total_loss_array = np.array([0., 0., 0., 0.])
        total_loss_array = np.array([0.] * len(self.g_loss_list))
        total_batch = int(len(self.batchList) / self.batchSize_G) + 1
        for batch_id in range(total_batch):
            if batch_id == total_batch - 1:
                batch_index = self.batchList[batch_id * self.batchSize_G:]
            else:
                batch_index = self.batchList[batch_id * self.batchSize_G: (batch_id + 1) * self.batchSize_G]

            batch_size = len(batch_index)
            batch_real_vector = self.train_matrix[batch_index]
            batch_mask = self.train_mask[batch_index]

            # sample zero for zero reconstruction
            batch_mask_ZR = batch_mask.copy()
            if self.ZR_ratio > 0:
                for idx in range(batch_size):
                    batch_mask_ZR[idx][self.selected_id_list] = \
                        [1 if i == 1 or random.random() < self.ZR_ratio else 0 for i in
                         batch_mask_ZR[idx][self.selected_id_list]]

            # sample fillers randomly
            batch_filler_indicator = self.filler_sampler(batch_index)

            batch_run_res = self.sess.run(
                [self.trainer_G, self.g_loss] + self.g_loss_list + [self.G_output, self.G_loss_merged],
                feed_dict={self.G_input: batch_real_vector,
                           self.filler_dims: batch_filler_indicator,
                           self.rating_matrix_mask: batch_mask_ZR})  # Update G

            total_loss_g += batch_run_res[1]
            total_loss_array += np.array(batch_run_res[2:2 + len(total_loss_array)])

        self.wirter.add_summary(batch_run_res[-1], self.step_G * self.epoch + self.epoch_G + 1)
        total_loss_array = [round(i, 2) for i in total_loss_array]
        g_out_seed = [round(i, 2) for i in np.mean(batch_run_res[-2], 0)]
        #
        fn_float_to_str = lambda x: str(round(x, 2))
        r = batch_real_vector.transpose()[self.selected_id_list].transpose()
        g = batch_run_res[-2]
        rmse = list(map(fn_float_to_str, np.sum(np.square(np.abs(r - g)), 0)))
        var_col = list(map(fn_float_to_str, np.var(g, 0)))
        self.add_loss_summary(type="var", info=np.var(g, 0))
        var_row = round(np.mean(np.var(g, 1)), 2)
        # var_col_ori = list(map(fn_float_to_str, np.var(r, 0)))
        # var_row_ori = round(np.mean(np.var(r, 1)), 2)
        log_info = "rmse : " + ','.join(rmse)
        log_info += "\tvar_col : " + ','.join(var_col) + "\tvar_row : " + str(var_row)
        # log_info += "\tvar_col_ori : " + ','.join(var_col_ori) + "\tvar_row_ori : " + str(var_row_ori)
        return total_loss_g, total_loss_array, g_out_seed, log_info  # [g_out_seed, mae, [var_col, var_row]]

    # def train_D(self):
    #     """
    #     每个epoch各产生self.batchSize_D个realData和fakeData
    #     """
    #     t1 = time.time()
    #     random.seed(int(t1))
    #     random.shuffle(self.batchList)
    #
    #     total_loss_d, total_loss_d_real, total_loss_d_fake = 0, 0, 0
    #     #
    #     batch_filler_indicator = None
    #
    #     total_batch = int(len(self.batchList) / self.batchSize_D) + 1
    #     for batch_id in range(total_batch):
    #         # prepare data
    #         if batch_id == total_batch - 1:
    #             batch_index = self.batchList[batch_id * self.batchSize_D:]
    #         else:
    #             batch_index = self.batchList[batch_id * self.batchSize_D: (batch_id + 1) * self.batchSize_D]
    #         batch_size = len(batch_index)
    #         batch_real_vector = self.train_matrix[batch_index]
    #         batch_filler_indicator = self.filler_sampler(batch_index)
    #
    #         # optimize
    #         _, total_loss_d_, total_loss_d_real_, total_loss_d_fake_ \
    #             = self.sess.run([self.trainer_D, self.d_loss, self.d_loss_real, self.d_loss_fake],
    #                             feed_dict={self.realData_: batch_real_vector,
    #                                        self.G_input: batch_real_vector,
    #                                        self.filler_dims: batch_filler_indicator,
    #                                        self.filler_dims_D: batch_filler_indicator})  # Update D
    #         total_loss_d += total_loss_d_
    #         total_loss_d_real += total_loss_d_real_
    #         total_loss_d_fake += total_loss_d_fake_
    #     self.add_loss_summary(type="D", info=[total_loss_d, total_loss_d_real, total_loss_d_fake])
    #     debug_info = [self.G_output, self.fakeData,
    #                   tf.squeeze(tf.nn.sigmoid(self.D_real)), tf.squeeze(tf.nn.sigmoid(self.D_fake))]
    #     info = self.sess.run(debug_info, feed_dict={self.realData_: batch_real_vector,
    #                                                 self.G_input: batch_real_vector,
    #                                                 self.filler_dims: batch_filler_indicator,
    #                                                 self.filler_dims_D: batch_filler_indicator})
    #
    #     D_real, D_fake = info[2:4]
    #     fake_data = info[1]
    #     # lower bound
    #     lower_bound = []
    #     for v in fake_data:
    #         t = v.copy()
    #         t[[self.target_id]] = 0.0  # 对判别器mask掉了target信息
    #         t[self.selected_id_list] = 5.0
    #         lower_bound.append(t)
    #     # upper bound
    #     upper_bound = []
    #     i = 0
    #     for v in fake_data:
    #         t = v.copy()
    #         t[self.selected_id_list] = batch_real_vector[i][self.selected_id_list]
    #         t[[self.target_id]] = 0.0  # 对判别器mask掉了target信息
    #         upper_bound.append(t)
    #         i += 1
    #     zero_data = []  # fake_data.copy()
    #     for v in fake_data:
    #         t = v.copy()
    #         t[[self.target_id]] = 0.0  # 对判别器mask掉了target信息
    #         t[self.selected_id_list] = 0.0
    #         zero_data.append(t)
    #     random_data = []
    #     for v in fake_data:
    #         t = v.copy()
    #         t[self.selected_id_list] = np.random.choice(list([1., 2., 3., 4., 5.]), size=self.selected_num,
    #                                                     replace=True)
    #         t[[self.target_id]] = 0.0  # 对判别器mask掉了target信息
    #         random_data.append(t)
    #
    #     D_lower_bound = self.sess.run(tf.squeeze(tf.nn.sigmoid(self.D_real)),
    #                                   feed_dict={self.realData_: lower_bound,
    #                                              self.filler_dims_D: batch_filler_indicator})
    #     D_upper_bound = self.sess.run(tf.squeeze(tf.nn.sigmoid(self.D_real)),
    #                                   feed_dict={self.realData_: upper_bound,
    #                                              self.filler_dims_D: batch_filler_indicator})
    #
    #     D_zero = self.sess.run(tf.squeeze(tf.nn.sigmoid(self.D_real)),
    #                            feed_dict={self.realData_: zero_data, self.filler_dims_D: batch_filler_indicator})
    #     D_random = self.sess.run(tf.squeeze(tf.nn.sigmoid(self.D_real)),
    #                              feed_dict={self.realData_: random_data, self.filler_dims_D: batch_filler_indicator})
    #     # filler=1通常会更假
    #
    #     d_info = [round(np.mean(D_real), 2), round(np.mean(D_fake), 2),
    #               [round(np.mean(D_lower_bound), 2), round(np.mean(D_upper_bound), 2)],
    #               round(np.mean(D_zero), 2), round(np.mean(D_random), 2)]
    #     # s = ["T:", "G:", "s=5:", "s=0:", "s=random:", "s=5,f=1:"]
    #     # s = ["real:", "fake:", "seed=5:", "seed=0:", "seed=random:", "seed=5,filler=1:"]
    #     # d_info = ' '.join([str(d_info[i]) for i in range(len(d_info))])  # s[i]+ str(d_info[i])
    #
    #     #
    #     fn_float_to_str = lambda x: str(round(x, 2))
    #     g_out_seed = list(map(fn_float_to_str, np.mean(info[0], 0)))  # [round(i, 2) for i in np.mean(info[0], 0)]
    #
    #     #
    #
    #     g = info[0]
    #     var_col = list(map(fn_float_to_str, np.var(g, 0)))
    #     var_row = round(np.mean(np.var(g, 1)), 2)
    #     log_info = "\tg_out_seed:" + ','.join(g_out_seed), "\tvar_col : " + ','.join(var_col) + "\tvar_row : " + str(
    #         var_row)
    #
    #     return total_loss_d, d_info, log_info

    def filler_sampler(self, uid_list):
        if self.filler_method == 0:
            batch_filler_indicator = []
            for uid in uid_list:
                filler_candi = np.array(
                    list(set(self.filler_candi_list) & set(self.train_mask[uid].nonzero()[0].tolist())))
                if len(filler_candi) > self.filler_num:
                    filler_candi = np.random.choice(filler_candi, size=self.filler_num, replace=False)
                filler_indicator = [1 if iid in filler_candi else 0 for iid in range(self.num_item)]
                batch_filler_indicator.append(filler_indicator)
            return batch_filler_indicator
        else:
            return self.filler_sampler_method(uid_list)

    def filler_sampler_method(self, uid_list):
        batch_filler_indicator = []
        for uid in uid_list:
            filler_candi = np.array(
                list(set(self.filler_candi_list) & set(self.train_mask[uid].nonzero()[0].tolist())))
            if len(filler_candi) > self.filler_num:
                # sample using a specific method
                # -------------------------
                prob = self.item_avg[filler_candi] if self.filler_method == 1 \
                    else self.item_pop[filler_candi] if self.filler_method == 2 \
                    else self.item_big_cap[filler_candi] if self.filler_method == 3 \
                    else None
                prob = None if prob is None else prob / sum(prob)
                # -------------------------

                filler_candi = np.random.choice(filler_candi, size=self.filler_num, replace=False, p=prob)
            filler_indicator = [1 if iid in filler_candi else 0 for iid in range(self.num_item)]
            batch_filler_indicator.append(filler_indicator)
        return batch_filler_indicator

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def add_loss_summary(self, type="G", info=None):
        # , total_loss_g, total_g_loss_gan, total_g_loss_seed, total_g_loss_reconstruct,total_g_loss_l2):
        if type == "G":
            # tf.summary.scalar('Generator/adversarial', self.g_loss_gan)
            if hasattr(self, 'g_loss_seed'):
                tf.summary.scalar('Generator/seed', self.g_loss_seed)
            tf.summary.scalar('Generator/selected_reconstruct', self.g_loss_reconstruct_seed)
            tf.summary.scalar('Generator/l2_normal', self.g_loss_l2)
            tf.summary.scalar('Generator/Sum', self.g_loss)
            self.G_loss_merged = tf.summary.merge_all()

        # elif type == 'D':
        #     total_loss_d, total_loss_d_real, total_loss_d_fake = info
        #     loss_summary = []
        #     tag_list = ['Discriminator/Sum', 'Discriminator/real', 'Discriminator/fake']
        #     simple_value_list = [total_loss_d, total_loss_d_real, total_loss_d_fake]
        #     for i in range(3):
        #         loss_summary.append(tf.Summary.Value(tag=tag_list[i], simple_value=simple_value_list[i]))
        #     self.wirter.add_summary(tf.Summary(value=loss_summary), self.epoch * self.step_D + self.epoch_D + 1)
        elif type == 'var':
            var_summary = []
            for i in range(self.selected_num):
                var_summary.append(tf.Summary.Value(tag='Var/' + str(i), simple_value=info[i]))
            self.wirter.add_summary(tf.Summary(value=var_summary), self.step_G * self.epoch + self.epoch_G + 1)
        else:
            print("summary type error")
```

```python cellView="form" id="Sbe-0hLiYkBu"
#@markdown class Train_GAN_Attacker
class Train_GAN_Attacker:
    def __init__(self, dataset_class, params_D, params_G, target_id, selected_id_list,
                 filler_num, attack_num, filler_method):
        # TODO:init refine
        # data set info
        self.dataset_class = dataset_class
        self.num_user = dataset_class.n_users
        self.num_item = dataset_class.n_items

        # attack info
        self.target_id = target_id
        self.selected_id_list = selected_id_list
        self.selected_num = len(self.selected_id_list)
        self.filler_num = filler_num
        self.attack_num = attack_num
        self.filler_method = filler_method

        # model params
        self.totalEpochs = 150
        self.ZR_ratio = 0.5
        # G
        if params_G is None:
            # MLP structure
            self.hiddenDim_G = 400
            # optimize params
            self.reg_G = 0.0001
            self.lr_G = 0.01
            self.opt_G = 'adam'
            self.step_G = 1
            self.batchSize_G = 128 * 2
            self.batchNum_G = 10
            self.G_loss_weights = [1, 1, 1, 1]
            self.decay_g = 3
        else:
            self.hiddenDim_G, self.hiddenLayer_G, self.scale, \
            self.reg_G, self.lr_G, self.opt_G, self.step_G, self.batchSize_G, self.batchNum_G, self.G_loss_weights = params_G

        if params_D is None:
            # MLP structure
            self.hiddenDim_D = 150
            self.hiddenLayer_D = 3
            # optimize params
            self.reg_D = 1e-05
            self.lr_D = 0.0001
            self.opt_D = 'adam'
            self.step_D = 1
            self.batchSize_D = 64
        else:
            self.hiddenDim_D, self.hiddenLayer_D, \
            self.reg_D, self.lr_D, self.opt_D, self.step_D, self.batchSize_D = params_D
        #
        self.log_dir = '_'.join(
            list(map(str, self.G_loss_weights + [self.step_G, self.step_D, self.ZR_ratio, str(target_id)])))

    def train_gan(self):
        for epoch in range(self.totalEpochs):
            self.epoch = epoch
            with open(self.log_path, "a+") as fout:
                fout.write("epoch:" + str(epoch) + "\n")
                fout.flush()

            for epoch_D in range(self.step_D):
                self.epoch_D = epoch_D
                loss_D, a, b = self.train_D()
                print('D', epoch_D, ':', round(loss_D, 5), a, end="")
                print(b[0])
                with open(self.log_path, "a+") as fout:
                    log_tmp = 'D' + str(epoch_D) + ':' + str(round(loss_D, 5)) + str(a) + str(b[0])
                    fout.write(log_tmp + "\n")
                    fout.flush()

            for epoch_G in range(self.step_G):
                self.epoch_G = epoch_G
                loss_G, loss_G_array, g_out_seed, log_info = self.train_G()
                with open(self.log_path, "a+") as fout:
                    log_tmp = 'G' + str(epoch_G) + ':' + str(round(loss_G, 5)) \
                              + str(loss_G_array) + str(g_out_seed) + str(log_info)
                    fout.write(log_tmp + "\n")
                    fout.flush()
                print('G', epoch_G, ':', round(loss_G, 5), loss_G_array, g_out_seed, log_info)

    def execute(self, is_train, model_path, final_attack_setting):
        self.log_path = 'logs/' + self.log_dir + '/' + "training_log.log"

        with tf.Graph().as_default():
            self._data_preparation()
            self._build_graph()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            # 训练或恢复模型
            if is_train == 0:
                if model_path != 'no':
                    self.restore(model_path)
            else:
                self.wirter = tf.summary.FileWriter('logs/' + self.log_dir + '/', self.sess.graph)
                self.train_gan()
                self.save(model_path)
            # 生成攻击文件
            fake_profiles, real_profiles_, filler_indicator_ \
                = self.fake_profiles_generator(final_attack_setting)
            return fake_profiles, real_profiles_, filler_indicator_

    def fake_profiles_generator(self, final_attack_setting):
        fake_num, real_vector, filler_indicator = final_attack_setting

        # input filler
        if real_vector is None or filler_indicator is None:
            batchList = self.batchList.copy()
            while fake_num > len(batchList):
                batchList += batchList
            random.shuffle(batchList)
            sampled_index = batchList[:fake_num]
            real_vector = self.dataset_class.train_matrix[sampled_index].toarray()
            filler_indicator = self.filler_sampler(sampled_index)

        # output fake profiles
        fake_profiles = self.sess.run(self.fakeData, feed_dict={self.G_input: real_vector,
                                                                self.filler_dims: filler_indicator})
        return fake_profiles, real_vector, filler_indicator
        # if return_real_filler == 0:
        #     return fake_profiles
        # else:
        #     # batchList = self.batchList.copy()
        #     # while fake_num > len(batchList):
        #     #     batchList += batchList
        #     # random.shuffle(batchList)
        #     # sampled_index = batchList[:fake_num]
        #     # # real_profiles = self.train_matrix[sampled_index]
        #     # real_profiles = self.dataset_class.train_matrix[sampled_index].toarray()
        #     # filler_indicator = np.array(self.filler_sampler(sampled_index))
        #     # for idx in range(filler_indicator.shape[0]):
        #     #     filler_indicator[idx][self.selected_id_list + [self.target_id]] = 1
        #     # return fake_profiles, real_profiles * filler_indicator
        #     return fake_profiles, real_vector, filler_indicator

    def _build_graph(self):
        self.filler_dims = tf.placeholder(tf.float32, [None, self.num_item])  # filler = 1, otherwise 0
        self.selected_dims = tf.squeeze(
            tf.reduce_sum(tf.one_hot([self.selected_id_list], self.num_item, dtype=tf.float32), 1))

        self.models = GAN_Attacker()
        # G
        with tf.name_scope("Generator"):
            self.G_input = tf.placeholder(tf.float32, [None, self.num_item], name="G_input")
            self.rating_matrix_mask = tf.placeholder(tf.float32, [None, self.num_item])  # rated = 1, otherwise 0
            self.G_output, self.G_L2norm = self.models.GEN(self.G_input * self.filler_dims, self.num_item,
                                                           self.hiddenDim_G, self.selected_num, 'sigmoid',
                                                           decay=self.decay_g, name="gen")

        with tf.name_scope("Fake_Data"):
            selected_patch = None
            for i in range(self.selected_num):
                one_hot = tf.one_hot(self.selected_id_list[i], self.num_item, dtype=tf.float32)
                mask = tf.boolean_mask(self.G_output, tf.one_hot(i, self.selected_num, dtype=tf.int32), axis=1)
                if i == 0:
                    selected_patch = one_hot * mask
                else:
                    selected_patch += one_hot * mask
            self.fakeData = selected_patch + self.target_patch + self.G_input * self.filler_dims
        # D
        with tf.name_scope("Discriminator"):
            self.realData_ = tf.placeholder(tf.float32, shape=[None, self.num_item], name="real_data")
            self.filler_dims_D = tf.placeholder(tf.float32, [None, self.num_item])  # filler = 1, otherwise 0
            self.realData = self.realData_ * (self.filler_dims_D + self.selected_dims)

            self.D_real = self.models.DIS(self.realData * self.target_mask, self.num_item * 1, self.hiddenDim_D,
                                          'sigmoid', self.hiddenLayer_D)

            self.D_fake = self.models.DIS(self.fakeData * self.target_mask, self.num_item * 1, self.hiddenDim_D,
                                          'sigmoid', self.hiddenLayer_D, _reuse=True)

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')

        # define loss & optimizer for G
        with tf.name_scope("loss_G"):
            self.g_loss_gan = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
            self.g_loss_seed = tf.reduce_mean(
                tf.reduce_mean(tf.square(self.G_output - 5.0), 1, keepdims=True))
            self.g_loss_reconstruct_seed = tf.reduce_mean(
                tf.reduce_sum(tf.square(self.fakeData - self.G_input) * self.rating_matrix_mask * self.selected_dims,
                              1, keepdims=True))
            self.g_loss_l2 = self.reg_G * self.G_L2norm

            self.g_loss_list = [self.g_loss_gan, self.g_loss_seed,
                                self.g_loss_reconstruct_seed, self.g_loss_l2]
            self.g_loss = sum(self.g_loss_list[i] * self.G_loss_weights[i] for i in range(len(self.G_loss_weights)))

        # tensorboard summary
        self.add_loss_summary(type='G')

        with tf.name_scope("optimizer_G"):
            if self.opt_G == 'sgd':
                self.trainer_G = tf.train.GradientDescentOptimizer(self.lr_G).minimize(self.g_loss,
                                                                                       var_list=self.g_vars,
                                                                                       name="GradientDescent_G")
            elif self.opt_G == 'adam':
                self.trainer_G = tf.train.AdamOptimizer(self.lr_G).minimize(self.g_loss, var_list=self.g_vars,
                                                                            name="Adam_G")

        # define loss & optimizer for D

        with tf.name_scope("loss_D"):
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real)),
                name="loss_real")
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)),
                name="loss_fake")
            D_L2norm = 0
            for pr in self.d_vars:
                D_L2norm += tf.nn.l2_loss(pr)
            self.d_loss = d_loss_real + d_loss_fake + self.reg_D * D_L2norm
            self.d_loss_real, self.d_loss_fake, self.D_L2norm = d_loss_real, d_loss_fake, D_L2norm
        with tf.name_scope("optimizer_D"):
            if self.opt_D == 'sgd':
                self.trainer_D = tf.train.GradientDescentOptimizer(self.lr_D).minimize(self.d_loss,
                                                                                       var_list=self.d_vars,
                                                                                       name="GradientDescent_D")
            elif self.opt_D == 'adam':
                self.trainer_D = tf.train.AdamOptimizer(self.lr_D).minimize(self.d_loss, var_list=self.d_vars,
                                                                            name="Adam_D")

    def _data_preparation(self):
        self.target_patch = tf.one_hot(self.target_id, self.num_item, dtype=tf.float32) * 5
        self.target_mask = 1 - tf.one_hot(self.target_id, self.num_item, dtype=tf.float32)
        # prepare train data
        # self.train_matrix = self.dataset_class.train_matrix.toarray().astype(np.float32)
        # self.train_mask = self.train_matrix.copy()
        # self.train_mask[self.train_mask > 0] = 1
        self.filler_candi_set = set(range(self.num_item)) - set(self.selected_id_list + [self.target_id])
        self.filler_candi_list = list(self.filler_candi_set)

        # sample filler>filler num
        self.batchList = []
        for i in range(self.num_user):
            set_rated = set(self.dataset_class.train_matrix[i].toarray()[0].nonzero()[0])
            # set_rated = set(self.train_mask[i].nonzero()[0])
            if len(self.filler_candi_set & set_rated) < self.filler_num: continue
            self.batchList.append(i)

        # 没有在train set对target item评分的用户，用来算all user的pred shift
        self.non_rated_users = self.dataset_class.get_item_nonrated_users(self.target_id)
        # item pop/avg
        self.item_pop = np.array(self.dataset_class.get_item_pop())
        _, _, self.item_avg, _ = self.dataset_class.get_all_mean_std()
        self.item_avg = np.array(self.item_avg)

        # big cap
        if self.filler_method == 3:
            print("\n==\n==\n修改路径！！\n==\n")
            attack_info_path = ["../data/data/filmTrust_selected_items", "../data/data/filmTrust_selected_items"]
            attack_info = load_attack_info(*attack_info_path)
            target_users = attack_info[self.target_id][1]
            uid_values = self.dataset_class.train_data.user_id.values
            idxs = [idx for idx in range(len(uid_values)) if uid_values[idx] in target_users]
            iid_values = self.dataset_class.train_data.loc[idxs, 'item_id']
            iid_values = iid_values.tolist()
            from collections import Counter
            iid_values = Counter(iid_values)
            self.item_big_cap = np.array([iid_values.get(iid, 0.5) for iid in range(self.num_item)])

    def train_G(self):
        t1 = time.time()
        random.seed(int(t1))
        random.shuffle(self.batchList)
        #
        batch_real_vector = None
        batch_run_res = None
        #
        total_loss_g = 0
        total_loss_array = np.array([0., 0., 0., 0.])
        total_batch = int(len(self.batchList) / self.batchSize_G) + 1
        for batch_id in range(total_batch):
            if batch_id == total_batch - 1:
                batch_index = self.batchList[batch_id * self.batchSize_G:]
            else:
                batch_index = self.batchList[batch_id * self.batchSize_G: (batch_id + 1) * self.batchSize_G]

            batch_size = len(batch_index)
            # batch_real_vector = self.train_matrix[batch_index]
            batch_real_vector = self.dataset_class.train_matrix[batch_index].toarray()
            # batch_mask = self.train_mask[batch_index]
            batch_mask = batch_real_vector.copy()
            batch_mask[batch_mask > 0] = 1

            # sample zero for zero reconstruction
            batch_mask_ZR = batch_mask.copy()
            if self.ZR_ratio > 0:
                for idx in range(batch_size):
                    batch_mask_ZR[idx][self.selected_id_list] = \
                        [1 if i == 1 or random.random() < self.ZR_ratio else 0 for i in
                         batch_mask_ZR[idx][self.selected_id_list]]

            # sample fillers randomly
            batch_filler_indicator = self.filler_sampler(batch_index)

            batch_run_res = self.sess.run(
                [self.trainer_G, self.g_loss] + self.g_loss_list + [self.G_output, self.G_loss_merged],
                feed_dict={self.G_input: batch_real_vector,
                           self.filler_dims: batch_filler_indicator,
                           self.rating_matrix_mask: batch_mask_ZR})  # Update G

            total_loss_g += batch_run_res[1]
            total_loss_array += np.array(batch_run_res[2:2 + len(total_loss_array)])

        self.wirter.add_summary(batch_run_res[-1], self.step_G * self.epoch + self.epoch_G + 1)
        total_loss_array = [round(i, 2) for i in total_loss_array]
        g_out_seed = [round(i, 2) for i in np.mean(batch_run_res[-2], 0)]
        #
        fn_float_to_str = lambda x: str(round(x, 2))
        r = batch_real_vector.transpose()[self.selected_id_list].transpose()
        g = batch_run_res[-2]
        rmse = list(map(fn_float_to_str, np.sum(np.square(np.abs(r - g)), 0)))
        var_col = list(map(fn_float_to_str, np.var(g, 0)))
        self.add_loss_summary(type="var", info=np.var(g, 0))
        var_row = round(np.mean(np.var(g, 1)), 2)
        # var_col_ori = list(map(fn_float_to_str, np.var(r, 0)))
        # var_row_ori = round(np.mean(np.var(r, 1)), 2)
        log_info = "rmse : " + ','.join(rmse)
        log_info += "\tvar_col : " + ','.join(var_col) + "\tvar_row : " + str(var_row)
        # log_info += "\tvar_col_ori : " + ','.join(var_col_ori) + "\tvar_row_ori : " + str(var_row_ori)
        return total_loss_g, total_loss_array, g_out_seed, log_info  # [g_out_seed, mae, [var_col, var_row]]

    def train_D(self):
        """
        每个epoch各产生self.batchSize_D个realData和fakeData
        """
        t1 = time.time()
        random.seed(int(t1))
        random.shuffle(self.batchList)

        total_loss_d, total_loss_d_real, total_loss_d_fake = 0, 0, 0
        #
        batch_filler_indicator = None

        total_batch = int(len(self.batchList) / self.batchSize_D) + 1
        for batch_id in range(total_batch):
            # prepare data
            if batch_id == total_batch - 1:
                batch_index = self.batchList[batch_id * self.batchSize_D:]
            else:
                batch_index = self.batchList[batch_id * self.batchSize_D: (batch_id + 1) * self.batchSize_D]
            batch_size = len(batch_index)
            batch_real_vector = self.dataset_class.train_matrix[batch_index].toarray()
            # batch_real_vector = self.train_matrix[batch_index]
            batch_filler_indicator = self.filler_sampler(batch_index)

            # optimize
            _, total_loss_d_, total_loss_d_real_, total_loss_d_fake_ \
                = self.sess.run([self.trainer_D, self.d_loss, self.d_loss_real, self.d_loss_fake],
                                feed_dict={self.realData_: batch_real_vector,
                                           self.G_input: batch_real_vector,
                                           self.filler_dims: batch_filler_indicator,
                                           self.filler_dims_D: batch_filler_indicator})  # Update D
            total_loss_d += total_loss_d_
            total_loss_d_real += total_loss_d_real_
            total_loss_d_fake += total_loss_d_fake_
        self.add_loss_summary(type="D", info=[total_loss_d, total_loss_d_real, total_loss_d_fake])
        debug_info = [self.G_output, self.fakeData,
                      tf.squeeze(tf.nn.sigmoid(self.D_real)), tf.squeeze(tf.nn.sigmoid(self.D_fake))]
        info = self.sess.run(debug_info, feed_dict={self.realData_: batch_real_vector,
                                                    self.G_input: batch_real_vector,
                                                    self.filler_dims: batch_filler_indicator,
                                                    self.filler_dims_D: batch_filler_indicator})

        D_real, D_fake = info[2:4]
        fake_data = info[1]
        # lower bound
        lower_bound = []
        for v in fake_data:
            t = v.copy()
            t[[self.target_id]] = 0.0  # 对判别器mask掉了target信息
            t[self.selected_id_list] = 5.0
            lower_bound.append(t)
        # upper bound
        upper_bound = []
        i = 0
        for v in fake_data:
            t = v.copy()
            t[self.selected_id_list] = batch_real_vector[i][self.selected_id_list]
            t[[self.target_id]] = 0.0  # 对判别器mask掉了target信息
            upper_bound.append(t)
            i += 1
        zero_data = []  # fake_data.copy()
        for v in fake_data:
            t = v.copy()
            t[[self.target_id]] = 0.0  # 对判别器mask掉了target信息
            t[self.selected_id_list] = 0.0
            zero_data.append(t)
        random_data = []
        for v in fake_data:
            t = v.copy()
            t[self.selected_id_list] = np.random.choice(list([1., 2., 3., 4., 5.]), size=self.selected_num,
                                                        replace=True)
            t[[self.target_id]] = 0.0  # 对判别器mask掉了target信息
            random_data.append(t)

        D_lower_bound = self.sess.run(tf.squeeze(tf.nn.sigmoid(self.D_real)),
                                      feed_dict={self.realData_: lower_bound,
                                                 self.filler_dims_D: batch_filler_indicator})
        D_upper_bound = self.sess.run(tf.squeeze(tf.nn.sigmoid(self.D_real)),
                                      feed_dict={self.realData_: upper_bound,
                                                 self.filler_dims_D: batch_filler_indicator})

        D_zero = self.sess.run(tf.squeeze(tf.nn.sigmoid(self.D_real)),
                               feed_dict={self.realData_: zero_data, self.filler_dims_D: batch_filler_indicator})
        D_random = self.sess.run(tf.squeeze(tf.nn.sigmoid(self.D_real)),
                                 feed_dict={self.realData_: random_data, self.filler_dims_D: batch_filler_indicator})
        # filler=1通常会更假

        d_info = [round(np.mean(D_real), 2), round(np.mean(D_fake), 2),
                  [round(np.mean(D_lower_bound), 2), round(np.mean(D_upper_bound), 2)],
                  round(np.mean(D_zero), 2), round(np.mean(D_random), 2)]
        # s = ["T:", "G:", "s=5:", "s=0:", "s=random:", "s=5,f=1:"]
        # s = ["real:", "fake:", "seed=5:", "seed=0:", "seed=random:", "seed=5,filler=1:"]
        # d_info = ' '.join([str(d_info[i]) for i in range(len(d_info))])  # s[i]+ str(d_info[i])

        #
        fn_float_to_str = lambda x: str(round(x, 2))
        g_out_seed = list(map(fn_float_to_str, np.mean(info[0], 0)))  # [round(i, 2) for i in np.mean(info[0], 0)]

        #

        g = info[0]
        var_col = list(map(fn_float_to_str, np.var(g, 0)))
        var_row = round(np.mean(np.var(g, 1)), 2)
        log_info = "\tg_out_seed:" + ','.join(g_out_seed), "\tvar_col : " + ','.join(var_col) + "\tvar_row : " + str(
            var_row)

        return total_loss_d, d_info, log_info

    def filler_sampler(self, uid_list):
        if self.filler_method == 0:
            batch_filler_indicator = []
            for uid in uid_list:
                # filler_candi = np.array(
                #     list(set(self.filler_candi_list) & set(self.train_mask[uid].nonzero()[0].tolist())))
                filler_candi = np.array(list(set(self.filler_candi_list)
                                             & set(self.dataset_class.train_matrix[uid].toarray()[0].nonzero()[0])))
                #
                if len(filler_candi) > self.filler_num:
                    filler_candi = np.random.choice(filler_candi, size=self.filler_num, replace=False)
                filler_indicator = [1 if iid in filler_candi else 0 for iid in range(self.num_item)]
                batch_filler_indicator.append(filler_indicator)
            return batch_filler_indicator
        else:
            return self.filler_sampler_method(uid_list)

    def filler_sampler_method(self, uid_list):
        batch_filler_indicator = []
        for uid in uid_list:
            # filler_candi = np.array(
            #     list(set(self.filler_candi_list) & set(self.train_mask[uid].nonzero()[0].tolist())))
            filler_candi = np.array(list(set(self.filler_candi_list)
                                         & set(self.dataset_class.train_matrix[uid].toarray()[0].nonzero()[0])))

            if len(filler_candi) > self.filler_num:
                # sample using a specific method
                # -------------------------
                prob = self.item_avg[filler_candi] if self.filler_method == 1 \
                    else self.item_pop[filler_candi] if self.filler_method == 2 \
                    else self.item_big_cap[filler_candi] if self.filler_method == 3 \
                    else None
                prob = None if prob is None else prob / sum(prob)
                # -------------------------

                filler_candi = np.random.choice(filler_candi, size=self.filler_num, replace=False, p=prob)
            filler_indicator = [1 if iid in filler_candi else 0 for iid in range(self.num_item)]
            batch_filler_indicator.append(filler_indicator)
        return batch_filler_indicator

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def add_loss_summary(self, type="G", info=None):
        # , total_loss_g, total_g_loss_gan, total_g_loss_seed, total_g_loss_reconstruct,total_g_loss_l2):
        if type == "G":
            tf.summary.scalar('Generator/adversarial', self.g_loss_gan)
            tf.summary.scalar('Generator/seed', self.g_loss_seed)
            tf.summary.scalar('Generator/selected_reconstruct', self.g_loss_reconstruct_seed)
            tf.summary.scalar('Generator/l2_normal', self.g_loss_l2)
            tf.summary.scalar('Generator/Sum', self.g_loss)
            self.G_loss_merged = tf.summary.merge_all()

        elif type == 'D':
            total_loss_d, total_loss_d_real, total_loss_d_fake = info
            loss_summary = []
            tag_list = ['Discriminator/Sum', 'Discriminator/real', 'Discriminator/fake']
            simple_value_list = [total_loss_d, total_loss_d_real, total_loss_d_fake]
            for i in range(3):
                loss_summary.append(tf.Summary.Value(tag=tag_list[i], simple_value=simple_value_list[i]))
            self.wirter.add_summary(tf.Summary(value=loss_summary), self.epoch * self.step_D + self.epoch_D + 1)
        elif type == 'var':
            var_summary = []
            for i in range(self.selected_num):
                var_summary.append(tf.Summary.Value(tag='Var/' + str(i), simple_value=info[i]))
            self.wirter.add_summary(tf.Summary(value=var_summary), self.step_G * self.epoch + self.epoch_G + 1)
        else:
            print("summary type error")
```

```python cellView="form" id="hqVzuSKciHPR"
#@markdown WGAN
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


# kernel_size = 5 * 5
def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        try:
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
        except ValueError as err:
            msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
            err.args = err.args + (msg,)
            raise
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def gen_random(size):
    # z - N(0,100)
    return np.random.normal(0, 100, size=size)


class WGAN(object):
    def __init__(self, sess, dataset_class,batch_size=64, height=29, width=58, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, max_to_keep=1):
        self.sess = sess
        self.dataset_class = dataset_class
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.max_to_keep = max_to_keep

        self.build_model()

    def build_model(self):
        self.inputs = tf.placeholder(tf.float32,
                                     [self.batch_size, self.height, self.width, 1],
                                     name='real_images')
        inputs = self.inputs
        # 生成器
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.G = self.generator(self.z)
        # 判别器 - real&fake
        self.D, self.D_logits = self.discriminator(inputs, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # def _cross_entropy_loss(self, logits, labels):
        #     xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, labels))
        #     return xentropy
        self.d_loss = tf.reduce_mean(tf.square(self.D_logits - self.D_logits_))
        self.g_loss = tf.reduce_mean(tf.square(self.D_logits_))
        # self.d_loss_real = tf.reduce_mean(
        #     _cross_entropy_loss(self.D_logits, tf.ones_like(self.D)))
        # self.d_loss_fake = tf.reduce_mean(
        #     _cross_entropy_loss(self.D_logits_, tf.zeros_like(self.D_)))
        #
        # self.g_loss = tf.reduce_mean(
        #     _cross_entropy_loss(self.D_logits_, tf.ones_like(self.D_)))
        # self.d_loss = self.d_loss_real + self.d_loss_fake
        #
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

    def train(self, config):
        d_optim = tf.train.RMSPropOptimizer(config.learning_rate, decay=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim =tf.train.RMSPropOptimizer(config.learning_rate, decay=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        train_idxs = list(range(self.dataset_class.train_matrix.shape[0]))
        for epoch in xrange(config.epoch):
            np.random.shuffle(train_idxs)
            for i in range(len(train_idxs) // self.batch_size):
                cur_idxs = train_idxs[i * self.batch_size:(i + 1) * self.batch_size]
                batch_inputs = self.dataset_class.train_matrix[cur_idxs].toarray()
                # transform range&shape
                batch_inputs = (batch_inputs - 2.5) / 2.5
                batch_inputs = np.reshape(batch_inputs, [self.batch_size, self.height, self.width, 1])
                # batch_inputs = np.random.random_sample([self.batch_size, self.height, self.width, 1])
                batch_z = gen_random(size=[config.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _ = self.sess.run(d_optim, feed_dict={self.inputs: batch_inputs, self.z: batch_z})

                # Update G network
                _ = self.sess.run(g_optim, feed_dict={self.z: batch_z})

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)

                errD= self.d_loss.eval({self.inputs: batch_inputs,self.z: batch_z})
                # errD_real = self.d_loss_real.eval({self.inputs: batch_inputs})
                errG = self.g_loss.eval({self.z: batch_z})

                print("Epoch:[%2d/%2d]d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, config.epoch, errD, errG))

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            # 论文中给的判别器结构:[conv+BN+LeakyRelu[64,128,256,512]]+[FC]+[sigmoid]
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.height, self.width
            # CONV stride=2
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # FC of 2*4*512&ReLU&BN
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)
            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            # four transposed CONV of [256,128,64] &ReLU&BN&kernel_size = 5 * 5
            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))
            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))
            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            # transposed CONV of [1] &tanh
            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, 1], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)
```

```python id="CsioaIWhjGKr"
def train_rec(data_set_name, model_name, attack_method, target_id, is_train):
    if attack_method == "no":
        attack_method = ""
        model_path = "../result/model_ckpt/" + '_'.join([model_name, data_set_name]) + ".ckpt"
    else:
        model_path = "../result/model_ckpt/" + '_'.join([model_name, data_set_name, attack_method]) + ".ckpt"
    path_train = "../data/data_attacked/" + '_'.join([data_set_name, str(target_id), attack_method]) + ".dat"
    path_test = "../data/data/" + data_set_name + "_test.dat"
    if attack_method == "": path_train = "../data/data/" + data_set_name + "_train.dat"

    # load_data
    dataset_class = load_data(path_train=path_train, path_test=path_test,
                              header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=True)
    # train rec
    if model_name in ["IAutoRec", "UAutoRec", "NNMF"]:
        predictions, hit_ratios = rec_trainer(model_name, dataset_class, target_id, is_train, model_path)
    else:
        predictions, hit_ratios = basic_rec(model_name, path_train, path_test, target_id)

    # write to file
    dst_path = "../result/pred_result/" + '_'.join([model_name, data_set_name, str(target_id), attack_method])
    dst_path = dst_path.strip('_')
    target_prediction_writer(predictions, hit_ratios, dst_path)


def parse_arg():
    parser = argparse.ArgumentParser()
    # 数据集名称，用来选择训练数据路径
    parser.add_argument('--dataset', type=str, default='automotive', help='input data_set_name,filmTrust or ml100k')
    # 要攻击的推荐模型的名称,其中NMF_25里的25指item/user的embedding size
    parser.add_argument('--model_name', type=str, default='NMF_25', help='NNMF,IAutoRec,UAutoRec,NMF_25')
    # 攻击方法
    parser.add_argument('--attack_method', type=str, default='G1',
                        help='no,gan,segment,average,random,bandwagon')
    # 目标item id
    # filmTrust:random = [5, 395, 181, 565, 254]    tail = [601, 623, 619, 64, 558]
    # ml100k:random = [62, 1077, 785, 1419, 1257]   tail = [1319, 1612, 1509, 1545, 1373]
    # 5,395,181,565,254,601,623,619,64,558
    # 62,1077,785,1419,1257,1319,1612,1509,1545,1373
    # 1166,1574,759,494,549,1272,1728,1662,450,1456,595,566,764,1187,1816,1478,1721,2294,2413,1148
    parser.add_argument('--target_ids', type=str, default='866',
                        help='attack target')
    # 参数 - 攻击数量，即往数据集里插入多少假用户
    parser.add_argument('--attack_num', type=int, default=50,
                        help='num of attack fake user,50 for ml100k and filmTrust')
    # 参数 - filler数量，可理解为是每个假用户有多少评分
    parser.add_argument('--filler_num', type=int, default=4,
                        help='num of filler items each fake user,90 for ml100k,36 for filmTrust')

    args = parser.parse_args()
    args.target_ids = list(map(int, args.target_ids.split(',')))
    return args


if __name__ == '__main__':
    """parse args"""
    args = parse_arg()

    """train"""
    if args.attack_method == 'no':
        attack_method_ = args.attack_method
    else:
        attack_method_ = '_'.join([args.attack_method, str(args.attack_num), str(args.filler_num)])
    is_train = 1
    train_rec(args.dataset, args.model_name, attack_method_, args.target_ids[0], is_train=is_train)
    for target in args.target_ids[1:]:
        if args.attack_method == 'no':
            is_train = 0
        train_rec(args.dataset, args.model_name, attack_method_, target, is_train=is_train)
```

```python id="fKcCIkdejGHB"
def gan_attack(data_set_name, attack_method, target_id, is_train, write_to_file=1, final_attack_setting=None):
    path_train = '../data/data/' + data_set_name + '_train.dat'
    path_test = '../data/data/' + data_set_name + '_test.dat'
    attack_info_path = ["../data/data/" + data_set_name + "_selected_items",
                        "../data/data/" + data_set_name + "_target_users"]
    # 读取seletced items和target users
    attack_info = load_attack_info(*attack_info_path)
    dataset_class = load_data(path_train=path_train, path_test=path_test, header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=True)
    # 攻击设置
    if len(attack_method.split('_')[1:]) == 2:
        attack_num, filler_num = map(int, attack_method.split('_')[1:])
        filler_method = 0
    else:
        attack_num, filler_num, filler_method = map(int, attack_method.split('_')[1:])
    # 0:重构 1:重构+seed
    loss_setting = int(attack_method.split('_')[0][-1])
    selected_items = attack_info[target_id][0]
    model_path = "../result/model_ckpt/" + '_'.join([data_set_name, attack_method, str(target_id)]) + ".ckpt"

    #
    gan_attacker = Train_G_Attacker(dataset_class, params_D=None, params_G=None, target_id=target_id,
                                    selected_id_list=selected_items,
                                    filler_num=filler_num, attack_num=attack_num, filler_method=filler_method,
                                    loss_setting=loss_setting)
    # if is_train:
    #     fake_profiles = gan_attacker.execute(is_train=True, model_path=model_path)
    # else:
    #     fake_profiles, real_profiles = gan_attacker.execute(is_train=False, model_path=model_path)
    #     if write_to_file == 0:
    #         return fake_profiles, real_profiles
    fake_profiles, real_profiles, filler_indicator = gan_attacker.execute(is_train=is_train, model_path=model_path,
                                                                          final_attack_setting=final_attack_setting)
    gan_attacker.sess.close()
    # """inject and write to file"""
    if write_to_file == 1:
        dst_path = "../data/data_attacked/" + '_'.join([data_set_name, str(target_id), attack_method]) + ".dat"
        attacked_file_writer(path_train, dst_path, fake_profiles, dataset_class.n_users)
    return fake_profiles, real_profiles, filler_indicator


def parse_arg():
    parser = argparse.ArgumentParser()
    # 数据集名称，用来选择训练数据路径
    parser.add_argument('--dataset', type=str, default='automotive', help='filmTrust/ml100k/grocery')
    # 目标item
    # filmTrust:random = [5, 395, 181, 565, 254]    tail = [601, 623, 619, 64, 558]
    # ml100k:random = [62, 1077, 785, 1419, 1257]   tail = [1319, 1612, 1509, 1545, 1373]
    # 5,395,181,565,254,601,623,619,64,558
    # 62,1077,785,1419,1257,1319,1612,1509,1545,1373
    # 1166,1574,759,494,549,1272,1728,1662,450,1456,595,566,764,1187,1816,1478,1721,2294,2413,1148
    # 88,22,122,339,1431,1141,1656,477,1089,866
    parser.add_argument('--target_ids', type=str, default='88,22,122,339,1431,1141,1656,477,1089,866',
                        help='attack target list')
    # 参数 - 攻击数量，即往数据集里插入多少假用户
    parser.add_argument('--attack_num', type=int, default=50,
                        help='num of attack fake user,50 for ml100k and filmTrust')
    # 参数 - filler数量，可理解为是每个假用户有多少评分
    parser.add_argument('--filler_num', type=int, default=4,
                        help='num of filler items each fake user,90 for ml100k,36 for filmTrust')
    # 参数 - 选择filler item的方法，0是随机
    parser.add_argument('--filler_method', type=str, default='', help='0/1/2/3')
    # 生成的攻击结果写入文件还是返回numpy矩阵，这里设置为1就好
    parser.add_argument('--write_to_file', type=int, default=1, help='write to fake profile to file or return array')
    # 0：损失函数只用重构损失,1：损失函数用重构损失+seed损失
    parser.add_argument('--loss', type=int, default=1, help='0:reconstruction,1:reconstruction+seed')
    #
    args = parser.parse_args()
    #
    args.target_ids = list(map(int, args.target_ids.split(',')))
    return args


if __name__ == '__main__':
    """parse args"""
    args = parse_arg()
    """train"""
    is_train = 1
    attack_method = '_'.join(
        ['G' + str(args.loss), str(args.attack_num), str(args.filler_num), str(args.filler_method)]).strip('_')
    #
    for target_id in args.target_ids:
        """读取生成攻击时的sample的filler"""
        attackSetting_path = '_'.join(map(str, [args.dataset, args.attack_num, args.filler_num, target_id]))
        attackSetting_path = "../data/data_attacked/" + attackSetting_path + '_attackSetting'
        real_profiles, filler_indicator = np.load(attackSetting_path + '.npy')
        final_attack_setting = [args.attack_num, real_profiles, filler_indicator]

        """训练模型并注入攻击"""
        _ = gan_attack(args.dataset, attack_method, target_id, is_train,
                       write_to_file=args.write_to_file,
                       final_attack_setting=final_attack_setting)
```

```python id="_Esdgb3_jGER"
def gan_attack(data_set_name, attack_method, target_id, is_train, write_to_file=1, final_attack_setting=None):
    # 路径设置
    path_train = '../data/data/' + data_set_name + '_train.dat'
    path_test = '../data/data/' + data_set_name + '_test.dat'
    attack_info_path = ["../data/data/" + data_set_name + "_selected_items",
                        "../data/data/" + data_set_name + "_target_users"]
    model_path = "../result/model_ckpt/" + '_'.join([data_set_name, attack_method, str(target_id)]) + ".ckpt"

    # 读取seletced items和target users
    attack_info = load_attack_info(*attack_info_path)
    dataset_class = load_data(path_train=path_train, path_test=path_test, header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=True)
    # 攻击设置
    if len(attack_method.split('_')[1:]) == 2:
        attack_num, filler_num = map(int, attack_method.split('_')[1:])
        filler_method = 0
    else:
        attack_num, filler_num, filler_method = map(int, attack_method.split('_')[1:])
    selected_items = attack_info[target_id][0]

    #
    gan_attacker = Train_GAN_Attacker(dataset_class, params_D=None, params_G=None, target_id=target_id,
                                      selected_id_list=selected_items,
                                      filler_num=filler_num, attack_num=attack_num, filler_method=filler_method)
    #
    # if is_train:
    #     # 训练->模型保存->生成fake_profiles
    #     fake_profiles = gan_attacker.execute(is_train=True, model_path=model_path,
    #                                          final_attack_setting=final_attack_setting)
    # else:
    #     # restore>模型保存->生成fake_profiles
    #     fake_profiles, real_profiles = gan_attacker.execute(is_train=False, model_path=model_path,
    #                                                         final_attack_setting=final_attack_setting)
    fake_profiles, real_profiles, filler_indicator = gan_attacker.execute(is_train=is_train, model_path=model_path,
                                                                          final_attack_setting=final_attack_setting)
    gan_attacker.sess.close()

    # """inject and write to file"""
    if write_to_file == 1:
        dst_path = "../data/data_attacked/" + '_'.join([data_set_name, str(target_id), attack_method]) + ".dat"
        attacked_file_writer(path_train, dst_path, fake_profiles, dataset_class.n_users)
    return fake_profiles, real_profiles, filler_indicator


def parse_arg():
    parser = argparse.ArgumentParser()
    # 数据集名称，用来选择训练数据路径
    parser.add_argument('--dataset', type=str, default='ml100k', help='filmTrust/ml100k/grocery')
    # 目标item
    # filmTrust:random = [5, 395, 181, 565, 254]    tail = [601, 623, 619, 64, 558]
    # ml100k:random = [62, 1077, 785, 1419, 1257]   tail = [1319, 1612, 1509, 1545, 1373]
    # 5,395,181,565,254,601,623,619,64,558
    # 62,1077,785,1419,1257,1319,1612,1509,1545,1373
    parser.add_argument('--target_ids', type=str, default='62,1077,785,1419,1257,1319,1612,1509,1545,1373',
                        help='attack target list')
    # 参数 - 攻击数量，即往数据集里插入多少假用户
    parser.add_argument('--attack_num', type=int, default=50,
                        help='num of attack fake user,50 for ml100k and filmTrust')
    # 参数 - filler数量，可理解为是每个假用户有多少评分
    parser.add_argument('--filler_num', type=int, default=90,
                        help='num of filler items each fake user,90 for ml100k,36 for filmTrust')
    # 参数 - 选择filler item的方法，0是随机
    parser.add_argument('--filler_method', type=str, default='', help='0/1/2/3')
    # 生成的攻击结果写入文件还是返回numpy矩阵，这里设置为1就好
    parser.add_argument('--write_to_file', type=int, default=1, help='write to fake profile to file or return array')
    #
    args = parser.parse_args()
    #
    args.target_ids = list(map(int, args.target_ids.split(',')))
    return args


if __name__ == '__main__':
    """parse args"""
    args = parse_arg()
    """train"""
    is_train = 1
    attack_method = '_'.join(['gan', str(args.attack_num), str(args.filler_num), str(args.filler_method)]).strip('_')

    #
    for target_id in args.target_ids:
        """读取生成攻击时的sample的filler"""
        attackSetting_path = '_'.join(map(str, [args.dataset, args.attack_num, args.filler_num, target_id]))
        attackSetting_path = "../data/data_attacked/" + attackSetting_path + '_attackSetting'
        real_profiles, filler_indicator = np.load(attackSetting_path + '.npy')
        final_attack_setting = [args.attack_num, real_profiles, filler_indicator]

        """训练模型并注入攻击"""
        _ = gan_attack(args.dataset, attack_method, target_id, is_train,
                       write_to_file=args.write_to_file,
                       final_attack_setting=final_attack_setting)
```

```python id="sE32fhaujGBd"
def get_data(data_set_name):
    path_train = '../data/data/' + data_set_name + '_train.dat'
    path_test = '../data/data/' + data_set_name + '_test.dat'
    dataset_class = load_data(path_train=path_train, path_test=path_test,
                              header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=False)
    attack_info_path = ["../data/data/" + data_set_name + "_selected_items",
                        "../data/data/" + data_set_name + "_target_users"]
    attack_info = load_attack_info(*attack_info_path)
    return dataset_class, attack_info


def baseline_attack(dataset_class, attack_info, attack_method, target_id, bandwagon_selected,
                    fixed_filler_indicator=None):
    """load data"""
    selected_ids, target_users = attack_info[target_id]
    attack_model, attack_num, filler_num = attack_method.split('_')
    attack_num, filler_num = int(attack_num), int(filler_num)

    """attack class"""
    global_mean, global_std, item_means, item_stds = dataset_class.get_all_mean_std()
    baseline_attacker = BaselineAttack(attack_num, filler_num, dataset_class.n_items, target_id,
                                       global_mean, global_std, item_means, item_stds, 5.0, 1.0,
                                       fixed_filler_indicator=fixed_filler_indicator)
    # fake profile array
    fake_profiles = None
    if attack_model == "random":
        fake_profiles = baseline_attacker.RandomAttack()
    elif attack_model == "bandwagon":
        fake_profiles = baseline_attacker.BandwagonAttack(bandwagon_selected)
    elif attack_model == "average":
        fake_profiles = baseline_attacker.AverageAttack()
    elif attack_model == "segment":
        fake_profiles = baseline_attacker.SegmentAttack(selected_ids)
    else:
        print('attack_method error')
        exit()
    return fake_profiles


def parse_arg():
    parser = argparse.ArgumentParser()
    # 数据集名称，用来选择训练数据路径
    parser.add_argument('--dataset', type=str, default='automotive', help='filmTrust/ml100k/grocery')
    # 攻击方法，逗号隔开
    parser.add_argument('--attack_methods', type=str, default='average',
                        help='average,segment,random,bandwagon')
    # 目标item，逗号隔开，这里前五个是随机target后五个是长尾target
    # filmTrust:random = [5, 395, 181, 565, 254]    tail = [601, 623, 619, 64, 558]
    # ml100k:random = [62, 1077, 785, 1419, 1257]   tail = [1319, 1612, 1509, 1545, 1373]
    # 1166,1574,759,494,549,1272,1728,1662,450,1456,595,566,764,1187,1816,1478,1721,2294,2413,1148
    # 62,1077,785,1419,1257,1319,1612,1509,1545,1373
    # 88,22,122,339,1431,1141,1656,477,1089,866
    parser.add_argument('--targets', type=str, default='88,22,122,339,1431,1141,1656,477,1089,866',
                        help='attack_targets')
    # 参数 - 攻击数量，即往数据集里插入多少假用户
    parser.add_argument('--attack_num', type=int, default=50, help='fixed 50')
    # 参数 - filler数量，可理解为是每个假用户有多少评分
    parser.add_argument('--filler_num', type=int, default=4, help='90 for ml100k,36 for filmTrust')
    parser.add_argument('--bandwagon_selected', type=str, default='180,99,49',
                        help='180,99,49 for ml100k,103,98,115 for filmTrust')
    #
    parser.add_argument('--sample_filler', type=int, default=1, help='sample filler')
    #

    args = parser.parse_args()
    #
    args.attack_methods = args.attack_methods.split(',')
    args.targets = list(map(int, args.targets.split(',')))
    args.bandwagon_selected = list(map(int, args.bandwagon_selected.split(',')))
    return args


if __name__ == '__main__':
    """parse args"""
    args = parse_arg()

    """attack"""
    dataset_class, attack_info = get_data(args.dataset)
    # 对每种攻击方法&攻击目标，生成fake profile并写入目标路径
    for target_id in args.targets:
        # 固定filler
        attackSetting_path = '_'.join(map(str, [args.dataset, args.attack_num, args.filler_num, target_id]))
        attackSetting_path = "../data/data_attacked/" + attackSetting_path + '_attackSetting'
        if args.sample_filler:
            gan_attacker = Train_GAN_Attacker(dataset_class, params_D=None, params_G=None, target_id=target_id,
                                              selected_id_list=attack_info[target_id][0],
                                              filler_num=args.filler_num, attack_num=args.attack_num, filler_method=0)
            _, real_profiles, filler_indicator = gan_attacker.execute(is_train=0, model_path='no',
                                                                      final_attack_setting=[args.attack_num,
                                                                                            None, None])

            np.save(attackSetting_path, [real_profiles, filler_indicator])
        else:
            real_profiles, filler_indicator = np.load(attackSetting_path + '.npy')
```

```python id="fsfaOuosjF-Y"
def attack_evaluate(real_preds_path, attacked_preds_file, non_rated_users, target_users):
    #
    names = ['uid', 'rating', 'HR_1', 'HR_3', 'HR_5', 'HR_10', 'HR_20', 'HR_50']
    real_preds = pd.read_csv(real_preds_path, sep='\t', names=names, engine='python')
    attacked_preds = pd.read_csv(attacked_preds_file, sep='\t', names=names, engine='python')
    # pred
    shift_target = np.mean(attacked_preds.iloc[target_users, 1].values - real_preds.iloc[target_users, 1].values)
    shift_all = np.mean(attacked_preds.iloc[non_rated_users, 1].values - real_preds.iloc[non_rated_users, 1].values)
    #
    HR_real_target = real_preds.iloc[target_users, range(2, 8)].mean().values
    HR_real_all = real_preds.iloc[non_rated_users, range(2, 8)].mean().values

    HR_attacked_target = attacked_preds.iloc[target_users, range(2, 8)].mean().values
    HR_attacked_all = attacked_preds.iloc[non_rated_users, range(2, 8)].mean().values
    return shift_target, HR_real_target, HR_attacked_target, shift_all, HR_real_all, HR_attacked_all


def eval_attack(data_set_name, rec_model_name, attack_method, target_id):
    dir = "../result/pred_result/"
    real_preds_path = dir + '_'.join([rec_model_name, data_set_name, str(target_id)])
    attacked_preds_file = real_preds_path + "_" + attack_method
    """
    ml100k
    """
    if data_set_name == 'ml100k':
        path_train = "../data/data/ml100k_train.dat"
        path_test = "../data/data/ml100k_test.dat"
        attack_info_path = ["../data/data/ml100k_selected_items", "../data/data/ml100k_target_users"]
    elif data_set_name == 'filmTrust':
        path_train = "../data/data/filmTrust_train.dat"
        path_test = "../data/data/filmTrust_test.dat"
        attack_info_path = ["../data/data/filmTrust_selected_items", "../data/data/filmTrust_target_users"]

    else:
        path_train = "../data/data/" + data_set_name + "_train.dat"
        path_test = "../data/data/" + data_set_name + "_test.dat"
        attack_info_path = ["../data/data/" + data_set_name + "_selected_items",
                            "../data/data/" + data_set_name + "_target_users"]

    attack_info = load_attack_info(*attack_info_path)
    dataset_class = load_data(path_train=path_train, path_test=path_test, header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=False)

    #
    target_users = attack_info[target_id][1]
    non_rated_users = dataset_class.get_item_nonrated_users(target_id)
    #
    res = attack_evaluate(real_preds_path, attacked_preds_file, non_rated_users, target_users)
    #
    target, all = res[:3], res[3:]
    target_str = '\t'.join([str(target[0]), '\t'.join(map(str, target[1])), '\t'.join(map(str, target[2]))])
    all_str = '\t'.join([str(all[0]), '\t'.join(map(str, all[1])), '\t'.join(map(str, all[2]))])

    # info
    info = '\t'.join([rec_model_name, attack_method, str(target_id)])
    # print(info + '\t' + target_str + '\t' + all_str)
    return info + '\t' + target_str + '\t' + all_str


def parse_arg():
    parser = argparse.ArgumentParser()
    # 数据集名称，用来选择训练数据路径
    parser.add_argument('--dataset', type=str, default='automotive', help='filmTrust/ml100k/office')
    # 参数 - 攻击数量，即往数据集里插入多少假用户
    parser.add_argument('--attack_num', type=int, default=50, help='50 for ml100k and filmTrust')
    # 参数 - filler数量，可理解为是每个假用户有多少评分
    parser.add_argument('--filler_num', type=int, default=4, help='90 for ml100k,36 for filmTrust')
    # 攻击方法
    parser.add_argument('--attack_methods', type=str, default='G0,G1',
                        help='gan,G0,G1,segment,average,random,bandwagon')
    # 目标模型
    parser.add_argument('--rec_model_names', type=str, default='NNMF,IAutoRec,UAutoRec,NMF_25',
                        help='NNMF,IAutoRec,UAutoRec,NMF_25')
    # 目标item，逗号隔开，这里前五个是随机target后五个是长尾target
    # filmTrust:5,395,181,565,254,601,623,619,64,558 - random*5+tail*5
    # ml100k:62,1077,785,1419,1257,1319,1612,1509,1545,1373 - random*5+tail*5
    # 1166,1574,759,494,549,1272,1728,1662,450,1456,595,566,764,1187,1816,1478,1721,2294,2413,1148
    # 88,22,122,339,1431,1141,1656,477,1089,866
    parser.add_argument('--target_ids', type=str, default='88,22,122,339,1431,1141,1656,477,1089,866',
                        help='target_id')

    #
    args = parser.parse_args()
    #
    args.attack_methods = args.attack_methods.split(',')
    args.rec_model_names = args.rec_model_names.split(',')
    args.target_ids = list(map(int, args.target_ids.split(',')))
    return args


if __name__ == '__main__':
    """parse args"""
    args = parse_arg()
    """eval"""
    result = []

    for attack_method in args.attack_methods:
        for rec_model_name in args.rec_model_names:
            for target_id in args.target_ids:
                attack_method_ = '_'.join([attack_method, str(args.attack_num), str(args.filler_num)])
                try:
                    result_ = eval_attack(args.dataset, rec_model_name, attack_method_, target_id)
                    result.append(result_.split('\t'))
                except:
                    print(attack_method, rec_model_name, target_id)

    result = np.array(result).transpose()
    result = pd.DataFrame(dict(zip(range(result.shape[0]), result)))
    result.to_excel(args.dataset + '_performance_all.xls', index=False)
```

```python id="MusordfrjF6r"
def eval_eigen_value(profiles):
    U_T_U = np.dot(profiles.transpose(), profiles)
    eig_val, _ = eig(U_T_U)
    top_10 = [i.real for i in eig_val[:10]]
    return top_10


def get_item_distribution(profiles):
    # [min(max(0, round(i)), 5) for i in a]
    profiles_T = profiles.transpose()
    fn_count = lambda item_vec: np.array(
        [sum([1 if (min(max(0, round(j)), 5) == i) else 0 for j in item_vec]) for i in range(6)])
    fn_norm = lambda item_vec: item_vec / sum(item_vec)
    item_distribution = np.array(list(map(fn_count, profiles_T)))
    item_distribution = np.array(list(map(fn_norm, item_distribution)))
    return item_distribution


def eval_TVD_JS(P, Q):
    # TVD
    dis_TVD = np.mean(np.sum(np.abs(P - Q) / 2, 1))
    # JS
    fn_KL = lambda p, q: scipy.stats.entropy(p, q)
    M = (P + Q) / 2
    js_vec = []
    for iid in range(P.shape[0]):
        p, q, m = P[iid], Q[iid], M[iid]
        js_vec.append((fn_KL(p, m) + fn_KL(q, m)) / 2)
    dis_JS = np.mean(np.array(js_vec))
    return dis_TVD, dis_JS


def print_eigen_result(real_profiles, fake_profiles_gan, baseline_fake_profiles, baseline_methods):
    top_10_res = []
    top_10_real = eval_eigen_value(real_profiles)
    top_10_res.append("real\t" + '\t'.join(map(str, top_10_real)))
    top_10_baseline = []
    for idx in range(len(baseline_fake_profiles)):
        top_10_baseline.append(eval_eigen_value(baseline_fake_profiles[idx]))
        top_10_res.append(baseline_methods[idx] + "\t" + '\t'.join(map(str, top_10_baseline[-1])))
    top_10_gan = eval_eigen_value(fake_profiles_gan)
    # top_10_sample_5 = eval_eigen_value(fake_profiles_sample_5)
    # top_10_real_sample = eval_eigen_value(real_profiles_gan)
    top_10_res.append("gan\t" + '\t'.join(map(str, top_10_gan)))
    # top_10_res.append("sample_5\t" + '\t'.join(map(str, top_10_sample_5)))
    # top_10_res.append("real_sample\t" + '\t'.join(map(str, top_10_real_sample)))
    print("\n".join(top_10_res))


def get_distance_result(target_id, real_profiles, fake_profiles_gan, baseline_fake_profiles, baseline_methods):
    k = ['target_id', 'attack_method', 'dis_TVD', 'dis_JS']
    v = [[], [], [], []]
    res_dis = []
    real_item_distribution = get_item_distribution(real_profiles)
    # real_gan_item_distribution = get_item_distribution(real_profiles_gan)
    fake_gan_distribution = get_item_distribution(fake_profiles_gan)
    # fake_sample_5_distribution = get_item_distribution(fake_profiles_sample_5)
    # dis_TVD, dis_JS = eval_TVD_JS(real_item_distribution, real_gan_item_distribution)
    # res_dis.append('\t'.join(map(str, ["real", "real_gan", dis_TVD, dis_JS])))
    # dis_TVD, dis_JS = eval_TVD_JS(real_gan_item_distribution, fake_gan_distribution)
    # res_dis.append('\t'.join(map(str, ["real_gan", "gan", dis_TVD, dis_JS])))
    # dis_TVD, dis_JS = eval_TVD_JS(real_item_distribution, fake_sample_5_distribution)
    # res_dis.append('\t'.join(map(str, ["real", "sample_5", dis_TVD, dis_JS])))
    # dis_TVD, dis_JS = eval_TVD_JS(real_gan_item_distribution, fake_sample_5_distribution)
    # res_dis.append('\t'.join(map(str, ["real_gan", "sample_5", dis_TVD, dis_JS])))
    dis_TVD, dis_JS = eval_TVD_JS(real_item_distribution, fake_gan_distribution)
    v[1] += ['gan']
    v[2] += [dis_TVD]
    v[3] += [dis_JS]
    # res_dis.append('\t'.join(map(str, [target_id, "gan", dis_TVD, dis_JS])))
    for idx in range(len(baseline_fake_profiles)):
        dis_TVD, dis_JS = eval_TVD_JS(real_item_distribution, get_item_distribution(baseline_fake_profiles[idx]))
        v[1] += [baseline_methods[idx]]
        v[2] += [dis_TVD]
        v[3] += [dis_JS]
        # res_dis.append('\t'.join(map(str, [target_id, baseline_methods[idx], dis_TVD, dis_JS])))
    v[0] = [target_id] * len(v[1])
    result = pd.DataFrame(dict(zip(k, v)))
    # print('\n'.join(res_dis))
    return result


def profiles_generator(target_id, dataset_class, attack_info, bandwagon_selected, sample_num, args, real_profiles,
                       filler_indicator, pre_fix, has_G=False):
    # baseline fake profiles
    baseline_methods = ["segment", "average", "random", "bandwagon"]
    baseline_fake_profiles = []
    for attack_method in baseline_methods:
        attack_model = '_'.join([attack_method, str(sample_num), str(args.filler_num)])
        fake_profiles = baseline_attack(dataset_class, attack_info, attack_model, target_id,
                                        bandwagon_selected, filler_indicator)
        baseline_fake_profiles.append(fake_profiles)

    for attack_method in baseline_methods:
        attack_model = '_'.join([attack_method, str(sample_num), str(args.filler_num)])
        fake_profiles = baseline_attack(dataset_class, attack_info, attack_model, target_id,
                                        bandwagon_selected, None)
        baseline_fake_profiles.append(fake_profiles)
    baseline_methods = baseline_methods + [i + '_rand' for i in baseline_methods]

    final_attack_setting = [sample_num, real_profiles, filler_indicator]
    # new_baseline
    if has_G:
        for attack_method in ['G0' + pre_fix, 'G1' + pre_fix]:
            baseline_methods.append(attack_method)
            fake_profiles_G, _, _ = gan_attack_baseline(args.dataset, attack_method, target_id, False, 0,
                                                        final_attack_setting=final_attack_setting)
            baseline_fake_profiles.append(fake_profiles_G)

    # gan profiles
    attack_method = "gan" + pre_fix
    fake_profiles_gan, _, _ = gan_attack(args.dataset, attack_method, target_id, False, write_to_file=0,
                                         final_attack_setting=final_attack_setting)
    return fake_profiles_gan, baseline_fake_profiles, baseline_methods


def parse_arg():
    parser = argparse.ArgumentParser()
    # 数据集名称，用来选择训练数据路径
    parser.add_argument('--dataset', type=str, default='ml100k',
                        help='input data_set_name,filmTrust or ml100k grocery')
    # 参数 - 攻击数量，即往数据集里插入多少假用户
    parser.add_argument('--attack_num', type=int, default=50,
                        help='num of attack fake user,50 for ml100k and filmTrust')
    # 参数 - filler数量，可理解为是每个假用户有多少评分
    parser.add_argument('--filler_num', type=int, default=90,
                        help='num of filler items each fake user,90 for ml100k,36 for filmTrust')
    # filmTrust:5,395,181,565,254,601,623,619,64,558 - random*5+tail*5
    # ml100k:62,1077,785,1419,1257,1319,1612,1509,1545,1373 - random*5+tail*5
    parser.add_argument('--targets', type=str, default='62,1077,785,1419,1257,1319,1612,1509,1545,1373', help='attack_targets')
    parser.add_argument('--bandwagon_selected', type=str, default='180,99,49',
                        help='180,99,49 for ml100k,103,98,115 for filmTrust')
    #
    args = parser.parse_args()
    #
    args.targets = list(map(int, args.targets.split(',')))
    args.bandwagon_selected = list(map(int, args.bandwagon_selected.split(',')))
    return args


if __name__ == '__main__':
    """
    step1 - load data
    step2 - 共所有攻击方法生成评分矩阵
    step3 - 真假评分矩阵的距离度量
    """

    #
    """parse args"""
    args = parse_arg()
    pre_fix = '_' + str(args.attack_num) + '_' + str(args.filler_num)

    """step1 - load data"""
    path_train = "../data/data/" + args.dataset + "_train.dat"
    path_test = "../data/data/" + args.dataset + "_test.dat"
    attack_info_path = ["../data/data/" + args.dataset + "_selected_items",
                        "../data/data/" + args.dataset + "_target_users"]
    dataset_class = load_data(path_train=path_train, path_test=path_test, header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=False)
    attack_info = load_attack_info(*attack_info_path)

    sample_num = dataset_class.n_users
    result = None
    for target_id in args.targets:
        selected = attack_info[target_id][0]
        """step2.1 - 生成固定的filler"""
        attackSetting_path = '_'.join(map(str, [args.dataset, sample_num, args.filler_num, target_id]))
        attackSetting_path = "../data/data_attacked/" + attackSetting_path + '_attackSetting'
        gan_attacker = Train_GAN_Attacker(dataset_class, params_D=None, params_G=None, target_id=target_id,
                                          selected_id_list=selected, filler_num=args.filler_num,
                                          attack_num=args.attack_num, filler_method=0)
        _, real_profiles, filler_indicator = gan_attacker.execute(is_train=0, model_path='no',
                                                                  final_attack_setting=[sample_num, None, None])
        np.save(attackSetting_path, [real_profiles, filler_indicator])
        """step2.2 - 为所有攻击方法生成评分矩阵"""
        fake_profiles_gan, baseline_fake_profiles, baseline_methods \
            = profiles_generator(target_id, dataset_class, attack_info, args.bandwagon_selected, sample_num, args,
                                 real_profiles, filler_indicator, pre_fix, has_G=True)

        """step3 - 真假评分矩阵的距离度量"""
        # result_ = get_distance_result(target_id, real_profiles, fake_profiles_gan, baseline_fake_profiles,
        #                               baseline_methods)
        result_ = get_distance_result(target_id, dataset_class.train_matrix.toarray(), fake_profiles_gan,
                                      baseline_fake_profiles,
                                      baseline_methods)

        result = result_ if result is None else pd.concat([result, result_])
    print(result)
    result.to_excel(args.dataset + '_distance_lianyun.xls', index=False)
```

```python id="oHPOQjPjjl1P"
def eval_eigen_value(profiles):
    U_T_U = np.dot(profiles.transpose(), profiles)
    eig_val, _ = eig(U_T_U)
    top_10 = [i.real for i in eig_val[:10]]
    return top_10


def get_item_distribution(profiles):
    # [min(max(0, round(i)), 5) for i in a]
    profiles_T = profiles.transpose()
    fn_count = lambda item_vec: np.array(
        [sum([1 if (min(max(0, round(j)), 5) == i) else 0 for j in item_vec]) for i in range(6)])
    fn_norm = lambda item_vec: item_vec / sum(item_vec)
    item_distribution = np.array(list(map(fn_count, profiles_T)))
    item_distribution = np.array(list(map(fn_norm, item_distribution)))
    return item_distribution


def eval_TVD_JS(P, Q):
    # TVD
    dis_TVD = np.mean(np.sum(np.abs(P - Q) / 2, 1))
    # JS
    fn_KL = lambda p, q: scipy.stats.entropy(p, q)
    M = (P + Q) / 2
    js_vec = []
    for iid in range(P.shape[0]):
        p, q, m = P[iid], Q[iid], M[iid]
        js_vec.append((fn_KL(p, m) + fn_KL(q, m)) / 2)
    dis_JS = np.mean(np.array(js_vec))
    return dis_TVD, dis_JS


def print_eigen_result(real_profiles, fake_profiles_gan, baseline_fake_profiles, baseline_methods):
    top_10_res = []
    top_10_real = eval_eigen_value(real_profiles)
    top_10_res.append("real\t" + '\t'.join(map(str, top_10_real)))
    top_10_baseline = []
    for idx in range(len(baseline_fake_profiles)):
        top_10_baseline.append(eval_eigen_value(baseline_fake_profiles[idx]))
        top_10_res.append(baseline_methods[idx] + "\t" + '\t'.join(map(str, top_10_baseline[-1])))
    top_10_gan = eval_eigen_value(fake_profiles_gan)
    # top_10_sample_5 = eval_eigen_value(fake_profiles_sample_5)
    # top_10_real_sample = eval_eigen_value(real_profiles_gan)
    top_10_res.append("gan\t" + '\t'.join(map(str, top_10_gan)))
    # top_10_res.append("sample_5\t" + '\t'.join(map(str, top_10_sample_5)))
    # top_10_res.append("real_sample\t" + '\t'.join(map(str, top_10_real_sample)))
    print("\n".join(top_10_res))


def get_distance_result(target_id, real_profiles, fake_profiles_list, method_name):
    k = ['target_id', 'attack_method', 'dis_TVD', 'dis_JS']
    v = [[], [], [], []]
    res_dis = []
    real_item_distribution = get_item_distribution(real_profiles)
    for idx in range(len(fake_profiles_list)):
        dis_TVD, dis_JS = eval_TVD_JS(real_item_distribution, get_item_distribution(fake_profiles_list[idx]))
        v[1] += [method_name[idx]]
        v[2] += [dis_TVD]
        v[3] += [dis_JS]
    v[0] = [target_id] * len(v[1])
    result = pd.DataFrame(dict(zip(k, v)))
    return result


def profiles_generator(target_id, dataset_class, attack_info, bandwagon_selected, sample_num, args, real_profiles,
                       filler_indicator, pre_fix, has_G=False):
    # baseline fake profiles
    baseline_methods = ["segment", "average", "random", "bandwagon"]
    baseline_fake_profiles = []
    for attack_method in baseline_methods:
        attack_model = '_'.join([attack_method, str(sample_num), str(args.filler_num)])
        fake_profiles = baseline_attack(dataset_class, attack_info, attack_model, target_id,
                                        bandwagon_selected, filler_indicator)
        baseline_fake_profiles.append(fake_profiles)

    for attack_method in baseline_methods:
        attack_model = '_'.join([attack_method, str(sample_num), str(args.filler_num)])
        fake_profiles = baseline_attack(dataset_class, attack_info, attack_model, target_id,
                                        bandwagon_selected, None)
        baseline_fake_profiles.append(fake_profiles)
    baseline_methods = baseline_methods + [i + '_rand' for i in baseline_methods]

    final_attack_setting = [sample_num, real_profiles, filler_indicator]
    # new_baseline
    if has_G:
        for attack_method in ['G0' + pre_fix, 'G1' + pre_fix]:
            baseline_methods.append(attack_method)
            fake_profiles_G, _, _ = gan_attack_baseline(args.dataset, attack_method, target_id, False, 0,
                                                        final_attack_setting=final_attack_setting)
            baseline_fake_profiles.append(fake_profiles_G)

    # gan profiles
    attack_method = "gan" + pre_fix
    fake_profiles_gan, _, _ = gan_attack(args.dataset, attack_method, target_id, False, write_to_file=0,
                                         final_attack_setting=final_attack_setting)
    return fake_profiles_gan, baseline_fake_profiles, baseline_methods


def parse_arg():
    parser = argparse.ArgumentParser()
    # 数据集名称，用来选择训练数据路径
    parser.add_argument('--dataset', type=str, default='ml100k',
                        help='input data_set_name,filmTrust or ml100k grocery')
    # 参数 - 攻击数量，即往数据集里插入多少假用户
    parser.add_argument('--attack_num', type=int, default=50,
                        help='num of attack fake user,50 for ml100k and filmTrust')
    # 参数 - filler数量，可理解为是每个假用户有多少评分
    parser.add_argument('--filler_num', type=int, default=90,
                        help='num of filler items each fake user,90 for ml100k,36 for filmTrust')
    # filmTrust:5,395,181,565,254,601,623,619,64,558 - random*5+tail*5
    # ml100k:62,1077,785,1419,1257,1319,1612,1509,1545,1373 - random*5+tail*5
    parser.add_argument('--targets', type=str, default='62,1077,785,1419,1257,1319,1612,1509,1545,1373',
                        help='attack_targets')
    parser.add_argument('--bandwagon_selected', type=str, default='180,99,49',
                        help='180,99,49 for ml100k,103,98,115 for filmTrust')
    #
    args = parser.parse_args()
    #
    args.targets = list(map(int, args.targets.split(',')))
    args.bandwagon_selected = list(map(int, args.bandwagon_selected.split(',')))
    return args


if __name__ == '__main__':
    """
    step1 - load data
    step2 - 共所有攻击方法生成评分矩阵
    step3 - 真假评分矩阵的距离度量
    """

    #
    """parse args"""
    args = parse_arg()
    pre_fix = '_' + str(args.attack_num) + '_' + str(args.filler_num)

    """step1 - load data"""
    path_train = "../data/data/" + args.dataset + "_train.dat"
    path_test = "../data/data/" + args.dataset + "_test.dat"
    attack_info_path = ["../data/data/" + args.dataset + "_selected_items",
                        "../data/data/" + args.dataset + "_target_users"]
    dataset_class = load_data(path_train=path_train, path_test=path_test, header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=False)
    attack_info = load_attack_info(*attack_info_path)

    sample_num = dataset_class.n_users
    result = None
    for target_id in args.targets:
        selected = attack_info[target_id][0]
        """step2.1 - real_profiles"""
        gan_attacker = Train_GAN_Attacker(dataset_class, params_D=None, params_G=None, target_id=target_id,
                                          selected_id_list=selected, filler_num=args.filler_num,
                                          attack_num=args.attack_num, filler_method=0)
        _, real_profiles, filler_indicator = gan_attacker.execute(is_train=0, model_path='no',
                                                                  final_attack_setting=[sample_num, None, None])
        """step2.2 - 为所有攻击方法生成评分矩阵"""
        # dcgan数据
        dir = None
        fake_profiles_list = []
        method_list = []
        for attack_method in ['IAutoRec', 'UAutoRec', 'NNMF', 'NMF_25']:
            path_dcgan = dir + 'D-%s-ml100k\\ml100k_%d_dcgan_50_90.dat' % (attack_method, target_id)
            dataset_class_dcgan = load_data(path_train=path_dcgan, path_test=path_test,
                                            header=['user_id', 'item_id', 'rating'],
                                            sep='\t', print_log=False)
            fake_profiles_ = dataset_class_dcgan.train_matrix.toarray()[dataset_class.n_users:]
            while fake_profiles_.shape[0] < dataset_class.n_users:
                fake_profiles_ = np.concatenate([fake_profiles_, fake_profiles_])
            fake_profiles_ = fake_profiles_[:dataset_class.n_users]
            # 同样的方法读入wgan数据
            path_wgan = dir + 'W-%s-ml100k\\ml100k_%d_wgan_50_90.dat' % (attack_method, target_id)
            dataset_class_dcgan = load_data(path_train=path_dcgan, path_test=path_test,
                                            header=['user_id', 'item_id', 'rating'],
                                            sep='\t', print_log=False)
            fake_profiles_w = dataset_class_dcgan.train_matrix.toarray()[dataset_class.n_users:]
            while fake_profiles_w.shape[0] < dataset_class.n_users:
                fake_profiles_w = np.concatenate([fake_profiles_w, fake_profiles_w])
            fake_profiles_w = fake_profiles_w[:dataset_class.n_users]
            #
            fake_profiles_list += [fake_profiles_, fake_profiles_w]
            method_list += ['dcgan', 'wgan']
        """step3 - 真假评分矩阵的距离度量"""
        result_ = get_distance_result(target_id, real_profiles, fake_profiles_list, method_list)
        result = result_ if result is None else pd.concat([result, result_])
    print(result)
    result.groupby('attack_method').mean().to_excel(args.dataset + '_distance_new.xls', index=False)
```

```python id="nch-c8tAjltW"
columns = ['Rec_model', 'attack_method', 'target_id']
# 后面是攻击效果
hr = ['HR_1', 'HR_3', 'HR_5', 'HR_10', 'HR_20', 'HR_50']
hr_ori = [i + '_ori' for i in hr]
# 段内用户的 预估分偏移量+攻击前的HR+攻击后的HR
columns += [i + '_inseg' for i in ['shift'] + hr_ori + hr]
# 全部用户的 预估分偏移量+攻击前的HR+攻击后的HR
columns += [i + '_all' for i in ['shift'] + hr_ori + hr]
# 需要统计的数值指标：
columns_r = [i + '_inseg' for i in ['shift'] + hr] + [i + '_all' for i in ['shift'] + hr]
""""""
# data = pd.read_excel('filmTrust_distance.xls')
# data.groupby('attack_method').mean()[['dis_TVD','dis_JS']].to_excel('filmTrust_distance_avg.xls')

# data = pd.read_excel('ml100k_performance_all.xls')
# data = pd.read_excel('../result_ijcai/filmTrust_performance_all.xls')
# data = pd.read_excel('../result_ijcai/ml100k_performance_all.xls')
# data = pd.read_excel('office_performance_all.xls')
data = pd.read_excel('automotive_performance_all.xls')
data.columns = columns
data = data[['Rec_model', 'attack_method', 'target_id', 'shift_inseg', 'HR_10_inseg', 'shift_all', 'HR_10_all']]
# target_type_dict = dict(
#     zip([62, 1077, 785, 1419, 1257] + [1319, 1612, 1509, 1545, 1373], ['random'] * 5 + ['tail'] * 5))
# target_type_dict = dict(zip([5, 395, 181, 565, 254] + [601, 623, 619, 64, 558], ['random'] * 5 + ['tail'] * 5))
target_type_dict = dict(zip([1141, 1656, 477, 1089, 866] + [88, 22, 122, 339, 1431], ['random'] * 5 + ['tail'] * 5))
data['target_type'] = data.target_id.apply(lambda x: target_type_dict[x])
data['attack_method'] = data.attack_method.apply(lambda x: x.split('_')[0])
result = data.groupby(['Rec_model','attack_method', 'target_type']).mean()[['shift_all', 'HR_10_all']]
result.to_excel('ml100k_performance_0119_sample_strategy.xlsx')
exit()
```
