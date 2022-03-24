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

<!-- #region id="2TRsPnfVTuGF" -->
# UltraGCN on ML-100k
<!-- #endregion -->

<!-- #region id="kDyGauIBgOO4" -->
## Setup
<!-- #endregion -->

<!-- #region id="J6YVYq5ggptl" -->
### Imports
<!-- #endregion -->

```python id="X49IEpJHiOSX" executionInfo={"status": "ok", "timestamp": 1635783462857, "user_tz": -330, "elapsed": 24781, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
from IPython import embed
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser
import time
from torch.utils.tensorboard import SummaryWriter
```

```python colab={"base_uri": "https://localhost:8080/"} id="OfBBansIXdbt" executionInfo={"status": "ok", "timestamp": 1635784729121, "user_tz": -330, "elapsed": 4376, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1675cc6b-ac3a-4015-e26c-b8b1dd7bfb63"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv -u -t -d
```

<!-- #region id="LHdL4aitgrfK" -->
### Params
<!-- #endregion -->

```python id="zLp8oVaZgsEu" executionInfo={"status": "ok", "timestamp": 1635783536872, "user_tz": -330, "elapsed": 508, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class Args:
    embedding_dim = 64
    ii_neighbor_num = 10
    model_save_path = './ultragcn_movielens.pt'
    max_epoch = 50
    enable_tensorboard = True
    initial_weight = 1e-4
    dataset = 'movielens'
    train_file_path = './train_ml.txt'
    gpu = '0' #need to specify the avaliable gpu index. If gpu is not avaliable, we will use cpu.
    lr = 1e-3
    batch_size = 128
    early_stop_epoch = 15
    w1 = 1e-4
    w2 = 1
    w3 = 1
    w4 = 1e-4
    negative_num = 50
    negative_weight = 50
    gamma = 1e-4 #weight of l2 normalization
    lambda_ = 2.75 #weight of L_I
    sampling_sift_pos = False #whether to sift the pos item when doing negative sampling
    test_batch_size = 128 #can be customized to your gpu size
    topk = 20
    test_file_path = './test_ml.txt'
```

```python id="nFPE3_IQi7wb" executionInfo={"status": "ok", "timestamp": 1635783539114, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args = Args()
```

```python id="Bzf5uleni67z" executionInfo={"status": "ok", "timestamp": 1635783539118, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
args.device = torch.device('cuda:'+ args.gpu if torch.cuda.is_available() else "cpu")
```

<!-- #region id="uCtXwV2lguvx" -->
## Dataset
<!-- #endregion -->

<!-- #region id="JgiLkumHEZM-" -->
Amazon-books
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="0h_a7Onwgx4d" executionInfo={"status": "ok", "timestamp": 1635783469430, "user_tz": -330, "elapsed": 6603, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="0a2aae3e-29e0-43fa-bf58-3270dc90adbd"
!rm -r *.txt

!wget -q --show-progress https://github.com/xue-pai/UltraGCN/raw/main/data/amazon/train.txt
!wget -q --show-progress https://github.com/xue-pai/UltraGCN/raw/main/data/amazon/test.txt
!wget -q --show-progress https://github.com/xue-pai/UltraGCN/raw/main/data/amazon/user_list.txt
!wget -q --show-progress https://github.com/xue-pai/UltraGCN/raw/main/data/amazon/item_list.txt

!wc -l train.txt
!wc -l test.txt
!wc -l user_list.txt
!wc -l item_list.txt

!head user_list.txt
```

<!-- #region id="EYZMnfFTJmeR" -->
Movielens
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="0G_cc6AMPAIZ" executionInfo={"status": "ok", "timestamp": 1635783495152, "user_tz": -330, "elapsed": 22491, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="baa9381a-d855-4cfa-b147-50648abf5783"
!pip install -q git+https://github.com/sparsh-ai/recochef
!wget -q --show-progress http://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip -qq ml-100k.zip

import os
import csv 
import pandas as pd
from pathlib import Path
from recochef.preprocessing.split import chrono_split

df = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['USERID','ITEMID','RATING','TIMESTAMP'])
df_train, df_test = chrono_split(df, ratio=0.8)

def preprocess(data):
  data = data.copy()
  data = data.sort_values(by=['USERID','TIMESTAMP'])
  data['USERID'] = data['USERID'] - 1
  data['ITEMID'] = data['ITEMID'] - 1
  data.drop(['TIMESTAMP','RATING'], axis=1, inplace=True)
  data = data.groupby('USERID')['ITEMID'].apply(list).reset_index(name='ITEMID')
  return data

def store(data, target_file='./data/movielens/train.txt'):
  Path(target_file).parent.mkdir(parents=True, exist_ok=True)
  with open(target_file, 'w+') as f:
    writer = csv.writer(f, delimiter=' ')
    for USERID, row in zip(data.USERID.values,data.ITEMID.values):
      row = [USERID] + row
      writer.writerow(row)

store(preprocess(df_train), 'train_ml.txt')
store(preprocess(df_test), 'test_ml.txt')
```

```python id="RaMBQdzmncNA" executionInfo={"status": "ok", "timestamp": 1635783540658, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def load_data(train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize = 0, 0
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)


    train_data = []
    test_data = []

    n_user += 1
    m_item += 1

    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0


    # construct degree matrix for graphmf

    items_D = np.sum(train_mat, axis = 0).reshape(-1)
    users_D = np.sum(train_mat, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))  # n_user * m_item
    constraint_mat = constraint_mat.flatten()

    return train_data, test_data, train_mat, n_user, m_item, constraint_mat
```

```python id="5-HbT9mZhsdN" executionInfo={"status": "ok", "timestamp": 1635783543652, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False):
    
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)	# I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    return res_mat.long(), res_sim_mat.float()
```

```python id="jXGAOOtunj4H" executionInfo={"status": "ok", "timestamp": 1635783543653, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
'''
Useful functions
'''

def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))
```

```python id="lHhb4HZ3jqSw" executionInfo={"status": "ok", "timestamp": 1635783543651, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def data_param_prepare(args):

    # dataset processing
    train_data, test_data, train_mat, user_num, item_num, constraint_mat = load_data(args.train_file_path, args.test_file_path)
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(list(range(user_num)), batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    args.user_num = user_num
    args.item_num = item_num

    # mask matrix for testing to accelarate testing speed
    mask = torch.zeros(user_num, item_num)
    interacted_items = [[] for _ in range(user_num)]
    for (u, i) in train_data:
        mask[u][i] = -np.inf
        interacted_items[u].append(i)

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(user_num)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)

    
    # Compute \Omega to extend UltraGCN to the item-item occurrence graph
    ii_cons_mat_path = './' + args.dataset + '_ii_constraint_mat'
    ii_neigh_mat_path = './' + args.dataset + '_ii_neighbor_mat'
    
    if os.path.exists(ii_cons_mat_path):
        ii_constraint_mat = pload(ii_cons_mat_path)
        ii_neighbor_mat = pload(ii_neigh_mat_path)
    else:
        ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, args.ii_neighbor_num)
        pstore(ii_neighbor_mat, ii_neigh_mat_path)
        pstore(ii_constraint_mat, ii_cons_mat_path)

    return args, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items
```

<!-- #region id="hNnbegiCYEpU" -->
## Negative Sampling
<!-- #endregion -->

```python id="sp1hjePBnoC3" executionInfo={"status": "ok", "timestamp": 1635783543654, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def Sampling(pos_train_data, item_num, neg_ratio, interacted_items, sampling_sift_pos):
	neg_candidates = np.arange(item_num)

	if sampling_sift_pos:
		neg_items = []
		for u in pos_train_data[0]:
			probs = np.ones(item_num)
			probs[interacted_items[u]] = 0
			probs /= np.sum(probs)

			u_neg_items = np.random.choice(neg_candidates, size = neg_ratio, p = probs, replace = True).reshape(1, -1)
	
			neg_items.append(u_neg_items)

		neg_items = np.concatenate(neg_items, axis = 0) 
	else:
		neg_items = np.random.choice(neg_candidates, (len(pos_train_data[0]), neg_ratio), replace = True)
	
	neg_items = torch.from_numpy(neg_items)
	
	return pos_train_data[0], pos_train_data[1], neg_items	# users, pos_items, neg_items
```

<!-- #region id="hQgcbO1AoARX" -->
## Model Definition
<!-- #endregion -->

<!-- #region id="ujj7eZoihU8x" -->
$ L = -(w1 + w2*\beta) * log(sigmoid(e_u e_i)) - \sum_{N^-} (w3 + w4*\beta) * log(sigmoid(e_u e_i')) $
<!-- #endregion -->

```python id="tSuNY4EOns8r" executionInfo={"status": "ok", "timestamp": 1635783544379, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
class UltraGCN(nn.Module):
    def __init__(self, args, constraint_mat, ii_constraint_mat, ii_neighbor_mat):
        super(UltraGCN, self).__init__()
        self.user_num = args.user_num
        self.item_num = args.item_num
        self.embedding_dim = args.embedding_dim
        self.w1 = args.w1
        self.w2 = args.w2
        self.w3 = args.w3
        self.w4 = args.w4

        self.negative_weight = args.negative_weight
        self.gamma = args.gamma
        self.lambda_ = args.lambda_

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        self.constraint_mat = constraint_mat
        self.ii_constraint_mat = ii_constraint_mat
        self.ii_neighbor_mat = ii_neighbor_mat

        self.initial_weight = args.initial_weight

        self.initial_weights()

    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)


    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()
        if self.w2 > 0:
            pos_weight = self.constraint_mat[users * self.item_num + pos_items].to(device)
            pow_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)
        
        users = (users * self.item_num).unsqueeze(0)

        if self.w4 > 0:
            neg_weight = self.constraint_mat[torch.cat([users] * neg_items.size(1)).transpose(1, 0) + neg_items].flatten().to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)


        weight = torch.cat((pow_weight, neg_weight))
        return weight


    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.get_device()
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)
      
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight
      
        return loss.sum()


    def cal_loss_I(self, users, pos_items):
        device = self.get_device()
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items].to(device))    # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].to(device)     # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users).unsqueeze(1)
        
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
      
        # loss = loss.sum(-1)
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2


    def forward(self, users, pos_items, neg_items):
        omega_weight = self.get_omegas(users, pos_items, neg_items)
        
        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss


    def test_foward(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)
         
        return user_embeds.mm(item_embeds.t())


    def get_device(self):
        return self.user_embeds.weight.device
```

<!-- #region id="JS_x_QQkoFnt" -->
## Training
<!-- #endregion -->

```python id="-xCwsQvXn-mf" executionInfo={"status": "ok", "timestamp": 1635783545214, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def train(model, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, args):
    device = args.device
    best_epoch, best_recall, best_ndcg = 0, 0, 0
    early_stop_count = 0
    early_stop = False

    batches = len(train_loader.dataset) // args.batch_size
    if len(train_loader.dataset) % args.batch_size != 0:
        batches += 1
    print('Total training batches = {}'.format(batches))
    
    if args.enable_tensorboard:
        writer = SummaryWriter()
    

    for epoch in range(args.max_epoch):
        model.train() 
        start_time = time.time()

        for batch, x in enumerate(train_loader): # x: tensor:[users, pos_items]
            users, pos_items, neg_items = Sampling(x, args.item_num, args.negative_num, interacted_items, args.sampling_sift_pos)
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            model.zero_grad()
            loss = model(users, pos_items, neg_items)
            if args.enable_tensorboard:
                writer.add_scalar("Loss/train_batch", loss, batches * epoch + batch)
            loss.backward()
            optimizer.step()
        
        train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
        if args.enable_tensorboard:
            writer.add_scalar("Loss/train_epoch", loss, epoch)

        need_test = True
        if epoch < 50 and epoch % 5 != 0:
            need_test = False
            
        if need_test:
            start_time = time.time()
            F1_score, Precision, Recall, NDCG = test(model, test_loader, test_ground_truth_list, mask, args.topk, args.user_num)
            if args.enable_tensorboard:
                writer.add_scalar('Results/recall@20', Recall, epoch)
                writer.add_scalar('Results/ndcg@20', NDCG, epoch)
            test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
            
            print('The time for epoch {} is: train time = {}, test time = {}'.format(epoch, train_time, test_time))
            print("Loss = {:.5f}, F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(loss.item(), F1_score, Precision, Recall, NDCG))

            if Recall > best_recall:
                best_recall, best_ndcg, best_epoch = Recall, NDCG, epoch
                early_stop_count = 0
                torch.save(model.state_dict(), args.model_save_path)

            else:
                early_stop_count += 1
                if early_stop_count == args.early_stop_epoch:
                    early_stop = True
        
        if early_stop:
            print('##########################################')
            print('Early stop is triggered at {} epochs.'.format(epoch))
            print('Results:')
            print('best epoch = {}, best recall = {}, best ndcg = {}'.format(best_epoch, best_recall, best_ndcg))
            print('The best model is saved at {}'.format(args.model_save_path))
            break

    writer.flush()

    print('Training end!')
```

<!-- #region id="EMPyQ8A4pGX6" -->
## Test and Metrics
<!-- #endregion -->

```python id="nq-_pk2ro-_S" executionInfo={"status": "ok", "timestamp": 1635783546716, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def RecallPrecision_ATk(test_data, r, k):
	"""
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
	right_pred = r[:, :k].sum(1)
	precis_n = k
	
	recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
	recall_n = np.where(recall_n != 0, recall_n, 1)
	recall = np.sum(right_pred / recall_n)
	precis = np.sum(right_pred) / precis_n
	return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
	"""
    Mean Reciprocal Rank
    """
	pred_data = r[:, :k]
	scores = np.log2(1. / np.arange(1, k + 1))
	pred_data = pred_data / scores
	pred_data = pred_data.sum(1)
	return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
	"""
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
	assert len(r) == len(test_data)
	pred_data = r[:, :k]

	test_matrix = np.zeros((len(pred_data), k))
	for i, items in enumerate(test_data):
		length = k if k <= len(items) else len(items)
		test_matrix[i, :length] = 1
	max_r = test_matrix
	idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
	dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
	dcg = np.sum(dcg, axis=1)
	idcg[idcg == 0.] = 1.
	ndcg = dcg / idcg
	ndcg[np.isnan(ndcg)] = 0.
	return np.sum(ndcg)
```

```python id="5H3IL1K6pLvJ" executionInfo={"status": "ok", "timestamp": 1635783547242, "user_tz": -330, "elapsed": 5, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue,r,k)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def test(model, test_loader, test_ground_truth_list, mask, topk, n_user):
    users_list = []
    rating_list = []
    groundTrue_list = []

    with torch.no_grad():
        model.eval()
        for idx, batch_users in enumerate(test_loader):
            
            batch_users = batch_users.to(model.get_device())
            rating = model.test_foward(batch_users) 
            rating = rating.cpu()
            rating += mask[batch_users]
            
            _, rating_K = torch.topk(rating, k=topk)
            rating_list.append(rating_K)

            groundTrue_list.append([test_ground_truth_list[u] for u in batch_users])

    X = zip(rating_list, groundTrue_list)
    Recall, Precision, NDCG = 0, 0, 0

    for i, x in enumerate(X):
        precision, recall, ndcg = test_one_batch(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg
        

    Precision /= n_user
    Recall /= n_user
    NDCG /= n_user
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    return F1_score, Precision, Recall, NDCG
```

```python colab={"base_uri": "https://localhost:8080/"} id="crj5MU85pOjY" executionInfo={"status": "ok", "timestamp": 1635783811899, "user_tz": -330, "elapsed": 262501, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="502cd752-1bba-4418-8be6-2dfd6e863955"
if __name__ == "__main__":

    print('###################### UltraGCN ######################')

    print('1. Loading Configuration...')
    args, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items = data_param_prepare(args)
    
    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(args)

    ultragcn = UltraGCN(args, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    ultragcn = ultragcn.to(args.device)
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=args.lr)

    train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, args)
    print('END')
```
