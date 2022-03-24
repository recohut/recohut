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

<!-- #region id="T5kY-7DqauRg" -->
# Training GMF with Truncated and Reweighted Denoising Losses on Yelp Dataset
<!-- #endregion -->

<!-- #region id="wVfZ0xpJb4nI" -->
## Executive summary

| | |
| --- | --- |
| Problem | It is of importance to account for the inevitable noises in implicit feedback. However, little work on recommendation has taken the noisy nature of implicit feedback into consideration. |
| Prblm Stmt. | We formulate a denoising recommender training task as $Θ^∗ = \text{argmin}_Θ \mathcal{L}(\textit{denoise}(\bar{D}))$ aiming to learn a reliable recommender model with parameters $Θ^∗$ by denoising implicit feedback $\bar{D}$. Formally, by assuming the existence of inconsistency between $y_{ui}^∗$ and $\bar{y}_{ui}$, we define noisy interactions (a.k.a. false-positive interactions) as $\{(u, i)|y_{ui}^∗ = 0 ∧ \bar{y}_{ui} = 1\}$. |
| Solution | Adaptive Denoising Training (ADT), which adaptively prunes the noisy interactions by two paradigms - Truncated Loss and Reweighted Loss. Furthermore, we consider extra feedback (e.g., rating) as auxiliary signal and employ three strategies to incorporate extra feedback into ADT: fine-tuning, warm-up training, and colliding inference. |
| Dataset | Adressa, Amazon-books, Yelp2018. |
| Preprocessing | We split the dataset into training, validation, and testing sets, and explored two experimental settings: 1) Extra feedback is unavailable during training. To evaluate the performance of denoising implicit feedback, we kept all interactions, including the false-positive ones, in training and validation, and tested the models only on true-positive interactions. 2) Sparse extra feedback is available during training. We assume that partial true-positive interactions have already been known, which will be used to verify the performance of the proposed three strategies: fine-tuning, warm-up training, and colliding inference. |
| Metrics | Recall, NDCG |
| Hyperparams | For GMF and NeuMF, the factor numbers of users and items are both 32. As to CDAE, the hidden size of MLP is set as 200. In addition, Adam is applied to optimize all the parameters with the learning rate initialized as 0.001 and he batch size set as 1,024. As to the ADT strategies, they have three hyper-parameters in total: α and max in the T-CE loss, and β in the R-CE loss. In detail, max is searched in {0.05, 0.1, 0.2} and β is tuned in {0.05, 0.1, ..., 0.25, 0.5, 1.0}. As for α, we controlled its range by adjusting the iteration number N to the maximum drop rate max, and N is adjusted in {1k, 5k, 10k, 20k, 30k}. In colliding inference, the number of neighbors Nu is tuned in {1, 3, 5, 10, 20, 50, 100}, wj is set as 1/|Nu|, and λ is searched in {0, 0.1, 0.2, ..., 1}. We used the validation set to tune the hyper-parameters and reported the performance on the testing set. |
| Models | GMF, NMF, CDAE, {GMF, NMF, CDAE}+T_CE, {GMF, NMF, CDAE}+R_CE |
| Cluster | Python 3.6+, PyTorch |
| Tags | `LossReweighting`, `TruncatedLoss`, `MatrixFactorization`, `Denoising` |
| Credits | Wenjie Wang |
<!-- #endregion -->

<!-- #region id="qHcxWaZPqFyE" -->
## Process flow

![](https://github.com/RecoHut-Stanzas/S063707/raw/main/images/process_flow.svg)
<!-- #endregion -->

<!-- #region id="009XG-W-b4is" -->
## Setup
<!-- #endregion -->

```python id="NhKMYe3Mb4gO"
import numpy as np 
import pandas as pd 
import scipy.sparse as sp
from copy import deepcopy
import random
import math
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F 
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
```

```python colab={"base_uri": "https://localhost:8080/"} id="K4L82dHCeb1H" executionInfo={"status": "ok", "timestamp": 1639311517343, "user_tz": -330, "elapsed": 722, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="32c39b27-4240-478a-da19-11af8f6d2dc4"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
	type = str,
	help = 'dataset used for training, options: amazon_book, yelp, adressa',
	default = 'yelp')
parser.add_argument('--model', 
	type = str,
	help = 'model used for training. options: GMF, NeuMF-end',
	default = 'GMF')
parser.add_argument('--alpha', 
	type = float, 
	default = 0.2, 
	help='hyperparameter in loss function')
parser.add_argument('--drop_rate', 
	type = float,
	help = 'drop rate',
	default = 0.2)
parser.add_argument('--num_gradual', 
	type = int, 
	default = 30000,
	help='how many epochs to linearly increase drop_rate')
parser.add_argument('--exponent', 
	type = float, 
	default = 1, 
	help='exponent of the drop rate {0.5, 1, 2}')
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=1024, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=10,
	help="training epoches")
parser.add_argument("--eval_freq", 
	type=int,
	default=2000,
	help="the freq of eval")
parser.add_argument("--top_k", 
	type=list, 
	default=[50, 100],
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=1, 
	help="sample negative items for training")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",
	help="gpu card ID")
args = parser.parse_args([])

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

torch.manual_seed(2019) # cpu
torch.cuda.manual_seed(2019) #gpu
np.random.seed(2019) #numpy
random.seed(2019) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id):
    np.random.seed(2019 + worker_id)

data_path = '{}/'.format(args.dataset)
model_path = './models/{}/'.format(args.dataset)
os.makedirs('./models', exist_ok=True)
print("arguments: %s " %(args))
print("config model", args.model)
print("config data path", data_path)
print("config model path", model_path)
```

<!-- #region id="GXK73SDob4dR" -->
## Data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="D6y9RjVob4ZS" executionInfo={"status": "ok", "timestamp": 1639310015219, "user_tz": -330, "elapsed": 24510, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a74f0442-81ed-4ad8-80fd-2ec961f58e1c"
!git clone --branch v4 https://github.com/RecoHut-Datasets/yelp.git
```

```python id="mpIf7UVqb7ln"
def load_all(dataset, data_path):

	train_rating = data_path + '{}.train.rating'.format(dataset)
	valid_rating = data_path + '{}.valid.rating'.format(dataset)
	test_negative = data_path + '{}.test.negative'.format(dataset)

	################# load training data #################	
	train_data = pd.read_csv(
		train_rating, 
		sep='\t', header=None, names=['user', 'item', 'noisy'], 
		usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})

	if dataset == "adressa":
		user_num = 212231
		item_num = 6596
	else:
		user_num = train_data['user'].max() + 1
		item_num = train_data['item'].max() + 1
	print("user, item num")
	print(user_num, item_num)
	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	train_data_list = []
	train_data_noisy = []
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0
		train_data_list.append([x[0], x[1]])
		train_data_noisy.append(x[2])

	################# load validation data #################
	valid_data = pd.read_csv(
		valid_rating, 
		sep='\t', header=None, names=['user', 'item', 'noisy'], 
		usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})
	valid_data = valid_data.values.tolist()
	valid_data_list = []
	for x in valid_data:
		valid_data_list.append([x[0], x[1]])
	
	user_pos = {}
	for x in train_data_list:
		if x[0] in user_pos:
			user_pos[x[0]].append(x[1])
		else:
			user_pos[x[0]] = [x[1]]
	for x in valid_data_list:
		if x[0] in user_pos:
			user_pos[x[0]].append(x[1])
		else:
			user_pos[x[0]] = [x[1]]


	################# load testing data #################
	test_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)

	test_data_pos = {}
	with open(test_negative, 'r') as fd:
		line = fd.readline()
		while line != None and line != '':
			arr = line.split('\t')
			if dataset == "adressa":
				u = eval(arr[0])[0]
				i = eval(arr[0])[1]
			else:
				u = int(arr[0])
				i = int(arr[1])
			if u in test_data_pos:
				test_data_pos[u].append(i)
			else:
				test_data_pos[u] = [i]
			test_mat[u, i] = 1.0
			line = fd.readline()


	return train_data_list, valid_data_list, test_data_pos, user_pos, user_num, item_num, train_mat, train_data_noisy
```

```python id="Jv8T8msTcSU1"
class NCFData(data.Dataset):
	def __init__(self, features,
				num_item, train_mat=None, num_ng=0, is_training=0, noisy_or_not=None):
		super(NCFData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features_ps = features
		if is_training == 0:
			self.noisy_or_not = noisy_or_not
		else:
			self.noisy_or_not = [0 for _ in range(len(features))]
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training
		self.labels = [0 for _ in range(len(features))]

	def ng_sample(self):
		assert self.is_training  != 2, 'no need to sampling when testing'

		self.features_ng = []
		for x in self.features_ps:
			u = x[0]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_ng.append([u, j])

		labels_ps = [1 for _ in range(len(self.features_ps))]
		labels_ng = [0 for _ in range(len(self.features_ng))]
		self.noisy_or_not_fill = self.noisy_or_not + [1 for _ in range(len(self.features_ng))]
		self.features_fill = self.features_ps + self.features_ng
		assert len(self.noisy_or_not_fill) == len(self.features_fill)
		self.labels_fill = labels_ps + labels_ng

	def __len__(self):
		return (self.num_ng + 1) * len(self.labels)

	def __getitem__(self, idx):
		features = self.features_fill if self.is_training != 2 \
					else self.features_ps
		labels = self.labels_fill if self.is_training != 2 \
					else self.labels
		noisy_or_not = self.noisy_or_not_fill if self.is_training != 2 \
					else self.noisy_or_not

		user = features[idx][0]
		item = features[idx][1]
		label = labels[idx]
		noisy_label = noisy_or_not[idx]

		return user, item, label, noisy_label
```

<!-- #region id="YutDztVXb7i2" -->
## Model
<!-- #endregion -->

```python id="eD0tBH5ub7gH"
class NCF(nn.Module):
	def __init__(self, user_num, item_num, factor_num, num_layers,
					dropout, model, GMF_model=None, MLP_model=None):
		super(NCF, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
		GMF_model: pre-trained GMF weights;
		MLP_model: pre-trained MLP weights.
		"""		
		self.dropout = dropout
		self.model = model
		self.GMF_model = GMF_model
		self.MLP_model = MLP_model

		self.embed_user_GMF = nn.Embedding(user_num, factor_num)
		self.embed_item_GMF = nn.Embedding(item_num, factor_num)
		self.embed_user_MLP = nn.Embedding(
				user_num, factor_num * (2 ** (num_layers - 1)))
		self.embed_item_MLP = nn.Embedding(
				item_num, factor_num * (2 ** (num_layers - 1)))

		MLP_modules = []
		for i in range(num_layers):
			input_size = factor_num * (2 ** (num_layers - i))
			MLP_modules.append(nn.Dropout(p=self.dropout))
			MLP_modules.append(nn.Linear(input_size, input_size//2))
			MLP_modules.append(nn.ReLU())
		self.MLP_layers = nn.Sequential(*MLP_modules)

		if self.model in ['MLP', 'GMF']:
			predict_size = factor_num 
		else:
			predict_size = factor_num * 2
		self.predict_layer = nn.Linear(predict_size, 1)

		self._init_weight_()

	def _init_weight_(self):
		""" We leave the weights initialization here. """
		if not self.model == 'NeuMF-pre':
			nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
			nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

			for m in self.MLP_layers:
				if isinstance(m, nn.Linear):
					nn.init.xavier_uniform_(m.weight)
			nn.init.kaiming_uniform_(self.predict_layer.weight, 
									a=1, nonlinearity='sigmoid')

			for m in self.modules():
				if isinstance(m, nn.Linear) and m.bias is not None:
					m.bias.data.zero_()
		else:
			# embedding layers
			self.embed_user_GMF.weight.data.copy_(
							self.GMF_model.embed_user_GMF.weight)
			self.embed_item_GMF.weight.data.copy_(
							self.GMF_model.embed_item_GMF.weight)
			self.embed_user_MLP.weight.data.copy_(
							self.MLP_model.embed_user_MLP.weight)
			self.embed_item_MLP.weight.data.copy_(
							self.MLP_model.embed_item_MLP.weight)

			# mlp layers
			for (m1, m2) in zip(
				self.MLP_layers, self.MLP_model.MLP_layers):
				if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
					m1.weight.data.copy_(m2.weight)
					m1.bias.data.copy_(m2.bias)

			# predict layers
			predict_weight = torch.cat([
				self.GMF_model.predict_layer.weight, 
				self.MLP_model.predict_layer.weight], dim=1)
			precit_bias = self.GMF_model.predict_layer.bias + \
						self.MLP_model.predict_layer.bias

			self.predict_layer.weight.data.copy_(0.5 * predict_weight)
			self.predict_layer.bias.data.copy_(0.5 * precit_bias)

	def forward(self, user, item):
		if not self.model == 'MLP':
			embed_user_GMF = self.embed_user_GMF(user)
			embed_item_GMF = self.embed_item_GMF(item)
			output_GMF = embed_user_GMF * embed_item_GMF
		if not self.model == 'GMF':
			embed_user_MLP = self.embed_user_MLP(user)
			embed_item_MLP = self.embed_item_MLP(item)
			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
			output_MLP = self.MLP_layers(interaction)

		if self.model == 'GMF':
			concat = output_GMF
		elif self.model == 'MLP':
			concat = output_MLP
		else:
			concat = torch.cat((output_GMF, output_MLP), -1)

		prediction = self.predict_layer(concat)
		return prediction.view(-1)
```

<!-- #region id="ADEh-IsUb7Xm" -->
## Evaluate
<!-- #endregion -->

```python id="I5oF1IG7dSFb"
def test_all_users(model, batch_size, item_num, test_data_pos, user_pos, top_k):
    
    predictedIndices = []
    GroundTruth = []
    for u in test_data_pos:
        batch_num = item_num // batch_size
        batch_user = torch.Tensor([u]*batch_size).long().cuda()
        st, ed = 0, batch_size
        for i in range(batch_num):
            batch_item = torch.Tensor([i for i in range(st, ed)]).long().cuda()
            pred = model(batch_user, batch_item)
            if i == 0:
                predictions = pred
            else:
                predictions = torch.cat([predictions, pred], 0)
            st, ed = st+batch_size, ed+batch_size
        ed = ed - batch_size
        batch_item = torch.Tensor([i for i in range(ed, item_num)]).long().cuda()
        batch_user = torch.Tensor([u]*(item_num-ed)).long().cuda()
        pred = model(batch_user, batch_item)
        predictions = torch.cat([predictions, pred], 0)
        test_data_mask = [0] * item_num
        if u in user_pos:
            for i in user_pos[u]:
                test_data_mask[i] = -9999
        predictions = predictions + torch.Tensor(test_data_mask).float().cuda()
        _, indices = torch.topk(predictions, top_k[-1])
        indices = indices.cpu().numpy().tolist()
        predictedIndices.append(indices)
        GroundTruth.append(test_data_pos[u])
    precision, recall, NDCG, MRR = compute_acc(GroundTruth, predictedIndices, top_k)
    return precision, recall, NDCG, MRR
```

```python id="gBInZ4_jdS8X"
def compute_acc(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        precision.append(sumForPrecision / len(predictedIndices))
        recall.append(sumForRecall / len(predictedIndices))
        NDCG.append(sumForNdcg / len(predictedIndices))
        MRR.append(sumForMRR / len(predictedIndices))
        
    return precision, recall, NDCG, MRR
```

<!-- #region id="K1tu6vgIfkyj" -->
## T_CE
<!-- #endregion -->

<!-- #region id="lfi9YaIBb7dY" -->
### Loss function
<!-- #endregion -->

```python id="SaBvZ5eVb7aw"
def loss_function(y, t, drop_rate):
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)

    loss_mul = loss * t
    ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = F.binary_cross_entropy_with_logits(y[ind_update], t[ind_update])

    return loss_update
```

<!-- #region id="et6OGWMBeJiE" -->
### Training & Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vK6RkgUteJdS" executionInfo={"status": "ok", "timestamp": 1639311150350, "user_tz": -330, "elapsed": 1134452, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="27de75fc-3cec-4162-e83a-1fcaf217206d"
############################## PREPARE DATASET ##########################

train_data, valid_data, test_data_pos, user_pos, user_num ,item_num, train_mat, train_data_noisy = load_all(args.dataset, data_path)

# construct the train and test datasets
train_dataset = NCFData(
		train_data, item_num, train_mat, args.num_ng, 0, train_data_noisy)
valid_dataset = NCFData(
		valid_data, item_num, train_mat, args.num_ng, 1)

train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
valid_loader = data.DataLoader(valid_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

print("data loaded! user_num:{}, item_num:{} train_data_len:{} test_user_num:{}".format(user_num, item_num, len(train_data), len(test_data_pos)))

########################### CREATE MODEL #################################
if args.model == 'NeuMF-pre': # pre-training. Not used in our work.
	GMF_model_path = model_path + 'GMF.pth'
	MLP_model_path = model_path + 'MLP.pth'
	NeuMF_model_path = model_path + 'NeuMF.pth'
	assert os.path.exists(GMF_model_path), 'lack of GMF model'
	assert os.path.exists(MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(GMF_model_path)
	MLP_model = torch.load(MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

model = NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, args.model, GMF_model, MLP_model)

model.cuda()
BCE_loss = nn.BCEWithLogitsLoss()

if args.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

# writer = SummaryWriter() # for visualization

# define drop rate schedule
def drop_rate_schedule(iteration):

	drop_rate = np.linspace(0, args.drop_rate**args.exponent, args.num_gradual)
	if iteration < args.num_gradual:
		return drop_rate[iteration]
	else:
		return args.drop_rate


########################### Eval #####################################
def eval(model, valid_loader, best_loss, count):
	
	model.eval()
	epoch_loss = 0
	valid_loader.dataset.ng_sample() # negative sampling
	for user, item, label, noisy_or_not in valid_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()

		prediction = model(user, item)
		loss = loss_function(prediction, label, drop_rate_schedule(count))
		epoch_loss += loss.detach()
	print("################### EVAL ######################")
	print("Eval loss:{}".format(epoch_loss))
	if epoch_loss < best_loss:
		best_loss = epoch_loss
		if args.out:
			if not os.path.exists(model_path):
				os.mkdir(model_path)
			torch.save(model, '{}{}_{}-{}.pth'.format(model_path, args.model, args.drop_rate, args.num_gradual))
	return best_loss

########################### Test #####################################
def test(model, test_data_pos, user_pos):
	top_k = args.top_k
	model.eval()
	_, recall, NDCG, _ = test_all_users(model, 4096, item_num, test_data_pos, user_pos, top_k)

	print("################### TEST ######################")
	print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
	print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))

########################### TRAINING #####################################
count, best_hr = 0, 0
best_loss = 1e9

for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).

	start_time = time.time()
	train_loader.dataset.ng_sample()

	for user, item, label, noisy_or_not in train_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()

		model.zero_grad()
		prediction = model(user, item)
		loss = loss_function(prediction, label, drop_rate_schedule(count))
		loss.backward()
		optimizer.step()

		if count % args.eval_freq == 0 and count != 0:
			print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))
			best_loss = eval(model, valid_loader, best_loss, count)
			model.train()

		count += 1

print("############################## Training End. ##############################")
test_model = torch.load('{}{}_{}-{}.pth'.format(model_path, args.model, args.drop_rate, args.num_gradual))
test_model.cuda()
test(test_model, test_data_pos, user_pos)
```

<!-- #region id="QVTERrOWhwQL" -->
## R_CE
<!-- #endregion -->

<!-- #region id="UgioesbdhwNS" -->
### Loss function
<!-- #endregion -->

```python id="xhBZZF99hwJM"
def loss_function(y, t, alpha):

    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)
    y_ = torch.sigmoid(y).detach()
    weight = torch.pow(y_, alpha) * t + torch.pow((1-y_), alpha) * (1-t)
    loss_ = loss * weight
    loss_ = torch.mean(loss_)
    return loss_
```

<!-- #region id="VcNPdUPYh6D3" -->
### Training & Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7Fr3rITCh6A7" executionInfo={"status": "ok", "timestamp": 1639312641090, "user_tz": -330, "elapsed": 1118311, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9321198d-3a69-4e28-e875-55fc46790edb"
############################## PREPARE DATASET ##########################

train_data, valid_data, test_data_pos, user_pos, user_num ,item_num, train_mat, train_data_noisy = load_all(args.dataset, data_path)

# construct the train and test datasets
train_dataset = NCFData(
		train_data, item_num, train_mat, args.num_ng, 0, train_data_noisy)
valid_dataset = NCFData(
		valid_data, item_num, train_mat, args.num_ng, 1)

train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
valid_loader = data.DataLoader(valid_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

print("data loaded! user_num:{}, item_num:{} train_data_len:{} test_user_num:{}".format(user_num, item_num, len(train_data), len(test_data_pos)))

########################### CREATE MODEL #################################
if args.model == 'NeuMF-pre': # pre-training. Not used in our work.
	GMF_model_path = model_path + 'GMF.pth'
	MLP_model_path = model_path + 'MLP.pth'
	NeuMF_model_path = model_path + 'NeuMF.pth'
	assert os.path.exists(GMF_model_path), 'lack of GMF model'
	assert os.path.exists(MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(GMF_model_path)
	MLP_model = torch.load(MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

model = NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, args.model, GMF_model, MLP_model)

model.cuda()
BCE_loss = nn.BCEWithLogitsLoss()

if args.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

# writer = SummaryWriter() # for visualization

########################### Eval #####################################
def eval(model, valid_loader, best_loss, count):
	
	model.eval()
	epoch_loss = 0
	valid_loader.dataset.ng_sample() # negative sampling
	for user, item, label, noisy_or_not in valid_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()

		prediction = model(user, item)
		loss = loss_function(prediction, label, args.alpha)
		epoch_loss += loss.detach()
	print("################### EVAL ######################")
	print("Eval loss:{}".format(epoch_loss))
	if epoch_loss < best_loss:
		best_loss = epoch_loss
		if args.out:
			if not os.path.exists(model_path):
				os.mkdir(model_path)
			torch.save(model, '{}{}_{}.pth'.format(model_path, args.model, args.alpha))
	return best_loss

########################### Test #####################################
def test(model, test_data_pos, user_pos):
	top_k = args.top_k
	model.eval()
	_, recall, NDCG, _ = test_all_users(model, 4096, item_num, test_data_pos, user_pos, top_k)

	print("################### TEST ######################")
	print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
	print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))

########################### TRAINING #####################################
count, best_hr = 0, 0
best_loss = 1e9

for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).

	start_time = time.time()
	train_loader.dataset.ng_sample()

	for user, item, label, noisy_or_not in train_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()

		model.zero_grad()
		prediction = model(user, item)
		loss = loss_function(prediction, label, args.alpha)
		loss.backward()
		optimizer.step()

		if count % args.eval_freq == 0 and count != 0:
			print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))
			best_loss = eval(model, valid_loader, best_loss, count)
			model.train()

		count += 1

print("############################## Training End. ##############################")
test_model = torch.load('{}{}_{}.pth'.format(model_path, args.model, args.alpha))
test_model.cuda()
test(test_model, test_data_pos, user_pos)
```

<!-- #region id="wH6KkvQ7xVob" -->
## References

1. [https://github.com/RecoHut-Stanzas/S063707](https://github.com/RecoHut-Stanzas/S063707)
2. [https://arxiv.org/pdf/2112.01160v1.pdf](https://arxiv.org/pdf/2112.01160v1.pdf)
3. [https://github.com/WenjieWWJ/DenoisingRec](https://github.com/WenjieWWJ/DenoisingRec)
<!-- #endregion -->

<!-- #region id="OHHk65wEo8uk" -->
---
<!-- #endregion -->

```python id="v0OE_IPAo8um"
!apt-get -qq install tree
!rm -r sample_data
```

```python colab={"base_uri": "https://localhost:8080/"} id="_-c-bayco8um" executionInfo={"status": "ok", "timestamp": 1639312774688, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e9f3917f-b46d-41c5-ec0a-14c8a0fc8bf7"
!tree -h --du .
```

```python colab={"base_uri": "https://localhost:8080/"} id="uHYQT0AMo8un" executionInfo={"status": "ok", "timestamp": 1639312785478, "user_tz": -330, "elapsed": 4282, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="666f7d5e-5e1f-48e8-f2dc-656995d46897"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="CrZCu7YIo8uo" -->
---
<!-- #endregion -->

<!-- #region id="YL6VsvtSo8uq" -->
**END**
<!-- #endregion -->
