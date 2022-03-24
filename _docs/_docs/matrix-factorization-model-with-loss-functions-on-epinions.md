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

```python id="NnG6ZPo6ctof"
!pip install torchsnooper
!pip install guppy3
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 17386, "status": "ok", "timestamp": 1637851739495, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="LFLhMq57fHng" outputId="03f43a57-be72-4216-d020-08ea9f4abe9a"
!mkdir -p /content/data/Epinions/th_0/fold_0
%cd /content/data/Epinions/th_0/fold_0
!gdown --id 1wE8sH6rhWGCnmiIQW8p_VBDSO1kEKDsx -O epinions_preprocessing.ipynb
%run epinions_preprocessing.ipynb
```

```python executionInfo={"elapsed": 6625, "status": "ok", "timestamp": 1637851746112, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="3CiwpYWBclgz"
import torch
import numpy as numpy
import torch.nn as nn
import torchsnooper

import os
import pandas as pd

import math
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
import pickle

from guppy import hpy

import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import json
```

```python executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1637851746113, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="gd2Kl01Iim_f"
class LambdaMF(nn.Module):
    def __init__(self, n_user, n_item, init_range, emb_size, 
    			weight_user=None, 
    			weight_item=None):

        super(LambdaMF, self).__init__()
        self.user_emb = nn.Embedding(n_user, emb_size)
        self.item_emb = nn.Embedding(n_item, emb_size)
          # initlializing weights
        if weight_user == None:
            self.user_emb.weight.data.uniform_(-init_range, init_range)
        else:
            self.user_emb = weight_user
           
        if weight_item == None:
            self.item_emb.weight.data.uniform_(-init_range, init_range)
        else:
            self.item_emb = weight_item
        
    def forward(self, userID, itemID, rels, mode):
        # score = torch.empty(size=(userID.size()[0], items.size()[1]))
        user = self.user_emb(userID)
        items = self.item_emb(itemID)

        if mode == 'train':
            pred = (user * items).sum(-1)
            idx_pad = (rels == 20).nonzero()
            pred[idx_pad[:, 0], idx_pad[:, 1]] = -100
            # rels[idx_pad[:, 0], idx_pad[:, 1]] = 0
            return pred, rels
        
        return (user * items).sum(-1)
```

```python executionInfo={"elapsed": 1829, "status": "ok", "timestamp": 1637851747936, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="TFqE_-hWcleg"
class lambda_loss:
	def __init__(self, device, rank_list, target_list, t, b, p, num_pos, num_neg):              
		# self.rank_list = rank_list.unsqueeze(1)
		# self.target_list = target_list.unsqueeze(1)
		self.rank_list = rank_list
		self.target_list = target_list
		self.num_pos = num_pos
		self.num_neg = num_neg
		self.device = device
		self.p = p

	
	def dcg(self, rank_list, target_list):
		return torch.sum((torch.pow(2, target_list) - 1)/ torch.log2(2 + rank_list.float()), dim=1)
	
	def rbp(self, rank_list, target_list, p):
		return torch.sum(target_list * torch.pow(p, rank_list), dim=1)

	def rr(self, rank_list, target_list):
		rank_list = rank_list + 1
		rr_all = target_list / rank_list
		values, _ = torch.max(rr_all, dim=1)
		# print(values)
		return values
	
	def smart_sort(self, x, permutation):
		d1, d2 = x.size()
		ret = x[
			torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
			permutation.flatten()
		].view(d1, d2)
		return ret
	

	def ap(self, rank_list, target_list):
		value, idxs = torch.sort(rank_list)        
		target_reorder = self.smart_sort(target_list, idxs)
		rank_list = value + 1
		ap_ind = target_reorder * target_reorder.cumsum(dim=1) / rank_list
		# print((ap_ind != 0).sum(dim=1))
		ap = ap_ind.sum(1) / (ap_ind != 0).sum(dim=1)
		return ap

	# @torchsnooper.snoop()
	def lambda_dcg(self):
		device = self.device
		rank_list = self.rank_list
		target_list = self.target_list

		num_doc, n_docs = rank_list.size()

		n_rel = (1.0 * (target_list == 1)).sum(-1).squeeze(-1).int()
		n_val = (1.0 * (target_list != 20)).sum(-1).squeeze(-1).int()

		if not n_rel.size():
			n_rel = n_rel.unsqueeze(0)
			n_val = n_val.unsqueeze(0)

		rank_list = rank_list.unsqueeze(1)
		
		(sorted_scores, sorted_idxs) = rank_list.permute(0, 2, 1).sort(dim=1, descending=True)
		# print(sorted_idxs)
		doc_ranks = torch.zeros(num_doc, n_docs).to(device)   

		for i in torch.arange(num_doc):
			doc_ranks[i, sorted_idxs[i]] = 1 + torch.arange(n_docs).view((n_docs, 1)).float().to(device)
		
		doc_ranks = doc_ranks.unsqueeze(1)
		doc_rank_ori = (doc_ranks - 1).squeeze(1)
		# doc_ranks = doc_ranks.permute(0, 2, 1)

		# print(rank_list[:, :n_rel].size())
		# print(rank_list[:, n_rel:].size())

		exped = torch.zeros([num_doc, n_docs, n_docs]).to(device)
		
	
		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]
			# print(n_docs, rel, val)
			# print(rank_list[i, :, :rel].shape)
			# print(rank_list[i, :, rel:val])
			rank_new = rank_list[i, :, :rel].permute(1, 0) - rank_list[i, :, rel:val] 
			# print(rank_new.shape)
			score_diffs = rank_new.exp()
			# print(exped.shape)
			exped[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(score_diffs) 

		N = 1
 
		dcg_diffs = torch.zeros([num_doc, n_docs, n_docs]).to(device) 

		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]

			diff_new = 1 / (1 + doc_ranks[i, :, :rel]).log2().permute(1, 0) - (1 / (1 + doc_ranks[i, :, rel:val]).log2())
			norm = (1 / (2 + torch.arange(rel).float()).log2()).sum().to(device)
			diff_new = diff_new / norm
			# print(n_docs-rel)
			# print(n_docs+rel-val)
			dcg_diffs[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(diff_new)

		lamb_updates = 1 / (1 + exped) * N * dcg_diffs.abs()
		loss = lamb_updates.sum()

	   
		return loss


	def lambda_ap(self):
		rank_list = self.rank_list
		target_list = self.target_list

		# n_docs = len(rank_list)
		# num_doc, _, n_docs = rank_list.size()
		num_doc, n_docs = rank_list.size()

		# n_rel = target_list.sum(dim=2)
		# n_rel = self.num_pos
		n_rel = (1.0 * (target_list == 1)).sum(-1).squeeze(-1).int()
		n_val = (1.0 * (target_list != 20)).sum(-1).squeeze(-1).int()

		if not n_rel.size():
			n_rel = n_rel.unsqueeze(0)
			n_val = n_val.unsqueeze(0)
			# print(n_rel)
			# print(n_val)
		# print(n_rel.shape)

		# rank_list = rank_list.permute(0, 2, 1)
		rank_list = rank_list.unsqueeze(1)

		(sorted_scores, sorted_idxs) = rank_list.permute(0, 2, 1).sort(dim=1, descending=True)
		# print(sorted_idxs)
		doc_ranks = torch.zeros(num_doc, n_docs).to(device)   

		for i in torch.arange(num_doc):
			doc_ranks[i, sorted_idxs[i]] = 1 + torch.arange(n_docs).view((n_docs, 1)).float().to(device)

		doc_ranks = doc_ranks.unsqueeze(1)
		doc_rank_ori = (doc_ranks - 1).squeeze(1)

		exped = torch.zeros([num_doc, n_docs, n_docs]).to(device)

		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]
			# print(n_docs, rel, val)
			# print(rank_list[i, :, :rel].shape)
			# print(rank_list[i, :, rel:val])
			rank_new = rank_list[i, :, :rel].permute(1, 0) - rank_list[i, :, rel:val] 
			# print(rank_new.shape)
			score_diffs = rank_new.exp()
			# print(exped.shape)
			exped[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(score_diffs) 
	

		N = 1
		ap_diffs = torch.zeros([num_doc, n_docs, n_docs]).to(device) 

		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]

			# print(n_docs, rel, val)
			# print(rank_list[i, :, :rel].shape)
			# print(rank_list[i, :, rel:val])
			# print(rel, val)

			rank_new = torch.zeros([rel, val-rel]).to(device)


			for j in range(rel):
				rank_p = doc_ranks[i, :, j].item()
				
				# print(1.0 * (doc_ranks[i] <= rank_p) * 1.0 * (target_list[i] == 1))
				# target_list[i, :val] == 1
				
				m = (1.0 * (doc_ranks[i] <= rank_p) * 1.0 * (target_list[i] == 1)).sum(-1)

				# m = (1.0 * ((target_list[i, :val] == 1) and (rank_list[i] <= rank_p))).sum(-1)
				term_2 = (m / rank_p).item()

				for k in range(val-rel):
					rank_n = (doc_ranks[i, :, val-rel+k]).item()
					
					if rank_p < rank_n:
						rank_new[j, k] = 0
					else:
						n = (1.0 * (doc_ranks[i] <= rank_n) * 1.0 * (target_list[i] == 1)).sum(-1)
						term_1 = (n + 1) / rank_n
						term_1 = term_1.item()

						prec = target_list[i, : val] * doc_ranks[i, :, :val]
						prec = prec.squeeze(0).double()
						# print(prec)
						# print(rank_p, rank_n)
			
						
					
						prec = torch.where(prec > rank_n, prec, 0.)
						prec = torch.where(prec < rank_p, prec, 0.)

						prec = prec[prec.nonzero()]

						if prec == torch.Size([]):
							term_3 = 0
						else:
							term_3 = (1.0 / prec).sum()
						# print(term_3)
						

						rank_new[j, k] = (term_1 - term_2 + term_3) / rel
				ap_diffs[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(rank_new) 

		lamb_updates = 1 / (1 + exped) * N * ap_diffs.abs()
		loss = lamb_updates.sum()
		return loss


	def lambda_rr(self):
		# device = self.device
		rank_list = self.rank_list
		target_list = self.target_list
		# p = self.p

		# n_docs = len(rank_list)
		# num_doc, _, n_docs = rank_list.size()
		num_doc, n_docs = rank_list.size()

		# n_rel = target_list.sum(dim=2)
		# n_rel = self.num_pos
		n_rel = (1.0 * (target_list == 1)).sum(-1).squeeze(-1).int()
		n_val = (1.0 * (target_list != 20)).sum(-1).squeeze(-1).int()

		if not n_rel.size():
			n_rel = n_rel.unsqueeze(0)
			n_val = n_val.unsqueeze(0)
			

		# rank_list = rank_list.permute(0, 2, 1)
		rank_list = rank_list.unsqueeze(1)
		
		(sorted_scores, sorted_idxs) = rank_list.permute(0, 2, 1).sort(dim=1, descending=True)
		# print(sorted_idxs)
		doc_ranks = torch.zeros(num_doc, n_docs).to(device)   

		for i in torch.arange(num_doc):
			doc_ranks[i, sorted_idxs[i]] = 1 + torch.arange(n_docs).view((n_docs, 1)).float().to(device)
		
		doc_ranks = doc_ranks.unsqueeze(1)
		doc_rank_ori = (doc_ranks - 1).squeeze(1)

		exped = torch.zeros([num_doc, n_docs, n_docs]).to(device)
		
	
		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]
			rank_new = rank_list[i, :, :rel].permute(1, 0) - rank_list[i, :, rel:val] 
			# print(rank_new.shape)
			score_diffs = rank_new.exp()
			# print(exped.shape)
			exped[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(score_diffs) 

		rr_diffs = torch.zeros([num_doc, n_docs, n_docs]).to(device) 

		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]

			rank_new = torch.zeros([rel, val-rel]).to(device)

			diff_new_ = 1 / doc_ranks[i, :, :rel].permute(1, 0) - 1 / doc_ranks[i, :, rel:val]
			diff_new_ = torch.clamp(diff_new_, max=0)
			top_rel = torch.argmin(doc_ranks[i, :, :rel])
			diff_new = torch.zeros_like(diff_new_)
			diff_new[top_rel] = diff_new_[top_rel]

			rr_diffs[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(diff_new)
		
		N = 1
		lamb_updates = 1 / (1 + exped) * N * rr_diffs.abs()
		loss = lamb_updates.sum()
		return loss


	def lambda_rbp(self):
		device = self.device
		rank_list = self.rank_list
		target_list = self.target_list
		p = self.p

		# n_docs = len(rank_list)
		# num_doc, _, n_docs = rank_list.size()
		num_doc, n_docs = rank_list.size()

		# n_rel = target_list.sum(dim=2)
		# n_rel = self.num_pos
		n_rel = (1.0 * (target_list == 1)).sum(-1).squeeze(-1).int()
		n_val = (1.0 * (target_list != 20)).sum(-1).squeeze(-1).int()

		if not n_rel.size():
			n_rel = n_rel.unsqueeze(0)
			n_val = n_val.unsqueeze(0)
			# print(n_rel)
			# print(n_val)
		# print(n_rel.shape)

		# rank_list = rank_list.permute(0, 2, 1)
		rank_list = rank_list.unsqueeze(1)
		
		(sorted_scores, sorted_idxs) = rank_list.permute(0, 2, 1).sort(dim=1, descending=True)
		# print(sorted_idxs)
		doc_ranks = torch.zeros(num_doc, n_docs).to(device)   

		for i in torch.arange(num_doc):
			doc_ranks[i, sorted_idxs[i]] = 1 + torch.arange(n_docs).view((n_docs, 1)).float().to(device)
		
		doc_ranks = doc_ranks.unsqueeze(1)
		doc_rank_ori = (doc_ranks - 1).squeeze(1)
		# doc_ranks = doc_ranks.permute(0, 2, 1)

		# print(rank_list[:, :n_rel].size())
		# print(rank_list[:, n_rel:].size())

		exped = torch.zeros([num_doc, n_docs, n_docs]).to(device)

		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]
			
			rank_new = rank_list[i, :, :rel].permute(1, 0) - rank_list[i, :, rel:val] 
			score_diffs = rank_new.exp()
			exped[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(score_diffs) 

		N = 1
 
		rbp_diffs = torch.zeros([num_doc, n_docs, n_docs]).to(device) 

		for i in range(num_doc):
			rel = n_rel[i]
			val = n_val[i]
			norm = 1.0 / (1 - torch.pow(p, rel))
			diff_new = torch.pow(p, doc_ranks[i, :, :rel]).permute(1, 0) - torch.pow(p, doc_ranks[i, :, rel:val])
			diff_new = (1 - p) * diff_new / norm
			rbp_diffs[i] = nn.ZeroPad2d((0, n_docs+rel-val, 0, n_docs-rel))(diff_new)

		lamb_updates = 1 / (1 + exped) * N * rbp_diffs.abs()
		loss = lamb_updates.sum()
	   
		return loss
```

```python executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1637851747937, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="BLq0koQSclcZ"
def hit(gt):
    for gt_item in gt:
        if gt_item == 1:
            return 1
    return 0


def dcg_at_k(gt, k):
    return np.sum((np.power(2, gt[: k]) - 1) / np.log2(np.arange(2, k + 2)))

def dcg(gt):
    return np.sum((np.power(2, gt) - 1) / np.log2(np.arange(2, len(gt) + 2)))

def evaluation(model, test_loader, max_rating, device, k, p, n_item):
    model.eval()
    NDCG_at_5, NDCG, RR, AP, RBP_80, RBP_90, RBP_95 = [], [], [], [], [], [], []
    
    for user, items, binary_rels, scale_rels in test_loader:
        user, items, binary_rels, scale_rels = user.to(device), items.to(device), binary_rels.to(device), scale_rels.to(device)
        for i in range(len(user)):
            gt_items = []
            u = user[i]
            item = items[i]
            binary_rel = binary_rels[i]
            scale_rel = scale_rels[i]

            prediction_i = model(u, item, -1, mode='test')

            # ratings, indices = torch.topk(prediction_i, k)
            ratings, indices = torch.topk(prediction_i, len(item))
            
            recommends = torch.take(item, indices).cpu().numpy().tolist()
#             print(u, recommends)
            binary_gt = binary_rel[indices].cpu().numpy()
            scale_gt = scale_rel[indices].cpu().numpy()
#             gt = recommends      
#             gt = [gt[j].cpu().numpy().tolist() for j in range(len(gt))]
            
            recommends = list(filter(lambda x: x != n_item, recommends))
            binary_gt = list(filter(lambda x: x != 20, binary_gt))
            scale_gt = list(filter(lambda x: x != 20, scale_gt))
            if len(scale_gt) < 5:
                scale_gt = scale_gt + [0] * (5 - len(scale_gt))


            non_zero = np.asarray(binary_gt).nonzero()[0]

            # # with cutoff
            # rr = 1. / (non_zero[0] + 1) if non_zero.size else 0.
            # ap = (binary_gt * np.cumsum(binary_gt) / (1 + np.arange(k))).mean()
            # rbp = (1 - p) * (binary_gt * np.power(p, range(k))).sum()

            # no cutoff
            rr = 1. / (non_zero[0] + 1) if non_zero.size else 0.
            ap = (binary_gt * np.cumsum(binary_gt) / (1 + np.arange(len(binary_gt))))
            ap = ap[np.nonzero(ap)].mean()
            rbp_80 = (1 - 0.8) * (binary_gt * np.power(0.8, range(len(binary_gt)))).sum()
            rbp_90 = (1 - 0.9) * (binary_gt * np.power(0.9, range(len(binary_gt)))).sum()
            rbp_95 = (1 - 0.95) * (binary_gt * np.power(0.95, range(len(binary_gt)))).sum()
        
        ###################################################### 
        # dcg with cutoff
            # full_mark = [max_rating] * k
        # dcg without cutoff
            idcg_gt = np.sort(scale_gt)[::-1]
            # score = 0
            # for j in range(len(item)):
            #     score = score + (np.power(2, idcg_gt[j]) - 1) / np.log2(j+2)
        #######################################################
#             print(hit(gt))
#             print(dcg(gt))
            # HR.append(hit(binary_gt))
            # NDCG.append(dcg(scale_gt) / score)
            NDCG_at_5.append(dcg_at_k(scale_gt, k) / dcg_at_k(idcg_gt, k))
            NDCG.append(dcg(scale_gt) / dcg(idcg_gt))
            RR.append(rr)
            AP.append(ap)
            RBP_80.append(rbp_80)
            RBP_90.append(rbp_90)
            RBP_95.append(rbp_95)

#             print(len(NDCG))
    
    # print("HR = %.4f" % np.mean(HR))
    print("NDCG@5 = %.4f" % np.mean(NDCG_at_5))
    print("NDCG = %.4f" % np.mean(NDCG))
    print("MRR = %.4f" % np.mean(RR))
    print("MAP = %.4f" % np.mean(AP))
    print("RBP_80 = %.4f" % np.mean(RBP_80))
    print("RBP_90 = %.4f" % np.mean(RBP_90))
    print("RBP_95 = %.4f" % np.mean(RBP_95))
        
    return NDCG_at_5, NDCG, RR, AP, RBP_80, RBP_90, RBP_95, np.mean(NDCG_at_5), np.mean(NDCG), np.mean(RR), np.mean(AP), np.mean(RBP_80), np.mean(RBP_90), np.mean(RBP_95)
```

```python executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1637851747938, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="liSYZe_IclaD"
# make folders for data, models and results
def dir_exists(path):
	if not os.path.exists(path):
		os.makedirs(path)

def save_model(epoch, model, best_result, optimizer, save_path):
	torch.save({
		'epoch': epoch + 1,
		'state_dict': model.state_dict(),
		'best_performance': best_result,
		'optimizer': optimizer.state_dict(),
		}, save_path)

# ---------------------------Amazon Dataset Preprocessing Functions---------------

def get_unique_id(data_pd: pd.DataFrame, column: str) -> (dict, pd.DataFrame):
	"""
	clear the ids
	:param data_pd: pd.DataFrame 
	:param column: specified col
	:return: dict: {value: id}
	"""
	new_column = '{}_id'.format(column)
	assert new_column not in data_pd.columns
	temp = data_pd.loc[:, [column]].drop_duplicates().reset_index(drop=True)
	temp[new_column] = temp.index
	temp.index = temp[column]
	del temp[column]
	# data_pd.merge()
	data_pd = pd.merge(left=data_pd,
		right=temp,
		left_on=column,
		right_index=True,
		how='left')

	return temp[new_column].to_dict(), data_pd


def load_data_amazon(df):
	user_ids, data_pd = get_unique_id(df, 'user')
	item_ids, data_pd = get_unique_id(df, 'item')


	data_pd = data_pd.loc[:, ['user', 'user_id', 'item', 'item_id', 'rating', 'timestamp']]
	data_pd = data_pd.drop(['user', 'item'], axis=1)

	return data_pd


def write_result(res_path, dataset, loss_type, num_neg, lr, temp, threshold, tradeoff, k, hr, ndcg, mrr, mAP, rbp):
	dir_exists(res_path)
	res_path = os.path.join(res_path, str(dataset) + '.txt')
	f = open(res_path, 'a+')
	f.write('loss type: ' + loss_type + ', num_neg: ' + str(num_neg) + ', learning rate: ' + str(lr) + ', temperature: ' + str(temp) + ', threshold: ' + str(threshold) + ', tradeoff:' + str(tradeoff) + ', k:' + str(k) + '\n')
	f.write('HR = ' + str(hr) + ', NDCG = ' + str(ndcg) + ', MRR = ' + str(mrr) + ', MAP = ' + str(mAP) + ', RBP = ' + str(rbp) + '\n\n' )
	f.close()


def choose_loss(loss_type, device, prediction, rel, t, b, temp, p, f_rbp, num_pos, num_neg):
	if loss_type == 'dcg':
		loss_value = ndcg_loss(device, prediction, rel, t, b, num_pos, num_neg, temp).to(device)
	elif loss_type == 'rr':
		loss_value = rr_loss(device, prediction, rel, temp).to(device)
	elif loss_type == 'ap':
		loss_value = ap_loss(device, prediction, rel, temp).to(device)
	elif loss_type == 'rbp':
		loss_value = rbp_loss(device, prediction, rel, temp, p, f_rbp).to(device)
	elif loss_type == 'nrbp':
		loss_value = nrbp_loss(device, prediction, rel, temp, p, f_rbp).to(device)
	elif loss_type == 'nrbp_1':
		loss_value = nrbp_loss_1(device, prediction, rel, temp, p, f_rbp).to(device)
	elif loss_type == 'lambda_dcg':
		loss_value = lambda_loss(device, prediction, rel, t, b, p, num_pos, num_neg).lambda_dcg().to(device)
	elif loss_type == 'lambda_rr':
		loss_value = lambda_loss(device, prediction, rel, t, b, p, num_pos, num_neg).lambda_rr().to(device)
	elif loss_type == 'lambda_ap':
		loss_value = lambda_loss(device, prediction, rel, t, b, p, num_pos, num_neg).lambda_ap().to(device)
	elif loss_type == 'lambda_rbp':
		loss_value = lambda_loss(device, prediction, rel, t, b, p, num_pos, num_neg).lambda_rbp().to(device)

	else: print('The loss function of your choice is not available.')

	return loss_value

def loss_train_test(model, optimizer, dataloader, loss_type, device, t, b, temp, p, f_rbp, num_pos, num_neg):
	loss = 0
	# if mode == 'train':
	for user, items, rels in dataloader:
		batch = len(user)
		user, items, rel = user.to(device), items.to(device), rels.type(torch.FloatTensor).to(device)       
		user = torch.unsqueeze(user, 1)  

		prediction, rel = model(user, items, rel, mode='train')
		# idx_pad = (rel == 20).nonzero()
		# print(prediction.size())
		# print(rels.size())
		loss_value = choose_loss(loss_type, device, \
			prediction, rel, t, b, temp, p, f_rbp, num_pos, num_neg)
		# for param in model.parameters():
		# 	regularization_loss += torch.sum(torch.abs(param))
		# loss_value += 0.0001 * regularization_loss
		# print(loss_value)
		# print(loss_value / user.size(0))
		optimizer.zero_grad()
		loss_value.sum().backward()
		optimizer.step()
		loss += loss_value.sum()
	return loss
```

```python executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1637851748708, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="MasDcDxJd6rS"
def load_data(path, dataset, threshold, fold):
	read_path = os.path.join(path, dataset, 'th_'+str(threshold))
	data_path = os.path.join(read_path, 'fold_'+str(fold), 'train.csv')
	stat_path = os.path.join(read_path, 'dataset_meta_info_'+str(threshold)+'.json')
	test_path = os.path.join(read_path, 'fold_'+str(fold), 'test.csv')

	df = pd.read_csv(data_path, header=0)
	df_test = pd.read_csv(test_path, header=0)
	print(df.shape)
	print(df_test.shape)

	with open(os.path.join(stat_path), 'r') as f:
		dataset_meta_info = json.load(f)


	n_user = dataset_meta_info['user_size']
	n_item = dataset_meta_info['item_size']
	
	train_row = []
	train_col = []
	train_rating = []

	for line in df.itertuples():
		# print(line)
		if line[3] >= threshold:
			u = line[4]
			i = line[5]
			train_row.append(u)
			train_col.append(i)
			train_rating.append(1)
	train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_user, n_item))

	test_row = []
	test_col = []
	test_rating = []

	for line in df_test.itertuples():
		if line[3] >= threshold:
			u = line[4]
			i = line[5]
			test_row.append(u)
			test_col.append(i)
			test_rating.append(1)
	test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_user, n_item))

	test_row = []
	test_col = []
	test_rel = []
	for line in df_test.itertuples():
		u = line[4]
		i = line[5]
		rel = line[3]
		test_row.append(u)
		test_col.append(i)
		test_rel.append(rel)
	test_rel_matrix = csr_matrix((test_rel, (test_row, test_col)), shape=(n_user, n_item))
	return n_user, n_item, train_matrix, test_matrix, test_rel_matrix


def train_preparation(n_user, n_item, matrix, frac):
	mat = []
	rels = []
	users = []

	all_items = set(np.arange(n_item))
	for user in range(n_user):
		pos_items = list(matrix.getrow(user).nonzero()[1])
		neg_pool = list(all_items - set(matrix.getrow(user).nonzero()[1]))
		len_pos = len(pos_items)
		num_neg = int(frac * len_pos)
		train_rel = [1] * len_pos  + [0] * num_neg
		neg_i = list(np.random.choice(neg_pool, size=num_neg, replace=False))
		items = pos_items + neg_i
		mat.append(items)
		rels.append(train_rel)
		users.append(user)

	max_cols = max([len(item) for item in mat])
	for line in mat:
		line += [n_item] * (max_cols - len(line))
	for line in rels:
		line += [20] * (max_cols - len(line))
	return mat, rels, users


def test_preparation(n_user, n_item, train_matrix, test_matrix, frac):
	mat = []
	rels = []
	users = []
	all_items = set(np.arange(n_item))

	for user in range(n_user):
		pos_items = list(test_matrix.getrow(user).nonzero()[1])
		neg_pool = list(all_items - set(train_matrix.getrow(user).nonzero()[1]) - set(test_matrix.getrow(user).nonzero()[1]))
		len_pos = len(pos_items)
		num_neg = int(frac * len_pos)
		test_rel = [1] * len_pos + [0] * num_neg
		neg_i = list(np.random.choice(neg_pool, size=num_neg, replace=False))
		items = pos_items + neg_i
		mat.append(items)
		rels.append(test_rel)
		users.append(user)

	max_cols = max([len(item) for item in mat])
	for line in mat:
		line += [n_item] * (max_cols - len(line))
	for line in rels:
		line += [20] * (max_cols - len(line))
	return mat, rels, users
```

```python executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1637851750125, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="dCW2Gaiqex8j"
!mv /content/data/Epinions/th_0/fold_0/dataset_meta_info.json /content/data/Epinions/th_0
# alson rename it -- add _0
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 513, "status": "ok", "timestamp": 1637851767980, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="qmXmbOXbhh5P" outputId="f2947102-d0d1-484e-c6ff-b10ee1145b27"
%cd /content
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11568, "status": "ok", "timestamp": 1637851976307, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="4vepnGO4IRYF" outputId="9ad50594-a68e-4065-9f8c-3155b743ef90"
sys.path.append('.')

# make possible new folders for data, models, and results
dir_exists('lambda_models')
dir_exists('logs')
dir_exists('lambda_results')

parser = argparse.ArgumentParser(description='Parameter settings')
parser.add_argument('--data_path', nargs='?', default='./data/',
						help='Input data path.')
parser.add_argument('--save_path', nargs='?', default='./lambda_models/',
                        help='Save data path.')
parser.add_argument('--res_path', nargs='?', default='./lambda_results/',
                        help='Save data path.')
parser.add_argument('--dataset', type=str, default='Epinions',
					choices=['Epinions', 'citeulike', 'ml-10m', 'Clothing_Shoes_and_Jewelry', 'Home_and_Kitchen', 'Sports_and_Outdoors'])  
parser.add_argument('--threshold', type=int, default=0,
					help='binary threshold for pos/neg') 
parser.add_argument('--fold', type=int, default=0,
					choices = [0, 1, 2, 3, 4],
					help='fold ID for experiments')
parser.add_argument('--num_pos', type=int, default=20,
					help='number of negative items sampled')
parser.add_argument('--num_neg', type=int, default=200,
					help='number of negative items sampled')
parser.add_argument('--batch_size', type=int, default=16, 
					help='input batch size for training (default: 128)')
parser.add_argument('--random_range', type=float, default=0.01,
					help='[-random_range, random_range] for initialization')
parser.add_argument('--emb_size', type=int, default=32,
					help='latent factor embedding size (default: 32)')
parser.add_argument('--no-cuda', action='store_true', default=True,
					help='enables CUDA training')
parser.add_argument('--loss_type', type=str, default='lambda_dcg', 
					choices=['lambda_dcg', 'lambda_rr', 'lambda_ap', 'lambda_rbp' ],
					help='listwise loss function selection')
parser.add_argument('--reg', type=float, default=0,
					help='l2 regularization')   
parser.add_argument('--lr', type=float, default=0.1, 
					help='learning rate')  
parser.add_argument('--epochs', type=int, default=5,
					help='number of epochs to train (default: 1000)') 
parser.add_argument('--p', type=float, default=0.95, 
					help='probability value for RBP')
parser.add_argument('--t', type=int, default=2, 
					help='power base for DCG')
parser.add_argument('--b', type=int, default=2, 
					help='log base for DCG')
parser.add_argument('--f_rbp', type=float, default=0.01,
					help='the value to make rankings smaller for RBP training')
parser.add_argument('--temp', type=float, default=1.0,
					help='temperature value for training acceleration')
parser.add_argument('--max_rating', type=float, default=1.0,
					help='max rating scale')
parser.add_argument('--k', type=int, default=5,
					help='cutoff')
parser.add_argument('--frac', type=float, default=1.0,
					help='negative sampling ratio')

args = parser.parse_args(args={})  
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
np.random.seed(2020)
# print(args.cuda)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


data_path = args.data_path
save_path = args.save_path
res_path = args.res_path
dataset = args.dataset

dir_exists(res_path)
dir_exists(os.path.join(res_path, dataset))


threshold = args.threshold
fold = args.fold
p = args.p
t = args.t
b = args.b 
f_rbp = args.f_rbp 
temp = args.temp
batch_size = args.batch_size

num_pos = args.num_pos
num_neg = args.num_neg 
reg = args.reg
emb_size = args.emb_size
random_range = args.random_range
lr = args.lr
k = args.k

epochs = args.epochs
loss_type = args.loss_type 
max_rating = args.max_rating

frac = args.frac                       


n_user, n_item, train_matrix, test_matrix, test_rel_matrix = load_data(data_path, dataset, threshold, fold)

# print(n_user, n_item)
# print(train_matrix)


# train_neg_items = get_neg_items(n_user, n_item, train_matrix, num_neg)
# print(np.array(train_neg_items).shape)

train_mat, train_rels, train_user = train_preparation(n_user, n_item, train_matrix, frac)
test_mat, test_rels_binary, test_user = test_preparation(n_user, n_item, train_matrix, test_matrix, frac)
_, test_rels_scale, _ = test_preparation(n_user, n_item, train_matrix, test_rel_matrix, frac)
# print(np.array(train_mat).shape)

train_tensor = TensorDataset(torch.from_numpy(np.array(train_user)),
							torch.from_numpy(np.array(train_mat)),
						   torch.from_numpy(np.array(train_rels)))
train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)


test_tensor = TensorDataset(torch.from_numpy(np.array(test_user)),
							torch.from_numpy(np.array(test_mat)),
							torch.from_numpy(np.array(test_rels_binary)),
							torch.from_numpy(np.array(test_rels_scale)))
test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
```

```python executionInfo={"elapsed": 19, "status": "ok", "timestamp": 1637851976308, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="CI9nTAbElQgQ"
import warnings
warnings.filterwarnings('ignore')
```

```python colab={"background_save": true, "base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 526334, "status": "ok", "timestamp": 1637852502625, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="06xQn_pqiV8Z"
# print(torch.cuda.memory_summary())

# --------------------------------MODEL---------------------------------
model = LambdaMF(n_user, n_item+1, 
			   init_range=random_range, emb_size=emb_size).to(device)

# ---------------------------Train and test---------------------------------------
train_loss_all = []
test_loss_all = []
epoch_all = [] 
# leave an interface for the epochID incase we save the loss value larger than 1
Best_ndcg = Best_ndcg_at_5 = Best_ap = Best_rr = Best_rbp_80 = Best_rbp_90 = Best_rbp_95 = 0
columns = ['loss_type', 'lr', 'threshold', 'reg', 'fold', 'frac', 'emb_size', 'NDCG@5', 'NDCG', 'RR', 'AP', 'RBP_80', 'RBP_90', 'RBP_95']
columns_indi = ['NDCG@5', 'NDCG', 'AP', 'RR', 'RBP_80', 'RBP_90', 'RBP_95']

# --------------------------Define optimizer----------------------------------
best_result = 0
weight_decay = reg

optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
 
# model_file = "model_" + dataset + '_' + loss_type + '_' + str(lr) + '_' + str(p) + '_' + str(threshold) + '_' + str(reg) +  '_' + str(num_neg) + '_' + str(emb_size) + '_' + str(fold) + ".pth.tar"
# save_path = os.path.join(args.save_path, model_file)
# checkpoint = torch.load(save_path)
# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# best_result = checkpoint['best_performance']
# epoch = check

for i in tqdm(range(epochs)):

	train_loss = loss_train_test(model, optimizer, train_loader, loss_type, device, \
								t, b, temp, p, f_rbp, num_pos, num_neg)
	# print(train_loss)
	## ------------Evaluation------------------------
	if i == 0 or (i + 1) % 5 == 0:
		NDCG_at_5, NDCG, RR, AP, RBP_80, RBP_90, RBP_95, ndcg_at_5, ndcg, mrr, mAP, rbp_80, rbp_90, rbp_95 = evaluation(model, test_loader, max_rating, device, k, p, n_item)
		if loss_type == 'lambda_dcg':
			result_current = ndcg
		elif loss_type == 'lambda_rr':
			result_current = mrr
		elif loss_type == 'lambda_ap':
			result_current = mAP
		elif loss_type == 'lambda_rbp' and p == 0.80:
			result_current = rbp_80
		elif loss_type == 'lambda_rbp' and p == 0.9:
			result_current = rbp_90
		elif loss_type == 'lambda_rbp' and p == 0.95:
			result_current = rbp_95
# 		else:
# 			print('loss function does not exist.')

		if result_current > best_result:
			epoch = i
			best_result = result_current
			# best_hr = hr
			best_ndcg = ndcg
			best_ndcg_at_5 = ndcg_at_5
			best_mrr = mrr
			best_mAP = mAP
			best_rbp_80 = rbp_80
			best_rbp_90 = rbp_90
			best_rbp_95 = rbp_95
			best_NDCG = NDCG
			best_RR = RR
			best_AP = AP
			best_RBP_80 = RBP_80
			best_RBP_90 = RBP_90
			best_RBP_95 = RBP_95
			model_file = "model_" + dataset + '_' + loss_type + '_' + str(lr) + '_' + str(p) + '_' + str(threshold) + '_' + str(reg) +  '_' + str(num_neg) + '_' + str(emb_size) + '_' + str(fold) + ".pth.tar"
			save_path = os.path.join(args.save_path, model_file)
			print("Best" + args.loss_type.upper() + ": %.4f" % best_result)
			# save_model(epoch, model, best_result, optimizer, save_path) 
			loss_type = loss_type
			# if loss_type == 'lambda_rbp':
			# 	if p == 0.80:
			# 		loss_type == 'lambda_rbp_80'
			# 	elif p == 0.90:
			# 		loss_typer == 'lambda_rbp_90'
			# 	elif p == 0.95:
			# 		loss_typer == 'lambda_rbp_95'

			result = pd.DataFrame([[loss_type, lr, threshold, reg, fold, frac, emb_size, best_ndcg_at_5, best_ndcg, best_mrr, best_mAP, best_rbp_80, best_rbp_90, best_rbp_95]], columns=columns).round(4)
			result_indi = list(zip(*[NDCG_at_5, NDCG, AP, RR, RBP_80, RBP_90, RBP_95]))
			result_indi = pd.DataFrame(result_indi, columns=columns_indi).round(4)
			if loss_type == 'lambda_rbp':
				name = 'loss_type_' + loss_type + '_' + str(p) + '_lr_' + str(lr) + '_th_' + str(threshold) + '_reg_' + str(reg) + '_fold_' + str(fold) + '_frac_' + str(frac) + '_emb_size_' + str(emb_size)
			else:
				name = 'loss_type_' + loss_type + '_lr_' + str(lr) + '_th_' + str(threshold) + '_reg_' + str(reg) + '_fold_' + str(fold) + '_frac_' + str(frac) + '_emb_size_' + str(emb_size)
			res_path = os.path.join('./results/', dataset, 'overall')
			res_path_indi = os.path.join('./results/', dataset, 'individual')

			dir_exists(res_path)
			dir_exists(res_path_indi)
			result.to_csv(os.path.join(res_path, name+'.csv'), index=False)
			result_indi.to_csv(os.path.join(res_path_indi,  name+'.csv'), index=False)
```
