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

<!-- #region id="4YLrsBbObFta" -->
# Discrete BCQ Model on IEEE 2021 RecSys dataset
<!-- #endregion -->

<!-- #region id="3GofkbEjwQCI" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="BaXpEKLdXNPa" executionInfo={"status": "ok", "timestamp": 1636123055589, "user_tz": -330, "elapsed": 1012, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e1ece07d-14e7-4895-a3d4-83a3eebe5f66"
import os
project_name = "ieee21cup-recsys"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', branch)

if not os.path.exists(project_path):
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

```python colab={"base_uri": "https://localhost:8080/"} id="MZvPHRyMXdlS" executionInfo={"status": "ok", "timestamp": 1636123057582, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b97f25bc-5ac9-44fd-fd4d-dc413b071b54"
%cd /content
```

```python id="2eRcpGL6XfDs"
!cd /content/main && git add . && git commit -m 'commit' && git push origin main
```

```python colab={"base_uri": "https://localhost:8080/"} id="DctyNOSdx-7h" executionInfo={"status": "ok", "timestamp": 1636120595541, "user_tz": -330, "elapsed": 5202, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="48a53075-14c3-444c-b70e-5d0b9ce8d245"
!pip install -q wget
```

```python id="vrEmNkAAsQlM"
import io
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm.notebook import tqdm
import copy
import sys
import wget
import os
import logging
import pandas as pd
from os import path as osp
from pathlib import Path

import bz2
import pickle
import _pickle as cPickle

import torch
import torch.nn as nn
import torch.nn.functional as F
```

```python id="M4swQxyAsQnj"
class Args:

    # Paths
    datapath_bronze = '/content/main/data/bronze'
    datapath_silver = '/content/main/data/silver/T719060'

    filename_trainset = 'train.csv'
    filename_iteminfo = 'item_info.csv'
    filename_track2_testset = 'track2_testset.csv'

    filename_trainset_processed = 'processed_trainset_data'
    filename_track2_testset_processed = 'processed_track2_testset_data'

    # Exploration
    start_timesteps = 1e3
    initial_eps = 0.1
    end_eps = 0.1
    eps_decay_period = 1

    # Evaluation
    eval_freq = 1000
    eval_eps = 0

    # Learning
    discount = 0.99
    epoch_num = 2
    batch_size = 512
    optimizer = 'Adam'
    optimizer_parameters = {'lr':3e-4}
    train_freq = 1
    polyak_target_update = True
    target_update_freq = 1
    tau = 0.005

    # Other
    data_sep = ' '
    state_dim = 273  
    num_actions = 381


args = Args()
```

```python id="YUnDjKgnKiIZ"
torch.manual_seed(2021)
np.random.seed(2021)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

```python id="wIDRSKqOtEdb"
logging.basicConfig(stream=sys.stdout,
                    level = logging.INFO,
                    format='%(asctime)s [%(levelname)s] : %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger('IEEE21 Logger')
```

<!-- #region id="N1bmqnvQv27E" -->
## Utilities
<!-- #endregion -->

```python id="tH7lmOJbAOIf"
def save_pickle(data, title):
 with bz2.BZ2File(title + '.pbz2', 'w') as f: 
    cPickle.dump(data, f)

def load_pickle(path):
    data = bz2.BZ2File(path+'.pbz2', 'rb')
    data = cPickle.load(data)
    return data
```

```python id="AGTVUdmtwWgZ"
def download_dataset():
    # create bronze folder if not exist
    Path(args.datapath_bronze).mkdir(parents=True, exist_ok=True)
    # also creating silver folder for later use
    Path(args.datapath_silver).mkdir(parents=True, exist_ok=True)
    # for each of the file, download if not exist
    datasets = ['train.parquet.snappy', 'item_info.parquet.snappy',
                'track1_testset.parquet.snappy', 'track2_testset.parquet.snappy']
    for filename in datasets:
        file_savepath = osp.join(args.datapath_bronze,filename)
        if not osp.exists(file_savepath):
            logger.info('Downloading {}'.format(filename))
            wget.download(url='https://github.com/sparsh-ai/ieee21cup-recsys/raw/main/data/bronze/{}'.format(filename),
                          out=file_savepath)
        else:
            logger.info('{} file already exists, skipping!'.format(filename))
```

```python id="vk93jRMwtEWP"
def parquet_to_csv(path):
    savepath = osp.join(str(Path(path).parent),str(Path(path).name).split('.')[0]+'.csv')
    pd.read_parquet(path).to_csv(savepath, index=False, sep=args.data_sep)
```

```python id="_F4vRpFCzYsf"
def convert_dataset():
    # for each of the file, convert into csv, if csv not exist
    datasets = ['train.parquet.snappy', 'item_info.parquet.snappy',
                'track1_testset.parquet.snappy', 'track2_testset.parquet.snappy']
    datasets = {x:str(Path(x).name).split('.')[0]+'.csv' for x in datasets}
    for sfilename, tfilename in datasets.items():
        file_loadpath = osp.join(args.datapath_bronze,sfilename)
        file_savepath = osp.join(args.datapath_bronze,tfilename)
        if not osp.exists(file_savepath):
            logger.info('Converting {} to {}'.format(sfilename, tfilename))
            parquet_to_csv(file_loadpath)
        else:
            logger.info('{} file already exists, skipping!'.format(tfilename))
```

<!-- #region id="igLLZV6gGu-v" -->
---
<!-- #endregion -->

```python id="f1Mi6dcZsQjW"
def load_dataset(path):
    i = 0
    user_id, user_click_history, user_protrait, exposed_items, labels, time = [], [], [], [], [], []
    with io.open(path,'r') as file:
        for line in file:
            if i > 0:
                user_id_1, user_click_history_1, user_protrait_1, exposed_items_1, labels_1, time_1 = line.split(' ')
                user_id.append(user_id_1)
                user_click_history.append(user_click_history_1)
                user_protrait.append(user_protrait_1)
                exposed_items.append(exposed_items_1)
                labels.append(labels_1)
                time.append(time_1)
            i = i + 1
    return user_id, user_click_history, user_protrait, exposed_items, labels, time
```

```python id="g7hdThsIsQg3"
def data_processing(user_click_history, user_protrait, exposed_items, labels, item_info_list):
    user_click_history_processed = []
    for item in user_click_history:
        user_click_history_row = []
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            click_history = float(item_2.split(':')[0])
            user_click_history_row.append(click_history)

        if len(user_click_history_row) < 249:
            for i in range(249-len(user_click_history_row)):
                user_click_history_row.append(0.0)
        
        if len(user_click_history_row) > 249:
            print("len(user_click_history_row): ", len(user_click_history_row))
            user_click_history_row = user_click_history_row[:249]

        user_click_history_processed.append(user_click_history_row)

    user_click_history_avg_processed = []
    for item in user_click_history:
        user_click_history_row_avg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            click_history_item_id = float(item_2.split(':')[0])
            
            if click_history_item_id == 0.0:
                continue

            item_info_dic = item_info_list[int(click_history_item_id)-1]
            item_info = item_info_dic[float(click_history_item_id)]
            user_click_history_row_avg = user_click_history_row_avg + np.array(item_info)
            
        user_click_history_row_avg = user_click_history_row_avg / len(item_split_list)
        user_click_history_row_avg = user_click_history_row_avg.tolist()

        user_click_history_avg_processed.append(user_click_history_row_avg)

    user_protrait_processed = []
    for item in user_protrait:
        user_protrait_row = []
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            user_protrait_row.append(float(item_2))
        user_protrait_processed.append(user_protrait_row)

    exposed_items_id = []
    for item in exposed_items:
        exposed_items_id_row = []
        item_split_list = item.split(',')
        for item_id in item_split_list:
            exposed_items_id_row.append(float(item_id))
        exposed_items_id.append(exposed_items_id_row)

    exposed_items_processed = []
    for item in exposed_items:
        exposed_items_row = []
        item_split_list = item.split(',')
        for item_id in item_split_list:
            item_info_dic = item_info_list[int(item_id)-1]
            item_info = item_info_dic[float(item_id)]

            exposed_items_row.append(item_info)
        exposed_items_processed.append(exposed_items_row)
    
    labels_processed = []
    for item in labels:
        labels_row = []
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            labels_row.append(float(item_2))
        labels_processed.append(labels_row)
        
    return user_click_history_processed, user_click_history_avg_processed, user_protrait_processed, exposed_items_processed, labels_processed, exposed_items_id
```

```python id="dQuZy-BOsvAu"
def load_item_info(path):
    price_max = 150.0
    price_min = 16621.0
    item_info_list = []
    item_id, item_vec, price, location = [], [], [], []
    i = 0
    with io.open(path,'r') as file:
        for line in file:
            if i > 0:
                item_info = {}
                item_vec_row = []

                item_id_1, item_vec_1, price_1, location_1 = line.split(' ')
                item_id_1 = float(item_id_1)
                price_1 = float(price_1)
                price_1 = (price_1 - price_min) / (price_max - price_min)
                location_1 = float(location_1)

                item_vec_list = item_vec_1.split(',')
                for item_2 in item_vec_list:
                    item_vec_row.append(float(item_2))

                item_id.append(item_id_1)
                item_vec.append(item_vec_row)
                price.append(price_1)
                location.append(location_1)

                item = []
                for j in range(len(item_vec_row)):
                    item.append(item_vec_row[j])
                item.append(price_1)
                item.append(location_1)
                item_info[item_id_1] = item
                
                item_info_list.insert(int(item_id_1)-1, item_info)

            i = i + 1
    return item_info_list
```

```python id="uqfsrDgcszTk"
def preprocess_trainset_data():

    trainset_path = osp.join(args.datapath_bronze, args.filename_trainset)
    iteminfo_path = osp.join(args.datapath_bronze, args.filename_iteminfo)
    savepath = osp.join(args.datapath_silver,args.filename_trainset_processed)

    if not osp.exists(savepath):
        logger.info('Loading Items Info')
        item_info_list = load_item_info(iteminfo_path)

        logger.info('Loading Trainset')
        user_id, user_click_history, user_protrait, exposed_items, labels, time = load_dataset(trainset_path)

        logger.info('Processing Trainset')
        user_click_history_processed, user_click_history_avg_processed, user_protrait_processed, exposed_items_processed, labels_processed, exposed_items_id = data_processing(user_click_history, user_protrait, exposed_items, labels, item_info_list)

        logger.info('Scaling Features')
        scaler = StandardScaler()
        user_click_history_processed = scaler.fit_transform(user_click_history_processed).tolist()
        user_protrait_processed = scaler.fit_transform(user_protrait_processed).tolist()

        processed_trainset_data = {
            'user_click_history_processed':user_click_history_processed,
            'user_click_history_avg_processed':user_click_history_avg_processed,
            'user_protrait_processed':user_protrait_processed,
            'exposed_items_processed':exposed_items_processed,
            'labels_processed':labels_processed,
            'exposed_items_id':exposed_items_id,
        }

        save_pickle(processed_trainset_data, savepath)
        logger.info('Processed data saved at {}'.format(savepath))
    else:
        logger.info('{} Processed data already exists, skipping!'.format(savepath))
```

<!-- #region id="3uQR8SuF2Szh" -->
---
<!-- #endregion -->

```python id="CG-uDDlHszRf"
def load_track2_test_dataset(path):
    i = 0
    user_id, user_click_history, user_protrait = [], [], []
    with io.open(path,'r') as file:
        for line in file:
            if i > 0:
                user_id_1, user_click_history_1, user_protrait_1 = line.split(' ')
                user_id.append(user_id_1)
                user_click_history.append(user_click_history_1)
                user_protrait.append(user_protrait_1)
            i = i + 1
    return user_id, user_click_history, user_protrait
```

```python id="DYSTaN_HszPe"
def data_track2_test_processing(user_click_history, user_protrait, item_info_list):
    user_click_history_processed = []
    for item in user_click_history:
        user_click_history_row = []
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            click_history = float(item_2.split(':')[0])
            user_click_history_row.append(click_history)

        if len(user_click_history_row) < 249:
            for i in range(249-len(user_click_history_row)):
                user_click_history_row.append(0.0)
        
        if len(user_click_history_row) > 249:
            user_click_history_row = user_click_history_row[:249]

        user_click_history_processed.append(user_click_history_row)

    user_click_history_avg_processed = []
    for item in user_click_history:
        user_click_history_row_avg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            click_history_item_id = float(item_2.split(':')[0])
            
            if click_history_item_id == 0.0:
                continue

            item_info_dic = item_info_list[int(click_history_item_id)-1]
            item_info = item_info_dic[float(click_history_item_id)]
            user_click_history_row_avg = user_click_history_row_avg + np.array(item_info)
            
        user_click_history_row_avg = user_click_history_row_avg / len(item_split_list)
        user_click_history_row_avg = user_click_history_row_avg.tolist()

        user_click_history_avg_processed.append(user_click_history_row_avg)

    user_protrait_processed = []
    for item in user_protrait:
        user_protrait_row = []
        item_split_list = item.split(',')
        for item_2 in item_split_list:
            user_protrait_row.append(float(item_2))
        user_protrait_processed.append(user_protrait_row)

    return user_click_history_processed, user_click_history_avg_processed, user_protrait_processed
```

```python id="XDIfH7NWG8bE"
def preprocess_track2_testset_data():

    track2_testset_path = osp.join(args.datapath_bronze, args.filename_track2_testset)
    iteminfo_path = osp.join(args.datapath_bronze, args.filename_iteminfo)
    savepath = osp.join(args.datapath_silver,args.filename_track2_testset_processed)

    if not osp.exists(savepath):
        logger.info('Loading Items Info')
        item_info_list = load_item_info(iteminfo_path)

        logger.info('Loading Track2 Testset')
        user_id, user_click_history, user_protrait = load_track2_test_dataset(track2_testset_path)

        logger.info('Processing Track 2 Testset')
        user_click_history_processed, user_click_history_avg_processed, user_protrait_processed = data_track2_test_processing(user_click_history, user_protrait, item_info_list)

        logger.info('Scaling Features')
        scaler = StandardScaler()
        user_click_history_processed = scaler.fit_transform(user_click_history_processed).tolist()
        user_protrait_processed = scaler.fit_transform(user_protrait_processed).tolist()

        processed_track2_testset_data = {
            'user_click_history_processed':user_click_history_processed,
            'user_click_history_avg_processed':user_click_history_avg_processed,
            'user_protrait_processed':user_protrait_processed,
            'item_info_list':item_info_list,
        }


        save_pickle(processed_track2_testset_data, savepath)
        logger.info('Processed data saved at {}'.format(savepath))
    else:
        logger.info('{} Processed data already exists, skipping!'.format(savepath))
```

<!-- #region id="Xaa771VJIqUF" -->
---
<!-- #endregion -->

```python id="5oyQnqTRsw9p"
def concat_feature_batch(user_click_history_processed_batch, user_click_history_avg_processed_batch, user_protrait_processed_batch, exposed_item_feature_processed_batch):
    feature_batch = []
    for i in range(len(user_click_history_processed_batch)):
        feature_row = user_click_history_processed_batch[i] + user_click_history_avg_processed_batch[i] + user_protrait_processed_batch[i] + exposed_item_feature_processed_batch[i]
        feature_batch.append(feature_row)
    return feature_batch
```

```python id="--iq5N9Ts6iU"
def get_action_info(action, item_info_list):
    item_info_dic = item_info_list[int(action)-1]
    item_info = item_info_dic[float(action)]
    return item_info
```

```python id="KHbgoGpKNYOG"
def write_csv(action_result_list):
    import pandas as pd
    import csv

    test2_set_path = osp.join(args.datapath_bronze, args.filename_track2_testset)
    test2_set = pd.read_csv(test2_set_path)
    item_id_list = test2_set['user_id'].tolist()
    res_list = []

    for row_list in action_result_list:
        row_list = list(map(str, row_list))
        row_str = ' '.join(row_list)
        res_list.append(row_str)

    path = osp.join(args.datapath_silver,'submissions.csv')
    with open(path, 'w', newline='', encoding='utf8') as f:
        csv_write = csv.writer(f)
        id = item_id_list
        id = [ str(i) for i in id]

        pred = res_list

        head = ('id', 'category')
        csv_write.writerow(head)
        for pair in zip(id, pred):
            csv_write.writerow(pair)
```

<!-- #region id="AmnI-JVAJ-h5" -->
---
<!-- #endregion -->

```python id="vC3toVkNKEln"
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC_Q, self).__init__()
		self.q1 = nn.Linear(state_dim, 512)
		self.q2 = nn.Linear(512, 256)
		self.q3 = nn.Linear(256, num_actions)

		self.i1 = nn.Linear(state_dim, 512)
		self.i2 = nn.Linear(512, 256)
		self.i3 = nn.Linear(256, num_actions)

	def forward(self, state):
		q = F.relu(self.q1(state))
		q = F.relu(self.q2(q))

		i = F.relu(self.i1(state))
		i = F.relu(self.i2(i))
		i = F.relu(self.i3(i))
		return self.q3(q), F.log_softmax(i, dim=1), i
```

```python id="_Ym3ws6KKGlU"
class discrete_BCQ(object):
	def __init__(
		self, 
		num_actions,
		state_dim,
		device,
		BCQ_threshold=0.3,
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=False,
		target_update_frequency = 1000,
		tau=0.005,
		initial_eps = 1,
		end_eps = 0.001,
		eps_decay_period = 25e4,
		eval_eps=0.001,
	):
	
		self.device = device

		# Determine network type
		self.Q = FC_Q(state_dim, num_actions).to(self.device)
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# Decay for eps
		self.initial_eps = initial_eps
		self.end_eps = end_eps
		self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

		# Evaluation hyper-parameters
		self.state_shape =  (-1, state_dim)
		self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Threshold for "unlikely" actions
		self.threshold = BCQ_threshold

		# Number of training iterations
		self.iterations = 0


	def select_action(self, state, step, action_list, eval=False):
		mask_list = []
		if 0<=step and step<=2:
			mask_list += [a for a in range(39, 381)]
		elif 3<=step and step<=5:
			mask_list += [a for a in range(0, 39)] + [b for b in range(147, 381)]
		elif 6<=step and step<=8:
			mask_list += [a for a in range(0, 147)]
		action_list = (np.array(action_list) - 1).tolist()
		mask_list += action_list
		mask_list = list(set(mask_list))

		# Select action according to policy with probability (1-eps) otherwise, select random action
		if np.random.uniform(0, 1) > self.eval_eps:
			with torch.no_grad():
				state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
				q, imt, i = self.Q(state)
				# action mask
				for idx in range(q.shape[0]):
					imt[idx][mask_list] += -1e10
				imt = imt.exp()
				imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
				return int((imt * q + (1. - imt) * -1e8).argmax(1))
		else:
			return np.random.randint(self.num_actions)


	def train_batch(self, state, action, next_state, reward, done):
		for bth_idx in range(len(action)):
			action[bth_idx] = action[bth_idx] - 1.0
	
		state = torch.FloatTensor(state).to(self.device)
		next_state = torch.FloatTensor(next_state).to(self.device)
		action = torch.LongTensor(action).unsqueeze(1).to(self.device)
		reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
		done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

		# Compute the target Q value
		with torch.no_grad():
			q, imt, i = self.Q(next_state)
			imt = imt.exp()
			imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

			# Use large negative number to mask actions from argmax
			next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

			q, imt, i = self.Q_target(next_state)
			target_Q = reward + (1.0 - done) * self.discount * q.gather(1, next_action).reshape(-1, 1)

		# Get current Q estimate
		current_Q, imt, i = self.Q(state)
		current_Q = current_Q.gather(1, action)

		# Compute Q loss
		q_loss = F.smooth_l1_loss(current_Q, target_Q)
		i_loss = F.nll_loss(imt, action.reshape(-1))

		Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

		# Optimize the Q
		self.Q_optimizer.zero_grad()
		Q_loss.backward()
		self.Q_optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()

		return Q_loss

	# hard update: theta' = theta
	# soft update(or Polyak update): theta' = tau*theta + (1-tau)*theta', tau is a little value, such as 0.001
	# θ_target = τ*θ_local + (1 - τ)*θ_target
	def polyak_target_update(self):
		for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
		   target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	# hard update: theta' = theta
	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
			 self.Q_target.load_state_dict(self.Q.state_dict())
	
	def train(self, state, action, next_state, reward, done):
		action = action - 1.0

		state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
		action = torch.LongTensor(np.array(action)).unsqueeze(0).unsqueeze(0).to(self.device)
		reward = torch.FloatTensor(np.array(reward)).unsqueeze(0).unsqueeze(0).to(self.device)
		done = torch.FloatTensor(np.array(done)).unsqueeze(0).unsqueeze(0).to(self.device)

		# Compute the target Q value
		# r_t + Q'(s_{t+1}, argmax_a Q(s_{t+1}, a))
		with torch.no_grad():
			q, imt, i = self.Q(next_state)#q:torch.Size([1, 381])
			imt = imt.exp()
			imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

			# Use large negative number to mask actions from argmax
			next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True) #torch.Size([1, 1])

			q, imt, i = self.Q_target(next_state) #torch.Size([1, 381])
			# target network
			target_Q = reward + (1.0 - done) * self.discount * q.gather(1, next_action).reshape(-1, 1)

		# Get current Q estimate
		# Q(s_t, a_t)
		current_Q, imt, i = self.Q(state)
		current_Q = current_Q.gather(1, action)

		# Compute Q loss
		q_loss = F.smooth_l1_loss(current_Q, target_Q)
		i_loss = F.nll_loss(imt, action.reshape(-1))

		Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

		# Optimize the Q
		self.Q_optimizer.zero_grad()
		Q_loss.backward()
		self.Q_optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()

		# 返回Q_loss以供打印
		return Q_loss
```

<!-- #region id="Zmq6G_83KEQD" -->
---
<!-- #endregion -->

```python id="C10TeZxeLVQN"
def eval_policy(policy):
    logger.info("Evaluating policy   ...")

    testset_path = osp.join(args.datapath_silver, args.filename_track2_testset_processed)
    track2_testset_processed = load_pickle(testset_path)
    test_user_click_history_processed = track2_testset_processed['user_click_history_processed']
    test_user_click_history_avg_processed = track2_testset_processed['user_click_history_avg_processed']
    test_user_protrait_processed = track2_testset_processed['user_protrait_processed']
    item_info_list = track2_testset_processed['item_info_list']

    num_track2_test_set = len(test_user_click_history_processed)

    action_result_list = []
    for test_iters in tqdm(range(num_track2_test_set)):
        test_user_click_history_processed_row = test_user_click_history_processed[test_iters]
        test_user_click_history_avg_processed_row = test_user_click_history_avg_processed[test_iters]
        test_user_protrait_processed_row = test_user_protrait_processed[test_iters]

        num_row = 9
        action_list = []
        reward_row = 0.0
        
        for i in range(num_row):
            if i == 0:
                action_item_feature = [0, 0, 0, 0, 0, 0, 0]
            else:
                action_item_feature = get_action_info(action_list[i-1], item_info_list)
            
            state = test_user_click_history_processed_row + test_user_click_history_avg_processed_row + test_user_protrait_processed_row + action_item_feature

            action = 1 + policy.select_action(state, i, action_list)
            
            action_list.append(action)

            reward = action_item_feature[-2]
            reward_row += reward
        
        action_result_list.append(action_list)

    return action_result_list
```

```python id="yHrGOevyMaDV"
def train_BCQ_batch():
    """Trains BCQ offline Batch"""

    logger.info('epoch_num: {}'.format(args.epoch_num))
    logger.info('batch_size: {}'.format(args.batch_size))

    logger.info('Initialize and load policy')
    policy = discrete_BCQ(
        args.num_actions,
        args.state_dim,
        device,
        0.3,
        args.discount,
        args.optimizer,
        args.optimizer_parameters,
        args.polyak_target_update,
        args.target_update_freq,
        args.tau,
        args.initial_eps,
        args.end_eps,
        args.eps_decay_period,
        args.eval_eps
	)
 
    logger.info('Load training dataset')
    trainset_path = osp.join(args.datapath_silver, args.filename_trainset_processed)
    trainset_processed = load_pickle(trainset_path)
    user_click_history_processed = trainset_processed['user_click_history_processed']
    user_click_history_avg_processed = trainset_processed['user_click_history_avg_processed']
    user_protrait_processed = trainset_processed['user_protrait_processed']
    exposed_items_processed = trainset_processed['exposed_items_processed']
    labels_processed = trainset_processed['labels_processed']
    exposed_items_id = trainset_processed['exposed_items_id']

    num_train_set = len(user_click_history_processed)
    
    logger.info("Training  ...")
    batch_num = num_train_set // args.batch_size
    q_loss_list = []
    reward_list = []
    for epoch in range(args.epoch_num):
        idx = np.random.permutation(num_train_set)
        q_loss_total = 0.0
        reward_total = 0.0
        for i in range(batch_num):
            if i%10==0:
                logger.info('epoch={},batch={}/{}'.format(epoch,i,batch_num))
            batch_idx = idx[i*args.batch_size:(i+1)*args.batch_size].tolist()
            user_click_history_processed_batch = []
            user_click_history_avg_processed_batch = []
            user_protrait_processed_batch = []
            exposed_item_batch = []
            labels_batch = []
            exposed_items_id_batch = []
            for i_idx in batch_idx:
                user_click_history_processed_batch.append(user_click_history_processed[i_idx])
                user_click_history_avg_processed_batch.append(user_click_history_avg_processed[i_idx])
                user_protrait_processed_batch.append(user_protrait_processed[i_idx])
                exposed_item_batch.append(exposed_items_processed[i_idx])
                labels_batch.append(labels_processed[i_idx])
                exposed_items_id_batch.append(exposed_items_id[i_idx])

            num_step = 9
            for step in range(num_step):                
                if step == 0:
                    exposed_item_feature_last_batch = []
                    for bth in range(args.batch_size):
                        exposed_item_feature_last_batch.append([0, 0, 0, 0, 0, 0, 0])
                else:
                    exposed_item_feature_last_batch = []
                    for bth in range(args.batch_size):
                        exposed_item_feature_last_batch.append(exposed_item_batch[bth][step - 1])
                exposed_item_feature_cur_batch = []
                for bth in range(args.batch_size):
                    exposed_item_feature_cur_batch.append(exposed_item_batch[bth][step])

                # state : user_click_history + user_protrait + last product features
                state = concat_feature_batch(user_click_history_processed_batch, user_click_history_avg_processed_batch, user_protrait_processed_batch, exposed_item_feature_last_batch) #list 266

                # next_state : user_click_history + user_protrait + current product features
                next_state = concat_feature_batch(user_click_history_processed_batch, user_click_history_avg_processed_batch, user_protrait_processed_batch, exposed_item_feature_cur_batch)#list 266

                # action
                reward = []
                action = []
                for bth in range(args.batch_size):
                    if labels_batch[bth][step] == 1.0:
                        reward.append(labels_batch[bth][step] * exposed_item_feature_cur_batch[bth][-2])
                    else:
                        reward.append((-0.25) * exposed_item_feature_cur_batch[bth][-2])
                    action.append(exposed_items_id_batch[bth][step])

                done = args.batch_size*[1.0] if (step == num_step-1) else args.batch_size*[0.0]
                q_loss = policy.train_batch(state, action, next_state, reward, done)
                q_loss_total += q_loss.item()
                reward_total += sum(reward)
        
        q_loss_list.append(q_loss_total)
        reward_list.append(reward_total)

        # evaluations
        if epoch > 0 and epoch % 30 == 0:
            _ = eval_policy(policy)

    logger.info("Predicting & writing csv file  ...")
    action_result_list = eval_policy(policy)
    write_csv(action_result_list)
```

<!-- #region id="koFQxtgos6gE" -->
## Jobs
<!-- #endregion -->

```python id="2y8mdDjds6dr"
logger.info('JOB START: DOWNLOAD_RAW_DATASET')
download_dataset()
logger.info('JOB END: DOWNLOAD_RAW_DATASET')
```

```python id="3ig3tPpB2Fx-"
logger.info('JOB START: DATASET_CONVERSION_PARQUET_TO_CSV')
convert_dataset()
logger.info('JOB END: DATASET_CONVERSION_PARQUET_TO_CSV')
```

```python id="pbQUhQUDGHlz"
logger.info('JOB START: TRAINSET_DATA_PREPROCESSING')
preprocess_trainset_data()
logger.info('JOB END: TRAINSET_DATA_PREPROCESSING')
```

```python id="hh3rb38eJU7b"
logger.info('JOB START: TRACK2_TESTSET_DATA_PREPROCESSING')
preprocess_track2_testset_data()
logger.info('JOB END: TRACK2_TESTSET_DATA_PREPROCESSING')
```

```python id="eHZDmoEUahTW"
logger.info('JOB START: BCQ_MODEL_TRAINING_AND_EVALUATION')
train_BCQ_batch()
logger.info('JOB END: BCQ_MODEL_TRAINING_AND_EVALUATION')
```
