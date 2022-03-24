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

<!-- #region id="jvhe3raF5V0J" -->
# DaRE Cross-domain recommender on Amazon Reviews dataset
<!-- #endregion -->

<!-- #region id="UZL_MAOGcEtj" -->
CDR utilizes information from source domains to alleviate the cold-start problem in the target domain. Early studies adopt feature mapping technique that requires overlapped users. For example, RC-DFM applies Stacked Denoising Autoencoder (SDAE) to each domain, where the learned knowledge of the same set of users are transferred from source to target domain. To overcome the restrictive requirement of overlapped users, CDLFM and CATN employ neighbor or similar user-based feature mapping. However, this kind of cross-domain algorithm implicates defects like filtering noises or requiring duplicate users.

## Problem Statement

Assume two datasets, $𝐷^𝑠$ and $𝐷^𝑡$, be the information from the source and target domains, respectively. Each dataset consists of tuples, $(𝑢,𝑖,𝑦_{𝑢,𝑖}, 𝑟_{𝑢,𝑖})$ which represents an individual review $𝑟_{𝑢,𝑖}$ written by a user 𝑢 for item 𝑖 with a rating $𝑦_{𝑢,𝑖}$. The two datasets take the form of $D^s = (𝑢^s,𝑖^s,𝑦^s_{𝑢,𝑖}, 𝑟^s_{𝑢,𝑖})$ and $D^t = (𝑢^t,𝑖^t,𝑦^t_{𝑢,𝑖}, 𝑟^t_{𝑢,𝑖})$, respectively. The goal of our task is to predict an accurate rating score $y^t_{u,i}$ using $𝐷^𝑠$ and a partial set of $𝐷^t$.
<!-- #endregion -->

<!-- #region id="LIfOZU8vcMrF" -->
## Model Architecture
<!-- #endregion -->

<!-- #region id="ItDbE2R52gDs" -->
<p><center><img src='_images/T519611_1.png'></center></p>
<!-- #endregion -->

<!-- #region id="3CxPWxDHcS19" -->
## Training Procedure

The training phase starts with review embedding layers followed by three types of feature extractors, ${𝐹𝐸}^𝑠$, ${𝐹𝐸}^c$, and ${𝐹𝐸}^t$, named source, common, and target, for the separation of domain-specific, domain-common knowledge. Integrated with domain discriminator, three FEs are trained independently for the parallel extraction of domain-specific $𝑂^𝑠$, $𝑂^𝑡$ and domain-common knowledge $𝑂^{𝑐,𝑠}$, $𝑂^{𝑐,𝑡}$.
<!-- #endregion -->

<!-- #region id="iXEwWiav2kLq" -->
<p><center><img src='_images/T519611_2.png'></center></p>
<!-- #endregion -->

<!-- #region id="yitawNqtcWwB" -->
Then, for each domain, the review encoder generates a single vector $𝐸^𝑠$, $𝐸^𝑡$ with extracted features 𝑂 by aligning them with individual review $𝐼^𝑠$, $𝐼^𝑡$. Finally, the regressor predicts an accurate rating that the user will give on an item. Here, shared parameters across two domains are common FE and a domain discriminator.
<!-- #endregion -->

<!-- #region id="9hNJcUY06EXn" -->
## Setup
<!-- #endregion -->

<!-- #region id="_IRLprNA6FsR" -->
### Imports
<!-- #endregion -->

```python id="-DE_6QV66J5t"
import re
import json
import numpy as np
from string import punctuation
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Function
```

```python id="v9gifAgn9x-9"
import warnings
warnings.filterwarnings('ignore') 
```

<!-- #region id="g9nByEAv6I_l" -->
### Params
<!-- #endregion -->

```python id="ZzYyF_7T6LZG"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 32
```

<!-- #region id="YQXExCxl51QU" -->
## Dataset
<!-- #endregion -->

<!-- #region id="4swld9pM8ZvD" -->
### Loading
<!-- #endregion -->

```python id="QQbU1RYS5jlj"
# !wget -q --show-progress https://anonymous.4open.science/api/repo/DaRE-9CC9/file/DaRE/Musical_Instruments.json
# !wget -q --show-progress https://anonymous.4open.science/api/repo/DaRE-9CC9/file/DaRE/Patio_Lawn_and_Garden.json

!wget -q --show-progress https://github.com/sparsh-ai/coldstart-recsys/raw/main/data/DaRE/Musical_Instruments.zip
!unzip Musical_Instruments.zip

!wget -q --show-progress https://github.com/sparsh-ai/coldstart-recsys/raw/main/data/DaRE/Patio_Lawn_and_Garden.zip
!unzip Patio_Lawn_and_Garden.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="lYVlwb_z5ut5" executionInfo={"status": "ok", "timestamp": 1635794798318, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ab8e536a-5e56-4bd2-9716-5a56e8be609c"
!head -1 Musical_Instruments.json
```

```python colab={"base_uri": "https://localhost:8080/"} id="_FtJHBUE8soB" executionInfo={"status": "ok", "timestamp": 1635794804188, "user_tz": -330, "elapsed": 5876, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="479bdb92-f0a1-4f2e-d17c-23d9b3706a14"
!wget -q --show-progress https://github.com/allenai/spv2/raw/master/model/glove.6B.100d.txt.gz
```

```python id="d0b_JFKC8vge"
!gunzip glove.6B.100d.txt.gz
```

<!-- #region id="1XsN56Eu8czS" -->
### Preprocessing
<!-- #endregion -->

```python id="34-E-Gl-8fLk"
def read_dataset(s_path, t_path):
    # Initialization
    s_dict, t_dict, w_embed = dict(), dict(), dict()
    s_data, t_train, t_valid, t_test = [], [], [], []
    len_t_data = 0

    print('\nProcessing Source & Target Data ... \n')

    f = open(s_path, 'r')

    # Read source data and generate user & item's review dict
    while True:
        line = f.readline()
        if not line: break

        # Convert str to json format
        line = json.loads(line)

        try:
            user, item, review, rating = line['reviewerID'], line['asin'], line['reviewText'], line['overall']

            review = review.lower()
            review = ''.join([c for c in review if c not in punctuation])

        except KeyError:
            continue

        s_data.append([user, item, rating])

        if user in s_dict:
            s_dict[user].append([item, review])
        else:
            s_dict[user] = [[item, review]]

        if item in s_dict:
            s_dict[item].append([user, review])
        else:
            s_dict[item] = [[user, review]]
    f.close()

    # For the separation of train / valid / test data in a target domain
    f = open(t_path, 'r')
    while True:
        len_t_data += 1
        line = f.readline()
        if not line: break

    len_train_data = int(len_t_data * 0.8)
    len_t_data = int(len_t_data * 0.2)
    f.close()

    # Read target domain's data
    f = open(t_path, 'r')
    while True:
        line = f.readline()
        if not line: break

        line = json.loads(line)

        try:
            user, item, review, rating = line['reviewerID'], line['asin'], line['reviewText'], line['overall']

            review = review.lower()
            review = ''.join([c for c in review if c not in punctuation])

        except KeyError:
            continue

        if user in t_dict and item in t_dict and len(t_valid) < len_t_data:
            t_valid.append([user, item, rating])
        else:
            if len(t_train) > len_train_data:
                break

            t_train.append([user, item, rating])

            if user in t_dict:
                t_dict[user].append([item, review])
            else:
                t_dict[user] = [[item, review]]
            if item in t_dict:
                t_dict[item].append([user, review])
            else:
                t_dict[item] = [[user, review]]

    f.close()

    # Split valid / test data
    t_test, t_valid = t_valid[int(len_t_data/2):len_t_data], t_valid[0:int(len_t_data/2)]

    print('Size of Train / Valid / Test data  : %d / %d / %d' % (len(t_train), len(t_valid), len(t_test)))

    # Dictionary for word embedding
    f = open('glove.6B.100d.txt')

    for line in f:
        word_vector = line.split()
        word = word_vector[0]
        word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
        w_embed[word] = word_vector_arr

    f.close()

    return s_data, s_dict, t_train, t_valid, t_test, t_dict, w_embed
```

<!-- #region id="KbQgd-Eu54a1" -->
**Define GRL for common feature extraction**
<!-- #endregion -->

```python id="Sz-LpCDL6wwX"
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.lambda_ = 1
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = 1
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None
```

<!-- #region id="Zwqh6Hxp6qNZ" -->
## Model Definition
<!-- #endregion -->

```python id="eZHvt9CL6o18"
class DaRE(nn.Module):
    def __init__(self):
        super(DaRE, self).__init__()
        # Num of CNN filter, CNN filter size 5x100
        self.filters_num = 100
        self.kernel_size = 5
        # Word embedding dimension
        self.word_dim = 100
        # Loss for siamese encoder
        self.dist = nn.MSELoss()

        self.s_user_feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim)),
            nn.BatchNorm2d(self.filters_num),
            nn.Sigmoid(),
            nn.MaxPool2d((496, 1)),
            nn.Dropout(),
        )

        self.s_item_feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim)),
            nn.BatchNorm2d(self.filters_num),
            nn.Sigmoid(),
            nn.MaxPool2d((496, 1)),
            nn.Dropout(),
        )

        self.t_user_feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim)),
            nn.BatchNorm2d(self.filters_num),
            nn.Sigmoid(),
            nn.MaxPool2d((496, 1)),
            nn.Dropout(),
        )

        self.t_item_feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim)),
            nn.BatchNorm2d(self.filters_num),
            nn.Sigmoid(),
            nn.MaxPool2d((496, 1)),
            nn.Dropout(),
        )

        self.c_user_feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim)),
            nn.BatchNorm2d(self.filters_num),
            nn.Sigmoid(),
            nn.MaxPool2d((496, 1)),
            nn.Dropout(),
        )

        self.c_item_feature_extractor = nn.Sequential(
            nn.Conv2d(1, self.filters_num, (self.kernel_size, self.word_dim)),
            nn.BatchNorm2d(self.filters_num),
            nn.Sigmoid(),
            nn.MaxPool2d((496, 1)),
            nn.Dropout(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(200, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
        )

        self.s_encoder = nn.Sequential(
            nn.Linear(200, 200)
        )

        self.s_classifier = nn.Sequential(
            nn.Linear(200, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
        )

        self.t_encoder = nn.Sequential(
            nn.Linear(200, 200)
        )

        self.t_classifier = nn.Sequential(
            nn.Linear(200, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
        )

        self.reset_para()

    def reset_para(self):
        for cnn in [self.s_user_feature_extractor[0], self.s_item_feature_extractor[0]]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for cnn in [self.t_user_feature_extractor[0], self.t_item_feature_extractor[0]]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for cnn in [self.c_user_feature_extractor[0], self.c_item_feature_extractor[0]]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for fc in [self.s_classifier[0]]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

        for fc in [self.t_classifier[0]]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

    def forward(self, user, item, ans, label):
        # Source individual review FE
        s_u_ans_fea = self.s_user_feature_extractor(ans).squeeze(2).squeeze(2)
        c_u_ans_fea = self.c_user_feature_extractor(ans).squeeze(2).squeeze(2)
        s_u_ans_fea = (s_u_ans_fea + c_u_ans_fea) / 2

        s_i_ans_fea = self.s_item_feature_extractor(ans).squeeze(2).squeeze(2)
        c_i_ans_fea = self.c_item_feature_extractor(ans).squeeze(2).squeeze(2)
        s_i_ans_fea = (s_i_ans_fea + c_i_ans_fea) / 2

        s_ans_fea = torch.cat((s_u_ans_fea, s_i_ans_fea), 1).squeeze(1)

        # Label of source individual review
        s_cls_out = self.s_classifier(s_ans_fea)

        # Output is [Source | Target] --> Masking target output for loss calculation
        masking = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).view(batch_size * 2, -1).to(device)
        s_ans_out, s_label = torch.mul(s_cls_out, masking), torch.mul(label, masking)

        # Source aggregated reviews FE
        s_u_fea = self.s_user_feature_extractor(user).squeeze(2).squeeze(2)
        s_i_fea = self.s_item_feature_extractor(item).squeeze(2).squeeze(2)

        s_c_u_fea = self.c_user_feature_extractor(user).squeeze(2).squeeze(2)
        s_c_i_fea = self.c_item_feature_extractor(item).squeeze(2).squeeze(2)

        s_u_fea = (s_u_fea + s_c_u_fea) / 2
        s_i_fea = (s_i_fea + s_c_i_fea) / 2

        s_fea = torch.cat((s_u_fea, s_i_fea), 1).squeeze(1)

        # Passing through encoder for aggregated review embedding
        s_fea = self.s_encoder(s_fea)

        s_cls_out = self.s_classifier(s_fea)
        s_out = torch.mul(s_cls_out, masking)

        # Distance between individual review & aggregated review
        s_dist = self.dist(torch.mul(s_ans_fea, masking), torch.mul(s_fea, masking))

        # Same for target domain
        t_u_ans_fea = self.t_user_feature_extractor(ans).squeeze(2).squeeze(2)
        c_u_ans_fea = self.c_user_feature_extractor(ans).squeeze(2).squeeze(2)
        t_u_ans_fea = (t_u_ans_fea + c_u_ans_fea) / 2

        t_i_ans_fea = self.t_item_feature_extractor(ans).squeeze(2).squeeze(2)
        c_i_ans_fea = self.c_item_feature_extractor(ans).squeeze(2).squeeze(2)
        t_i_ans_fea = (t_i_ans_fea + c_i_ans_fea) / 2

        t_ans_fea = torch.cat((t_u_ans_fea, t_i_ans_fea), 1).squeeze(1)

        t_cls_out = self.t_classifier(t_ans_fea)

        masking = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)]).view(batch_size * 2, -1).to(device)
        t_ans_out, t_label = torch.mul(t_cls_out, masking), torch.mul(label, masking)

        # Target classification loss
        t_u_fea = self.t_user_feature_extractor(user).squeeze(2).squeeze(2)
        t_i_fea = self.t_item_feature_extractor(item).squeeze(2).squeeze(2)

        t_c_u_fea = self.c_user_feature_extractor(user).squeeze(2).squeeze(2)
        t_c_i_fea = self.c_item_feature_extractor(item).squeeze(2).squeeze(2)

        t_u_fea = (t_u_fea + t_c_u_fea) / 2
        t_i_fea = (t_i_fea + t_c_i_fea) / 2

        t_fea = torch.cat((t_u_fea, t_i_fea), 1).squeeze(1)

        t_fea = self.t_encoder(t_fea)

        t_cls_out = self.t_classifier(t_fea)
        t_out = torch.mul(t_cls_out, masking)

        t_dist = self.dist(torch.mul(t_ans_fea, masking), torch.mul(t_fea, masking))

        # Discriminator label
        s_domain_specific = torch.zeros(batch_size).to(device)
        t_domain_specific = torch.ones(batch_size).to(device)

        # Common source discriminator loss
        s_c_d_fea = torch.cat((s_c_u_fea, s_c_i_fea), 1)
        s_c_d_fea = GradientReversalFunction.apply(s_c_d_fea)
        s_c_d_fea = self.discriminator(s_c_d_fea).squeeze(1)[0:batch_size]
        s_c_domain_loss = F.binary_cross_entropy_with_logits(s_c_d_fea, s_domain_specific)

        # Common target discriminator loss
        t_c_d_fea = torch.cat((t_c_u_fea, t_c_i_fea), 1)
        t_c_d_fea = GradientReversalFunction.apply(t_c_d_fea)
        t_c_d_fea = self.discriminator(t_c_d_fea).squeeze(1)[batch_size:batch_size * 2]
        t_c_domain_loss = F.binary_cross_entropy_with_logits(t_c_d_fea, t_domain_specific)

        domain_common_loss = (s_c_domain_loss + t_c_domain_loss) / 2

        # Source specific discriminator loss
        s_d_fea = torch.cat((s_u_fea, s_i_fea), 1)
        s_d_fea = self.discriminator(s_d_fea).squeeze(1)[0:batch_size]

        # Target specific discriminator loss
        t_d_fea = torch.cat((t_u_fea, t_i_fea), 1)
        t_d_fea = self.discriminator(t_d_fea).squeeze(1)[batch_size:batch_size * 2]

        s_domain_specific = torch.zeros(batch_size).to(device)
        s_domain_loss = F.binary_cross_entropy_with_logits(s_d_fea, s_domain_specific)
        t_domain_specific = torch.ones(batch_size).to(device)
        t_domain_loss = F.binary_cross_entropy_with_logits(t_d_fea, t_domain_specific)
        domain_specific_loss = (s_domain_loss + t_domain_loss) / 2

        return s_ans_out, s_out, s_label, s_dist, t_ans_out, t_out, t_label, t_dist, domain_common_loss, domain_specific_loss
```

<!-- #region id="UVCoOitl6lhA" -->
## Clean strings for reviews
<!-- #endregion -->

```python id="WG9z1Sr26h8g"
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)

    return string.strip().lower()
```

<!-- #region id="xwe21etL6dy6" -->
## Review embedding layer
<!-- #endregion -->

```python id="rFxm4B3I6e_M"
def pre_processing(s_data, s_dict, t_data, t_dict, w_embed, valid_idx):
    # Return embedded vector [user, item, rev_ans, rat]
    u_embed, i_embed, ans_embed, label = [], [], [], []
    limit = 500

    for idx in range(batch_size):
        u, i, rat = s_data[0][idx], s_data[1][idx], s_data[2][idx]

        u_rev, i_rev, ans_rev = [], [], []

        reviews = s_dict[u]
        for review in reviews:
            if review[0] != i:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        u_rev.append(rev)
                        if len(u_rev) > limit:
                            break
                    except KeyError:
                        continue

        reviews = s_dict[i]
        for review in reviews:
            if review[0] != u:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        i_rev.append(rev)
                        if len(i_rev) > limit:
                            break
                    except KeyError:
                        continue

        reviews = s_dict[u]
        for review in reviews:
            if review[0] == i:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        ans_rev.append(rev)
                        if len(ans_rev) > limit:
                            break
                    except KeyError:
                        continue

        if len(u_rev) > limit:
            u_rev = u_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(u_rev)
            for p in range(pend):
                u_rev.append(lis)

        if len(i_rev) > limit:
            i_rev = i_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(i_rev)
            for p in range(pend):
                i_rev.append(lis)

        if len(ans_rev) > limit:
            ans_rev = ans_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(ans_rev)
            for p in range(pend):
                ans_rev.append(lis)

        u_embed.append(u_rev)
        i_embed.append(i_rev)
        ans_embed.append(ans_rev)
        label.append([rat])

    if valid_idx:
        u_embed = torch.tensor(u_embed, requires_grad=True).view(batch_size, 1, 500, 100).to(device)
        i_embed = torch.tensor(i_embed, requires_grad=True).view(batch_size, 1, 500, 100).to(device)
        ans_embed = torch.tensor(ans_embed, requires_grad=True).view(batch_size, 1, 500, 100).to(device)
        label = torch.FloatTensor(label).to(device)

        return u_embed, i_embed, ans_embed, label

    for idx in range(batch_size):
        u, i, rat = t_data[0][idx], t_data[1][idx], t_data[2][idx]

        u_rev, i_rev, ans_rev = [], [], []

        reviews = t_dict[u]
        for review in reviews:
            if review[0] != i:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        u_rev.append(rev)
                        if len(u_rev) > limit:
                            break
                    except KeyError:
                        continue

        reviews = t_dict[i]
        for review in reviews:
            if review[0] != u:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        i_rev.append(rev)
                        if len(i_rev) > limit:
                            break
                    except KeyError:
                        continue

        reviews = t_dict[u]
        for review in reviews:
            if review[0] == i:
                review = review[1].split(' ')
                for rev in review:
                    try:
                        rev = clean_str(rev)
                        rev = w_embed[rev]
                        ans_rev.append(rev)
                        if len(ans_rev) > limit:
                            break
                    except KeyError:
                        continue

        if len(u_rev) > limit:
            u_rev = u_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(u_rev)
            for p in range(pend):
                u_rev.append(lis)

        if len(i_rev) > limit:
            i_rev = i_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(i_rev)
            for p in range(pend):
                i_rev.append(lis)

        if len(ans_rev) > limit:
            ans_rev = ans_rev[0:limit]
        else:
            lis = [0.0] * 100
            pend = limit - len(ans_rev)
            for p in range(pend):
                ans_rev.append(lis)

        u_embed.append(u_rev)
        i_embed.append(i_rev)
        ans_embed.append(ans_rev)
        label.append([rat])

    u_embed = torch.tensor(u_embed, requires_grad=True).view(batch_size * 2, 1, 500, 100).to(device)
    i_embed = torch.tensor(i_embed, requires_grad=True).view(batch_size * 2, 1, 500, 100).to(device)
    ans_embed = torch.tensor(ans_embed, requires_grad=True).view(batch_size * 2, 1, 500, 100).to(device)
    label = torch.FloatTensor(label).to(device)

    return u_embed, i_embed, ans_embed, label
```

<!-- #region id="3Lt9LqV06WWs" -->
## Training function
<!-- #endregion -->

```python id="gMnwLfpW6U9e"
def learning(s_data, s_dict, t_data, t_dict, w_embed, save, idx):
    # Model
    print('Start Training ... \n')
    enc_loss_ratio, domain_loss_ratio = 0.05, 0.1
    model = DaRE()
    # After 1 epoch, load trained parameters
    if idx == 1:
        model.load_state_dict(torch.load(save, map_location=device))
    model.to(device)
    model.train()

    criterion = nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Make batch
    batch_size = 32
    s_batch = DataLoader(s_data, batch_size=batch_size, shuffle=True, num_workers=2)
    t_batch = DataLoader(t_data, batch_size=batch_size, shuffle=True, num_workers=2)

    batch_data, zip_size = zip(s_batch, t_batch), min(len(s_batch), len(t_batch))

    for source_x, target_x in tqdm(batch_data, leave=False, total=zip_size):
        # Pre processing
        if len(source_x[0]) != batch_size or len(target_x[0]) != batch_size:
            continue

        # Get embedding of user and item reviews
        u_embed, i_embed, ans_embed, label = pre_processing(source_x, s_dict, target_x, t_dict, w_embed, 0)

        s_ans_out, s_out, s_label, s_dist, t_ans_out, t_out, t_label, t_dist, \
        c_domain_loss, domain_loss = model(u_embed, i_embed, ans_embed, label)

        # Loss
        s_ans_loss, s_loss = criterion(s_ans_out, s_label) * 2, criterion(s_out, s_label) * 2
        t_ans_loss, t_loss = criterion(t_ans_out, t_label) * 2, criterion(t_out, t_label) * 2

        # Train
        loss_func = (s_loss + t_loss + s_ans_loss + t_ans_loss) / 2 + \
                    (s_dist + t_dist) * enc_loss_ratio + (c_domain_loss + domain_loss) * domain_loss_ratio

        optim.zero_grad()
        loss_func.backward()
        optim.step()

        torch.save(model.state_dict(), save)
              
        print('Prediction Loss / Encoder Loss / Domain Loss: %.2f %.2f %.2f %.2f %.2f %.2f' %
              (s_loss, t_loss, s_dist, t_dist, c_domain_loss, domain_loss))
```

<!-- #region id="6r-C9tV56ZPK" -->
## Validation & Inference function
<!-- #endregion -->

```python id="c00Kd_KW6aIp"
def valid(v_data, t_data, t_dict, w_embed, save, write_file):
    model = DaRE()
    model.load_state_dict(torch.load(save, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.MSELoss()

    t_user_feature_extractor = model.t_user_feature_extractor
    t_item_feature_extractor = model.t_item_feature_extractor
    t_encoder = model.t_encoder
    t_clf = model.t_classifier

    c_user_feature_extractor = model.c_user_feature_extractor
    c_item_feature_extractor = model.c_item_feature_extractor

    v_batch = DataLoader(v_data, batch_size=batch_size, shuffle=True, num_workers=2)
    v_loss, idx = 0, 0

    for v_data in tqdm(v_batch, leave=False):
        if len(v_data[0]) != batch_size:
            continue
        u_embed, i_embed, ans_embed, label = pre_processing(v_data, t_dict, v_data, t_dict, w_embed, 1)

        with torch.no_grad():
            # Target rating encoder
            c_u_fea = c_user_feature_extractor(u_embed).squeeze(2).squeeze(2)
            c_i_fea = c_item_feature_extractor(i_embed).squeeze(2).squeeze(2)

            t_u_fea = t_user_feature_extractor(u_embed).squeeze(2).squeeze(2)
            t_i_fea = t_item_feature_extractor(i_embed).squeeze(2).squeeze(2)

            u_fea, i_fea = (c_u_fea + t_u_fea) / 2, (c_i_fea + t_i_fea) / 2

            t_fea = t_encoder(torch.cat((u_fea, i_fea), 1).squeeze(1))

            t_out = t_clf(t_fea)

            v_loss += criterion(t_out, label)
        idx += 1
    v_loss = v_loss / idx

    t_batch = DataLoader(t_data, batch_size=batch_size, shuffle=True, num_workers=2)
    t_loss, idx = 0, 0

    for t_data in tqdm(t_batch, leave=False):
        if len(t_data[0]) != batch_size:
            continue
        u_embed, i_embed, ans_embed, label = pre_processing(t_data, t_dict, t_data, t_dict, w_embed, 1)

        with torch.no_grad():
            # Target rating encoder
            c_u_fea = c_user_feature_extractor(u_embed).squeeze(2).squeeze(2)
            c_i_fea = c_item_feature_extractor(i_embed).squeeze(2).squeeze(2)

            t_u_fea = t_user_feature_extractor(u_embed).squeeze(2).squeeze(2)
            t_i_fea = t_item_feature_extractor(i_embed).squeeze(2).squeeze(2)

            u_fea, i_fea = (c_u_fea + t_u_fea) / 2, (c_i_fea + t_i_fea) / 2

            t_fea = t_encoder(torch.cat((u_fea, i_fea), 1).squeeze(1))

            t_out = t_clf(t_fea)

            t_loss += criterion(t_out, label)
        idx += 1

    t_loss = t_loss / idx

    print('Loss: %.4f %.4f' % (v_loss, t_loss))

    w = open(write_file, 'a')
    w.write('%.6f %.6f\n' % (v_loss, t_loss))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["71af347650b942369ec81944117b690d", "64c03e117862428caa0345c121dc1e65", "8a558da37a33403882178cb4a3aef6dd", "a950962c7bbc482894dac5691960a48c", "feb69d523c2a42368eba2e6c862aed1f", "cc2b25addf0a4c1a9c8397a7fa0b7a0b", "4f16a579d9ca49e2982612b7a69eb937", "f220f3f708ca4520bf8af56ddbfcddae", "99eb423c1fbe4f5a9003b2d42a4bdfb5", "ef71f61de89946b09497f919da483edf", "939d579f7b064b798a94a96b73826e63", "4f4509f59ac048e3b4066b5aa5e6aebc", "bb6ce32472674f07ba801c01c1afd3fb", "38015bb960f04061a49f4e2aae55b945", "13474a1fd3f74bed8efff76fbecf4a94", "7aa5e54bcece46cab79637f18a96d05f", "1b1826b54e824f4b8155ff7d1b5014a5", "ef8e674873b54bcdae218ac80261824d", "d87c8408a4344c5ca33c9ea8946ff1d7", "10274b50d21047d48c056b4e20420515", "7b7da642f2e64aa7b895c56c816dfe92", "6cfedfbd6aa740b3a0ad5b5b69266359", "278978e363ce4ad39dc9683c3a875358", "a45ba79505e84f669d3eed41ab3ea6a1", "4200609081684d48844888c7db2d9980", "bb4da70fa81b47e6b9a799c715dc5396", "0519e08c1d06482c8dce555b334bf0c4", "5c6b77244fb34c94928f306c8b5c177f", "81902f8f148d4e97b30e8ddb4c7790ca", "5454cb78d1504a93bff233ae47f0862e", "23f0caef6a194857ba14d024fa40ccc3", "3bfad4f7189c4284a879cecf44bdcb0f", "62a692be31ac4463aaca39a729e11550", "554de49e9f334bbe9e6de81e00aa6bff", "cb19273aadd04a598ab097ecd71d881a", "1139d0692c5442ad9e7855a922ff0008", "3ffe23dc45cc4a7095f3160b7c02ce12", "39dd7912df95465b8922ddab77a4c40b", "243da63d3ff6408792f11ec2bf88a88e", "dd2b38765377478a91f93799d25a6162", "de8e36541f424b8aa64682bb0b1b5251", "d9cbd681c8394e89aa2f62c2f9fe18c2", "72ee96fd766e4400bca24d7a45994bd2", "ee5bd63317644e9dbd4e3132e6c7af4a", "e01a372c721b4ad987379973cba79488", "d6d1d0235bd8428b867438df43b760ab", "e1fff4d24bbe4761b096ca4d2ba3118f", "d46cd04d4ffb4fc0998bca89036cf901", "70acccddbae2473c83e76af5558a9953", "f49280fdd2634c88a1096386bc073005", "89ee1a0ff2354807b96f5894009b5329", "4b870844c785414dbba8b6e9518cd015", "c69e47fe6b8c4372ad421f5dd62da39e", "73bc130a63f140bfb6a3852b9514352a", "6afe98930b594112b35cb6bd952de3d7", "1efeebb504b742f8b4019823650a32ef", "c5455de90d1941aeba7116538b183d16", "11b807095d544a79ade2293208e9ec41", "7ba1e4cd7a2745bebb2767accde223f3", "f3801bffc9a74e49992fc14ee0e01d51", "e8c5ae46821743588463426f237a492b", "3a3c190733c24867b92c2114d31ed3c4", "ad36ce8ecdaf4f95b28fe7035777cdd4", "8ef14db4028145d59b3311008ffac1dd", "0fe5dcd76f7c4286bd9b827649e27397", "c1682c045acd4296b48a0a76cd6b5a51", "571c9d9e2b5547d1af69e1d4f650d17c", "a820912ebccc49c18991ef5ef9165373", "fb8e04f667b649799866b17e2f169889", "6cf577f733dc4115b5d7b5f505dbe297", "5590435c86664c06ac80757bd2203d75", "c4ab49b8fb7a4dbb95e1f064e7a0ff11", "02285f8e717142be93115df03ef4c3cd", "ad88f5e7de984426abd2f5cb8e8335ab", "4af93b684f684ca587e85b009301cfec", "7a4e6883a10544c9a1041ed8896a3347", "8b352644e4f144378b9d2b28009b6875", "f54b9b78c1cd49afa93c27ee9f5dc91b", "fa7f688d3ddb41958ac16b30a461d72f", "4b44d5abe1f54eabbc5255581d147cc4", "4663e4aba9334486b9523efa33cd14de", "76b579d195c14d958ef4a029fe8e8103", "0c60a7e3ca7240a1b455d2055b9ff294", "8e9fbd6a4ec9488cb90b5bef74e6d9e0", "08c8671698ca491789e31ecc4986dc30", "7a07ee3ec0684a74bc791e7ae3a8a993", "0e991b9cc3f84c06b4ed7401462cd2de", "1088366c3ee54635ad41bed2cbf92289"]} id="HLWtIMWF8-Yq" outputId="bebcb283-5e75-4667-f06a-617f09ac4615"
if __name__ == '__main__':
    # Define paths for source & target domain
    source_path = './Musical_Instruments.json'
    target_path = './Patio_Lawn_and_Garden.json'

    iteration = 5

    path = source_path[2:-5] + '_plus_' + target_path[2:-5]
    print('Source & Target domain: ', path)

    save = './' + path + '.pth'
    write_file = './Performance_' + path + '.txt'

    s_data, s_dict, t_train, t_valid, t_test, t_dict, w_embed = read_dataset(source_path, target_path)

    for i in range(iteration):
        # After 1 epoch of training -> load trained parameter
        if i > 0:
            learning(s_data, s_dict, t_train, t_dict, w_embed, save, 1)
        # First training
        else:
            learning(s_data, s_dict, t_train, t_dict, w_embed, save, 0)

        # Validation and Test
        valid(t_valid, t_test, t_dict, w_embed, save, write_file)
```

<!-- #region id="Qdf0K8U0ccnr" -->
## Extra Notes
<!-- #endregion -->

<!-- #region id="QN10zXQhcdxN" -->
### Inference Process
<!-- #endregion -->

<!-- #region id="LPRaVnTr2pRY" -->
<p><center><img src='_images/T519611_3.png'></center></p>
<!-- #endregion -->

<!-- #region id="hKLpv5VCcgtM" -->
### Domain-Aware Feature Extraction Example
<!-- #endregion -->

<!-- #region id="1GDyHYD3cuZP" -->
Following is the example of domain-aware feature extraction from a real-world benchmark dataset Amazon.
<!-- #endregion -->

<!-- #region id="cqTLXp_I2rCs" -->
<p><center><img src='_images/T519611_4.png'></center></p>

<p><center><img src='_images/T519611_5.png'></center></p>
<!-- #endregion -->

<!-- #region id="gEPw_zBdcwIP" -->
We assume two phases: training and inference, with two different domains: Musical Instruments and Toys & Games for cross-domain recommendation scenario. The scenario assumes a training phase with source (upper) and target (lower) domain. The difference is that a common FE (red-box) is shared across domains, while the source and target FEs (green and blue boxes) are domain-specific networks. **The objective is predicting a rating that a user 𝐴 gives on item 2**. Excluding individual review, user 𝐴′𝑠 review on item 2, the source and common extractors distillate latent of user and item respectively. Specifically, for user 𝐴 in 𝑀𝑢𝑠𝑖𝑐𝑎𝑙 𝐼𝑛𝑠𝑡𝑟𝑢𝑚𝑒𝑛𝑡𝑠, a source FE captures domain-specific knowledge that she makes much of sound quality, while common FE extracts domain-common information like beautiful, and nice price. The analysis for item 2 follows the same mechanism. To summarize, DaRE model not only considers domain-shareable knowledge with common FE but also reflects domain-specific information through the source and target FE.
<!-- #endregion -->

<!-- #region id="eLSOoBHEcgnm" -->
### Review Encoder Example
<!-- #endregion -->

<!-- #region id="Ilf85u992u8N" -->
<p><center><img src='_images/T519611_6.png'></center></p>

<p><center><img src='_images/T519611_7.png'></center></p>
<!-- #endregion -->

<!-- #region id="NTrmae_6cgiL" -->
For the training of a review encoder, we utilize individual review that user 𝐴 has written on item 2 (blue box) as another label. Taking the above figure as an example, the review encoder (purple box) takes four types of inputs which are extracted from the source and common FEs. Then, the encoder generates a single output, which contains mixed information of user 𝐴 and item 2. Here, the encoder is trained to infer an individual review, negative feedback of user 𝐴 who takes sound quality into account. Likewise, another encoder in a target domain can be trained in a same manner. With user and item’s previous reviews, the encoder assumes a real feedback that user will leave after purchasing an item.
<!-- #endregion -->

<!-- #region id="gSfOe-Gocyhv" -->
**END**
<!-- #endregion -->
