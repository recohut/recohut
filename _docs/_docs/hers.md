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

<!-- #region id="uazyzsOz3cvK" -->
# HERS Cold-start Recommendations on LastFM dataset

> Modeling Influential Contexts with Heterogeneous Relations for Sparse and Cold-Start Recommendation.
<!-- #endregion -->

<!-- #region id="f91qUW98GVIi" -->
HERS consists of three heterogeneous relations: user-user, item-item, and user-item. Each user’s choice is relevant to the corresponding user’s and item’s influential contexts.
<!-- #endregion -->

<!-- #region id="YH6hSPByyc9J" -->
![](https://github.com/recohut/coldstart-recsys/raw/da72950ca514faee94f010a2cb6e99a373044ec1/docs/_images/T229879_1.png)
<!-- #endregion -->

<!-- #region id="wx2zMNhRGLND" -->
### Model Architecture
<!-- #endregion -->

<!-- #region id="qByNsRigGTDG" -->
The architecture of HERS for modeling user-item interaction with user’s and item’s influential contexts.
<!-- #endregion -->

<!-- #region id="WHmX_cCgygwp" -->
![](https://github.com/recohut/coldstart-recsys/raw/da72950ca514faee94f010a2cb6e99a373044ec1/docs/_images/T229879_2.png)
<!-- #endregion -->

<!-- #region id="Dt4u_t3qGav9" -->
Influential-Context Aggregation Unit (ICAU): A two-stage aggregation model to construct ICE.
<!-- #endregion -->

<!-- #region id="RBySJCXQyjt2" -->
![](https://github.com/recohut/coldstart-recsys/raw/da72950ca514faee94f010a2cb6e99a373044ec1/docs/_images/T229879_3.png)
<!-- #endregion -->

<!-- #region id="V4GXZQBdGEhZ" -->
## CLI Run
<!-- #endregion -->

```python id="O5TFa17hynTk"
!pip install fastFM==0.2.9
```

```python colab={"base_uri": "https://localhost:8080/"} id="0peCFDE_215A" executionInfo={"status": "ok", "timestamp": 1635692303792, "user_tz": -330, "elapsed": 1742, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="eca4468e-e757-451a-e489-c862538eca07"
%tensorflow_version 1.x
```

```python colab={"base_uri": "https://localhost:8080/"} id="nOi-m36_yTaO" executionInfo={"status": "ok", "timestamp": 1635691954353, "user_tz": -330, "elapsed": 8921, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1b1a7fa8-273f-46d2-e8db-10d089002223"
!apt-get -qq install tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="VEGCLgXxyZ1q" executionInfo={"status": "ok", "timestamp": 1635691957026, "user_tz": -330, "elapsed": 2684, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="fee4e6b0-4059-4c2c-b5a5-8c8b73305aff"
!git clone https://github.com/rainmilk/aaai19hers.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="HJ3vn2Rhyxb7" executionInfo={"status": "ok", "timestamp": 1635692305929, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a57301a7-09db-453e-9f32-16ed0f334b22"
%cd aaai19hers
```

```python colab={"base_uri": "https://localhost:8080/"} id="WW_bdf-s1n9l" executionInfo={"status": "ok", "timestamp": 1635691970090, "user_tz": -330, "elapsed": 604, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c06d2ab4-884d-4301-afcd-b7e579c05753"
!tree --du -h -C .
```

```python id="LBV41lkB10Mw"
import sys
sys.path.insert(0,'.')
```

```python colab={"base_uri": "https://localhost:8080/"} id="TTZRdXwGyccq" outputId="b6c2bd7a-34e0-4407-df40-5267cf07e997"
from model.graph_utilities import  read_graph
from model.losses import infinite_margin_loss, max_margin_loss
from keras.regularizers import l2
from model.srs_model import NetworkRS
import numpy as np
import math
from model.RSbatch import ItemGenerator,TripletGenerator
from sklearn.utils import shuffle
from model.socialRC import test_recommendation
from model.mlmr import mlmf
from model.scorer import nn_scoremodel, inner_prod_scoremodel, fm_scoremodel


data_name='lastfm'
user_net_path='datasets/%s/%s_userNet.txt'%(data_name,data_name)
ui_net_path ='datasets/%s/%s_rating.txt'%(data_name,data_name)
item_path = 'datasets/%s/%s_itemNet.txt'%(data_name,data_name)
#
# train_path = "networkRS/%s_rating_train.txt"%data_name
# test_path = "networkRS/%s_rating_test.txt"%data_name
# neg_test_path = "networkRS/%s_rating_test_neg.csv"%data_name

# item_rep_path = "networkRS/%s_item_rep_user.txt"%data_name
# user_rep_path = "networkRS/%s_user_rep_user.txt"%data_name

neg_test_path= "datasets/%s/%s_rating_test_cold_user_neg.txt"%(data_name,data_name)
train_path = "datasets/%s/%s_rating_train_cold_user.txt"%(data_name,data_name)
test_path = "datasets/%s/%s_rating_test_cold_user.txt"%(data_name,data_name)

item_rep_path = "datasets/%s/%s_item_rep_user_cold.txt"%(data_name,data_name)
user_rep_path = "datasets/%s/%s_user_rep_user_cold.txt"%(data_name,data_name)

#
# neg_test_path= "networkRS/%s_rating_test_cold_item_neg.txt"%data_name
# train_path = "networkRS/%s_rating_train_cold_item.txt"%data_name
# test_path = "networkRS/%s_rating_test_cold_item.txt"%data_name
#
# item_rep_path = "networkRS/%s_item_rep_item_cold.txt"%data_name
# user_rep_path = "networkRS/%s_user_rep_item_cold.txt"%data_name


def get_user_rep(model, nx_G, embed_len, user_rep_path, batch_size=100, save=False):
    node_size = nx_G.number_of_nodes()
    memory_output = np.zeros((node_size + 1, embed_len))

    node_list = list(nx_G.nodes())
    num_node=len(node_list)
    nb_batch = math.ceil(len(node_list) / batch_size)
    for j in range(nb_batch):
        batch_node = node_list[j * batch_size:min(num_node, (j + 1) * batch_size)]
        first_batch_data, second_batch_data = batchGenerator.get_batch_data_topk(batch_node=batch_node, topK=topK)
        memory_out = model.user_model.predict_on_batch([np.array(batch_node), first_batch_data, second_batch_data])
        memory_output[batch_node, :] = memory_out

    if save:
        np.savetxt(user_rep_path, memory_output[1:])
        print("save memory successfully")

    return memory_output[1:]

def get_cold_start_user_rep(model, embed_len, test_users, batch_size=100):
    memory_output = np.zeros((user_size + 1, embed_len))

    node_list = test_users
    num_node=len(node_list)
    nb_batch = math.ceil(len(node_list) / batch_size)
    for j in range(nb_batch):
        batch_node = node_list[j * batch_size:min(num_node, (j + 1) * batch_size)]
        first_batch_data, second_batch_data = batchGenerator.get_batch_data_topk(batch_node=batch_node, topK=topK)
        memory_out = model.first_model.predict_on_batch([np.array(batch_node), first_batch_data, second_batch_data])
        memory_output[batch_node, :] = np.squeeze(memory_out, axis=1)

    return memory_output[1:]


def get_item_rep(model, G_item, embed_len, item_rep_path, batch_size=100, save=False):
    node_list = list(G_item.nodes())
    node_size = len(node_list)
    memory_output = np.zeros((node_size + 1, embed_len))
    nb_batch = math.ceil(len(node_list) / batch_size)

    for j in range(nb_batch):
        batch_node = node_list[j * batch_size:min(node_size, (j + 1) * batch_size)]
        first_batch_data, _ =  batchGenerator.itemGenerate.get_batch_data_topk(batch_node=batch_node, topK=topK, predict_batch_size=100, order=1)

        memory_out = model.item_model.predict_on_batch([np.array(batch_node), first_batch_data])
        memory_output[batch_node, :] = memory_out

    # embedding_nodeset = embedding_matrix[node_set]
    # np.savetxt(config.embedding_path, embedding_matrix[1:])
    # np.savetxt(config.memory_path, memory_output)
    if save:
        np.savetxt(item_rep_path, memory_output[1:])
        print("save item representation successfully")

    return memory_output[1:]

def model_testembed_zero(model, test_path):
    test_data=np.loadtxt(test_path,dtype=np.int32)
    test_user_list=list(set(test_data[:,0]))
    user_embed = model.user_emb.get_weights()[0]
    user_embed[test_user_list] = 0
    model.user_embed.set_weights(user_embed)

test_data=np.loadtxt(test_path,dtype=np.int32)
test_user_list = list(set(test_data[:,0]))

G_user=read_graph(user_net_path)
G_item=read_graph(item_path)
G_ui= np.loadtxt(train_path, dtype=np.int32)



directed=False
user_list=list(G_user.nodes())
item_list=list(G_item.nodes())

user_size=len(user_list)
item_size = len(item_list)
edges = G_ui
num_edges = len(edges)



embed_len=128
topK=10
fliter_theta=16
aggre_theta=64
batch_size = 400
samples = 3
margin=20
iter_without_att = 5
iter_with_att = 25
max_iter = iter_without_att + iter_with_att
batch_num = math.ceil(num_edges / batch_size)

loss = max_margin_loss

# score_model = nn_scoremodel((embed_len,), embed_len, score_act=None)
score_model = inner_prod_scoremodel((embed_len,), score_rep_norm=False)
# score_model = fm_scoremodel((embed_len,), score_rep_norm=False, score_act=None)

pretrain_model=mlmf(nb_user=user_size+1, nb_item=item_size+1, embed_dim=embed_len,
                    score_model=score_model, reg=l2(1e-7))
pretrain_model.contrast_model.compile(loss=loss, optimizer='adam')
pretrain_model.contrast_model.summary()

pretrain_samples = 3
pretrain_batch_sz = 200
pretrain_batch_num = math.ceil(num_edges / pretrain_batch_sz)
pretrain_iter = 3

for i in range(pretrain_iter):
    shuffle(edges)
    train_loss = 0
    # print("Running on iteration %d/%d:"%(i, max_iter))
    for s in range(pretrain_samples):
        for j in range(pretrain_batch_num):
            edge_batch = np.array(edges[j * pretrain_batch_sz:min(num_edges, (j + 1) * pretrain_batch_sz)])
            batch_node_array, positive_batch_array, negative_batch_array = \
                (edge_batch[:,0], edge_batch[:,1], np.random.randint(low=1,high=item_size,size=len(edge_batch)))
            train_loss_temp = pretrain_model.contrast_model.train_on_batch(
                x=[batch_node_array, positive_batch_array, negative_batch_array,], y=margin * np.ones([len(edge_batch)]))
            train_loss += train_loss_temp

        print("Training on sample %d and iter %d" % (s + 1, i + 1))
    print("Finish iteration %d/%d with loss: %f" % (i + 1, pretrain_iter, train_loss / (pretrain_batch_num * pretrain_samples)))
    user_rep = pretrain_model.user_emb.get_weights()[0][1:]
    item_rep = pretrain_model.item_emb.get_weights()[0][1:]
    test_recommendation(user_rep, item_rep, pretrain_model.score_model, test_path, neg_test_path)
    #test_recommendation(item_rep, user_rep, test_path, neg_test_path) #for cold start item


model = NetworkRS(user_size, item_size, embed_len, score_model,
                  topK, topK, embed_regularizer=l2(5e-7), directed=directed,
                  mem_filt_alpha=fliter_theta, mem_agg_alpha=aggre_theta,
                  user_mask=None)
model.triplet_model.compile(loss=loss, optimizer='adam')
model.triplet_model.summary()

if pretrain_iter > 0:
    model.user_embed.set_weights(pretrain_model.user_emb.get_weights())
    model.item_embed.set_weights(pretrain_model.item_emb.get_weights())


batchGenerator = TripletGenerator(G_user, model, G_ui, G_item)


# model.user_embed.set_weights([user_embed])

for i in range(max_iter):
    edges = shuffle(edges)
    train_loss = 0
    # print("Running on iteration %d/%d:"%(i, max_iter))

    spl = samples if i < iter_without_att else samples
    for s in range(spl):
        for j in range(batch_num):
            edge_batch = edges[j * batch_size:min(num_edges, (j + 1) * batch_size)]
            batch_node, positive_batch, negative_batch, \
            first_batch_data, second_batch_data, \
            positive_first_batch, \
            negative_first_batch = \
                batchGenerator.generate_triplet_batch(edge_batch=edge_batch, topK=topK,
                                                       attention_sampling=i >= iter_without_att)

            batch_node_array = np.asarray(batch_node)
            positive_batch_array = np.asarray(positive_batch)
            negative_batch_array = np.asarray(negative_batch)
            train_loss_temp = model.triplet_model.train_on_batch(
                x=[batch_node_array, first_batch_data, second_batch_data,
                   positive_batch_array, positive_first_batch,
                   negative_batch_array, negative_first_batch],
                y=margin * np.ones((len(batch_node),)))
            train_loss += train_loss_temp

            if (j + 1) % 100 == 0:
                print("Training on batch %d/%d sample %d and iter %d on dataset %s" % (j + 1, batch_num, s + 1, i + 1, data_name))
    print("Finish iteration %d/%d with loss: %f" % (i + 1, max_iter, train_loss / (batch_num * spl)))

    batchGenerator.clear_node_cache()

    saveMem = (i + 1) % 5 == 0 or i == max_iter - 1
    item_rep= get_item_rep(model, G_item, embed_len, item_rep_path, batch_size=batch_size, save=saveMem)

    # user_rep = get_cold_start_user_rep(model, embed_len, test_user_list, batch_size=batch_size)
    # test_recommendation(user_rep, item_rep, test_path, neg_test_path)

    user_rep = get_user_rep(model, G_user, embed_len, user_rep_path, batch_size=batch_size, save=saveMem)
    test_recommendation(user_rep, item_rep, model.score_model, test_path, neg_test_path)
    #test_recommendation(item_rep, user_rep, test_path, neg_test_path)  # for cold start item

#
# from model.construct_RS_train import get_attention_graph_RS
# att_graph_path="./%s_att_graph.csv"%data_name
# edge=[41,2589]
# get_attention_graph_RS(model, G_user, G_item, edge, topK, att_graph_path, order=2)

# model_save_path="./%s_model.h5"%data_name
# model.triplet_model.save(model_save_path)
# print("save triplet model successfully")
```

<!-- #region id="7MzbkGTy2NW5" -->
## Citations

HERS: Modeling Influential Contexts with Heterogeneous Relations for Sparse and Cold-Start Recommendation. Hu et. al.. 2019. arXiv. [https://ojs.aaai.org//index.php/AAAI/article/view/4270](https://ojs.aaai.org//index.php/AAAI/article/view/4270)
<!-- #endregion -->
