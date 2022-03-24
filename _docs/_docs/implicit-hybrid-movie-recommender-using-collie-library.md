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

<!-- #region id="meP8PulpcavH" -->
# Implicit Hybrid Movie Recommender using Collie Library

> Testing out the features of Collie Recs library on MovieLens-100K. Training Factorization and Hybrid models with Pytorch Lightning.
<!-- #endregion -->

<!-- #region id="UDwkPy1gcA94" -->
## Setup
<!-- #endregion -->

```python id="P43fS4H27gCt"
!pip install -q collie_recs
!pip install -q git+https://github.com/sparsh-ai/recochef.git
```

```python id="PD0n8kefAb67" executionInfo={"status": "ok", "timestamp": 1635313819498, "user_tz": -330, "elapsed": 25682, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import os
import joblib
import numpy as np
import pandas as pd

from collie_recs.interactions import Interactions
from collie_recs.interactions import ApproximateNegativeSamplingInteractionsDataLoader
from collie_recs.cross_validation import stratified_split
from collie_recs.metrics import auc, evaluate_in_batches, mapk, mrr
from collie_recs.model import CollieTrainer, MatrixFactorizationModel, HybridPretrainedModel
from collie_recs.movielens import get_recommendation_visualizations

import torch
from pytorch_lightning.utilities.seed import seed_everything

from recochef.datasets.movielens import MovieLens
from recochef.preprocessing.encode import label_encode as le

from IPython.display import HTML
```

```python colab={"base_uri": "https://localhost:8080/"} id="ClWobbREN4VU" executionInfo={"status": "ok", "timestamp": 1635313819500, "user_tz": -330, "elapsed": 23, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="678bdbf8-84ec-44ec-e700-f4ffe39c10b5"
# this handy PyTorch Lightning function fixes random seeds across all the libraries used here
seed_everything(22)
```

```python colab={"base_uri": "https://localhost:8080/"} id="xfNy6xZKS5HT" executionInfo={"status": "ok", "timestamp": 1635313874039, "user_tz": -330, "elapsed": 3735, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2f6167f7-caaa-4986-ed58-1f402ff03994"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv -u -t -d -p collie_recs,pytorch_lightning,recochef
```

<!-- #region id="T0N-kmIrcDZr" -->
## Data Loading
<!-- #endregion -->

```python id="LEuHGsYgGv-e"
data_object = MovieLens()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="g3lLxQE1G120" outputId="90249953-b9c8-4dc2-a8f9-710fadac69b9"
df = data_object.load_interactions()
df.head()
```

<!-- #region id="3X8-bE9YcGBg" -->
## Preprocessing
<!-- #endregion -->

```python id="tJvifZdrLsGK"
# drop duplicate user-item pair records, keeping recent ratings only
df.drop_duplicates(subset=['USERID','ITEMID'], keep='last', inplace=True)
```

```python id="eq7GY0lEINgA"
# convert the explicit data to implicit by only keeping interactions with a rating ``>= 4``
df = df[df.RATING>=4].reset_index(drop=True)
df['RATING'] = 1
```

```python id="Jo4FnRzSHPzs"
# label encode
df, umap = le(df, col='USERID')
df, imap = le(df, col='ITEMID')

df = df.astype('int64')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="LA9BtiDrJ_wf" outputId="bbc203d9-ee6a-4874-fce6-15a4a565c907"
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="W3hUWib-MyjB" outputId="6b50d658-91b7-4609-b101-29633e7ebbfb"
user_counts = df.groupby(by='USERID')['ITEMID'].count()
user_list = user_counts[user_counts>=3].index.tolist()
df = df[df.USERID.isin(user_list)]

df.head()
```

<!-- #region id="37Y7qEEJNiE4" -->
### Interactions
While we have chosen to represent the data as a ``pandas.DataFrame`` for easy viewing now, Collie uses a custom ``torch.utils.data.Dataset`` called ``Interactions``. This class stores a sparse representation of the data and offers some handy benefits, including: 

* The ability to index the data with a ``__getitem__`` method 
* The ability to sample many negative items (we will get to this later!) 
* Nice quality checks to ensure data is free of errors before model training 

Instantiating the object is simple! 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dLcqt57TI-6l" outputId="3a8be733-de25-4117-f76a-d1e097db238e"
interactions = Interactions(
    users=df['USERID'],
    items=df['ITEMID'],
    ratings=df['RATING'],
    allow_missing_ids=True,
)

interactions
```

<!-- #region id="4TaWM_OFNZzn" -->
### Data Splits 
With an ``Interactions`` dataset, Collie supports two types of data splits. 

1. **Random split**: This code randomly assigns an interaction to a ``train``, ``validation``, or ``test`` dataset. While this is significantly faster to perform than a stratified split, it does not guarantee any balance, meaning a scenario where a user will have no interactions in the ``train`` dataset and all in the ``test`` dataset is possible. 
2. **Stratified split**: While this code runs slower than a random split, this guarantees that each user will be represented in the ``train``, ``validation``, and ``test`` dataset. This is by far the most fair way to train and evaluate a recommendation model. 

Since this is a small dataset and we have time, we will go ahead and use ``stratified_split``. If you're short on time, a ``random_split`` can easily be swapped in, since both functions share the same API! 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="U_4IPy2aNLVE" outputId="3e580753-b31b-4d82-bffa-9b226a940eb2"
train_interactions, val_interactions = stratified_split(interactions, test_p=0.1, seed=42)
train_interactions, val_interactions
```

<!-- #region id="PMN-6ktJb7qR" -->
## Train a Matrix Factorization Model
<!-- #endregion -->

<!-- #region id="ZWi4VUjeghUA" -->
### Model Architecture 
With our data ready-to-go, we can now start training a recommendation model. While Collie has several model architectures built-in, the simplest by far is the ``MatrixFactorizationModel``, which use ``torch.nn.Embedding`` layers and a dot-product operation to perform matrix factorization via collaborative filtering. 


Digging through the code of [``collie_recs.model.MatrixFactorizationModel``](../collie_recs/model.py) shows the architecture is as simple as we might think. For simplicity, we will include relevant portions below so we know exactly what we are building: 

````python
def _setup_model(self, **kwargs) -> None:
    self.user_biases = ZeroEmbedding(num_embeddings=self.hparams.num_users,
                                     embedding_dim=1,
                                     sparse=self.hparams.sparse)
    self.item_biases = ZeroEmbedding(num_embeddings=self.hparams.num_items,
                                     embedding_dim=1,
                                     sparse=self.hparams.sparse)
    self.user_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_users,
                                           embedding_dim=self.hparams.embedding_dim,
                                           sparse=self.hparams.sparse)
    self.item_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_items,
                                           embedding_dim=self.hparams.embedding_dim,
                                           sparse=self.hparams.sparse)

        
def forward(self, users: torch.tensor, items: torch.tensor) -> torch.tensor:
    user_embeddings = self.user_embeddings(users)
    item_embeddings = self.item_embeddings(items)

    preds = (
        torch.mul(user_embeddings, item_embeddings).sum(axis=1)
        + self.user_biases(users).squeeze(1)
        + self.item_biases(items).squeeze(1)
    )

    if self.hparams.y_range is not None:
        preds = (
            torch.sigmoid(preds)
            * (self.hparams.y_range[1] - self.hparams.y_range[0])
            + self.hparams.y_range[0]
        )

    return preds
````

Let's go ahead and instantiate the model and start training! Note that even if you are running this model on a CPU instead of a GPU, this will still be relatively quick to fully train. 
<!-- #endregion -->

<!-- #region id="L7o3t9vNN-Lt" -->
Collie is built with PyTorch Lightning, so all the model classes and the ``CollieTrainer`` class accept all the training options available in PyTorch Lightning. Here, we're going to set the embedding dimension and learning rate differently, and go with the defaults for everything else
<!-- #endregion -->

```python id="LNfxzlruN1xx"
model = MatrixFactorizationModel(
    train=train_interactions,
    val=val_interactions,
    embedding_dim=10,
    lr=1e-2,
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 540, "referenced_widgets": ["a532ba9270d346ecbf3361663f592c42", "fe06af14c6e14038b9e26691d97ad6b9", "4c4086c1224d41d9a52225d60dadae2a", "41cc7bf0562949f5a321dae682d9f18e", "98fde3f8c1bc4c2793f579a2e4e5aa38", "e73145e05dbe48a384d3ecdb6c69576c", "3e739cd7fa3047bca9759bda69650053", "29264e6d6a7f48118c2341c601f505ad", "e5af3beb370e48ec819d910bd3180e49", "c25e70e717de476aaf9b7925f0422376", "71352845df0b47f8ab7c48cd14c54f35", "62bbe8a9f9b3405a87e3c3a66100beab", "02c2dc5e928749bea02660a2e5132634", "32b687331314412bbdbac67679c34213", "2df4d456c10243a5ae07b5b7ce0a014e", "c4655e70154e4f5eb7dac1d695cfaf47", "e8fc918ec6844698acce9356a152d958", "e66aea70cef140d49917ecb50e8a85c2", "ea76e671074540b2a59f8f97c9add463", "66b0a5b605614de2abe0f18c0b7fee2c", "84dd3e9145be4b83ab7f98e6e182bed0", "903e0772d9e841ab93f737492ed06c47", "5c77c749fbd74b608f7a1009c611e971", "9a435425b7504085830d382b633a063e", "e1b1a5bef2354532a53c097ff3fffd46", "ebbde52678c0485bbb22d46a5701b874", "013d2784809e43be9306f49755874495", "c1484c213024483fae047f0f50026c8e", "9d1c8157d9c9426b9c73bf3e51a9176d", "59a51739a1f54a978cbd3a5e1236a4ff", "a4e4acea17774d5c95a3467afcd569fe", "4bc57309d6ac40059ffc21b9e4742d53", "44bad8cce24f4d3f8c06728f1982503c", "842ea507e1924e438ce06a6b7098331f", "a30e84e6336f4bb99767553a7693ca2e", "0e6c23659ddf4e3c8ac6cde5206b2179", "081fd50f691347bcba03000ea35bcff6", "5b5e335ba4ab4f58820fb5eccded3907", "5b0d797f65e74340baf1f76ecc5fa529", "07453b8d1f7b4452ab232d2df9aa57eb", "03d70195faa841bb9852840f84d1854c", "fa497ae9f01a427ca8cb83ca8382e07a", "497d7963e20244c484623950d7952582", "4c048c5851784d74b0038f3e5021be44", "9f6eafabad0f48329b254811eb585f14", "5d65f2138f26455fb45706b378c7b17c", "49c976510a144871b321ef28557b33a8", "f0ae0cadeace4a949f7ac32fffa399e6", "7179bd8f092a484c95a288ff603cc036", "178365af316b4a48893027027e255e03", "98a49dcf60cd4743b534915f2586869d", "7015bd2119a74aa0a5a887226f0fb187", "79bfa8ebee134b0ea6f95f647a301532", "6d420a05a21a430b8fe8ce031396c682", "df95e8f0eeeb4b949c85439ad3c1ad68", "6b1043e2720949359b3c46e3507ee315", "6969b545595e4ef7bc03963777c7381f", "7e2bc6952bd24fc1ab82160aafa55922", "21f9824f96b347bbbe24e77b8993fdb9", "1a6a4d1a9fe04729924dda3198142c30", "0e49d268b84a41c092d0b539b50e7676", "55be339ad48e45808e5f2cd77dc1c139", "5e4d214774ec4743b64e47c5b8421068", "663cfd47f8754745a6556d1804981bc6", "ee93de1b5c7b4983b0f8289bd8de62de", "c1f847aaea814f37a259ea9421bdefaa", "f67b51dae8f64dc9853e1a770e25cd6d", "c73eb00e351245499f38a4772dd9acba", "c38ef7ca20174a83ada7c238f5109cb0", "f86d88358dc848b68eaa912c09a820d4", "5fe2c6d2521c4068a49b854a3e927839", "c15e1e0c7b8c4bfe94cb50e788d88940", "fe0bf03cbd0248759c449b79e96be5e7", "dc841e06353642dc8426aa8db09a99bc", "00c84840a1cd4ddbafb10b14f62aeddd", "25bf19659d0c46f6adae4df3ef01df56", "c120854a3f1b481ab2533a551d4e7691", "af872e4f132e4008a234a999cae375e5", "092223e489bb4afbbbb8519dd1ef5ee4", "70da32049909489ebe2eb09172fdc31d", "a379e734b9e04078b7189c60477dfa02", "0e3fb6e0ada84704a8a9fc526b0dfe65", "0af8980abb8c47b2a067a2db725149a7", "4aa501bf754b4cc5a8dbd35f6207e92b", "a1a1db4342954e0db629db93ccf9da1a", "3b8e83c7236f470cbf7885803977f902", "e708dac9f06948d28260b3aff2d51b08", "061ccc416a0940dc8cdabc64e66afebe", "72a766ae6b76435d85b4c73b8ea925d8", "2ecf461577224c7abb49da80479627c2", "cc2d61893eaa45459fc4a4091d06471d", "c02d5f3ec1db4a5793f5e42c6465c2e7", "8fd5ff0a1ad3417bb37b4c446f527ba1", "aa7fc30167d14cd9b51135fe7ad4f5e5", "5dbf11d4ba234f358531e15fa28cb3b8", "85ef0885231941f78b844be13907d9eb"]} id="TKxeyMsMN1vg" outputId="72f60b35-2123-460d-d9af-5fe81b250af5"
trainer = CollieTrainer(model, max_epochs=10, deterministic=True)

trainer.fit(model)
```

<!-- #region id="b1PhE5nDghUC" -->
### Evaluate the Model 
We have a model! Now, we need to figure out how well we did. Evaluating implicit recommendation models is a bit tricky, but Collie offers the following metrics that are built into the library. They use vectorized operations that can run on the GPU in a single pass for speed-ups. 

* [``Mean Average Precision at K (MAP@K)``](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) 
* [``Mean Reciprocal Rank``](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) 
* [``Area Under the Curve (AUC)``](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) 

We'll go ahead and evaluate all of these at once below. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["55d2d78bb84d42dba55c98cefd715c0b", "3a61376a87544d23a759ce1c0bc3d006", "77a96f3fc52c40fc9e86c0008d823d3f", "5ada3f7dbe2043239abfb04edb711eaa", "b544feed25d24c7f8760504e4bb6ef67", "9d9bf1e370ba406bb234fd2220c2f4ce", "6abb7828d21e4649a21fe4c1e95a7568", "e1708e5ae59c46c4a857ee927ef70cfd"]} id="fUFAraDsOHtL" outputId="21ad00eb-4416-4c13-e82a-0522ff736a71"
model.eval()  # set model to inference mode
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc], val_interactions, model)

print(f'MAP@10 Score: {mapk_score}')
print(f'MRR Score:    {mrr_score}')
print(f'AUC Score:    {auc_score}')
```

<!-- #region id="zlqMa7eWOMGl" -->
### Inference
<!-- #endregion -->

<!-- #region id="z7y-J7KUghUD" -->
We can also look at particular users to get a sense of what the recs look like. 
<!-- #endregion -->

```python id="sIQsXedgghUD" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="0fe6ff83-b1cb-4cdf-ed59-32c2fdfb82b0"
# select a random user ID to look at recommendations for
user_id = np.random.randint(10, train_interactions.num_users)

display(
    HTML(
        get_recommendation_visualizations(
            model=model,
            user_id=user_id,
            filter_films=True,
            shuffle=True,
            detailed=True,
        )
    )
)
```

<!-- #region id="DbZ9ufGvghUE" -->
### Save and Load a Standard Model 
<!-- #endregion -->

```python id="WqJbXHXgghUG"
# we can save the model with...
os.makedirs('models', exist_ok=True)
model.save_model('models/matrix_factorization_model.pth')
```

```python id="Dz_8miLPghUG" colab={"base_uri": "https://localhost:8080/"} outputId="8efcaec9-f018-409b-a01b-e8cc137c8957"
# ... and if we wanted to load that model back in, we can do that easily...
model_loaded_in = MatrixFactorizationModel(load_model_path='models/matrix_factorization_model.pth')

model_loaded_in
```

<!-- #region id="cQpWlCuughUH" -->
Now that we've built our first model and gotten some baseline metrics, we now will be looking at some more advanced features in Collie's ``MatrixFactorizationModel``. 
<!-- #endregion -->

<!-- #region id="Q3Ne8ETsgzLe" -->
## Faster Data Loading Through Approximate Negative Sampling 
<!-- #endregion -->

<!-- #region id="8nBu6PZhgzLe" -->
With sufficiently large enough data, verifying that each negative sample is one a user has *not* interacted with becomes expensive. With many items, this can soon become a bottleneck in the training process. 

Yet, when we have many items, the chances a user has interacted with most is increasingly rare. Say we have ``1,000,000`` items and we want to sample ``10`` negative items for a user that has positively interacted with ``200`` items. The chance that we accidentally select a positive item in a random sample of ``10`` items is just ``0.2%``. At that point, it might be worth it to forgo the expensive check to assert our negative sample is true, and instead just randomly sample negative items with the hope that most of the time, they will happen to be negative. 

This is the theory behind the ``ApproximateNegativeSamplingInteractionsDataLoader``, an alternate DataLoader built into Collie. Let's train a model with this below, noting how similar this procedure looks to that in the previous tutorial. 
<!-- #endregion -->

```python id="MgSijz04gzLf"
train_loader = ApproximateNegativeSamplingInteractionsDataLoader(train_interactions, batch_size=1024, shuffle=True)
val_loader = ApproximateNegativeSamplingInteractionsDataLoader(val_interactions, batch_size=1024, shuffle=False)
```

```python id="n9wQLd9-gzLf"
model = MatrixFactorizationModel(
    train=train_loader,
    val=val_loader,
    embedding_dim=10,
    lr=1e-2,
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 557, "referenced_widgets": ["cdad48dfe56b42a9a921302dd1bb30af", "2fb34fac3cca4d9bb52ed2c18f00083d", "6ce262ade8fa446688c504c560563c04", "1be318bdc3eb4545b3e550e27c077a3e", "5da8fa8e34e140fc9daa8272ddd6d60e", "b02505f30bd04902a9884ca2a2d20cfd", "1f229f51b7334b61a320e9746c4f7725", "cefe9ed1237b497ebe84829ba4477cc4", "d0a2f3c7539f47d787665ae5dbd9c708", "be56d62fe4084218854ca1f345e8f86b", "9315661301f84e0890a82458e3864a69", "a9a9955356ac4229983c6b8f6487fa49", "84fcc5cefd89414da17801503dd3a211", "502ea894a9f649c8ad2537e946ee745e", "94ce2722931a47f590f76a73703de201", "a1e486b195604237a7cf8a72aa0e5a2e", "8a9429e4228a4bfeaf9694679a62c9b4", "1ca12c9e073f40c7a9cde5edd02081df", "b6825244b2d84f90858bfb55799a5bd9", "62f1c9537a5540be804ca19e1f8df401", "fc78a79812b24c219762e6146fb48aed", "9f6be4f1d42e4c399a76b0d9b188e138", "1845ba2a1329476bb389c890acabca8d", "f37d75b9df794546a66e3c7f89413bcc", "0878d639c1ab47db9df0434da5cd8451", "3bd890a25d9244178b8820772894caea", "b4ebed201f62472990a026a79b0c5679", "191a4352c4eb4a59b828cfa664a0680c", "dc2bcf31d6594da1b0828083e5b43915", "e90b2e2d207e42c2b2003b97e643691c", "b9cd462f5a844ff487ecdb324c76dc52", "e50d28dd314047ca9b87a13b94940a98", "c3edba0cab41447f89e1626efacaf763", "071c9dac147b418a8df4173b1f7e5c70", "b66900b838cc4ddc87ff8aaef7872cd8", "3516fa6a6b874e0d90e37c5ab8512db8", "d04e3f2f71fe423ea057975e2f2646ed", "ad24e46c04764f01afa7d5508f528f45", "4ea67205161641aba764f2fb117c918a", "efe8c0fd7bff4a6c8c0944981b21a933", "bafe7d36094f4697b15caff40fe153f4", "e170c73163954c9c800272bcca90d5db", "b28be4918c3d4ff8b29c0443d5822059", "7999001a61a946b2970e6726a2a8b7de", "7a7c1f74e81e462faef5a6d5b0597991", "c0aeb3a6257642c5a23c246d8bfd8e9d", "3dab29511c3e4368a6c2730f9b9ddaed", "39597cbc7ec74eed8d93e0b728139458", "6954ebb5b9154db2acd936945a6dd53d", "133116d7b1df4c408295d4435f18b416", "5c25caef56aa4d42a723edc792a37f52", "b0d9f5b75a4e4124ac975fd93011dec2", "d559f9388cdd44eeab9827d92514ffbb", "7b75a718d36b4462bc2d861ea37e198c", "5ce80420c5a74df080509a95c87ebc19", "a1bc2d5efa864ff3946e9fb788a25146", "bfee2e200b1345bcb8c0e888c51e574e", "cff209ba8d8a412897689f0f5676a524", "5458d5ce8b9443a29ca42f134f6dc78b", "595fb69b3a264155a7bfbe40f9bfb95b", "6795d71bc5e8445cbef35af337d4cc31", "7ba945ab3b724c8ab991474ed78269c5", "c3ae3bfff3b3441daa3ad9fa156617e5", "cf913bb565c84426aa90cab0b0e975c9", "04f49fbe1f73490bab41a2ea7cb82e0d", "e26653fbae42413880c61ae01d1a4731", "8321973a6a434b6f89a6871b36fc2893", "60260e6bcc244ef3a2970f2e881658ef", "678baf588f5b4177bf2ba5cfec8cebe1", "ef4447f28ad44167886a7b4221704416", "1e4c604275fa4c20afe7dbd8c7070e12", "773d9fc887e64c579a20af0a92646cbf", "f01ae7c057ab476c9da1530cd3db2f85", "3bf36fd8bb0448b6898ed8e7f22e159a", "d954fc0e43a6400c960c1a40ab7d33fb", "90db3526a5a1470cb24f639d27360fc4", "2cb667b093c74e84852d8fea2c70a565", "7eae1a0a5bb54eabbba3d3c2edd407fe", "4f63bc5d82c647bf865aa5cda0fac339", "5b9b7fef876547a29d7b66887f82eb07", "800fd8df329e43aea1f2aa66708711b1", "9ee9e7ac09c3435b990ef3aa0db002a5", "0c9a232a319e437c9ba50419fe1115be", "e722a30414aa4d05964955c420ac9319", "154620319753457280e583209e360f61", "72a760d2f3394b508a70e14916ea813e", "410b1388d62a4f91a34d1b6be9db4dfe", "3d74d26efda54df18d8a1a3e027b94b0", "2deab300c43a4fed95e629c6e287d015", "c0d56896a368411c9ec059197cade4b0", "10ea78570f0d41f69c9c3916c153daf5", "8d63897d48044c52ad204ceab7319b0a", "87fbca480f774101b15c5118cb4842ff", "ce97ee72c90b49eabb16de8a45f36864", "62c4b4ec0cc144a0893220bc0f351e97", "b2e0c1050787481f936429c24738e39c"]} id="AbYirCSNgzLg" outputId="f04d8302-ad3f-4dd4-bc7f-e39819349e95"
trainer = CollieTrainer(model, max_epochs=10, deterministic=True)

trainer.fit(model)
model.eval()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["d7c34c7be74246acb1a7da702b490029", "8ca15bf2e3be4ba88b7dbbf4ed26820d", "3e63ae601740455e81c35d4fbab2db6e", "b289ad3cdd3f4a25a9e92ed22c37aeed", "3d7d55e46c7d4e2caa6680229fe73b4e", "e841d95562314124b242c2e4225afbdb", "7b9d13b53df04e2082d4e236f50b807d", "9a4137c369cc4f26817ed3ab568a5ef7"]} id="xLKDSq-hgzLg" outputId="44183d74-a567-418d-a85e-946bf1443713"
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc], val_interactions, model)

print(f'MAP@10 Score: {mapk_score}')
print(f'MRR Score:    {mrr_score}')
print(f'AUC Score:    {auc_score}')
```

<!-- #region id="qipOCQzxgzLh" -->
We're seeing a small hit on performance and only a marginal improvement in training time compared to the standard ``MatrixFactorizationModel`` model because MovieLens 100K has so few items. ``ApproximateNegativeSamplingInteractionsDataLoader`` is especially recommended for when we have more items in our data and training times need to be optimized. 

For more details on this and other DataLoaders in Collie (including those for out-of-memory datasets), check out the [docs](https://collie.readthedocs.io/en/latest/index.html)! 
<!-- #endregion -->

<!-- #region id="iGgM-FaegzLi" -->
## Multiple Optimizers 
<!-- #endregion -->

<!-- #region id="4xcFspsbgzLi" -->
Training recommendation models at ShopRunner, we have encountered something we call "the curse of popularity." 

This is best thought of in the viewpoint of a model optimizer - say we have a user, a positive item, and several negative items that we hope have recommendation scores that score lower than the positive item. As an optimizer, you can either optimize every single embedding dimension (hundreds of parameters) to achieve this, or instead choose to score a quick win by optimizing the bias terms for the items (just add a positive constant to the positive item and a negative constant to each negative item). 

While we clearly want to have varied embedding layers that reflect each user and item's taste profiles, some models learn to settle for popularity as a recommendation score proxy by over-optimizing the bias terms, essentially just returning the same set of recommendations for every user. Worst of all, since popular items are... well, popular, **the loss of this model will actually be decent, solidifying the model getting stuck in a local loss minima**. 

To counteract this, Collie supports multiple optimizers in a ``MatrixFactorizationModel``. With this, we can have a faster optimizer work to optimize the embedding layers for users and items, and a slower optimizer work to optimize the bias terms. With this, we impel the model to do the work actually coming up with varied, personalized recommendations for users while still taking into account the necessity of the bias (popularity) terms on recommendations. 

At ShopRunner, we have seen significantly better metrics and results from this type of model. With Collie, this is simple to do, as shown below. 
<!-- #endregion -->

```python id="GxUMsr61gzLj"
model = MatrixFactorizationModel(
    train=train_interactions,
    val=val_interactions,
    embedding_dim=10,
    lr=1e-2,
    bias_lr=1e-1,
    optimizer='adam',
    bias_optimizer='sgd',
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 557, "referenced_widgets": ["49f97e2e5f55454bab313b1570b4b9c4", "b3c7b293fe8b44e0920b060f00382401", "80b4510b6a7e4d77879516c59662180c", "f1e818c6ef934704ae0ef1f80eaa7b5b", "695c6402e0a1475eb2965bee589c2bf4", "06f180111dfe4f57a12723bccb253f98", "316eaa17f9cd4558b8c7f6951b98d6f0", "75fc6c2f339b4e47bb22790d37e7ebab", "8d714e45f0d64e85b637961f4a25f3d0", "b5a4c2dd499c4d5c8d3ab2e890b44642", "d1dd8920ddfc4c97beffe89ece6b982e", "fc7d1f4bea4145af93d9059177a477fd", "ef66bb3e20eb4d25a6dc3af0fccf2e2a", "9275275798974477a5f5858880d1b429", "2bcffbf7f05b41688936d51d6c3ffcac", "28198fbab991465583818c47d8ddbb6d", "83658275a6c346d7822fd1368345e09a", "3389371180f64e088e7c6549a25b3ee1", "54d47349fff54cd0bca2e4903b06e5d8", "ec37483292aa4715a97c6f0de133cb24", "78845118227b40adac2503e57e1f38a7", "ce20fc80baf0494f976d7ebb54303833", "b0606438031a4d0fbe1b3b1bc4d49e9e", "38cb1183eb294363af2d2762d8a49a13", "b9a2d6afed7641d995fba91c5a94f8ec", "6f42c45ff32f46deb83569801ab976c8", "58d6593d8810421bb8bd39099e82feb9", "7cdc87e9c27f48e2b081bdbf2d6228e8", "45b86b0dc3394da4b2f87191d23dfed7", "585898b88c044e2e8d6fd0c46c3b575d", "a0259f8bb2264993a6d27acd097b6c05", "5b0ed212d376472987768ea06e314f9d", "a59ea6641fca44bbaa553a1624deab51", "671b80df47dc444d817c51101c7adf49", "d56ba50fc09c4502a1055ff9a3199062", "1bd1fa4ccb564778be34c5fe9036e731", "1b3ff709422c4864acff0ca42ec89e07", "5b6f8c0d04be4ec19c55259caf622c02", "b8903c665b7c4a3695101aa7d6e8f929", "518b4b044b4d411fb43dac5fc993c02c", "36d600f8e9be46a6b05ec580025aac20", "f14194bdbf724d20bdd7d5b572374608", "b62d3801dd3d4f45b53de6d6c25e26bd", "066ce24dcd2e46acb54f1fd437963b85", "bd80f8f37cb247c69f969d96b146ea8c", "4100968fe609430f8cd1f73c48a356f2", "9e6c6c48df824c548aa0425888adac9b", "27866404913f4820bb46d1641f63dc1f", "d75b9faa61b242279706a77458e55276", "ba792412f171494d937051219e01607d", "f43bba18ba134b269006e100f8de55d1", "63c6fb5bfa3147a89962641d1a7d215b", "6d7a19d43c4846d99d3d470fd5d7d66e", "90af7f79f905435f8c81ca21e59cab59", "fad0632c95f74aca92b6e72751da0632", "7eb49fb403564170b47e0c6f867f81e1", "57322a343ddd4ad2bc408a840eca0e98", "18b3142ac5c24401941cd4f79bac783b", "6d730313b939447bbd396526cdee3fad", "73cccc23b004406e8521b0cc26401c9c", "7520241c3b2c4591bc0532d20dafccda", "5dd8ebbab9ce418cae4eaf9f568ab5c9", "0afd2fe607b04590813dc297c3bfc4da", "c14405b6297a435eaab8032dbbb9966f", "0351e40f8c0f46708aeea7233565800a", "d7e152362cd54b4380a6cd3e506e7b86", "73dff796a59d444a96016b2578f277d7", "89a5fbcb3a4e4732a72c8cee79a346b6", "bcde36e5444444568f7a6fa34d85ccaa", "53e3249bbef446208f54a6216420062f", "235ef9871d5443a9ab3b31f1c231f53d", "b329742fdb3e44e2974d3a31700072a3", "93fbdf1f9df1464888cdecc6285ef575", "7ef0973e658740528843af596067fe8e", "dad5c7f76e20437ca651d4d39ac68660", "3a61c8ddd8b74af78c6ceffb7001da02", "fec250de7d6f45688eb48b51d837b53a", "54d415a20e204d53bbe9baddd4598e2f", "136520e97ebe4b04b9b46a44e46297fc", "c4d1f1810116465b9dcb9282eed0f113", "2c8df37c614447d5b5064dbbaa837081", "dc9b62bd3eb94dc8afd9c21491faff0b", "2416bc3ad43149ff9e8d10572693f115", "93054dc57a344b9daae9e6e36e14b594", "4d9cf1e48cb74afb91410b46b2d601b5", "897727bf519648e0bc3f204771ded957", "795429863893411cab25dfe5771a2e4a", "93a527dd2045487393fb8280ff3945b1", "3d3ca53ac0cb41d1bbb58aa754a79619", "600e766c4f4c4136875db902954f9ac7", "2cf5bd88c0c74729a89a70f6ba9dd02e", "a247748059184bbaac041b8fffd99e3f", "010c2bf6e2af470d950542394f8ceef5", "5bb0a358fcd64106acdac950f82d12a7", "16202361258644f6ab7b168990f73136", "c2fffba8678b48f9957b9a581aae9e2e"]} id="dQ_tTRfOgzLj" outputId="4d3af437-c62d-4b9b-ddb5-eaf3731556da"
trainer = CollieTrainer(model, max_epochs=10, deterministic=True)

trainer.fit(model)
model.eval()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["5c2218b13d6345ff91c4d8980b2b2ca0", "309d24f05b1b4bc5b7d0148795fd5cf4", "ccfb5dc3eef14f1f8324bc48b3dc0e7f", "a84e5372c5c24417a08487dd3efcebe8", "c28c07424b7f45088e73ac25f2bb712a", "5d01bda6f6354f25bb6dcb15ef4596a0", "758cba83e9414f60bf87799a71b3cd7f", "48d71d4847be4a25a237568371ae4493"]} id="ENZSOd1DgzLk" outputId="7b1844fb-b81a-43cb-8a98-5206b87b4506"
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc], val_interactions, model)

print(f'MAP@10 Score: {mapk_score}')
print(f'MRR Score:    {mrr_score}')
print(f'AUC Score:    {auc_score}')
```

<!-- #region id="blXcVpg3gzLk" -->
Again, we're not seeing as much performance increase here compared to the standard model because MovieLens 100K has so few items. For a more dramatic difference, try training this model on a larger dataset, such as MovieLens 10M, adjusting the architecture-specific hyperparameters, or train longer. 
<!-- #endregion -->

<!-- #region id="1Dt0IsWJgzLk" -->
## Item-Item Similarity 
<!-- #endregion -->

<!-- #region id="sqQJpyKVgzLl" -->
While we've trained every model thus far to work for member-item recommendations (given a *member*, recommend *items* - think of this best as "Personalized recommendations for you"), we also have access to item-item recommendations for free (given a seed *item*, recommend similar *items* - think of this more like "People who interacted with this item also interacted with..."). 

With Collie, accessing this is simple! 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 343} id="RRMouFweUOhw" outputId="02e6281a-f64e-43c4-a151-84601d91ab50"
df_item = data_object.load_items()
df_item.head()
```

```python id="Ycl8PcLIgzLl" colab={"base_uri": "https://localhost:8080/", "height": 343} outputId="27f2b6aa-9b6f-4fe4-9d8f-3637cadd4bbe"
df_item = le(df_item, col='ITEMID', maps=imap)
df_item.head()
```

```python id="TvtbOrbtgzLl" colab={"base_uri": "https://localhost:8080/", "height": 117} outputId="67ef757a-91e9-4b32-c2ac-5f43c4742634"
df_item.loc[df_item['TITLE'] == 'GoldenEye (1995)']
```

```python id="Uom6EVG8gzLm" colab={"base_uri": "https://localhost:8080/"} outputId="a4f6c8a4-c472-48c2-f240-14763b5d1d8e"
# let's start by finding movies similar to GoldenEye (1995)
item_similarities = model.item_item_similarity(item_id=160)

item_similarities
```

```python id="MW741iOcgzLm" colab={"base_uri": "https://localhost:8080/", "height": 428} outputId="ad4b1617-bc85-4698-ebf6-65b70161e7ce"
df_item.iloc[item_similarities.index][:5]
```

<!-- #region id="_py39oQ8gzLm" -->
Unfortunately, not seen these movies. Can't say if these are relevant.

``item_item_similarity`` method is available in all Collie models, not just ``MatrixFactorizationModel``! 

Next, we will incorporate item metadata into recommendations for even better results.
<!-- #endregion -->

<!-- #region id="8hkoWyfVg9AK" -->
## Partial Credit Loss
Most of the time, we don't *only* have user-item interactions, but also side-data about our items that we are recommending. These next two notebooks will focus on incorporating this into the model training process. 

In this notebook, we're going to add a new component to our loss function - "partial credit". Specifically, we're going to use the genre information to give our model "partial credit" for predicting that a user would like a movie that they haven't interacted with, but is in the same genre as one that they liked. The goal is to help our model learn faster from these similarities. 
<!-- #endregion -->

<!-- #region id="4iFhjr7eg9AK" -->
### Read in Data
<!-- #endregion -->

<!-- #region id="bK4bGSUEWe9F" -->
To do the partial credit calculation, we need this data in a slightly different form. Instead of the one-hot-encoded version above, we're going to make a ``1 x n_items`` tensor with a number representing the first genre associated with the film, for simplicity. Note that with Collie, we could instead make a metadata tensor for each genre
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="mi7IZRHbWwsc" outputId="20906809-3682-4bee-eee3-1d98fe9b03ca"
df_item.columns[5:]
```

```python id="bWBxAUUXYvZ3"
metadata_df = df_item[df_item.columns[5:]]
```

```python id="LSy_-Jsxg9AL" colab={"base_uri": "https://localhost:8080/"} outputId="a1d9aa32-d10c-454a-dc7b-9cfcda410071"
genres = (
    torch.tensor(metadata_df.values)
    .topk(1)
    .indices
    .view(-1)
)

genres
```

<!-- #region id="LhaQLQQig9AM" -->
### Train a model with our new loss
<!-- #endregion -->

<!-- #region id="5NrYwTFVXCco" -->
now, we will pass in ``metadata_for_loss`` and ``metadata_for_loss_weights`` into the model ``metadata_for_loss`` should have a tensor containing the integer representations for metadata we created above for every item ID in our dataset ``metadata_for_loss_weights`` should have the weights for each of the keys in ``metadata_for_loss``
<!-- #endregion -->

```python id="Sysr04kSg9AN"
model = MatrixFactorizationModel(
    train=train_interactions,
    val=val_interactions,
    embedding_dim=10,
    lr=1e-2,
    metadata_for_loss={'genre': genres},
    metadata_for_loss_weights={'genre': 0.4},
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 472, "referenced_widgets": ["7f9d29d6c1f04d4b9adcda6292a698fb", "4d95aa15fb8e45168224c12f033db89e", "567edd5b84854cc3ba9d8b34702716ce", "3ccc6ebee63d473aba28adc868b5b5c0", "5db3013f6fca47428b7f05d89d611441", "650154b034e843648fdd31198e6b5c8b", "b85c146244634db5a8ec5bb8c37a6a3f", "71e7704f60a44170988aed05dc8a4788", "71fb17475171409ab7b30f0915d3c3f9", "ec82867f02de401faf33bdeb69a1bf2e", "4ffc6e6cdbbb47239b7da5a07f16b5fb", "cce5557f779742e3932bb8ef2016a17c", "668011e6db324325974aa6a6b1fdc782", "ebcdd1df171847a38f7655716f2d6061", "9d11e361b37a4849a83531e249455615", "30fefc2060004b2bb028ed54ea9383b6", "85b96bc221984d08815cf3c09fde7c9f", "1abf2c736a91486eb8621d1cd679f9b2", "89b238d17aad4ef389c20de16ed1e55c", "70d5a3c3cf41478b83813ec40298cd22", "bdf376ea2f8e4b9e9d495a1fe44b2b59", "c6213cfdba3d4228aad387361e062fc7", "a28aa408fe444c179080e2ef9433a877", "6e3313408a184dcca85da1f7514d8bfe", "b6f3004c83574b379bc3520a410b5df5", "57e0d47b70b744898e4e548c28d27095", "d5f9ad23ba314382a2ffa29ecc311b5c", "75bcb22fb4514a288fdd43cdb2084b11", "d7bf1b026d10490fa70221af127986e5", "bef2ed39c203462a80a1f217adfc301b", "1038bcbdb10a4da08c7b2ffbc08c7e81", "2a435432b9914e3faf2d49a0645a9fd5", "127ad96af11c41519ded336bbc246c42", "b5fe20560cc64837af9e13936fd702d3", "2260ec6d57724ecca381bf65a151e97f", "add9b2d1bfeb4b3baaa97d39ca4eb8e3", "2baf39d33c2a4f28bfd557ba7b9f8c78", "bdf0ea6c2c8042d68c50cd5e19b1c15b", "143ecbd32aa14efeb1d2eae6873609ce", "b4eee80b60f748b4886ede6073b37a7c", "1997f9f7a4d04e5ebec4b8907e32b797", "21380c3a63df4a7592b0e580535603df", "0bd610aa3a55452dae1e3fbc721cb87a", "07e2308d6aff45c5bd0ad77e246e9981", "2354778e0ce6427385e2e107ea30332b", "717568c66bc444ddb60b415c5169588c", "13a5b3039ac04a2dbd6049992f822c89", "96773fc1d61d4b9faf8f8e7f0d0c3786", "2bb338780a1a40318d91fc689692c9ae", "869cd72009244cd997fc6b8bca19cd53", "8f341ddecce34a2e9967972e901123b6", "4f53235cc013444c8a71141df07ccc4b", "d76f878252a54a79bd98b1e093285964", "69b2d3fe2f5d4f73ba34142ccea3dbf8", "818427165e3c40d6b9c60f5fc26ea0db", "a62b8b75b5bb4b1a9a35a98b9f87edc0", "b066cece2abe4a74b1aed6c238cd82b3", "5a5b267987ed4ad0a143744fe30f27b1", "2666a159d7f1406887a6bff3efdd1dd6", "7b3b88973bc74728a3ea7eb04a122371", "7e684d1ba34a49e1b108909220e0a487", "a96a161717124b54ba9473f5d4476177", "2137633428de4deb8056902144e7551d", "2454cc03bb5342eca1ab02501d3ac39e", "2da2765da24841709884a7cca16f1107", "9d9f37ef8c9844a9ad5f4022ce904af3", "b7febc3da8714283b798aafaab3a8d8f", "f44ea695c0b2485faff83a6bbe12407e", "1e1888b6dc3e4c47aadf843a62f03233", "d75a669ab0a147ddb4185d6950c8267e", "a1eab9ad9a6147c2a3a13ed1d837f94b", "dd40e90d90a34827bb07d9d86c4234ce", "2391e70d0de74d71ab5372f39a4fa49b", "05fa6eea049e4b41a40604eed4c798d0", "e271494364c24f4e901c42b64e464056", "69b72f3ef4ae40a0a7b060be0849a354", "3e756777c21b4c56a06b537ca232ffc3", "0d4e9b5e9a3241dd920146b8be70d1ce", "d6e7bf43dc2346c0a2278d6b19078c90", "852593a171a048bc81b0cd3466366bd0", "13c2754aca2a42c2ac06d3640062b6d4", "1ac7aa00d4b24eccaf9b0cb622a65b05", "50691a81b511451aa9f78cbd0803a652", "43a743d7e9f14787a606cea5d0176931", "8561cb62512f47d9929c3abd557ef739", "52ac26d1722c40b68b18e0d92c6ac994", "cbf2e61a7dea4b91a8f86774bcfc8ff7", "8e58b5a80dca4cca8b03635c8d529072", "9adf92080dd2458a942dbd76a285ce48", "9a6a243411e84f049dfaeb99cccbc562", "5458638e910744509182e8c4ace883ec", "e692df9e6fb949c6a345dbaf9235d195", "7409891326ad4a259842211e306e980a", "0f57dd28b6954df693a160125075b506", "ae151ed064b547b6a8bca5181668f491", "5e8575da45094823bce0d16b9fe01f09"]} id="ZAk1C815g9AN" outputId="0c5600e3-d81d-4f6e-d621-c7882a767fb1"
trainer = CollieTrainer(model=model, max_epochs=10, deterministic=True)

trainer.fit(model)
```

<!-- #region id="NQdGTCPvg9AN" -->
### Evaluate the Model 
<!-- #endregion -->

<!-- #region id="ISQzTUnVg9AO" -->
Again, we'll evaluate the model and look at some particular users' recommendations to get a sense of what these recommendations look like using a partial credit loss function during model training. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["b1c126acba99422c9aad4bc9b1b0293f", "3e7789b77e3342acabd132d924906d5e", "a9db4a7b06d34e2eaee4933580e8e28e", "2e28af584886415c869f079215546e5e", "e078e83f8216456997e974acfc7f4816", "12b2182ba29547a8845685618988224e", "6d1261287d6042b486877c29f1056305", "6715a50cf1074e98b4e8e8f9aca4932e"]} id="6STWH4Ozg9AO" outputId="861755ec-75be-40fe-975b-17accf21477f"
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc], val_interactions, model)

print(f'MAP@10 Score: {mapk_score}')
print(f'MRR Score:    {mrr_score}')
print(f'AUC Score:    {auc_score}')
```

<!-- #region id="Bk75mVQWg9AP" -->
Broken record alert: we're not seeing as much performance increase here compared to the standard model because MovieLens 100K has so few items. For a more dramatic difference, try training this model on a larger dataset, such as MovieLens 10M, adjusting the architecture-specific hyperparameters, or train longer. 
<!-- #endregion -->

<!-- #region id="9X25yfucbbKx" -->
### Inference
<!-- #endregion -->

```python id="dB6eeXWfg9AP" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="805ffd4f-38fd-4f4b-d9d1-b9ddfbdda60c"
user_id = np.random.randint(10, train_interactions.num_users)

display(
    HTML(
        get_recommendation_visualizations(
            model=model,
            user_id=user_id,
            filter_films=True,
            shuffle=True,
            detailed=True,
        )
    )
)
```

<!-- #region id="ZoIh0oHfg9AQ" -->
Partial credit loss is useful when we want an easy way to boost performance of any implicit model architecture, hybrid or not. When tuned properly, partial credit loss more fairly penalizes the model for more egregious mistakes and relaxes the loss applied when items are more similar. 

Of course, the loss function isn't the only place we can incorporate this metadata - we can also directly use this in the model (and even use a hybrid model combined with partial credit loss). Next, we will train a hybrid Collie model! 
<!-- #endregion -->

<!-- #region id="DZpMvb3-bidk" -->
## Hybrid Factorization Model
<!-- #endregion -->

<!-- #region id="Laxa0vh1hE3o" -->
### Train a ``MatrixFactorizationModel`` 
<!-- #endregion -->

<!-- #region id="Fj3tJg-1hE3o" -->
The first step towards training a Collie Hybrid model is to train a regular ``MatrixFactorizationModel`` to generate rich user and item embeddings. We'll use these embeddings in a ``HybridPretrainedModel`` a bit later. 
<!-- #endregion -->

```python id="m75xWkQLhE3o"
model = MatrixFactorizationModel(
    train=train_interactions,
    val=val_interactions,
    embedding_dim=30,
    lr=1e-2,
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 438, "referenced_widgets": ["62986af6e0394e969bb8942274ba6784", "4e51bc72f9fe41ca82096d01e26a5c23", "a3b4a6f432f7406b97c59bdd31eec848", "d6b7dc0b58874a93927b9182fe92ae63", "2528199682404110adb35e51bf1c39ff", "16dc9e31d8bb483faa575fd6db3915d7", "d9b41c7acb1741d98f5edf4e942e3b77", "7aabb759b56a4fc295687e0a24b6cb62", "b1ccac573c5c4c79b4357c14f99a08b6", "1c45f04d9cdc4876969a8a85d9087f18", "84065182993c4cdcbc3e5a37c769447b", "13ef2e644cae4a48ad8ab142e8b6435e", "191a266c1f034c9fa6e7b4138805d60c", "b8f64ba5faab4995a8b6f676b5a8e5d5", "22cb3e6197d34688914565f07d26ecef", "cc2768fdc75942c4a894bcf6006ad11f", "952b7b171ddb4f4eb96cc91f1257ca9a", "660daa560af241f2911156e2b7187c7e", "847fd69ae9924b03929b1a0e59e80f87", "e1cfc5099d1b4cc7bedcc7a53da34453", "9b62c6c326784d799f790398e7fa22e9", "65115b3c64504cc09ed3459b778995bf", "069b286ddde94f77a425f288e6e14d09", "f051d817821744d78a30577f69178e1e", "c0632ad487cf400e906da9a52efa239a", "12183c2f615f435a9cd920100f2820d4", "839eb1025574493ca1cf19f5a2f2010e", "ca5087b11f79495e8d32cfe8cca64f7b", "949ae8d78ed644cebf9b8fd86a6be7fc", "38f2d3b8e5ad41d1af054ef4c2ea47be", "195cc632420d4e85aadf55741dafe166", "e6ee962e992e44cca46b92e93b9f2c37", "b06e9ed8cacb428ca888b8c7e0558e54", "77d14fa2a785407181e4a9da9d40a3d0", "fd4cec7630304a159031e309fdbd715c", "cffb8b1a826b429488fc7d6af390df80", "c58a9370d7484bf8bf50c8e1fef3da12", "8d8c48b368c943aca716058d8a6c7485", "4ff6be2f8fe8422fa6ed67aa4e4bb2cc", "741722e693c344cca086aacee15eb874", "d227738e76a3420ba0a79a6684de0dcf", "0057a0667a434d1e8ac207b121049b2c", "c6316fd570b741628ea906f553c52679", "845c79c672984d0bab35df090dc632e5", "52d654208b6b4001a56c287a822354d3", "c47d7500fff14be39a2184e2d30b3795", "f6273966027146e9b345c02f935e65be", "72d633d88bd6413cad4cd3d2717dc7ca", "b423c474bb5d4ba98c14502c6b02d95d", "75241cec097b4fba877099eede78e3e7", "2cef863fdeb0413a8a3ce9bc68d1f985", "be5f89ff1dbc41b2a8263817d4a59f1f", "9459414718e44c6a9b88760f1fb1b46a", "1c8e2a3fde7942e697e111b07aefe26b", "637168904e464ac9a6782ada14784eb1", "a069be3b5a6d46e083524d14053bcb9f", "71907183c5584961bfd232d68e685d3d", "9e35548cd1714a0c9bf3ccc52d268fb1", "5a131f872f8d4fcb96b713b60e56656b", "7e72190f3fd44d1489badb8d28b6bb29", "82dc472697914b1aaf5866e1573c662c", "dd1af81809254587aa2a3d0deacff0a0", "b61d4544c6a54060a2dc120eee5362f3", "3b096679a00843148f3b874b9c70e901", "2e2eb1edad7748ee92249d555840d612", "8b2752894a8b4a69af0ad6be380c8c23", "09a7ea71de884bcc94aa330697ab728b", "29c1f761f5634be2bf0ff25736fd7d61", "be4983b296534b4ba8991262f81d6c0a", "058111735ae7416fab7f7adb8c981d14", "1e7ed32472134c508325266efe272c40", "d46475c0add24876b714a409bc9933b2", "51757d5291844c6981c0258a245ad5be", "36090e9a0cc94f56a377ff368eca2b64", "cb77a4c364704cdca658427246c1a3fb", "921e0906b18841d4b68b0b72dc911121", "86439582332e43cea3d1eea6062510f2", "edb0b8167858489c8f94c95337647d9f", "92b45c3467f44cdfb4b12fc7d9f25f5a", "1b13814df36f42969c8795c40b351ccf", "d1a11fbd7c3a4dadbdee14b744937c30", "248753435c404995812b93422ca8c062", "2b3ae44914934e31ab2229c3b7d1b9a7", "08c6b30a70024da28e546ff11dce4f94", "ce736cb44cec467e8d6db28528e30780", "d0ea236e247c46c78d6ee489433b76cf", "33e822baba3b4f0bbec237b48deba7f4", "fb8e91ab0228428487907e96827a91b9", "c7fec0dc1e8648ca869c65caefa05751", "acaa2bdd5eef41a49a99c02f384a8173", "2a62b154d02740988eaf6df9ce39cd93", "369e4fbc8a1b4c6d91f502a7fe5b18f4", "f2ff83d1d99d40c7bb9d563ad623e164", "e359213b91d841e3a28cdc293668a104", "86a3bee8f0604c9a922c94e4161fc24b", "f260359ad48e4f4c99e7aec1fc5cd09f"]} id="bjh0jE0yhE3p" outputId="34e09142-f66f-4d32-9052-5e41d3e7d166"
trainer = CollieTrainer(model=model, max_epochs=10, deterministic=True)

trainer.fit(model)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["9b69ca9af02e4dcead166064746dfcb2", "1748fb86e5824e739a39350e4777a4eb", "27157458047e4a769ae205d23b8e6221", "2c7ac2735ed649cfafc52ca48eb38d1a", "ed61762550634ce6a906b6da4550f4c7", "b1ab812f3fae427db2b8cef8fe572f83", "08103e370a6047068d6e7d54653c1f3a", "1d603bf9ea684001b97a63be7f446e3b"]} id="6tvE66cfhE3p" outputId="1424a753-c468-48c2-dfc9-3b2118195955"
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc], val_interactions, model)

print(f'Standard MAP@10 Score: {mapk_score}')
print(f'Standard MRR Score:    {mrr_score}')
print(f'Standard AUC Score:    {auc_score}')
```

<!-- #region id="CRq29RVfhE3q" -->
### Train a ``HybridPretrainedModel`` 
<!-- #endregion -->

<!-- #region id="lFh1LEcChE3q" -->
With our trained ``model`` above, we can now use these embeddings and additional side data directly in a hybrid model. The architecture essentially takes our user embedding, item embedding, and item metadata for each user-item interaction, concatenates them, and sends it through a simple feedforward network to output a recommendation score. 

We can initially freeze the user and item embeddings from our previously-trained ``model``, train for a few epochs only optimizing our newly-added linear layers, and then train a model with everything unfrozen at a lower learning rate. We will show this process below. 
<!-- #endregion -->

```python id="27SwhWuMYNkQ"
HybridPretrainedModel??
```

```python id="RPgUTdR1hE3r"
# we will apply a linear layer to the metadata with ``metadata_layers_dims`` and
# a linear layer to the combined embeddings and metadata data with ``combined_layers_dims``
hybrid_model = HybridPretrainedModel(
    train=train_interactions,
    val=val_interactions,
    item_metadata=metadata_df,
    trained_model=model,
    metadata_layers_dims=[8],
    combined_layers_dims=[16],
    lr=1e-2,
    freeze_embeddings=True,
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 438, "referenced_widgets": ["1b0f046d63ec46909920000d416a956b", "2010b6b4500e4b139e93077008383a3c", "3a31cc9116d7465bbeea6471c6e93968", "560c2c1f4ca94b73a47aeb344ee81ab1", "cf1581e1b6614caa92b7845465c9de39", "346d17d614ee426fbe2be794a34737f3", "52271d9aa0d64d0dacb8d2d0d827adc0", "eb62d6f5333a4cad90fe00ecab4f4de3", "73d516b143a94e0a9ff2115a975dfe0b", "cabcae1934154c03a2ddbfa47b2b4501", "726b226d57474815ac786d93f4039902", "1a2825490a2043aca84c13d316d11ed2", "293790e04a5a44bb9f14edf7ac0ab8cc", "5ccacb85b1214ee1a0e5665364c235b7", "fb3d481dfea04ea3b6d1f8d57682826c", "044c05d2a4824890a595a2f196d64459", "ad5390af70b14da0aa0d3a0b5d20a90a", "0f93d0dc68984a279b6382fbedda6630", "e20d6099034e4567a726f6a7b038395c", "76f0fc24f91d40e4869be6e1b5b4b0d3", "a0666c429914487faf1fac59402e7364", "d6c7474eb9114cde9ff9adde036f3bc2", "83e1559c03c543c88418b270c16c88f6", "01c61537bfe842649a60c70fde9a51a3", "32b3c6d930f94d56b9f6e652303cc1ca", "27b7e22b72b94b9da9d22190c338ac84", "e39a8caba8854f5696d3ff03277bfaa0", "ad367116739e4f4c8717c5b5b741b2ee", "916dfdca43f74c058da5b45dff1eb621", "564e25c0cc9d4f8bb9a00e8f480eaf7b", "0cfb4e7dc348480d9cae85167df7a73e", "24f32e83468d4a4eb03b91130a526df5", "cf3dcff653a3443184d0942defe61230", "7f75f89935e84173bf5d914bc8f50150", "8c86d4c10827436b90b7f62885c57ef6", "3971fc6570fc4f3ca85df75030b1a984", "b5be05a8a7c045c1bce4eda3852a9256", "5802083d9af4473684eadd22aa17d5c0", "00a30c0b9eac4a38b0bd3be2cc05476b", "dc0817dca939450caa82a751377451d4", "aa23f91202f7432ab937a60fd0cc69c2", "fa12cd0edffc4459aadb255b2f85309c", "cfa25ee4254c479498e117189651d8c4", "7e6c87609e254706b7e6ee6ac7bf78b4", "a50062a20b2a475fb760c5ddf6266197", "82c055fff9914a1d8a6c838062046b99", "c16130a0a9624f848af1b5553264b78d", "1f2cb0a8875946439822315ad3a7e1e5", "22fbff68d90c43d6960fea7ddd420810", "6f02c441db1a45488e41b29c656de74e", "1408cb730c964bd39fd32f56916b9701", "095d3de7d1e1464eb81d927e6edc3b0c", "e5379796148347bc8f6ab506e1f5b1cd", "bb09a545620c4dd1af632339d6fdd04a", "08ee77ef97744d6d92a86674b9247b25", "ef2fb16bb26f4a489fe456a33a9aaabf", "c4588fb79e384aeabc04a175fe8d6548", "d519f414b012498bafdd6ebd8503bbf0", "a571a202b33d4a0dae7428d244d64cf6", "f8841a4feab24ea1ace015833bbf0891", "e0eb2bb0242e44aba710350121762fad", "2cbdcb710a39477494596ee8e941152a", "7ef288048daf401699f47fcf24ae2b2a", "7927e1af2f644102b87c7c13ad22546a", "c560a29789ca48129d338f88feddfeee", "a2dc9ff9d3b6413489e37645e5a81969", "e23fffe8236f4f9ab8b76858b371ed8c", "7a1cdaf584e44e3392d963cd39f65cc0", "0303b557637e45078eb791c8e6091f61", "7025cdb08e7646169e13713fc442b5e3", "c3216b05839d47d781220cc4dc762e46", "41b780dbe1394396a24bb180a7945c3d", "c58dc7499eef43f39a406d873aa9bf2f", "3a7fe9e14ff94e14a3ea9be6a3685299", "65d8dc2103ba4794a94ddbf4e1ef9825", "16a6b8dc6f36458da47ce830fac15d26", "5be8f16d753c47a6a68248c561a9605e", "c85c795ddd734662b1277c3b4dd4bb6f", "37aacb2f901c46c4bc7b9c580890a3c6", "f837d58642d247cf9ffb657002d83ae6", "109fb55f40334538966bbdb2961cfdf9", "1a1fa94f37524d4cb93f5a2bfde53b12", "7978b82dcd874c2782f4821d859ef918", "e4f3bb71ec4645649a3e81bfd8b3e310", "0374e23a55c249bb955a605926214583", "14acab5f63754fd0970ca0a6e204d47e", "720887ecd42045539a4f03a57b8a2d5e", "94f0ec5f8e5e43de8397102e899dde43", "fcbf9f9aecf54c1f97c3e2e3915c36c2", "2b71910b6bf74406a468ef78e914db9f", "7bfe5a6f9f2f4787acb15eedd6e23341", "365f4e59afcb443f8c3cad99137dcd40", "d9068f85980247bc89628635dd221f3c", "c98453288d7e416c8ae6067db3220b5f", "6657854d04e446339437228998a63325", "27a1d1d028474aab8ee3cdc1c00166ad"]} id="vyyUg5ilhE3r" outputId="77ffdf06-0964-4a7d-f491-a83d53d3e210"
hybrid_trainer = CollieTrainer(model=hybrid_model, max_epochs=10, deterministic=True)

hybrid_trainer.fit(hybrid_model)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["f3842767c3b9491ca0e39c7d83c3bba7", "42e789d0121c43df87d03eeb4e58d3fe", "bdebaa528fe345f58ed0e85d0015187b", "08455bd7b312416a8828902e10a0a6de", "4a1d091431614ff2a9a94baa79f4ebdd", "78f224d19e054aa3a6787b9c3aa536eb", "b6cff5f6b6004d6999a277ef1ed64fe1", "9c1fd029256144a68b8c996e201c3e08"]} id="I8eEYwcfhE3s" outputId="9babb089-060c-4636-f3dd-ebbf91d939e6"
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc], val_interactions, hybrid_model)

print(f'Hybrid MAP@10 Score: {mapk_score}')
print(f'Hybrid MRR Score:    {mrr_score}')
print(f'Hybrid AUC Score:    {auc_score}')
```

```python id="EEw83cTUhE3s"
hybrid_model_unfrozen = HybridPretrainedModel(
    train=train_interactions,
    val=val_interactions,
    item_metadata=metadata_df,
    trained_model=model,
    metadata_layers_dims=[8],
    combined_layers_dims=[16],
    lr=1e-4,
    freeze_embeddings=False,
)

hybrid_model.unfreeze_embeddings()
hybrid_model_unfrozen.load_from_hybrid_model(hybrid_model)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 472, "referenced_widgets": ["b776627648254a919b1a18b4025fbcb3", "2b4a275a432241ec91b5fe2a67fff267", "b9fed3cbac434c5e8e8eccc37fdc7909", "518844d0b3cc4deabe8b0ebb657eac8f", "20a1726ffbf9408b8a00ccd5c44b81a8", "f9e3ccff7f41406eb5ec4f9658b24718", "d354c4d6b73d412f87a41f8cbb7bb9eb", "01f11efae5cc4bc48726f6554c346655", "bede57139cc14b599167088250548b43", "512af7cae0c84552b736431802ed4402", "2a5f13e1cce34787ba032800b8a59ef6", "2dd137e524324334b5f7c56e7d4d8877", "e7f0ad3f8d204c1a9c611a897655abbe", "dbaaabf67d1640d5b988ca6e095577c2", "9814e7c9d49245e7802828d385800a6d", "7efc774340274141a6a6d4c3e5bfcf12", "fe7f069fe1f94fe3a98c44c3cf5c3ed1", "82de05cc4d4f4b32bf7127007ba17bef", "3e5ca461bd1f4d6ba6e3f5d2a5c708b4", "4d565f9c62d84ba487c79a1e2053770d", "7beb2d7bb4b8421fa6a3fdc5a0a578b1", "b56b52a8d05c4c2496e18f32df63ee5b", "55a7ecf02b1f492e82b95e5460e04f22", "7d93dfd6d22843108d76c8a2416bee21", "3b0d7093adde4f64bfddffdb4444e1f2", "aa9c6996aa3c40d18f0b3f0b8cd1705d", "7dc6a6399f604fef8dda1b5a1d7b2920", "c1ab4cda390d4a99b92162421e86718a", "c894ed9774704d88bff3e9d1ad542900", "4968d6ba2303488c9256042f3a7f8206", "27da441d4d55492ebc526ca00dd7d01e", "ad048e4075c847f1b911080e51548cdf", "84cd0ec866694431a1bc3f6eb7686107", "8b4924ef271d4097abd6e57303794327", "200eca6c62424921ab682d2ab8a0785d", "80117a933a694fd9914f88917466a00f", "47c26c18cccf44f4b6a5caf3ccdd4e83", "4fdf8e0a36244ff1bdf34e9060e3f035", "7478d08325084ef28fa9f1c5a6ab18f6", "1b0c6cff674c4930b18748d9fa4f9090", "73232374fc7f4880a87dec552959d3a4", "03dad58a3f004920aff305a2357ad121", "fe02f58536de4d49b132a067fc065671", "7805d39a07414d199a012bad80e90acb", "094a99863ecc425dae15237f990ffb3e", "b91817bbe07449b2a9a8d1b6be2ce378", "82c6bf962cd04ee8bf02d24b03952b28", "3e392b5c457a4ef2b7c883982f60c36b", "5ebee37d75004546b316d10399b69431", "b7fbdb82ca614754a12635170518e0cb", "6cb3f43666734c869fcc960645e129e9", "ad4885e1e6ab41f48ac113c30aa13b39", "87093d6b3c924e6087e9d2b78ba0c6eb", "0a4702427f524e768b8d1b2379e65499", "3c9b57c8d7e24cb181e50d3b5e9a89d4", "1f6a03090e2f4bd5a08b973a2e31a48e", "43ff46cfff674d37a04bf926feca9048", "7fb80bf0ab9549989de36323648126cd", "34c2e6b19152468e8e8bbdcbf1e7d87e", "521b635587644d588e28f0efba61aea2", "6408b9869970482d949d6a794700716b", "61a742fb4fde49fb9e4026e330cf2159", "05bdc0f323064d359a32a0b8d345dd78", "424e11f8cff442ee8a7643e56ffb36af", "2ac6cf2e1f304f06bdc354d04507fecd", "44ac79c4dfec4caa9bd2e4f987aeca9d", "ebe6a36807834fb38ff46654e27075c1", "ac257e025a9545d5b924d2946b422735", "4606964ad6b5447db1d7498178cb5a78", "e823a52deb434d5f81deb90ae34adc5e", "9dc20f131e8642ddae5a90537309d835", "44eea8a841af4ddb92d93bfe06c4c6fc", "5a147c6e4b59428ebd7e2e3412cc52fc", "49bf717484f84a25a7ac27b01b2606a3", "c91bb1aa56074b398362c4326c9b9b13", "1aad34c1efeb4af6a9ec6809a12a3569", "7214617bec6645ba891a2071f6bc6442", "ab827148c2884c728fe50dd09ce55912", "d48633090c514ddd9e9fc2baa4bf347d", "6315ef0a9eb14469a4de8063b9e745ca", "afe817a0babe466cbbf8e4b802ef3360", "60f26b29709e462381c221393c45e76b", "cd7d3ce3534447b98fafe4d5aab0194b", "f44f6872eb08434ea56d62708770cd25", "cb38f538344b43928074a1184cd05997", "286f2daf07f745d891110f78c116823c", "c456a6bba5804e8e911a89abf34e5670", "b4b847b559944ce2ba7a4ce34c472120", "975f803a94994f91891c3fb7f187a135", "47d4f8168a3049b09471968aca76fa08", "0af18a7ade304878bdcb8dbca2ef1074", "a521fbb49c5946b1afca37cbc4052b52", "dac2cbf4f3f2477ab18adce6db8e77a8", "5e9949db3b97478a905b23c0a437dd45", "544a63b30c714580be37d69eb8669328", "33fbfb7b8c2d4a5bb0e979c626862b13"]} id="yiA-EylqhE3t" outputId="aefcb665-c88d-43ab-eb7a-322cbc58a262"
hybrid_trainer_unfrozen = CollieTrainer(model=hybrid_model_unfrozen, max_epochs=10, deterministic=True)

hybrid_trainer_unfrozen.fit(hybrid_model_unfrozen)
```

<!-- #region id="2Txbqf3fbqvD" -->
### Evaluate the Model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 117, "referenced_widgets": ["0d9f2c4528634f63a26527a755b33773", "6d196fda01ee4b63a7731b169c2f36b7", "20d9caea7c744c09a8efd05793d2c6db", "1760eeca2a084b1c8e57a9989139cdf6", "8ccaa38836fd4fa588723ba2516f635b", "113a8ae2c462494abf3ca97f65e51d06", "c0282a49c81f4dce83322f9734eb0efd", "9848dcb0b0e048c88f9c6243dc5ede33"]} id="sof4rqMbhE3u" outputId="a344f1bb-0627-4e46-968e-f89bc81a5224"
mapk_score, mrr_score, auc_score = evaluate_in_batches([mapk, mrr, auc],
                                                       val_interactions,
                                                       hybrid_model_unfrozen)

print(f'Hybrid Unfrozen MAP@10 Score: {mapk_score}')
print(f'Hybrid Unfrozen MRR Score:    {mrr_score}')
print(f'Hybrid Unfrozen AUC Score:    {auc_score}')
```

<!-- #region id="0FzTQc6WbtJA" -->
### Inference
<!-- #endregion -->

```python id="EwM1pkf_hE3v" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="650e4d07-8566-484c-c7b7-543e7c7428a9"
user_id = np.random.randint(10, train_interactions.num_users)

display(
    HTML(
        get_recommendation_visualizations(
            model=hybrid_model_unfrozen,
            user_id=user_id,
            filter_films=True,
            shuffle=True,
            detailed=True,
        )
    )
)
```

<!-- #region id="fNwj-u-AhE3w" -->
The metrics and results look great, and we should only see a larger difference compared to a standard model as our data becomes more nuanced and complex (such as with MovieLens 10M data). 

If we're happy with this model, we can go ahead and save it for later! 
<!-- #endregion -->

<!-- #region id="xYmFQZEhhE3w" -->
### Save and Load a Hybrid Model 
<!-- #endregion -->

```python id="2ZDlfmAVhE3w"
# we can save the model with...
os.makedirs('models', exist_ok=True)
hybrid_model_unfrozen.save_model('models/hybrid_model_unfrozen')
```

```python id="qW3kPpenhE3x" colab={"base_uri": "https://localhost:8080/"} outputId="da7c6d72-d7ca-4913-98fc-e85691a771f4"
# ... and if we wanted to load that model back in, we can do that easily...
hybrid_model_loaded_in = HybridPretrainedModel(load_model_path='models/hybrid_model_unfrozen')


hybrid_model_loaded_in
```

<!-- #region id="1rDDCMkehE3x" -->
Thus far, we keep our focus only on the implicit feedback based matrix factorization model on small movielens dataset. In future, we will be expanding this MVP in the following directions:
1. Large scale industrial datasets - Yoochoose, Trivago
2. Other available models in [this](https://github.com/ShopRunner/collie_recs/tree/main/collie_recs/model) repo
3. Really liked the poster carousel. Put it in dash/streamlit app.
<!-- #endregion -->
