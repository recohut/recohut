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

<!-- #region id="NK_4bvxlXaML" -->
# News Recommender
> Finding best news to recommender using NCF model, training in pytorch lightening on MIND news dataset

- toc: true
- badges: true
- comments: true
- categories: [NCF, News, PytorchLightning, Pytorch, Visualization, Wordcloud, NLP, Transformers]
- image:
<!-- #endregion -->

<!-- #region id="Vq8TD4AsWoiX" -->
## Setup
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ebTmBDSV1006" outputId="c873071a-9dd7-4700-c727-f04c5f20fb76"
#hide
!pip install -q sentence-transformers
!pip install -q pytorch-lightning
!pip install -q scikit-plot
!pip install -q wordcloud
```

```python id="CQHvdZCW2ZGF"
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import scikitplot as skplt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)
%reload_ext tensorboard
%config InlineBackend.figure_format = 'retina'

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-notebook')
```

```python colab={"base_uri": "https://localhost:8080/"} id="I2DTPtmnUIdS" outputId="a5fe6a05-ad9d-463e-ccdf-20e5f1e8611e"
!pip install -q watermark
%reload_ext watermark
%watermark -m -iv
```

<!-- #region id="luwdf-m5Wqjy" -->
## Data Loading
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="P4un071DQTQM" outputId="49487e34-a173-4bbf-8544-53bf5b389a2d"
#hide
!wget https://github.com/sparsh-ai/reco-data/raw/newsdata/newsdata/news_text.parquet.gzip
!wget https://github.com/sparsh-ai/reco-data/raw/newsdata/newsdata/user_news_clicks.parquet.gzip
```

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="Vu3anGm_VUrT" outputId="fbd0f778-55b9-417e-c08a-88f82fb2a593"
news_df = pd.read_parquet("news_text.parquet.gzip")
news_df['title'] = news_df['title'].fillna("")
news_df['title'] = news_df['title'].str.lower()
news_df['title'] = news_df.apply(lambda z: z.get("title", "")+"." if z.get("title") and z.get("title", "")[-1] not in string.punctuation else z.get("title"), axis=1)
news_df['abstract'] = news_df['abstract'].fillna("")
news_df['abstract'] = news_df['abstract'].str.lower()
news_df['text'] = news_df.apply(lambda z: z.get("title", "")+ " " + z.get("abstract", ""), axis=1)
news_df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="vFok8U3JjMzK" outputId="ea7eaf49-5f0b-4c1d-9007-c4892fb25888"
news_df.info()
```

```python id="i9lrf_AEGGbG" colab={"base_uri": "https://localhost:8080/"} outputId="4c06ad21-4a91-4262-de53-790c9c294762"
print("unique items:", len(news_df.news_id.unique()))
print("unique categories:", len(news_df.category.unique()))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 560} id="_ClKc4IHDxN-" outputId="28119e7e-408f-4236-f7de-e87f55af87db"
news_df.category.value_counts().plot(kind='bar', title='distribution of category values', figsize=(20, 10))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="TCUt5i6kXGPv" outputId="0bd84c43-2e04-4b01-9560-8ec5ec07b474"
clicks_df = pd.read_parquet("user_news_clicks.parquet.gzip")
clicks_df
```

```python id="17RqDc7AvDYF" colab={"base_uri": "https://localhost:8080/"} outputId="702bb09a-acd7-43b5-c2fa-920a3900bdfc"
print("unique users: ", len(clicks_df.user_id.unique()))
print("unique items: ", len(clicks_df.item.unique()))
print("unique interactions: ",len(clicks_df.click.unique()))
```

```python colab={"base_uri": "https://localhost:8080/"} id="xQGpkQOmKkKb" outputId="fd12a392-410f-4e19-a0ac-2f225d7a6770"
clicks_df.shape, clicks_df.drop_duplicates(subset=["user_id", "item", "click"]).shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 278} id="4Uw0KZNNJbCG" outputId="6ae318f5-5044-4817-ad9d-ede055610e6b"
clicks_df.click.value_counts().plot(kind='bar', title='distribution of non-clicks vs clicks');
```

<!-- #region id="Z0ysds68Wv_b" -->
## Encoding
<!-- #endregion -->

<!-- #region id="cNIiT6w2JKd6" -->
### Encode news articles text with embeddings from SentenceTransformers
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 732, "referenced_widgets": ["de36c163055c4958b4df5e063fdd5b6a", "1514ce55a32a4eca91587e769f6ef594", "319d47a5abab46c387c44b5436f393a8", "2ce854887eba4245a430b7515bb9cab2", "ab1c9a272ab84bf1aa1e4e2b7d2e81ab", "cc2365fba7d54594abbb6c75ea5e25bf", "7040fefd2a1c4cc4a91fab9cd3e1eeb6", "78a326f0371f481eb670b7da58b33ba1", "ddf88d1c94ce4662bd937f2ae7af6b34", "3cb62cb35415434d9fa03ad494b3b6dd", "788d4d3d77fd4066a34a32a395264118", "0ba663b8bf374534bfe9ab0fb9de1395", "998a7f2f5918498e9bacad59170f8d98", "8e2a7447783e4f55bdfe393f33bc8a27", "872edd463392476fb978c0d06cb94252", "663e61a07c89415d876b9cd6d9312f0b", "d80b984b5f564c5c9c372e4264b3a43c", "0a101d9b0c5548bebc4f828c0943cc04", "6f01e9a3bbbe4618bac8f738e232b3f5", "d1b80e6900444485bcd62ec30408191f", "3d8f2d3ab74e453abc2c02e83d27ea06", "9181a2fc7bae4f65b405f5256b24fbce", "8c6008ba07b64853b8f459a66461649e", "64a1331e07a64b1695b3d59eaf64c6f5", "84b3471e09db4b5c9e71e3f0c03765ed", "2d7fb371470348a1a5c1c0393ee18569", "cc80ae6ad7a94d7db78faf4a9d329ce9", "ea4c920020d546059df4e2c2c9f57f04", "1aaf45fd54f648fc81db2d61275f569f", "81bfe87f8d2d4d3ebd3951eca4f82e2d", "6caa64a3536847bda76f6078210bc90d", "0662aec4ed3e4ace8c2bab521492ebea", "58494a6f7346445d8828b694f4994e4d", "2640e67b5489458980b05b5cdb151d72", "0dffdd65a61b44b78af662b154615042", "28d55ab72437420b97ed6275f07b35e1", "9f3d31e5cf384a9e9ab35877266c8cfc", "28d8ea868fde4f65acd2e26d2440d928", "3e6e32d93ee44b50868694bf812699f5", "e4d7a37138a3404596360a8d24e129be", "439726c4f11d45a7bd26ff01e7848757", "4fe69605f43040728c8d3f7f0523fbdc", "1e9e51ad0bb94c0d8e57f14ef98576de", "5b8ca585742d4edc9c55137cd32b9bd1", "79885a59991b4738a77598123b4c8cc6", "8d23d600674d46cc868bcddd8baee482", "d50d5b5030b346e796ed649757afd75f", "0ca63ed648a5417c80bf2579e5d8bfd6", "4955e4c6efeb422aa4c77556171150cd", "a5889b6485394fa7bae3a4ce86cec689", "a8389596ec914dd5998e711585f61f89", "3cda9430315247b59dc711ca22b798e2", "adb2ee28e0784ff6aec904c3a1a251cb", "7b321fc358d14ab49c523c7faff71711", "45e19c6a5903495eac75f9f33d195e15", "4ed5a136e6844aca8e3fc43b124b9fb0", "d99e01a806c64a66ad7aac4432327104", "c65de05ef43042e0b0ba4fcecac5a0bf", "5b0d96646500462fa3de8558a41cd04e", "4ae5a88efc7f4bc78d65d2f8d358a647", "2ba710aebe9042d0b2ec7958e7a3739a", "ab08941cf4d9473c9f0225e3adabd5a4", "9ced448788b0454ba0bd8bf97a3c37d6", "abe8be478ac047f395b5bcb3c11bd165"]} id="8shqsHG4JLOZ" outputId="f1387bd0-daf9-4ffa-d8ac-70bcaae49547"
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('average_word_embeddings_komninos')

#Our sentences we like to encode
sentences = news_df['text'].tolist()

# run the encoder
embeddings = model.encode(sentences)

news_df['text_embedding'] = embeddings.tolist()

print(embeddings.shape)

news_df.head()
```

<!-- #region id="sUeMZJ3_58kw" -->
### Encode news category column
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 391} id="7FecqQx46AL6" outputId="6aa28315-4d62-4eb6-b899-f6b1bb2db359"
news_category_encoder = LabelEncoder()
news_df['category_encoded'] = news_category_encoder.fit_transform(news_df["category"])
news_dict = {r['news_id']: r for r in news_df.to_dict("rows")}
len(news_dict)
news_df.head()
```

<!-- #region id="9Ty26G_P683w" -->
### Encode user_id and item_id
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="xqyLvVwrS_Lo" outputId="e9f957fc-ef57-491e-dc02-a6131b81626f"
label_encoders = {}
label_encoders["user_id"] = LabelEncoder()
label_encoders["item"] = LabelEncoder()
clicks_df["user_id_encoded"] = label_encoders["user_id"].fit_transform(clicks_df["user_id"])
clicks_df["item_id_encoded"] = label_encoders["item"].fit_transform(clicks_df["item"])
clicks_df.head()
```

<!-- #region id="ANDCkqqp7Kgh" -->
## Data Preparation

- Random Sampling data for faster training with limited hardware resources
- There are lots of user-item interactions with more than one event, to keep it simple dropping these duplicates from the data as we are using these interactions as implicit feedback. This will also prevent training data leakage 
- Train and Test split will be done randomly as we don't have any timestamp values for the user-item interactions to split based on chronology of events.
- Not going for negative sampling to keep it simple here. The distribution of clicks vs non-clicks looks balanced already.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1QMX0kTJOEGS" outputId="b76ea307-d18b-42e9-fde7-6b573f2fc405"
traindf = clicks_df.drop_duplicates(subset=["user_id", "item", "click"])
traindf = traindf.sample(frac=0.4)
testdf = traindf.sample(frac=0.025)
traindf = traindf.drop(testdf.index)
traindf.shape, testdf.shape
```

```python id="CPRebwwrT5uX" colab={"base_uri": "https://localhost:8080/"} outputId="3279e65d-69e4-4e97-cb9d-d5524aa0c2c2"
traindf.click.value_counts()
```

```python colab={"base_uri": "https://localhost:8080/"} id="c3qnhVw6ImN8" outputId="10a55d07-2158-4143-b80c-8e493be4b167"
testdf.click.value_counts()
```

<!-- #region id="FDRHiZk9XCDB" -->
## Pytorch Dataset
<!-- #endregion -->

```python id="IXrDXI9JUAN9"
class MINDTrainDataset(Dataset):
    """
    PyTorch Dataset for Training MIND dataset
    """

    def __init__(self, interactions, all_news_ids):
        """
        Args:
        interactions (pd.DataFrame): Dataframe containing the interactions
        all_news_ids (dict): dict containing all news ids and its metadata
        """
        self.users, self.items, self.item_cats, self.labels = self.get_dataset(interactions, all_news_ids)

    def __len__(self):
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.item_cats[idx], self.labels[idx]

    def get_dataset(self, interactions, all_news_ids):
        users, items, labels, item_cats = [], [], [], []
        user_item_set = set(zip(interactions['user_id_encoded'], interactions['item'], interactions['click']))

        for u, i, l in user_item_set:
            users.append(u)
            items.append(all_news_ids[i]['text_embedding'])
            item_cats.append(all_news_ids[i]['category_encoded'])
            labels.append(l)
    
        return torch.tensor(users), torch.tensor(items), torch.tensor(item_cats), torch.tensor(labels)
```

<!-- #region id="tvzp62GYB2W2" -->
## Modelling
<!-- #endregion -->

<!-- #region id="4kTFFs84W9qq" -->
### Neural Collaborating Filtering Model
<!-- #endregion -->

```python id="KaTYF0XCUGbW"
class NCF(pl.LightningModule):
    """ 
    Neural Collaborative Filtering (NCF)
    """
    
    def __init__(self, num_users, num_item_cats, text_embedding_dim, interactions, all_news_ids, embedding_hidden_dim=16):
        """
         Args:
            num_users (int): Number of unique users
            num_item_cats (int): Number of unique item cats
            text_embedding_dim (int): dimensions of the text embedding
            interactions (pd.DataFrame): Dataframe containing the news clicks
            all_news_ids (dict): dict containing all news ids and its metadata
        """
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_hidden_dim)
        self.item_cat_embedding = nn.Embedding(num_embeddings=num_item_cats, embedding_dim=embedding_hidden_dim)
        self.input_feature_shape = text_embedding_dim+embedding_hidden_dim+embedding_hidden_dim
        print(self.input_feature_shape)
        self.fc1 = nn.Linear(in_features=self.input_feature_shape, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.interactions = interactions
        self.all_news_ids = all_news_ids
        
    def forward(self, user_input, item_cat_input, item_embedding_input):
        
        # Compute embeddings 
        user_embedded = self.user_embedding(user_input)
        item_cat_embedded = self.item_cat_embedding(item_cat_input)

        # Concat the embeddings
        vector = torch.cat([user_embedded, item_cat_embedded, item_embedding_input], dim=-1)

        # Pass through fully connected
        out = nn.ReLU()(self.fc1(vector))
        out = nn.ReLU()(self.fc2(out))
        out = nn.ReLU()(self.fc3(out))

        # Output layer
        pred = nn.Sigmoid()(self.output(out))

        return pred
    
    def training_step(self, batch, batch_idx):
        # compute the loss
        user_input, item_embedding, item_cat_input, labels = batch
        predicted_labels = self(user_input, item_cat_input, item_embedding)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MINDTrainDataset(self.interactions, self.all_news_ids),
                          batch_size=512, num_workers=4)
```

<!-- #region id="PcW3-nMgKHeQ" -->
### Training the model
<!-- #endregion -->

```python id="WtjdO7H9UOu2" colab={"base_uri": "https://localhost:8080/"} outputId="4bfd2605-beb8-4e4f-e999-2170e6da30e7"
num_users = max(traindf['user_id_encoded'].max()+1, testdf['user_id_encoded'].max()+1)
num_items_cat = news_df['category_encoded'].max()+1
text_embedding_dim = embeddings.shape[1]

model = NCF(num_users, num_items_cat, text_embedding_dim, traindf, news_dict)
```

```python colab={"base_uri": "https://localhost:8080/"} id="_smiy5rZgrim" outputId="420bdcdb-33b0-44f9-93db-99dd2f0fef1a"
text_embedding_dim, num_users, num_items_cat
```

```python id="hp-oN2P1dTNH"
logger = TensorBoardLogger("tb_logs", name="NCF_SBERT")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 338, "referenced_widgets": ["27f6a54c44f849e7842a0fb54d65b6c5", "88cf3e13f9b04eadb7268f5336f51696", "7d7d5354fb7647ff9ccbb1c8dfc7fe0e", "0e4416db9c644c39bb3bab8c8c74a95e", "3fe7f80799b34df88b52bf7caa4b69a4", "a3eedfe255d54ccc9cfb6b491dec4bf7", "5e71c51fd2434e45b79b6694581cd7ae", "bfc3b2cf9a094a49bcafdeaa03064860"]} id="u8IOj248UQ4u" outputId="745329b1-8b11-4088-9056-d208b525a735"
trainer = pl.Trainer(max_epochs=5, reload_dataloaders_every_epoch=True, progress_bar_refresh_rate=50, logger=logger, checkpoint_callback=False)

trainer.fit(model)
```

```python id="r8B7d-ERepjL"
test_item_embeddings = [news_dict.get(i).get("text_embedding") for i in testdf['item'].values]
test_itemcat_embeddings = [news_dict.get(i).get("category_encoded") for i in testdf['item'].values]

trainer.logger.experiment.add_graph(model, input_to_model=(torch.tensor(testdf['user_id_encoded'].values[0]), torch.tensor(test_itemcat_embeddings[0]), torch.tensor(test_item_embeddings[0])))
```

```python id="wjn7JdBlSgGj"
# %tensorboard --logdir tb_logs/
```

<!-- #region id="vs86TWb9J-LP" -->
## Evaluating the model



*   Calculate the classification metrics precision, recall and f1 score with a default threshold of 0.5
*   Plot confusion matrix, PR Curves and ROC curves.
*   From the metrics, we can observe that the precision is high overall but the recall is lower for clicks==1.



<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="8ZZ81DNkUVnn" outputId="0e6e1f30-6f50-41a0-ab35-bb9d849a0686"
predicted_labels = np.squeeze(model(torch.tensor(testdf['user_id_encoded'].values), torch.tensor(test_itemcat_embeddings), torch.tensor(test_item_embeddings)).detach().numpy())

testdf['prediction_conf1'] = predicted_labels.tolist()
testdf['prediction_conf0'] = (1-predicted_labels).tolist()
testdf['prediction'] = testdf['prediction_conf1'].apply(lambda z: 1 if z>0.5 else 0)
testdf
```

```python id="P6dSoi9KUWGx" colab={"base_uri": "https://localhost:8080/", "height": 382} outputId="6243f4b5-bcfe-4ef1-d720-51a1c95cb8e5"
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
    cmd.plot()

plot_confusion_matrix(testdf['click'], testdf['prediction'], [1, 0])
```

```python colab={"base_uri": "https://localhost:8080/"} id="27S2hVYzUY2L" outputId="b29d21dd-3395-47f8-d9b5-8b4dba838e80"
print(classification_report(testdf['click'], testdf['prediction'], [1, 0]))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 668} id="QjHWZQYxC3xy" outputId="48f0e9a1-d0b2-41a9-c66b-35ee26128d4b"
skplt.metrics.plot_precision_recall(testdf['click'].values, testdf[['prediction_conf0', 'prediction_conf1']].values, classes_to_plot=[0, 1], figsize=(10,10));
```

```python colab={"base_uri": "https://localhost:8080/", "height": 672} id="EOJJOdZBPrL0" outputId="e02103e3-ace8-4d58-c838-602a84e06641"
skplt.metrics.plot_roc_curve(testdf['click'].values, testdf[['prediction_conf0', 'prediction_conf1']].values, figsize=(10,10));
```

<!-- #region id="IGU94rXwDhar" -->
## Visualizing outputs
Doing some sanity checks to figure if the model outputs make sense. Randomly taking an user id and checking the titles from train clicks and predicted clicks.
<!-- #endregion -->

```python id="oUFrqf5SDgT1"
def get_wordcloud_for_user(text_list):

    stopwords = set(STOPWORDS).union([np.nan, 'NaN', 'S'])

    wordcloud = WordCloud(
                   max_words=50000,
                   min_font_size =12,
                   max_font_size=50,
                   relative_scaling = 0.9,
                   stopwords=set(STOPWORDS),
                   normalize_plurals= True
    )

    clean_titles = [word for word in text_list if word not in stopwords]
    title_wordcloud = wordcloud.generate(' '.join(clean_titles))

    plt.figure(figsize = (10,10))
    plt.imshow(title_wordcloud, interpolation='bilinear',)
    plt.axis("off")
    plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Vg_o5GTn-ve-" outputId="6e651556-4bd4-466b-ba97-31f5d2e1fbf5"
# traindf.user_id.value_counts()[44300:44305]
ii = traindf[(traindf['user_id']=="U84756") & (traindf['click']==1)]['item'].unique()
tlist = [news_dict[i]['title'] for i in ii]
# tlist
news_df[news_df['news_id'].isin(ii)]['category'].value_counts()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 344} id="30BP3ZgXJmbW" outputId="5b6d46c5-3079-4648-e7d8-70bbac0786bf"
get_wordcloud_for_user(tlist)
```

```python colab={"base_uri": "https://localhost:8080/"} id="jmgu9bloI4WZ" outputId="3dc351f4-cc47-4c5e-d889-e807cdbb4d41"
ii = testdf[(testdf['user_id']=="U84756") & (testdf['prediction']==1)]['item'].unique()
news_df[news_df['news_id'].isin(ii)]['category'].value_counts()
```

```python id="cvfXdeYPIyva" colab={"base_uri": "https://localhost:8080/", "height": 344} outputId="d9a719ed-ee43-4ffb-f448-8f5d57fa8121"
# testdf[testdf['user_id']=="U84756"]
tlist1 = [news_dict[i]['title'] for i in ii]
get_wordcloud_for_user(tlist1)
```

<!-- #region id="Qeq4tXXm_Ggj" -->
## Further Improvements

- More feature engineering: user-category affinity score, chronological based train-test split and evaluation, etc
- Hyperparamter tuning: finding the best values for different hyperparams like text embedding method, Neural Net layers and hidden sizes, batch size, epochs, etc.
- Finetuning the prediction probabilty thresholds to find the right balance. i.e, Precision-Recall tradeoff.
<!-- #endregion -->
