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

```python id="P6EIvBGnS7PW"
# Managing the category type
# The categories can be S (for promotion), 0 (when unknown), 
# a number between 1-12 when it came from a category on the page
# or a 8-10 digit number that represents a brand

# def assign_cat(x):
#   if x == "S":
#       return "PROMOTION"
#   elif np.int(x) == 0:
#       return "NONE"
#   elif np.int(x) < 13:
#       return "CATEGORY"
#   else:
#       return "BRAND"

# df_clicks["Item_Type"] = df_clicks.iloc[:,3].map(assign_cat)
```

```python id="9gdI5SlLS9mZ"
# fraction = 64

# PATH_TO_PROCESSED_DATA = '../../data/'

# data = pd.read_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_tr.txt', sep='\t', dtype={'ItemId':np.int64})
# train = data
# length = len(data['ItemId'])

# print('Full Training Set:\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(data), data.SessionId.nunique(), data.ItemId.nunique()))

# print("\nGetting most recent 1/{} fraction of training test...\n".format(fraction))
# first_session = train.iloc[length-length//fraction].SessionId
# train = train.loc[train['SessionId'] >= first_session]

# itemids = train['ItemId'].unique()
# n_items = len(itemids)

# print('Fractioned train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
# train.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_fraction_1_{}.txt'.format(fraction), sep='\t', index=False)
```

```python id="6O2xnAE_S_4r"

```

```python cellView="form" id="nECbnLVMpovT"
#@title extractDwellTime.py

from matplotlib import pyplot as plt
import argparse
import numpy as np
import pandas as pd


def preprocess_df(df):    
    n_items = len(train_data['ItemId'].unique())
    aux = list(train_data['ItemId'].unique())
    itemids = np.array(aux)
    itemidmap = pd.Series(data=np.arange(n_items), index=itemids)  # (id_item => (0, n_items))
    
    item_key = 'ItemId'
    session_key = 'SessionId'
    time_key = 'Time'
    
    data = pd.merge(df, pd.DataFrame({item_key:itemids, 'ItemIdx':itemidmap[itemids].values}), on=item_key, how='inner')
    data.sort_values([session_key, time_key], inplace=True)

    length = len(data['ItemId'])
        
    return data


def compute_dwell_time(df):
    times_t = np.roll(df['Time'], -1)  # Take time row
    times_dt  = df['Time']             # Copy, then displace by one
    
    diffs = np.subtract(times_t, times_dt)  # Take the pairwise difference
    
    length = len(df['ItemId'])
    
    # cummulative offset start for each session
    offset_sessions = np.zeros(df['SessionId'].nunique()+1, dtype=np.int32)
    offset_sessions[1:] = df.groupby('SessionId').size().cumsum() 
    
    offset_sessions = offset_sessions - 1
    offset_sessions = np.roll(offset_sessions, -1)
    
    # session transition implies zero-dwell-time
    # note: paper statistics do not consider null entries, 
    # though they are still checked when augmenting
    np.put(diffs.values, offset_sessions, np.zeros((offset_sessions.shape)), mode='raise')
    return diffs


def get_statistics(dts):
    filtered = np.array(list(filter(lambda x: int(x) != 0, dts)))
    pd_dts = pd.DataFrame(filtered)
    pd_dts.boxplot(vert=False, showfliers=False) # no outliers in boxplot
    plt.show()
    pd_dts.describe()


def join_dwell_reps(df, dt, threshold=2000):
    # Calculate d_ti/threshold + 1, add column to dataFrame
    dt //= threshold
    dt += 1   
    df['DwellReps'] = pd.Series(dt.astype(np.int64), index=dt.index)


def augment(df):    
    col_names = list(df.columns.values)[:3]
    print(col_names)
    augmented = np.repeat(df.values, df['DwellReps'], axis=0) 
    print(augmented[0][:3])  
    augmented = pd.DataFrame(data=augmented[:,:3],
                             columns=col_names)
    dtype = {'SessionId': np.int64, 
             'ItemId': np.int64, 
             'Time': np.float32}
    
    for k, v in dtype.items():
        augmented[k] = augmented[k].astype(v)

    return augmented


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DwellTime extractor')
    parser.add_argument('--train-path', type=str, default='../processedData/rsc15_train_tr.txt')
    parser.add_argument('--output-path', type=str, default='../processedData/augmented_train.csv')
    args = parser.parse_args()

    # load RSC15 preprocessed train dataframe
    train_data = pd.read_csv(args.train_path, sep='\t', dtype={'ItemId':np.int64})

    new_df = preprocess_df(train_data)
    dts = compute_dwell_time(new_df)

    # get_statistics(dts)

    join_dwell_reps(new_df, dts, threshold=200000)

    # Now, we augment the sessions copying each entry an additional (dwellReps[i]-1) times
    df_aug = augment(new_df)
    df_aug.to_csv(args.output_path, index=False, sep='\t')
```

<!-- #region id="S2ElKay9pp6u" -->
## Feature Engineering
<!-- #endregion -->

<!-- #region id="L6EgBZy6psH3" -->
| Name                               | Type    |
| ---------------------------------- | ------- |
| No. of clicks                      | Session |
| No. of unique items                | Session |
| Avg. no. of clicks per unique item | Session |
| Session duration in seconds        | Session |
| Average time between two clicks    | Session |
| Maximal time between two clicks    | Session |
| Day of the week                    | Session |
| Month of the year                  | Session |
| Time during the day                | Session |
| Total clicks on the item           | Item    |
| Total buys on the item             | Item    |
| Max price of the item              | Item    |
| Min price of the item              | Item    |
| Item id                            | Item    |
| Category id                        | Item    |
<!-- #endregion -->

```python id="5vBjkxSdTcI7"
#@title GRU4Rec.py
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class GRU4Rec(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.
    Note:
        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, dataset):
        super(GRU4Rec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.loss_type = config['loss_type']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
```
