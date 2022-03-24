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

```python id="ryZiTbDtHQcs" executionInfo={"status": "ok", "timestamp": 1609390158193, "user_tz": -330, "elapsed": 1203, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import torch
import pandas as pd
from tqdm.notebook import tqdm
```

```python colab={"base_uri": "https://localhost:8080/"} id="XrIMNLkeHGLj" executionInfo={"status": "ok", "timestamp": 1609390009558, "user_tz": -330, "elapsed": 1374, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6973bd10-9780-4f4e-e2b7-98acc1169c8a"
!wget 'https://raw.githubusercontent.com/Kausthub8/Emotion-Analysis-Using-BERT-With-Pytorch/master/smile-annotations-final.csv'
```

```python id="q8wVuN7BHXyk" executionInfo={"status": "ok", "timestamp": 1609390531833, "user_tz": -330, "elapsed": 1764, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
df = pd.read_csv('smile-annotations-final.csv', names=['id','text','category'])
df.set_index('id', inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="feg3HCRsIB5B" executionInfo={"status": "ok", "timestamp": 1609390531836, "user_tz": -330, "elapsed": 1486, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2e4b85e5-ee43-4b69-e958-ec56af7ab0ac"
df.sample(10)
```

```python colab={"base_uri": "https://localhost:8080/"} id="1SK_XUIOICrs" executionInfo={"status": "ok", "timestamp": 1609390531839, "user_tz": -330, "elapsed": 1215, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cc30ca4c-efa1-4744-fd78-7475e21a9f3e"
df.category.value_counts()
```

```python colab={"base_uri": "https://localhost:8080/"} id="5kVEOkcqIk1O" executionInfo={"status": "ok", "timestamp": 1609390533786, "user_tz": -330, "elapsed": 1657, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dca6ae0e-8a9c-4e65-a8c5-ff86aa6c30a4"
df = df[~df.category.str.contains('\|')]
df = df[df.category!='nocode']
df.category.value_counts()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="uLdQiC5nJQh1" executionInfo={"status": "ok", "timestamp": 1609390516426, "user_tz": -330, "elapsed": 1512, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="08203cb3-09ad-406d-b220-3f729d3227d4"
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="F1FsHNssJh7p" executionInfo={"status": "ok", "timestamp": 1609390737537, "user_tz": -330, "elapsed": 4164, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1ad9b876-d5b1-40e8-f78c-ce0e8d78687a"
possible_labels = df.category.unique()
label_dict = {}
for index, label in enumerate(possible_labels):
  label_dict[label] = index
label_dict
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="0hG5O5v6KENu" executionInfo={"status": "ok", "timestamp": 1609390746276, "user_tz": -330, "elapsed": 3086, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="520a710a-6927-42c9-9e5b-6535d057b600"
df['label'] = df.category.replace(label_dict)
df.sample(5)
```

```python id="js4KtxLeKd4Z" executionInfo={"status": "ok", "timestamp": 1609391407741, "user_tz": -330, "elapsed": 1438, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                  df.label.values,
                                                  test_size=0.2,
                                                  random_state=40,
                                                  stratify=df.label.values)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="ButDu2gCMSGH" executionInfo={"status": "ok", "timestamp": 1609391410448, "user_tz": -330, "elapsed": 1629, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f517051e-720d-4668-f928-766aa2be82ed"
df['data_type'] = ['not_set']*df.shape[0]
df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'
df.sample(5)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 452} id="l7c3wnDiMeRH" executionInfo={"status": "ok", "timestamp": 1609391492865, "user_tz": -330, "elapsed": 1206, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a7cf1e07-428f-4e44-b058-fe5017a8b9a3"
df.groupby(['category','label','data_type']).count()
```

```python id="upFlhLxnOggp"
!pip install -q transformers
```

```python id="neF30_p-NSLn" executionInfo={"status": "ok", "timestamp": 1609391810510, "user_tz": -330, "elapsed": 3778, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
```

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["5ec39ce234dd4de9ada27d84e48e852d", "c4fcfb0f0a554c6c95a9cfb9af835f23", "c4e81ef2f3194303afe2fa0d490f67d0", "0318a3e032574f21b2e4aface099acf4", "ec2095b2dba6400499f7370e71394403", "292f69c4c83e4527b2793a9a29bcaa75", "f9a2cda402864401b53be91bbe185dff", "81f7a49d812c468da8f6ec7fbf559b18"]} id="xtuqcQeyRt87" executionInfo={"status": "ok", "timestamp": 1609392688688, "user_tz": -330, "elapsed": 2294, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="31b3ab0a-c9af-4166-af08-90cfe3f0f74e"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
```

```python id="IcQAsppPOPDa" executionInfo={"status": "ok", "timestamp": 1609393604440, "user_tz": -330, "elapsed": 2000, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
encoded_data_train = tokenizer.batch_encode_plus(df[df.data_type=='train'].text.values, 
                                                     add_special_tokens = True, 
                                                     return_attention_mask = True, 
                                                     padding = True, 
                                                     max_length = 256, 
                                                     return_tensors = 'pt')

encoded_data_val = tokenizer.batch_encode_plus(df[df.data_type=='val'].text.values,
                                                     add_special_tokens = True,
                                                     return_attention_mask = True,
                                                     padding = True,
                                                     max_length = 256,
                                                     return_tensors = 'pt')
```

```python id="-PMT2Dl-ORLA" executionInfo={"status": "ok", "timestamp": 1609394645436, "user_tz": -330, "elapsed": 1224, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = encoded_data_train['token_type_ids']

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = encoded_data_val['token_type_ids']
```

```python id="iHOz8ZzxPxGW" executionInfo={"status": "ok", "timestamp": 1609394799856, "user_tz": -330, "elapsed": 847, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
```

```python colab={"base_uri": "https://localhost:8080/"} id="G_2kPFW8Veuu" executionInfo={"status": "ok", "timestamp": 1609394854416, "user_tz": -330, "elapsed": 1085, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f9fedf4d-361e-4e98-ced3-9b40c618aa81"
len(dataset_train)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 220, "referenced_widgets": ["69a8ebcfe0fc4cec9d9618e58babe7cc", "47e6381203784a5ba26d2aaf856e1cad", "fd0f9738cce74a11b0693f96f213290a", "cc1ce6aea5154734b357dde86cb4eeac", "eb7981a100ea429fa6030df1c52a533c", "0006713ece7f44058f6c90fb3d0e3537", "01564ce7f9464ad8a7a119955d0e5b7a", "7579d88cf0ab431a8d66c66717fdf069", "128c75b6495e4244a7dd5696afbde6cb", "1fb4a1f273194216a9f49885a5ff5f66", "d637d9f167a04d289bf75eb549e8a0fa", "16eaba89673b46afb842cb3df2bd2aa0", "dcad758ec5ff45068a14f29942cc76d6", "b65dc934da6a42f5a849972a67da75a6", "cd361ca1d5784cacb477e961f868cc57", "957a51e35f294ea983c6a8d6907d449f"]} id="d1SIQ1OHaJVu" executionInfo={"status": "ok", "timestamp": 1609395444067, "user_tz": -330, "elapsed": 19997, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9771b405-dd2c-4f40-f8de-1ee759191895"
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
```

```python id="yWeFcmkMbuba" executionInfo={"status": "ok", "timestamp": 1609396490818, "user_tz": -330, "elapsed": 1261, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 4 #32

dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
dataloader_val = DataLoader(dataset_val, sampler=RandomSampler(dataset_val), batch_size=batch_size)
```

```python id="Vhfmifj4cBs9" executionInfo={"status": "ok", "timestamp": 1609396549733, "user_tz": -330, "elapsed": 1141, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from transformers import AdamW, get_linear_schedule_with_warmup

epochs = 10
optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*epochs)
```

```python id="7H3eM0SMggBg" executionInfo={"status": "ok", "timestamp": 1609396915121, "user_tz": -330, "elapsed": 1311, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
from sklearn.metrics import f1_score

def f1_score_func(preds, labels):
  preds_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return f1_score(labels_flat, preds_flat, average='weighted')
```

```python id="E188qFZ9iAWL" executionInfo={"status": "ok", "timestamp": 1609397545978, "user_tz": -330, "elapsed": 4777, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def accuracy_per_class(preds, labels):
  label_dict_inverse = {v: k for k, v in label_dict.items()}

  preds_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()

  for label in np.unique(labels_flat):
    y_preds = preds_flat[labels_flat==label]
    y_true = labels_flat[labels_flat==label]
    print(f'Class: {label_dict_inverse[label]}')
    print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}')
```

```python colab={"base_uri": "https://localhost:8080/"} id="hDegwGQhkTea" executionInfo={"status": "ok", "timestamp": 1609399928166, "user_tz": -330, "elapsed": 1088, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4a92dc35-a496-427b-bb19-662eb2929699"
import random

seed = 40
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)
```

```python id="GKdoosVwkkeQ" executionInfo={"status": "ok", "timestamp": 1609400051761, "user_tz": -330, "elapsed": 1115, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals
```

```python colab={"base_uri": "https://localhost:8080/", "height": 438, "referenced_widgets": ["a99951125f0b437f88fb0242b2dbdc5f", "683139851a0c4486bafedd55b83fd493", "402591a8e8d942498bc445fe75b9b1dc", "f666c31cd0274e04a4900adeabe8a668", "1dc7d56bca224057af37789492587eae", "c70fab65380d444ba5d5b94674a4bf71", "123bfbae1e904996993aec361992fef6", "c522df53eeb54d1d8daccfe2ae993e78", "b89ef450ad0c4a048eafec8f93e526be", "bdc7029b30fb4ac4b98c0a2ed219fcf8", "74d0015e3ef141a98f94070ef7be8b3c", "9f7241eef86448b9ae06f497e0f60c45", "0747df4ac7b242c78e377e6f64c5b28e", "6f3c0b6b55844677bf3506c8d1c80c2e", "c326ebf2ba91451c86b5aed835770a6d", "3504a7655b2442a18f2a4facd5a69a69"]} id="J6ttp2-dt-Lv" executionInfo={"status": "error", "timestamp": 1609401308860, "user_tz": -330, "elapsed": 1912, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6dd992d1-cced-4c95-a84d-dd4fb8d9d3fc"
for epoch in tqdm(range(1, epochs+1)):

  model.train()
  loss_train_total = 0
  progress_bar = tqdm(dataloader_train, 
                      desc='Epoch {:1d}'.format(epoch),
                      leave=False,
                      disable=False)
  
  for batch in progress_bar:
    model.zero_grad()
    batch = tuple(b.to(device) for b in batch)
    
    inputs = {'input_ids':      batch[0],
              'attention_mask': batch[1],
              'labels':         batch[2],
              }
    outputs = model(**inputs)
    loss = outputs[0]
    loss_train_total+=loss.item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    progress_bar.set_postfix('training loss: {:.3f}'.format(loss.item()/len(batch)))

  torch.save(model.state_dict(), f'BERT_ft_epoch{epoch}.model')

  tqdm.write(f'\nEpoch {epoch}')

  loss_train_avg = loss_train_total/len(dataloader_train)
  tqdm.write(f'Training loss: {loss_train_avg}')

  val_loss, predictions, true_vals = evaluate(dataloader_val)
  val_f1 = f1_score_func(predictions, true_vals)
  tqdm.write(f'Validation loss: {val_loss}')
  tqdm.write(f'F1 score (weighted): {val_f1}')
```

```python id="dun8INOHxpPN"
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

model.load_state_dict(torch.load('BERT_ft_epoch1.model'))
```
