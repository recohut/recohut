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

<!-- #region id="TsOmtZXQ8m9p" -->
# Name and Address Parsing
> Recognising Person names and Addresses in a text using NER and NLP modeling techniques
<!-- #endregion -->

<!-- #region id="VrUPAMQG8m2C" -->
## Get data from MongoDB
<!-- #endregion -->

```python id="PHeH-3lxcJOU"
!pip install pymongo[tls,srv]
```

```python id="Y8H-4V3hV-P-" executionInfo={"status": "ok", "timestamp": 1644417906793, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}}
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
%reload_ext google.colab.data_table
```

```python id="Z4Aq0xjrbJMK"
import pymongo 
import pprint
mongo_uri = "mongodb+srv://<userid>:<pass>@<server>.azure.mongodb.net/<db>?retryWrites=true&w=majority"
client = pymongo.MongoClient(mongo_uri)
```

```python id="_Cck8Qk9bi3o" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1598257716586, "user_tz": -330, "elapsed": 2217, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c6675435-fd2e-48e7-9d9f-29159e085eca"
listdb = client.list_database_names(); listdb
```

```python id="xnE-iMxOcXQk" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1598257716589, "user_tz": -330, "elapsed": 2184, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="79239908-5b07-4e45-d38c-bf3dea3fd6d9"
db = client['testdb']
db.list_collection_names()
```

```python id="N8nLlaTfZrw0" colab={"base_uri": "https://localhost:8080/", "height": 323} executionInfo={"status": "ok", "timestamp": 1598254717305, "user_tz": -330, "elapsed": 1326, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0c3d35b0-0c7b-4cb3-8f17-0f6fead8d7be"
# print(db.command("collstats", "events"))
db.command("dbstats")
```

```python id="5Df44YMqcfhw"
# collection1 = db['entities']
# df1 = pd.DataFrame(list(collection1.find()))
# print(df1.info())
# # print(df1.describe())
# df1.sample(5)
```

```python id="C7nJmPzxaPDj" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1598255037308, "user_tz": -330, "elapsed": 1313, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bc9d4931-64f4-48ab-e5c2-c55af474b008"
# df1.to_pickle('/content/drive/My Drive/df_entities.p')
# del df1
# import gc
# gc.collect()
```

```python id="V3EpjYPZke20" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1598258271633, "user_tz": -330, "elapsed": 224945, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e050bb73-1e87-4204-c78d-a0a6a620859d"
collection2 = db['parcels']
cursor = collection2.find()
max = 0
maxL = {}
for i in range(cnt):
  xx = next(cursor)
  if len(xx)>max:
    max = len(xx)
    maxL = xx
max
```

```python id="pNBAGKuIi7z0"
cnt = collection2.count()
```

```python id="CH6eXfVHnwBV" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1598259865546, "user_tz": -330, "elapsed": 1933, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dd0f7751-5a52-4a0c-8d87-b9e14b8d83d8"
list(maxL)
```

```python id="-BQj2kP3bu2p" colab={"base_uri": "https://localhost:8080/"} outputId="aac4f6a1-57b0-43fe-be21-b234916b7eb8"
collection2 = db['parcels']
cursor = collection2.find()
XX = pd.DataFrame(columns=list(maxL))
XX.to_csv('df_parcels.csv', index=False)
for i in tqdm(range((cnt//10000)+1)):
  YY = {}
  for j in range(10000):
    YY[j] = next(cursor)
  pd.DataFrame(YY).T.to_csv('df_parcels.csv', mode='a', index=False, header=False)
# pd.DataFrame(next(cursor), index=[j]).to_csv('df_parcels.csv', mode='a', index=False, header=False)
```

```python id="HzraWr_bicHu"
pd.read_csv('df_parcels.csv')
```

```python id="-RAQq6ZNWauK"
collection2 = db['parcels']
df2 = pd.DataFrame(list(collection2.find()))
print(df2.info())
# print(df2.describe())
# df2.sample(5)
```

<!-- #region id="J-3mWKGVWXB4" -->
## Data wrangling
<!-- #endregion -->

```python id="BAkSUQd_Rho5"
!pip install -q probablepeople
!pip install -q usaddress
```

```python id="EdO9-A8zqi8-"
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import  probablepeople as pp
import usaddress as ua
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

import warnings
warnings.filterwarnings("ignore")

tqdm.pandas()
%reload_ext autoreload
%autoreload 2
%reload_ext google.colab.data_table
%config InlineBackend.figure_format = 'retina'

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-notebook')
```

```python id="rfSy_7dlqkBx" colab={"base_uri": "https://localhost:8080/", "height": 170} executionInfo={"status": "ok", "timestamp": 1598880959613, "user_tz": -330, "elapsed": 5556, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fa353d72-75bd-4443-a5ec-22413cbae08d"
df = pd.read_pickle(os.path.join(path,'sample.p'))
df.info()
data = df.oor.tolist()
```

```python id="L5n4FVhjIbIE"
data[:10]
```

```python id="UP0OzFuhb6ok"
combos = [' US ',' USA ',' U.S. ',' U.S.A. ']
def patch_country(text):
  cflagz = ''
  otext = text
  text = text[-10:]
  text = ' ' + text + ' '
  for i in combos:
    if i in text:
      cflagz = i
  text = otext[:-10] + re.sub(cflagz, '', text)
  return text, cflagz
```

```python id="2450Yq-SzpAp"
def patch_household(xz):
  xx = pd.DataFrame(pp.tag(xz)[0], index=[0])
  xx['Type'] = 'Person'
  if 'And' in xx.columns.tolist():
    xx = pd.DataFrame({'Household':xz}, index=[0])
    xx['Type'] = 'Household'
  return xx
```

```python id="xV9zxm4zWMK2"
tags = pp.LABELS
tags.extend(ua.LABELS)
tags = list(set(tags))
additionals = ['CountryName', 'Household', 'Type', 'NameConfidence', 'AddrConfidence', 'Text']
tags.extend(additionals)
```

```python id="AJhLnVY6SE9c"
dfx = pd.DataFrame(columns=tags)
dfx = dfx.loc[:,~dfx.columns.duplicated()]
```

```python id="4oN4623MPkpn"
# def temp(text):
#   # replace long country name with short one
#   text = re.sub('UNITED STATES OF AMERICA','US',text)
#   # country patch
#   text, cflag = patch_country(text)
#   # try and catch
#   try:
#     df1 = pd.DataFrame(ua.tag(text)[0], index=[0])
#     xz = df1.Recipient.values[0]
#     df2 = patch_household(xz)
#     xx = pd.concat([df1,df2], axis=1)
#     xx['Confidence'] = ua.tag(text)[1]
#   except:
#     try:
#       df2 = patch_household(text)
#       xx = df2
#       xx['Confidence'] = 'AddressError'
#     except:
#       xx = pd.DataFrame({'Recipient':text}, index=[0])
#       xx['Confidence'] = 'PersonError'
#   # add country label
#   if cflag!='':
#     xx['CountryName'] = cflag
    
#   return xx#.T.to_dict()[0]
```

```python id="xmmrxUxw9-Hf"
def temp(text):
  # country patch
  text = re.sub('UNITED STATES OF AMERICA','US',text)
  text, cflag = patch_country(text)
  # address parsing
  try:
    df1 = pd.DataFrame(ua.tag(text)[0], index=[0])
    df1['AddrConfidence'] = ua.tag(text)[1]
  except:
    df1 = pd.DataFrame(ua.parse(text)).groupby(1).agg({0: lambda x: ' '.join(x)}).T
    df1['AddrConfidence'] = 'Error'
  # address to name linking
  try:
    xz = df1.Recipient.values[0]
  except:
    xz = text
  # name parsing
  try:
    df2 = pd.DataFrame(pp.tag(xz)[0], index=[0])
    df2['NameConfidence'] = pp.tag(xz)[1]
  except:
    df2 = pd.DataFrame(pp.parse(xz)).groupby(1).agg({0: lambda x: ' '.join(x)}).T
    df2['NameConfidence'] = 'Error'
  # person name patch
  df2['Type'] = 'Person'
  if 'MiddleName' in df2.columns.tolist():
    gname, mname, sname = df2.MiddleName, df2.Surname, df2.GivenName
    df2.MiddleName, df2.Surname, df2.GivenName = mname[0], sname[0], gname[0]
  elif 'GivenName' in df2.columns.tolist():
    gname, sname = df2.Surname, df2.GivenName
    df2.Surname, df2.GivenName = sname[0], gname[0]
  # household name patch
  if 'And' in df2.columns.tolist():
    df2 = pd.DataFrame({'Household':xz}, index=[0])
    df2['Type'] = 'Household'
  # concatenation
  df = pd.concat([df1,df2], axis=1)
  df['Text'] = text  
  df['CountryName'] = cflag if cflag!='' else np.NaN
  try:
    df.loc[pd.notnull(df['CorporationName']), 'Type'] = 'Corporation'
  except:
    pass
  return df
```

```python id="Pqf1BTMnQ4xw" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1598880979797, "user_tz": -330, "elapsed": 20370, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c052486b-601f-40f7-8170-9491d89f795c"
errs = {}
dfx = pd.DataFrame(columns=tags)
dfx = dfx.loc[:,~dfx.columns.duplicated()]
for idx, text in tqdm(enumerate(data)):
  try:
    dfx = dfx.append(temp(text))
    dfx.loc[pd.notnull(dfx['CorporationName']), 'Type'] = 'Corporation'
  except Exception as e:
    errs[idx] = e
    pass
dfx = dfx.fillna('')
```

```python id="ympXVT2WAzO-"
err_df = pd.DataFrame(errs, index=[0]).T
err_df = err_df.merge(pd.Series(data, name='text'), left_index=True, right_index=True)
err_df.columns = ['error','text']
err_df = err_df[['text','error']]
err_df.iloc[:]
```

```python id="9FtzHpHQZ02Y"
dfbackup = dfx.copy()
```

```python id="6XzAt00oPkrr" colab={"base_uri": "https://localhost:8080/", "height": 799} executionInfo={"status": "ok", "timestamp": 1598880980594, "user_tz": -330, "elapsed": 8107, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f43402ea-bbe4-4c8b-a845-5286d611f52d"
dfx = dfbackup[tags]
dfx.drop(['Recipient', 'CountryName'], axis=1, inplace=True)
dfx = dfx.replace(r'^\s*$', np.nan, regex=True).dropna(axis=1, how='all')
dfx.info()
```

```python id="Ut4vvaNfvAKl"
def func(text):
  text = str(text)
  xx = temp(text)
  xx = xx.drop(['AddrConfidence','NameConfidence','Text'], axis=1)
  try:
    xx = xx.drop(['Recipient'], axis=1)
  except:
    pass
  xx = xx.T.to_dict()[0]
  return str(xx)
```

```python id="alKerkOZxUKn"
# !pip install -q gradio
# import gradio as gr
gr.Interface(fn=func, inputs="text", outputs="text").launch()
```

<!-- #region id="bQgQT_LdwWTj" -->
## Sampling
<!-- #endregion -->

```python id="oLfPSe_OMMH_"
path = '.'
```

```python id="Bp5OfflRMZCy"
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
tqdm.pandas()
warnings.filterwarnings("ignore")
```

```python id="UCOY1SBvMg2K"
# df = pd.read_pickle(os.path.join(path,'df_entities.p'))
# df.info()
```

```python id="5Q19mhNfMnZ_"
# def process_text(text):
#   text = re.sub('"','',text)
#   text = re.sub(',','',text)
#   text = ' '.join(text.split())
#   return text

# sampledf = df[['oor']].sample(5000, random_state=40)
# sampledf['oor'] = sampledf.oor.apply(process_text)
# sampledf = sampledf.replace(r'^\s*$', np.nan, regex=True).dropna()
# msk = np.random.rand(len(sampledf)) < 0.8
# sampledf[msk].to_csv('train_28118_21.csv', index=False, header=None)
# sampledf[~msk].to_csv('test_28118_21.csv', index=False, header=None)
```

```python id="vBtmcL2qdzga"
!pip install doccano-transformer

import spacy
import random
from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl
dataset = read_jsonl(filepath='file.json1', dataset=NERDataset, encoding='utf-8')
dataset.to_spacy(tokenizer=str.split)

TRAIN_DATA = []
for x in dataset:
  xx = {}
  try:
    xx['entities'] = list(x.labels.values())[0]
    TRAIN_DATA.append((x.text.lower(), xx))
  except:
    pass
```

```python id="PEqMJCyUOevA" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1599732604029, "user_tz": -330, "elapsed": 5438068, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1b4461a1-df02-4be4-d378-0f3661926393"
def train_spacy(data, iterations, model=None):
    TRAIN_DATA = data
    if model is not None:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model") 
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")
    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.1,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp

prdnlp = train_spacy(TRAIN_DATA, 200)
```

```python id="6IROcg0WSYSt"
# Save our trained Model
# modelfile = input("Enter your Model Name: ")
# prdnlp.to_disk(modelfile)

prdnlp.to_disk('ner233')
import shutil
shutil.make_archive(os.path.join(path,'ner233'),'zip','/content/ner233')
```

```python id="WgeN6BEKZUwi"
# test = pd.read_csv('/content/test_28118_21.csv', header=None)
# test.head()
```

```python id="wjQ6WFnup-oc" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1599668534705, "user_tz": -330, "elapsed": 1350, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="710046ce-da2c-4b4b-a8e7-f3a0f328b7a8"
# from spacy.gold import GoldParse
# from spacy.scorer import Scorer

# def evaluate(ner_model, examples):
#     scorer = Scorer()
#     for input_, annot in examples:
#         doc_gold_text = ner_model.make_doc(input_)
#         gold = GoldParse(doc_gold_text, entities=annot)
#         pred_value = ner_model(input_)
#         scorer.score(pred_value, gold)
#     return scorer.scores

# examples = [
#     ('Who is Shaka Khan?',
#      [(7, 17, 'PERSON')]),
#     ('I like London and Berlin.',
#      [(7, 13, 'LOC'), (18, 24, 'LOC')])
# ]

# results = evaluate(prdnlp, examples)

# def report_scores(scores, e):
#     """
#     prints precision recall and f_measure
#     :param scores:
#     :return:
#     """

#     precision = '%.2f' % scores['ents_p']
#     recall = '%.2f' % scores['ents_r']
#     f_measure = '%.2f' % scores['ents_f']
#     print('%-25s %-10s %-10s %-10s' % (e, precision, recall, f_measure))

# report_scores(results, 'x')
```

```python id="rnoAYRVlVsKH"
# Household -> Pass again for name tags
```

```python id="9RRLg-hRdTfk"
tags = ['EntityType','Recipient','Address',
        'GivenName','MiddleName','SurName','Household','Corporation',
        'StreetAddress','City','State','Zipcode','Country']
```

```python id="CXvulQXDuHBm"
PAD_TAIL = '<redacted>'
```

```python id="f61JpO2Gk3au"
def household_patch(row):
  if row.EntityType=='Household':
    text = row.Household.split('&')[0] if '&' in row.Household else ' '.join(row.Household.split()[:2])
    text = parseit(text+PAD_TAIL)
    row['GivenName'] = text.loc[0,'GivenName']
    row['SurName'] = text.loc[0,'SurName']
    row['MiddleName'] = text.loc[0,'MiddleName']
  return row
```

```python id="cYy3cYdtfh_r"
def add_on(xx):
  # Address is the combination of street, city, state, zip and country
  xx['Address'] = xx['StreetAddress'] +' '+ xx['City'] +' '+ xx['State'] +' '+ xx['Zipcode'] +' '+ xx['Country']
  # default is person, if household column is not empty, then household, same for corporation
  xx['EntityType'] = 'Person'
  xx.loc[xx.Household!='','EntityType'] = 'Household'
  xx.loc[xx.Corporation!='','EntityType'] = 'Corporation'
  # default is full name of person, if entity is corporation ,then corporation name, same for household
  xx['Recipient'] = xx['SurName'] +' '+ xx['GivenName'] +' '+ xx['MiddleName']
  xx['Recipient'] = xx.apply(lambda row: row.Corporation if row.EntityType=='Corporation' else row.Recipient, axis=1)
  xx['Recipient'] = xx.apply(lambda row: row.Household if row.EntityType=='Household' else row.Recipient, axis=1)
  # adding household to name field patch
  xx = xx.apply(household_patch, axis=1)
  # converting to dictionary format
  xx = xx.replace(r'^\s*$', np.nan, regex=True).dropna(axis=1, how='any')
  xx = xx.T.to_dict()[0]
  # return the processed data
  return xx
```

```python id="wIDjsgAwalUj"
def parseit(text):
  text = str(text).upper()
  text = re.sub('"','',text)
  text = re.sub(',','',text)
  text = ' '.join(text.split())
  output = {}
  doc = prdnlp(text)
  for ent in doc.ents:
    output[ent.label_] = ent.text
  
  df = pd.DataFrame(output, index=[0])
  dfx = pd.DataFrame(columns=tags)
  dfx = dfx.append(df).fillna('')
  
  return dfx
```

```python id="VeOX3tA_zhiI"
def func(text):
  X = parseit(text)
  X = add_on(X)
  X = str(X)
  return X
```

```python id="ZglmehQyZ6Ur"
# text = test[0].sample().tolist()[0]
text = '<redacted>'
# xx = parseit(text); xx
# xx = pd.DataFrame(xx, index=[0]).T.reset_index(); xx.columns = ['Tag','Entity']; xx
```

```python id="ylrk0jP9zKiP"
func('<redacted>')
```

```python id="DWhuQEEmw6Ot"
# !pip install -q gradio
# import gradio as gr
# inputs = gr.inputs.Textbox(lines=3, label='Input')
# outputs = gr.outputs.Textbox(label='Output')
gr.Interface(fn=func, inputs=inputs, outputs=outputs).launch()
```

<!-- #region id="2cqkiFIL9lkg" -->
## Flair NER Model
<!-- #endregion -->

```python id="k4VKjq42L2h0"
!pip install doccano-transformer
from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl
dataset = read_jsonl(filepath='file_2.json1', dataset=NERDataset, encoding='utf-8')
dataset.to_spacy(tokenizer=str.split)
```

```python id="03DIsZSGMEnr"
TRAIN_DATA = []
for x in dataset:
  xx = {}
  try:
    xx['entities'] = list(x.labels.values())[0]
    TRAIN_DATA.append((x.text, xx))
  except:
    pass
```

```python id="2-1bvOKMNiTi"
import spacy
from spacy.gold import biluo_tags_from_offsets
nlp = spacy.load('en_core_web_sm')
docs = []
for text, annot in TRAIN_DATA:
    doc = nlp(text)
    tags = biluo_tags_from_offsets(doc, annot['entities'])
    tags = [x.replace('U-','B-') for x in tags]
    tags = [x.replace('L-','I-') for x in tags]
    tmpdocs = [(x,y) for x,y in zip(text.split(),tags)]
    docs.append(tmpdocs)
```

```python id="7i8vCu0RNuTM"
import random
import pandas as pd
X = pd.DataFrame(docs[0])
X['break'] = 'train'
X = X.append(pd.Series(dtype='str'), ignore_index=True)
for x in docs[1:]:
  Xt = pd.DataFrame(x)
  Xt = Xt.append(pd.Series(dtype='str'), ignore_index=True) 
  Xt['break'] = random.choices(['train','val','test'], weights=(90,5,5), k=1)[0]
  X = X.append(Xt)
X = X.fillna('')
X = X.reset_index(drop=True)

train = X.loc[X['break']=='train',[0,1]]; train.to_csv('train.txt', sep=' ', header=None, index=False)
val = X.loc[X['break']=='val',[0,1]]; val.to_csv('val.txt', sep=' ', header=None, index=False)
test = X.loc[X['break']=='test',[0,1]]; test.to_csv('test.txt', sep=' ', header=None, index=False)
```

```python id="_T8F7ynowCY7"
import re
import numpy as np
```

```python id="cRpBVQr6O_OW"
!pip install -q flair
from flair.data import Corpus
from flair.datasets import ColumnCorpus

columns = {0:'text', 1:'ner'}

data_folder = '/content'
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file = 'train.txt',
                              test_file = 'test.txt',
                              dev_file = 'val.txt')
```

```python id="8dVyIvPdTba4"
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
```

```python id="kZLuva-kdxjA"
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings, CharacterEmbeddings

embedding_types = [

    WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
]

embeddings : StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
```

```python id="cW_64WOQfZnX"
from flair.models import SequenceTagger
tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                       embeddings=embeddings,
                                       tag_dictionary=tag_dictionary,
                                       tag_type=tag_type,
                                       use_crf=True)
```

```python id="XYMKQqVhffhi" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1599816744463, "user_tz": -330, "elapsed": 133453, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1a32b71f-38cc-464c-a6f1-2f87dc2d4d2d"
from flair.trainers import ModelTrainer
trainer : ModelTrainer = ModelTrainer(tagger, corpus)
trainer.train('/content',
              learning_rate=0.5,
              mini_batch_size=32,
              max_epochs=150,
              )
```

```python id="lwtMkZr4f7VT" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1599816802678, "user_tz": -330, "elapsed": 3559, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0750c25d-d3ed-44d9-a0fb-162c7360361d"
from flair.data import Sentence
from flair.models import SequenceTagger
model = SequenceTagger.load('/content/final-model.pt')

text = '<redacted>'
sentence = Sentence(text)
model.predict(sentence)
out = {}
for entity in sentence.get_spans('ner'):
  out[entity.tag] = entity.text
```

```python id="_wfIor6Zk2md"
tags = ['EntityType','Recipient','Address',
        'GivenName','MiddleName','SurName','Household','Corporation',
        'StreetAddress','City','State','Zipcode','Country', 'Confidence']

PAD_TAIL = '<redacted>'

def household_patch(row):
  if row.EntityType=='Household':
    text = row.Household.split('&')[0] if '&' in row.Household else ' '.join(row.Household.split()[:2])
    text = parseit(text+PAD_TAIL)
    row['GivenName'] = text.loc[0,'GivenName']
    row['SurName'] = text.loc[0,'SurName']
    row['MiddleName'] = text.loc[0,'MiddleName']
  return row

def add_on(xx):
  # Address is the combination of street, city, state, zip and country
  xx['Address'] = xx['StreetAddress'] +' '+ xx['City'] +' '+ xx['State'] +' '+ xx['Zipcode'] +' '+ xx['Country']
  # default is person, if household column is not empty, then household, same for corporation
  xx['EntityType'] = 'Person'
  xx.loc[xx.Household!='','EntityType'] = 'Household'
  xx.loc[xx.Corporation!='','EntityType'] = 'Corporation'
  # default is full name of person, if entity is corporation ,then corporation name, same for household
  xx['Recipient'] = xx['SurName'] +' '+ xx['GivenName'] +' '+ xx['MiddleName']
  xx['Recipient'] = xx.apply(lambda row: row.Corporation if row.EntityType=='Corporation' else row.Recipient, axis=1)
  xx['Recipient'] = xx.apply(lambda row: row.Household if row.EntityType=='Household' else row.Recipient, axis=1)
  # adding household to name field patch
  xx = xx.apply(household_patch, axis=1)
  # converting to dictionary format
  xx = xx.replace(r'^\s*$', np.nan, regex=True).dropna(axis=1, how='any')
  xx = xx.T.to_dict()[0]
  # return the processed data
  return xx

def parseit(text):
  text = str(text).upper()
  text = re.sub('"','',text)
  text = re.sub(',','',text)
  text = ' '.join(text.split())
  output = {}
  sentence = Sentence(text)
  model.predict(sentence)
  for entity in sentence.get_spans('ner'):
    output[entity.tag] = entity.text
  df = pd.DataFrame(output, index=[0])
  dfx = pd.DataFrame(columns=tags)
  dfx = dfx.append(df).fillna('')
  dfx['Confidence'] = ','.join([f'[{x.tag} {str(round(x.score,2))}]'for x in sentence.get_spans('ner')])
  return dfx

def func(text):
  X = parseit(text)
  X = add_on(X)
  X = str(X)
  return X
```

```python id="UjhE8pMImoAF"
func('<redacted>')
```

```python id="k1idZJroCUmR"
!pip install -q gradio
import gradio as gr
inputs = gr.inputs.Textbox(lines=3, label='Input')
outputs = gr.outputs.Textbox(label='Output')
```

```python id="A1u08rpDG0C9"
gr.Interface(fn=func, inputs=inputs, outputs=outputs).launch()
```

<!-- #region id="-e9acRTB_tbj" -->
## API
<!-- #endregion -->

```python id="IakPL8Fxwl4L"
!pip install -q probablepeople
!pip install -q usaddress
```

```python id="Ihr48DD5wnI2"
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import  probablepeople as pp
import usaddress as ua

import warnings
warnings.filterwarnings("ignore")

tqdm.pandas()
%reload_ext autoreload
%autoreload 2
%reload_ext google.colab.data_table
%config InlineBackend.figure_format = 'retina'

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-notebook')
```

```python id="NCw5hexdwsj9"
df = pd.read_pickle(os.path.join(path,'sample.p'))
data = df.oor.tolist()
data[:10]
```

```python id="_sxWicuvwyp-"
def patch_country(text):
  combos = [' US ',' USA ',' U.S. ',' U.S.A. ']
  cflagz = ''
  otext = text
  text = text[-10:]
  text = ' ' + text + ' '
  for i in combos:
    if i in text:
      cflagz = i
  text = otext[:-10] + re.sub(cflagz, '', text)
  cflagz = cflagz.strip()
  return text, cflagz

def patch_household(xz):
  xx = pd.DataFrame(pp.tag(xz)[0], index=[0])
  xx['Type'] = 'Person'
  if 'And' in xx.columns.tolist():
    xx = pd.DataFrame({'Household':xz}, index=[0])
    xx['Type'] = 'Household'
  return xx

tags = pp.LABELS
tags.extend(ua.LABELS)
tags = list(set(tags))
additionals = ['CountryName', 'Household', 'Type', 
               'NameConfidence', 'AddrConfidence', 'Text']
tags.extend(additionals)

dfx = pd.DataFrame(columns=tags)
dfx = dfx.loc[:,~dfx.columns.duplicated()]
```

```python id="rM6gLffxE8E5"
def process_text(text):
  text = re.sub(',','',text)
  text = ' '.join(text.split())
  text = re.sub('UNITED STATES OF AMERICA','US',text)
  text = re.sub('TEXAS','TX',text)
  return text
```

```python id="wStcHr-dxBB4"
def temp(text):
  text = process_text(text)
  text, cflag = patch_country(text)
  # address parsing
  try:
    df1 = pd.DataFrame(ua.tag(text)[0], index=[0])
    df1['AddrConfidence'] = ua.tag(text)[1]
  except:
    df1 = pd.DataFrame(ua.parse(text)).groupby(1).agg({0: lambda x: ' '.join(x)}).T
    df1['AddrConfidence'] = 'Error'
  # address to name linking
  try:
    xz = df1.Recipient.values[0]
  except:
    xz = text
  # name parsing
  try:
    df2 = pd.DataFrame(pp.tag(xz)[0], index=[0])
    df2['NameConfidence'] = pp.tag(xz)[1]
  except:
    df2 = pd.DataFrame(pp.parse(xz)).groupby(1).agg({0: lambda x: ' '.join(x)}).T
    df2['NameConfidence'] = 'Error'
  df2['Type'] = 'Person'
  # household name patch
  if 'And' in df2.columns.tolist():
    df2 = pd.DataFrame({'Household':xz}, index=[0])
    df2['Type'] = 'Household'
  # concatenation
  df = pd.concat([df1,df2], axis=1)
  df['Text'] = text  
  df['CountryName'] = cflag if cflag!='' else np.NaN
  dfx = pd.DataFrame(columns=tags)
  dfx = dfx.loc[:,~dfx.columns.duplicated()]
  df = dfx.append(df)
  df = df.fillna('')
  df.loc[df['CorporationName']!='', 'Type'] = 'Corporation'
  if (df['MiddleInitial'][0]=='') & (df['MiddleName'][0]!=''):
    gname, mname, sname = df.MiddleName, df.Surname, df.GivenName
    df['MiddleName'], df['Surname'], dfx['GivenName'] = mname[0], sname[0], gname[0]
  # elif (df['MiddleInitial'][0]=='') & (df['MiddleName'][0]==''):
  #   gname, sname = df.Surname, df.GivenName
  #   df['Surname'], df['GivenName'] = sname[0], gname[0]
  return df
```

```python id="qemvICfa0Y_g"
tagmap = pd.read_excel(os.path.join(path,'tag_map.xlsx')).reset_index()
tagmap.columns = ['_'.join(col.split()) for col in tagmap.columns]
tagmap
tagmap = pd.melt(tagmap, id_vars=['index'], value_vars=['FIRSTNAME','MIDDLENAME','LASTNAME',
                                                        'OTHERNAME','HOUSEHOLD','CORPORATION',
                                                        'STREET_ADDRESS', 'CITY', 'STATE', 'PINCODE', 
                                                        'COUNTRY', 'META', 'OTHER']).dropna()[['variable','value']].set_index('value').to_dict()['variable']
```

```python id="KFwypUBTLSGA"
def custom_patch(text):
  text = re.sub(' DALLA S ', ' DALLAS ', text)
  text = re.sub(' PRAIRI E ', ' PRAIRIE ', text)
  text = re.sub(' EL PAS O ', ' EL PASO ', text)
  text = re.sub(' SAN ANT ONIO ', ' SAN ANTONIO ', text)
  text = re.sub(' THE WOODL ANDS ', ' THE WOODLANDS ', text)
  text = re.sub(' FORT WORT H ', ' FORT WORTH ', text)
  text = re.sub(' ARLINGTO N ', ' ARLINGTON ', text)
  text = re.sub(' SOCORR O ', ' SOCORRO ', text)
  text = re.sub(' HURS T ', ' HURST ', text)
  text = re.sub(' HOLLYWOO D ', ' HOLLYWOOD ', text)
  return text
```

```python id="OPh5efJ-DEOn"
def oor_api(text):
  texto = text
  xx = temp(text)
  xx.columns = xx.columns.to_series().map(tagmap)
  xx = xx.drop('META', axis=1)
  xx = xx.replace(r'^\s*$', np.nan, regex=True).dropna(axis=1, how='any')
  try:
    xx['PINCODE'] = xx['PINCODE'].apply(lambda x: re.sub(' ','',x))
  except:
    pass
  xx = xx.T
  xx = xx.reset_index()
  xx.index = xx.index.map(str)
  xx = ' '+ xx + ' '
  xx[0] = xx[0].apply(custom_patch)
  xx = xx.to_dict()
  label2tag = {str(v): ' '+str(k)+' ' for k, v in xx[0].items()}
  text = process_text(text)
  text = ' ' + text + ' '
  for key, value in label2tag.items():
    text = text.replace(key, value)
  x = pd.DataFrame({'entity':text.split(), 'tag':text.split()})
  x = x.replace({'entity': xx[0], 'tag': xx['index']})
  x['entity'] = x.groupby(['tag'])['entity'].transform(lambda x: ' '.join(x))
  x = x.drop_duplicates()
  x['tag'] = x['tag'].apply(process_text)
  x['entity'] = x['entity'].apply(process_text)
  x = x.append(pd.Series({'tag':'TEXT','entity':texto}), ignore_index=True)
  x = x.set_index('tag')
  x = x.to_dict()['entity']
  return x
```

```python id="BsVtlnx3QwvZ"
# zz = data[5]
# oor_api(zz)
```

```python id="NRYQ0XZBCI4i" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1599292456543, "user_tz": -330, "elapsed": 65542, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b8c7647b-e65b-466b-d1ca-410a03b4576b"
oor_cols = ['TEXT','FIRSTNAME','MIDDLENAME','LASTNAME', 'OTHERNAME', 'HOUSEHOLD', 
            'CORPORATION', 'STREET_ADDRESS', 'CITY', 'STATE', 'PINCODE', 'COUNTRY']
dfx = pd.DataFrame(columns=oor_cols)
for idx, text in tqdm(enumerate(data)):
  dfx = dfx.append(pd.DataFrame(oor_api(text), index=[idx]))
dfx = dfx.fillna('')
dfx = dfx[oor_cols]
```

```python id="D0ib1GecEMh2"
dfx = dfx.replace(r'^\s*$', np.nan, regex=True)
```

```python id="TCU9tFJtZxzj" colab={"base_uri": "https://localhost:8080/", "height": 340} executionInfo={"status": "ok", "timestamp": 1599292456548, "user_tz": -330, "elapsed": 63971, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a712624c-96b6-4833-d53a-7e070c1b428b"
dfx.info()
```

```python id="iBfXkHZXZz_p"
dfx.to_csv('x.csv')
```

```python id="fr6Rco32akIo"
x = '<redacted>'
temp(x).T.replace('',np.nan).dropna()
```

```python id="fQ5v0CiU99o9"
oor_api(x)
```

<!-- #region id="At_JoRYLAykF" -->
## Deliverables
<!-- #endregion -->

```python id="fIl-OFnoOXyg"
# !pip install cookiecutter
# !cookiecutter https://github.com/drivendata/cookiecutter-data-science
```

```python id="zVCyma0HVh2X"
imports _requirements.txt_
!pip install flair
```

```python id="CTZUlBjTWodq"
# utils

import pandas as pd

tags = ['EntityType','Recipient','Address',
        'GivenName','MiddleName','SurName','Household','Corporation',
        'StreetAddress','City','State','Zipcode','Country', 'Confidence']
name_tags = ['EntityType','Recipient', 'GivenName','MiddleName','SurName','Household','Corporation']
addr_tags = ['Address','StreetAddress','City','State','Zipcode','Country']

PAD_TAIL = '<redacted>'
PAD_HEAD = '<redacted>'

def household_patch(row):
  if row.EntityType=='Household':
    text = row.Household.split('&')[0] if '&' in row.Household else ' '.join(row.Household.split()[:2])
    text = parseit(text+PAD_TAIL)
    row['GivenName'] = text.loc[0,'GivenName']
    row['SurName'] = text.loc[0,'SurName']
    row['MiddleName'] = text.loc[0,'MiddleName']
  return row

def add_on(xx):
  # Address is the combination of street, city, state, zip and country
  xx['Address'] = xx['StreetAddress'] +' '+ xx['City'] +' '+ xx['State'] +' '+ xx['Zipcode'] +' '+ xx['Country']
  # default is person, if household column is not empty, then household, same for corporation
  xx['EntityType'] = 'Person'
  xx.loc[xx.Household!='','EntityType'] = 'Household'
  xx.loc[xx.Corporation!='','EntityType'] = 'Corporation'
  # default is full name of person, if entity is corporation ,then corporation name, same for household
  xx['Recipient'] = xx['SurName'] +' '+ xx['GivenName'] +' '+ xx['MiddleName']
  xx['Recipient'] = xx.apply(lambda row: row.Corporation if row.EntityType=='Corporation' else row.Recipient, axis=1)
  xx['Recipient'] = xx.apply(lambda row: row.Household if row.EntityType=='Household' else row.Recipient, axis=1)
  # adding household to name field patch
  xx = xx.apply(household_patch, axis=1)
  # converting to dictionary format
  xx = xx.replace(r'^\s*$', np.nan, regex=True).dropna(axis=1, how='any')
  # return the processed data
  return xx
```

```python id="WiQhXopPXOJC" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1599829364233, "user_tz": -330, "elapsed": 3908, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0df25d4f-7375-454f-ab74-22059d9e2ffb"
# pred

import re
import numpy as np
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger

model = SequenceTagger.load('models/final-model.pt')

def _parse(text):
  text = str(text).upper()
  text = re.sub('"','',text)
  text = re.sub(',','',text)
  text = ' '.join(text.split())
  output = {}
  sentence = Sentence(text)
  model.predict(sentence)
  for entity in sentence.get_spans('ner'):
    output[entity.tag] = entity.text
  df = pd.DataFrame(output, index=[0])
  dfx = pd.DataFrame(columns=tags)
  dfx = dfx.append(df).fillna('')
  dfx['Confidence'] = ','.join([f'[{x.tag} {str(round(x.score,2))}]'for x in sentence.get_spans('ner')])
  return dfx
```

```python id="iQXElBhcQ5xz"
def _predict(text):
  X = _parse(text)
  X = add_on(X)
  X = X.T.to_dict()[0]
  return X

def _predict_name(text):
  text = text + PAD_TAIL
  X = _parse(text)
  X = add_on(X)
  rqd_cols = list(set(X.columns) & set(name_tags))
  X = X[rqd_cols]
  X = X.T.to_dict()[0]
  return X

def _predict_address(text):
  text = PAD_HEAD + text
  X = _parse(text)
  X = add_on(X)
  rqd_cols = list(set(X.columns) & set(addr_tags))
  X = X[rqd_cols]
  X = X.T.to_dict()[0]
  return X
```

```python id="Jdt5jDAVWqE3"
# app
_predict('<redacted>')
```

<!-- #region id="xCWWvms8hv_W" -->
---
<!-- #endregion -->

```python id="Fb4ddrm-jW0l" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1599830827753, "user_tz": -330, "elapsed": 1907, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="051634ac-31d7-4130-f05c-f6ff0e3b2afd"
%%writefile src/app.py

import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
from flair.data import Sentence
from flair.models import SequenceTagger

from utils import *
from model import loadmodel

path = Path(__file__)
_PPATH = str(path.parents[1])+'/'

start_time = time.time()
model = loadmodel.finalmodel
print("---Encoder--- %s seconds ---" % (time.time() - start_time))

def _parse(text):
  text = str(text).upper()
  text = re.sub('"','',text)
  text = re.sub(',','',text)
  text = ' '.join(text.split())
  output = {}
  sentence = Sentence(text)
  model.predict(sentence)
  for entity in sentence.get_spans('ner'):
    output[entity.tag] = entity.text
  df = pd.DataFrame(output, index=[0])
  dfx = pd.DataFrame(columns=tags)
  dfx = dfx.append(df).fillna('')
  dfx['Confidence'] = ','.join([f'[{x.tag} {str(round(x.score,2))}]'for x in sentence.get_spans('ner')])
  return dfx

def _predict(text):
  X = _parse(text)
  X = add_on(X)
  X = X.T.to_dict()[0]
  return X

def _predict_name(text):
  text = text + PAD_TAIL
  X = _parse(text)
  X = add_on(X)
  rqd_cols = list(set(X.columns) & set(name_tags))
  X = X[rqd_cols]
  X = X.T.to_dict()[0]
  return X

def _predict_address(text):
  text = PAD_HEAD + text
  X = _parse(text)
  X = add_on(X)
  rqd_cols = list(set(X.columns) & set(addr_tags))
  X = X[rqd_cols]
  X = X.T.to_dict()[0]
  return X

########## FLASK API ##########

from flask import Flask, request, jsonify, send_file
import json

app = Flask(__name__)

@app.route("/oor", methods=["POST"])
def parse_oor():
  req_data = request.get_json()
  text = req_data['query']
  preds = _predict(text)
  return jsonify(preds)

@app.route("/name", methods=["POST"])
def parse_name():
  req_data = request.get_json()
  text = req_data['query']
  preds = _predict_name(text)
  return jsonify(preds)

@app.route("/address", methods=["POST"])
def parse_address():
  req_data = request.get_json()
  text = req_data['query']
  preds = _predict_address(text)
  return jsonify(preds)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
```

```python id="F_fQ5snNsvt-" colab={"base_uri": "https://localhost:8080/", "height": 52} executionInfo={"status": "ok", "timestamp": 1600272321665, "user_tz": -330, "elapsed": 5649, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="825017c1-d76c-480b-bcf4-d928e0732e48"
!pip install -q pyngrok
from pyngrok import ngrok
!ngrok authtoken <redacted>
```

```python id="KrqPRFnQskIH" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1600272321667, "user_tz": -330, "elapsed": 4824, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f366f6f2-76c3-4d68-82bb-b4e0b354990a"
ngrok.kill()
public_url = ngrok.connect(port='5000'); public_url
```

```python id="udHTbEJ1sWTr" colab={"base_uri": "https://localhost:8080/", "height": 277} executionInfo={"status": "ok", "timestamp": 1600272564397, "user_tz": -330, "elapsed": 241055, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f6966754-d3ae-4878-da3c-be241503f6d0"
!python src/app.py
```

<!-- #region id="GAoFBYgrDfOF" -->
## Pipeline
<!-- #endregion -->

```python id="0Lp1rKZ2NX5b"
!pip install -q probablepeople
!pip install -q usaddress
```

```python id="iUxe6E8VNX5c"
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import  probablepeople as pp
import usaddress as ua

import shutil
# shutil.make_archive(os.path.join(path,'customNER'),'zip','/content/customNER')
shutil.unpack_archive(os.path.join(path,'customNER.zip'), '/content/customNER', 'zip')
```

```python id="oTc1Csj_NX5c"
df = pd.read_pickle(os.path.join(path,'sample.p'))
data = df.oor.tolist()
data[100:110]
```

```python id="5APu8pBoFGQs" colab={"base_uri": "https://localhost:8080/", "height": 170} executionInfo={"status": "ok", "timestamp": 1599579636238, "user_tz": -330, "elapsed": 19992, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e2939ead-925a-46e7-e9fd-b034ff570ce1"
df.info()
```

```python id="IIIURNHjUM3c"
import spacy
model = spacy.load('/content/customNER')
```

```python id="hRP78cDPJvs-"
text1 = '<redacted>'

def name_parsing(text):
  try:
    text = pp.tag(text)
    textdf = pd.DataFrame(text[0], index=[0])
    textdf['NameType'] = text[1]
    textdf['NameConfidence'] = 'Tag'
    textdf = textdf.T.to_dict()[0]
  except:
    textdf = pp.parse(text)
    textdf = pd.DataFrame(textdf).groupby(1).agg({0: lambda x: ' '.join(x)}).T
    textdf['NameType'] = 'Ambiguous'
    textdf['NameConfidence'] = 'Parse'
    textdf = textdf.T.to_dict()[0]
  return textdf

name_parsing(text3)
```

```python id="FTTvJxiiPK6m"
def preprocess(text):
  text = re.sub(',','',text)
  text = ' '.join(text.split())
  text = re.sub('UNITED STATES OF AMERICA','US',text)
  return text
```

```python id="iWzRe8I8INJw"
text1 = '<redacted>'

def address_parsing(text):
  try:
    text = preprocess(text)
    text = ua.tag(text)
    textdf = pd.DataFrame(text[0], index=[0])
    textdf['AddressType'] = text[1]
    textdf['AddrConfidence'] = 'Tag'
    textdf = textdf.T.to_dict()[0]
  except:
    textdf = ua.parse(text)
    textdf = pd.DataFrame(textdf).groupby(1).agg({0: lambda x: ' '.join(x)}).T
    textdf['AddressType'] = 'Ambiguous'
    textdf['AddrConfidence'] = 'Parse'
    textdf = textdf.T.to_dict()[0]
  return textdf

xx = address_parsing(text2); xx
```

```python id="OFiP7Zm8wb4E" colab={"base_uri": "https://localhost:8080/", "height": 170} executionInfo={"status": "ok", "timestamp": 1599586085970, "user_tz": -330, "elapsed": 1126, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="15491f28-6808-46ac-806a-f5aa1cd708d7"
list(xx.keys())
```

```python id="IzZtirGAN91a"
text1 = '<redacted>'

def full_parsing(text):
  doc = model(text)
  ners = {ent.label_:ent.text for ent in doc.ents}
  name = address = {}
  try:
    address = address_parsing(ners['Address'])
  except:
    pass
  try:
    name = name_parsing(ners['Name'])
  except:
    pass
  address.update(name)
  try:
    del address['Recipient']
  except:
    pass
  if 'And' in list(address.keys()):
    xx = pd.DataFrame({'Household':xz}, index=[0])
    xx['Type'] = 'Household'
  return address

xx = full_parsing(text2); xx
```

```python id="hC1-HKnIQ9VB"
tags = pp.LABELS
tags.extend(ua.LABELS)
tags = list(set(tags))
additionals = ['Text', 'AddressType','AddrConfidence','NameType','NameConfidence']
tags.extend(additionals)
dfx = pd.DataFrame(columns=tags)
dfx = dfx.loc[:,~dfx.columns.duplicated()]
for idx, text in tqdm(enumerate(data)):
  dfx = dfx.append(full_parsing(text), ignore_index=True)
  dfx.loc[idx, 'Text'] = text
```

```python id="CXuLYZ5Yyv_-"
sorted_collist = dfx.isna().sum().sort_values().index
dfx = dfx[sorted_collist]
```

```python id="bOMIw6rMs3ER" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1599553735678, "user_tz": -330, "elapsed": 1103, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="04eafcb3-9b5f-472e-d202-e381b3d54451"
dfx.info()
```

```python id="iuipqNrMTOe8" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1599553743693, "user_tz": -330, "elapsed": 2068, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="32312c01-0bd1-4626-bc51-a2dd7c55db33"
dfx0 = dfx.dropna(how='all').reset_index(drop=True); print(dfx0.info())
dfx1 = dfx.dropna(how='all', axis=1).dropna(how='all').reset_index(drop=True); dfx1.info()
```

```python id="Hogu3vvsTi-c"
# dfx0.to_csv('dfx0.csv')
dfx1.to_csv('dfx1.csv')
```

```python id="1Yd-Ws-ku2wQ"
def func(text):
  text = str(text)
  output = full_parsing(text)
  return str(output)

txt = '<redacted>'
func(txt)
```

```python id="WgZkY-uxYIOU"
!pip install -q gradio
import gradio as gr
```

```python id="fyRPhmbuYJfr"
gr.Interface(fn=func, inputs="text", outputs="text").launch()
```
