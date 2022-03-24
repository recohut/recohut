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
    language: python
    name: python3
---

```python
import pandas as pd
import sklearn as sk
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import sys
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
sys.setrecursionlimit(10**6) 
```

```python tags=[]
csv_file = pd.read_csv("pres2_v2.csv")
print(csv_file.shape)
print(csv_file[csv_file["class"]==0].shape, csv_file[csv_file["class"]==1].shape)

```

### Cleaning all the data 

```python
csv_file.pop('index')
csv_file["city"] = csv_file.city.str.lower()
csv_file["county"] = csv_file.county.str.lower()
csv_file["ipcity"] = csv_file.ipcity.str.lower()
csv_file["ipstate"] = csv_file.ipstate.str.lower()
csv_file["browser"] = csv_file.browser.str.lower()
csv_file["os"] = csv_file.os.str.lower()
csv_file["os_v"] = csv_file.os_v.str.lower()
csv_file["device"] = csv_file.device.str.lower()
csv_file["device_family"] = csv_file.device_family.str.lower()
csv_file["device_model"] = csv_file.device_model.str.lower()

```

```python
csv_file["city"] = csv_file.city.astype('category').cat.codes
csv_file["county"] = csv_file.county.astype('category').cat.codes
csv_file["ipcity"] = csv_file.ipcity.astype('category').cat.codes
csv_file["ipstate"] = csv_file.ipstate.astype('category').cat.codes
csv_file["browser"] = csv_file.browser.astype('category').cat.codes
csv_file["os"] = csv_file.os.astype('category').cat.codes
csv_file["os_v"] = csv_file.os_v.astype('category').cat.codes
csv_file["device"] = csv_file.device.astype('category').cat.codes
csv_file["device_family"] = csv_file.device_family.astype('category').cat.codes
csv_file["device_model"] = csv_file.device_model.astype('category').cat.codes
csv_file["ippostcode"] = csv_file.ippostcode.astype('category').cat.codes

csv_file.head()
```

### Check for the class imbalance and take equal entries from both

```python tags=[]
majority = csv_file[csv_file["class"]==0]
minority = csv_file[csv_file["class"]==1]
print("Majority shape : ", majority.shape)
print("Minority shape : ", minority.shape)

new_df = pd.concat([majority.sample(minority.shape[0],), minority])
new_df.shape
```

```python
# Split your train and test data in 7:3 ration
train_x,test_x= sk.model_selection.train_test_split(new_df, test_size=0.30)
train_x.shape

train_y = train_x.pop("class") 
test_y = test_x.pop("class") 
```

### Train a gradient boosting classifier

```python tags=[]
clf = GradientBoostingClassifier(random_state=0,n_estimators=100)
selector = RFE(clf)
selector = selector.fit(train_x, train_y)
print(selector.support_)

pred = selector.predict(test_x)
```

```python tags=[]
print("Macro f1 : ",f1_score(test_y, pred, average="macro"))
print("Macro Precission : ",precision_score(test_y, pred, average="macro"))
print("Macro Recall : ",recall_score(test_y, pred, average="macro"))  
print("Macro f1 : ",f1_score(test_y, pred, average="micro"))
print("Macro Precission : ",precision_score(test_y, pred, average="micro"))
print("Macro Recall : ",recall_score(test_y, pred, average="micro"))  
print("Binary f1 : ",f1_score(test_y, pred, average="binary"))
print("Binary Precission : ",precision_score(test_y, pred, average="binary"))
print("Binary Recall : ",recall_score(test_y, pred, average="binary"))  
print(confusion_matrix(test_y,pred))
```

```python
output_df = pd.DataFrame()
```

```python
output_df["actual_class"] = test_y
output_df["predicted_class"] = pred
output_df.head()
```

```python

```
