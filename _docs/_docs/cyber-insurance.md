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

# Cyber Insurance Analytics

<!-- #region id="F4kunXX35Oeq" -->
## Module 1 Simulator code
<!-- #endregion -->

```python id="iVnIDgpH433j"
import scipy
from scipy import stats
from scipy.stats import beta
import numpy as np
import random
```

```python id="Jd6YyIVL433p"
def beta_simulator(x):
    minimum = x[0]
    mode = x[1]
    maximum = x[2]
    conf = x[3]
    if conf=='low':
        d = (minimum + mode + 4*maximum)/6
    elif conf=='medium':
        d = (minimum + 4*mode + maximum)/6
    else:
        d = (4*minimum + mode + maximum)/6
    alpha = 6*((d - minimum)/(maximum - minimum))
    beta = 6*((maximum - d)/(maximum - minimum))
    location = minimum
    scale = maximum - minimum
    return scipy.stats.beta.ppf(np.random.rand(), alpha, beta, location, scale)
```

```python id="e493-i4V433t"
tef = [2,5,20,'medium']
difficulty = [70,90,95,'high']
tcap = [80,85,90,'low']
```

```python id="y3WKt0G2433x"
n = 10000
lef = []
for i in range(0,n+1):
    if(beta_simulator(tcap) > beta_simulator(difficulty)):
        lef.append(beta_simulator(tef))
    else:
        lef.append(0)
```

```python id="aTEmp2AF4331" outputId="b3711950-80ee-4a71-ddae-7ca1b3706c56"
sum(lef)/n
```

<!-- #region id="CET-AhUC43He" -->
## Modelling Intrusion Detection: Analysis of a Feature Selection Mechanism


### Step 1: Data preprocessing:
All features are made numerical using one-Hot-encoding. The features are scaled to avoid features with large values that may weigh too much in the results.

### Step 2: Feature Selection:
Eliminate redundant and irrelevant data by selecting a subset of relevant features that fully represents the given problem.
Univariate feature selection with ANOVA F-test. This analyzes each feature individually to detemine the strength of the relationship between the feature and labels. Using SecondPercentile method (sklearn.feature_selection) to select features based on percentile of the highest scores. 
When this subset is found: Recursive Feature Elimination (RFE) is applied.

### Step 4: Build the model:
Decision tree model is built.

### Step 5: Prediction & Evaluation (validation):
Using the test data to make predictions of the model.
Multiple scores are considered such as:accuracy score, recall, f-measure, confusion matrix.
perform a 10-fold cross-validation.
<!-- #endregion -->

<!-- #region id="G4OMbS9x43Hf" -->
## Version Check
<!-- #endregion -->

```python id="t3p15jMO43Hg" outputId="2229afb1-f248-4e6f-95f1-ded5cc532673"
import pandas as pd
import numpy as np
import sys
import sklearn
print(pd.__version__)
print(np.__version__)
print(sys.version)
print(sklearn.__version__)
```

<!-- #region id="QUWCxOTf43Hp" -->
## Load the Dataset
<!-- #endregion -->

```python id="UOr4yHGc43Hs" outputId="435250ce-5df4-42e0-8ac1-e79f67d32cf8"
# attach the column names to the dataset
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

# KDDTrain+_2.csv & KDDTest+_2.csv are the datafiles without the last column about the difficulty score
# these have already been removed.
df = pd.read_csv("KDDTrain+_2.csv", header=None, names = col_names)
df_test = pd.read_csv("KDDTest+_2.csv", header=None, names = col_names)

# shape, this gives the dimensions of the dataset
print('Dimensions of the Training set:',df.shape)
print('Dimensions of the Test set:',df_test.shape)
```

<!-- #region id="J0nu6mfW43Hw" -->
## Sample view of the training dataset
<!-- #endregion -->

```python id="kW9XaJtS43Hx" outputId="b85d6dec-b2ca-422f-e113-a8073a59bc80"
# first five rows
df.head(5)
```

<!-- #region id="4pxGijm943H1" -->
## Statistical Summary
<!-- #endregion -->

```python id="AbtMqTAO43H2" outputId="9a0b572c-040d-4075-b254-8940617f9a90"
df.describe()
```

<!-- #region id="uq4flel_43H7" -->
## Label Distribution of Training and Test set
<!-- #endregion -->

```python id="_kFgoet143H8" outputId="ae359e55-b3b3-4837-86ec-6fbb686eff73"
print('Label distribution Training set:')
print(df['label'].value_counts())
print()
print('Label distribution Test set:')
print(df_test['label'].value_counts())
```

<!-- #region id="m9yqm_1443IA" -->
## Step 1: Data preprocessing:
One-Hot-Encoding (one-of-K) is used to to transform all categorical features into binary features. 
Requirement for One-Hot-encoding:
"The input to this transformer should be a matrix of integers, denoting the values taken on by categorical (discrete) features. The output will be a sparse matrix where each column corresponds to one possible value of one feature. It is assumed that input features take on values in the range [0, n_values)."

Therefore the features first need to be transformed with LabelEncoder, to transform every category to a number.
<!-- #endregion -->

<!-- #region id="iXrIRtyx43IA" -->
## Identify categorical features
<!-- #endregion -->

```python id="Gi5MNBz043IC" outputId="153cf1fc-9638-47b1-a73a-6981281a8606"
# colums that are categorical and not binary yet: protocol_type (column 2), service (column 3), flag (column 4).
# explore categorical features
print('Training set:')
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

#see how distributed the feature service is, it is evenly distributed and therefore we need to make dummies for all.
print()
print('Distribution of categories in service:')
print(df['service'].value_counts().sort_values(ascending=False).head())
```

```python id="xalSTsJ243IG" outputId="ca57aeb2-f474-4e05-e744-c4392b8a54f3"
# Test set
print('Test set:')
for col_name in df_test.columns:
    if df_test[col_name].dtypes == 'object' :
        unique_cat = len(df_test[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))
```

<!-- #region id="Uh7lWbgD43IJ" -->
Conclusion: Need to make dummies for all categories as the distribution is fairly even. In total: 3+70+11=84 dummies.

Comparing the results shows that the Test set has fewer categories (6), these need to be added as empty columns.
<!-- #endregion -->

<!-- #region id="H_MIE60x43IL" -->
## LabelEncoder
<!-- #endregion -->

<!-- #region id="6MCvLQaO43IL" -->
### Insert categorical features into a 2D numpy array
<!-- #endregion -->

```python id="eVbxDEv043IM" outputId="9f01d1b9-81d7-4ddf-8b8a-c05282d5e741"
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_columns=['protocol_type', 'service', 'flag']
# insert code to get a list of categorical columns into a variable, categorical_columns
categorical_columns=['protocol_type', 'service', 'flag'] 
 # Get the categorical values into a 2D numpy array
df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]
df_categorical_values.head()
```

<!-- #region id="R8BEZwEm43IR" -->
### Make column names for dummies
<!-- #endregion -->

```python id="8eBCEkZN43IS" outputId="5e23a936-d6e2-4e2f-c7a2-2d3201bae3bb"
# protocol type
unique_protocol=sorted(df.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
# service
unique_service=sorted(df.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
# flag
unique_flag=sorted(df.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]
# put together
dumcols=unique_protocol2 + unique_service2 + unique_flag2
print(dumcols)

#do same for test set
unique_service_test=sorted(df_test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2
```

<!-- #region id="gvkExAc243IZ" -->
## Transform categorical features into numbers using LabelEncoder()
<!-- #endregion -->

```python id="n4N6nNHB43Ia" outputId="a3a74aeb-967f-46a1-827a-427ff237b796"
df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)
print(df_categorical_values_enc.head())
# test set
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)
```

<!-- #region id="2QoRab9u43Id" -->
## One-Hot-Encoding
<!-- #endregion -->

```python id="in1kmSvl43Ie" outputId="dc3a2afc-c910-46c5-b415-55a00ed6bf85"
enc = OneHotEncoder()
df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)
# test set
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)

df_cat_data.head()
```

<!-- #region id="HHjqfSTP43Ii" -->
### Add 6 missing categories from train set to test set
<!-- #endregion -->

```python id="8x3tJcuE43Ij" outputId="76919ecb-4e09-483f-ee21-2a0cc840f23b"
trainservice=df['service'].tolist()
testservice= df_test['service'].tolist()
difference=list(set(trainservice) - set(testservice))
string = 'service_'
difference=[string + x for x in difference]
difference
```

```python id="118WdwZe43Im" outputId="c1bf4bfa-8dd5-49d2-f341-46eb6da16957"
for col in difference:
    testdf_cat_data[col] = 0

testdf_cat_data.shape
```

<!-- #region id="5FsBWzGe43Iq" -->
## Join encoded categorical dataframe with the non-categorical dataframe
<!-- #endregion -->

```python id="MYRs544a43Iq" outputId="8ea4b904-f8c8-4f25-c9db-80338c8295c7"
newdf=df.join(df_cat_data)
newdf.drop('flag', axis=1, inplace=True)
newdf.drop('protocol_type', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)
# test data
newdf_test=df_test.join(testdf_cat_data)
newdf_test.drop('flag', axis=1, inplace=True)
newdf_test.drop('protocol_type', axis=1, inplace=True)
newdf_test.drop('service', axis=1, inplace=True)
print(newdf.shape)
print(newdf_test.shape)
```

<!-- #region id="MKC4jReu43Iv" -->
- Split Dataset into 4 datasets for every attack category
- Rename every attack label: 0=normal, 1=DoS, 2=Probe, 3=R2L and 4=U2R.
- Replace labels column with new labels column
- Make new datasets

<!-- #endregion -->

```python id="54pY7N1R43Iw" outputId="ed059383-3294-44f9-db0f-75ae3a267142"
# take label column
labeldf=newdf['label']
labeldf_test=newdf_test['label']
# change the label column
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
# put the new label column back
newdf['label'] = newlabeldf
newdf_test['label'] = newlabeldf_test
print(newdf['label'].head())
```

```python id="Tq3bQdpl43Iz" outputId="ced78a2d-ef44-4c5d-d881-318d320a0ff5"
to_drop_DoS = [2,3,4]
to_drop_Probe = [1,3,4]
to_drop_R2L = [1,2,4]
to_drop_U2R = [1,2,3]
DoS_df=newdf[~newdf['label'].isin(to_drop_DoS)];
Probe_df=newdf[~newdf['label'].isin(to_drop_Probe)];
R2L_df=newdf[~newdf['label'].isin(to_drop_R2L)];
U2R_df=newdf[~newdf['label'].isin(to_drop_U2R)];

#test
DoS_df_test=newdf_test[~newdf_test['label'].isin(to_drop_DoS)];
Probe_df_test=newdf_test[~newdf_test['label'].isin(to_drop_Probe)];
R2L_df_test=newdf_test[~newdf_test['label'].isin(to_drop_R2L)];
U2R_df_test=newdf_test[~newdf_test['label'].isin(to_drop_U2R)];
print('Train:')
print('Dimensions of DoS:' ,DoS_df.shape)
print('Dimensions of Probe:' ,Probe_df.shape)
print('Dimensions of R2L:' ,R2L_df.shape)
print('Dimensions of U2R:' ,U2R_df.shape)
print('Test:')
print('Dimensions of DoS:' ,DoS_df_test.shape)
print('Dimensions of Probe:' ,Probe_df_test.shape)
print('Dimensions of R2L:' ,R2L_df_test.shape)
print('Dimensions of U2R:' ,U2R_df_test.shape)
```

<!-- #region id="TtX2Dimo43I2" -->
# Step 2: Feature Scaling:
<!-- #endregion -->

```python id="wwjTu3mJ43I3"
# Split dataframes into X & Y
# assign X as a dataframe of feautures and Y as a series of outcome variables
X_DoS = DoS_df.drop('label',1)
Y_DoS = DoS_df.label
X_Probe = Probe_df.drop('label',1)
Y_Probe = Probe_df.label
X_R2L = R2L_df.drop('label',1)
Y_R2L = R2L_df.label
X_U2R = U2R_df.drop('label',1)
Y_U2R = U2R_df.label
# test set
X_DoS_test = DoS_df_test.drop('label',1)
Y_DoS_test = DoS_df_test.label
X_Probe_test = Probe_df_test.drop('label',1)
Y_Probe_test = Probe_df_test.label
X_R2L_test = R2L_df_test.drop('label',1)
Y_R2L_test = R2L_df_test.label
X_U2R_test = U2R_df_test.drop('label',1)
Y_U2R_test = U2R_df_test.label
```

<!-- #region id="kLOaA_Ke43I5" -->
### Save a list of feature names for later use (it is the same for every attack category). Column names are dropped at this stage.
<!-- #endregion -->

```python id="_GoLWgI_43I6"
colNames=list(X_DoS)
colNames_test=list(X_DoS_test)
```

<!-- #region id="lI8HLDOg43I-" -->
## Use StandardScaler() to scale the dataframes
<!-- #endregion -->

```python id="-qLX4PWD43I-"
from sklearn import preprocessing
scaler1 = preprocessing.StandardScaler().fit(X_DoS)
X_DoS=scaler1.transform(X_DoS) 
scaler2 = preprocessing.StandardScaler().fit(X_Probe)
X_Probe=scaler2.transform(X_Probe) 
scaler3 = preprocessing.StandardScaler().fit(X_R2L)
X_R2L=scaler3.transform(X_R2L) 
scaler4 = preprocessing.StandardScaler().fit(X_U2R)
X_U2R=scaler4.transform(X_U2R) 
# test data
scaler5 = preprocessing.StandardScaler().fit(X_DoS_test)
X_DoS_test=scaler5.transform(X_DoS_test) 
scaler6 = preprocessing.StandardScaler().fit(X_Probe_test)
X_Probe_test=scaler6.transform(X_Probe_test) 
scaler7 = preprocessing.StandardScaler().fit(X_R2L_test)
X_R2L_test=scaler7.transform(X_R2L_test) 
scaler8 = preprocessing.StandardScaler().fit(X_U2R_test)
X_U2R_test=scaler8.transform(X_U2R_test) 
```

<!-- #region id="ZOJ353-k43JB" -->
### Check that the Standard Deviation is 1
<!-- #endregion -->

```python id="u7Bku2O843JC" outputId="cec9b1d8-8929-4c75-ec29-cb6c9006c48c"
print(X_DoS.std(axis=0))
```

```python id="RFzVgFFh43JE"
X_Probe.std(axis=0);
X_R2L.std(axis=0);
X_U2R.std(axis=0);
```

<!-- #region id="MisiuRS143JH" -->
# Step 3: Feature Selection:
<!-- #endregion -->

<!-- #region id="Q-U6BTvl43JH" -->
# 1. Univariate Feature Selection using ANOVA F-test
<!-- #endregion -->

```python id="PEXtpidi43JI" outputId="dbba2677-045b-4c67-b6fb-324ed83c081e"
#univariate feature selection with ANOVA F-test. using secondPercentile method, then RFE
#Scikit-learn exposes feature selection routines as objects that implement the transform method
#SelectPercentile: removes all but a user-specified highest scoring percentage of features
#f_classif: ANOVA F-value between label/feature for classification tasks.
from sklearn.feature_selection import SelectPercentile, f_classif
np.seterr(divide='ignore', invalid='ignore');
selector=SelectPercentile(f_classif, percentile=10)
X_newDoS = selector.fit_transform(X_DoS,Y_DoS)
X_newDoS.shape
```

<!-- #region id="5V5AzsSZ43JK" -->
### Get the features that were selected: DoS
<!-- #endregion -->

```python id="alj46et043JL" outputId="081555fd-ad8a-4d80-fbbd-9a114064f55a"
true=selector.get_support()
newcolindex_DoS=[i for i, x in enumerate(true) if x]
newcolname_DoS=list( colNames[i] for i in newcolindex_DoS )
newcolname_DoS
```

```python id="LTLOh6Zu43JO" outputId="23020b8b-993d-4e45-d39f-ea379143c576"
X_newProbe = selector.fit_transform(X_Probe,Y_Probe)
X_newProbe.shape
```

<!-- #region id="_iCVfhuU43JR" -->
### Get the features that were selected: Probe
<!-- #endregion -->

```python id="eDpUyX2443JS" outputId="c6388e2e-5710-4b3e-da36-4d3181d82ff1"
true=selector.get_support()
newcolindex_Probe=[i for i, x in enumerate(true) if x]
newcolname_Probe=list( colNames[i] for i in newcolindex_Probe )
newcolname_Probe
```

```python id="3dH2EBt243JV" outputId="c446c712-7828-4aad-fb2a-81a206538040"
X_newR2L = selector.fit_transform(X_R2L,Y_R2L)
X_newR2L.shape
```

<!-- #region id="gCkrifjx43JY" -->
### Get the features that were selected: R2L
<!-- #endregion -->

```python id="uGhXwApB43JY" outputId="617bf27d-137e-4cd2-c4f2-8a03ad154b0c"
true=selector.get_support()
newcolindex_R2L=[i for i, x in enumerate(true) if x]
newcolname_R2L=list( colNames[i] for i in newcolindex_R2L)
newcolname_R2L
```

```python id="18a9rYHb43Je" outputId="51ae2e43-222a-43c7-f0ec-73014795b77d"
X_newU2R = selector.fit_transform(X_U2R,Y_U2R)
X_newU2R.shape
```

<!-- #region id="ReCZtuaL43Ji" -->
### Get the features that were selected: U2R
<!-- #endregion -->

```python id="43kDHjgf43Ji" outputId="d5a53ea8-3bf7-456e-e5d1-603ac362b53b"
true=selector.get_support()
newcolindex_U2R=[i for i, x in enumerate(true) if x]
newcolname_U2R=list( colNames[i] for i in newcolindex_U2R)
newcolname_U2R
```

<!-- #region id="PRxE2ef-43Jl" -->
# Summary of features selected by Univariate Feature Selection
<!-- #endregion -->

```python id="bkC4btyC43Jl" outputId="cc4c1676-f071-48a3-b9a2-d459f7c6dda5"
print('Features selected for DoS:',newcolname_DoS)
print()
print('Features selected for Probe:',newcolname_Probe)
print()
print('Features selected for R2L:',newcolname_R2L)
print()
print('Features selected for U2R:',newcolname_U2R)
```

<!-- #region id="pgWe1YZM43Jn" -->
## The authors state that "After obtaining the adequate number of features during the univariate selection process, a recursive feature elimination (RFE) was operated with the number of features passed as parameter to identify the features selected". This either implies that RFE is only used for obtaining the features previously selected but also obtaining the rank. This use of RFE is however very redundant as the features selected can be obtained in another way (Done in this project). One can also not say that the features were selected by RFE, as it was not used for this. The quote could however also imply that only the number 13 from univariate feature selection was used. RFE is then used for feature selection trying to find the best 13 features. With this use of RFE one can actually say that it was used for feature selection. However the authors obtained different numbers of features for every attack category, 12 for DoS, 15 for Probe, 13 for R2L and 11 for U2R. This concludes that it is not clear what mechanism is used for feature selection. 

## To procede with the data mining, the second option is considered as this uses RFE. From now on the number of features for every attack category is 13.
<!-- #endregion -->

<!-- #region id="QUi3iD_r43Jo" -->
# 2. Recursive Feature Elimination for feature ranking (Option 1: get importance from previous selected)
<!-- #endregion -->

```python id="059seeWI43Jp" outputId="ac70c406-efe0-48b7-8f1c-68f716ba6bf4"
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
# Create a decision tree classifier. By convention, clf means 'classifier'
clf = RandomForestClassifier(n_jobs=2)

#rank all features, i.e continue the elimination until the last one
rfe = RFE(clf, n_features_to_select=1)
rfe.fit(X_newDoS, Y_DoS)
print ("DoS Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_DoS)))
```

```python id="Wy3fKV_X43Jt" outputId="03735f5c-d390-40d5-aeb1-7272a6077f66"
rfe.fit(X_newProbe, Y_Probe)
print ("Probe Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_Probe)))
```

```python id="i9QJu96o43Jv" outputId="185a842c-f089-49e2-8291-0596117d1830"
rfe.fit(X_newR2L, Y_R2L)
 
print ("R2L Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_R2L)))
```

```python id="Ufw976Ne43Jz" outputId="f4a21ab3-4f38-4657-91f0-793138b5ad8e"
rfe.fit(X_newU2R, Y_U2R)
 
print ("U2R Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_U2R)))
```

<!-- #region id="E-ESkiow43J5" -->
# 2. Recursive Feature Elimination, select 13 features each of 122 (Option 2: get 13 best features from 122 from RFE)
<!-- #endregion -->

```python id="jCeKO4oV43J5"
from sklearn.feature_selection import RFE
clf = RandomForestClassifier(n_jobs=2)
rfe = RFE(estimator=clf, n_features_to_select=13, step=1)
rfe.fit(X_DoS, Y_DoS)
X_rfeDoS=rfe.transform(X_DoS)
true=rfe.support_
rfecolindex_DoS=[i for i, x in enumerate(true) if x]
rfecolname_DoS=list(colNames[i] for i in rfecolindex_DoS)
```

```python id="X6REn2wQ43J7"
rfe.fit(X_Probe, Y_Probe)
X_rfeProbe=rfe.transform(X_Probe)
true=rfe.support_
rfecolindex_Probe=[i for i, x in enumerate(true) if x]
rfecolname_Probe=list(colNames[i] for i in rfecolindex_Probe)
```

```python id="z4IzHcwj43J_"
rfe.fit(X_R2L, Y_R2L)
X_rfeR2L=rfe.transform(X_R2L)
true=rfe.support_
rfecolindex_R2L=[i for i, x in enumerate(true) if x]
rfecolname_R2L=list(colNames[i] for i in rfecolindex_R2L)
```

```python id="Mix_c6aN43KD"
rfe.fit(X_U2R, Y_U2R)
X_rfeU2R=rfe.transform(X_U2R)
true=rfe.support_
rfecolindex_U2R=[i for i, x in enumerate(true) if x]
rfecolname_U2R=list(colNames[i] for i in rfecolindex_U2R)
```

<!-- #region id="Pk9hbn5Y43KG" -->
# Summary of features selected by RFE
<!-- #endregion -->

```python id="9ghYFPwh43KI" outputId="3ee30fbd-7bc8-4efe-d7f2-70d72e6f1200"
print('Features selected for DoS:',rfecolname_DoS)
print()
print('Features selected for Probe:',rfecolname_Probe)
print()
print('Features selected for R2L:',rfecolname_R2L)
print()
print('Features selected for U2R:',rfecolname_U2R)
```

```python id="zqi0O8re43KM" outputId="d5317fb8-9311-4898-c0ce-b1dea7c8db2e"
print(X_rfeDoS.shape)
print(X_rfeProbe.shape)
print(X_rfeR2L.shape)
print(X_rfeU2R.shape)
```

<!-- #region id="lmmUD1mp43KQ" -->
## Step 4: Build the model:
### Classifier is trained for all features and for reduced features, for later comparison.
#### The classifier model itself is stored in the clf variable.
<!-- #endregion -->

```python id="h67xMXw643KR" outputId="24d5821a-26ec-4f08-fed5-df1ad86eb541"
# all features
clf_DoS=RandomForestClassifier(n_jobs=2)
clf_Probe=RandomForestClassifier(n_jobs=2)
clf_R2L=RandomForestClassifier(n_jobs=2)
clf_U2R=RandomForestClassifier(n_jobs=2)
clf_DoS.fit(X_DoS, Y_DoS)
clf_Probe.fit(X_Probe, Y_Probe)
clf_R2L.fit(X_R2L, Y_R2L)
clf_U2R.fit(X_U2R, Y_U2R)
```

```python id="PCQMakAu43KU" outputId="329945ab-d34b-44dd-e409-3795bb9545ae"
# selected features
clf_rfeDoS=RandomForestClassifier(n_jobs=2)
clf_rfeProbe=RandomForestClassifier(n_jobs=2)
clf_rfeR2L=RandomForestClassifier(n_jobs=2)
clf_rfeU2R=RandomForestClassifier(n_jobs=2)
clf_rfeDoS.fit(X_rfeDoS, Y_DoS)
clf_rfeProbe.fit(X_rfeProbe, Y_Probe)
clf_rfeR2L.fit(X_rfeR2L, Y_R2L)
clf_rfeU2R.fit(X_rfeU2R, Y_U2R)
```

<!-- #region id="dJis8mjA43KX" -->
## Step 5: Prediction & Evaluation (validation):
<!-- #endregion -->

<!-- #region id="0ep24_km43KX" -->
> Note: Using all Features for each category
<!-- #endregion -->

<!-- #region id="4xPEDeF_43KY" -->
## Confusion Matrices
## DoS
<!-- #endregion -->

```python id="3IcwvttU43KZ" outputId="c0bc8997-3027-4cb4-8bc4-e213df489a53"
# Apply the classifier we trained to the test data (which it has never seen before)
clf_DoS.predict(X_DoS_test)
```

```python id="8AfSpQxt43Kb" outputId="57f853b7-bc65-4d0e-dfb8-7fe73f7270e3"
# View the predicted probabilities of the first 10 observations
clf_DoS.predict_proba(X_DoS_test)[0:10]
```

```python id="9dQl_qGi43Kd" outputId="37a1d9ef-d6ff-4992-8c0a-61e6a7f966d1"
Y_DoS_pred=clf_DoS.predict(X_DoS_test)
# Create confusion matrix
pd.crosstab(Y_DoS_test, Y_DoS_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])
```

<!-- #region id="Cv3KybF143Ki" -->
## Probe
<!-- #endregion -->

```python id="y878UapA43Kj" outputId="ea5437cb-9154-4d07-81dd-1842ee7d56e9"
Y_Probe_pred=clf_Probe.predict(X_Probe_test)
# Create confusion matrix
pd.crosstab(Y_Probe_test, Y_Probe_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])
```

<!-- #region id="uBjqzEtM43Kn" -->
## R2L
<!-- #endregion -->

```python id="qj4L0So143Kn" outputId="935defa2-ef02-4f28-efb0-0a6354bf8039"
Y_R2L_pred=clf_R2L.predict(X_R2L_test)
# Create confusion matrix
pd.crosstab(Y_R2L_test, Y_R2L_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])
```

<!-- #region id="NrPmSwRM43Kx" -->
## U2R
<!-- #endregion -->

```python id="F0bcVPG743Ky" outputId="88bd91de-70ff-444b-93e1-cd36a3be3ee5"
Y_U2R_pred=clf_U2R.predict(X_U2R_test)
# Create confusion matrix
pd.crosstab(Y_U2R_test, Y_U2R_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])
```

<!-- #region id="y086HgaC43K3" -->
## Cross Validation: Accuracy, Precision, Recall, F-measure
<!-- #endregion -->

<!-- #region id="6mCZmy5v43K3" -->
## DoS
<!-- #endregion -->

```python id="bGV5Nx9n43K3" outputId="a1def5ed-1437-4483-b567-38d69b618dde"
from sklearn.model_selection import cross_val_score
from sklearn import metrics
accuracy = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='precision')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='recall')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='f1')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
```

<!-- #region id="euWk6wou43K5" -->
## Probe
<!-- #endregion -->

```python id="Xj7T8WSQ43K6" outputId="18074f5b-49c8-4b42-c2f3-0859d174ea3c"
accuracy = cross_val_score(clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='precision_macro')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='recall_macro')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='f1_macro')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
```

<!-- #region id="QS8nOIVJ43K-" -->
## R2L
<!-- #endregion -->

```python id="ytgJAuN043K_" outputId="bba64ecf-e9f0-4480-fee6-f226e06c5248"
accuracy = cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='precision_macro')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='recall_macro')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='f1_macro')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
```

<!-- #region id="PW5H19k-43LD" -->
## U2R
<!-- #endregion -->

```python id="yo1vWoli43LD" outputId="b7e3591b-1457-4f84-9c26-c586e90db8db"
accuracy = cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='precision_macro')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='recall_macro')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='f1_macro')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
```

<!-- #region id="LxmHnqOx43LH" -->
## RFECV for illustration
<!-- #endregion -->

```python id="WOmvXvY343LM" outputId="abc40f01-ca70-4623-c44e-7dd90d3fceba"
print(__doc__)

import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv_DoS = RFECV(estimator=clf_DoS, step=1, cv=10, scoring='accuracy')
rfecv_DoS.fit(X_DoS_test, Y_DoS_test)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.title('RFECV DoS')
plt.plot(range(1, len(rfecv_DoS.grid_scores_) + 1), rfecv_DoS.grid_scores_)
plt.show()
```

```python id="uryvbpyN43LT" outputId="07ae0413-0090-4472-d424-538810f15fe5"
rfecv_Probe = RFECV(estimator=clf_Probe, step=1, cv=10, scoring='accuracy')
rfecv_Probe.fit(X_Probe_test, Y_Probe_test)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.title('RFECV Probe')
plt.plot(range(1, len(rfecv_Probe.grid_scores_) + 1), rfecv_Probe.grid_scores_)
plt.show()
```

```python id="6nJ_n9X-43LX" outputId="ac2505c0-28f4-4b5b-c99a-8c270e2843c5"
rfecv_R2L = RFECV(estimator=clf_R2L, step=1, cv=10, scoring='accuracy')
rfecv_R2L.fit(X_R2L_test, Y_R2L_test)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.title('RFECV R2L')
plt.plot(range(1, len(rfecv_R2L.grid_scores_) + 1), rfecv_R2L.grid_scores_)
plt.show()
```

```python id="GkPffIhy43Lb" outputId="bb6a54f8-ca23-4cec-9d42-30d54f78298b"
rfecv_U2R = RFECV(estimator=clf_U2R, step=1, cv=10, scoring='accuracy')
rfecv_U2R.fit(X_U2R_test, Y_U2R_test)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.title('RFECV U2R')
plt.plot(range(1, len(rfecv_U2R.grid_scores_) + 1), rfecv_U2R.grid_scores_)
plt.show()
```

<!-- #region id="eklRCB3m43Ld" -->
> Note: Using 13 Features for each category
<!-- #endregion -->

<!-- #region id="rBM5SDC843Le" -->
## Confusion Matrices
## DoS
<!-- #endregion -->

```python id="B7xeq4Pn43Le" outputId="197b9743-51d5-476a-e763-44f5d674380a"
# reduce test dataset to 13 features, use only features described in rfecolname_DoS etc.
X_DoS_test2=X_DoS_test[:,rfecolindex_DoS]
X_Probe_test2=X_Probe_test[:,rfecolindex_Probe]
X_R2L_test2=X_R2L_test[:,rfecolindex_R2L]
X_U2R_test2=X_U2R_test[:,rfecolindex_U2R]
X_U2R_test2.shape
```

```python id="EJTwYTXx43Lg" outputId="f0bcd9eb-9f48-452d-bdd6-59b5fa44177a"
Y_DoS_pred2=clf_rfeDoS.predict(X_DoS_test2)
# Create confusion matrix
pd.crosstab(Y_DoS_test, Y_DoS_pred2, rownames=['Actual attacks'], colnames=['Predicted attacks'])
```

<!-- #region id="YbGLgWOw43Lj" -->
## Probe
<!-- #endregion -->

```python id="t8Q_oQr243Lk" outputId="bf7befce-06c3-48f9-9e2a-0b6f50190143"
Y_Probe_pred2=clf_rfeProbe.predict(X_Probe_test2)
# Create confusion matrix
pd.crosstab(Y_Probe_test, Y_Probe_pred2, rownames=['Actual attacks'], colnames=['Predicted attacks'])
```

<!-- #region id="H6scCTJv43Ln" -->
## R2L
<!-- #endregion -->

```python id="OLIoycOP43Ln" outputId="20e78e2e-402f-43b4-f1af-accfb3f42906"
Y_R2L_pred2=clf_rfeR2L.predict(X_R2L_test2)
# Create confusion matrix
pd.crosstab(Y_R2L_test, Y_R2L_pred2, rownames=['Actual attacks'], colnames=['Predicted attacks'])
```

<!-- #region id="MiQfOlfi43Lr" -->
## U2R
<!-- #endregion -->

```python id="gvzWwRnT43Ls" outputId="66a5e4ce-40bf-4b64-d8d5-dc8705f3f946"
Y_U2R_pred2=clf_rfeU2R.predict(X_U2R_test2)
# Create confusion matrix
pd.crosstab(Y_U2R_test, Y_U2R_pred2, rownames=['Actual attacks'], colnames=['Predicted attacks'])
```

<!-- #region id="XxHGrOV243Ly" -->
## Cross Validation: Accuracy, Precision, Recall, F-measure
<!-- #endregion -->

<!-- #region id="tpSpUeJf43Ly" -->
## DoS
<!-- #endregion -->

```python id="RyQb44QQ43Lz" outputId="1a703ed9-f681-4e2d-99e7-3b75d9e3121c"
accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='precision')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='recall')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='f1')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
```

<!-- #region id="oWSY50OD43MA" -->
## Probe
<!-- #endregion -->

```python id="n24l4FsR43MB" outputId="d6154c40-8a1f-4cf8-a524-c002ae933e79"
accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=10, scoring='precision_macro')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=10, scoring='recall_macro')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=10, scoring='f1_macro')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
```

<!-- #region id="QlemawJB43MD" -->
## R2L
<!-- #endregion -->

```python id="70a4p2I-43MD" outputId="68d4e4d7-89f0-4b3b-9984-4f275eb70409"
accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=10, scoring='precision_macro')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=10, scoring='recall_macro')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=10, scoring='f1_macro')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
```

<!-- #region id="sxHqBoRe43MF" -->
## U2R
<!-- #endregion -->

```python id="jUr0yeSN43MH" outputId="d1d6b1d0-fa6b-45c5-93f1-67899a0026ef"
accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=10, scoring='precision_macro')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=10, scoring='recall_macro')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=10, scoring='f1_macro')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
```

<!-- #region id="t1VduJNs43MJ" -->
## Stratified CV => Stays the same
<!-- #endregion -->

```python id="XBLIwnRx43MJ" outputId="7ca1829f-da1c-4dd9-d56d-aa3daf42b5d2"
from sklearn.model_selection import StratifiedKFold
accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=StratifiedKFold(10), scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="9BsiRzT_43ML" outputId="a70cba15-2380-4766-9f83-399e1c29d82a"
accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=StratifiedKFold(10), scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="LqGX9bMh43MN" outputId="002a5124-557c-4a61-a777-8622a75d644f"
accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=StratifiedKFold(10), scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="wMvp9cKQ43MP" outputId="07a11830-e43c-4af0-e662-f621a3805e57"
accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=StratifiedKFold(10), scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

<!-- #region id="GaNN6Ipx43MR" -->
## CV 2, 5, 10, 30, 50 fold
<!-- #endregion -->

<!-- #region id="RI2aGmOu43MS" -->
## DoS
<!-- #endregion -->

```python id="KLA9fpTi43MT" outputId="85a8f413-df02-4442-a303-2d853433b1af"
accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=2, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="OwH816gm43MV" outputId="6081a5b1-c5ed-4f14-ef21-ad2e8051f2fe"
accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=5, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="wTSwSO2Q43MY" outputId="a0b980e6-ebd2-4a64-ed0c-ff468595a032"
accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="62HQjYAn43Ma" outputId="303c1da6-e3af-469a-cd12-bf2f883a4b44"
accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=30, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="gJZxfoU543Mc" outputId="f75f78a7-1e2b-4557-91f8-c1947afd9c8b"
accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=50, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

<!-- #region id="zH_pM2Sh43Me" -->
## Probe
<!-- #endregion -->

```python id="kv8ZDx5643Me" outputId="22f4881d-64aa-4555-f186-2b469b1abcde"
accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=2, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="ireXRnON43Mg" outputId="9af01eaf-7f17-4bec-d87a-d0c822b68b8c"
accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=5, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="AnVzMUvh43Mi" outputId="cfc11d90-2d0f-4723-fb48-4e4150e1fdf2"
accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="B3hedeCL43Mk" outputId="0beb3ea4-e5ae-4105-f07c-ad805db92bd0"
accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=30, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="6aw963CX43Ml" outputId="93784360-aebf-4c3c-a166-20e64ee6f6d5"
accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=50, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

<!-- #region id="GCLMA0wi43Mn" -->
## R2L
<!-- #endregion -->

```python id="YNRSr9Cd43Mo" outputId="74c5666b-2e77-4da4-b1b6-2189a012ab84"
accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=2, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="NuGe_f3g43Mr" outputId="1093ff8a-914d-4828-b8fc-ca7a3d031873"
accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=5, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="_mu1L1qB43Ms" outputId="88347505-20a2-47fa-dbde-5f3244135249"
accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="xdIsKcTJ43Mu" outputId="f8392317-2373-4e53-d02c-7226c48d0eae"
accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=30, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="muBUWfth43Mx" outputId="eeaf208d-e2c7-4638-d36f-9487e1c823b3"
accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=50, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

<!-- #region id="vagrSRoG43Mz" -->
## U2R
<!-- #endregion -->

```python id="DZ7Izuw943Mz" outputId="a970c2bd-4d70-4f1e-b06c-ece9d9129e39"
accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=2, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="GhBPLc8_43M2" outputId="c357c18c-954e-4286-a54e-ed48b3f1c112"
accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=5, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="hysG4gnl43M4" outputId="60a7406d-caa1-4218-d21a-8e1ef1a25983"
accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="bXURwCur43M7" outputId="5c6143c0-b3ad-4891-ce53-a6bb790d670f"
accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=30, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="8WDQ-ELK43M8" outputId="65a3cac5-6c86-418c-cbf9-48dabc257794"
accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=50, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
```

```python id="d_c0Ax2F6TJw"

```

```python id="i-azh95o41t3"
import pandas as pd
import numpy as np
df_read =  pd.read_csv("./login_transformed.csv")
df=df_read
```

```python id="FgR77bYv41t7"
#Encode the categorical features
from sklearn.preprocessing import LabelEncoder

encs = dict()
for column in df.columns:
    if df[column].dtype == "object":
        encs[column] = LabelEncoder()
        df[column] = encs[column].fit_transform(df[column])
```

```python id="-jCC-ajh41t_" outputId="0317ded3-1724-4f86-aaa5-c1b2327873d2"
from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators=200, max_samples=300)
df.head()
```

```python id="EpKs9EiK41uD" outputId="e4166719-5e0d-4b5a-bd43-3ae078ec69be"
#Select the features you wan't to use by dropping all the other features:

df['hour_sin'] = np.sin(df.hour*(2.*np.pi/24))
df['hour_cos'] = np.cos(df.hour*(2.*np.pi/24))
df['month_sin'] = np.sin((df.month-1)*(2.*np.pi/12))
df['month_cos'] = np.cos((df.month-1)*(2.*np.pi/12))
df['weekday_sin'] = np.sin((df.weekday-1)*(2.*np.pi/12))
df['weekday_cos'] = np.cos((df.weekday-1)*(2.*np.pi/12))
df=df.drop('hour',1)
df=df.drop('month',1)
df=df.drop('weekday',1)

df=df.drop('id',1)
df=df.drop('day',1)
df=df.drop('user',1)
df=df.drop('pc',1)
df=df.drop('month_sin',1)
df=df.drop('month_cos',1)
df=df.drop('weekday_sin',1)
df=df.drop('weekday_cos',1)

df.head()
```

```python id="hVVKnfAJ41uH" outputId="10ea26f7-084e-4412-d2f9-3132f744f257"
from scipy import stats

print("STARTING")

#If you want to produce lots of data, loop:
#for _ in range(500):

train_data=df.drop('threat',1)

try:
    train_data=df.drop('anomaly_score',1)
except:
    print("No anomaly_score column!")

### Try Isolation forest with different nummer of features ###
### The training parameters, i.e n_estimators, max_samples should also be evaluated ###

# Train the model
clf.fit(train_data)

# Calculate the anomaly score
anomaly_score = clf.decision_function(train_data)

# Add the anomaly score to the data frame
df['anomaly_score'] = anomaly_score

acctual_false = df.loc[df.threat==0]    #Data frame with true negatives
acctual_true = df.loc[df.threat==1]     #Data frame with true positives

features=[]
for element in train_data.columns:
    features.append(element)

(ks_stat, pval) = stats.ks_2samp(acctual_true.anomaly_score, acctual_false.anomaly_score)

#Print the features used together with the KS stat and p-value
print(str(features)+";"+str(ks_stat)+";"+str(pval)+"\n")
```

```python id="MtbS83HT41uK" outputId="119e32e8-247f-450e-ef17-c43c319d939b"
#Plot the combined distribution of the scores 
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 4), dpi=600, facecolor='w', edgecolor='k')

normal = plt.hist(acctual_false.anomaly_score, 50,density=True)

plt.xlabel('Anomaly score')
plt.ylabel('Percentage')
plt.title("Distribution of anomaly score for non threats")
```

```python id="Vv9wsCs141uO" outputId="b951fddb-5ad7-430f-879e-f5b6fd292c17"
#Plot the combined distribution of the scores 
%matplotlib inline
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4), dpi=600, facecolor='w', edgecolor='k')

normal = plt.hist(acctual_true.anomaly_score, 50, density=True)

plt.xlabel('Anomaly score')
plt.ylabel('Percentage')
plt.title("Distribution of anomaly score for threats")

```

```python id="w69xICa641uR" outputId="87f8b1ae-f4d0-401c-e25d-4aa013b8a68c"
#Let's create some fancy plots!

import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
all_data = [acctual_true['anomaly_score'],acctual_false['anomaly_score']]

axes[0].set_title('Violin plot')
# plot violin plot
axes[0].violinplot(all_data,
                   showmeans=False,
                   showmedians=True)

# plot box plot
axes[1].boxplot(all_data)
axes[1].set_title('Box Plot')

for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(all_data))])
    ax.set_xlabel('Density', fontsize=20)
    ax.set_ylabel('Anomaly score', fontsize=20)

plt.setp(axes, xticks=[y+1 for y in range(len(all_data))],
         xticklabels=['Positives', 'Negatives'])

plt.show()

```

```python id="Eni-cm6J41uU" outputId="678aa571-af56-43e7-e5ac-dff28efdf0a7"
# Now let's create the confusion matrix!
# This code will try to find the best limit value to apply on the anomaly score. 
# All data points with a score lower than the limit will be considered as threats (positives).
# We use Matthews Correlation Coefficient to evaluate the results and pick the "best" limit.

from sklearn.metrics import matthews_corrcoef

def calculateMatthews(limit):
    #Predict all anomaly scores < limit as threats:
    df['pred_threat'] = np.where(df.anomaly_score < limit, 1, 0)
    
    #Evaluate..
    
    total_negatives = len(acctual_false.index)
    total_positives = len(acctual_true.index)

    true_positives = acctual_true[(acctual_true['anomaly_score'] < limit)]
    false_positives = acctual_false[(acctual_false['anomaly_score'] < limit)]

    false_negatives = acctual_true[(acctual_true['anomaly_score'] >= limit)]
    true_negatives = acctual_false[(acctual_false['anomaly_score'] >= limit)]

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)
    tn = len(true_negatives)

    tp_rate = tp/total_positives
    fp_rate = fp/total_negatives
    fn_rate = fn/total_positives
    tn_rate = tn/total_negatives
    
    if(tp + fp == 0):
        precision = 0
    else:
        precision = tp/(tp + fp)
        
    if(tp + fn == 0):
        recall = 0
    else:
        recall = tp/(tp + fn)        

    predicted_true = df.loc[df.anomaly_score < limit]

    mcc = matthews_corrcoef(df['threat'], df['pred_threat'])
    
    #print("tp_rate: " + str(tp_rate))
    #print("fp_rate: " + str(fp_rate))
    #print("fn_rate: " + str(fn_rate))
    #print("tn_rate: " + str(tn_rate))      
    #print("mcc "+str(mcc))
    return(mcc, precision, recall, tp_rate, fp_rate, fn_rate, tn_rate)


#Place the upper and lower limit (start/stop) just below and above the min and max anomaly score value.
lower_limit = -0.25

lower_limit = df.anomaly_score.min()-0.05
upper_limit = df.anomaly_score.max()+0.05

max_mat=0.0
max_limit=-100.0
best_precision = 0
best_recall = 0
best_tp = 0
best_fp = 0
best_tn = 0
best_tp = 0

limit=lower_limit
mat_array=[]
limit_array=[]
tp_array = []
fp_array = []
fn_array = []
tn_array = []

#Try a few different step sizes:
step_size = 0.011

#Try a lot of different limits to find the one that scores the highest MCC
while(limit < upper_limit):
    limit=limit+step_size
    (mat, precision, recall, tp_rate, fp_rate, fn_rate, tn_rate)=calculateMatthews(limit)
    
    print("MCC: "+str(mat)+"  Limit: "+str(limit))
    mat_array.append(mat)
    limit_array.append(limit)
    tp_array.append(tp_rate)
    fp_array.append(fp_rate)
    fn_array.append(fn_rate)
    tn_array.append(tn_rate)
    if abs(mat) > max_mat:
        max_mat = mat
        max_limit = limit
        best_precision = precision
        best_tp = tp_rate
        best_fp = fp_rate
        best_tn = tn_rate
        best_fn = fn_rate
        best_recall = recall
        
print("Best limit    : " + str(max_limit))
print("Best MCC      : " + str(max_mat))
print("Best Precision: " + str(best_precision))
print("Best Recall   : " + str(best_recall))
print("Best TN       : " + str(best_tn))
print("Best TP       : " + str(best_tp))
print("Best FN       : " + str(best_fn))
print("Best FP       : " + str(best_fp))
```

```python id="AgSrzjf841uY" outputId="31005bda-0fe3-490c-9ebe-c41e8a86ffb0"
fig = plt.figure(figsize=(8,3),dpi=600,facecolor='w', edgecolor='k')
plt.xlabel('Anomaly score limit')
plt.ylabel('MCC score')
plt.title('')
plt.plot(limit_array, mat_array)
#Highlight the best limit manually..
plt.axvspan(max_limit, max_limit, color='red', alpha=0.6)
plt.show()
```

```python id="g2IhiJXv41ub" outputId="67b49580-8bee-4f32-acbe-3206b42cd6ef"
#Plot true positive rate vs false positive rate

fig = plt.figure(figsize=(8,3),dpi=600,facecolor='w', edgecolor='k')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive Rate')
plt.title('')
plt.axvspan(best_fp, best_fp, color='red', alpha=0.6)
plt.plot(fp_array, tp_array)
plt.show()

```

```python id="6S2jCxJU6pMG"
from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D, Activation
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.utils import np_utils
import keras.backend as K
import numpy as np
import tensorflow as tf
import keras.callbacks
import sys
import os
import cPickle as pickle
from timeit import default_timer as timer
import glob
import time

LSTM_UNITS = 92
MINI_BATCH = 10
TRAIN_STEPS_PER_EPOCH = 12000
VALIDATION_STEPS_PER_EPOCH = 800
DATA_DIR = '/root/data/PreprocessedISCX2012_5class_pkl/'
CHECKPOINTS_DIR = './iscx2012_cnn_rnn_5class_new_checkpoints/'

dict_5class = {0:'Normal', 1:'BFSSH', 2:'Infilt', 3:'HttpDoS', 4:'DDoS'}

def update_confusion_matrix(confusion_matrix, actual_lb, predict_lb):
    for idx, value in enumerate(actual_lb):
        p_value = predict_lb[idx]
        confusion_matrix[value, p_value] += 1
    return confusion_matrix

# function: find an element in a list
def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1

def truncate(f, n):
    trunc_f = np.math.floor(f * 10 ** n) / 10 ** n
    return '{:.2f}'.format(trunc_f) # only for 0.0 => 0.00

def binarize(x, sz=256):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 256

def byte_block(in_layer, nb_filter=(64, 100), filter_length=(3, 3), subsample=(2, 1), pool_length=(2, 2)):
    block = in_layer
    for i in range(len(nb_filter)):
        block = Conv1D(filters=nb_filter[i],
                       kernel_size=filter_length[i],
                       padding='valid',
                       activation='tanh',
                       strides=subsample[i])(block)
        # block = BatchNormalization()(block)
        # block = Dropout(0.1)(block)
        if pool_length[i]:
            block = MaxPooling1D(pool_size=pool_length[i])(block)

    # block = Lambda(max_1d, output_shape=(nb_filter[-1],))(block)
    block = GlobalMaxPool1D()(block)
    block = Dense(128, activation='relu')(block)
    return block

def mini_batch_generator(sessions, labels, indices, batch_size):
    Xbatch = np.ones((batch_size, PACKET_NUM_PER_SESSION, PACKET_LEN), dtype=np.int64) * -1
    Ybatch = np.ones((batch_size,5), dtype=np.int64) * -1
    batch_idx = 0
    while True:
        for idx in indices:
            for i, packet in enumerate(sessions[idx]):
                if i < PACKET_NUM_PER_SESSION:
                    for j, byte in enumerate(packet[:PACKET_LEN]):
                        Xbatch[batch_idx, i, (PACKET_LEN - 1 - j)] = byte            
            Ybatch[batch_idx] = np_utils.to_categorical(labels[idx], num_classes=5)[0]
            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0
                yield (Xbatch, Ybatch)

# read argv
print ("Script name: %s" % str(sys.argv[0]))
checkpoint = None
if len(sys.argv) == 2:
    if os.path.exists(str(sys.argv[1])):
        print ("Checkpoint : %s" % str(sys.argv[1]))
        checkpoint = str(sys.argv[1])        
# if len(sys.argv) == 5:
#     if os.path.exists(str(sys.argv[4])):
#         print ("Checkpoint : %s" % str(sys.argv[4]))
#         checkpoint = str(sys.argv[4])

# read sessions and labels from pickle files
sessions = []
labels = []
t1 = timer()
num_pkls = len(glob.glob(DATA_DIR + 'ISCX2012_labels_*.pkl'))
for i in range(num_pkls):
    session_pkl = DATA_DIR + 'ISCX2012_pcaps_' + str(i) + '.pkl'
    session_lists  = pickle.load(open(session_pkl, 'rb'))
    sessions.extend(session_lists)
    label_pkl = DATA_DIR + 'ISCX2012_labels_' + str(i) + '.pkl'
    label_lists = pickle.load(open(label_pkl, 'rb'))    
    labels.extend(label_lists)    
    print i
t2 = timer()
print t2 - t1
print('Sample doc{}'.format(sessions[1200]))
labels = np.array(labels)

# arg_list = [[50,6],[100,6],[200,6],[300,6],[400,6],[500,6],[600,6],[700,6],[800,6],[900,6],[1000,6],
#             [100,8],[100,10],[100,12],[100,14],[100,16],[100,18],[100,20],[100,22],[100,24],[100,26],[100,28],[100,30]]
# arg_list = [[100,12],[100,14],[100,16],[100,18],[100,20],[100,22],[100,24],[100,26],[100,28],[100,30]]
arg_list = [[600,14],[700,14],[800,14],[900,14],[1000,14]]
# arg_list = [[100,6]]
for arg in arg_list:
    PACKET_LEN = arg[0]
    PACKET_NUM_PER_SESSION = arg[1]
    TRAIN_EPOCHS = 8

    # create train/validate data generator 
    normal_indices = np.where(labels == 0)[0]
    attack_indices = [np.where(labels == i)[0] for i in range(1,5)]
    print len(normal_indices)
    print len(attack_indices)
    test_normal_indices = np.random.choice(normal_indices, int(len(normal_indices)*0.4))
    test_attack_indices = np.concatenate([np.random.choice(attack_indices[i], int(len(attack_indices[i])*0.4)) for i in range(4)])
    test_indices = np.concatenate([test_normal_indices, test_attack_indices]).astype(int)
    train_indices = np.array(list(set(np.arange(len(labels))) - set(test_indices)))
    train_data_generator  = mini_batch_generator(sessions, labels, train_indices, MINI_BATCH)
    val_data_generator    = mini_batch_generator(sessions, labels, test_indices, MINI_BATCH)
    test_data_generator   = mini_batch_generator(sessions, labels, test_indices, MINI_BATCH)

    # create model
    session = Input(shape=(PACKET_NUM_PER_SESSION, PACKET_LEN), dtype='int64')
    input_packet = Input(shape=(PACKET_LEN,), dtype='int64')
    embedded = Lambda(binarize, output_shape=binarize_outshape)(input_packet)
    block2 = byte_block(embedded, (128, 256), filter_length=(5, 5), subsample=(1, 1), pool_length=(2, 2))
    block3 = byte_block(embedded, (192, 320), filter_length=(7, 5), subsample=(1, 1), pool_length=(2, 2))
    packet_encode = concatenate([block2, block3], axis=-1)
    # packet_encode = Dropout(0.2)(packet_encode)
    encoder = Model(inputs=input_packet, outputs=packet_encode)
    encoder.summary()
    encoded = TimeDistributed(encoder)(session)
    lstm_layer = LSTM(LSTM_UNITS, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, implementation=0)(encoded)
    lstm_layer2 = LSTM(LSTM_UNITS, return_sequences=False, dropout=0.1, recurrent_dropout=0.1, implementation=0)(lstm_layer)
    dense_layer = Dense(5, name='dense_layer')(lstm_layer2)
    output = Activation('softmax')(dense_layer)
    model = Model(outputs=output, inputs=session)
    model.summary()

    # if input checkpoint, test with saved model and save predicted results
    if checkpoint:
        model.load_weights(checkpoint)
        sub_model = Model(inputs=model.input,
                        outputs=model.get_layer('dense_layer').output)
        test_steps = np.math.ceil(float(len(test_indices)) / MINI_BATCH)
        embd = sub_model.predict_generator(test_data_generator, steps=test_steps)
        print type(embd)
        print embd.shape
        print len(embd)
        print embd[0]
        np.save('./embeddings_iscx2012.npy', embd)
        np.save('./labels_iscx2012.npy', labels[test_indices])
        break

    # train and validate model
    script_name = os.path.basename(sys.argv[0]).split('.')[0]
    weight_file = CHECKPOINTS_DIR + script_name + '_' + str(PACKET_LEN) + '_' + str(PACKET_NUM_PER_SESSION) + '_{epoch:02d}_{val_loss:.2f}.hdf5'
    check_cb = keras.callbacks.ModelCheckpoint(weight_file, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min')
    earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    train_time = 0
    start_train = timer()
    model.fit_generator(
        generator=train_data_generator, 
        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
        epochs=TRAIN_EPOCHS,
        callbacks=[check_cb, earlystop_cb],
        validation_data=val_data_generator,
        validation_steps=VALIDATION_STEPS_PER_EPOCH)
    end_train = timer()
    train_time = end_train - start_train

    # test model
    start_test = timer()
    test_steps = np.math.ceil(float(len(test_indices)) / MINI_BATCH)
    predictions = model.predict_generator(test_data_generator, steps=test_steps)
    end_test = timer()
    test_time = end_test - start_test

    # stat and save
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = labels[test_indices]
    if len(predicted_labels) > len(true_labels):
        num_pad = len(predicted_labels) - len(true_labels)
        true_labels = np.concatenate([true_labels, true_labels[0:num_pad]])
    print len(predicted_labels)
    print len(true_labels)
    len_test = len(true_labels)
    cf_ma = np.zeros((5,5), dtype=int)
    update_confusion_matrix(cf_ma, true_labels, predicted_labels)
    metrics_list = []
    for i in range(5):
        if i == 0:
            metrics_list.append([dict_5class[i], str(i), str(cf_ma[i,0]), str(cf_ma[i,1]), str(cf_ma[i,2]), str(cf_ma[i,3]), str(cf_ma[i,4]), '--', '--', '--'])
        else:
            acc = truncate((float(len_test-cf_ma[:,i].sum()-cf_ma[i,:].sum()+cf_ma[i,i]*2)/len_test)*100, 2)
            tpr = truncate((float(cf_ma[i,i])/cf_ma[i].sum())*100, 2)
            fpr = truncate((float(cf_ma[0,i])/cf_ma[0].sum())*100, 2)
            metrics_list.append([dict_5class[i], str(i), str(cf_ma[i,0]), str(cf_ma[i,1]), str(cf_ma[i,2]), str(cf_ma[i,3]), str(cf_ma[i,4]), str(acc), str(tpr), str(fpr)])
    overall_acc = truncate((float(cf_ma[0,0]+cf_ma[1,1]+cf_ma[2,2]+cf_ma[3,3]+cf_ma[4,4])/len_test)*100, 2)
    overall_tpr = truncate((float(cf_ma[1,1]+cf_ma[2,2]+cf_ma[3,3]+cf_ma[4,4])/cf_ma[1:].sum())*100, 2)
    overall_fpr = truncate((float(cf_ma[0,1:].sum())/cf_ma[0,:].sum())*100, 2)
    with open('iscx12_cnn_rnn_5class_new.txt','a') as f:
        f.write("\n")
        t = time.strftime('%Y-%m-%d %X',time.localtime())
        f.write(t + "\n")
        f.write('CLASS_NUM: 5\n')
        f.write('PACKET_LEN: ' + str(PACKET_LEN) + "\n")
        f.write('PACKET_NUM_PER_SESSION: ' + str(PACKET_NUM_PER_SESSION) + "\n")
        f.write('MINI_BATCH: ' + str(MINI_BATCH) + "\n")
        f.write('TRAIN_EPOCHS: ' + str(TRAIN_EPOCHS) + "\n")
        f.write('DATA_DIR: ' + DATA_DIR + "\n")
        f.write("label\tindex\t0\t1\t2\t3\t4\tACC\tTPR\tFPR\n")
        for metrics in metrics_list:
            f.write('\t'.join(metrics) + "\n")
        f.write('Overall accuracy: ' + str(overall_acc) + "\n")
        f.write('Overall TPR: ' + str(overall_tpr) + "\n")
        f.write('Overall FPR: ' + str(overall_fpr) + "\n")
        f.write('Train time(second): ' + str(int(train_time)) + "\n")
        f.write('Test time(second): ' + str(int(test_time)) + "\n\n")
```
