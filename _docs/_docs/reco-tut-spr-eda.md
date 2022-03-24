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

```python colab={"base_uri": "https://localhost:8080/"} id="PYvHGli8ukum" executionInfo={"status": "ok", "timestamp": 1627923278562, "user_tz": -330, "elapsed": 430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="13ebf5a3-cc81-48f5-a72e-da8c43407b2c"
import os
project_name = "reco-tut-spr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "recotut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="nz2yphhRkbhY"
!git status
!git add . && git commit -m 'commit' && git push origin main
```

```python id="4DRkSJS6doCK" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1627922633540, "user_tz": -330, "elapsed": 15555, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cf307ca9-44dd-4abf-cc81-1e835ead723e"
!pip install -q git+https://github.com/sparsh-ai/recochef
```

```python id="arnAepMAeH0m" executionInfo={"status": "ok", "timestamp": 1627923363965, "user_tz": -330, "elapsed": 477, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from recochef.datasets.santander import Santander

%reload_ext google.colab.data_table
```

```python id="0ZZNOmlKePE4" executionInfo={"status": "ok", "timestamp": 1627923283488, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
data = Santander()
```

```python colab={"base_uri": "https://localhost:8080/"} id="zdwamklYgkg4" executionInfo={"status": "ok", "timestamp": 1627923436079, "user_tz": -330, "elapsed": 20644, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="29b87f97-8d5d-47f4-a070-a663e93d0536"
%%time
# train = data.load_train()
train = pd.read_parquet('train.parquet.gz')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 309} id="Zf568IexxqeI" executionInfo={"status": "ok", "timestamp": 1627923436082, "user_tz": -330, "elapsed": 29, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ddf58b88-a2e0-466e-d9f4-257e1749f20b"
train.head()
```

<!-- #region id="PJsQYXmBgpsL" -->
Let's rename all the column names with english name to understand what's going on...
<!-- #endregion -->

```python id="COrnc_76hW2N" executionInfo={"status": "ok", "timestamp": 1627923436084, "user_tz": -330, "elapsed": 26, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train.columns = ['Month_status_date', 'Customer_ID', 'Employee_Index', 'Customer_country', 'Sex', 'Age', 'Join_date',
                'New_customer', 'Relnshp_Mnths', 'Relnshp_flag','Last_date_Prim_Cust', 'Cust_type_beg_Mth', 'Cust_Reln_type_beg_mth',
                'Residence_flag', 'Forigner_flag', 'Emp_spouse_flag', 'Channel_when_joined', 'Deceased_flag', 
                'Address_type', 'Customer_address', 'Address_detail', 'Activity_flag', 'Gross_household_income',
                'Segment', 'Saving_account', 'Guarantees', 'Cur_account', 'Derivative_account', 'Payroll_account',
                'Junior_account', 'Particular_acct1', 'Particular_acct2', 'Particular_acct3', 'Short_term_deposites',
                'Med_term_deposites', 'Long_term_deposites', 'e-account', 'Funds', 'Mortgage', 'Pension', 'Loans',
                'Taxes', 'Credit_card', 'Securities', 'Home_account', 'Payroll', 'Pensions', 'Direct_debit']
```

```python colab={"base_uri": "https://localhost:8080/"} id="OaEJ7mgIgogQ" executionInfo={"status": "ok", "timestamp": 1627919099382, "user_tz": -330, "elapsed": 736, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6f54c747-fde4-475c-84b7-1f0951516194"
train.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 640} id="WxiFBEgehcvs" executionInfo={"status": "ok", "timestamp": 1627919146282, "user_tz": -330, "elapsed": 899, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2ce885ad-26bb-4721-b619-f28ea87577ea"
desc = train.describe()
desc.loc['Unique'] = [len(train[col].unique()) for col in desc.columns]
desc.loc["Missing"] = [train[col].isnull().sum() for col in desc.columns]
desc.loc['Datatype'] = [train[col].dtype for col in desc.columns]
desc.T
```

```python id="9SxG84Dmx-IG" executionInfo={"status": "ok", "timestamp": 1627923307956, "user_tz": -330, "elapsed": 1858, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# test = data.load_test()
test = pd.read_parquet('test.parquet.gz')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 292} id="5KKRZR0eh0y5" executionInfo={"status": "ok", "timestamp": 1627923163313, "user_tz": -330, "elapsed": 28, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9524050d-9e99-4fe9-f1a4-3cb0b00db2de"
test.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="_z5B02Djktj6" executionInfo={"status": "ok", "timestamp": 1627923307958, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7dde3101-90d4-4d2c-a89a-3bab475a3781"
test.columns = ['Month_status_date', 'Customer_ID', 'Employee_Index', 'Customer_country', 'Sex', 'Age', 'Join_date',
                'New_customer', 'Relnshp_Mnths', 'Relnshp_flag','Last_date_Prim_Cust', 'Cust_type_beg_Mth', 'Cust_Reln_type_beg_mth',
                'Residence_flag', 'Forigner_flag', 'Emp_spouse_flag', 'Channel_when_joined', 'Deceased_flag', 
                'Address_type', 'Customer_address', 'Address_detail', 'Activity_flag', 'Gross_household_income',
                'Segment']

desc = test.describe()
desc.loc['Unique'] = [len(test[col].unique()) for col in desc.columns]
desc.loc["Missing"] = [test[col].isnull().sum() for col in desc.columns]
desc.loc['Datatype'] = [test[col].dtype for col in desc.columns]
desc.T
```

<!-- #region id="W0iiBpbhk1r9" -->
> Note: we have far less numeric features in test data. This is because we do not have any of the 24 products information in test data, as the objective of the project is to predict the products a customer is going to buy
<!-- #endregion -->

```python id="JSTaWABEk2oW" executionInfo={"status": "ok", "timestamp": 1627923307959, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
eng_num_features = ['Employee_Index','Age','New_customer', 'Relnshp_Mnths', 'Relnshp_flag','Residence_flag',
                    'Forigner_flag', 'Emp_spouse_flag','Deceased_flag', 'Activity_flag', 'Gross_household_income']

eng_target_features = ['Saving_account', 'Guarantees', 'Cur_account', 'Derivative_account', 'Payroll_account',
                'Junior_account', 'Particular_acct1', 'Particular_acct2', 'Particular_acct3', 'Short_term_deposites',
                'Med_term_deposites', 'Long_term_deposites', 'e-account', 'Funds', 'Mortgage', 'Pension', 'Loans',
                'Taxes', 'Credit_card', 'Securities', 'Home_account', 'Payroll', 'Pensions', 'Direct_debit']

span_eng_feat_dict = {'fecha_dato': 'Month_status_date', 'ncodpers': 'Customer_ID', 'ind_empleado': 'Employee_Index',
                     'pais_residencia':'Customer_country', 'sexo': 'Sex', 'age': 'Age', 'fecha_alta': 'Join_date',
                     'ind_nuevo': 'New_customer', 'antiguedad':'Relnshp_Mnths', 'indrel': 'Relnshp_flag',
                     'ult_fec_cli_1t': 'Last_date_Prim_Cust', 'indrel_1mes': 'Cust_type_beg_Mth', 'tiprel_1mes':'Cust_Reln_type_beg_mth',
                     'indresi': 'Residence_flag', 'indext': 'Forigner_flag', 'conyuemp': 'Emp_spouse_flag', 'canal_entrada':'Channel_when_joined',
                     'indfall': 'Deceased_flag','tipodom':'Address_type', 'cod_prov':'Customer_address','nomprov': 'Address_detail', 
                     'ind_actividad_cliente': 'Activity_flag', 'renta': 'Gross_household_income', 'segmento' :'Segment' }
```

```python colab={"base_uri": "https://localhost:8080/"} id="V-I4T84dlHvR" executionInfo={"status": "ok", "timestamp": 1627919849517, "user_tz": -330, "elapsed": 1794, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e81d9f9d-4ce1-4f92-8559-cd8c342a4f9c"
print("Unique customers in train:", len(train['Customer_ID'].unique()))
print("Unique customers in test:", len(test['Customer_ID'].unique()))
print("Common customers in train and test:", len(set(train['Customer_ID'].unique()).intersection(set(test['Customer_ID'].unique()))))
```

<!-- #region id="hDS40PvvlUVA" -->
> Tip: Happy to see that every customer in test is also there in train data
<!-- #endregion -->

<!-- #region id="BI2D3sVfyQv2" -->
Let's first take a random sample, because it would be hard to do eda with the full data
<!-- #endregion -->

```python id="7kUfUrXNyZPD" executionInfo={"status": "ok", "timestamp": 1627923439810, "user_tz": -330, "elapsed": 3749, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train = train.sample(frac=0.1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="hUGRhWqkyvbh" executionInfo={"status": "ok", "timestamp": 1627923439815, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7329e083-1e80-45ef-fef3-00e9cdd2f015"
gc.collect()
```

<!-- #region id="RE_WG3KjlVt7" -->
> Note: Let's first explore all numeric features
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="rfCSHXgxlkQZ" executionInfo={"status": "ok", "timestamp": 1627923439816, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="da547704-afad-4556-917c-d1c1a6a7aed4"
train["Age"] = train["Age"].replace(to_replace = ' NA', value = np.nan)
train["Age"] = train["Age"].astype("float")
train["Age"].isnull().sum()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 223} id="lG3rYgwzluLh" executionInfo={"status": "ok", "timestamp": 1627923441097, "user_tz": -330, "elapsed": 1298, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="57464d5e-9aff-4d46-c160-059abb97698f"
age_series = train.Age.value_counts()
plt.figure(figsize=(20,4), dpi=80)
sns.barplot(age_series.index.astype('int'), age_series.values)
plt.ylabel('Number of customers', fontsize=12)
plt.xlabel('Age', fontsize=10)
plt.show()
```

<!-- #region id="cWrsgUqBmFGI" -->
**Observations**

- We have a bimodal distribution for the age. Let's see if we can find any reason for this.
- Also we have customer ages from 0 to 164.
- Looks like there might be some products for small children under 18, some product for young generation.
- It is not possible to have customers having age 164.. Lets cap the age at 100.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 256} id="4-6xB7MCnbkz" executionInfo={"status": "ok", "timestamp": 1627923442062, "user_tz": -330, "elapsed": 974, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="64eac178-b010-4514-9fa0-24383f5b416e"
#Age vs Segment
train.groupby(["Segment", "Age"])["Customer_ID"].nunique("Customer_ID").unstack()
```

<!-- #region id="x0fuFDAInf8z" -->
Looks like only **PARTICULARS** segment has age group <16. This segment might be served some specific products...

Lets look at segments for young generations only
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 152} id="oobJpcd3oDdL" executionInfo={"status": "ok", "timestamp": 1627923461447, "user_tz": -330, "elapsed": 686, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ec2ee4a9-8a00-4a0c-db6b-f12c89206ed0"
young = train[(train["Age"] > 18) & (train["Age"] < 30)]
young.groupby(["Segment", "Age"])["Customer_ID"].nunique("Customer_ID").unstack()
```

<!-- #region id="qxJPeG4hpg7J" -->
Let's have a look at the box plot...
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="bys9ZsB8pj-w" executionInfo={"status": "ok", "timestamp": 1627923464680, "user_tz": -330, "elapsed": 1614, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3c9c9fb8-20f7-4fee-e60c-7460ffc171df"
sns.boxplot(train["Age"].values, train["Segment"])
plt.xlabel("Age")
plt.title("Age Box plot")
plt.show()
```

<!-- #region id="MvM27lkopdnP" -->
The customers in university segemnt seems to have median age of 24 years while other to 2 segments have median age of 46 and 52
<!-- #endregion -->

<!-- #region id="kyt7q4PzoNYX" -->
Looks like these are university students, as most of the young customers belong to university segment... It makes sense why we have a bimodal distribution. 3 things to notice here:-
1. We have some population under 18 having bank accounts. these may be students or junior account holders where there parents have created an account for them.
2. We seem to have a group of people between 18 and 30 who could be students or early job starters. This segment has very high number of people than working people.
3. There are some people with age 164. It's better to cap the age at 100.


<!-- #endregion -->

```python id="ohriMi7NobBy" executionInfo={"status": "ok", "timestamp": 1627923497497, "user_tz": -330, "elapsed": 508, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train.loc[train.Age > 100,"Age"] = train.loc[(train.Age >= 30) & (train.Age <= 100),"Age"].median(skipna=True)
train["Age"].fillna(train["Age"].mean(),inplace=True)
train["Age"] = train["Age"].astype(int)
```

<!-- #region id="ZeCuK3O_qiT2" -->
Let's have a look at the distribution of age vs all the products
<!-- #endregion -->

```python id="SadMWhHtqliR" colab={"base_uri": "https://localhost:8080/", "height": 720} executionInfo={"status": "ok", "timestamp": 1627923542928, "user_tz": -330, "elapsed": 8969, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5d7fd15a-1ef7-40ce-c649-da820c661270"
fig, ax = plt.subplots(4,6, figsize=(30,15))
fig.suptitle('Distribution of Age vs All products')

for i, col in enumerate(eng_target_features):
    sns.boxplot(train[col], train["Age"].values, ax=ax[i//6][i%6])
    plt.xlabel(col)
    plt.ylabel("Age")
    plt.title("Age Box plot")
plt.show()
```

<!-- #region id="lww3Ii6_qwr4" -->
This boxplot confirms our belief about the median ages of cutsomer's having various types of accounts. We can see that customer's having junior account are very young.
<!-- #endregion -->

<!-- #region id="1z1No7m2qxvC" -->
Let's explore new_customer column
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1gM39MlivfSL" executionInfo={"status": "ok", "timestamp": 1627923580953, "user_tz": -330, "elapsed": 824, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c4d1fb9a-368e-4e91-8178-c8257d91e129"
train["New_customer"].value_counts(dropna=False)
```

```python id="1eUJojcxw02P" colab={"base_uri": "https://localhost:8080/", "height": 241} executionInfo={"status": "ok", "timestamp": 1627923600452, "user_tz": -330, "elapsed": 1281, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ccb08de1-d3ec-400e-9c5e-ca9e5dc03e7e"
missing_new_cust = train[train["New_customer"].isnull()]
missing_new_cust.sort_values(by="Customer_ID").head()
```

<!-- #region id="dqZtQ1Z-w5RI" -->
New_customer, Relationship_months and Join_date are all corelated variables.Customer would have joined before the observation period, but we do not have any information on that.Hence, I think, it is best to impute the join date first by finding the first month_status_date.
<!-- #endregion -->

```python id="m77Wn8RFw7_K" colab={"base_uri": "https://localhost:8080/", "height": 326} executionInfo={"status": "ok", "timestamp": 1627923698714, "user_tz": -330, "elapsed": 9635, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f08aa89e-9cd6-4144-971a-4d70e8c35e36"
First_month = train.groupby(["Customer_ID"])["Month_status_date"].first()
train = train.merge(First_month, on="Customer_ID", how = "outer")
train.loc[train["Join_date"].isnull(), "Join_date"] = train["Month_status_date_y"]
train.drop("Month_status_date_y", axis=1).head(5)
```

<!-- #region id="8Z-GPwAnz4Su" -->
Let's calculate relationship months
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="PMJwi9o-0JdA" executionInfo={"status": "ok", "timestamp": 1627923934975, "user_tz": -330, "elapsed": 1511, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9cdfd162-7e59-4679-a2a6-ac693724912e"
from datetime import datetime
train["Join_date"] = pd.to_datetime(train["Join_date"])
train["Month_status_date_x"] = pd.to_datetime(train["Month_status_date_x"])
train["Relnshp_Mnths"] = train["Relnshp_Mnths"].str.strip()
train.loc[train["Relnshp_Mnths"]=='NA',"Relnshp_Mnths"] = (train.loc[train["Relnshp_Mnths"]=='NA']["Month_status_date_x"] -  train.loc[train["Relnshp_Mnths"]=='NA']["Join_date"])/2678400000000000
train["Relnshp_Mnths"].value_counts().head()
```

```python id="cPjrydRp1f0R" executionInfo={"status": "ok", "timestamp": 1627924075114, "user_tz": -330, "elapsed": 2476, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train["New_customer"] = train.loc[train["New_customer"].isnull(), "New_customer"] = 1
```

```python colab={"base_uri": "https://localhost:8080/", "height": 191} id="L1mjNg161gG-" executionInfo={"status": "ok", "timestamp": 1627924102533, "user_tz": -330, "elapsed": 2496, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ea2cb58e-672c-49e9-9370-1e241556408c"
pd.crosstab(train["Relnshp_flag"], train["Relnshp_Mnths"])
```

<!-- #region id="M9wr5c6r1m1D" -->
The newer the customer, there is a high likelihood of a customer to have reltionship_flag to be 99. However, the percentage of new customers having 99 Relationship flag is less than 0.1%, so it will be better to impoute the values by most frequent value
<!-- #endregion -->

```python id="i3sI7aCl1uKa" executionInfo={"status": "ok", "timestamp": 1627924132232, "user_tz": -330, "elapsed": 995, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
train.loc[train["Relnshp_flag"].isnull(), "Relnshp_flag"] = 1
```

<!-- #region id="2L-rHCZR1ucB" -->
Great.. 4 more variables have been imputed with values!!!
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="BJSIG1ny1yDH" executionInfo={"status": "ok", "timestamp": 1627924169101, "user_tz": -330, "elapsed": 674, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e31845f9-820d-41ef-8c99-46341ec0591a"
train["Employee_Index"].value_counts()
```

<!-- #region id="FkcQMATe13r7" -->
Employee index: A active, B ex employed, F filial, N not employee, P passive
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ieupswJa2Bs8" executionInfo={"status": "ok", "timestamp": 1627924211361, "user_tz": -330, "elapsed": 430, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f267b217-5889-4e17-f6be-3c3a77ae57b2"
train["Employee_Index"].isnull().sum()
```

<!-- #region id="mLdBcndJ2CDz" -->
We do not have any more information about the employee status, so it will be safe to impute the employee index to be the most frequent value
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pRGiYG0n2FY3" executionInfo={"status": "ok", "timestamp": 1627924233486, "user_tz": -330, "elapsed": 774, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4b9a3c77-5c22-40f6-81bf-0302f441c4a3"
train.loc[train["Employee_Index"].isnull(), "Employee_Index"] = 'N'
train["Employee_Index"].isnull().sum()
```

<!-- #region id="54D2pwmt2HdF" -->
Customer's Country residence
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Tlpbry6W2Ol6" executionInfo={"status": "ok", "timestamp": 1627924263570, "user_tz": -330, "elapsed": 775, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e1014ddc-4ef8-4e63-e999-377d06dea5ce"
train["Customer_country"].value_counts().head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ov7o7B0c2Oxq" executionInfo={"status": "ok", "timestamp": 1627924269128, "user_tz": -330, "elapsed": 520, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="35aacfb5-5653-435d-cd41-252583b12f3f"
train["Customer_country"].isnull().sum()
```

<!-- #region id="1ZteRxRg2QI7" -->
Lets check if we have customer's address information in the data.


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="Bhg4h6g82R3P" executionInfo={"status": "ok", "timestamp": 1627924286626, "user_tz": -330, "elapsed": 523, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4b4d657c-20a9-4429-d38c-61fea60f36ec"
train.loc[train["Customer_country"].isnull(), ["Address_detail", "Customer_address"]].head(10)
```

<!-- #region id="JT8JL-di2UdJ" -->
Nope.. we do not have customer's data. So again, we can impute the customer country as the most frequent country which is Spain
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QCUwcpk52VGc" executionInfo={"status": "ok", "timestamp": 1627924301286, "user_tz": -330, "elapsed": 462, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0af435d1-c52b-46f3-911f-c4ef1d533910"
train.loc[train["Customer_country"].isnull(), "Customer_country"] = 'ES'
train["Customer_country"].isnull().sum()
```

<!-- #region id="e943fmqf2YCF" -->
Residence flag and forigner flag
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 131} id="tdOgz8Ql2gZF" executionInfo={"status": "ok", "timestamp": 1627924339563, "user_tz": -330, "elapsed": 717, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cf6e18c7-6a55-4993-c581-687110cc8b3a"
pd.crosstab(train["Residence_flag"], train["Forigner_flag"])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 110} id="uDgJhkcz2hSY" executionInfo={"status": "ok", "timestamp": 1627924361544, "user_tz": -330, "elapsed": 533, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="29f809d4-c03b-4bcb-c3f9-03b6d1751f10"
pd.crosstab(train.loc[train["Residence_flag"]=='S',"Customer_country"], train["Forigner_flag"])
```

<!-- #region id="s3pBbPbW2muF" -->
As we have imputed missing customer_country to be Spain, it will be safe to impute the residence flag to be Y and Forigner flag to be No, as we have only 4% of the spanish customer;s to have forigner flag
<!-- #endregion -->

```python id="7Oq673482x9Z"
train.loc[train["Residence_flag"].isnull(), "Residence_flag"] = "S"
train.loc[train["Forigner_flag"].isnull(), "Forigner_flag"] = "N"
```

<!-- #region id="2O8Jqa9Y3DRU" -->
## To be continued...
<!-- #endregion -->

<!-- #region id="QgjF_dQ13CT0" -->
https://nbviewer.jupyter.org/github/Sahoopa/My-Projects/blob/master/Santander_Data_Exploration_EDA_Submission.ipynb
<!-- #endregion -->

<!-- #region id="AjH7hpoy3C0A" -->
https://nbviewer.jupyter.org/github/Sahoopa/My-Projects/blob/master/Santander%20Data%20Prep.ipynb
<!-- #endregion -->

<!-- #region id="E_i_8Psi3erM" -->
https://nbviewer.jupyter.org/github/Sahoopa/My-Projects/blob/master/Santander%20Product%20Recommendation/Santander%20Data%20Prep%20-%20Part2.ipynb
<!-- #endregion -->

<!-- #region id="IolvOIDu3MOS" -->
https://nbviewer.jupyter.org/github/Sahoopa/My-Projects/blob/master/Santander%20Product%20Recommendation/Santander_Data_Exploration_EDA%20-%20Part1.ipynb
<!-- #endregion -->
