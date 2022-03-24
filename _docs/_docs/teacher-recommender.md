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

<!-- #region id="fkXkWczy-MHI" -->
### Recommending `Teachers` to `Students` for the K-12 online one-on-one courses

1. Pseudo Matching Score Module - provide reliable training labels
2. Ranking Module - scores every candidate teacher
3. Novelty Boosting Module - gives additional opportunities to new teachers
4. Diversity Module - guardrails the recommended results to reduce the chance of collision
<!-- #endregion -->

<!-- #region id="nvT1h1H6AJd6" -->
| Challenge          | Description                        | Workaround        |
| ------------------ | ---------------------------------- | ----------------- |
| Limited sizes of demand and supply | The number of teachers in supply side is incredibly smaller compared to Internet-scaled inventories. Moreover, different from item based recommendation where popular items can be suggested to millions of users simultaneously, a teacher can only take a very limited amount of students and students may only take one or two classes at each semester. | - |
| Lack of gold standard | There is no ground truth showing how good a match is between a teacher and a student. The rating based mechanism doesn’t work since ratings from K-12 students are very noisy and unreliable. | Pseudo Matching Score Module |
| Cold-start teachers | The online educational marketplace is dynamic and there are always new teachers joining. It is important to give such new teachers opportunities to let them take students instead of keeping recommending existing best performing teachers. | Novelty Boosting Module |
| High-demand diversity | It is undesirable to recommend the same set of teachers to students and the teacher recommender systems are supposed to reduce chances that two students want to book the same teacher at the same time. | Diversity Module |
<!-- #endregion -->

<!-- #region id="7O1mHvFBHQry" -->
## Setup
<!-- #endregion -->

```python id="SKBBv_cKYTPj"
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

np.random.seed(42)
%matplotlib inline
tqdm.pandas()
```

<!-- #region id="aI7vJIiJG-Zp" -->
## Generate Synthetic Data
<!-- #endregion -->

```python id="wQ6bPXvudGKv" executionInfo={"status": "ok", "timestamp": 1627106824034, "user_tz": -330, "elapsed": 856, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
teacher_set = np.arange(200)
student_set = np.arange(300)
gender_set = ['male','female']
school_set = ['school1','school2']
talking_time = ['short','long']
```

```python colab={"base_uri": "https://localhost:8080/"} id="slyX4L_4dfJW" executionInfo={"status": "ok", "timestamp": 1627106824037, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e67448e8-44fd-4b9b-8cee-d0e50629747e"
df = pd.DataFrame(list(product(teacher_set, student_set)), columns=['teacherid', 'studentid'])

df2 = pd.DataFrame(list(product(np.random.choice(teacher_set, int(len(teacher_set)/2)),
                                np.random.choice(student_set, int(len(student_set)/2)))),
                       columns=['teacherid', 'studentid'])

df = pd.concat([df, df2], ignore_index=True, sort=False)

df = df.rename_axis('session_id').reset_index()

df['status'] = np.random.choice(['completed','dropped','not_taken'],len(df),p=[0.05,0.1,0.85])

df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 359} id="mNXPc9INdRpp" executionInfo={"status": "ok", "timestamp": 1627106824861, "user_tz": -330, "elapsed": 834, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3b54a813-7e4a-4c62-c343-9baea94124ce"
df.head(10)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="Z5-DPvIobtXm" executionInfo={"status": "ok", "timestamp": 1627106824862, "user_tz": -330, "elapsed": 22, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8d035894-bfe7-483e-cacb-b8a645469119"
df_s = pd.DataFrame(student_set, columns=['studentid'])
df_s['gender'] = np.random.choice(gender_set,len(df_s),p=[0.4, 0.6])
df_s['school'] = np.random.choice(school_set,len(df_s),p=[0.3, 0.7])
df_s['talking_time'] = np.random.choice(talking_time,len(df_s),p=[0.8, 0.2])
df_s
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="4UeobhQDbvSr" executionInfo={"status": "ok", "timestamp": 1627106824864, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ba8bf47d-cd55-4aa8-e40b-2c388735fd6a"
df_t = pd.DataFrame(teacher_set, columns=['teacherid'])
df_t['gender'] = np.random.choice(gender_set,len(df_t),p=[0.7, 0.3])
df_t['school'] = np.random.choice(school_set,len(df_t),p=[0.4, 0.6])
df_t['talking_time'] = np.random.choice(talking_time,len(df_t),p=[0.6, 0.4])
df_t
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="Dk-FqnU9dedR" executionInfo={"status": "ok", "timestamp": 1627106826754, "user_tz": -330, "elapsed": 39, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fe1b5546-9861-46f4-8b52-39544c88a2cf"
df = df.merge(df_s, on='studentid').merge(df_t, on='teacherid', suffixes=('_student','_teacher'))
df
```

```python id="zcWXB_5bctmQ" executionInfo={"status": "ok", "timestamp": 1627106826756, "user_tz": -330, "elapsed": 34, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
dfx = df.copy()

# logical filter - student and teachers belong to same school
dfx = dfx[dfx['school_student']==dfx['school_teacher']]
dfx.drop('school_teacher', axis=1, inplace=True)
dfx.rename(columns={'school_student':'school'}, inplace=True)

# in school1, female students prefer female teacher
subset = dfx.loc[(dfx['school']=='school1') & (dfx['gender_student']=='female') & (dfx['gender_teacher']=='female'),'status']
dfx.loc[subset.index,'status'] = np.random.choice(['completed','dropped','not_taken'],len(subset),p=[0.4,0.1,0.5])

# in school2, male students prefer female teacher
subset = dfx.loc[(dfx['school']=='school2') & (dfx['gender_student']=='male') & (dfx['gender_teacher']=='female'),'status']
dfx.loc[subset.index,'status'] = np.random.choice(['completed','dropped','not_taken'],len(subset),p=[0.4,0.1,0.5])

# in school1, students prefer teacher who talk long
subset = dfx.loc[(dfx['school']=='school1') & (dfx['talking_time_teacher']=='long'),'status']
dfx.loc[subset.index,'status'] = np.random.choice(['completed','dropped','not_taken'],len(subset),p=[0.4,0.1,0.5])

# in school2, students prefer teacher who talk short
subset = dfx.loc[(dfx['school']=='school2') & (dfx['talking_time_teacher']=='short'),'status']
dfx.loc[subset.index,'status'] = np.random.choice(['completed','dropped','not_taken'],len(subset),p=[0.4,0.1,0.5])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="FETaLUgUfTEX" executionInfo={"status": "ok", "timestamp": 1627106826757, "user_tz": -330, "elapsed": 27, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="31792d49-cccc-4d97-81ea-a237c1d8f16b"
dfx
```

<!-- #region id="ymtuySskmQ4K" -->
Let's checkout some patterns that we engineered
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 281} id="rXe5KNvkmNPi" executionInfo={"status": "ok", "timestamp": 1627106827488, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="be8fa03a-57d5-4565-a8ef-e344ff1fd985"
# status count in general
figure, axes = plt.subplots(1, 2, figsize=(12,4))
df.loc[:,'status'].value_counts().plot(kind='barh', ax=axes[0], title='Before pattern engineering');
dfx.loc[:,'status'].value_counts().plot(kind='barh', ax=axes[1], title='After pattern engineering');
```

```python colab={"base_uri": "https://localhost:8080/", "height": 281} id="eRky64qvfTmH" executionInfo={"status": "ok", "timestamp": 1627106828353, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c5305371-7bc5-47c8-a94c-f0195b6258e0"
# status count for long-talking students to short-talking teachers
figure, axes = plt.subplots(1, 2, figsize=(12,4))
df.loc[(df['talking_time_student']=='long') & (df['talking_time_teacher']=='short'),'status'].value_counts().plot(kind='barh', ax=axes[0], title='Before pattern engineering');
dfx.loc[(dfx['talking_time_student']=='long') & (dfx['talking_time_teacher']=='short'),'status'].value_counts().plot(kind='barh', ax=axes[1], title='After pattern engineering');
```

```python colab={"base_uri": "https://localhost:8080/", "height": 281} id="oHuW3qvzkENq" executionInfo={"status": "ok", "timestamp": 1627106828922, "user_tz": -330, "elapsed": 584, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b198bb2e-2add-4f1c-aa30-79350811da26"
# status count for female student preference
figure, axes = plt.subplots(1, 2, figsize=(12,4))
df.loc[(df['gender_student']=='female'),'status'].value_counts().plot(kind='barh', ax=axes[0], title='Before pattern engineering');
dfx.loc[(dfx['gender_student']=='female'),'status'].value_counts().plot(kind='barh', ax=axes[1], title='After pattern engineering');
```

```python id="1L6Wul12kl8s" executionInfo={"status": "ok", "timestamp": 1627106843568, "user_tz": -330, "elapsed": 464, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
dfx.to_csv('data.csv', index=False)
```

<!-- #region id="EoJ1dirnDDC3" -->
## Pseudo Matching Score Module
<!-- #endregion -->

<!-- #region id="0MmrqmE1DUqH" -->
- Signal 1 - If student $s_i \in \mathbb{S}$ dropped out from a session of teacher $t_j \in \mathbb{T}$, it signals a negative feedback of $s_i$ towards $t_j$.
- Signal 2 - The recency of this feedback is also an important signal because if $s_i$ dropped from $t_j$'s session yesterday and from $t_k$'s session a month ago, then as per our intuition, we can assume that again recommending $t_j$ to this student would cause more dissatisfaction to the student.
<!-- #endregion -->

<!-- #region id="DUTvz0ZqLkBa" -->
Positive Pseudo Matching Score $ \mathcal{P}_{{s_i},{t_j}} = \dfrac{Total\ no.\ of\ courses\ taught\ by\ t_j\ to\ s_i}{Total\ no.\ of\ courses\ taken\ by\ s_i}$
<!-- #endregion -->

<!-- #region id="BiioQ7Ogh0r8" -->
Total pseudo score would be the sum of positive and negative psuedo scores. This score tanges from -1 to 1.

- Best case scenario: student s only prefer teacher t, and therefore always completed sessions taught by t. In this case psuedo score would be 1.
- Neutral case scenario: student s never took any lesson from t. In this case, psuedo score would be 0.
- Worst case scenario: Student s always dropped the courses taught by teacher t. And score would be -1.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="1yD-Yw6yIBW3" executionInfo={"status": "ok", "timestamp": 1627111104921, "user_tz": -330, "elapsed": 823, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ff8bba26-4061-41ba-d493-1a4d53d77202"
df = pd.read_csv('data.csv')
df
```

```python id="jZhnrOhDJWQM" executionInfo={"status": "ok", "timestamp": 1627109653413, "user_tz": -330, "elapsed": 1059, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def calc_ppos(df, si='s1', tj='t1'):
    sum_tj_si = len(df[(df['studentid']==si) & \
                       (df['teacherid']==tj) & \
                       (df['status']=='completed')])
    sum_si = len(df[(df['studentid']==si) & \
                    (df['status']=='completed')]) + 1e-5
    return sum_tj_si/sum_si


def calc_pneg(df, si='s1', tj='t1'):
    sum_tj_si = len(df[(df['studentid']==si) & \
                       (df['teacherid']==tj) & \
                       (df['status']=='dropped')])  
    return np.exp(-sum_tj_si)-1
```

```python id="DAf9mPYDWirk" executionInfo={"status": "ok", "timestamp": 1627111113542, "user_tz": -330, "elapsed": 521, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def calc_score(df):
    sid = df.head(1).studentid.values[0]
    tid = df.head(1).teacherid.values[0]
    pos_score = calc_ppos(df, sid, tid)
    neg_score = calc_pneg(df, sid, tid)
    return pos_score + neg_score
```

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["2b1a6e1fa73c4739ad126de2a867dd95", "cb4c60d4e12741fe95711b45700973c0", "5b234f1c29004ca7bdf427b954794e73", "b0bf1f7d312e4458b8ad9e2c376cda39", "f51ee85cbbe540f1a9c97bbeedf5e861", "4c9cb0b7b7784227aa2f475beca982ef", "c7a2eb5b7e534d85a26a9ccc19d0e52d", "b4396bd24ce7490891b6695adf0595f4"]} id="1s_V7BqIVnyZ" executionInfo={"status": "ok", "timestamp": 1627111235350, "user_tz": -330, "elapsed": 108187, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4d56b345-0a67-4edb-fc84-e5bd9faab3ae"
_df = df.groupby(['studentid','teacherid']).progress_apply(calc_score)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="DQ2zl7z3X6gD" executionInfo={"status": "ok", "timestamp": 1627111245914, "user_tz": -330, "elapsed": 847, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2318ea7e-7a37-419c-c730-164955165114"
_df = _df.reset_index()
_df.columns=['studentid','teacherid','score']
_df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Gbc16VegYf_2" executionInfo={"status": "ok", "timestamp": 1627111252110, "user_tz": -330, "elapsed": 704, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5470b4a8-3193-42c2-f9be-0453c7e00636"
df = df.merge(_df, on=['studentid','teacherid'])
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="nOJzWC90QoqE" executionInfo={"status": "ok", "timestamp": 1627111259357, "user_tz": -330, "elapsed": 1180, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="73674470-b202-44b4-e5c9-9f44379f089d"
df.score.describe()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="ZGRnUYgcQW6p" executionInfo={"status": "ok", "timestamp": 1627111325826, "user_tz": -330, "elapsed": 582, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="de8dbbb6-5d59-4fbd-c93c-4d750ceb3545"
sns.kdeplot(df.score.values);
```

```python id="ESdkW8n3SC5V"

```
