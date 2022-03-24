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

```python id="xolR7gYoJCEb" executionInfo={"status": "ok", "timestamp": 1628951003710, "user_tz": -330, "elapsed": 467, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-elf"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Yq6V2furI-Qf" executionInfo={"status": "ok", "timestamp": 1628951009879, "user_tz": -330, "elapsed": 5650, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="125d3a45-af8c-48c9-99fe-81c80a370c43"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="lRc_TZM6I-Ql" executionInfo={"status": "ok", "timestamp": 1628951116442, "user_tz": -330, "elapsed": 498, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7a02864f-1d8d-499f-ef67-4378da543774"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="PW8XIzLQI-Qm" executionInfo={"status": "ok", "timestamp": 1628951121888, "user_tz": -330, "elapsed": 1198, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ea188a6f-637b-49db-8d26-542cadad1223"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="sdD-8ZiRJSG-"
# !cd /content && git clone https://github.com/KhalilDMK/EBPR.git
# !mkdir ./data/bronze && mv /content/EBPR/Data/* ./data/bronze
```

```python id="pjDBBUbYI-Qo" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1628951028054, "user_tz": -330, "elapsed": 444, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c6cb2413-2534-4192-d765-873ed666c2d7"
import glob
files = sorted(glob.glob(f'/content/drive/MyDrive/Colab Notebooks/{project_name}*.ipynb'))
files
```

```python id="swdfCgeII-Qp" executionInfo={"status": "ok", "timestamp": 1628951035288, "user_tz": -330, "elapsed": 689, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from shutil import copy
for fpath in files[1:]:
    copy(fpath, './notebooks')
```

```sh colab={"base_uri": "https://localhost:8080/"} id="BRQFlohOC4NF" executionInfo={"status": "ok", "timestamp": 1628951048471, "user_tz": -330, "elapsed": 11283, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1f60a7be-fa42-43a8-db9e-1b8d00058470"
mkdir -p ./code/nbs
for file in ./notebooks/*.ipynb
do
    jupyter nbconvert --output-dir="./code/nbs" --to python "$file"
done
```

```python colab={"base_uri": "https://localhost:8080/"} id="l1vbJbECC4NG" executionInfo={"status": "ok", "timestamp": 1628951058572, "user_tz": -330, "elapsed": 10121, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="12ad9815-54b7-43a4-c3ac-82262f5d4ca2"
!sudo apt-get install -qq tree
```

```python colab={"base_uri": "https://localhost:8080/"} id="4m4vd_w4C4NG" executionInfo={"status": "ok", "timestamp": 1628951058573, "user_tz": -330, "elapsed": 25, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8458d9ff-2141-45c1-d940-41d3640f0599"
!tree -L 3 .
```

````python colab={"base_uri": "https://localhost:8080/"} id="QIW4Rm6dC4NG" executionInfo={"status": "ok", "timestamp": 1628951110614, "user_tz": -330, "elapsed": 627, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c0cc5080-bdf0-4adb-a8f1-160ba4792b84"
%%writefile README.md
# Debiased Explainable Pairwise Ranking

Pairwise ranking model BPR not only outperforms pointwise counterparts but also able to handle implicit feedback. But it is a black-box model and vulnerable to exposure bias. This exposure bias usually translates into an unfairness against the least popular items because they risk being under-exposed by the recommender system. **One approach to address this problem is to use EBPR (Explainable BPR) loss function.**

## Project structure
```
.
├── code
│   ├── EBPR_model.py
│   ├── engine_EBPR.py
│   ├── hyperparameter_tuning.py
│   ├── __init__.py
│   ├── metrics.py
│   ├── nbs
│   │   ├── reco-tut-elf-ml100k-01-data-preparation.py
│   │   └── reco-tut-elf-notebook.py
│   ├── preprocess.py
│   ├── train_EBPR.py
│   └── utils.py
├── data
│   └── bronze
│       ├── lastfm-2k
│       ├── ml-100k
│       └── ml-1m
├── docs
├── LICENSE
├── models
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_100_l2reg_0_Epoch31_NDCG@10_0.4173_HR@10_0.6946_MEP@10_0.9274_WMEP@10_0.3581_Avg_Pop@10_0.4685_EFD@10_1.2144_Avg_Pair_Sim@10_0.2616.model
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_100_l2reg_0_Epoch35_NDCG@10_0.4222_HR@10_0.6946_MEP@10_0.9244_WMEP@10_0.3534_Avg_Pop@10_0.4667_EFD@10_1.2195_Avg_Pair_Sim@10_0.2609.model
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_100_l2reg_0_Epoch48_NDCG@10_0.4130_HR@10_0.6925_MEP@10_0.9176_WMEP@10_0.3472_Avg_Pop@10_0.4662_EFD@10_1.2205_Avg_Pair_Sim@10_0.2588.model
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_20_l2reg_0_Epoch42_NDCG@10_0.3980_HR@10_0.6713_MEP@10_0.9238_WMEP@10_0.3579_Avg_Pop@10_0.4765_EFD@10_1.1796_Avg_Pair_Sim@10_0.2717.model
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_20_l2reg_0_Epoch48_NDCG@10_0.3986_HR@10_0.6649_MEP@10_0.9255_WMEP@10_0.3589_Avg_Pop@10_0.4745_EFD@10_1.1934_Avg_Pair_Sim@10_0.2696.model
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_20_l2reg_0_Epoch49_NDCG@10_0.4022_HR@10_0.6776_MEP@10_0.9211_WMEP@10_0.3576_Avg_Pop@10_0.4723_EFD@10_1.1993_Avg_Pair_Sim@10_0.2701.model
│   ├── EBPR_ml-100k_batchsize_100_opt_adam_lr_0.001_latent_50_l2reg_0.0_Epoch49_NDCG@10_0.3838_HR@10_0.6628_MEP@10_0.9282_WMEP@10_0.3593_Avg_Pop@10_0.4658_EFD@10_1.2251_Avg_Pair_Sim@10_0.2607.model
│   ├── EBPR_ml-100k_batchsize_500_opt_adam_lr_0.001_latent_10_l2reg_0_Epoch47_NDCG@10_0.3482_HR@10_0.5949_MEP@10_0.9060_WMEP@10_0.3486_Avg_Pop@10_0.4998_EFD@10_1.0814_Avg_Pair_Sim@10_0.2918.model
│   ├── EBPR_ml-100k_batchsize_500_opt_adam_lr_0.001_latent_10_l2reg_0_Epoch48_NDCG@10_0.3429_HR@10_0.6045_MEP@10_0.9030_WMEP@10_0.3479_Avg_Pop@10_0.5051_EFD@10_1.0634_Avg_Pair_Sim@10_0.2960.model
│   ├── EBPR_ml-100k_batchsize_500_opt_adam_lr_0.001_latent_10_l2reg_0_Epoch49_NDCG@10_0.3384_HR@10_0.5885_MEP@10_0.9035_WMEP@10_0.3497_Avg_Pop@10_0.5100_EFD@10_1.0507_Avg_Pair_Sim@10_0.2977.model
│   ├── EBPR_ml-100k_batchsize_500_opt_adam_lr_0.001_latent_20_l2reg_0.001_Epoch48_NDCG@10_0.3709_HR@10_0.6405_MEP@10_0.9205_WMEP@10_0.3583_Avg_Pop@10_0.4934_EFD@10_1.1147_Avg_Pair_Sim@10_0.2861.model
│   ├── EBPR_ml-100k_batchsize_500_opt_adam_lr_0.001_latent_20_l2reg_0.001_Epoch49_NDCG@10_0.3733_HR@10_0.6416_MEP@10_0.9175_WMEP@10_0.3582_Avg_Pop@10_0.4913_EFD@10_1.1205_Avg_Pair_Sim@10_0.2877.model
│   ├── EBPR_ml-100k_batchsize_500_opt_adam_lr_0.001_latent_20_l2reg_0.001_Epoch49_NDCG@10_0.3822_HR@10_0.6437_MEP@10_0.9137_WMEP@10_0.3551_Avg_Pop@10_0.4894_EFD@10_1.1286_Avg_Pair_Sim@10_0.2858.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_100_l2reg_0.001_Epoch22_NDCG@10_0.4080_HR@10_0.6914_MEP@10_0.9295_WMEP@10_0.3576_Avg_Pop@10_0.4741_EFD@10_1.1954_Avg_Pair_Sim@10_0.2659.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_100_l2reg_0.001_Epoch32_NDCG@10_0.4043_HR@10_0.6808_MEP@10_0.9273_WMEP@10_0.3537_Avg_Pop@10_0.4700_EFD@10_1.2071_Avg_Pair_Sim@10_0.2609.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_100_l2reg_0.001_Epoch34_NDCG@10_0.4192_HR@10_0.7010_MEP@10_0.9275_WMEP@10_0.3507_Avg_Pop@10_0.4679_EFD@10_1.2125_Avg_Pair_Sim@10_0.2592.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_20_l2reg_0.001_Epoch46_NDCG@10_0.3952_HR@10_0.6681_MEP@10_0.9316_WMEP@10_0.3612_Avg_Pop@10_0.4728_EFD@10_1.1976_Avg_Pair_Sim@10_0.2690.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_20_l2reg_0.001_Epoch46_NDCG@10_0.4058_HR@10_0.6819_MEP@10_0.9234_WMEP@10_0.3559_Avg_Pop@10_0.4736_EFD@10_1.1930_Avg_Pair_Sim@10_0.2691.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_20_l2reg_0.001_Epoch48_NDCG@10_0.4040_HR@10_0.6776_MEP@10_0.9285_WMEP@10_0.3585_Avg_Pop@10_0.4702_EFD@10_1.2034_Avg_Pair_Sim@10_0.2659.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_20_l2reg_1e-05_Epoch45_NDCG@10_0.4015_HR@10_0.6808_MEP@10_0.9211_WMEP@10_0.3546_Avg_Pop@10_0.4748_EFD@10_1.1922_Avg_Pair_Sim@10_0.2700.model
│   ├── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_20_l2reg_1e-05_Epoch46_NDCG@10_0.4003_HR@10_0.6755_MEP@10_0.9221_WMEP@10_0.3572_Avg_Pop@10_0.4717_EFD@10_1.2022_Avg_Pair_Sim@10_0.2683.model
│   └── EBPR_ml-100k_batchsize_50_opt_adam_lr_0.001_latent_20_l2reg_1e-05_Epoch46_NDCG@10_0.4042_HR@10_0.6734_MEP@10_0.9231_WMEP@10_0.3548_Avg_Pop@10_0.4729_EFD@10_1.1968_Avg_Pair_Sim@10_0.2673.model
├── notebooks
│   ├── reco-tut-elf-ml100k-01-data-preparation.ipynb
│   └── reco-tut-elf-notebook.ipynb
├── outputs
│   └── Hyperparameter_tuning_EBPR_ml-100k.csv
└── README.md  
```
````

```python colab={"base_uri": "https://localhost:8080/"} id="bIwHEGctC4NH" executionInfo={"status": "ok", "timestamp": 1628951143854, "user_tz": -330, "elapsed": 8946, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="59b5751a-18f0-4c91-eb66-c77d420d0075"
xproject_name = "reco-nb-stage"; xbranch = "queued"; xaccount = "recohut"
xproject_path = os.path.join('/content', xproject_name)

if not os.path.exists(xproject_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + xproject_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "recotut@recohut.com"
    !git config --global user.name  "reco-tut"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{xaccount}"/"{xproject_name}".git
    !git pull origin "{xbranch}"
else:
    %cd "{xproject_path}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="A2OY1BAPC4NI" executionInfo={"status": "ok", "timestamp": 1628951145445, "user_tz": -330, "elapsed": 1612, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="64478067-fc53-4fa8-9450-236b11d3fc85"
!git checkout -b queued
!cp $project_path/notebooks/*.ipynb ./_notebooks
!git add . && git commit -m 'commit' && git push origin queued
```

```python id="9NPjEuUXDcCK"

```
