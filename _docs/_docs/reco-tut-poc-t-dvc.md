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

```python id="Iqka8EK_-i3J"
import os
project_name = "reco-tut-poc"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="SNRTJUnc-2Gv" executionInfo={"status": "ok", "timestamp": 1629551962286, "user_tz": -330, "elapsed": 3145, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ae304153-1130-4f87-f5c7-531cad83a960"
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

```python colab={"base_uri": "https://localhost:8080/"} id="40cRJp8h-2G2" executionInfo={"status": "ok", "timestamp": 1629555097580, "user_tz": -330, "elapsed": 569, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="021453cd-d274-4dd1-9012-ad592a8a4557"
!git status
```

```python id="pR2reD8e_Drw"
!git pull --rebase origin "{branch}"
!git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="E0MFwfEh-2G3" executionInfo={"status": "ok", "timestamp": 1629554672437, "user_tz": -330, "elapsed": 1982, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="10f9d4d2-65a5-48df-fcc8-f015ce8a11b7"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="nOzxKkK43bT0" -->
---
<!-- #endregion -->

```python id="vncq0FEd3zf6"
!pip install -q git+https://github.com/sparsh-ai/recochef
!pip install -U -q PyDrive dvc dvc[gdrive]
from recochef.utils.gdrive import *
drive_handler = GoogleDriveHandler()
test_subfolder_id = drive_handler.create_folder('reco-tut-poc')
!dvc init
!dvc remote add -d myremote gdrive://"{test_subfolder_id}"
```

```python id="-g-1sXBs48nE"
# !cd /content && wget https://code.dvc.org/get-started/code.zip && unzip code.zip -d ./code
# !cat /content/code/.github/workflows/cml.yaml
# !cp /content/code/src/*.py ./code
# !cp /content/code/src/requirements.txt .
# !cp /content/code/params.yaml .
# !mkdir -p ./data/bronze ./data/silver ./data/gold
# !cp /content/temp/data/data.xml ./data/bronze
# !dvc add ./data/bronze/data.xml
# !git add data/bronze/.gitignore data/bronze/data.xml.dvc
# !dvc commit
# !git commit -m 'commit'
# !dvc push
```

<!-- #region id="96GIvImBBgTS" -->
Building pipeline
<!-- #endregion -->

```python id="f5crQlfmA6zy"
# %%sh
# dvc run -n prepare \
#           -p prepare.seed,prepare.split \
#           -d code/prepare.py -d data/bronze/data.xml \
#           -o data/silver \
#           python code/prepare.py data/bronze/data.xml

# dvc run -n featurize \
#           -p featurize.max_features,featurize.ngrams \
#           -d code/featurization.py -d data/silver \
#           -o data/gold \
#           python code/featurization.py data/silver data/gold

# dvc run -n train \
#           -p train.seed,train.n_est,train.min_split \
#           -d code/train.py -d data/gold \
#           -o artifacts/model.pkl \
#           python code/train.py data/gold artifacts/model.pkl
```

```python id="XkwEpEcm3c9W"
# !git add dvc.lock dvc.yaml data/.gitignore
# !git add dvc.lock data/.gitignore dvc.yaml
# !git add dvc.yaml dvc.lock artifacts/.gitignore
# !dvc commit && dvc push
```

<!-- #region id="0KgqbZxxBmJG" -->
Changing params and reproducing pipeline again
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="mRCUc5CrCTvs" executionInfo={"status": "ok", "timestamp": 1629555785225, "user_tz": -330, "elapsed": 656, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="737bc210-2926-4046-a05b-095f117333a8"
%%writefile params.yaml
prepare:
  split: 0.20
  seed: 20170428

featurize:
  max_features: 500
  ngrams: 1

train:
  seed: 20170428
  n_est: 60
  min_split: 2
```

```python id="S_kbXO44CjOO"
!dvc repro
```

<!-- #region id="JmVka8rSEf89" -->
Let's save this iteration, so we can compare it with previous pipeline config
<!-- #endregion -->

```python id="7LmcfVMCEyAH"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="liB0yIWqFJ3o" executionInfo={"status": "ok", "timestamp": 1629555832169, "user_tz": -330, "elapsed": 2736, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bec9fc88-7f02-4387-d501-9a2ab124b988"
!dvc params diff
```

<!-- #region id="CEUjijUhCk_5" -->
Visualize
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="gjFELFnfC5O0" executionInfo={"status": "ok", "timestamp": 1629554981767, "user_tz": -330, "elapsed": 2278, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="006bc8a1-4e4c-4599-9af2-57693dd00677"
!dvc dag
```

<!-- #region id="A5s5qm0DC6db" -->
Add evaluate stage in pipeline
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="bpI7dgRLDQXs" executionInfo={"status": "ok", "timestamp": 1629555235459, "user_tz": -330, "elapsed": 4129, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e0f50143-af40-4a10-b658-38f9e37bc097"
!dvc run -n evaluate \
          -d code/evaluate.py -d artifacts/model.pkl -d data/gold \
          -M outputs/scores.json \
          --plots-no-cache outputs/prc.json \
          --plots-no-cache outputs/roc.json \
          python code/evaluate.py artifacts/model.pkl \
                 data/gold outputs/scores.json outputs/prc.json outputs/roc.json
```

```python id="wTdteU8sD4ch"
!git add dvc.lock dvc.yaml
```

<!-- #region id="MpMWvCwLD_x0" -->
Visualize
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xc6l-w9yEONq" executionInfo={"status": "ok", "timestamp": 1629555333910, "user_tz": -330, "elapsed": 3764, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7c5ab5e1-34df-461f-cc5b-ad8f2cd6c0bc"
!dvc metrics show
```

<!-- #region id="k5Ig1qbaEQj1" -->
To view plots, first specify which arrays to use as the plot axes. We only need to do this once, and DVC will save our plot configurations.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_WCdhM-uEWn2" executionInfo={"status": "ok", "timestamp": 1629555400419, "user_tz": -330, "elapsed": 6938, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e276a32f-dfec-4fcf-909d-8aaa075f6f30"
!dvc plots modify outputs/prc.json -x recall -y precision
!dvc plots modify outputs/roc.json -x fpr -y tpr
!dvc plots show
```

<!-- #region id="iyztkVb8E1Db" -->
Comparing iterations by updating pipeline
<!-- #endregion -->

```python id="4BN2zI9BGn1_" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629556033717, "user_tz": -330, "elapsed": 763, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f52b2c00-05a4-45ef-9da1-3be106b5349a"
%%writefile params.yaml
prepare:
  split: 0.20
  seed: 20170428

featurize:
  max_features: 1500
  ngrams: 2

train:
  seed: 20170428
  n_est: 60
  min_split: 2
```

```python id="PelhBfIkGqIr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629556038591, "user_tz": -330, "elapsed": 2402, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="056acba5-d5a5-4c1f-9ad1-5625b0345e42"
!dvc params diff
```

```python colab={"base_uri": "https://localhost:8080/"} id="AP8Gfy8sG8ve" executionInfo={"status": "ok", "timestamp": 1629556093760, "user_tz": -330, "elapsed": 33685, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="094fe045-54d2-4e65-c1c7-95b01c7ad511"
!dvc repro
```

```python colab={"base_uri": "https://localhost:8080/"} id="2Xi2YBz2HCiG" executionInfo={"status": "ok", "timestamp": 1629556098668, "user_tz": -330, "elapsed": 2758, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5f1d8f93-5d49-4a68-fb0b-4e09c8e6d422"
!dvc metrics diff
```

```python colab={"base_uri": "https://localhost:8080/"} id="gU6L-EuoHINW" executionInfo={"status": "ok", "timestamp": 1629556099770, "user_tz": -330, "elapsed": 1119, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="98260a28-bf7a-4645-82f5-119ca2f264a1"
!dvc plots diff
```

```python colab={"base_uri": "https://localhost:8080/", "height": 368} id="XOiGNzPEHPAX" executionInfo={"status": "ok", "timestamp": 1629556136449, "user_tz": -330, "elapsed": 10, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a59b058d-137f-48fb-f3aa-b2cbfdf41ae9"
import IPython
IPython.display.HTML(filename='./plots.html')
```

<!-- #region id="df5maxdVHVWz" -->
### Experiments
<!-- #endregion -->

<!-- #region id="zQWHKfjqKBTC" -->
Let's further increase the params to see how model performance change
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="veJ5RSMOKIqe" executionInfo={"status": "ok", "timestamp": 1629556924685, "user_tz": -330, "elapsed": 39976, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bf5c7f06-3d23-4715-ff0f-0aa0e056c37a"
!dvc exp run --set-param featurize.max_features=3000
```

```python colab={"base_uri": "https://localhost:8080/"} id="RBUyfIsLKL33" executionInfo={"status": "ok", "timestamp": 1629556931919, "user_tz": -330, "elapsed": 3031, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="22bcf419-bc66-4144-857a-710bc7c997aa"
!dvc exp diff
```

<!-- #region id="XqM6IQBMKPg5" -->
Queueing experiments
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6HoyeXHQKf7q" executionInfo={"status": "ok", "timestamp": 1629557007343, "user_tz": -330, "elapsed": 13341, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f73643e6-540b-4481-a687-a9088d5a4d5d"
!dvc exp run --queue -S train.min_split=8
!dvc exp run --queue -S train.min_split=64
!dvc exp run --queue -S train.min_split=2 -S train.n_est=100
!dvc exp run --queue -S train.min_split=8 -S train.n_est=100
!dvc exp run --queue -S train.min_split=64 -S train.n_est=100
```

```python colab={"base_uri": "https://localhost:8080/"} id="UJu40o6xKmh_" executionInfo={"status": "ok", "timestamp": 1629557058727, "user_tz": -330, "elapsed": 44446, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c73cd4e8-972f-4334-a838-039d68c57e2d"
!dvc exp run --run-all --jobs 2
```

```python colab={"base_uri": "https://localhost:8080/"} id="R3-7CQjWKrn0" executionInfo={"status": "ok", "timestamp": 1629557127529, "user_tz": -330, "elapsed": 2446, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="235c9873-e48f-48a6-b0c6-48fd1857bfbb"
!dvc exp show --no-timestamp \
               --include-params train.n_est,train.min_split
```

<!-- #region id="8aDZkMx6LUno" -->
Now that we know the best parameters, let's keep that experiment and ignore the rest.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MH1vGzbZLVc9" executionInfo={"status": "ok", "timestamp": 1629557225761, "user_tz": -330, "elapsed": 2574, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="606c1322-e3ab-4efb-a195-03264623a745"
!dvc exp apply exp-f6e3f
```

```python colab={"base_uri": "https://localhost:8080/"} id="5gG2B4ALLetX" executionInfo={"status": "ok", "timestamp": 1629557262307, "user_tz": -330, "elapsed": 2361, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3350def1-fbb5-46af-f0c7-f5cc4b972b6c"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="psqBipNZLnsj"
!dvc commit && dvc push
```
