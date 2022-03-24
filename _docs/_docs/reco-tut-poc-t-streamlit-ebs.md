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

```python colab={"base_uri": "https://localhost:8080/"} id="SNRTJUnc-2Gv" executionInfo={"status": "ok", "timestamp": 1629529395382, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="64bdd018-aec2-48a2-c2fa-e6e9264cd89d"
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

```python colab={"base_uri": "https://localhost:8080/"} id="40cRJp8h-2G2" executionInfo={"status": "ok", "timestamp": 1629529399108, "user_tz": -330, "elapsed": 459, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b5c2bd4b-ee9c-4dc4-d53f-4059e9591894"
!git status
```

```python colab={"base_uri": "https://localhost:8080/"} id="pR2reD8e_Drw" executionInfo={"status": "ok", "timestamp": 1629490140454, "user_tz": -330, "elapsed": 1148, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aa53fb96-47b7-4f4f-93bf-5e32c70d956a"
!git pull --rebase origin "{branch}"
!git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="E0MFwfEh-2G3" executionInfo={"status": "ok", "timestamp": 1629529403788, "user_tz": -330, "elapsed": 1177, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="48797259-fc17-42c6-c609-a2108237c01b"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="q2RGRn4BAhI8" executionInfo={"status": "ok", "timestamp": 1629520333962, "user_tz": -330, "elapsed": 20061, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="67f945cd-0e2e-4eca-fa5c-25adc85a69a7"
!pip install -q awscli
```

```python colab={"base_uri": "https://localhost:8080/"} id="ekk4PiEcURSB" executionInfo={"status": "ok", "timestamp": 1629526392359, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fc982041-ee77-4810-9df4-0e9f261d9152"

```

```python id="qv-h0EdF_OAd"

```

````python colab={"base_uri": "https://localhost:8080/"} id="bahGFSEmCGGz" executionInfo={"status": "ok", "timestamp": 1629527143940, "user_tz": -330, "elapsed": 439, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3067bef5-b123-4204-bd3c-486e149e48ad"
%%writefile /content/reco-tut-poc/apps/st_ebs/README.md
# Streamlit app on AWS ElasticBeanstalk

## Step 1 - Create the streamlit app
Build the streamlit app and put all required python code inside ```app``` directory.
## Step 2 - Dockerize it
Git action will do this automatically. It is currently configured to trigger manually.
## Step 3 - Configure ```Dockerrun.aws.json```
Update the docker image name in this file.
Copy this file to public S3 location. We can use command like ```aws s3 cp ./apps/st_ebs/Dockerrun.aws.json s3://recotut/poc/``` to do this.
## Step 4 - Create Beanstalk App

```
!mkdir -p ~/.aws && cp /content/drive/MyDrive/AWS/d01_admin/* ~/.aws
!aws s3 cp /content/reco-tut-poc/apps/st_ebs/Dockerrun.aws.json s3://recotut/poc/
!aws elasticbeanstalk update-environment --application-name reco-tut-poc --environment-name Recotutpoc-env --version-label reco-tut-poc-v1
```

```
!pip install awsebcli --upgrade --user
%cd ./apps/st_ebs
!/root/.local/bin/eb --version
!/root/.local/bin/eb init
!/root/.local/bin/eb create
!/root/.local/bin/eb status
!/root/.local/bin/eb deploy
```
````

```python colab={"base_uri": "https://localhost:8080/"} id="ztZFcClLJC1_" executionInfo={"status": "ok", "timestamp": 1629529093635, "user_tz": -330, "elapsed": 59187, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="92408bab-bb15-4617-c174-5afc211d4dff"
# !pip install awsebcli --upgrade --user
# %cd ./apps/st_ebs
# !/root/.local/bin/eb --version
# !/root/.local/bin/eb init
# !/root/.local/bin/eb init -p docker reco-tut-poc-2
# !/root/.local/bin/eb create
# !/root/.local/bin/eb status
# !/root/.local/bin/eb deploy
```

```python colab={"base_uri": "https://localhost:8080/"} id="PmJSyJPlbHZ1" executionInfo={"status": "ok", "timestamp": 1629528978056, "user_tz": -330, "elapsed": 522, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8efeb9bd-9cc8-4acf-e7f7-2cbbbdf24327"
%%writefile ./.elasticbeanstalk/config.yml
branch-defaults:
  default:
    environment: reco-tut-poc-2-dev
deploy:
  artifact: Dockerrun.aws.json
environment-defaults:
  reco-tut-poc-2-dev:
    branch: null
    repository: null
global:
  application_name: reco-tut-poc-2
  branch: null
  default_ec2_keyname: null
  default_platform: Docker
  default_region: us-east-1
  include_git_submodules: true
  instance_profile: null
  platform_name: null
  platform_version: null
  profile: null
  repository: null
  sc: null
  workspace_type: Application
```

```python id="mQgSXsixenZe"

```
