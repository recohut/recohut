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

<!-- #region id="Kt-tbcBerP0Y" -->
# Amazon Personalize Generic Module - Model Layer
<!-- #endregion -->

```python id="C4xjNqJZcw2a"
!pip install -q boto3
```

```python id="1oZPIuJFcXN1"
!mkdir -p ~/.aws && cp /content/drive/MyDrive/AWS/d01_admin/* ~/.aws
```

<!-- #region id="vx2Q0KoDtTay" -->
In Amazon Personalize, a specific variation of an algorithm is called a recipe. Different recipes are suitable for different situations. A trained model is called a solution, and each solution can have many versions that relate to a given volume of data when the model was trained.
<!-- #endregion -->

<!-- #region id="G5UZT21LsLrU" -->
In this notebook, you will train several models using Amazon Personalize, specifically:
1. User Personalization - what items are most relevant to a specific user.
2. Similar Items - given an item, what items are similar to it.
3. Personalized Ranking - given a user and a collection of items, in what order are they most releveant.
<!-- #endregion -->

```python id="pP4Hab1Gt94Y"
import pickle
import time
from time import sleep
import json
from datetime import datetime
import uuid
import random
import pandas as pd

import boto3
import botocore
from botocore.exceptions import ClientError

from generic_modules.import_dataset import personalize_dataset
```

```python id="z1zTY8rOGWyz"
# Loading old state containing resouce arns
with open('./artifacts/etc/personalize_item_meta.pkl', 'rb') as outp:
    personalize_item_meta = pickle.load(outp)
```

```python id="x_cde5DTHbS-"
# Configure the SDK to Personalize
personalize = boto3.client('personalize')
personalize_runtime = boto3.client('personalize-runtime')
```

```python colab={"base_uri": "https://localhost:8080/"} id="_g4x5LwCutEr" executionInfo={"status": "ok", "timestamp": 1630170620896, "user_tz": -330, "elapsed": 513, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0f590413-3b80-40a5-8b82-64568c6ae86e"
personalize.list_recipes()
```

```python id="B4pLA79DvZ5g"
# define the solution arn
user_personalization_recipe_arn = "arn:aws:personalize:::recipe/aws-user-personalization"
```

```python colab={"base_uri": "https://localhost:8080/"} id="QwbWV-Y7vjBy" executionInfo={"status": "ok", "timestamp": 1630170784469, "user_tz": -330, "elapsed": 445, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c83aee67-cba5-4a06-d12f-91f255e44bca"
# Create the solution
"""First you create a solution using the recipe. Although you provide
the dataset ARN in this step, the model is not yet trained. See this as
an identifier instead of a trained model."""

user_personalization_create_solution_response = personalize.create_solution(
    name = "personalize-poc-userpersonalization",
    datasetGroupArn = personalize_item_meta.dataset_group_arn,
    recipeArn = user_personalization_recipe_arn
)

user_personalization_solution_arn = user_personalization_create_solution_response['solutionArn']
user_personalization_solution_arn
```

<!-- #region id="MP8FE7XEwdD4" -->
The training can take a while to complete,
upwards of 25 minutes, and an average of 90 minutes for this recipe with
our dataset. Normally, we would use a while loop to poll until the task is
completed. However the task would block other cells from executing, and the
goal here is to create many models and deploy them quickly. So we will set
up the while loop for all of the solutions further down in the notebook. There,
you will also find instructions for viewing the progress in the AWS console.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="uqQnH0pjwC6E" executionInfo={"status": "ok", "timestamp": 1630170926788, "user_tz": -330, "elapsed": 629, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cd3e71e0-7986-4b29-a726-07d3d4d11038"
# Create the solution version
"""Once you have a solution, you need to create a version in order to
complete the model training."""

userpersonalization_create_solution_version_response = personalize.create_solution_version(
    solutionArn = user_personalization_solution_arn
)
userpersonalization_solution_version_arn = userpersonalization_create_solution_version_response['solutionVersionArn']
userpersonalization_solution_version_arn
```

<!-- #region id="lnlLEbuXw1NU" -->
### SIMS Recipe
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="7F_vg302w2HX" executionInfo={"status": "ok", "timestamp": 1630171065784, "user_tz": -330, "elapsed": 724, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4ae16fa7-de66-426e-9d65-c6489d091502"
SIMS_recipe_arn = "arn:aws:personalize:::recipe/aws-sims"

sims_create_solution_response = personalize.create_solution(
    name = "personalize-poc-sims",
    datasetGroupArn = personalize_item_meta.dataset_group_arn,
    recipeArn = SIMS_recipe_arn
)

sims_solution_arn = sims_create_solution_response['solutionArn']
sims_solution_arn
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="_N-KmDyUxC08" executionInfo={"status": "ok", "timestamp": 1630171068910, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="82c9c5c0-dde2-48bf-b8cd-d502543a6d1a"
sims_create_solution_version_response = personalize.create_solution_version(
    solutionArn = sims_solution_arn
)

sims_solution_version_arn = sims_create_solution_version_response['solutionVersionArn']
sims_solution_version_arn
```

<!-- #region id="TsoFHQnKxHWn" -->
### Ranking Recipe
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="Hh7b_Sl8xLdl" executionInfo={"status": "ok", "timestamp": 1630171139163, "user_tz": -330, "elapsed": 507, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b7b211b2-b14d-41f5-8902-3eb230433773"
rerank_recipe_arn = "arn:aws:personalize:::recipe/aws-personalized-ranking"

rerank_create_solution_response = personalize.create_solution(
    name = "personalize-poc-rerank",
    datasetGroupArn = personalize_item_meta.dataset_group_arn,
    recipeArn = rerank_recipe_arn
)

rerank_solution_arn = rerank_create_solution_response['solutionArn']
rerank_solution_arn
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="P63rp7XPxYdM" executionInfo={"status": "ok", "timestamp": 1630171139869, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aa26f8e1-bce2-46d0-da05-c225de21db11"
rerank_create_solution_version_response = personalize.create_solution_version(
    solutionArn = rerank_solution_arn
)
rerank_solution_version_arn = rerank_create_solution_version_response['solutionVersionArn']
rerank_solution_version_arn
```

<!-- #region id="Z3yTII-4xYtL" -->
## Check the status
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Sd5a7KN5yYqS" executionInfo={"status": "ok", "timestamp": 1630172427535, "user_tz": -330, "elapsed": 1023329, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f915e923-0ccb-470a-8700-cdb71659a813"
in_progress_solution_versions = [
    userpersonalization_solution_version_arn,
    sims_solution_version_arn,
    rerank_solution_version_arn
]

max_time = time.time() + 10*60*60 # 10 hours
while time.time() < max_time:
    for solution_version_arn in in_progress_solution_versions:
        version_response = personalize.describe_solution_version(
            solutionVersionArn = solution_version_arn
        )
        status = version_response["solutionVersion"]["status"]
        
        if status == "ACTIVE":
            print("Build succeeded for {}".format(solution_version_arn))
            in_progress_solution_versions.remove(solution_version_arn)
        elif status == "CREATE FAILED":
            print("Build failed for {}".format(solution_version_arn))
            in_progress_solution_versions.remove(solution_version_arn)
    
    if len(in_progress_solution_versions) <= 0:
        break
    else:
        print("At least one solution build is still in progress")
        
    time.sleep(60)
```

<!-- #region id="PR1mBKFdyzyd" -->
## Hyperparameter tuning
<!-- #endregion -->

```python id="aYPt8hogyZOE"
sims_create_solution_response = personalize.create_solution(
    name = "personalize-poc-sims-hpo",
    datasetGroupArn = dataset_group_arn,
    recipeArn = SIMS_recipe_arn,
    performHPO=True
)
```

<!-- #region id="QtzYaUxCy677" -->
If you already know the values you want to use for a specific hyperparameter, you can also set this value when you create the solution. The code below shows how you could set the value for the popularity_discount_factor for the SIMS recipe.
<!-- #endregion -->

```python id="DRZiRDMcy6Dl"
sims_create_solution_response = personalize.create_solution(
    name = "personalize-poc-sims-set-hp",
    datasetGroupArn = dataset_group_arn,
    recipeArn = SIMS_recipe_arn,
    solutionConfig = {
        'algorithmHyperParameters': {
            'popularity_discount_factor': '0.7'
        }
    }
)
```

<!-- #region id="qhLIGXA5zaPl" -->
## Evaluation
<!-- #endregion -->

```python id="a08tGbfeza_J"
user_personalization_solution_metrics_response = personalize.get_solution_metrics(
    solutionVersionArn = userpersonalization_solution_version_arn
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="NAhkVimaBxZD" executionInfo={"status": "ok", "timestamp": 1630175436268, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f5700418-b1c6-44a7-9768-dbb0d363c43b"
user_personalization_solution_metrics_response
```

<!-- #region id="ke4wQiRpzq8c" -->
## Deploying Campaigns and Filters
<!-- #endregion -->

```python id="jmuEQiYbzuK_"
# Establish a connection to Personalize's event streaming
personalize_events = boto3.client(service_name='personalize-events')
```

```python id="hNPOYILHz-Ig"
userpersonalization_create_campaign_response = personalize.create_campaign(
    name = "personalize-poc-userpersonalization",
    solutionVersionArn = userpersonalization_solution_version_arn,
    minProvisionedTPS = 1
)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="7rcUZAsECCwZ" executionInfo={"status": "ok", "timestamp": 1630175511987, "user_tz": -330, "elapsed": 428, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="49aa11e9-6c87-4328-fbd6-e377e5bca19e"
userpersonalization_campaign_arn = userpersonalization_create_campaign_response['campaignArn']
userpersonalization_campaign_arn
```

<!-- #region id="NIWvKPZL0gFZ" -->
## View Campaign Creation Status
<!-- #endregion -->

```python id="l0GuG1T80ie7"
in_progress_campaigns = [
    userpersonalization_campaign_arn
]

max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    for campaign_arn in in_progress_campaigns:
        version_response = personalize.describe_campaign(
            campaignArn = campaign_arn
        )
        status = version_response["campaign"]["status"]
        
        if status == "ACTIVE":
            print("Build succeeded for {}".format(campaign_arn))
            in_progress_campaigns.remove(campaign_arn)
        elif status == "CREATE FAILED":
            print("Build failed for {}".format(campaign_arn))
            in_progress_campaigns.remove(campaign_arn)
    
    if len(in_progress_campaigns) <= 0:
        break
    else:
        print("At least one campaign build is still in progress")
        
    time.sleep(60)
```

<!-- #region id="-B6-JJ-u0lVQ" -->
## Create Static Filters
<!-- #endregion -->

```python id="o3Q_k-8X0mxe"
creategenrefilter_response = personalize.create_filter(name='Genre',
    datasetGroupArn=dataset_group_arn,
    filterExpression='INCLUDE ItemID WHERE Items.GENRE IN ($GENRE)'
    )
genre_filter_arn = creategenrefilter_response['filterArn']

decades_to_filter = [1950,1960,1970,1980,1990,2000,2010]

# Create a list for the filters:
meta_filter_decade_arns = []

# Iterate through Genres
for decade in decades_to_filter:
    # Start by creating a filter
    current_decade = str(decade)
    next_decade = str(decade + 10)
    try:
        createfilter_response = personalize.create_filter(
            name=current_decade + "s",
            datasetGroupArn=dataset_group_arn,
            filterExpression='INCLUDE ItemID WHERE Items.YEAR >= '+ current_decade +' AND Items.YEAR < '+ next_decade +''
    )
        # Add the ARN to the list
        meta_filter_decade_arns.append(createfilter_response['filterArn'])
        print("Creating: " + createfilter_response['filterArn'])
    
    # If this fails, wait a bit
    except ClientError as error:
        # Here we only care about raising if it isnt the throttling issue
        if error.response['Error']['Code'] != 'LimitExceededException':
            print(error)
        else:    
            time.sleep(120)
            createfilter_response = personalize.create_filter(
                name=current_decade + "s",
                datasetGroupArn=dataset_group_arn,
                filterExpression='INCLUDE ItemID WHERE Items.YEAR >= '+ current_decade +' AND Items.YEAR < '+ next_decade +''
    )
            # Add the ARN to the list
            meta_filter_decade_arns.append(createfilter_response['filterArn'])
            print("Creating: " + createfilter_response['filterArn'])
```

```python id="RNgJOseH1JIx"
# Lets also create 2 event filters for watched and unwatched content

createwatchedfilter_response = personalize.create_filter(name='watched',
    datasetGroupArn=dataset_group_arn,
    filterExpression='INCLUDE ItemID WHERE Interactions.event_type IN ("watch")'
    )

createunwatchedfilter_response = personalize.create_filter(name='unwatched',
    datasetGroupArn=dataset_group_arn,
    filterExpression='EXCLUDE ItemID WHERE Interactions.event_type IN ("watch")'
    )

interaction_filter_arns = [createwatchedfilter_response['filterArn'], createunwatchedfilter_response['filterArn']]
```

<!-- #region id="bMiFBo991pw5" -->
## Generic Module
<!-- #endregion -->

```python id="WBHCudKc1raD"
from generic_modules.import_model import personalize_model
```

<!-- #region id="nkLk2nB6FU_3" -->
### User Personalization Model
<!-- #endregion -->

```python id="So4eBquiFBhJ"
dataset_group_arn = 'arn:aws:personalize:us-east-1:746888961694:dataset-group/immersion-day-dataset-group-movielens-latest'
solution_arn = 'arn:aws:personalize:us-east-1:746888961694:solution/personalize-poc-sims'
solution_version_arn = 'arn:aws:personalize:us-east-1:746888961694:solution/personalize-poc-sims/edb3d46e'
campaign_arn = 'arn:aws:personalize:us-east-1:746888961694:campaign/personalize-poc-userpersonalization'
```

```python id="qMBwqrkhFBcS"
pers_model = personalize_model(dataset_group_arn = dataset_group_arn,
                               solution_arn = solution_arn,
                               solution_version_arn = solution_version_arn,
                               campaign_arn = campaign_arn)
```

```python colab={"base_uri": "https://localhost:8080/"} id="bs-DjrcgF-ib" executionInfo={"status": "ok", "timestamp": 1630179063536, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="51e63436-503a-4d06-d8fc-8645f038b870"
pers_model.setup_connection()
```

```python colab={"base_uri": "https://localhost:8080/"} id="2PogGtHcGFd5" executionInfo={"status": "ok", "timestamp": 1630179067727, "user_tz": -330, "elapsed": 735, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3217f436-2c0b-46ec-aae3-a74e09e786c8"
pers_model.recipe_list()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="y3-ZfAy2FREp" executionInfo={"status": "ok", "timestamp": 1630179068415, "user_tz": -330, "elapsed": 696, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7f22e681-c160-44da-d7ff-a1b894dd6724"
pers_model.check_campaign_creation_status()
```

```python id="4kSS-Sv0Olxs"
pers_model.create_filter(name='Genre', expression='INCLUDE ItemID WHERE Items.GENRE IN ($GENRE)')
pers_model.create_filter(name='watched', expression='INCLUDE ItemID WHERE Interactions.event_type IN ("watch")')
pers_model.create_filter(name='unwatched', expression='EXCLUDE ItemID WHERE Interactions.event_type IN ("watch")')
```

<!-- #region id="19dBOcc8PzlM" -->
<!-- #endregion -->

<!-- #region id="BXYQcdpsFBYT" -->
### SIMS Model
<!-- #endregion -->

```python id="SJWEhp_lESNW"
dataset_group_arn = 'arn:aws:personalize:us-east-1:746888961694:dataset-group/immersion-day-dataset-group-movielens-latest'
solution_arn = 'arn:aws:personalize:us-east-1:746888961694:solution/personalize-poc-sims'
solution_version_arn = 'arn:aws:personalize:us-east-1:746888961694:solution/personalize-poc-sims/edb3d46e'
```

```python id="4KNxZdOaEBab"
sims_model = personalize_model(dataset_group_arn = dataset_group_arn,
                               solution_arn = solution_arn,
                               solution_version_arn = solution_version_arn)
```

```python colab={"base_uri": "https://localhost:8080/"} id="1pwlsB7rG_G1" executionInfo={"status": "ok", "timestamp": 1630179471709, "user_tz": -330, "elapsed": 626, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="819a88b6-822d-468e-a2d4-cbdc31449d31"
sims_model.setup_connection()
```

```python colab={"base_uri": "https://localhost:8080/"} id="FBLgN0lVE433" executionInfo={"status": "ok", "timestamp": 1630179296022, "user_tz": -330, "elapsed": 681, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="faa59be6-996c-4823-809f-b3cc8fcde60d"
sims_model.get_evaluation_metrics()['metrics']
```

```python id="jjL3Rs04G92e"
sims_model.create_campaign(name='personalize-poc-sims-campaign')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="LLQ4KsAMHszO" executionInfo={"status": "ok", "timestamp": 1630179475228, "user_tz": -330, "elapsed": 645, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="eb0fd625-669f-4cef-93ba-adebbc430ba7"
sims_model.check_campaign_creation_status()
```

<!-- #region id="wD-d4ZdnJasJ" -->
### Popularity Model
<!-- #endregion -->

```python id="sCvbfI0iJvwU"
dataset_group_arn = 'arn:aws:personalize:us-east-1:746888961694:dataset-group/immersion-day-dataset-group-movielens-latest'
```

```python id="HKXUty7JJvwV"
pop_model = personalize_model(dataset_group_arn = dataset_group_arn)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ebzHhXvZJ0Ci" executionInfo={"status": "ok", "timestamp": 1630179604071, "user_tz": -330, "elapsed": 444, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="05ce6973-361b-4e3d-a30f-8c1e0edec816"
pop_model.setup_connection()
```

```python colab={"base_uri": "https://localhost:8080/"} id="1VHljedrJ24j" executionInfo={"status": "ok", "timestamp": 1630177566106, "user_tz": -330, "elapsed": 616, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="01a268b2-d3fa-4b8a-f71e-9bee249b4414"
pop_model.recipe_list()
```

```python id="JeB8mAw3KCzT"
pop_model.create_solution(name='aws-popularity-count')
```

```python id="ffwZch3QKTCi"
pop_model.create_solution_version()
```

<!-- #region id="jRSB6noIKelO" -->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="X1DyBLMwKWDr" executionInfo={"status": "ok", "timestamp": 1630177693894, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ee9452bc-2ef0-4efc-f330-f8233d7219a1"
pop_model.check_solution_version_status()
```

```python colab={"base_uri": "https://localhost:8080/"} id="mdTDH11bLE9A" executionInfo={"status": "ok", "timestamp": 1630178748944, "user_tz": -330, "elapsed": 721732, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4e252ac1-670e-4735-d4fe-03599adf144e"
max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time: 
    status = pop_model.check_solution_version_status()
    print(status)
    if status=='ACTIVE':
        break
    time.sleep(30)
```

<!-- #region id="a0GcGp5lLhbr" -->
### Save States
<!-- #endregion -->

```python id="r00laugjP3NS"
with open('./artifacts/etc/personalize_model_pers.pkl', 'wb') as outp:
    pickle.dump(pers_model, outp, pickle.HIGHEST_PROTOCOL)
```

```python id="lHiLMdpFQRLO"
with open('./artifacts/etc/personalize_model_sims.pkl', 'wb') as outp:
    pickle.dump(sims_model, outp, pickle.HIGHEST_PROTOCOL)
```

```python id="aiGsmDGLQZfo"
with open('./artifacts/etc/personalize_model_pop.pkl', 'wb') as outp:
    pickle.dump(pop_model, outp, pickle.HIGHEST_PROTOCOL)
```

```python colab={"base_uri": "https://localhost:8080/"} id="dFp4YXhLQZ1l" executionInfo={"status": "ok", "timestamp": 1630179638082, "user_tz": -330, "elapsed": 584, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e8ab05d3-f0e9-4282-d94b-5c9e0066f234"
pers_model.__getstate__()
```

```python colab={"base_uri": "https://localhost:8080/"} id="rA10BHEdRzZV" executionInfo={"status": "ok", "timestamp": 1630179647399, "user_tz": -330, "elapsed": 652, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2bb92edb-e7c7-4265-9f11-ef5f56b38d8e"
sims_model.__getstate__()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6R61f4SoR1mX" executionInfo={"status": "ok", "timestamp": 1630179651848, "user_tz": -330, "elapsed": 458, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="20aa6e7e-b7d3-4b05-e1b5-357b47a5255e"
pop_model.__getstate__()
```

```python id="1cHXYuKEfopX"
import boto3
import json
import time


class personalize_inference:
    def __init__(self,
                 dataset_group_arn = None,
                 campaign_arn = None,
                 event_tracker_arn = None,
                 role_arn=None,
                 solution_version_arn=None,
                 batch_job_arn=None
                 ):
        self.personalize = None
        self.personalize_runtime = None
        self.personalize_events = None
        self.dataset_group_arn = dataset_group_arn
        self.campaign_arn = campaign_arn
        self.event_tracker_arn = event_tracker_arn
        self.event_tracker_id = event_tracker_id
        self.role_arn = role_arn
        self.solution_version_arn = solution_version_arn
        self.batch_job_arn = batch_job_arn

    def setup_connection(self):
        try:
            self.personalize = boto3.client('personalize')
            self.personalize_runtime = boto3.client('personalize-runtime')
            self.personalize_events = boto3.client(service_name='personalize-events')
            print("SUCCESS | We can communicate with Personalize!")
        except:
            print("ERROR | Connection can't be established!")

    def get_recommendations(self, itemid=None, userid=None, k=5,
                            filter_arn=None, filter_values=None):
        get_recommendations_response = self.personalize_runtime.get_recommendations(
            campaignArn = self.campaign_arn,
            itemId = str(itemid),
            userId = str(userid),
            filterArn = filter_arn,
            filterValues = filter_values,
            numResults = k
            )
        
    def get_rankings(self, userid=None, inputlist=None):
        get_recommendations_response = self.personalize_runtime.get_personalized_ranking(
            campaignArn = self.campaign_arn,
            userId = str(userid),
            inputList = inputlist
            )
        
    def create_event_tracker(self, name=None):
        response = self.personalize.create_event_tracker(
            name=name,
            datasetGroupArn=self.dataset_group_arn
            )
        self.event_tracker_arn = response['eventTrackerArn']
        self.event_tracker_id = response['trackingId']
    
    def put_events(self, userid=None, sessionid=None, eventlist=None):
        self.personalize_events.put_events(
            trackingId = self.event_tracker_id,
            userId = userid, 
            sessionId = sessionid,
            eventList = eventlist
            )
        
    def create_batch_job(self, jobname=None, input_path=None, output_path=None):
        response = self.personalize.create_batch_inference_job(
            solutionVersionArn = self.solution_version_arn,
            jobName = jobname,
            roleArn = self.role_arn,
            jobInput = {"s3DataSource": {"path": input_path}},
            jobOutput = {"s3DataDestination": {"path": output_path}}
            )
        self.batch_job_arn = response['batchInferenceJobArn']
    
    def check_batch_job_status(self):
        batch_job_response = self.personalize.describe_batch_inference_job(
        batchInferenceJobArn = self.batch_job_arn
        )
        status = batch_job_response["batchInferenceJob"]['status']
        return status

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['personalize']
        del attributes['personalize_runtime']
        del attributes['personalize_events']
        return attributes
```
