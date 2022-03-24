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

<!-- #region id="K81MlTZ4J2Cx" -->
# Operationalize end-to-end Amazon Personalize model deployment process using AWS Step Functions Data Science SDK

1. [Introduction](#Introduction)
2. [Setup](#Setup)
3. [Task-States](#Task-States)
4. [Wait-States](#Wait-States)
5. [Choice-States](#Choice-States)
6. [Workflow](#Workflow)
7. [Generate-Recommendations](#Generate-Recommendations)


<!-- #endregion -->

<!-- #region id="qAClB9a8J2C5" -->
## Introduction

This notebook describes using the AWS Step Functions Data Science SDK to create and manage an Amazon Personalize workflow. The Step Functions SDK is an open source library that allows data scientists to easily create and execute machine learning workflows using AWS Step Functions. For more information on Step Functions SDK, see the following.
* [AWS Step Functions](https://aws.amazon.com/step-functions/)
* [AWS Step Functions Developer Guide](https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html)
* [AWS Step Functions Data Science SDK](https://aws-step-functions-data-science-sdk.readthedocs.io)

In this notebook we will use the SDK to create steps to create Personalize resources, link them together to create a workflow, and execute the workflow in AWS Step Functions. 

For more information, on Amazon Personalize see the following.

* [Amazon Personalize](https://aws.amazon.com/personalize/)

<!-- #endregion -->

<!-- #region id="wGyw9o-NJ2C8" -->
## Setup
<!-- #endregion -->

<!-- #region id="HKpgQxE7J2C-" -->
### Import required modules from the SDK
<!-- #endregion -->

```python id="ccBDjF7vIvvv"
!mkdir -p ~/.aws && cp /content/drive/MyDrive/AWS/d01_admin/* ~/.aws
```

```python id="glGX0q2SGqV9"
!pip install -q boto3
```

```python id="iLG-4lmeJihL"
!pip install -q stepfunctions
```

```python id="cuz-quS0KTtE"
import boto3
import json
import numpy as np
import pandas as pd
import time
import logging

import stepfunctions
from stepfunctions.steps import *
from stepfunctions.workflow import Workflow
```

```python id="MzYEDVPkJl51"
personalize = boto3.client('personalize')
personalize_runtime = boto3.client('personalize-runtime')
stepfunctions.set_stream_logger(level=logging.INFO)
```

```python colab={"base_uri": "https://localhost:8080/"} id="nj8sUJyEqjRV" executionInfo={"status": "ok", "timestamp": 1629783598934, "user_tz": -330, "elapsed": 523, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="82c0bd55-f04e-4e06-a60c-3b9e2151d421"
!git clone https://github.com/aws-samples/personalize-data-science-sdk-workflow.git
%cd personalize-data-science-sdk-workflow
```

<!-- #region id="HVarZRLNJ2DD" -->
### Setup S3 location and filename
create an Amazon S3 bucket to store the training dataset and provide the Amazon S3 bucket name and file name in the walkthrough notebook  step Setup S3 location and filename below:
<!-- #endregion -->

```python id="WfrLHb5Rntt7"
bucket = "reco-tut-aps"       # replace with the name of your S3 bucket
filename = "namemovie-lens-100k.csv"  # replace with a name that you want to save the dataset under
```

<!-- #region id="WhYT5n4WJ2DH" -->
### Setup IAM Roles

#### Create an execution role for Step Functions

You need an execution role so that you can create and execute workflows in Step Functions.

1. Go to the [IAM console](https://console.aws.amazon.com/iam/)
2. Select **Roles** and then **Create role**.
3. Under **Choose the service that will use this role** select **Step Functions**
4. Choose **Next** until you can enter a **Role name**
5. Enter a name such as `StepFunctionsWorkflowExecutionRole` and then select **Create role**


Attach a policy to the role you created. The following steps attach a policy that provides full access to Step Functions, however as a good practice you should only provide access to the resources you need.  

1. Under the **Permissions** tab, click **Add inline policy**
2. Enter the following in the **JSON** tab

```json
{
    "Version": "2012-10-17",
    "Statement": [
    
        {
            "Effect": "Allow",
            "Action": [
                "personalize:*"
            ],
            "Resource": "*"
        },   

        {
            "Effect": "Allow",
            "Action": [
                "lambda:InvokeFunction"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:PassRole"
            ],
            "Resource": "*",
        },
        {
            "Effect": "Allow",
            "Action": [
                "events:PutTargets",
                "events:PutRule",
                "events:DescribeRule"
            ],
            "Resource": "*"
        }
    ]
}
```

3. Choose **Review policy** and give the policy a name such as `StepFunctionsWorkflowExecutionPolicy`
4. Choose **Create policy**. You will be redirected to the details page for the role.
5. Copy the **Role ARN** at the top of the **Summary**


<!-- #endregion -->

```python id="I1pQH4QupYOR"
workflow_execution_role = "arn:aws:iam::746888961694:role/StepFunctionsWorkflowExecutionRole" # paste the StepFunctionsWorkflowExecutionRole ARN from above
```

```python id="sCRfs1XrpeXn"
lambda_state_role = LambdaStep(
    state_id="create bucket and role",
    parameters={  
        "FunctionName": "stepfunction_create_personalize_role", #replace with the name of the function you created
        "Payload": {  
           "bucket": bucket
        }
    },
    result_path='$'
 
)

lambda_state_role.add_retry(Retry(
    error_equals=["States.TaskFailed"],
    interval_seconds=5,
    max_attempts=1,
    backoff_rate=4.0
))

lambda_state_role.add_catch(Catch(
    error_equals=["States.TaskFailed"],
    next_step=Fail("CreateRoleTaskFailed")
))
```

<!-- #region id="XyKX8VzwJ2DO" -->
#### Attach Policy to S3 Bucket
<!-- #endregion -->

```python id="Z_doAlx5J2DO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629783731289, "user_tz": -330, "elapsed": 2003, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="64c63124-ad2c-43fa-f76f-3afc18ca099b"
s3 = boto3.client("s3")

policy = {
    "Version": "2012-10-17",
    "Id": "PersonalizeS3BucketAccessPolicy",
    "Statement": [
        {
            "Sid": "PersonalizeS3BucketAccessPolicy",
            "Effect": "Allow",
            "Principal": {
                "Service": "personalize.amazonaws.com"
            },
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::{}".format(bucket),
                "arn:aws:s3:::{}/*".format(bucket)
                
            ]
        }
    ]
}

s3.put_bucket_policy(Bucket=bucket, Policy=json.dumps(policy))

# AmazonPersonalizeFullAccess provides access to any S3 bucket with a name that includes "personalize" or "Personalize" 
# if you would like to use a bucket with a different name, please consider creating and attaching a new policy
# that provides read access to your bucket or attaching the AmazonS3ReadOnlyAccess policy to the role
```

<!-- #region id="jZl3hpHIJ2DR" -->
#### Create Personalize Role

<!-- #endregion -->

```python id="vsRt2r1jJ2DS" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1629783896480, "user_tz": -330, "elapsed": 742, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6e23a954-e80e-4ce0-b736-4963c6cca740"
iam = boto3.client("iam")

role_name = "personalize-role" # Create a personalize role

assume_role_policy_document = {
    "Version": "2012-10-17",
    "Statement": [
        {
          "Effect": "Allow",
          "Principal": {
            "Service": "personalize.amazonaws.com"
          },
          "Action": "sts:AssumeRole"
        }
    ]
}

create_role_response = iam.create_role(
    RoleName = role_name,
    AssumeRolePolicyDocument = json.dumps(assume_role_policy_document)
)

policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonPersonalizeFullAccess"
iam.attach_role_policy(
    RoleName = role_name,
    PolicyArn = policy_arn
)

time.sleep(60) # wait for a minute to allow IAM role policy attachment to propagate

role_arn = create_role_response["Role"]["Arn"]
role_arn
```

```python id="u1hGPMbbJ2DT"
role_arn = "arn:aws:iam::746888961694:role/personalize-role"
```

<!-- #region id="jCV-PvhcJ2DT" -->
## Data-Preparation
<!-- #endregion -->

<!-- #region id="XWtmxyFdJ2DT" -->
### Download, Prepare, and Upload Training Data
<!-- #endregion -->

```python id="5vmvfKd-J2DU" colab={"base_uri": "https://localhost:8080/", "height": 816} executionInfo={"status": "ok", "timestamp": 1629783985307, "user_tz": -330, "elapsed": 1844, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1a35ed69-0252-4c2f-ad17-369080c5da69"
!wget -N http://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip -o ml-100k.zip
data = pd.read_csv('./ml-100k/u.data', sep='\t', names=['USER_ID', 'ITEM_ID', 'RATING', 'TIMESTAMP'])
pd.set_option('display.max_rows', 5)
data.head()
```

```python id="OLo1rRrcJ2DV"
data = data[data['RATING'] > 2]                # keep only movies rated 2 and above
data2 = data[['USER_ID', 'ITEM_ID', 'TIMESTAMP']] 
data2.to_csv(filename, index=False)

boto3.Session().resource('s3').Bucket(bucket).Object(filename).upload_file(filename)
```

<!-- #region id="yBpHHwSRJ2DV" -->
## Task-States
<!-- #endregion -->

<!-- #region id="O200GRvTJ2DW" -->
### Lambda Task state

A `Task` State in Step Functions represents a single unit of work performed by a workflow. Tasks can call Lambda functions and orchestrate other AWS services. See [AWS Service Integrations](https://docs.aws.amazon.com/step-functions/latest/dg/concepts-service-integrations.html) in the *AWS Step Functions Developer Guide*.

The following creates a [LambdaStep](https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/compute.html#stepfunctions.steps.compute.LambdaStep) called `lambda_state`, and then configures the options to [Retry](https://docs.aws.amazon.com/step-functions/latest/dg/concepts-error-handling.html#error-handling-retrying-after-an-error) if the Lambda function fails.

#### Create a Lambda functions

The Lambda task states in this workflow uses Lambda function **(Python 3.x)** that returns a Personalize resources such as Schema, Datasetgroup, Dataset, Solution, SolutionVersion, etc. Create the following functions in the [Lambda console](https://console.aws.amazon.com/lambda/).

1. stepfunction-create-schema
2. stepfunctioncreatedatagroup
3. stepfunctioncreatedataset
4. stepfunction-createdatasetimportjob
5. stepfunction_select-recipe_create-solution
6. stepfunction_create_solution_version
7. stepfunction_getsolution_metric_create_campaign

Copy/Paste the corresponding lambda function code from ./Lambda/ folder in the repo

<!-- #endregion -->

<!-- #region id="elVncKGOJ2DX" -->
#### Create Schema

Before you add a dataset to Amazon Personalize, you must define a schema for that dataset. Once you define the schema and create the dataset, you can't make changes to the schema.for more information refer this documentation.
<!-- #endregion -->

```python id="UMqA0zdUJ2DY"
lambda_state_schema = LambdaStep(
    state_id="create schema",
    parameters={  
        "FunctionName": "stepfunction-create-schema", #replace with the name of the function you created
        "Payload": {  
           "input": "personalize-stepfunction-schema3484"
        }
    },
    result_path='$'    
)

lambda_state_schema.add_retry(Retry(
    error_equals=["States.TaskFailed"],
    interval_seconds=5,
    max_attempts=1,
    backoff_rate=4.0
))

lambda_state_schema.add_catch(Catch(
    error_equals=["States.TaskFailed"],
    next_step=Fail("CreateSchemaTaskFailed")
))
```

<!-- #region id="oAjTD8KEJ2DZ" -->
#### Create Datasetgroup

Craete Datasetgroup: Creates an empty dataset group. A dataset group contains related datasets that supply data for training a model. A dataset group can contain at most three datasets, one for each type of dataset:
•	Interactions
•	Items
•	Users
To train a model (create a solution), a dataset group that contains an Interactions dataset is required. Call CreateDataset to add a dataset to the group.

After you have created a schema , we will create another Stepfunction state based on this lambda function stepfunctioncreatedatagroup.py  below in github lambdas folder by running the Create Datasetgroup¶ step of the notebook. We are using python boto3 APIs to create_dataset_group.
<!-- #endregion -->

```python id="j4_o-p77J2Da"
lambda_state_datasetgroup = LambdaStep(
    state_id="create dataset Group",
    parameters={  
        "FunctionName": "stepfunctioncreatedatagroup", #replace with the name of the function you created
        "Payload": {  
           "input": "personalize-stepfunction-dataset-group", 
           "schemaArn.$": '$.Payload.schemaArn'
        }
    },

    result_path='$'
)



lambda_state_datasetgroup.add_retry(Retry(
    error_equals=["States.TaskFailed"],
    interval_seconds=5,
    max_attempts=1,
    backoff_rate=4.0
))


lambda_state_datasetgroup.add_catch(Catch(
    error_equals=["States.TaskFailed"],
    next_step=Fail("CreateDataSetGroupTaskFailed")
))
```

<!-- #region id="XQ1PtIUyJ2Db" -->
#### Create Dataset

Creates an empty dataset and adds it to the specified dataset group. Use CreateDatasetImportJob to import your training data to a dataset.

There are three types of datasets:

Interactions

Items

Users

Each dataset type has an associated schema with required field types. Only the Interactions dataset is required in order to train a model (also referred to as creating a solution).
<!-- #endregion -->

```python id="UWLRhD7IJ2Dc"
lambda_state_createdataset = LambdaStep(
    state_id="create dataset",
    parameters={  
        "FunctionName": "stepfunctioncreatedataset", #replace with the name of the function you created
#        "Payload": {  
#           "schemaArn.$": '$.Payload.schemaArn',
#           "datasetGroupArn.$": '$.Payload.datasetGroupArn',
            
            
#        }
        
        "Payload": {  
           "schemaArn.$": '$.schemaArn',
           "datasetGroupArn.$": '$.datasetGroupArn',        
        } 
        
        
    },
    result_path = '$'
)

lambda_state_createdataset.add_retry(Retry(
    error_equals=["States.TaskFailed"],
    interval_seconds=5,
    max_attempts=1,
    backoff_rate=4.0
))

lambda_state_createdataset.add_catch(Catch(
    error_equals=["States.TaskFailed"],
    next_step=Fail("CreateDataSetTaskFailed")
))
```

<!-- #region id="SMetj_NqJ2Dc" -->
#### Create Dataset Import Job

When you have completed Step 1: Creating a Dataset Group and Step 2: Creating a Dataset and a Schema, you are ready to import your training data into Amazon Personalize. When you import data, you can choose to import records in bulk, import records individually, or both, depending on your business requirements and the amount of historical data you have collected. If you have a large amount of historical records, 
we recommend you first import data in bulk and then add data incrementally as necessary.
<!-- #endregion -->

```python id="DP4zbP-WJ2Dd"
lambda_state_datasetimportjob = LambdaStep(
    state_id="create dataset import job",
    parameters={  
        "FunctionName": "stepfunction-createdatasetimportjob", #replace with the name of the function you created
        "Payload": {  
           "datasetimportjob": "stepfunction-createdatasetimportjob",
           "dataset_arn.$": '$.Payload.dataset_arn',
           "datasetGroupArn.$": '$.Payload.datasetGroupArn',
           "bucket_name": bucket,
           "file_name": filename,
           "role_arn": role_arn
            
        }
    },

    result_path = '$'
)

lambda_state_datasetimportjob.add_retry(Retry(
    error_equals=["States.TaskFailed"],
    interval_seconds=5,
    max_attempts=1,
    backoff_rate=4.0
))

lambda_state_datasetimportjob.add_catch(Catch(
    error_equals=["States.TaskFailed"],
    next_step=Fail("DatasetImportJobTaskFailed")
))
```

<!-- #region id="3qpJV8laJ2Dd" -->
#### Create Solution

Once you have finished Preparing and Importing Data, you are ready to create a Solution. A Solution refers to the combination of an Amazon Personalize recipe, customized parameters, and one or more solution versions (trained models). Once you create a solution with a solution version, you can create a campaign to deploy the solution version and get recommendations.

To create a solution in Amazon Personalize, you do the following:

Choose a recipe – A recipe is an Amazon Personalize term specifying an appropriate algorithm to train for a given use case. See Step 1: Choosing a Recipe.

Configure a solution – Customize solution parameters and recipe-specific hyperparameters so the model meets your specific business needs. See Step 2: Configuring a Solution.

Create a solution version (train a model) – Train the machine learning model Amazon Personalize will use to generate recommendations for your customers. See Step 3: Creating a Solution Version.

Evaluate the solution version – Use the metrics Amazon Personalize generates from the new solution version to evaluate the performance of the model. See Step 4: Evaluating the Solution Version.

<!-- #endregion -->

<!-- #region id="jFdpKpBCJ2De" -->
#### Choosing a Recipe and Configuring a Solution

A recipe is an Amazon Personalize term specifying an appropriate algorithm to train for a given use case. 
<!-- #endregion -->

```python id="ZfJ2o9hXJ2De"
lambda_state_select_receipe_create_solution = LambdaStep(
    state_id="select receipe and create solution",
    parameters={  
        "FunctionName": "stepfunction_select-recipe_create-solution", #replace with the name of the function you created
        "Payload": {  
           #"dataset_group_arn.$": '$.Payload.datasetGroupArn' 
            "dataset_group_arn.$": '$.datasetGroupArn'
        }
    },
    result_path = '$'
)

lambda_state_select_receipe_create_solution.add_retry(Retry(
    error_equals=["States.TaskFailed"],
    interval_seconds=5,
    max_attempts=1,
    backoff_rate=4.0
))

lambda_state_select_receipe_create_solution.add_catch(Catch(
    error_equals=["States.TaskFailed"],
    next_step=Fail("DatasetReceiptCreateSolutionTaskFailed")
))
```

<!-- #region id="QOf9vztIJ2Df" -->
#### Create Solution Version

Once you have completed Choosing a Recipe and Configuring a Solution, you are ready to create a Solution Version. A Solution Version refers to a trained machine learning model you can deploy to get recommendations for customers. You can create a solution version using the console, AWS Command Line Interface (AWS CLI), or AWS SDK.
<!-- #endregion -->

```python id="fHluaQ2GJ2Df"
lambda_create_solution_version = LambdaStep(
    state_id="create solution version",
    parameters={  
        "FunctionName": "stepfunction_create_solution_version", 
        "Payload": {  
           "solution_arn.$": '$.Payload.solution_arn'           
        }
    },
    result_path = '$'
)

lambda_create_solution_version.add_retry(Retry(
    error_equals=["States.TaskFailed"],
    interval_seconds=5,
    max_attempts=1,
    backoff_rate=4.0
))

lambda_create_solution_version.add_catch(Catch(
    error_equals=["States.TaskFailed"],
    next_step=Fail("CreateSolutionVersionTaskFailed")
))
```

<!-- #region id="1UX-hOvUJ2Df" -->
#### Create Campaign

A campaign is used to make recommendations for your users. You create a campaign by deploying a solution version
<!-- #endregion -->

```python id="WCh3ylXEJ2Dg"
lambda_create_campaign = LambdaStep(
    state_id="create campaign",
    parameters={  
        "FunctionName": "stepfunction_getsolution_metric_create_campaign", 
        "Payload": {  
            #"solution_version_arn.$": '$.Payload.solution_version_arn'  
            "solution_version_arn.$": '$.solution_version_arn'
        }
    },
    result_path = '$'
)

lambda_create_campaign.add_retry(Retry(
    error_equals=["States.TaskFailed"],
    interval_seconds=5,
    max_attempts=1,
    backoff_rate=4.0
))

lambda_create_campaign.add_catch(Catch(
    error_equals=["States.TaskFailed"],
    next_step=Fail("CreateCampaignTaskFailed")
))
```

<!-- #region id="k5G5abaGJ2Dg" -->
## Wait-States

#### A `Wait` state in Step Functions waits a specific amount of time. See [Wait](https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/states.html#stepfunctions.steps.states.Wait) in the AWS Step Functions Data Science SDK documentation.
<!-- #endregion -->

<!-- #region id="f9-_p09KJ2Dg" -->
#### Wait for Schema to be ready
<!-- #endregion -->

```python id="asINKENUJ2Dh"
wait_state_schema = Wait(
    state_id="Wait for create schema - 5 secs",
    seconds=5
)
```

<!-- #region id="aQaJYp7mJ2Dj" -->
#### Wait for Datasetgroup to be ready
<!-- #endregion -->

```python id="QSWOitkzJ2Dk"
wait_state_datasetgroup = Wait(
    state_id="Wait for datasetgroup - 30 secs",
    seconds=30
)
```

<!-- #region id="VygShtL1J2Dk" -->
#### Wait for Dataset to be ready
<!-- #endregion -->

```python id="rp0AXddVJ2Dk"
wait_state_dataset = Wait(
    state_id="wait for dataset - 30 secs",
    seconds=30
)
```

<!-- #region id="fjTGtNNZJ2Dl" -->
#### Wait for Dataset Import Job to be ACTIVE
<!-- #endregion -->

```python id="2Ptw09z8J2Dl"
wait_state_datasetimportjob = Wait(
    state_id="Wait for datasetimportjob - 30 secs",
    seconds=30
)
```

<!-- #region id="UR3BtVUGJ2Dl" -->
#### Wait for Receipe to ready
<!-- #endregion -->

```python id="5wgQMQwPJ2Dl"
wait_state_receipe = Wait(
    state_id="Wait for receipe - 30 secs",
    seconds=30
)
```

<!-- #region id="FX5m4LZQJ2Dl" -->
#### Wait for Solution Version to be ACTIVE
<!-- #endregion -->

```python id="TTtvsSW3J2Dm"
wait_state_solutionversion = Wait(
    state_id="Wait for solution version - 60 secs",
    seconds=60
)
```

<!-- #region id="aE9j-RvgJ2Dm" -->
#### Wait for Campaign to be ACTIVE
<!-- #endregion -->

```python id="WpkBzy5RJ2Dm"
wait_state_campaign = Wait(
    state_id="Wait for Campaign - 30 secs",
    seconds=30
)
```

<!-- #region id="_uPinnogJ2Dn" -->


### Check status of the lambda task and take action accordingly
<!-- #endregion -->

<!-- #region id="b9F_mwJiJ2Dn" -->
#### If a state fails, move it to `Fail` state. See [Fail](https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/states.html#stepfunctions.steps.states.Fail) in the AWS Step Functions Data Science SDK documentation.
<!-- #endregion -->

<!-- #region id="TDBopEIcJ2Dn" -->
### check datasetgroup status
<!-- #endregion -->

```python id="h93BkFW0J2Do"
lambda_state_datasetgroupstatus = LambdaStep(
    state_id="check dataset Group status",
    parameters={  
        "FunctionName": "stepfunction_waitforDatasetGroup", #replace with the name of the function you created
        "Payload": {  
           "input.$": '$.Payload.datasetGroupArn',
           "schemaArn.$": '$.Payload.schemaArn'
        }
    },
    result_path = '$'
)

lambda_state_datasetgroupstatus.add_retry(Retry(
    error_equals=["States.TaskFailed"],
    interval_seconds=5,
    max_attempts=1,
    backoff_rate=4.0
))

lambda_state_datasetgroupstatus.add_catch(Catch(
    error_equals=["States.TaskFailed"],
    next_step=Fail("DatasetGroupStatusTaskFailed")
))
```

<!-- #region id="HKoSh3sXJ2Do" -->
### check dataset import job status
<!-- #endregion -->

```python id="NuD_ceTSJ2Do"
lambda_state_datasetimportjob_status = LambdaStep(
    state_id="check dataset import job status",
    parameters={  
        "FunctionName": "stepfunction_waitfordatasetimportjob", #replace with the name of the function you created
        "Payload": {  
           "dataset_import_job_arn.$": '$.Payload.dataset_import_job_arn',
           "datasetGroupArn.$": '$.Payload.datasetGroupArn'
        }
    },
    result_path = '$'
)

lambda_state_datasetimportjob_status.add_retry(Retry(
    error_equals=["States.TaskFailed"],
    interval_seconds=5,
    max_attempts=1,
    backoff_rate=4.0
))

lambda_state_datasetimportjob_status.add_catch(Catch(
    error_equals=["States.TaskFailed"],
    next_step=Fail("DatasetImportJobStatusTaskFailed")
))
```

<!-- #region id="ceFrym6bJ2Do" -->
### check solution version status
<!-- #endregion -->

```python id="Y7Em6gevJ2Dp"

solutionversion_succeed_state = Succeed(
    state_id="The Solution Version ready?"
)
```

```python id="rN3K8w3UJ2Dp"
lambda_state_solutionversion_status = LambdaStep(
    state_id="check solution version status",
    parameters={  
        "FunctionName": "stepfunction_waitforSolutionVersion", #replace with the name of the function you created
        "Payload": {  
           "solution_version_arn.$": '$.Payload.solution_version_arn'           
        }
    },
    result_path = '$'
)

lambda_state_solutionversion_status.add_retry(Retry(
    error_equals=["States.TaskFailed"],
    interval_seconds=5,
    max_attempts=1,
    backoff_rate=4.0
))

lambda_state_solutionversion_status.add_catch(Catch(
    error_equals=["States.TaskFailed"],
    next_step=Fail("SolutionVersionStatusTaskFailed")
))
```

<!-- #region id="BZXi905gJ2Dp" -->
### check campaign status
<!-- #endregion -->

```python id="L6JKukjJJ2Dp"
lambda_state_campaign_status = LambdaStep(
    state_id="check campaign status",
    parameters={  
        "FunctionName": "stepfunction_waitforCampaign", #replace with the name of the function you created
        "Payload": {  
           "campaign_arn.$": '$.Payload.campaign_arn'           
        }
    },
    result_path = '$'
)

lambda_state_campaign_status.add_retry(Retry(
    error_equals=["States.TaskFailed"],
    interval_seconds=5,
    max_attempts=1,
    backoff_rate=4.0
))

lambda_state_campaign_status.add_catch(Catch(
    error_equals=["States.TaskFailed"],
    next_step=Fail("CampaignStatusTaskFailed")
))
```

<!-- #region id="B-Epm1VwJ2Dq" -->
## Choice-States

Now, attach branches to the Choice state you created earlier. See *Choice Rules* in the [AWS Step Functions Data Science SDK documentation](https://aws-step-functions-data-science-sdk.readthedocs.io) .
<!-- #endregion -->

<!-- #region id="0tmx9V9SJ2Dq" -->
#### Chain together steps for the define the workflow path

The following cell links together the steps you've created above into a sequential group. The new path sequentially includes the Lambda state, Wait state, and the Succeed state that you created earlier.

#### After chaining together the steps for the workflow path, we will define and visualize the workflow.
<!-- #endregion -->

```python id="0AX1lwVVJ2Dq"
create_campaign_choice_state = Choice(
    state_id="Is the Campaign ready?"
)
```

```python id="S03MmV_NJ2Dq"
create_campaign_choice_state.add_choice(
    rule=ChoiceRule.StringEquals(variable=lambda_state_campaign_status.output()['Payload']['status'], value='ACTIVE'),
    next_step=Succeed("CampaignCreatedSuccessfully")     
)
create_campaign_choice_state.add_choice(
    ChoiceRule.StringEquals(variable=lambda_state_campaign_status.output()['Payload']['status'], value='CREATE PENDING'),
    next_step=wait_state_campaign
)
create_campaign_choice_state.add_choice(
    ChoiceRule.StringEquals(variable=lambda_state_campaign_status.output()['Payload']['status'], value='CREATE IN_PROGRESS'),
    next_step=wait_state_campaign
)

create_campaign_choice_state.default_choice(next_step=Fail("CreateCampaignFailed"))

```

```python id="8_WGDhgvJ2Dr"
solutionversion_choice_state = Choice(
    state_id="Is the Solution Version ready?"
)
```

```python id="XhLvfSiLJ2Dr"
solutionversion_succeed_state = Succeed(
    state_id="The Solution Version ready?"
)
```

```python id="wPly7Zk0J2Dr"
solutionversion_choice_state.add_choice(
    rule=ChoiceRule.StringEquals(variable=lambda_state_solutionversion_status.output()['Payload']['status'], value='ACTIVE'),
    next_step=solutionversion_succeed_state   
)
solutionversion_choice_state.add_choice(
    ChoiceRule.StringEquals(variable=lambda_state_solutionversion_status.output()['Payload']['status'], value='CREATE PENDING'),
    next_step=wait_state_solutionversion
)
solutionversion_choice_state.add_choice(
    ChoiceRule.StringEquals(variable=lambda_state_solutionversion_status.output()['Payload']['status'], value='CREATE IN_PROGRESS'),
    next_step=wait_state_solutionversion
)

solutionversion_choice_state.default_choice(next_step=Fail("create_solution_version_failed"))

```

```python id="D8RHrx4xJ2Dr"
datasetimportjob_succeed_state = Succeed(
    state_id="The Solution Version ready?"
)
```

```python id="Zncia-QOJ2Dr"
datasetimportjob_choice_state = Choice(
    state_id="Is the DataSet Import Job ready?"
)
```

```python id="ZW6uELggJ2Ds"
datasetimportjob_choice_state.add_choice(
    rule=ChoiceRule.StringEquals(variable=lambda_state_datasetimportjob_status.output()['Payload']['status'], value='ACTIVE'),
    next_step=datasetimportjob_succeed_state   
)
datasetimportjob_choice_state.add_choice(
    ChoiceRule.StringEquals(variable=lambda_state_datasetimportjob_status.output()['Payload']['status'], value='CREATE PENDING'),
    next_step=wait_state_datasetimportjob
)
datasetimportjob_choice_state.add_choice(
    ChoiceRule.StringEquals(variable=lambda_state_datasetimportjob_status.output()['Payload']['status'], value='CREATE IN_PROGRESS'),
    next_step=wait_state_datasetimportjob
)


datasetimportjob_choice_state.default_choice(next_step=Fail("dataset_import_job_failed"))

```

```python id="vny-Ir9zJ2Ds"
datasetgroupstatus_choice_state = Choice(
    state_id="Is the DataSetGroup ready?"
)
```

<!-- #region id="WyQP4BX3J2Ds" -->
## Workflow
<!-- #endregion -->

<!-- #region id="sc4tPUYiJ2Ds" -->
### Define Workflow

In the following cell, you will define the step that you will use in our workflow.  Then you will create, visualize and execute the workflow. 

Steps relate to states in AWS Step Functions. For more information, see [States](https://docs.aws.amazon.com/step-functions/latest/dg/concepts-states.html) in the *AWS Step Functions Developer Guide*. For more information on the AWS Step Functions Data Science SDK APIs, see: https://aws-step-functions-data-science-sdk.readthedocs.io. 



<!-- #endregion -->

<!-- #region id="IW5Zs1iFJ2Ds" -->
### Dataset workflow
<!-- #endregion -->

```python id="kC-LSu5yJ2Dt"
Dataset_workflow_definition=Chain([lambda_state_schema,
                                   wait_state_schema,
                                   lambda_state_datasetgroup,
                                   wait_state_datasetgroup,
                                   lambda_state_datasetgroupstatus
                                  ])
```

```python id="dj4lFc3DJ2Dt"
Dataset_workflow = Workflow(
    name="Dataset-workflow",
    definition=Dataset_workflow_definition,
    role=workflow_execution_role
)
```

```python id="ZIReHZHSJ2Dt"
Dataset_workflow.render_graph()
```

```python id="Rfy9NZ4KJ2Dt" outputId="77778fd2-2eda-4dda-ec90-e0ed370fd37c"
DatasetWorkflowArn = Dataset_workflow.create()
```

<!-- #region id="DAfPk0bOJ2Du" -->
### DatasetImportWorkflow
<!-- #endregion -->

```python id="QwwOlnDGJ2Du"
DatasetImport_workflow_definition=Chain([lambda_state_createdataset,
                                   wait_state_dataset,
                                   lambda_state_datasetimportjob,
                                   wait_state_datasetimportjob,
                                   lambda_state_datasetimportjob_status,
                                   datasetimportjob_choice_state
                                  ])
```

```python id="JCXHOq0gJ2Du"
DatasetImport_workflow = Workflow(
    name="DatasetImport-workflow",
    definition=DatasetImport_workflow_definition,
    role=workflow_execution_role
)
```

```python id="WEhqxRT2J2Du"
DatasetImport_workflow.render_graph()
```

```python id="eN9ZSS--J2Dv" outputId="34aa754b-58d4-4716-dfd8-4719b92eb3e7"
DatasetImportflowArn = DatasetImport_workflow.create()
```

<!-- #region id="ukKZGxmGJ2Dw" -->
Recepie and Solution workflow
<!-- #endregion -->

```python id="MF4Ywk3PJ2Dw"
Create_receipe_sol_workflow_definition=Chain([lambda_state_select_receipe_create_solution,
                                   wait_state_receipe,
                                   lambda_create_solution_version,
                                   wait_state_solutionversion,
                                   lambda_state_solutionversion_status,
                                   solutionversion_choice_state
                                  ])
```

```python id="zADxSkQnJ2Dx"
Create_receipe_sol_workflow = Workflow(
    name="Create_receipe_sol-workflow",
    definition=Create_receipe_sol_workflow_definition,
    role=workflow_execution_role
)
```

```python id="PdZ-CA_UJ2D0"
Create_receipe_sol_workflow.render_graph()
```

```python id="zT4gW6YTJ2D1" outputId="b59ea1d1-2681-4672-ab60-6d3aabbd75e4"
CreateReceipeArn = Create_receipe_sol_workflow.create()
```

<!-- #region id="MOvlFUOTJ2D2" -->
Create Campaign Workflow
<!-- #endregion -->

```python id="gPw3BTGnJ2D3"
Create_Campaign_workflow_definition=Chain([lambda_create_campaign,
                                   wait_state_campaign,
                                   lambda_state_campaign_status,
                                   wait_state_datasetimportjob,
                                   create_campaign_choice_state
                                  ])
```

```python id="RWDpc6MjJ2D4"
Campaign_workflow = Workflow(
    name="Campaign-workflow",
    definition=Create_Campaign_workflow_definition,
    role=workflow_execution_role
)
```

```python id="LUYkcXRTJ2D4"
Campaign_workflow.render_graph()
```

```python id="NH6mxXIjJ2D4" outputId="66119000-5388-47ef-cf0f-eb896b50b529"
CreateCampaignArn = Campaign_workflow.create()
```

<!-- #region id="G0pfkXRAJ2D5" -->
Main workflow
<!-- #endregion -->

```python id="kMD1jsRYJ2D5"
call_dataset_workflow_state = Task(
    state_id="DataSetWorkflow",
    resource="arn:aws:states:::states:startExecution.sync:2",
    parameters={
                                "Input": "true",
                                #"StateMachineArn": "arn:aws:states:us-east-1:444602785259:stateMachine:Dataset-workflow",
                                "StateMachineArn": DatasetWorkflowArn
                }
)
```

```python id="6cWeYi2rJ2D5"
call_datasetImport_workflow_state = Task(
    state_id="DataSetImportWorkflow",
    resource="arn:aws:states:::states:startExecution.sync:2",
    parameters={
                                 "Input":{
                                    "schemaArn.$": "$.Output.Payload.schemaArn",
                                    "datasetGroupArn.$": "$.Output.Payload.datasetGroupArn"
                                   },
                                "StateMachineArn": DatasetImportflowArn,
                }
)
```

```python id="c0alenEpJ2D6"
call_receipe_solution_workflow_state = Task(
    state_id="ReceipeSolutionWorkflow",
    resource="arn:aws:states:::states:startExecution.sync:2",
    parameters={
                                 "Input":{
                                    "datasetGroupArn.$": "$.Output.Payload.datasetGroupArn"

                                   },
                                "StateMachineArn": CreateReceipeArn
                }
)
```

```python id="uM_Z_eAkJ2D6"
call_campaign_solution_workflow_state = Task(
    state_id="CampaignWorkflow",
    resource="arn:aws:states:::states:startExecution.sync:2",
    parameters={
                                 "Input":{
                                    "solution_version_arn.$": "$.Output.Payload.solution_version_arn"

                                   },
                                "StateMachineArn": CreateCampaignArn
                }
)
```

```python id="lnfdK7nnJ2D6"
Main_workflow_definition=Chain([call_dataset_workflow_state,
                                call_datasetImport_workflow_state,
                                call_receipe_solution_workflow_state,
                                call_campaign_solution_workflow_state
                               ])
```

```python id="qOA8ka1ZJ2D7"
Main_workflow = Workflow(
    name="Main-workflow",
    definition=Main_workflow_definition,
    role=workflow_execution_role
)
```

```python id="jmP40EZOJ2D7"
Main_workflow.render_graph()
```

```python id="hnmL_wd3J2D7" outputId="0bd7e7c0-2ced-4f22-e232-5161943f02e6"
Main_workflow.create()
```

```python id="5aybLT6IJ2D8" outputId="1ca77a2f-e382-40ba-ccc5-a00639d4569c"
Main_workflow_execution = Main_workflow.execute()
```

<!-- #region id="psUCkhxRJ2D8" -->
Main_workflow_execution = Workflow(
    name="Campaign_Workflow",
    definition=path1,
    role=workflow_execution_role
)

<!-- #endregion -->

```python id="v58VTqGYJ2D8"

```

```python id="aJn_ZPAZJ2D8"
#Main_workflow_execution.render_graph()
```

<!-- #region id="jDLfzQa6J2D9" -->
### Create and execute the workflow

In the next cells, we will create the branching happy workflow in AWS Step Functions with [create](https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.create) and execute it with [execute](https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.execute).

<!-- #endregion -->

```python id="M0mkIbrZJ2D9"
#personalize_workflow.create()
```

```python id="WKSKUVduJ2D9"
#personalize_workflow_execution = happy_workflow.execute()
```

<!-- #region id="NRGHAcZAJ2D9" -->
###  Review the workflow progress

Review the workflow progress with the [render_progress](https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Execution.render_progress).

Review the execution history by calling [list_events](https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Execution.list_events) to list all events in the workflow execution.
<!-- #endregion -->

```python id="GL_pv7j5J2D9"
Main_workflow_execution.render_progress()
```

```python id="H0Cbo_KcJ2D-"
Main_workflow_execution.list_events(html=True)
```

<!-- #region id="XcjmxIS9J2D-" -->
## Generate-Recommendations
<!-- #endregion -->

<!-- #region id="MlR8uXCdJ2D-" -->
### Now that we have a successful campaign, let's generate recommendations for the campaign
<!-- #endregion -->

<!-- #region id="j3GGtZRNJ2D-" -->
#### Select a User and an Item
<!-- #endregion -->

```python id="DNqNpKWgJ2D_" outputId="56e00557-43eb-4dc4-cbf0-de2ddc13a95b"
items = pd.read_csv('./ml-100k/u.item', sep='|', usecols=[0,1], encoding='latin-1')
items.columns = ['ITEM_ID', 'TITLE']


user_id, item_id, rating, timestamp = data.sample().values[0]

user_id = int(user_id)
item_id = int(item_id)

print("user_id",user_id)
print("items",items)


item_title = items.loc[items['ITEM_ID'] == item_id].values[0][-1]
print("USER: {}".format(user_id))
print("ITEM: {}".format(item_title))
print("ITEM ID: {}".format(item_id))

```

```python id="Wup_JzVPJ2D_"
wait_recommendations = Wait(
    state_id="Wait for recommendations - 10 secs",
    seconds=10
)
```

<!-- #region id="GhmMBRCvJ2D_" -->
#### Lambda Task
<!-- #endregion -->

```python id="f_FBoyjUJ2D_"
lambda_state_get_recommendations = LambdaStep(
    state_id="get recommendations",
    parameters={  
        "FunctionName": "stepfunction_getRecommendations", 
        "Payload": {  
           "campaign_arn": 'arn:aws:personalize:us-east-1:261602857181:campaign/stepfunction-campaign',            
           "user_id": user_id,  
           "item_id": item_id             
        }
    },
    result_path = '$'
)

lambda_state_get_recommendations.add_retry(Retry(
    error_equals=["States.TaskFailed"],
    interval_seconds=5,
    max_attempts=1,
    backoff_rate=4.0
))

lambda_state_get_recommendations.add_catch(Catch(
    error_equals=["States.TaskFailed"],
    next_step=Fail("GetRecommendationTaskFailed")
    #next_step=recommendation_path   
))
```

<!-- #region id="wCHTwoYOJ2EA" -->
#### Create a Succeed State
<!-- #endregion -->

```python id="p9oPY3XwJ2EA"
workflow_complete = Succeed("WorkflowComplete")
```

```python id="lXKaj9ySJ2EA"
recommendation_path = Chain([ 
lambda_state_get_recommendations,
wait_recommendations,
workflow_complete
])
```

<!-- #region id="UnFfflboJ2EA" -->
### Define, Create, Render, and Execute Recommendation Workflow

In the next cells, we will create a workflow in AWS Step Functions with [create](https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.create) and execute it with [execute](https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Workflow.execute).
<!-- #endregion -->

```python id="FfZgDTMrJ2EB"
recommendation_workflow = Workflow(
    name="Recommendation_Workflow4",
    definition=recommendation_path,
    role=workflow_execution_role
)


```

```python id="1dYso8YqJ2EB"
recommendation_workflow.render_graph()
```

```python id="Oj3jJUC5J2EB" outputId="b8fd34a3-d0fc-45b5-901a-f14806474501"
recommendation_workflow.create()
```

```python id="zRSuIQwxJ2EB" outputId="c9c43a18-30d3-408b-f3ba-03c8c76172e4"
recommendation_workflow_execution = recommendation_workflow.execute()
```

```python id="zwsdMiVVJ2EC"

```

<!-- #region id="ZC4j4ZDLJ2EE" -->
### Review progress

Review workflow progress with the [render_progress](https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Execution.render_progress).

Review execution history by calling [list_events](https://aws-step-functions-data-science-sdk.readthedocs.io/en/latest/workflow.html#stepfunctions.workflow.Execution.list_events) to list all events in the workflow execution.
<!-- #endregion -->

```python id="9LeIbX-WJ2EE"
recommendation_workflow_execution.render_progress()
```

```python id="rdf-Qk5WJ2EF"
recommendation_workflow_execution.list_events(html=True)

```

```python id="1JxN8KAvJ2EF"
item_list = recommendation_workflow_execution.get_output()['Payload']['item_list']
```

<!-- #region id="Ru-i3odAJ2EF" -->
### Get Recommendations
<!-- #endregion -->

```python id="TtHJWsX2J2EF" outputId="8980d474-18fe-4ca8-ec1b-56d33e0d3696"
item_list = recommendation_workflow_execution.get_output()['Payload']['item_list']

print("Recommendations:")
for item in item_list:
    np.int(item['itemId'])
    item_title = items.loc[items['ITEM_ID'] == np.int(item['itemId'])].values[0][-1]
    print(item_title)

```

<!-- #region id="ulmWlmDtJ2EG" -->
## Clean up Amazon Personalize resources
<!-- #endregion -->

<!-- #region id="0r-vw0cdJ2EG" -->
Make sure to clean up the Amazon Personalize and the state machines created blog. Login to Amazon Personalize console and delete resources such as Dataset Groups, Dataset, Solutions, Receipts, and Campaign. 
<!-- #endregion -->

<!-- #region id="0_XJaJhvJ2EG" -->
## Clean up State Machine resources
<!-- #endregion -->

```python id="FXwtOF8rJ2EG" outputId="b96b903c-975f-4eb3-e842-9400fc6961fb"
Campaign_workflow.delete()

recommendation_workflow.delete()

Main_workflow.delete()

Create_receipe_sol_workflow.delete()

DatasetImport_workflow.delete()

Dataset_workflow.delete()

```
