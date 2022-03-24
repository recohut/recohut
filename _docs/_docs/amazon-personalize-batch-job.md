---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: conda_python3
    language: python
    name: conda_python3
---

<!-- #region id="gdNLiW3fYNJW" -->
# Amazon Personalize Batch Job
> Train and create batch recommendations using Amazon Personalize. Expected completion time is 1.5-2 hours

- toc: true
- badges: true
- comments: true
- categories: [amazonpersonalize, batch, movie, hrnn]
- image: 
<!-- #endregion -->

<!-- #region id="MmXmEqzGYJ7Q" -->
**Amazon Personalize Workflow**

The general workflow for training, deploying, and getting recommendations from a campaign is as follows:

1. Prepare data

2. Create related datasets and a dataset group.

3. Get training data.

    - Import historical data to the dataset group.

    - Record user events to the dataset group.

4. Create a solution version (trained model) using a recipe.

5. Evaluate the solution version using metrics.

6. Create a campaign (deploy the solution version).

7. Provide recommendations for users by running Batch Recommendation.

In this lab, we will step through the workflow and with some additional steps to setup your IAM permissions and S3 buckets as a data source for your dataset and output for the batch recommendations. 

**Note:** This lab will not cover the deployment of a real-time personalize campaign.
<!-- #endregion -->

<!-- #region id="H7j_gy-NYJ7S" -->
## Prepare Data
<!-- #endregion -->

<!-- #region id="FhhAxqIkYJ7T" -->
### Get dataset
In thie lab, we will be using the the [Movielens dataset](http://grouplens.org/datasets/movielens/) to train and make movie recommendations.

Movielens provide several datasets. To achieve better model accuracy, it is recommendeded to train the Personalize model with a large dataset, however the tradeoff would mean a longer training time. To minimise the time required to complete this lab, we will be sacrificing accuracy for time and will be using the small dataset.
<!-- #endregion -->

```python id="Uh3XWFHMYJ7V" outputId="db24260f-19ce-4302-924c-faff1d9fd681"
data_dir = "movie_data"
!mkdir $data_dir
!cd $data_dir && wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!cd $data_dir && unzip ml-latest-small.zip
```

<!-- #region id="WokCPN9cYJ7Z" -->
### Prepare data
<!-- #endregion -->

```python id="6Xarb5ZYYJ7a"
import time
from time import sleep
import json
from datetime import datetime

import boto3
import pandas as pd
import uuid
```

<!-- #region id="nfXGh3bIYJ7b" -->
Load the dataset and preview the data.
<!-- #endregion -->

```python id="Bn4I2lpbYJ7c" outputId="e66b796b-d1b6-4147-f640-1f0556f6089c"
original_data = pd.read_csv(data_dir + '/ml-latest-small/ratings.csv')
original_data.head(10)
```

<!-- #region id="b4y6j3NHYJ7e" -->
In the lab, we will be using the movie rating dataset and considering movies with ratings greater or equal to 4 to use for the recommendation as we only want to recommend movies that have been positively rated.
<!-- #endregion -->

```python id="tR1DeDnLYJ7f" outputId="0fd1b46d-d2f0-48b6-9f55-09b6b287764f"
interactions_df = original_data.copy()

# Only want ratings greater or equal to 4, filter out ratings less than 4
interactions_df = interactions_df[interactions_df['rating'] >= 4.0]

interactions_df.head(10)
```

<!-- #region id="a1kJRg6gYJ7g" -->
The next step is to map the dataset to the personalize schema by renaming the column name.

For more information about the schema, refer to the following URL: https://docs.aws.amazon.com/personalize/latest/dg/how-it-works-dataset-schema.html
<!-- #endregion -->

```python id="m5enBhlJYJ7h"
interactions_df = interactions_df[['userId', 'movieId', 'timestamp']]
interactions_df.head()
interactions_df.rename(columns = {'userId':'USER_ID', 'movieId':'ITEM_ID', 
                              'timestamp':'TIMESTAMP'}, inplace = True)
```

<!-- #region id="V6Wo4Ip8YJ7h" -->
Finally, we save the dataset to CSV file which we will later upload to S3 for Personalize to use.
<!-- #endregion -->

```python id="hFXUZ8FoYJ7i"
interactions_filename = "interactions.csv"
interactions_df.to_csv((data_dir+"/"+interactions_filename), index=False, float_format='%.0f')
```

<!-- #region id="M2uc816aYJ7j" -->
## Create related datasets and a dataset group.

In this section, we will setup the Amazon Personalize dataset group and load the inteaction dataset into Amazon Personalize which will be used for training.

Amazon Personalize requires data, stored in Amazon Personalize datasets, in order to train a model.

There are two ways to provide the training data. You can import historical data from an Amazon S3 bucket, and you can record event data as it is created.

A dataset group contains related datasets. Three types of historical datasets are created by the customer (users, items, and interactions), and one type is created by Amazon Personalize for live-event interactions. A dataset group can contain only one of each kind of dataset.

You can create dataset groups to serve different purposes. For example, you might have an application that provides recommendations for purchasing shoes and another that provides recommendations for places to visit in Europe. In Amazon Personalize, each application would have its own dataset group.

Historical data must be provided in a CSV file. Each dataset type has a unique schema that specifies the contents of the CSV file.

There is a [minimum amount of data](https://docs.aws.amazon.com/personalize/latest/dg/limits.html) that is necessary to train a model. Using existing data allows you to immediately start training a model. If you rely on recorded data as it is created, and there is no historical data, it can take a while before training can begin.
<!-- #endregion -->

```python id="GnkHMWk7YJ7k"
# Configure the SDK to Personalize:
personalize = boto3.client('personalize')
personalize_runtime = boto3.client('personalize-runtime')
```

<!-- #region id="HJngfs71YJ7l" -->
### Create the personalize dataset group
We start by creating the personalize dataset group named "**personalize-devlab-movies-dataset-group**" which will be used to to store our interactive (ratings) dataset we prepared earlier.
<!-- #endregion -->

```python id="XsbY7FHHYJ7m" outputId="0808569c-6f3b-4e4b-fc15-57b0c8c73868"
create_dataset_group_response = personalize.create_dataset_group(
    name = "personalize-devlab-movies-dataset-group"
)

dataset_group_arn = create_dataset_group_response['datasetGroupArn']
print(json.dumps(create_dataset_group_response, indent=2))
```

<!-- #region id="l_g_mGaoYJ7m" -->
### CHECKPOINT #1 - Wait for dataset group creation to complete
The dataset group will take some time to be created. **Execute the following cell and wait for it to show "ACTIVE" before proceeding to the next step.**
<!-- #endregion -->

```python id="9N-x3w80YJ7n" outputId="990d0937-149f-45d3-ac28-d12f50aee19c"
current_time = datetime.now()
print("Started on: ", current_time.strftime("%I:%M:%S %p"))

max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    describe_dataset_group_response = personalize.describe_dataset_group(
        datasetGroupArn = dataset_group_arn
    )
    status = describe_dataset_group_response["datasetGroup"]["status"]
    print("DatasetGroup: {}".format(status))
    
    if status == "ACTIVE" or status == "CREATE FAILED":
        break
        
    time.sleep(60)

current_time = datetime.now()
print("Completed on: ", current_time.strftime("%I:%M:%S %p"))
```

<!-- #region id="l2TIRnBTYJ7o" -->
### Create the dataset
Once the dataset group have been complete, the next step is to defined the interaction schema and we will name it "**personalize-devlab-movies-interactions-schema**".
<!-- #endregion -->

```python id="AUCNSq1eYJ7o" outputId="4e263641-5991-450c-dbb1-9459530d99f7"
interactions_schema = schema = {
    "type": "record",
    "name": "Interactions",
    "namespace": "com.amazonaws.personalize.schema",
    "fields": [
        {
            "name": "USER_ID",
            "type": "string"
        },
        {
            "name": "ITEM_ID",
            "type": "string"
        },
        {
            "name": "TIMESTAMP",
            "type": "long"
        }
    ],
    "version": "1.0"
}

create_schema_response = personalize.create_schema(
    name = "personalize-devlab-movies-interactions-schema",
    schema = json.dumps(interactions_schema)
)

schema_arn = create_schema_response['schemaArn']
print(json.dumps(create_schema_response, indent=2))
```

<!-- #region id="x9s_stG4YJ7o" -->
Once the schema has been defined, we will define the interactiion dataset using the schema we created above and provide it with the following name "personalize-devlab-movies-interactions-dataset"
<!-- #endregion -->

```python id="-ARNtsrMYJ7p" outputId="d8fde9f3-1281-448b-e909-391d828fb6cd"
dataset_type = "INTERACTIONS"
create_dataset_response = personalize.create_dataset(
    name = "personalize-devlab-movies-interactions-dataset",
    datasetType = dataset_type,
    datasetGroupArn = dataset_group_arn,
    schemaArn = schema_arn
)

dataset_arn = create_dataset_response['datasetArn']
print(json.dumps(create_dataset_response, indent=2))
```

```python id="MBO7X8yRYJ7p"
# Record the interaction dataset arn to be used later
interactions_dataset_arn = dataset_arn
```

<!-- #region id="kSKWcCO9YJ7p" -->
## Configuring S3 and IAM 


Amazon Personalize will need an S3 bucket to act as the source of your data, as well as IAM roles for accessing it. The code below will set all that up.
<!-- #endregion -->

<!-- #region id="JjyKXo4zYJ7p" -->
Now using the metada stored on this instance of a SageMaker Notebook determine the region we are operating in. If you are using a Jupyter Notebook outside of SageMaker simply define region as the string that indicates the region you would like to use for Forecast and S3.

<!-- #endregion -->

```python id="Ymh7wuwmYJ7q" outputId="d9f0b97c-4b49-48b5-d84c-4acd71d7cc7e"
with open('/opt/ml/metadata/resource-metadata.json') as notebook_info:
    data = json.load(notebook_info)
    resource_arn = data['ResourceArn']
    region = resource_arn.split(':')[3]
    
session = boto3.Session(region_name=region)

print(region)
s3 = boto3.client('s3')
account_id = boto3.client('sts').get_caller_identity().get('Account')
bucket_name = account_id + "personalizedevlab"
print(bucket_name)
if region != "us-east-1":
    s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': region})
else:
    s3.create_bucket(Bucket=bucket_name)
```

<!-- #region id="sWAfdnn-YJ7q" -->
### Attach Policy to S3 Bucket
Amazon Personalize needs to be able to read the content of your S3 bucket that you created earlier. The lines below will do that.
<!-- #endregion -->

```python id="XFWQOgdJYJ7q" outputId="84b51a0e-094a-4fa3-c465-12c1d1041f3e"
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
                "s3:*Object",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::{}".format(bucket_name),
                "arn:aws:s3:::{}/*".format(bucket_name)
            ]
        }
    ]
}

s3.put_bucket_policy(Bucket=bucket_name, Policy=json.dumps(policy))
```

<!-- #region id="-4CtiR6iYJ7r" -->
### Create Personalize Role
Also Amazon Personalize needs the ability to assume Roles in AWS in order to have the permissions to execute certain tasks, the lines below grant that.
<!-- #endregion -->

```python id="cV8ia7jJYJ7r" outputId="129fc6e6-c344-4cbe-e94e-7a33d2406e4d"
iam = boto3.client("iam")

role_name = "PersonalizeRoleDevLab"
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

# AmazonPersonalizeFullAccess provides access to any S3 bucket with a name that includes "personalize" or "Personalize" 
# if you would like to use a bucket with a different name, please consider creating and attaching a new policy
# that provides read access to your bucket or attaching the AmazonS3ReadOnlyAccess policy to the role
policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonPersonalizeFullAccess"
iam.attach_role_policy(
    RoleName = role_name,
    PolicyArn = policy_arn
)

# Now add S3 support
iam.attach_role_policy(
    PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess',
    RoleName=role_name
)
time.sleep(60) # wait for a minute to allow IAM role policy attachment to propagate

role_arn = create_role_response["Role"]["Arn"]
print(role_arn)
```

<!-- #region id="pLb6oC84YJ7t" -->
## Upload dataset to S3

Before Personalize can import the data, it needs to be in S3.
<!-- #endregion -->

```python id="8aKLxG4KYJ7t"
# Upload Interactions File
interactions_file_path = data_dir + "/" + interactions_filename
boto3.Session().resource('s3').Bucket(bucket_name).Object(interactions_filename).upload_file(interactions_file_path)
interactions_s3DataPath = "s3://"+bucket_name+"/"+interactions_filename
```

<!-- #region id="qpyzK67_YJ7u" -->
## Importing the Interactions Data

Earlier you created the DatasetGroup and Dataset to house your information, now you will execute an import job that will load the data from S3 into Amazon Personalize for usage building your model.

#### Create Dataset Import Job
<!-- #endregion -->

```python id="q1FngPBJYJ7u" outputId="4d5f189f-2e5a-4ecf-8013-dc1415d3e2d4"
create_dataset_import_job_response = personalize.create_dataset_import_job(
    jobName = "personalize-devlab-import1",
    datasetArn = interactions_dataset_arn,
    dataSource = {
        "dataLocation": "s3://{}/{}".format(bucket_name, interactions_filename)
    },
    roleArn = role_arn
)

dataset_import_job_arn = create_dataset_import_job_response['datasetImportJobArn']
print(json.dumps(create_dataset_import_job_response, indent=2))
```

<!-- #region id="ET7p3lQNYJ7v" -->
### CHECKPOINT #2 - Wait for Dataset Import Job to Have ACTIVE Status

It can take a while before the import job completes. **Execute the following cell and wait for it to show "ACTIVE" before proceeding to the next step.**
<!-- #endregion -->

```python id="Y16w9gC2YJ7w" outputId="c91b3910-32cd-433d-e37b-7a72c7ca84b2"
current_time = datetime.now()
print("Started on: ", current_time.strftime("%I:%M:%S %p"))

max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    describe_dataset_import_job_response = personalize.describe_dataset_import_job(
        datasetImportJobArn = dataset_import_job_arn
    )
    status = describe_dataset_import_job_response["datasetImportJob"]['status']
    print("DatasetImportJob: {}".format(status))
    
    if status == "ACTIVE" or status == "CREATE FAILED":
        break
        
    time.sleep(60)

current_time = datetime.now()
print("Completed on: ", current_time.strftime("%I:%M:%S %p"))
```

<!-- #region id="hoEIVqp2YJ7w" -->
## Create solution
<!-- #endregion -->

<!-- #region id="bWid0wrlYJ70" -->
In this section we will define a solution using the HRNN recipe to generate user personalization recommendation. There are several other recipes available such as HRNN-Metadata, HRNN-Coldstart etc. More information about the additional recipes can be found here:
https://docs.aws.amazon.com/personalize/latest/dg/working-with-predefined-recipes.html
<!-- #endregion -->

```python id="jOqsLWEkYJ70"
HRNN_recipe_arn = "arn:aws:personalize:::recipe/aws-hrnn"
```

```python id="l8AxBjoXYJ71" outputId="99fdcc2e-7f87-4e70-810f-f5680c682255"
hrnn_create_solution_response = personalize.create_solution(
    name = "personalize-devlab-hrnn",
    datasetGroupArn = dataset_group_arn,
    recipeArn = HRNN_recipe_arn
)

hrnn_solution_arn = hrnn_create_solution_response['solutionArn']
print(json.dumps(hrnn_create_solution_response, indent=2))
```

<!-- #region id="pK-eqHBgYJ71" -->
## Create the solution version

In this section we will train a solution version using the dataset that we loaded and using the HRNN recipe.
<!-- #endregion -->

```python id="exUD_T6DYJ72"
hrnn_create_solution_version_response = personalize.create_solution_version(
    solutionArn = hrnn_solution_arn
)
```

```python id="SkUVxOyNYJ72" outputId="8fc4e5a3-e536-4088-d132-26b2c2df5a88"
hrnn_solution_version_arn = hrnn_create_solution_version_response['solutionVersionArn']
print(json.dumps(hrnn_create_solution_version_response, indent=2))
```

<!-- #region id="HdVamw3GYJ72" -->
### CHECKPOINT #3 - Wait for solution version creation to be completed
Training the solution version will take some time to complete training. **Execute the following cell and wait for it to show "ACTIVE" before proceeding to the next step.**

#### Viewing Solution Creation Status

You can also view the status updates in the console. To do so,

* In another browser tab you should already have the AWS Console up from opening this notebook instance. 
* Switch to that tab and search at the top for the service `Personalize`, then go to that service page. 
* Click `View dataset groups`.
* Click the name of your dataset group, most likely something with DevLab in the name.
* Click `Solutions and recipes`.
* You will now see a list of all of the solutions you created above. Click any one of them. 
* Note in `Solution versions` the job that is in progress. Once it is `Active` you solution is ready to be reviewed. It is also capable of being deployed.
<!-- #endregion -->

```python id="KkJvTtvfYJ73" outputId="5f0b6f62-1fa1-41db-c3f9-2a64a1930c8b"
current_time = datetime.now()
print("Started on: ", current_time.strftime("%I:%M:%S %p"))

max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    describe_solution_version_response = personalize.describe_solution_version(
        solutionVersionArn = hrnn_solution_version_arn
    )
    status = describe_solution_version_response["solutionVersion"]['status']
    print("SolutionVersion Status: {}".format(status))
    
    if status == "ACTIVE" or status == "CREATE FAILED":
        break
        
    time.sleep(60)
    
current_time = datetime.now()
print("Completed on: ", current_time.strftime("%I:%M:%S %p"))
```

<!-- #region id="J3qg6QyGYJ73" -->
## Evaluate solution metrics

In this section we will run the function to get the solution metrics. More information about the Personalize solution metrics can be found here: https://docs.aws.amazon.com/personalize/latest/dg/working-with-training-metrics.html
<!-- #endregion -->

```python id="xFA2CKFFYJ74" outputId="d358e73b-bf26-4b94-eeda-c3f277d19efb"
hrnn_solution_metrics_response = personalize.get_solution_metrics(
    solutionVersionArn = hrnn_solution_version_arn
)

print(json.dumps(hrnn_solution_metrics_response, indent=2))
```

<!-- #region id="Z-da6aESYJ74" -->
## Batch Recommendation
In the section, we will generate a random sample of users to generate batch recommendations for.

First, we will load the movie database so that we can visualize the recommended movie.
<!-- #endregion -->

```python id="XleAFQtLYJ75" outputId="460b115c-55c2-4609-bdf2-a9ebec30ceef"
# Create a dataframe for the items by reading in the correct source CSV.
items_df = pd.read_csv(data_dir + '/ml-latest-small/movies.csv', index_col=0)
# Render some sample data
items_df.head(5)
```

```python id="ju4euVo6YJ76"
# Create a function to get movie by id
def get_movie_by_id(movie_id, movie_df=items_df):
    try:
        return movie_df.loc[int(movie_id)]['title'] + " - " + movie_df.loc[int(movie_id)]['genres']
    except:
        return "Error obtaining movie" + movie_id
```

```python id="H66T7iIVYJ76"
# Get the user list
users_df = pd.read_csv(data_dir + '/ml-latest-small/ratings.csv', index_col=0)

batch_users = users_df.sample(3).index.tolist()

# Write the file to disk
json_input_filename = "json_input.json"
with open(data_dir + "/" + json_input_filename, 'w') as json_input:
    for user_id in batch_users:
        json_input.write('{"userId": "' + str(user_id) + '"}\n')
```

```python id="yHRqcOsGYJ76" outputId="f344c548-1e89-4d80-8a2c-07edf69f1d5d"
# Showcase the input file:
!cat $data_dir"/"$json_input_filename
```

<!-- #region id="L_Mb6WlCYJ77" -->
Upload the users generate batch recommendations for to S3
<!-- #endregion -->

```python id="zXpiDDB_YJ77" outputId="8793787e-8ed0-4cd8-db30-366c55c8489e"
# Upload files to S3
boto3.Session().resource('s3').Bucket(bucket_name).Object(json_input_filename).upload_file(data_dir+"/"+json_input_filename)
s3_input_path = "s3://" + bucket_name + "/" + json_input_filename
print(s3_input_path)
```

<!-- #region id="PIUfX7ZkYJ78" -->
In the next cell, we define output bucket of where we will store the batch recommendation results.
<!-- #endregion -->

```python id="q71Z7gAiYJ78" outputId="aad75e6f-2cb9-4016-f236-814f5f672f91"
# Define the output path
s3_output_path = "s3://" + bucket_name + "/"
print(s3_output_path)
```

<!-- #region id="DtDmY1PeYJ78" -->
Run the batch inference process
<!-- #endregion -->

```python id="phNDv1zsYJ79"
current_time = datetime.now()
batchInferenceJobArn = personalize.create_batch_inference_job (
    solutionVersionArn = hrnn_solution_version_arn,
    jobName = "Personalize-devlab-Batch-Inference-Job-HRNN"+current_time.strftime("%I%M%S"),
    roleArn = role_arn,
    jobInput = 
     {"s3DataSource": {"path": s3_input_path}},
    jobOutput = 
     {"s3DataDestination":{"path": s3_output_path}}
)
batchInferenceJobArn = batchInferenceJobArn['batchInferenceJobArn']
```

<!-- #region id="z8yqiKFOYJ79" -->
### CHECKPOINT #4 - Wait for batch recommendation job to complete

It can take a while before the batch recommendation job completes. **Execute the following cell and wait for it to show "ACTIVE" before proceeding to the next step.**
<!-- #endregion -->

```python id="i87sKMrNYJ7-" outputId="dfd20926-7d21-4042-9a7f-578a6896f68a"
current_time = datetime.now()
print("Import Started on: ", current_time.strftime("%I:%M:%S %p"))

max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    describe_dataset_inference_job_response = personalize.describe_batch_inference_job(
        batchInferenceJobArn = batchInferenceJobArn
    )
    status = describe_dataset_inference_job_response["batchInferenceJob"]['status']
    print("DatasetInferenceJob: {}".format(status))
    
    if status == "ACTIVE" or status == "CREATE FAILED":
        break
        
    time.sleep(60)
    
current_time = datetime.now()
print("Import Completed on: ", current_time.strftime("%I:%M:%S %p"))
```

<!-- #region id="_9ndY8QdYJ7-" -->
**Download and Visualize batch recommendation**

Once the batch recommendation job has been completed, we will now download and visualize the results from the batch job.

The results of the batch job will be stored in the S3 output folder that was specified earlier. It will be returned in a json format similar to the following:

```
{"input":{"userId":"448"},"output":{"recommendedItems":["5810","53322","2003","6957","92535","8917","3105","6873","1249","26133","2657","4865","2420","1345","4621","34437","2010","4128","2076","1203","52973","4246","2871","8641","162"],"scores":[0.0031413,0.0022093,0.0021377,0.0020497,0.001922,0.0018058,0.0017834,0.0017671,0.0017457,0.0016255,0.0015854,0.001539,0.0014838,0.0014573,0.001374,0.001372,0.0013563,0.0013385,0.0013196,0.0013065,0.0012714,0.0012507,0.001228,0.0012243,0.0012083]},"error":null}
{"input":{"userId":"409"},"output":{"recommendedItems":["2571","50","527","296","1196","111","110","1258","2858","1214","6874","1265","648","750","912","588","608","2329","858","2762","1291","541","1387","260","1200"],"scores":[0.0151337,0.0129099,0.0094722,0.0081178,0.0070135,0.0061934,0.0059672,0.0049986,0.0048773,0.0048134,0.0046837,0.0044422,0.0044251,0.0042917,0.0042376,0.0042309,0.0039795,0.0038287,0.0037793,0.0036532,0.0036527,0.0035218,0.0035178,0.0034306,0.0034158]},"error":null}
{"input":{"userId":"288"},"output":{"recommendedItems":["5989","49272","6377","4963","4995","68157","4886","48394","8368","80463","54001","8961","91658","8950","5418","5445","109374","8360","1136","81834","5618","1265","48780","8949","4720"],"scores":[0.0177912,0.0114374,0.0107343,0.0098669,0.009329,0.0067548,0.0057784,0.0057611,0.0056522,0.0053493,0.0050671,0.0050291,0.0047089,0.0046453,0.0045933,0.0045281,0.0041894,0.0040642,0.0040296,0.003935,0.0037718,0.0036896,0.0036616,0.0036362,0.0036019]},"error":null}

```

An typical use case for this is to use the batch recommendation output to generate a personalized recommendation email that can be fed into a popular email marketing service such as **[Amazon Pinpoint](https://aws.amazon.com/pinpoint/)** or your favourate email marketing service.
<!-- #endregion -->

```python id="UCoA3l3uYJ7_" outputId="4f53dbbe-3485-4a82-9c6f-80b6037e22d2"
s3 = boto3.client('s3')
export_name = json_input_filename + ".out"
s3.download_file(bucket_name, export_name, data_dir+"/"+export_name)

# Update DF rendering
pd.set_option('display.max_rows', 30)
with open(data_dir+"/"+export_name) as json_file:
    # Get the first line and parse it
    line = json.loads(json_file.readline())
    # Do the same for the other lines
    while line:
        # extract the user ID 
        col_header = "User: " + line['input']['userId']
        # Create a list for all the artists
        recommendation_list = []
        # Add all the entries
        for item in line['output']['recommendedItems']:
            movie = get_movie_by_id(item)
            recommendation_list.append(movie)
        if 'bulk_recommendations_df' in locals():
            new_rec_DF = pd.DataFrame(recommendation_list, columns = [col_header])
            bulk_recommendations_df = bulk_recommendations_df.join(new_rec_DF)
        else:
            bulk_recommendations_df = pd.DataFrame(recommendation_list, columns=[col_header])
        try:
            line = json.loads(json_file.readline())
        except:
            line = None
bulk_recommendations_df
```

<!-- #region id="JmYuYJj2YJ7_" -->
Congratulations, you have now completed the lab. You can either continue to the challenge section to see if you can improve the model by using a larger dataset or proceed to the cleanup section to delete the resources from this lab.
<!-- #endregion -->

<!-- #region id="DlqVkxe6YJ8A" -->
## Challenge
Before wrapping up the lab, let's see if you can try and improve the recommendation accuracy by usingn the large dataset from [movielens](https://grouplens.org/datasets/movielens/).

You can use the same dataset group, however, please note that you don't need to redefine the data set schema.
<!-- #endregion -->

<!-- #region id="wWubD_JCYJ8A" -->
## Cleanup

**IMPORTANT**
Once you're done with the lab, the final step is to clean up your environment by decommissioning the resources we created for this devlab. Please run the following cells in the following order to clean up your environment.
<!-- #endregion -->

<!-- #region id="moVOx3PjYJ8A" -->
**1. Delete the solution**

Delete the HRNN solution we created for the Personalize dataset group
<!-- #endregion -->

```python id="3ua9goKhYJ8B"
personalize.delete_solution(
    solutionArn = hrnn_solution_arn
)
```

<!-- #region id="WDq7eQ_vYJ8B" -->
**2. Delete the dataset**

Delete the datasets created for the personalize dataset group.
<!-- #endregion -->

```python id="ypMsJbY0YJ8B"
personalize.delete_dataset(
    datasetArn = dataset_arn
)
```

<!-- #region id="FZNWAFvFYJ8B" -->
Run the following cell to verify that non-required dataset has been deleted and if required run the subsequent cell with the correct ARN.
<!-- #endregion -->

```python id="v04CD74DYJ8B"
paginator = personalize.get_paginator('list_datasets')
for paginate_result in paginator.paginate():
    for datasets in paginate_result["datasets"]:
        print(datasets["datasetArn"])
```

```python id="2hBY--ZKYJ8C"
# Replace the ARN and run the following cell to delete any additional datasets
personalize.delete_dataset(
    datasetArn = "INSERT ARN HERE"
)
```

<!-- #region id="kYFJ9JMZYJ8C" -->
**3. Delete the schema**

Delete the personalize schema used for the datasets.
<!-- #endregion -->

```python id="dmoPLnMOYJ8C"
personalize.delete_schema(
    schemaArn = schema_arn
)
```

<!-- #region id="_Ys-Xp1uYJ8C" -->
Run the following cell to verify that non-required schema has been deleted and if required run the subsequent cell with the correct ARN.
<!-- #endregion -->

```python id="yS5oNjo6YJ8D"
paginator = personalize.get_paginator('list_schemas')
for paginate_result in paginator.paginate():
    for schema in paginate_result["schemas"]:
        print(schema["schemaArn"])
```

```python id="QTpZYNP3YJ8E"
# Replace the ARN and run the following cell to delete any additional schema
personalize.delete_schema(
    schemaArn = "INSERT ARN HERE"
)
```

<!-- #region id="p3Ln05nxYJ8F" -->
**4. Delete the dataset group**

Deletes the personalize dataset group
<!-- #endregion -->

```python id="wBGFw8FVYJ8G"
personalize.delete_dataset_group(
    datasetGroupArn = dataset_group_arn
)
```

<!-- #region id="gBVledKNYJ8G" -->
**5. Detach the policy from the personalize devlab role**
<!-- #endregion -->

```python id="6P8BZiasYJ8G"
iam.detach_role_policy(
    RoleName = "PersonalizeRoleDevLab",
    PolicyArn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
)
```

```python id="nS2EQRaEYJ8H"
iam.detach_role_policy(
    RoleName = "PersonalizeRoleDevLab",
    PolicyArn = "arn:aws:iam::aws:policy/service-role/AmazonPersonalizeFullAccess"
)
```

<!-- #region id="mPQcf158YJ8H" -->
Check that all policies have been detached from the role, if not run the subsequent cell to detached the appropriate roles
<!-- #endregion -->

```python id="7vzDRK8tYJ8H"
# Lists all policities attached to the personalize devlab role
iam.list_attached_role_policies(
    RoleName = "PersonalizeRoleDevLab"
)
```

```python id="T514IOjdYJ8I"
# Detach policy from rule
iam.detach_role_policy(
    RoleName = "PersonalizeRoleDevLab",
    PolicyArn = "INSERT ARN HERE"
)
```

<!-- #region id="6RUp0f5HYJ8I" -->
**6. Delete the IAM role**
<!-- #endregion -->

```python id="QYTJ_edLYJ8J"
iam.delete_role(
    RoleName = "PersonalizeRoleDevLab"
)
```
