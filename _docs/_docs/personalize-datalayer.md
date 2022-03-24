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

<!-- #region id="ac4NjDi0-JIz" -->
# Amazon Personalize Generic Module - Data Layer
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7ZJPAbfaJcTE" executionInfo={"status": "ok", "timestamp": 1630047493290, "user_tz": -330, "elapsed": 720, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bded3fca-1d3b-4955-8b9a-f767ea8c65b9"
!mkdir -p code/cloudformation
!wget -q --show-progress -O code/cloudformation/immersion_day.yaml https://personalization-at-amazon.s3.amazonaws.com/amazon-personalize/AmazonPersonalizeImmersionDay.yaml
```

```python colab={"base_uri": "https://localhost:8080/"} id="dco-CyHmKTAb" executionInfo={"status": "ok", "timestamp": 1630047507321, "user_tz": -330, "elapsed": 737, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="5e012369-a54c-4841-9897-d1b7f1892473"
!cat code/cloudformation/immersion_day.yaml
```

<!-- #region id="am9RumpUcLWx" -->
## Data Preparation
<!-- #endregion -->

```python id="MVMVDeLWKYzk"
import time
from time import sleep
import json
from datetime import datetime
import pandas as pd
```

```python colab={"base_uri": "https://localhost:8080/"} id="2UVWuFVVaF81" executionInfo={"status": "ok", "timestamp": 1630047692472, "user_tz": -330, "elapsed": 437, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="111803e5-aa5d-4010-965e-a041645eead2"
original_data = pd.read_csv('./data/bronze/ml-latest-small/ratings.csv')
original_data.info()
```

<!-- #region id="GLJtDOC7aZmV" -->
The int64 format is clearly suitable for userId and movieId. However, we need to dive deeper to understand the timestamps in the data. To use Amazon Personalize, you need to save timestamps in Unix Epoch format. Currently, the timestamp values are not human-readable. So let's grab an arbitrary timestamp value and figure out how to interpret it. Do a quick sanity check on the transformed dataset by picking an arbitrary timestamp and transforming it to a human-readable format.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5KhFzjnHaoIV" executionInfo={"status": "ok", "timestamp": 1630047743488, "user_tz": -330, "elapsed": 461, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f2966174-a31b-4484-e4bd-edce90c03721"
arb_time_stamp = original_data.iloc[50]['timestamp']
print(arb_time_stamp)
print(datetime.utcfromtimestamp(arb_time_stamp).strftime('%Y-%m-%d %H:%M:%S'))
```

<!-- #region id="LfF4C69ea9CK" -->
Since this is a dataset of an explicit feedback movie ratings, it includes movies rated from 1 to 5. We want to include only moves that were "liked" by the users, and simulate a dataset of data that would be gathered by a VOD platform. In order to do that, we will filter out all interactions under 2 out of 5, and create two EVENT_Types "click" and and "watch". We will then assign all movies rated 2 and above as "click" and movies rated 4 and above as both "click" and "watch".

Note that this is to correspond with the events we are modeling, for a real data set you would actually model based on implicit feedback such as clicks, watches and/or explicit feedback such as ratings, likes etc.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 578} id="3TmlOUgNa-Sr" executionInfo={"status": "ok", "timestamp": 1630047929545, "user_tz": -330, "elapsed": 613, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="69a4ab49-f41c-4db4-b4d3-25ad8e3dc7ae"
watched_df = original_data.copy()
watched_df = watched_df[watched_df['rating'] > 3]
watched_df = watched_df[['userId', 'movieId', 'timestamp']]
watched_df['EVENT_TYPE']='watch'
display(watched_df.head())

clicked_df = original_data.copy()
clicked_df = clicked_df[clicked_df['rating'] > 1]
clicked_df = clicked_df[['userId', 'movieId', 'timestamp']]
clicked_df['EVENT_TYPE']='click'
display(clicked_df.head())

interactions_df = clicked_df.copy()
interactions_df = interactions_df.append(watched_df)
interactions_df.sort_values("timestamp", axis = 0, ascending = True, 
                 inplace = True, na_position ='last') 
interactions_df.info()
```

<!-- #region id="NXV-FREFbX4_" -->
Amazon Personalize has default column names for users, items, and timestamp. These default column names are USER_ID, ITEM_ID, AND TIMESTAMP. So the final modification to the dataset is to replace the existing column headers with the default headers.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="JuhWwOV7bbnJ" executionInfo={"status": "ok", "timestamp": 1630048048895, "user_tz": -330, "elapsed": 545, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f1f62fd8-566c-4333-a124-df2150d74259"
interactions_df.rename(columns = {'userId':'USER_ID', 'movieId':'ITEM_ID', 
                              'timestamp':'TIMESTAMP'}, inplace = True)
interactions_df.head()
```

```python id="X_bHOc_0b1HK"
interactions_df.to_csv('./data/silver/ml-latest-small/interactions.csv', index=False, float_format='%.0f')
```

```python colab={"base_uri": "https://localhost:8080/"} id="H03qEFm25Otd" executionInfo={"status": "ok", "timestamp": 1630055796089, "user_tz": -330, "elapsed": 537, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a4c83035-39c1-4159-b564-9c65bab224c2"
original_data = pd.read_csv('./data/bronze/ml-latest-small/movies.csv')
original_data.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="GR6mdTge5Yzz" executionInfo={"status": "ok", "timestamp": 1630055932177, "user_tz": -330, "elapsed": 622, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="0891fc8c-67cc-45c5-92a1-21fafab26734"
original_data['year'] =original_data['title'].str.extract('.*\((.*)\).*',expand = False)
original_data = original_data.dropna(axis=0)

itemmetadata_df = original_data.copy()
itemmetadata_df = itemmetadata_df[['movieId', 'genres', 'year']]
itemmetadata_df.head()
```

<!-- #region id="XwgGUdk654pB" -->
We will add a new dataframe to help us generate a creation timestamp. If you don’t provide the CREATION_TIMESTAMP for an item, the model infers this information from the interaction dataset and uses the timestamp of the item’s earliest interaction as its corresponding release date. If an item doesn’t have an interaction, its release date is set as the timestamp of the latest interaction in the training set and it is considered a new item. For the current dataset we will set the CREATION_TIMESTAMP to 0.
<!-- #endregion -->

```python id="pHUtEDT2522e"
itemmetadata_df['CREATION_TIMESTAMP'] = 0
itemmetadata_df.rename(columns = {'genres':'GENRE', 'movieId':'ITEM_ID', 'year':'YEAR'}, inplace = True) 
itemmetadata_df.to_csv('./data/silver/ml-latest-small/item-meta.csv', index=False, float_format='%.0f')
```

<!-- #region id="eBy5V4Z4b2rC" -->
## AWS Personalize
<!-- #endregion -->

```python id="C4xjNqJZcw2a"
!pip install -q boto3
import boto3
import json
import time
```

```python id="1oZPIuJFcXN1"
!mkdir -p ~/.aws && cp /content/drive/MyDrive/AWS/d01_admin/* ~/.aws
```

<!-- #region id="XeBuVJpGFsvb" -->
### ETL Job for Interactions data without using generic data loading module
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="iwv-AGiWcJeD" executionInfo={"status": "ok", "timestamp": 1630048308723, "user_tz": -330, "elapsed": 402, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8c08cc2d-607c-4ed2-8077-ae0cb909bce9"
# Configure the SDK to Personalize:
personalize = boto3.client('personalize')
personalize_runtime = boto3.client('personalize-runtime')
print("We can communicate with Personalize!")
```

```python id="XkvOHA9ack0x"
# create the dataset group (the highest level of abstraction)
create_dataset_group_response = personalize.create_dataset_group(
    name = "immersion-day-dataset-group-movielens-latest"
)

dataset_group_arn = create_dataset_group_response['datasetGroupArn']
print(json.dumps(create_dataset_group_response, indent=2))

# wait for it to become active
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
```

```python id="k-6qvw4oflsH"
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
            "name": "EVENT_TYPE",
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
    name = "personalize-poc-movielens-interactions",
    schema = json.dumps(interactions_schema)
)

interaction_schema_arn = create_schema_response['schemaArn']
print(json.dumps(create_schema_response, indent=2))

dataset_type = "INTERACTIONS"
create_dataset_response = personalize.create_dataset(
    name = "personalize-poc-movielens-ints",
    datasetType = dataset_type,
    datasetGroupArn = dataset_group_arn,
    schemaArn = interaction_schema_arn
)

interactions_dataset_arn = create_dataset_response['datasetArn']
print(json.dumps(create_dataset_response, indent=2))
```

```python colab={"base_uri": "https://localhost:8080/"} id="8fN7IWbagYkT" executionInfo={"status": "ok", "timestamp": 1630049380307, "user_tz": -330, "elapsed": 472, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="998ba781-2064-492f-9ff0-9bbdc8e1aef3"
region = 'us-east-1'
s3 = boto3.client('s3')
account_id = boto3.client('sts').get_caller_identity().get('Account')
bucket_name = account_id + "-" + region + "-" + "personalizepocvod"
print(bucket_name)
if region == "us-east-1":
    s3.create_bucket(Bucket=bucket_name)
else:
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={'LocationConstraint': region}
        )
```

```python id="SQsQs56hg6I2"
interactions_file_path = './data/silver/ml-latest-small/interactions.csv'
interactions_filename = 'interactions.csv'
boto3.Session().resource('s3').Bucket(bucket_name).Object(interactions_filename).upload_file(interactions_file_path)
interactions_s3DataPath = "s3://"+bucket_name+"/"+interactions_filename
```

```python id="uj3Ua9zOhfFW"
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

```python colab={"base_uri": "https://localhost:8080/"} id="jI5D1AlViHAZ" executionInfo={"status": "ok", "timestamp": 1630049797032, "user_tz": -330, "elapsed": 60956, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="69e83310-bb91-4ed2-d439-a5783e33a56a"
iam = boto3.client("iam")

role_name = "PersonalizeRolePOC"
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

```python id="asaTiB0oiRGr"
create_dataset_import_job_response = personalize.create_dataset_import_job(
    jobName = "personalize-poc-import1",
    datasetArn = interactions_dataset_arn,
    dataSource = {
        "dataLocation": "s3://{}/{}".format(bucket_name, interactions_filename)
    },
    roleArn = role_arn
)

dataset_import_job_arn = create_dataset_import_job_response['datasetImportJobArn']
print(json.dumps(create_dataset_import_job_response, indent=2))

# wait fir this import job to gets activated

max_time = time.time() + 6*60*60 # 6 hours
while time.time() < max_time:
    describe_dataset_import_job_response = personalize.describe_dataset_import_job(
        datasetImportJobArn = dataset_import_job_arn
    )
    status = describe_dataset_import_job_response["datasetImportJob"]['status']
    print("DatasetImportJob: {}".format(status))
    
    if status == "ACTIVE" or status == "CREATE FAILED":
        break
        
    time.sleep(60)
```

<!-- #region id="7jPFKY3qFHwh" -->
### ETL Job for Item meta using generic data loading module
<!-- #endregion -->

```python id="Y5zv_4760056"
import sys
sys.path.insert(0,'./code')

from generic_modules.import_dataset import personalize_dataset
```

```python id="91OcwOQm9Ai6"
dataset_group_arn = 'arn:aws:personalize:us-east-1:746888961694:dataset-group/immersion-day-dataset-group-movielens-latest'
bucket_name = '746888961694-us-east-1-personalizepocvod'
role_arn = 'arn:aws:iam::746888961694:role/PersonalizeRolePOC'

dataset_type = 'ITEMS'
source_data_path = './data/silver/ml-latest-small/item-meta.csv'
target_file_name = 'item-meta.csv'
```

```python id="rIv3OWn18qVV"
personalize_item_meta = personalize_dataset(
    dataset_group_arn = dataset_group_arn,
    bucket_name = bucket_name,
    role_arn = role_arn,
    dataset_type = dataset_type,
    source_data_path = source_data_path,
    target_file_name = target_file_name,
    dataset_arn = dataset_arn,
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="pVwE0fkL-b51" executionInfo={"status": "ok", "timestamp": 1630057962385, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="369468f7-275c-45f1-8794-d503260db547"
personalize_item_meta.setup_connection()
```

```python id="2OaRUWs8-hAE"
itemmetadata_schema = {
    "type": "record",
    "name": "Items",
    "namespace": "com.amazonaws.personalize.schema",
    "fields": [
        {
            "name": "ITEM_ID",
            "type": "string"
        },
        {
            "name": "GENRE",
            "type": "string",
            "categorical": True
        },{
            "name": "YEAR",
            "type": "int",
        },
        {
            "name": "CREATION_TIMESTAMP",
            "type": "long",
        }
    ],
    "version": "1.0"
}
```

```python id="ELGoa3ng-uD1"
personalize_item_meta.create_dataset(schema=itemmetadata_schema,
                                     schema_name='personalize-poc-movielens-item',
                                     dataset_name='personalize-poc-movielens-items')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="BrwOj2VpCemy" executionInfo={"status": "ok", "timestamp": 1630058213530, "user_tz": -330, "elapsed": 635, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bd706745-e758-43d2-ac2e-4b44e47baff7"
personalize_item_meta.dataset_arn
```

```python id="9ZksZbihBmBz"
personalize_item_meta.upload_data_to_s3()
```

```python id="0fYXJO3lB6EP"
personalize_item_meta.import_data_from_s3(import_job_name='personalize-poc-item-import1')
```

```python id="2M5MUMAeGpFN"

import boto3
import json
import time


class personalize_dataset:
    def __init__(self,
                 dataset_group_arn=None,
                 schema_arn=None,
                 dataset_arn=None,
                 dataset_type='INTERACTIONS',
                 region='us-east-1',
                 bucket_name=None,
                 role_arn=None,
                 source_data_path=None,
                 target_file_name=None,
                 dataset_import_job_arn=None
                 ):
        self.personalize = None
        self.personalize_runtime = None
        self.s3 = None
        self.iam = None
        self.dataset_group_arn = dataset_group_arn
        self.schema_arn = schema_arn
        self.dataset_arn = dataset_arn
        self.dataset_type = dataset_type
        self.region = region
        self.bucket_name = bucket_name
        self.role_arn = role_arn
        self.source_data_path = source_data_path
        self.target_file_name = target_file_name
        self.dataset_import_job_arn = dataset_import_job_arn

    def setup_connection(self):
        try:
            self.personalize = boto3.client('personalize')
            self.personalize_runtime = boto3.client('personalize-runtime')
            self.s3 = boto3.client('s3')
            self.iam = boto3.client("iam")
            print("SUCCESS | We can communicate with Personalize!")
        except:
            print("ERROR | Connection can't be established!")
    
    def create_dataset_group(self, dataset_group_name=None):
        """
        The highest level of isolation and abstraction with Amazon Personalize
        is a dataset group. Information stored within one of these dataset groups
        has no impact on any other dataset group or models created from one. they
        are completely isolated. This allows you to run many experiments and is
        part of how we keep your models private and fully trained only on your data.
        """
        create_dataset_group_response = self.personalize.create_dataset_group(name=dataset_group_name)
        self.dataset_group_arn = create_dataset_group_response['datasetGroupArn']
        # print(json.dumps(create_dataset_group_response, indent=2))

        # Before we can use the dataset group, it must be active. 
        # This can take a minute or two. Execute the cell below and wait for it
        # to show the ACTIVE status. It checks the status of the dataset group
        # every minute, up to a maximum of 3 hours.
        max_time = time.time() + 3*60*60 # 3 hours
        while time.time() < max_time:
            status = self.check_dataset_group_status()
            print("DatasetGroup: {}".format(status))
            if status == "ACTIVE" or status == "CREATE FAILED":
                break
            time.sleep(60)

    def check_dataset_group_status(self):
        """
        Check the status of dataset group
        """
        describe_dataset_group_response = self.personalize.describe_dataset_group(
            datasetGroupArn = self.dataset_group_arn
            )
        status = describe_dataset_group_response["datasetGroup"]["status"]
        return status

    def create_dataset(self, schema=None, schema_name=None, dataset_name=None):
        """
        First, define a schema to tell Amazon Personalize what type of dataset
        you are uploading. There are several reserved and mandatory keywords
        required in the schema, based on the type of dataset. More detailed
        information can be found in the documentation.
        """
        create_schema_response = self.personalize.create_schema(
            name = schema_name,
            schema = json.dumps(schema)
        )
        self.schema_arn = create_schema_response['schemaArn']

        """
        With a schema created, you can create a dataset within the dataset group.
        Note that this does not load the data yet, it just defines the schema for
        the data. The data will be loaded a few steps later.
        """
        create_dataset_response = self.personalize.create_dataset(
            name = dataset_name,
            datasetType = self.dataset_type,
            datasetGroupArn = self.dataset_group_arn,
            schemaArn = self.schema_arn
        )
        self.dataset_arn = create_dataset_response['datasetArn']
    
    def create_s3_bucket(self):
        if region == "us-east-1":
            self.s3.create_bucket(Bucket=self.bucket_name)
        else:
            self.s3.create_bucket(
                Bucket=self.bucket_name,
                CreateBucketConfiguration={'LocationConstraint': self.region}
                )
    
    def upload_data_to_s3(self):
        """
        Now that your Amazon S3 bucket has been created, upload the CSV file of
        our user-item-interaction data.
        """
        boto3.Session().resource('s3').Bucket(self.bucket_name).Object(self.target_file_name).upload_file(self.source_data_path)
        s3DataPath = "s3://"+self.bucket_name+"/"+self.target_file_name
    
    def set_s3_bucket_policy(self, policy=None):
        """
        Amazon Personalize needs to be able to read the contents of your S3
        bucket. So add a bucket policy which allows that.
        """
        if not policy:
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
                            "arn:aws:s3:::{}".format(self.bucket_name),
                            "arn:aws:s3:::{}/*".format(self.bucket_name)
                        ]
                    }
                ]
            }

        self.s3.put_bucket_policy(Bucket=self.bucket_name, Policy=json.dumps(policy))

    def create_iam_role(self, role_name=None):
        """
        Amazon Personalize needs the ability to assume roles in AWS in order to
        have the permissions to execute certain tasks. Let's create an IAM role
        and attach the required policies to it. The code below attaches very permissive
        policies; please use more restrictive policies for any production application.
        """
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
        create_role_response = self.iam.create_role(
            RoleName = role_name,
            AssumeRolePolicyDocument = json.dumps(assume_role_policy_document)
        )

        # AmazonPersonalizeFullAccess provides access to any S3 bucket with a name that includes "personalize" or "Personalize" 
        # if you would like to use a bucket with a different name, please consider creating and attaching a new policy
        # that provides read access to your bucket or attaching the AmazonS3ReadOnlyAccess policy to the role
        policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonPersonalizeFullAccess"
        self.iam.attach_role_policy(
            RoleName = role_name,
            PolicyArn = policy_arn
        )
        # Now add S3 support
        self.iam.attach_role_policy(
            PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess',
            RoleName=role_name
        )
        time.sleep(60) # wait for a minute to allow IAM role policy attachment to propagate
        self.role_arn = create_role_response["Role"]["Arn"]

    def import_data_from_s3(self, import_job_name=None):
        """
        Earlier you created the dataset group and dataset to house your information,
        so now you will execute an import job that will load the data from the S3
        bucket into the Amazon Personalize dataset.
        """
        create_dataset_import_job_response = self.personalize.create_dataset_import_job(
        jobName = import_job_name,
        datasetArn = self.dataset_arn,
        dataSource = {
            "dataLocation": "s3://{}/{}".format(self.bucket_name, self.target_file_name)
        },
        roleArn = self.role_arn
        )
        self.dataset_import_job_arn = create_dataset_import_job_response['datasetImportJobArn']

        """
        Before we can use the dataset, the import job must be active. Execute the
        cell below and wait for it to show the ACTIVE status. It checks the status
        of the import job every minute, up to a maximum of 6 hours.
        Importing the data can take some time, depending on the size of the dataset.
        In this workshop, the data import job should take around 15 minutes.
        """
        max_time = time.time() + 6*60*60 # 6 hours
        while time.time() < max_time:
            describe_dataset_import_job_response = personalize.describe_dataset_import_job(
                datasetImportJobArn = dataset_import_job_arn
            )
            status = self.check_import_job_status()
            print("DatasetImportJob: {}".format(status))
            if status == "ACTIVE" or status == "CREATE FAILED":
                break
            time.sleep(60)
    
    def check_import_job_status(self):
        describe_dataset_import_job_response = self.personalize.describe_dataset_import_job(
            datasetImportJobArn = self.dataset_import_job_arn
        )
        status = describe_dataset_import_job_response["datasetImportJob"]['status']
        return status

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['personalize']
        del attributes['personalize_runtime']
        del attributes['s3']
        del attributes['iam']
        return attributes
```

```python colab={"base_uri": "https://localhost:8080/"} id="fq131onaDHkM" executionInfo={"status": "ok", "timestamp": 1630059835081, "user_tz": -330, "elapsed": 613, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c101b288-131e-457b-d2c5-20dcf755308c"
dataset_arn = 'arn:aws:personalize:us-east-1:746888961694:dataset/immersion-day-dataset-group-movielens-latest/ITEMS'
dataset_import_job_arn = 'arn:aws:personalize:us-east-1:746888961694:dataset-import-job/personalize-poc-item-import1'

personalize_item_meta = personalize_dataset(
    dataset_group_arn = dataset_group_arn,
    bucket_name = bucket_name,
    role_arn = role_arn,
    dataset_type = dataset_type,
    source_data_path = source_data_path,
    target_file_name = target_file_name,
    dataset_arn = dataset_arn,
    dataset_import_job_arn = dataset_import_job_arn
)

personalize_item_meta.setup_connection()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="Au1JEi8UDSI7" executionInfo={"status": "ok", "timestamp": 1630059836609, "user_tz": -330, "elapsed": 11, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7f2a0e7b-f374-4eb7-d0d5-ed9502f8d361"
personalize_item_meta.check_import_job_status()
```

<!-- #region id="-AU3W_1IEub6" -->
### Saving the state
<!-- #endregion -->

```python id="z1zTY8rOGWyz"
import pickle

with open('./artifacts/etc/personalize_item_meta.pkl', 'wb') as outp:
    pickle.dump(personalize_item_meta, outp, pickle.HIGHEST_PROTOCOL)
```

```python colab={"base_uri": "https://localhost:8080/"} id="x_cde5DTHbS-" executionInfo={"status": "ok", "timestamp": 1630059841764, "user_tz": -330, "elapsed": 426, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="fe78a27b-cc40-42b4-b621-6860f8277727"
personalize_item_meta.__getstate__()
```
