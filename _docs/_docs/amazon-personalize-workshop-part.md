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
    language: python
    name: python3
---

# Amazon Personalize Workshop Part 4 - Security best practices (optional)
> We are using a pre-made dataset that hasn't been encrypted so there is no need to decrypt this dataset. However, it would be a good security practice to store your datasets encrypted.
- toc: true
- badges: true
- comments: true
- categories: [amazonpersonalize, movie, security, privacy]
- image: 


### Get the Personalize boto3 Client

```python
import boto3

import json
import numpy as np
import pandas as pd
import time

personalize = boto3.client('personalize')
personalize_runtime = boto3.client('personalize-runtime')
iam = boto3.client("iam")
s3 = boto3.client("s3")
```

### Specify a Bucket and Data Output Location

```python
bucket = "personalize-demo"       # replace with the name of your S3 bucket
filename = "movie-lens-100k.csv"  # replace with a name that you want to save the dataset under
```

### Download, Prepare, and Upload Training Data


#### Download and Explore the Dataset

```python
!wget -N http://files.grouplens.org/datasets/movielens/ml-100k.zip
!unzip -o ml-100k.zip
data = pd.read_csv('./ml-100k/u.data', sep='\t', names=['USER_ID', 'ITEM_ID', 'RATING', 'TIMESTAMP'])
pd.set_option('display.max_rows', 5)
data
```

#### Optional security practice: Protect data at rest - Encrypt/decrypt your dataset
We are using a pre-made dataset that hasn't been encrypted so there is no need to decrypt this dataset. However, it would be a good security practice to store your datasets encrypted.

For more information on encrypting your data when using S3, visit https://docs.aws.amazon.com/AmazonS3/latest/dev/KMSUsingRESTAPI.html


#### Optional security practice: Protect data in transit - SSL access only for S3 bucket

```python
requires_ssl_access_policy = {
    "Version": "2012-10-17",
    "Id": "RequireSSLAccess",
    "Statement": [
        {
            "Sid": "RequireSSLAccess",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "*",
            "Resource": [
                "arn:aws:s3:::{}".format(bucket),
                "arn:aws:s3:::{}/*".format(bucket)
            ],
            "Condition": {
                "Bool": {
                    "aws:SecureTransport": "false"
                }
            }
        }
    ]
}

s3.put_bucket_policy(Bucket=bucket, Policy=json.dumps(requires_ssl_access_policy))
```

#### Additional security note:
Some users prevent accidental information disclosure by limiting S3 access to only come from a VPC. Another common security practice is to validate this limited access. It should be noted that this security check will fail when performed against S3 buckets used for Personalize - as Personalize copies data from the user's S3 into the internal systems used by Personalize (during dataset import jobs).


#### Optional security practice: validate bucket owner matches your account canonical id
[More information about canonical ids here](https://docs.aws.amazon.com/general/latest/gr/acct-identifiers.html#FindingCanonicalId)

```python
bucket_owner_id = boto3.client('s3').get_bucket_acl(Bucket=bucket)['Owner']['ID']
print("This bucket belongs to: {} ".format(bucket_owner_id))
```

#### Optional security practice: Protect data integrity - Enable S3 bucket versioning


```python
s3_resource = boto3.resource('s3')
bucket_versioning = s3_resource.BucketVersioning(bucket)
bucket_versioning.enable()
```

#### Prepare and Upload Data

```python
data = data[data['RATING'] > 3.6]                # keep only movies rated 3.6 and above
data = data[['USER_ID', 'ITEM_ID', 'TIMESTAMP']] # select columns that match the columns in the schema below
data.to_csv(filename, index=False)

boto3.Session().resource('s3').Bucket(bucket).Object(filename).upload_file(filename)
```

### Create Schema

```python
schema = {
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
    name = "DEMO-schema",
    schema = json.dumps(schema)
)

schema_arn = create_schema_response['schemaArn']
print(json.dumps(create_schema_response, indent=2))
```

#### Optional security practice - Protect data at rest - Encrypt datasets under Personalize
If you skip this step, do not pass kmsKeyArn or roleArn when you create your dataset group.

```python
kmsKeyArn = boto3.client('kms').create_key(Description="personalize-data")['KeyMetadata']['Arn']
print(kmsKeyArn)
key_accessor_policy_name = "AccessPersonalizeDatasetPolicy"
key_accessor_policy = {
  "Version": "2012-10-17",
  "Statement": {
    "Effect": "Allow",
    "Action": [
      "kms:*"
    ],
    "Resource": [ kmsKeyArn ]
  }
}
key_access_policy = iam.create_policy(
    PolicyName = key_accessor_policy_name,
    PolicyDocument = json.dumps(key_accessor_policy)
)

key_access_role_name = "AccessPersonalizeDatasetRole"
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
    RoleName = key_access_role_name,
    AssumeRolePolicyDocument = json.dumps(assume_role_policy_document)
)
iam.attach_role_policy(
    RoleName = create_role_response["Role"]["RoleName"],
    PolicyArn = key_access_policy["Policy"]["Arn"]
)

keyAccessRoleArn = create_role_response["Role"]["Arn"]
print(keyAccessRoleArn)
```

### Create and Wait for Encrypted Dataset Group


#### Create Encrypted Dataset Group
If you did not create a KMS Key and IAM role from the last step, then do not pass in roleArn or kmsKeyArn.

```python
create_dataset_group_response = personalize.create_dataset_group(
    name = "DEMO-dataset-group",
    roleArn = keyAccessRoleArn,
    kmsKeyArn = kmsKeyArn
)

dataset_group_arn = create_dataset_group_response['datasetGroupArn']
print(json.dumps(create_dataset_group_response, indent=2))
```

#### Wait for Dataset Group to Have ACTIVE Status

```python
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

### Create Dataset

```python
dataset_type = "INTERACTIONS"
create_dataset_response = personalize.create_dataset(
    name = "DEMO-dataset",
    datasetType = dataset_type,
    datasetGroupArn = dataset_group_arn,
    schemaArn = schema_arn
)

dataset_arn = create_dataset_response['datasetArn']
print(json.dumps(create_dataset_response, indent=2))
```

### Prepare, Create, and Wait for Dataset Import Job


#### Attach Policy to S3 Bucket

```python
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
```

#### Create Personalize Role

```python
role_name = "PersonalizeRole"
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

time.sleep(60) # wait for a minute to allow IAM role policy attachment to propagate

role_arn = create_role_response["Role"]["Arn"]
print(role_arn)
```

#### Create Dataset Import Job

```python
create_dataset_import_job_response = personalize.create_dataset_import_job(
    jobName = "DEMO-dataset-import-job",
    datasetArn = dataset_arn,
    dataSource = {
        "dataLocation": "s3://{}/{}".format(bucket, filename)
    },
    roleArn = role_arn
)

dataset_import_job_arn = create_dataset_import_job_response['datasetImportJobArn']
print(json.dumps(create_dataset_import_job_response, indent=2))
```

#### Wait for Dataset Import Job to Have ACTIVE Status

```python
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
```

### Select Recipe

```python
list_recipes_response = personalize.list_recipes()
recipe_arn = "arn:aws:personalize:::recipe/aws-hrnn" # aws-hrnn selected for demo purposes
list_recipes_response
```

### Create and Wait for Solution


#### Create Solution

```python
create_solution_response = personalize.create_solution(
    name = "DEMO-solution",
    datasetGroupArn = dataset_group_arn,
    recipeArn = recipe_arn
)

solution_arn = create_solution_response['solutionArn']
print(json.dumps(create_solution_response, indent=2))
```

#### Create Solution Version

```python
create_solution_version_response = personalize.create_solution_version(
    solutionArn = solution_arn
)

solution_version_arn = create_solution_version_response['solutionVersionArn']
print(json.dumps(create_solution_version_response, indent=2))
```

#### Wait for Solution Version to Have ACTIVE Status

```python
max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    describe_solution_version_response = personalize.describe_solution_version(
        solutionVersionArn = solution_version_arn
    )
    status = describe_solution_version_response["solutionVersion"]["status"]
    print("SolutionVersion: {}".format(status))
    
    if status == "ACTIVE" or status == "CREATE FAILED":
        break
        
    time.sleep(60)
```

#### Get Metrics of Solution

```python
get_solution_metrics_response = personalize.get_solution_metrics(
    solutionVersionArn = solution_version_arn
)

print(json.dumps(get_solution_metrics_response, indent=2))
```

### Create and Wait for Campaign


#### Create Campaign

```python
create_campaign_response = personalize.create_campaign(
    name = "DEMO-campaign",
    solutionVersionArn = solution_version_arn,
    minProvisionedTPS = 1
)

campaign_arn = create_campaign_response['campaignArn']
print(json.dumps(create_campaign_response, indent=2))
```

#### Wait for Campaign to Have ACTIVE Status

```python
max_time = time.time() + 3*60*60 # 3 hours
while time.time() < max_time:
    describe_campaign_response = personalize.describe_campaign(
        campaignArn = campaign_arn
    )
    status = describe_campaign_response["campaign"]["status"]
    print("Campaign: {}".format(status))
    
    if status == "ACTIVE" or status == "CREATE FAILED":
        break
        
    time.sleep(60)
```

### Get Recommendations


#### Select a User and an Item

```python
items = pd.read_csv('./ml-100k/u.item', sep='|', usecols=[0,1], encoding='latin-1')
items.columns = ['ITEM_ID', 'TITLE']

user_id, item_id, _ = data.sample().values[0]
item_title = items.loc[items['ITEM_ID'] == item_id].values[0][-1]
print("USER: {}".format(user_id))
print("ITEM: {}".format(item_title))

items
```

#### Call GetRecommendations

```python
get_recommendations_response = personalize_runtime.get_recommendations(
    campaignArn = campaign_arn,
    userId = str(user_id),
    itemId = str(item_id)
)

item_list = get_recommendations_response['itemList']
title_list = [items.loc[items['ITEM_ID'] == np.int(item['itemId'])].values[0][-1] for item in item_list]

print("Recommendations: {}".format(json.dumps(title_list, indent=2)))
```
