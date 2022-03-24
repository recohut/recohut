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

<!-- #region id="77ZN-BAylPoU" -->
# Serverless Machine Learning in Action
<!-- #endregion -->

<!-- #region id="AQ7LHTvvjE-t" -->
> Note: https://livebook.manning.com/book/serverless-machine-learning-in-action/chapter-2?a_aid=osipov&a_bid=fa913283&
<!-- #endregion -->

<!-- #region id="NCaMbSI-OFtB" -->
## Download the DC taxi dataset

`gdown` is a Python utility for downloading files stored in Google Drive. The bash script in the following cell iterates through a collection of Google Drive identifiers that match files `taxi_2015.zip` through `taxi_2019.zip` stored in a shared Google Drive. This script uses these files instead of the original files from https://opendata.dc.gov/search?categories=transportation&q=taxi&type=document%20link since the originals cannot be easily downloaded using a bash script.

* **the next cell should take about a minute to finish**
<!-- #endregion -->

```bash id="_dnRoxG2J-BY"
pip install gdown
for ID in '1yF2hYrVjAZ3VPFo1dDkN80wUV2eEq65O'\
          '1Z7ZVi79wKEbnc0FH3o0XKHGUS8MQOU6R'\
          '1I_uraLKNbGPe3IeE7FUa9gPfwBHjthu4'\
          '1MoY3THytktrxam6_hFSC8kW1DeHZ8_g1'\
          '1balxhT6Qh_WDp4wq4OsG40iwqFa86QgW'
do

  gdown --id $ID

done
```

<!-- #region id="dcuxUE1oP5zB" -->
## Unzip the dataset

The script in the following cell unzips the downloaded dataset files to the `dctaxi` subdirectory in the current directory of the notebook. The `-o` flag used by the `unzip` command overwrites existing files in case you execute the next cell more than once.

* **the next cell should take about 3 minutes to complete**
<!-- #endregion -->

```bash id="RqJ9uEg_LDR5"

mkdir -p dctaxi

for YEAR in '2015' \
            '2016' \
            '2017' \
            '2018' \
            '2019'
do

  unzip -o taxi_$YEAR.zip -d dctaxi
  
done
```

<!-- #region id="kvYQMYZSQjpK" -->
## Report on the disk space used by the dataset files

The next cell reports on the disk usage (`du`) by the files from the DC taxi dataset. All of the files in the dataset have the `taxi_` prefix. Since the entire output of the `du` command lists the disk usage of all of the files, the `tail` command is used to limit the output to just the last line. You can remove the `tail` command (in other words, leave just `du -cha taxi_*.txt` in the next cell) if you wish to report on the disk usage by the individual files in the dataset.

For reference, the entire report on disk usage is also available as a Github Gist here: https://gist.github.com/osipov/032505a9c7e7388a2384f893be9e0681
<!-- #endregion -->

```python id="5TXVXaanNUJE"
!du -cha --block-size=1MB dctaxi/taxi_*.txt | tail -n 1
```

<!-- #region id="AJrzrRFaSejc" -->
## Scan the dataset documentation

The dataset includes a `README_DC_Taxicab_trip.txt` file with a brief documentation about the dataset contents. Run the next cell and take a moment to review the documentation, focusing on the schema used by the dataset.
<!-- #endregion -->

```bash id="OKbux5ecLSCK"
cat dctaxi/README_DC_Taxicab_trip.txt
```

<!-- #region id="gCyrbSReS55v" -->
## Preview the dataset

Run the next cell to confirm that the dataset consists of pipe (`|`) separated values organized according to the schema described by the documentation. The `taxi_2015_09.txt` file used in the next cell was picked arbitrarily, just to illustrate the dataset.
<!-- #endregion -->

```python id="lCm6l_DLS9VZ"
!head dctaxi/taxi_2015_09.txt
```

<!-- #region id="GZ2rTEBfU20C" -->
## Download and install AWS CLI
<!-- #endregion -->

```bash id="ei0Vm3p9UkT1"
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -o awscliv2.zip
sudo ./aws/install
```

<!-- #region id="1xoSKwf7U77e" -->
## Specify AWS credentials

Modify the contents of the next cell to specify your AWS credentials as strings. 

If you see the following exception:

`TypeError: str expected, not NoneType`

It means that you did not specify the credentials correctly.
<!-- #endregion -->

```python id="CaRjFdSoT-q1"
import os
# *** REPLACE None in the next 2 lines with your AWS key values ***
os.environ['AWS_ACCESS_KEY_ID'] = None
os.environ['AWS_SECRET_ACCESS_KEY'] = None
```

<!-- #region id="aAMFo90AVJuI" -->
## Confirm the credentials

Run the next cell to validate your credentials.
<!-- #endregion -->

```bash id="VZqAz5PjS_f1"
aws sts get-caller-identity
```

<!-- #region id="66DsruTZWERS" -->
If you have specified the correct credentials as values for the `AWS_ACCESS_KEY_ID` and the `AWS_SECRET_ACCESS_KEY` environment variables, then `aws sts get-caller-identity` used by the previous cell should have returned back the `UserId`, `Account` and the `Arn` for the credentials, resembling the following

```
{
    "UserId": "█████████████████████",
    "Account": "████████████",
    "Arn": "arn:aws:iam::████████████:user/█████████"
}
```
<!-- #endregion -->

<!-- #region id="wywu4hC-WPxV" -->
## Specify the region

Replace the `None` in the next cell with your AWS region name, for example `us-west-2`.
<!-- #endregion -->

```python id="IowJTSN1e8B-"
# *** REPLACE None in the next line with your AWS region ***
os.environ['AWS_DEFAULT_REGION'] = None
```

<!-- #region id="ZwJSUTvlfSE0" -->
If you have specified the region correctly, the following cell should return back the region that you have specifies.
<!-- #endregion -->

```bash id="2CssvgRfUSu9"
echo $AWS_DEFAULT_REGION
```

<!-- #region id="_s0XszRAi4sO" -->
## Create unique bucket ID

Use the bash `$RANDOM` pseudo-random number generator and the first 32 characters of the `md5sum` output to produce a unique bucket ID.
<!-- #endregion -->

```python id="R99zeeQkfmak"
BUCKET_ID = !echo $(echo $RANDOM | md5sum | cut -c -32)
os.environ['BUCKET_ID'] = next(iter(BUCKET_ID))
os.environ['BUCKET_ID']
```

<!-- #region id="9KPk03iKkQnq" -->
## Save a backup copy of the `BUCKET_ID`

The next cell saves the contents of the `BUCKET_ID` environment variable to a `BUCKET_ID` file as a backup.
<!-- #endregion -->

```python id="UiKvBqAof63B"
val = os.environ['BUCKET_ID']
%store val > BUCKET_ID
!cat BUCKET_ID
```

<!-- #region id="GSR-WujrkVhF" -->
## <font color=red>Download the `BUCKET_ID` file</font>

Ensure that you have a backup copy of the `BUCKET_ID` file created by the previous cell before proceeding. The contents of the `BUCKET_ID` file are going to be reused later in this notebook and in the other notebooks.

<!-- #endregion -->

<!-- #region id="2RRQXy_AmfrD" -->
## Create an AWS bucket
<!-- #endregion -->

```bash id="C4OrMK8bmjUM"
aws s3api create-bucket --bucket dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION --create-bucket-configuration LocationConstraint=$AWS_DEFAULT_REGION
```

<!-- #region id="jRg9sMTFmqEV" -->
If the previous cell executed successfully, then it should have produced an output resembling the following:

```
{
    "Location": "http:/dc-taxi-████████████████████████████████-█████████.s3.amazonaws.com/"
}
```

You can return back the name of the bucket by running the following cell:
<!-- #endregion -->

```python id="VMekJlTGm-11"
!echo s3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION
```

<!-- #region id="4CJ2C81YnIO6" -->
You can also use the AWS CLI `list-buckets` command to print out all the buckets that exist in your AWS account, however the printed names will not show the `s3://` prefix:
<!-- #endregion -->

```python id="FM6SjdPBkuwH"
!aws s3api list-buckets
```

<!-- #region id="IlTfJ8S5ruJU" -->
## Upload the DC taxi dataset to AWS S3

Synchronize the contents of the `dctaxi` directory (where you unzipped the dataset) to the `csv` folder in the S3 bucket you just created. 

* **the next cell should take about 4 minutes to complete**
<!-- #endregion -->

```bash id="BQ3JWPb5kPBU"
aws s3 sync \
  --exclude 'README*' \
  dctaxi/ s3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/csv/
```

<!-- #region id="m6cK-rVvsrQx" -->
## Confirm a successful copy

You can check whether the `aws sync` command completed successfully, by listing the contents of the newly created bucket. Run the following cell:
<!-- #endregion -->

```python id="yD0AuIsbrmhR"
!aws s3 ls --recursive --summarize --human-readable s3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/csv/ | tail -n 2
```

<!-- #region id="Cwjo2RU0tCkB" -->
which should have returned

```
Total Objects: 54
   Total Size: 11.2 GiB
```

if the dataset was copied to S3 successfully.

As before you can remove the `tail -n 2` part in the previous cell to report the entire contents of the `csv` folder on S3.
<!-- #endregion -->

<!-- #region id="H8BgQUXCufMo" -->
## Create AWS role and policy to allow Glue to access the S3 bucket
<!-- #endregion -->

```bash id="JJNiedyRspHT"
aws iam detach-role-policy --role-name AWSGlueServiceRole-dc-taxi --policy-arn arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole && \
aws iam delete-role-policy --role-name AWSGlueServiceRole-dc-taxi --policy-name GlueBucketPolicy && \
aws iam delete-role --role-name AWSGlueServiceRole-dc-taxi

aws iam create-role --path "/service-role/" --role-name AWSGlueServiceRole-dc-taxi --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "glue.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'

aws iam attach-role-policy --role-name AWSGlueServiceRole-dc-taxi --policy-arn arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole

aws iam put-role-policy --role-name AWSGlueServiceRole-dc-taxi --policy-name GlueBucketPolicy --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:*"
            ],
            "Resource": [
                "arn:aws:s3:::dc-taxi-'$BUCKET_ID'-'$AWS_DEFAULT_REGION'/*"
            ]
        }
    ]
}'
```

<!-- #region id="aZgCUjxAxAu2" -->
## Create an AWS Glue database
<!-- #endregion -->

```bash id="joKA9y_9wLtH"
aws glue delete-database --name dc_taxi_db 2> /dev/null

aws glue create-database --database-input '{
  "Name": "dc_taxi_db"
}'

aws glue get-database --name 'dc_taxi_db'
```

<!-- #region id="GFmG7ra7xlaD" -->
## Create an AWS Glue crawler

Save the results of crawling the S3 bucket with the DC taxi dataset to the AWS Glue database created in the previous cell.
<!-- #endregion -->

```bash id="n-BMVq29xXi0"
aws glue delete-crawler --name dc-taxi-csv-crawler 2> /dev/null

aws glue create-crawler \
  --name dc-taxi-csv-crawler \
  --database-name dc_taxi_db \
  --table-prefix dc_taxi_ \
  --role $( aws iam get-role \
              --role-name AWSGlueServiceRole-dc-taxi \
              --query 'Role.Arn' \
              --output text ) \
   --targets '{
  "S3Targets": [
    {
      "Path": "s3://dc-taxi-'$BUCKET_ID'-'$AWS_DEFAULT_REGION'/csv"
    }]
}'

aws glue start-crawler --name dc-taxi-csv-crawler
```

<!-- #region id="nS7INrVvyHL8" -->
## Check the status of the AWS Glue crawler

When running this notebook, you need to re-run the next cell to get updates on crawler status. It should cycle through `STARTING`, `RUNNING`, `STOPPING`, and `READY`. 

It will take the crawler about a minute to finish crawling the DC taxi dataset.
<!-- #endregion -->

```bash id="m_XxAWuDx66T"
aws glue get-crawler --name dc-taxi-csv-crawler --query 'Crawler.State' --output text
```

<!-- #region id="I_DisBrCIQaL" -->
Poll the crawler state every minute to wait for it to finish.
<!-- #endregion -->

```bash id="i5PfPixmKlVs"
printf "Waiting for crawler to finish..."
until echo "$(aws glue get-crawler --name dc-taxi-csv-crawler --query 'Crawler.State' --output text)" | grep -q "READY"; do
   sleep 60
   printf "..."
done
printf "done\n"
```

<!-- #region id="RvmEu4JiyNPq" -->
## Find out the last known status of the AWS Glue crawler
<!-- #endregion -->

```python id="Nm30IriYx_yH"
!aws glue get-crawler --name dc-taxi-csv-crawler --query 'Crawler.LastCrawl'
```

<!-- #region id="I5CH23_WyZNS" -->
## Describe the table created by the AWS Glue crawler
<!-- #endregion -->

```python id="xHPAzHfGyLzF"
!aws glue get-table --database-name dc_taxi_db --name dc_taxi_csv
```

<!-- #region id="osbkOyeDLHvi" -->
## Create a PySpark job to convert CSV to Parquet

The next cell uses the Jupyter `%%writefile` magic to save the source code for the PySpark job to the `dctaxi_csv_to_parquet.py` file.
<!-- #endregion -->

```python id="880hw7aeKqXB"
%%writefile dctaxi_csv_to_parquet.py
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ['JOB_NAME',
                                     'BUCKET_SRC_PATH',
                                     'BUCKET_DST_PATH',
									                    'DST_VIEW_NAME'])

BUCKET_SRC_PATH = args['BUCKET_SRC_PATH']
BUCKET_DST_PATH = args['BUCKET_DST_PATH']
DST_VIEW_NAME = args['DST_VIEW_NAME']

sc = SparkContext()
glueContext = GlueContext(sc)
logger = glueContext.get_logger()
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

df = ( spark.read.format("csv")
		.option("header", True)
		.option("inferSchema", True)
    .option("multiLine", True)  
		.option("delimiter", "|")
		.load(f"{BUCKET_SRC_PATH}") )

df.createOrReplaceTempView(f"{DST_VIEW_NAME}")

query_df = spark.sql(f"""
SELECT 

  CAST(fareamount AS DOUBLE) AS fareamount_double,
  CAST(fareamount AS STRING) AS fareamount_string,

  CAST(origindatetime_tr AS STRING) AS origindatetime_tr,

  CAST(origin_block_latitude AS DOUBLE) AS origin_block_latitude_double,
  CAST(origin_block_latitude AS STRING) AS origin_block_latitude_string,

  CAST(origin_block_longitude AS DOUBLE) AS origin_block_longitude_double,
  CAST(origin_block_longitude AS STRING) AS origin_block_longitude_string,

  CAST(destination_block_latitude AS DOUBLE) AS destination_block_latitude_double,
  CAST(destination_block_latitude AS STRING) AS destination_block_latitude_string,

  CAST(destination_block_longitude AS DOUBLE) AS destination_block_longitude_double,
  CAST(destination_block_longitude AS STRING) AS destination_block_longitude_string,

  CAST(mileage AS DOUBLE) AS mileage_double,
  CAST(mileage AS STRING) AS mileage_string 

FROM {DST_VIEW_NAME}""".replace('\n', ''))

query_df.write.parquet(f"{BUCKET_DST_PATH}", mode="overwrite")

job.commit()
```

<!-- #region id="b5kw70zRLskj" -->
## Copy the PySpark job code to the S3 bucket
<!-- #endregion -->

```bash id="sy84Mn4YLsDH"
aws s3 cp dctaxi_csv_to_parquet.py s3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/glue/
aws s3 ls s3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/glue/dctaxi_csv_to_parquet.py
```

<!-- #region id="m2LA3Y2IgM9p" -->
## Create and start the PySpark job
<!-- #endregion -->

```bash id="8djrkqIFL1qY"
aws glue delete-job --job-name dc-taxi-csv-to-parquet-job 2> /dev/null

aws glue create-job \
  --name dc-taxi-csv-to-parquet-job \
  --role $(aws iam get-role --role-name AWSGlueServiceRole-dc-taxi --query 'Role.Arn' --output text) \
  --default-arguments '{"--TempDir":"s3://dc-taxi-'$BUCKET_ID'-'$AWS_DEFAULT_REGION'/glue/"}' \
  --command '{
    "ScriptLocation": "s3://dc-taxi-'$BUCKET_ID'-'$AWS_DEFAULT_REGION'/glue/dctaxi_csv_to_parquet.py",
    "Name": "glueetl",
    "PythonVersion": "3"
  }' \
  --glue-version "2.0"

aws glue start-job-run \
  --job-name dc-taxi-csv-to-parquet-job \
  --arguments='--BUCKET_SRC_PATH="'$(
      echo s3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/csv/*.txt
    )'",
  --BUCKET_DST_PATH="'$(
      echo s3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/parquet/
    )'",
  --DST_VIEW_NAME="dc_taxi_csv"'
```

<!-- #region id="zx1NQZYLgdel" -->
In case of a successful completion, the last cell should have produced an output similar to the following:

```
{
    "Name": "dc-taxi-csv-to-parquet-job"
}
{
    "JobRunId": "jr_925ab8ea6e5bdd64d4491c6f641bcc58f5c7b0140edcdba9896052c70d3675fe"
}
```
<!-- #endregion -->

<!-- #region id="FYrM2iS9gnmX" -->
## Monitor the job execution

* **it should take about 3 minutes for the job to complete**

Once the PySpark job completes successfully, the job execution status should  change from `RUNNING` to `SUCCEEDED`. You can re-run the next cell to get the updated job status.
<!-- #endregion -->

```python id="oYwrYdVFL4WH"
!aws glue get-job-runs --job-name dc-taxi-csv-to-parquet-job --output text --query 'JobRuns[0].JobRunState'
```

<!-- #region id="U2iFXtBuKB9i" -->
Poll the job every minute to wait for it to finish
<!-- #endregion -->

```bash id="HDNKiOukOgco"
printf "Waiting for the job to finish..."
while echo "$(aws glue get-job-runs --job-name dc-taxi-csv-to-parquet-job --query 'JobRuns[0].JobRunState')" | grep -q -E "STARTING|RUNNING|STOPPING"; do
   sleep 60
   printf "..."
done
aws glue get-job-runs --job-name dc-taxi-csv-to-parquet-job --output text --query 'JobRuns[0].JobRunState'
```

<!-- #region id="MYxJwg-cgu3a" -->
## Confirm the CSV to Parquet convertion job completed successfully


<!-- #endregion -->

```python id="9uoI1v1Jgszl"
!aws s3 ls --recursive --summarize --human-readable s3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/parquet/ | tail -n 2
```

<!-- #region id="ZJguy-X4hoKi" -->
Assuming the Parquet files have been correctly created in the S3 bucket, the previous cell should output the following:

```
Total Objects: 53
   Total Size: 941.3 MiB
```
<!-- #endregion -->

<!-- #region id="3xxR3fpLiIQc" -->
## Crawl the Parquet dataset
<!-- #endregion -->

```bash id="F_wYQ0GXg2q7"
aws glue delete-crawler --name dc-taxi-parquet-crawler  2> /dev/null

aws glue create-crawler --name dc-taxi-parquet-crawler --database-name dc_taxi_db --table-prefix dc_taxi_ --role `aws iam get-role --role-name AWSGlueServiceRole-dc-taxi --query 'Role.Arn' --output text` --targets '{
  "S3Targets": [
    {
      "Path": "s3://dc-taxi-'$BUCKET_ID'-'$AWS_DEFAULT_REGION'/parquet/"
    }]
}'

aws glue start-crawler --name dc-taxi-parquet-crawler
```

<!-- #region id="rjpuQT4Tijp_" -->
## Monitor the Parquet crawler status

* **the crawler should take about 2 minutes to finish**

Once done, the crawler should return to a `READY` state.
<!-- #endregion -->

```python id="97lxt-GHiUaI"
!aws glue get-crawler --name dc-taxi-parquet-crawler --query 'Crawler.State' --output text
```

<!-- #region id="thUt_02jKmRS" -->
Poll the crawler status every minute to wait for it to finish
<!-- #endregion -->

```bash id="Ys-tvL_xO3ly"
printf "Waiting for crawler to finish..."
until echo "$(aws glue get-crawler --name dc-taxi-parquet-crawler --query 'Crawler.State' --output text)" | grep -q "READY"; do
   sleep 10
   printf "..."
done
aws glue get-crawler --name dc-taxi-parquet-crawler --query 'Crawler.State' --output text
```

<!-- #region id="8omIXmOojBlA" -->
## Confirm that the crawler successfully created the `dc_taxi_parquet` table

If the crawler completed successfully, the number of records in the `dc_taxi_parquet` table as reported by the following command should be equal to `53173692`
<!-- #endregion -->

```python id="3Id27YhNihfz"
!aws glue get-table --database-name dc_taxi_db --name dc_taxi_parquet --query "Table.Parameters.recordCount" --output text
```
