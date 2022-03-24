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
# Data exploration at scale with AWS Athena
<!-- #endregion -->

<!-- #region id="8HPLiT_xlliJ" -->
> Note: https://learning.oreilly.com/library/view/mlops-engineering-at/9781617297762/OEBPS/Text/03.htm
<!-- #endregion -->

<!-- #region id="nWyD1Bt9k26w" -->
## <font color=red>Upload the `BUCKET_ID` file</font>

Before proceeding, ensure that you have a backup copy of the `BUCKET_ID` file created in the [Chapter 2](https://colab.research.google.com/github/osipov/smlbook/blob/master/ch2.ipynb) notebook before proceeding. The contents of the `BUCKET_ID` file are reused later in this notebook and in the other notebooks.

<!-- #endregion -->

```python id="cwPOIYDdnXKN"
import os
from pathlib import Path
assert Path('BUCKET_ID').exists(), "Place the BUCKET_ID file in the current directory before proceeding"

BUCKET_ID = Path('BUCKET_ID').read_text().strip()
os.environ['BUCKET_ID'] = BUCKET_ID
os.environ['BUCKET_ID']
```

<!-- #region id="GZ2rTEBfU20C" -->
## **OPTIONAL:** Download and install AWS CLI

This is unnecessary if you have already installed AWS CLI in a preceding notebook.
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

<!-- #region id="2RRQXy_AmfrD" -->
## Download a tiny sample

Download a tiny sample of the dataset from https://gist.github.com/osipov/1fc0265f8f829d9d9eee8393657423a9 to a `trips_sample.csv` file which you are going to use to learn about using the Athena interface.
<!-- #endregion -->

```bash id="D23pmPM2p3Mk"
wget -q https://gist.githubusercontent.com/osipov/1fc0265f8f829d9d9eee8393657423a9/raw/9957c1f09cdfa64f8b8d89cfec532a0e150d5178/trips_sample.csv
ls -ltr trips_sample.csv
cat trips_sample.csv
```

<!-- #region id="BrZqHUzLrHdQ" -->
Assuming the previous cell executed successfully, it should have returned the following result:

```
-rw-r--r-- 1 root root 378 Nov 23 19:50 trips_sample.csv
fareamount_double,origin_block_latitude,origin_block_longitude,destination_block_latitude,destination_block_longitude
8.11,38.900769,-77.033644,38.912239,-77.036514
5.95,38.912609,-77.030788,38.906445,-77.023978
7.57,38.900773,-77.03655,38.896131,-77.024975
11.61,38.892101000000004,-77.044208,38.905969,-77.06564399999999
4.87,38.899615000000004,-76.980387,38.900638,-76.97023
```
<!-- #endregion -->

<!-- #region id="ifWi_kpRrSIp" -->
## Upload `trips_sample.csv` to your object storage bucket
<!-- #endregion -->

```bash id="_t1IzWylptua"
aws s3 cp trips_sample.csv s3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/samples/trips_sample.csv
aws s3 ls s3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/samples/trips_sample.csv
```

<!-- #region id="RaWojJMWuV0Z" -->
The output of the `aws s3 ls` command above should include the following, confirming that the file was uploaded successfully.

```
2020-11-23 20:07:31        378 trips_sample.csv
```
<!-- #endregion -->

<!-- #region id="UtGQ6KAx3Yed" -->
## Create an Athena workgroup

Create a `dc_taxi_athena_workgroup` for your Athena project, assuming one does not exist yet.
<!-- #endregion -->

```bash id="B9DhccSFxZ35"
aws athena delete-work-group --work-group dc_taxi_athena_workgroup --recursive-delete-option 2> /dev/null
aws athena create-work-group --name dc_taxi_athena_workgroup \
--configuration "ResultConfiguration={OutputLocation=s3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/athena},EnforceWorkGroupConfiguration=false,PublishCloudWatchMetricsEnabled=false"
```

<!-- #region id="Mcn4o2CrDbpW" -->
## Query Athena and Report Query Status
<!-- #endregion -->

```bash id="iQP_USHQu2Kx"
SQL="
CREATE EXTERNAL TABLE IF NOT EXISTS dc_taxi_db.dc_taxi_csv_sample_strings(
        fareamount STRING,
        origin_block_latitude STRING,
        origin_block_longitude STRING,
        destination_block_latitude STRING,
        destination_block_longitude STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
LOCATION 's3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/samples/'
TBLPROPERTIES ('skip.header.line.count'='1');"

ATHENA_QUERY_ID=$(aws athena start-query-execution \
--work-group dc_taxi_athena_workgroup \
--query 'QueryExecutionId' \
--output text \
--query-string "$SQL")

echo $SQL

echo $ATHENA_QUERY_ID
until aws athena get-query-execution \
  --query 'QueryExecution.Status.State' \
  --output text \
  --query-execution-id $ATHENA_QUERY_ID | grep -v "RUNNING";
do
  printf '.'
  sleep 1; 
done
```

<!-- #region id="Zduq4AQbDlTp" -->
## Download and Preview a Utility Script to Query Athena

The script is downloaded as `utils.sh` and is loaded in the upcoming cells using `source utils.sh` command.
<!-- #endregion -->

```bash id="-1A2IUgK62N2"
wget -q https://raw.githubusercontent.com/osipov/smlbook/master/utils.sh
ls -l utils.sh
```

<!-- #region id="de1AFcAUDt3R" -->
## Output Athena Query to a Text Table
<!-- #endregion -->

```bash id="siQ1_XcX8jn2"
source utils.sh
SQL="
SELECT

origin_block_latitude || ' , ' || origin_block_longitude
    AS origin,

destination_block_latitude || '  , ' || destination_block_longitude
    AS destination

FROM
    dc_taxi_db.dc_taxi_csv_sample_strings
"
athena_query_to_table "$SQL" "ResultSet.Rows[*].[Data[0].VarCharValue,Data[1].VarCharValue]"
```

<!-- #region id="2U6ugJFND0ga" -->
## Output Athena Query to JSON for a Pandas DataFrame
<!-- #endregion -->

```bash id="Ba8rMXvhBWdU"
source utils.sh ; athena_query_to_pandas """
SELECT

origin_block_latitude || ' , ' || origin_block_longitude
    AS origin,

destination_block_latitude || '  , ' || destination_block_longitude
    AS destination

FROM
    dc_taxi_db.dc_taxi_csv_sample_strings
"""
```

<!-- #region id="5TmJx0tQD40z" -->
## Create a Utility Function to Read AWS CLI JSON as Pandas

Note that the `utils.sh` script saves the output from Athena to `/tmp/awscli.json`
<!-- #endregion -->

```python id="VT20bpOrCVZ9"
import pandas as pd
def awscli_to_df():
  json_df = pd.read_json('/tmp/awscli.json')
  df = pd.DataFrame(json_df[0].tolist(), index = json_df.index, columns = json_df[0].tolist()[0]).drop(0, axis = 0)
  return df
```

```python id="MQi-CfOHAqgU"
awscli_to_df()
```

<!-- #region id="m0THrt-DOtdv" -->
## Apply Athena schema-on-read with columns as `DOUBLE`
<!-- #endregion -->

```bash id="bScMRJ-L28J-"
source utils.sh ; athena_query "
CREATE EXTERNAL TABLE IF NOT EXISTS dc_taxi_db.dc_taxi_csv_sample_double(
        fareamount DOUBLE,
        origin_block_latitude DOUBLE,
        origin_block_longitude DOUBLE,
        destination_block_latitude DOUBLE,
        destination_block_longitude DOUBLE
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
LOCATION 's3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/samples/'
TBLPROPERTIES ('skip.header.line.count'='1');
"
```

```bash id="hGIsmGgezora"
source utils.sh ; athena_query_to_pandas "
SELECT ROUND(MAX(fareamount) - MIN(fareamount), 2)
FROM dc_taxi_db.dc_taxi_csv_sample_double
"
```

```python id="K2kyee1VFAIx"
awscli_to_df()
```

<!-- #region id="Rp71QW-ZOj2Q" -->
## Explore 10 records from the DC taxi dataset
<!-- #endregion -->

```bash id="ZV4CZ8Zd8wNL"
source utils.sh ; athena_query_to_pandas "
SELECT fareamount_double,
         origin_block_latitude_double,
         origin_block_longitude_double,
         destination_block_latitude_double,
         destination_block_longitude_double,
         origindatetime_tr
FROM dc_taxi_db.dc_taxi_parquet
LIMIT 10
"
```

```python id="_lhTTGCKFId5"
awscli_to_df()
```

<!-- #region id="W-6X9lnOOWHv" -->
## What is the number of the timestamps with NULL values?
<!-- #endregion -->

```bash id="kIEuIU3tJMXZ"
source utils.sh ; athena_query_to_pandas "
SELECT
    (SELECT COUNT(*) FROM dc_taxi_db.dc_taxi_parquet) AS total,
    COUNT(*) AS null_origindate_time_total
FROM
    dc_taxi_db.dc_taxi_parquet
WHERE
    origindatetime_tr IS NULL
"
```

```python id="vmilocpPF5F4"
awscli_to_df()
```

<!-- #region id="Yt5PI2xyOLhQ" -->
## How many timestamps are un-parsable?
<!-- #endregion -->

```bash id="y-iugDT4KZDW"
source utils.sh ; athena_query_to_pandas "
SELECT
    (SELECT COUNT(*) FROM dc_taxi_db.dc_taxi_parquet)
    - COUNT(DATE_PARSE(origindatetime_tr, '%m/%d/%Y %H:%i'))
    AS origindatetime_not_parsed
FROM
    dc_taxi_db.dc_taxi_parquet
WHERE
    origindatetime_tr IS NOT NULL;
"
```

```python id="qovgo1ToGByv"
awscli_to_df()
```

<!-- #region id="ZJOQp7FvN81f" -->
## How often are parts of the pick up location coordinate missing?
<!-- #endregion -->

```bash id="SzuPPwdlRkQo"
source utils.sh ; athena_query_to_pandas "
SELECT
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*)
                        FROM dc_taxi_db.dc_taxi_parquet), 2)

        AS percentage_null,

    (SELECT COUNT(*)
     FROM dc_taxi_db.dc_taxi_parquet
     WHERE origin_block_longitude_double IS NULL
     OR origin_block_latitude_double IS NULL)

        AS either_null,

    (SELECT COUNT(*)
     FROM dc_taxi_db.dc_taxi_parquet
     WHERE origin_block_longitude_double IS NULL
     AND origin_block_latitude_double IS NULL)

        AS both_null

FROM
    dc_taxi_db.dc_taxi_parquet
WHERE
    origin_block_longitude_double IS NULL
    OR origin_block_latitude_double IS NULL
"
```

```python id="i1-i7wx-R9Op"
awscli_to_df()
```

<!-- #region id="fJMCC1D0MgoL" -->
## How often are parts of the drop off coordinates missing?

Repeat the previous analysis
<!-- #endregion -->

```bash id="pWp6SMuySyXM"
source utils.sh ; athena_query_to_pandas "
SELECT
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*)
                        FROM dc_taxi_db.dc_taxi_parquet), 2)

        AS percentage_null,

    (SELECT COUNT(*)
     FROM dc_taxi_db.dc_taxi_parquet
     WHERE destination_block_longitude_double IS NULL
     OR destination_block_latitude_double IS NULL)

        AS either_null,

    (SELECT COUNT(*)
     FROM dc_taxi_db.dc_taxi_parquet
     WHERE destination_block_longitude_double IS NULL
     AND destination_block_latitude_double IS NULL)

        AS both_null

FROM
    dc_taxi_db.dc_taxi_parquet
WHERE
    destination_block_longitude_double IS NULL
    OR destination_block_latitude_double IS NULL
"
```

```python id="os9ckvk4TQSd"
awscli_to_df()
```

<!-- #region id="FL-blOX9MZtV" -->
 ## Find the count and the fraction of the missing coordinates
<!-- #endregion -->

```bash id="LosJk-AyTxO8"
source utils.sh ; athena_query_to_pandas "
SELECT
    COUNT(*)
      AS total,

    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*)
                              FROM dc_taxi_db.dc_taxi_parquet), 2)
      AS percent

FROM
    dc_taxi_db.dc_taxi_parquet

WHERE
    origin_block_latitude_double IS NULL
    OR origin_block_longitude_double IS NULL
    OR destination_block_latitude_double IS NULL
    OR destination_block_longitude_double IS NULL
"
```

```python id="znSbval0T5OV"
awscli_to_df()
```

<!-- #region id="PN3LuLQkMVcs" -->
## Query for the values and quantities of fareamount_string that failed to parse as a double
<!-- #endregion -->

```bash id="ydZ17NanUF4I"
source utils.sh ; athena_query_to_pandas "
SELECT
    fareamount_string,
    COUNT(fareamount_string) AS rows,
    ROUND(100.0 * COUNT(fareamount_string) /
          ( SELECT COUNT(*)
            FROM dc_taxi_db.dc_taxi_parquet), 2)

      AS percent
FROM
    dc_taxi_db.dc_taxi_parquet
WHERE
    fareamount_double IS NULL
    AND fareamount_string IS NOT NULL
GROUP BY
    fareamount_string;
"
```

```python id="mhQDbvztULpN"
awscli_to_df()
```

<!-- #region id="2xWUyJ1hMLq6" -->
## Explore summary statistics of the `fareamount_double` column
<!-- #endregion -->

```bash id="Glbl854gVSdD"
source utils.sh ; athena_query_to_pandas "
WITH
src AS (SELECT
            fareamount_double AS val
        FROM
            dc_taxi_db.dc_taxi_parquet),

stats AS
    (SELECT
        MIN(val) AS min,
        APPROX_PERCENTILE(val, 0.25) AS q1,
        APPROX_PERCENTILE(val ,0.5) AS q2,
        APPROX_PERCENTILE(val, 0.75) AS q3,
        AVG(val) AS mean,
        STDDEV(val) AS std,
        MAX(val) AS max
    FROM
        src)

SELECT
    DISTINCT min, q1, q2, q3, max

FROM
    dc_taxi_db.dc_taxi_parquet, stats
"
```

```python id="lTHagOwHVaBf"
awscli_to_df()
```

<!-- #region id="wHlNB-JnMAmX" -->
## What percentage of fare amount values are null or below the minimum threshold?
<!-- #endregion -->

```bash id="OFQcU6sVVqtc"
source utils.sh ; athena_query_to_pandas "
WITH
src AS (SELECT
            COUNT(*) AS total
        FROM
            dc_taxi_db.dc_taxi_parquet
        WHERE
            fareamount_double IS NOT NULL)

SELECT
    ROUND(100.0 * COUNT(fareamount_double) / MIN(total), 2) AS percent
FROM
    dc_taxi_db.dc_taxi_parquet, src
WHERE
    fareamount_double < 3.25
    AND fareamount_double IS NOT NULL
"
```

```python id="T5WqqGHjVzAv"
awscli_to_df()
```

<!-- #region id="gfNCCzsILt6c" -->
## Compute summary statistics for the cases where the fareamount_string failed to parse
<!-- #endregion -->

```bash id="8fnCYXMUV-Da"
source utils.sh ; athena_query_to_pandas "
SELECT
    fareamount_string,
    ROUND( MIN(mileage_double), 2) AS min,
    ROUND( APPROX_PERCENTILE(mileage_double, 0.25), 2) AS q1,
    ROUND( APPROX_PERCENTILE(mileage_double ,0.5), 2) AS q2,
    ROUND( APPROX_PERCENTILE(mileage_double, 0.75), 2) AS q3,
    ROUND( MAX(mileage_double), 2) AS max
FROM
    dc_taxi_db.dc_taxi_parquet
WHERE
    fareamount_string LIKE 'NULL'
GROUP BY
    fareamount_string
"
```

```python id="CJ5QxWdbWEFD"
awscli_to_df()
```

<!-- #region id="492StbYgLLf3" -->
## Figure out the lower left and upper right boundary locations

Plugging the latitude and longitude coordinates reported by the query into OpenStreetMap ( https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route=38.8110%2C-77.1130%3B38.9950%2C-76.9100#map=11/38.9025/-77.0094 ) yields 21.13 miles or an estimate of $ \$48.89 (21.13 * \$2.16/mile + \$3.25) $.

<!-- #endregion -->

```bash id="u-IkoJL4Uf1N"
source utils.sh ; athena_query_to_pandas "
SELECT 
  MIN(lat) AS lower_left_latitude,
  MIN(lon) AS lower_left_longitude,
  MAX(lat) AS upper_right_latitude,
  MAX(lon) AS upper_right_longitude

 FROM (
  SELECT 
    MIN(origin_block_latitude_double) AS lat,
    MIN(origin_block_longitude_double) AS lon 
  FROM "dc_taxi_db"."dc_taxi_parquet" 
  
  UNION

  SELECT 
    MIN(destination_block_latitude_double) AS lat, 
    MIN(destination_block_longitude_double) AS lon 
  FROM "dc_taxi_db"."dc_taxi_parquet" 
  
  UNION

  SELECT 
    MAX(origin_block_latitude_double) AS lat, 
    MAX(origin_block_longitude_double) AS lon 
  FROM "dc_taxi_db"."dc_taxi_parquet" 
  
  UNION

  SELECT 
    MAX(destination_block_latitude_double) AS lat, 
    MAX(destination_block_longitude_double) AS lon 
  FROM "dc_taxi_db"."dc_taxi_parquet"

)
"
```

```python id="K4jRkxVhWZeu"
awscli_to_df()
```

<!-- #region id="gV-yM36xJpMK" -->
## Compute normally distributed averages of random samples
<!-- #endregion -->

```bash id="N8SORcI7XFGd"
source utils.sh ; athena_query_to_pandas "
WITH dc_taxi AS 
(SELECT *, 
    origindatetime_tr 
    || fareamount_string 
    || origin_block_latitude_string 
    || origin_block_longitude_string 
    || destination_block_latitude_string 
    || destination_block_longitude_string 
    || mileage_string AS objectid

    FROM "dc_taxi_db"."dc_taxi_parquet"

    WHERE fareamount_double >= 3.25
            AND fareamount_double IS NOT NULL
            AND mileage_double > 0 )

SELECT AVG(mileage_double) AS average_mileage
FROM dc_taxi
WHERE objectid IS NOT NULL
GROUP BY  MOD( ABS( from_big_endian_64( xxhash64( to_utf8( objectid ) ) ) ), 1000)
" ResultSet.Rows[*].[Data[].VarCharValue] 1000
```

```python id="lNf2VXJ6XFN_"
awscli_to_df()
```

<!-- #region id="knnL4gesLAiy" -->
## Visually confirm that the means of samples are normally distributed
<!-- #endregion -->

```python id="oDL4h5yiKHO_"
%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize = (12, 9))

df = awscli_to_df()
df.average_mileage = df.average_mileage.astype(float)
df.average_mileage -= df.average_mileage.mean()
df.average_mileage /= df.average_mileage.std()
df.average_mileage.plot.hist(bins = 30);
```

<!-- #region id="7ccBsRpiJa6j" -->
## Produce a statistical upper bound estimate for the mileage
<!-- #endregion -->

```bash id="f0NKqmy6toiR"
source utils.sh ; athena_query_to_pandas "
WITH dc_taxi AS 
(SELECT *, 
    origindatetime_tr 
    || fareamount_string 
    || origin_block_latitude_string 
    || origin_block_longitude_string 
    || destination_block_latitude_string 
    || destination_block_longitude_string 
    || mileage_string AS objectid

    FROM "dc_taxi_db"."dc_taxi_parquet"

    WHERE fareamount_double >= 3.25
            AND fareamount_double IS NOT NULL
            AND mileage_double > 0 ),

dc_taxi_samples AS (
    SELECT AVG(mileage_double) AS average_mileage
    FROM dc_taxi
    WHERE objectid IS NOT NULL
    GROUP BY  MOD( ABS( from_big_endian_64( xxhash64( to_utf8( objectid ) ) ) ) , 1000)
)
SELECT AVG(average_mileage) + 4 * STDDEV(average_mileage)
FROM dc_taxi_samples
"
```

```python id="LoQWBPVTt0P9"
awscli_to_df()
```

```python id="OJUb_ncZwust"
upper_mileage = 2.16 * awscli_to_df().mean().item() + 3.25
upper_mileage
```

<!-- #region id="4cKoV6ZOJSvb" -->
## Produce the final estimate for the upper bound on fare amount
<!-- #endregion -->

```bash id="UslfD8rgxviR"
source utils.sh ; athena_query_to_pandas "
WITH dc_taxi AS 
(SELECT *, 
    origindatetime_tr 
    || fareamount_string 
    || origin_block_latitude_string 
    || origin_block_longitude_string 
    || destination_block_latitude_string 
    || destination_block_longitude_string 
    || mileage_string AS objectid

    FROM "dc_taxi_db"."dc_taxi_parquet"

    WHERE fareamount_double >= 3.25
            AND fareamount_double IS NOT NULL
            AND mileage_double > 0 ),

dc_taxi_samples AS (
    SELECT AVG(fareamount_double) AS average_fareamount
    FROM dc_taxi
    WHERE objectid IS NOT NULL
    GROUP BY  MOD( ABS( from_big_endian_64( xxhash64( to_utf8( objectid ) ) ) ) , 1000)
)
SELECT AVG(average_fareamount) + 4 * STDDEV(average_fareamount)
FROM dc_taxi_samples
"
```

```python id="0zl0aN90xvo9"
awscli_to_df()
```

```python id="_Ws1qkOJyJ6A"
upper_fareamount = awscli_to_df().mean().item()
upper_fareamount
```

```python id="HnRDFMN6yUmF"
means = [15.96, 29.19, 48.89, 560, 2,422.45]
sum(means) / len(means)
```

<!-- #region id="ZHWQVcIHJDuz" -->
## Check the percentage of the dataset above the upper fare amount bound
<!-- #endregion -->

```bash id="jxNv5mAVyh7K"
source utils.sh ; athena_query_to_pandas "
SELECT
    100.0 * COUNT(fareamount_double) / 
      (SELECT COUNT(*) 
      FROM dc_taxi_db.dc_taxi_parquet 
      WHERE fareamount_double IS NOT NULL) AS percent
FROM
    dc_taxi_db.dc_taxi_parquet
WHERE (fareamount_double < 3.25 OR fareamount_double > 179.75)
        AND fareamount_double IS NOT NULL;
"
```

```python id="-L_WoWDAyjeR"
awscli_to_df()
```

<!-- #region id="ntK1nMfLI8wk" -->
## Produce final summary statistics
<!-- #endregion -->

```bash id="FSHcmmTlyvvK"
source utils.sh ; athena_query_to_pandas "
WITH src AS (SELECT fareamount_double AS val
             FROM dc_taxi_db.dc_taxi_parquet
             WHERE fareamount_double IS NOT NULL
             AND fareamount_double >= 3.25
             AND fareamount_double <= 180.0),
stats AS
    (SELECT
     ROUND(MIN(val), 2) AS min,
     ROUND(APPROX_PERCENTILE(val, 0.25), 2) AS q1,
     ROUND(APPROX_PERCENTILE(val, 0.5), 2) AS q2,
     ROUND(APPROX_PERCENTILE(val, 0.75), 2) AS q3,
     ROUND(AVG(val), 2) AS mean,
     ROUND(STDDEV(val), 2) AS std,
     ROUND(MAX(val), 2) AS max
    FROM src)
SELECT min, q1, q2, q3, max, mean, std
FROM stats;
"
```

```python id="yDcKPEtPyv0c"
awscli_to_df()
```

<!-- #region id="8Rb-Rba7Iied" -->
## Check that minimum and maximum locations are within DC boundaries

Using the SQL query and OpenStreetMap ( https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route=38.8106%2C-77.1134%3B38.9940%2C-76.9100#map=11/38.9025/-77.0210 )  check that the minimum and maximum coordinates for the origin latitude and longitude columns confirm that resulting pairs (38.81138, -77.113633) and (38.994217, -76.910012) as well as ( 38.994217,	-76.910012) and (38.81138,	-77.113633) (https://www.openstreetmap.org/directions?engine=fossgis_osrm_car&route=38.994217%2C-76.910012%3B38.81138%2C-77.113633#map=11/38.9025/-77.0210 ) are within DC boundaries. 




<!-- #endregion -->

```bash id="EGxcZAdzzCxg"
source utils.sh ; athena_query_to_pandas "
SELECT
    MIN(origin_block_latitude_double) AS olat_min,
    MIN(origin_block_longitude_double) AS olon_min,
    MAX(origin_block_latitude_double) AS olat_max,
    MAX(origin_block_longitude_double) AS olon_max,
    MIN(destination_block_latitude_double) AS dlat_min,
    MIN(destination_block_longitude_double) AS dlon_min,
    MAX(destination_block_latitude_double) AS dlat_max,
    MAX(destination_block_longitude_double) AS dlon_max
FROM
    dc_taxi_db.dc_taxi_parquet
"
```

```python id="-zAPUH8yzCua"
awscli_to_df()
```

<!-- #region id="rQVARZpYD9gw" -->
## Use a PySpark job to create a VACUUMed dataset

The next cell uses the Jupyter `%%writefile` magic to save the source code for the PySpark job to the `dctaxi_parquet_vacuum.py` file.
<!-- #endregion -->

```python id="b6lJXTYXyN5U"
%%writefile dctaxi_parquet_vacuum.py
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ['JOB_NAME',
                                     'BUCKET_SRC_PATH',
                                     'BUCKET_DST_PATH',
                                     ])

BUCKET_SRC_PATH = args['BUCKET_SRC_PATH']
BUCKET_DST_PATH = args['BUCKET_DST_PATH']

sc = SparkContext()
glueContext = GlueContext(sc)
logger = glueContext.get_logger()
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

df = ( spark
        .read
        .parquet(f"{BUCKET_SRC_PATH}") )

```

<!-- #region id="sh4FbJVrzr44" -->
Apply the SQL query developed though the VACUUM-based analysis of the data to prepare a version of the dataset without any `NULL` values and with an interval applied to the `fareamount_double` column.
<!-- #endregion -->

```python id="E0Upxn7oyScc"
%%writefile -a dctaxi_parquet_vacuum.py

df.createOrReplaceTempView("dc_taxi_parquet")

query_df = spark.sql("""
SELECT
    fareamount_double,
    origindatetime_tr,
    origin_block_latitude_double,
    origin_block_longitude_double,
    destination_block_latitude_double,
    destination_block_longitude_double 
FROM 
  dc_taxi_parquet 
WHERE 
    origindatetime_tr IS NOT NULL
    AND fareamount_double IS NOT NULL
    AND fareamount_double >= 3.25
    AND fareamount_double <= 180.0
    AND origin_block_latitude_double IS NOT NULL
    AND origin_block_longitude_double IS NOT NULL
    AND destination_block_latitude_double IS NOT NULL
    AND destination_block_longitude_double IS NOT NULL
""".replace('\n', ''))


```

<!-- #region id="T08Ergub2N9j" -->
Convert the original, `STRING` formatted `origindatetime_tr` column into a SQL `TIMESTAMP` column named `origindatetime_ts`. The conversion is needed to extract the year, month, day of the week (`dow`), and hour of the taxi trip as separate numeric, `INTEGER` columns for machine learning. Lastly, drop any records that are missing values (for example due to failed conversion), or are duplicated in the dataset.
<!-- #endregion -->

```python id="D7AcexukD9EI"
%%writefile -a dctaxi_parquet_vacuum.py


#parse to check for valid value of the original timestamp
from pyspark.sql.functions import col, to_timestamp, dayofweek, year, month, hour
from pyspark.sql.types import IntegerType

#convert the source timestamp into numeric data needed for machine learning
query_df = (query_df
  .withColumn("origindatetime_ts", to_timestamp("origindatetime_tr", "dd/MM/yyyy HH:mm"))
  .where(col("origindatetime_ts").isNotNull())
  .drop("origindatetime_tr")
  .withColumn( 'year_integer',  year('origindatetime_ts').cast(IntegerType()) )
  .withColumn( 'month_integer',  month('origindatetime_ts').cast(IntegerType()) )
  .withColumn( 'dow_integer',  dayofweek('origindatetime_ts').cast(IntegerType()) )
  .withColumn( 'hour_integer',  hour('origindatetime_ts').cast(IntegerType()) )
  .drop('origindatetime_ts') )

#drop missing data and duplicates
query_df = ( query_df
            .dropna()
            .drop_duplicates() )


```

<!-- #region id="ioxdeCsb39h3" -->
Persists the cleaned up dataset as a Parquet formatted dataset in the AWS S3 location specified by the `BUCKET_DST_PATH` parameter. The `save_stats_metadata` function computes summary statistics of the clean up dataset and saves the statistics as a single CSV file located in a AWS S3 subfolder named `.meta/stats` under the S3 location from the `BUCKET_DST_PATH` parameter.

<!-- #endregion -->

```python id="dygcsspizkU0"
%%writefile -a dctaxi_parquet_vacuum.py


(query_df
 .write
 .parquet(f"{BUCKET_DST_PATH}", mode="overwrite"))

def save_stats_metadata(df, dest, header = 'true', mode = 'overwrite'):
  return (df.describe()
    .coalesce(1)
    .write
    .option("header", header)
    .csv(dest, mode = mode))

save_stats_metadata(query_df, f"{BUCKET_DST_PATH}/.meta/stats")

job.commit()

```

<!-- #region id="IAqxnQVKEHhb" -->
## Run and monitor the PySpark job
* **it should take about 5 minutes for the job to complete**

Once the PySpark job completes successfully, the job execution status should  change from `RUNNING` to `SUCCEEDED`.

<!-- #endregion -->

```bash id="QO2lJ5rPEJ2_"
source utils.sh

PYSPARK_SRC_NAME=dctaxi_parquet_vacuum.py \
PYSPARK_JOB_NAME=dc-taxi-parquet-vacuum-job \
BUCKET_SRC_PATH=s3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/parquet \
BUCKET_DST_PATH=s3://dc-taxi-$BUCKET_ID-$AWS_DEFAULT_REGION/parquet/vacuum \
run_job
```

<!-- #region id="-td5JdiHEPhx" -->
In case of a successful completion, the last cell should have produced an output similar to the following:

```
2021-06-01 23:34:56       1840 dctaxi_parquet_vacuum.py
{
    "JobName": "dc-taxi-parquet-vacuum-job"
}
{
    "Name": "dc-taxi-parquet-vacuum-job"
}
{
    "JobRunId": "jr_59eee7f229f448b39286f1bd19428c9082aaf6bed232342cc05e68f9246d131e"
}
Waiting for the job to finish...............SUCCEEDED
```

Once the PySpark job completes successfully, the job execution status should  change from `RUNNING` to `SUCCEEDED`. You can run the next cell to get the updated job status.

<!-- #endregion -->

```python id="SHBy543wESN9"
!aws glue get-job-runs --job-name dc-taxi-parquet-vacuum-job --output text --query 'JobRuns[0].JobRunState'
```
