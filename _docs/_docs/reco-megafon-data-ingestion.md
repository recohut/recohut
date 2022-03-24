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

<!-- #region id="Bszk6pYpOl-D" -->
### References
- https://github.com/iesnekei/megafon_recommendation_success_predict
<!-- #endregion -->

<!-- #region id="9ppBIsLgOiTG" -->
### Predict result of recommendation for Megafon
<!-- #endregion -->

<!-- #region id="SoFLgIPXO6sX" -->
• Got data sets with 860052 clients of Megafon Co. and theirs transaction or behavior history or client infos. Based on this data I need to create a model for predict a result of Megafon’s recommendation for other group in test data set.
<!-- #endregion -->

```python id="-tR0-bkoOHoO" executionInfo={"status": "ok", "timestamp": 1626722709913, "user_tz": -330, "elapsed": 95201, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!cp /content/drive/MyDrive/TempData/Megafon/* /content
```

```python colab={"base_uri": "https://localhost:8080/"} id="Az4i11_IOx_S" executionInfo={"status": "ok", "timestamp": 1626723176427, "user_tz": -330, "elapsed": 425039, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a97ff65e-a87e-4486-aede-600d875c918e"
!unzip features.csv.zip
```

```python colab={"base_uri": "https://localhost:8080/"} id="K1tss2qTO1D9" executionInfo={"status": "ok", "timestamp": 1626723197121, "user_tz": -330, "elapsed": 571, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c6bfbf7e-9fce-4af0-dd62-121b57be45b3"
!head features.csv
```

```python colab={"base_uri": "https://localhost:8080/"} id="wsO2ELrKRVLE" executionInfo={"status": "ok", "timestamp": 1626723413211, "user_tz": -330, "elapsed": 509, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6ba67ae8-e0e0-4427-b367-7da202be0916"
!tail features.csv
```

```python colab={"base_uri": "https://localhost:8080/"} id="LjZcFuzmQhhg" executionInfo={"status": "ok", "timestamp": 1626723229073, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="03bd4f65-5172-413c-dabb-b81c45f27db3"
!head data_train.csv
```

```python colab={"base_uri": "https://localhost:8080/"} id="MukQQWfSQpmz" executionInfo={"status": "ok", "timestamp": 1626723248914, "user_tz": -330, "elapsed": 412, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e6087c39-8da1-42b4-b507-53232964739b"
!head data_test.csv
```

```python colab={"base_uri": "https://localhost:8080/"} id="4cFPfDd3QuJP" executionInfo={"status": "ok", "timestamp": 1626723314840, "user_tz": -330, "elapsed": 680, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6979ea62-ebb1-4025-bc9d-cf0337228555"
!head /content/megafon_recommendation_success_predict/answers_test.csv
```

```python id="gEi0u_R-Q-b5" executionInfo={"status": "ok", "timestamp": 1626725102308, "user_tz": -330, "elapsed": 496, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import glob
import pandas as pd
```

```python id="HareiaU-ZEPu"
chunksize = 100000
idx = 0
for chunk in pd.read_csv('features.csv', chunksize=chunksize, sep='\t', index_col=0):
  idx = idx+1
  print(idx)
  chunk.to_parquet('features_chunk_{}.parquet.gzip'.format(idx), compression='gzip')
```

```python id="WIItI2DmZClI"
extension = 'parquet.gzip'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
```

```python id="Xqe4konLTmgw" executionInfo={"status": "ok", "timestamp": 1626725142847, "user_tz": -330, "elapsed": 38826, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
combined_df = pd.concat([pd.read_parquet(f) for f in all_filenames[:10]])
combined_df = combined_df.astype('int32')
combined_df.to_parquet("features1.parquet.gzip", compression='gzip')
```

```python id="VsByxR3zWdZG" executionInfo={"status": "ok", "timestamp": 1626725181463, "user_tz": -330, "elapsed": 33691, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
combined_df = pd.concat([pd.read_parquet(f) for f in all_filenames[10:20]])
combined_df = combined_df.astype('int32')
combined_df.to_parquet("features2.parquet.gzip", compression='gzip')
```

```python id="vN5SFs7UWedQ"
combined_df = pd.concat([pd.read_parquet(f) for f in all_filenames[20:30]])
combined_df = combined_df.astype('int32')
combined_df.to_parquet("features3.parquet.gzip", compression='gzip')
```

```python id="atPer8-QX_2a"
combined_df = pd.concat([pd.read_parquet(f) for f in all_filenames[30:40]])
combined_df = combined_df.astype('int32')
combined_df.to_parquet("features4.parquet.gzip", compression='gzip')
```

```python id="S0lXDZEyYDHs" executionInfo={"status": "ok", "timestamp": 1626725281408, "user_tz": -330, "elapsed": 1553, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
combined_df = pd.concat([pd.read_parquet(f) for f in all_filenames[40:]])
combined_df = combined_df.astype('int32')
combined_df.to_parquet("features5.parquet.gzip", compression='gzip')
```

```python id="yfYTkB2qYG9C" executionInfo={"status": "ok", "timestamp": 1626725317289, "user_tz": -330, "elapsed": 443, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!rm -r /content/features_chunk_*.gzip
```

```python colab={"base_uri": "https://localhost:8080/"} id="7LEUKD6aYnZA" executionInfo={"status": "ok", "timestamp": 1626725348361, "user_tz": -330, "elapsed": 410, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="11501d03-b6ff-4670-d185-bfb686ee3e0d"
extension = 'parquet.gzip'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
all_filenames
```

```python id="U1oXRoeMYu-b" executionInfo={"status": "ok", "timestamp": 1626725534610, "user_tz": -330, "elapsed": 155095, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
combined_df = pd.concat([pd.read_parquet(f) for f in all_filenames])
combined_df.to_parquet("features.parquet.gzip", compression='gzip')
```

```python id="CsNEvgiAY2sg" executionInfo={"status": "ok", "timestamp": 1626725936752, "user_tz": -330, "elapsed": 408, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
pd.read_csv('data_test.csv', index_col=0).to_parquet('test.parquet.gzip', compression='gzip')
```

```python id="Y2wonfFhZhS8" executionInfo={"status": "ok", "timestamp": 1626725788797, "user_tz": -330, "elapsed": 414, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!mkdir -p /content/recodata/megafon/v1 && mv /content/features.parquet.gzip /content/recodata/megafon/v1
```
