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

```python id="r9M_yX9JsdiP"
!pip install pyoptimus
```

```python id="16soPIGwutca"
!pip install py-dateinfer url_parser
!pip install -U pandas
!pip install python-libmagic
```

```python id="DWA1du7xsf6-"
!mkdir /content/x
!git clone https://github.com/PacktPublishing/Data-Processing-with-Optimus.git /content/x
%cd /content
```

```python executionInfo={"elapsed": 729, "status": "ok", "timestamp": 1630841402844, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="fOgcTSS1t_jw"
!mv /content/x/Chapter01/foo.txt .
!mv /content/x/Chapter01/path/to/file.csv .
```

```python colab={"base_uri": "https://localhost:8080/", "height": 173} executionInfo={"elapsed": 484, "status": "ok", "timestamp": 1630841880824, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="loWWlyhHwAc9" outputId="ddec3204-2b43-441c-faba-316acb8f5198"
import pandas as pd

pd.read_csv('file.csv')
```

```python executionInfo={"elapsed": 613, "status": "ok", "timestamp": 1630841743804, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="3qc5tJTwuGnQ"
from optimus import Optimus
```

```python executionInfo={"elapsed": 1764, "status": "ok", "timestamp": 1630841762607, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="CnOx75kZuyb-"
op = Optimus('pandas')
```

<!-- #region id="4wgHRGMl2sEx" -->
### Basics
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 232} executionInfo={"elapsed": 495, "status": "ok", "timestamp": 1630841834405, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="3zTFi-6-uq0l" outputId="6939fe24-65ca-4fa6-e168-f9e444999360"
# df = op.load.file('file.csv')
df = op.load.csv('file.csv')
df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 232} executionInfo={"elapsed": 500, "status": "ok", "timestamp": 1630841962329, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="lbb5syYjwXQP" outputId="27294abe-e99a-4b67-9471-c842f5798ea5"
df = df.cols.rename("function", "job")
df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 232} executionInfo={"elapsed": 527, "status": "ok", "timestamp": 1630841989413, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="TowuWmVWwcgT" outputId="b714eb29-67de-4f2a-e13e-e51dd10c1234"
df = df.cols.upper("name").cols.lower("job")
df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 232} executionInfo={"elapsed": 692, "status": "ok", "timestamp": 1630842013063, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="veisGK30wjg3" outputId="a6389d01-6541-479e-87ee-31469582f4fa"
df.cols.drop("name") 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 214} executionInfo={"elapsed": 460, "status": "ok", "timestamp": 1630842026921, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="DhmGCtbqwk6Y" outputId="d166592e-2eb0-48f7-e5b4-a3a331732b91"
df.rows.drop(df["name"]=="MEGATRON") 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 232} executionInfo={"elapsed": 829, "status": "ok", "timestamp": 1630842039275, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="6fzLnqGwwp9s" outputId="4824837d-98b4-416b-a4f4-5c36e769ffee"
df.display()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 232} executionInfo={"elapsed": 764, "status": "ok", "timestamp": 1630842128885, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="lfsZIN20w4nw" outputId="d4f3de7c-808c-4697-a79f-d474472875ee"
df.cols.capitalize("name", output_cols="cap_name") 
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 514, "status": "ok", "timestamp": 1630842161857, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ZswWqKcAxH6R" outputId="c1f42c9d-b925-4b65-911a-a9986bdad728"
df.profile(bins=10) 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 250} executionInfo={"elapsed": 626, "status": "ok", "timestamp": 1630842077372, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="lyW_kbvBwy0c" outputId="da4787f9-92c0-4555-edc8-d8d965702ed5"
dfn = op.create.dataframe({"A":["1",2,"4","!",None]})
dfn
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 716, "status": "ok", "timestamp": 1630842093292, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="bIzShbFLw2ZJ" outputId="f55a2991-63b9-408e-b573-a569d680cc8b"
dfn.cols.min("A"), dfn.cols.max("A")
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 723, "status": "ok", "timestamp": 1630842190933, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="WWWiSF1_xPFu" outputId="91286f42-e9f1-4970-8c57-111b2a5ca751"
df = op.create.dataframe({
    "A":["1",2,"4","!",None],
    "B":["Optimus","Bumblebee", "Eject", None, None]
})  

df.profile(bins=10) 
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 617, "status": "ok", "timestamp": 1630842204042, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="kt3_qU7hxSSG" outputId="a696862f-f176-43a7-93fa-95b30b5a6a1b"
df.columns_sample("*") 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 250} executionInfo={"elapsed": 623, "status": "ok", "timestamp": 1630842213676, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="SbcbmuJUxUcx" outputId="488e4cfa-5de6-4b38-f949-3d299a478933"
df.execute()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1630842232814, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="oETBou6BxZTh" outputId="ca079dc4-1f02-4018-cbbc-ec60220ed00a"
df = op.load.csv("foo.txt", sep=",") 
type(df.data)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 214} executionInfo={"elapsed": 837, "status": "ok", "timestamp": 1630842265177, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="CwlQA7lqxg2Y" outputId="22b55ae1-a93e-45fd-a384-c4c7ae63a921"
df
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 723, "status": "ok", "timestamp": 1630842275108, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="X1xwlz88vtRD" outputId="c0b752f5-ad06-4dc2-e8dc-7eb2f0770c15"
df = df.cols.upper("*") 
df.meta["transformations"] 
```

<!-- #region id="vXcLT_qr1tx1" -->
### Read some rows from parquet
<!-- #endregion -->

<!-- #region id="SevVeUhh1wxE" -->
Pandas still doesn't support reading few of the rows, instead of full. So we can use Optimus in this case.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 173} executionInfo={"elapsed": 498, "status": "ok", "timestamp": 1630843261876, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="ox6F9pW3v0eX" outputId="9ef7f2e1-2712-4cc8-fea8-6e0eeae4252c"
import pandas as pd

df = pd.read_csv('file.csv')
df.to_parquet('file.parquet.snappy', compression='snappy')
df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 323} executionInfo={"elapsed": 639, "status": "error", "timestamp": 1630843314759, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="REb5nrE61WQd" outputId="7045e037-fb94-4083-f383-c6d6811fe3c7"
pd.read_parquet('file.parquet.snappy', nrows=2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 213} executionInfo={"elapsed": 467, "status": "ok", "timestamp": 1630843412818, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="nNi5uJfD0-iT" outputId="894bc96b-c905-4d20-b5ce-a598631c777b"
df = op.load.parquet('file.parquet.snappy', n_rows=2)
df
```

<!-- #region id="FfH3DzyA15y1" -->
### Optimize memory
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 621, "status": "ok", "timestamp": 1630843636760, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="wiGJpECL2pAg" outputId="a25187ae-9080-4887-93fd-d03b0d33ff5a"
df = op.create.dataframe({ 
    "a": [1000,2000,3000,4000,5000]*10000, 
    "b": [1,2,3,4,5]*10000 
}) 
df.size() 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 340} executionInfo={"elapsed": 18, "status": "error", "timestamp": 1630843638883, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}, "user_tz": -330} id="DrNluEdz2wWk" outputId="4d097caf-645b-4075-b760-59dd94b91155"
df = df.optimize()
df.size() 
```

```python id="ujFvdpbt2xC3"

```
