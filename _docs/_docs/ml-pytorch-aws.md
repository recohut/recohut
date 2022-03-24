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
# Serverless machine learning at scale with pytorch and AWS
<!-- #endregion -->

<!-- #region id="8HPLiT_xlliJ" -->
> Note: https://learning.oreilly.com/library/view/mlops-engineering-at/9781617297762/OEBPS/Text/07.htm#sigil_toc_id_83
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

<!-- #region id="87N5zM7SA8bo" -->
## Using ObjectStorageDataset


`ObjectStorageDataset` provides support for tensor-based, out-of-memory datasets for the iterable-style `torch.utils.data.DataLoader` interface. The `ObjectStorageDataset` is not available by default when you install PyTorch, so you need to install it separately in your Python environment using:
<!-- #endregion -->

```python id="SdeYRTfyBCDE"
!pip install kaen[osds]
```

<!-- #region id="zYEobnJEBGTa" -->
and once installed, import the class in your runtime and create an instance using:
<!-- #endregion -->

```python id="ByXQQ8hqEkFR"
from kaen.torch import ObjectStorageDataset as osds
BUCKET_ID = os.environ['BUCKET_ID']
AWS_DEFAULT_REGION = os.environ['AWS_DEFAULT_REGION']

BATCH_SIZE = 2 ** 20 # 1_048_576 

train_ds = osds(f"s3://dc-taxi-{BUCKET_ID}-{AWS_DEFAULT_REGION}/csv/dev/part*.csv", 
                storage_options = {'anon': False},
                batch_size = BATCH_SIZE)
train_ds
```

<!-- #region id="X5V_4JTGjqwP" -->
The `shards_glob` parameter of the `ObjectStorageDataset` points to the metadata file about the CSV part files that match the `/csv/dev/part*.csv` glob. You can preview the metadata as a Pandas data frame.
<!-- #endregion -->

```python id="sT8RAt1piEAS"
import pandas as pd
shards_df = pd.read_csv(f"s3://dc-taxi-{BUCKET_ID}-{AWS_DEFAULT_REGION}/csv/dev/.meta/shards/*.csv")
print(shards_df[:5])
```

```python id="valsQoixFijy"
from torch.utils.data import DataLoader
batch = next(iter(DataLoader(train_ds)))

batch
```

```python id="D4B29s74Fqpg"
batch.dtype
```

```python id="PRREdltuslJp"
batch.shape
```

```python id="U49NbWPyFsO5"
import os
import time
import torch as pt

from torch.utils.data import TensorDataset, DataLoader
from kaen.torch import ObjectStorageDataset as osds

pt.manual_seed(0);
pt.set_default_dtype(pt.float64)

BUCKET_ID = os.environ['BUCKET_ID']
AWS_DEFAULT_REGION = os.environ['AWS_DEFAULT_REGION']

BATCH_SIZE = 2 ** 20 #evaluates to 1_048_576
train_ds = osds(f"s3://dc-taxi-{BUCKET_ID}-{AWS_DEFAULT_REGION}/csv/dev/part*.csv", 
                storage_options = {'anon': False},
                batch_size = BATCH_SIZE)

train_dl = DataLoader(train_ds, batch_size=None)

FEATURE_COUNT = 8

w = pt.nn.init.kaiming_uniform_(pt.empty(FEATURE_COUNT, 1, requires_grad=True))
b = pt.nn.init.kaiming_uniform_(pt.empty(1, 1, requires_grad = True))

def batchToXy(batch):
  batch = batch.squeeze_()  
  return batch[:, 1:], batch[:, 0]

def forward(X):
  y_est = X @ w + b
  return y_est.squeeze_()

LEARNING_RATE = 0.03
optimizer = pt.optim.SGD([w, b], lr = LEARNING_RATE)

GRADIENT_NORM = .5

ITERATION_COUNT = 50

for iter_idx, batch in zip(range(ITERATION_COUNT), train_dl):
  start_ts = time.perf_counter()

  X, y = batchToXy(batch)

  y_est = forward(X)
  mse = pt.nn.functional.mse_loss(y_est, y)
  mse.backward()

  pt.nn.utils.clip_grad_norm_([w, b], GRADIENT_NORM) if GRADIENT_NORM else None

  optimizer.step()
  optimizer.zero_grad()

  sec_iter = time.perf_counter() - start_ts

  print(f"Iteration: {iter_idx:03d}, Seconds/Iteration: {sec_iter:.3f} MSE: {mse.data.item():.2f}")
```

<!-- #region id="yy7PSJjXLFcF" -->
## Faster PyTorch tensor operations with Graphical Processing Units
<!-- #endregion -->

<!-- #region id="RjtvOdWLLMaI" -->
To find out exactly the number of CPU cores available to PyTorch
<!-- #endregion -->

```python id="4ObUE9pTLGq-"
import os
print(os.cpu_count())
```

<!-- #region id="Or7L3elLLTnf" -->
In PyTorch it is customary to initialize the device variable as follows before using the GPU
<!-- #endregion -->

```python id="V0yL6-X3LKP6"
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(device)
```

<!-- #region id="hFBAs5j1Ld7w" -->
To find out the number of ALUs you have available, you need to first use the `get_device_capability` method to find out your CUDA compute capability profile:




<!-- #endregion -->

```python id="a2_NcF0mLeO8"
import torch as pt
print([pt.cuda.get_device_properties(i) for i in range(pt.cuda.device_count())])
```

```python id="XB_qLOlLLtVw"
!nvidia-smi
```

<!-- #region id="kdVVXb_9MWDC" -->
PyTorch defaults to using the CPU-based tensors
<!-- #endregion -->

```python id="pMHqdgo0MFUU"
pt.set_default_dtype(pt.float64)

tensor = pt.empty(1)
print(tensor.dtype, tensor.device)
```

<!-- #region id="JJgih4OiMcSD" -->
To specify the CUDA-based implementation as default you can use the `set_default_tensor_type` method
<!-- #endregion -->

```python id="PicdqWqfMOch"
pt.set_default_tensor_type(pt.cuda.FloatTensor)
pt.set_default_dtype(pt.float64)

tensor = pt.empty(1)
print(tensor.dtype, tensor.device)
```

<!-- #region id="VAtZ3fcQM5u5" -->
A better practice when using a GPU for tensor operations, is to create tensors directly on a desired device. Assuming that you initialize the `device` variable, you can create a tensor on a specific device by setting the `device` named parameter as shown here:
<!-- #endregion -->

```python id="pRaP2VcjM0bs"
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

tensor = pt.empty(1, dtype=int, device=device)
print(tensor.dtype, tensor.device)
```

```python id="eOFOgr1xNGgh"
import timeit
MAX_SIZE = 28

def benchmark_cpu_gpu(n, sizes):
  for device in ["cpu", "cuda"]:
    for size in sizes:
      a = pt.randn(size).to(device)
      b = pt.randn(size).to(device)
      yield timeit.timeit(lambda: a + b, number = n)

sizes = [2 ** i for i in range(MAX_SIZE)]
measurements = list(benchmark_cpu_gpu(1, sizes))
cpu = measurements[:MAX_SIZE]
gpu = measurements[MAX_SIZE:]
ratios = [cpu[i] / gpu[i] for i in range(len(cpu))]
ratios
```

```python id="qZo5jffVNSq_"
import matplotlib.pyplot as plt
%matplotlib inline
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.figure(figsize=(12, 6))
plt.plot([2 ** i for i in range(MAX_SIZE)], ratios)
plt.xscale("log", basex=2)
```

```python id="6SBtUE8eN-sN"
plt.figure(figsize=(12,6))

plt.plot(sizes[:16], ratios[:16])
plt.xscale("log", basex=2);

plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
```

<!-- #region id="INnKfnDHLAzV" -->
## Scaling up to use GPU cores
<!-- #endregion -->

```python id="e8iVoMrtLAWX"
import os
import torch as pt
from torch.utils.data import DataLoader
from kaen.torch import ObjectStorageDataset as osds

pt.manual_seed(0);
pt.set_default_dtype(pt.float64)

BATCH_SIZE = 1_048_576 # = 2 ** 20

train_ds = osds(f"s3://dc-taxi-{os.environ['BUCKET_ID']}-{os.environ['AWS_DEFAULT_REGION']}/csv/dev/part*.csv", 
    storage_options = {'anon': False},
    batch_size = BATCH_SIZE)
  
train_dl = DataLoader(train_ds, 
                      pin_memory = True) 

FEATURE_COUNT = 8

w = pt.nn.init.kaiming_uniform_(pt.empty(FEATURE_COUNT, 1, requires_grad=True, device=device))
b = pt.nn.init.kaiming_uniform_(pt.empty(1, 1, requires_grad = True, device=device))

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

def batchToXy(batch):
  batch = batch.squeeze_().to(device)
  return batch[:, 1:], batch[:, 0]

def forward(X):
  y_pred = X @ w + b
  return y_pred.squeeze_()

def loss(y_est, y):
  mse_loss = pt.mean((y_est - y) ** 2)
  return mse_loss

LEARNING_RATE = 0.03
optimizer = pt.optim.SGD([w, b], lr = LEARNING_RATE)

GRADIENT_NORM = 0.5

ITERATION_COUNT = 50

for iter_idx, batch in zip(range(ITERATION_COUNT), train_dl):
  start_ts = time.perf_counter()

  X, y = batchToXy(batch)

  y_est = forward(X)
  mse = loss(y_est, y)
  mse.backward()

  pt.nn.utils.clip_grad_norm_([w, b], GRADIENT_NORM) if GRADIENT_NORM else None

  optimizer.step()
  optimizer.zero_grad()

  sec_iter = time.perf_counter() - start_ts

  print(f"Iteration: {iter_idx:03d}, Seconds/Iteration: {sec_iter:.3f} MSE: {mse.data.item():.2f}")
```

<!-- #region id="mYt-3VQUk5jv" -->
Copyright 2021 CounterFactual.AI LLC. All Rights Reserved.

Licensed under the GNU General Public License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. 

You may obtain a copy of the License at

https://github.com/osipov/smlbook/blob/master/LICENSE

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
<!-- #endregion -->
