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

<!-- #region id="xf10Ini7TZxK" -->
# Modeling Multi-Destination Trips with EMDE Model
<!-- #endregion -->

```python id="qTL31U4-q5uG"
!wget https://github.com/Synerise/cleora/releases/download/v1.1.0/cleora-v1.1.0-x86_64-unknown-linux-gnu
!chmod +x cleora-v1.1.0-x86_64-unknown-linux-gnu
```

```python id="7Y7oA76TriEK"
!pip install torch==1.7.1
!pip install pytorch_lightning==1.1.0
!pip install tqdm==4.50.2
!pip install pandas==1.1.5
!pip install numpy==1.19.1
!pip install scikit_learn==0.24.1
```

```python id="OQHRRhmpwfvj"
import os
project_name = "booking-trip-recommender"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', branch)

if not os.path.exists(project_path):
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    !mkdir "{project_path}"
    %cd "{project_path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout -b "{branch}"
    %reload_ext autoreload
    %autoreload 2
else:
    %cd "{project_path}"
```

```python id="YrjmlGO9wfvk"
!git add .
!git commit -m ''
!git push origin "{branch}"
```

```python id="JZx1WUTBsO2o"
!pip install -U -q dvc dvc[gdrive]
!dvc get https://github.com/sparsh-ai/reco-data booking/v1/train.parquet.snappy
!dvc get https://github.com/sparsh-ai/reco-data booking/v1/test.parquet.snappy
!dvc get https://github.com/sparsh-ai/reco-data booking/v1/ground_truth.parquet.snappy
!dvc get https://github.com/sparsh-ai/reco-data booking/v1/submission.parquet.snappy
```

```python id="ybaBeLpawyhj"
!mkdir -p data/bronze data/silver
!mv *.snappy data/bronze
```

<!-- #region id="FLD2ETZVsTII" -->
## Data split

> Split dataset into training and validation set that imitates the hidden test set
<!-- #endregion -->

```python id="nQVIICEzsxgg"
import argparse
import logging
import os
import random
import pandas as pd
import numpy as np
from collections import Counter
```

```python id="w_5kHOArsvr_"
log = logging.getLogger(__name__)


class Args:
    train = 'data/bronze/train.parquet.snappy' # Filename of training dataset provided by challenge organizers
    validation_size = 70662 # Number of trips in validation dataset
    train_output_filename = 'train.csv' # Filename of output training dataset
    valid_output_filename = 'valid.csv' # Filename of output validation dataset
    ground_truth_filename = 'ground_truth.csv' # Filename of ground truth for validation dataset
    working_dir = 'data/silver' # Directory where files will be saved

args = Args()


def preprocess_data(data, utrips_counter):
    """
    Released test set contains additional columns: `row_num`, `total_rows`.
    Those columns are added here.
    """
    data['total_rows'] = data.apply(lambda row: utrips_counter[row['utrip_id']], axis = 1)
    row_num = []
    counter = 1
    for row in data.itertuples():
        row_num.append(counter)
        counter += 1
        if counter > row.total_rows:
            counter = 1
    data['row_num'] = row_num


def get_validation_utrips(data, utrips_less_than_4, validation_size):
    val_utrips = list()
    train_cities_so_far = set()
    utrip_cities = []

    for row in data.itertuples():
        utrip_id = row.utrip_id
        if utrip_id in utrips_less_than_4:
            # test set has at least 4 cities in a trip
            continue

        utrip_cities.append(row.city_id)
        if row.total_rows == row.row_num:
            if all(elem in train_cities_so_far for elem in utrip_cities) and random.random() < 0.5:
                val_utrips.append(row.utrip_id)
            else:
                train_cities_so_far.update(set(utrip_cities))
            utrip_cities = []

        if len(val_utrips) == validation_size:
            break

    log.info(f"Number of validation trips: {len(val_utrips)}")
    return val_utrips


def get_ground_truth(test):
    ground_truth = []
    for i, row in test.iterrows():
        if row['row_num'] == row['total_rows']:
            # this city should be predicted
            ground_truth.append({'utrip_id': row['utrip_id'],
                                'city_id': row['city_id'],
                                'hotel_country': row['hotel_country']})
            test.at[i, 'city_id'] = np.int64(0)
            test.at[i, 'hotel_country'] = ''
    return ground_truth


def main(params):
    os.makedirs(params.working_dir, exist_ok=True)
    data = pd.read_parquet(params.train)
    data['checkin'] = pd.to_datetime(data['checkin'])
    data['checkout'] = pd.to_datetime(data['checkout'])
    utrips_counter = Counter(data['utrip_id'])
    utrips_single = []
    utrips_less_than_4 = []
    for k, v in utrips_counter.items():
        if v == 1:
            utrips_single.append(k)
        if v < 4:
            utrips_less_than_4.append(k)

    log.info(f"Remove {len(utrips_single)} trips with single row")
    data = data.loc[~data['utrip_id'].isin(utrips_single)]
    data = data.sort_values(['utrip_id', 'checkin'])
    data.reset_index(inplace=True, drop=True)
    preprocess_data(data, utrips_counter)
    val_utrips = get_validation_utrips(data, utrips_less_than_4, params.validation_size)

    train = data.loc[~data['utrip_id'].isin(val_utrips)]
    train.reset_index(inplace=True, drop=True)

    test = data.loc[data['utrip_id'].isin(val_utrips)]
    test.reset_index(inplace=True, drop=True)
    log.info(f"Length of train set: {len(train)}")
    log.info(f"Length of test set: {len(test)}")

    ground_truth = get_ground_truth(test)
    pd.DataFrame(ground_truth).to_csv(os.path.join(params.working_dir, params.ground_truth_filename), index=False, sep='\t')
    train.to_csv(os.path.join(params.working_dir, params.train_output_filename), index=False, sep='\t')
    test.to_csv(os.path.join(params.working_dir, params.valid_output_filename), index=False, sep='\t')
```

```python colab={"base_uri": "https://localhost:8080/"} id="Lbjia868svpx" executionInfo={"status": "ok", "timestamp": 1634566364877, "user_tz": -330, "elapsed": 120326, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="44a95441-666f-4cf1-a5d2-8b5c75c04a29"
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(args)
```

<!-- #region id="T8qUoTpqsvnl" -->
This will create three files: `data/valid.csv`, `data/train.csv` and `data/ground_truth.csv`.
<!-- #endregion -->

<!-- #region id="ijjN4-JuxS2c" -->
## City embedding

> Compute city sketches using Cleora. This will create LSH codes for each city and save it into data/codes.
<!-- #endregion -->

```python id="TM3Mhv5SyFLe"
import logging
import os
import subprocess
import numpy as np
import pandas as pd
from typing import List
```

<!-- #region id="xMxZ8H9XxttA" -->
### Cleora
<!-- #endregion -->

```python id="h16-lwjHyHN1"
log = logging.getLogger(__name__)


def prepare_cleora_directed_input(filename: str, data: pd.DataFrame):
    """
    Prepare file such as, for trip X->H->Z->X:
    START X_B
    X_A H_B
    H_A Z_B
    Z_A X_B
    X_A END
    """
    log.info("Preparing input file to Cleora")
    data_grouped_utrip = data.groupby('utrip_id')
    with open(filename, 'w') as f:
        for utrip_id, rows in data_grouped_utrip:
            for i in range(0, len(rows)+1):
                if i == 0:
                    f.write(f"START\t{rows['city_id'].tolist()[0]}_B\n")
                elif i == len(rows):
                    f.write(f"{rows['city_id'].tolist()[-1]}_A\tEND\n")
                else:
                    f.write(f"{rows['city_id'].tolist()[i-1]}_A\t{rows['city_id'].tolist()[i]}_B\n")


def get_cleora_output_directed(filename: str, all_cities: List[str]):
    """
    Read embeddings from file generated by cleora.
    """
    id2embedding = {}
    with open(filename, 'r') as f:
        next(f) # skip cleora header
        for index, line in enumerate(f):
            line_splitted = line.split(sep=' ')
            id = str(line_splitted[0])
            embedding = np.array([float(i) for i in line_splitted[2:]])
            id2embedding[id] = embedding

    ids = []
    embeddings = []
    for city in all_cities:
        ids.append(city)
        embeddings.append(np.concatenate((id2embedding[f'{city}_A'], id2embedding[f'{city}_B'])))

    return ids, np.stack(embeddings)


def train_cleora(dim: int, iter_: int, columns: str, input_filename: str, working_dir: str):
    """
    Training Cleora. See more details: https://github.com/Synerise/cleora/
    """
    command = ['/content/cleora-v1.1.0-x86_64-unknown-linux-gnu',
               '--columns', columns,
               '--dimension', str(dim),
               '-n', str(iter_),
               '--input', input_filename,
               '--output-dir', working_dir]
    subprocess.run(command, check=True)


def run_cleora_directed(working_dir: str, input_file: str, dim: int, iter_: int ,all_cities: List[str]):
    train_cleora(dim, iter_, 'nodeStart nodeEnd', input_file, working_dir)
    return get_cleora_output_directed(os.path.join(working_dir, 'emb__nodeStart__nodeEnd.out'), all_cities)
```
