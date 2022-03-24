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

<!-- #region id="DI5sTNMd54MI" -->
### Setup
<!-- #endregion -->

```python id="l6AvHIS_yvGW" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1640516373916, "user_tz": -330, "elapsed": 43565, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="bb61d554-8f23-4d9d-82be-b3c37b5d8915"
import os
project_name = "recobase"; branch = "US565244"; account = "recohut"
project_path = os.path.join('/content', branch)

if not os.path.exists(project_path):
    !pip install -U -q dvc dvc[gdrive]
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    !mkdir "{project_path}"
    %cd "{project_path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout -b "{branch}"
else:
    %cd "{project_path}"
```

```python id="i5OiQm3vyvGd"
!git add .
!git commit -m 'commit'
!git push origin "{branch}"
```

<!-- #region id="xnnDLTFL6CuY" -->
### Imports
<!-- #endregion -->

```python id="_DoCdhmRZrUR" executionInfo={"status": "ok", "timestamp": 1640516477117, "user_tz": -330, "elapsed": 1025, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f28405a2-c683-4b6e-f0aa-08ce0852eec5" colab={"base_uri": "https://localhost:8080/"}
!git checkout main
```

<!-- #region id="guNVJXI958DV" -->
### Data Bronze Layer
<!-- #endregion -->

```python id="12pV0SCxzd-r" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1640516436173, "user_tz": -330, "elapsed": 10268, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="11fb27b2-8c54-4ca4-9703-63c32c5c5549"
!mkdir -p data/bronze
!dvc import -o data/bronze/confirmation.parquet.gzip https://github.com/recohut/recodata trivago/confirmation.parquet.gzip
!dvc import -o data/bronze/validation.parquet.gzip https://github.com/recohut/recodata trivago/validation.parquet.gzip
!dvc import -o data/bronze/item_metadata.parquet.gzip https://github.com/recohut/recodata trivago/item_metadata.parquet.gzip
!dvc import -o data/bronze/test.parquet.gzip https://github.com/recohut/recodata trivago/test.parquet.gzip
!dvc import -o data/bronze/train.parquet.gzip https://github.com/recohut/recodata trivago/train.parquet.gzip
```

<!-- #region id="bxNfTZke1xEv" -->
### Preprocessing
<!-- #endregion -->

```python id="7Ac_Zexj8CTs"
def merge_duplicates(df):
    """
    Deletes from df consecutive actions of same type performed on the same reference within the same session.
    It keeps the first occurrence of those consecutive actions and for those it saves
    how many consecutive actions are occurred in column 'frequence'.
    For the non-consecutive actions, frequence is set to 1.
    :param df: DataFrame to preprocess
    :return: df: preprocessed DataFrame df with 'frequence' column
    """
    tqdm.pandas()

    duplicates_indices = []
    # points to the next valid row
    indices = df.index.values
    totlen = len(df)
    i = 0
    j = 0
    next_index = indices[j]

    for index in tqdm(indices):
        if i >= j:
            curr_actiontype = df.at[index,'action_type']
            count = 1
            j += 1
            # check next interactions
            while j < totlen:
                next_index = indices[j]

                # iterate while the interactions are duplicated
                if curr_actiontype != 'clickout item' and \
                    df.at[index, 'user_id'] == df.at[next_index, 'user_id'] and \
                    df.at[index, 'session_id'] == df.at[next_index, 'session_id'] and \
                    df.at[index, 'reference'] == df.at[next_index, 'reference'] and \
                    curr_actiontype == df.at[next_index, 'action_type']:

                    # current interaction can be merged
                    j += 1
                    duplicates_indices.append(next_index)
                    count += 1
                else:
                    break

            # different interaction reached
            df.at[index, 'frequence'] = count
        i += 1

    # drop the duplicated indices
    return df.drop(duplicates_indices)
```

```python id="Ar1dVzqz8WEz"
def get_small_dataset(df, maximum_rows=1000000):
    """
    Return a dataframe from the original dataset containing a maximum number of rows. The actual total rows
    extracted may vary in order to avoid breaking the last session.
    :param df: dataframe
    :param maximum_rows:
    :return: dataframe
    """
    if len(df) < maximum_rows:
      return df
    # get the last row
    last_row = df.iloc[[maximum_rows]]
    last_session_id = last_row.session_id.values[0]

    # OPTIMIZATION: last_user_id = last_row.user_id.values[0]

    # slice the dataframe from the target row on
    temp_df = df.iloc[maximum_rows:]
    # get the number of remaining interactions of the last session
    # OPTIMIZATION: remaining_rows = temp_df[(temp_df.session_id == last_session_id) & (temp_df.user_id == last_user_id)].shape[0]
    remaining_rows = temp_df[temp_df.session_id == last_session_id].shape[0]
    # slice from the first row to the final index
    return df.iloc[0:maximum_rows+remaining_rows]
```

```python id="revyb3eR8xCR"
def split(df, save_path, perc_train=80):
    """
    Split a timestamp-ordered dataset into train and test, saving them as train.csv and test.csv in the
    specififed path. Also save the target indices file containing indices of missing clickout interactions.
    :param df: dataframe to split in train and test
    :param save_path: path where to save
    :param perc_train: percentage of the df to keep in the TRAIN split
    :return:
    """
    print('Splitting...', end=' ', flush=True)
    # train-test split
    print('sorting')
    sorted_session_ids = df.groupby('session_id').first().sort_values('timestamp').reset_index()['session_id']
    print('slicing')
    slice_sorted_session_ids = sorted_session_ids.head(int(len(sorted_session_ids) * (perc_train / 100)))
    df_train = df.loc[df['session_id'].isin(slice_sorted_session_ids)]
    df_test = df.loc[~df['session_id'].isin(slice_sorted_session_ids)]

    # remove clickout from test and save an handle
    # just those who are for real into the list of impressions
    groups = df_test[df_test['action_type'] == 'clickout item'].groupby('user_id', as_index=False)
    remove_reference_tuples = groups.apply(lambda x: x.sort_values(by=['timestamp'], ascending=True).tail(1))

    for index, row in tqdm(remove_reference_tuples.iterrows()):
        if int(row['reference']) not in list(map(int, row['impressions'].split('|'))):
            remove_reference_tuples.drop(index, inplace=True)

    for e in tqdm(remove_reference_tuples.index.tolist()):
        df_test.at[e[1], 'reference'] = np.nan

    # save them all
    df_train.to_csv(os.path.join(save_path, "train.csv"))
    df_test.to_csv(os.path.join(save_path, "test.csv"))
    np.save(os.path.join(save_path, 'target_indices'), get_target_indices(df_test))
    np.save(os.path.join(save_path, 'train_indices'), df_train.index)
    np.save(os.path.join(save_path, 'test_indices'), df_test.index)
    print('Done!')
```

```python id="yYQFEXD39BqL"
def append_missing_accomodations(mode):
    found_ids = []

    joined_df = data.train_df(mode).append(data.test_df(mode))

    # add references if valid
    refs = joined_df.reference
    refs = refs[refs.notnull()].values
    for r in tqdm(refs):
        try:
            v = int(r)
            found_ids.append(v)
        except ValueError:
            continue

    # add impressions
    imprs = joined_df.impressions
    imprs = imprs[imprs.notnull()].values
    for i in tqdm(imprs):
        found_ids.extend(list(map(int, i.split('|'))))

    found_ids = set(found_ids)
    acs = data.accomodations_ids()
    accomod_known = set(map(int, acs))
    missing = found_ids.difference(accomod_known)
    missing_count = len(missing)
    print('Found {} missing accomodations'.format(missing_count))

    del joined_df

    # add those at the end of the dataframe
    if missing_count > 0:
        new_acc_df = pd.DataFrame({ 'item_id': list(missing) }, columns=['item_id', 'properties'] )

        new_acs = data.accomodations_df().append(new_acc_df, ignore_index=True)
        new_acs.to_csv(data.ITEMS_PATH, index=False)
        print('{} successfully updated'.format(data.ITEMS_PATH))
```

```python id="PS0q_dI19FCz"
def preprocess_accomodations_df(preprocessing_fns):
    """
    Preprocess and save the item metadata csv using the supplied functions. Each function will be applied
    sequentially to each row of the dataframe. The function will receive as param each dataframe row and
    should return a tuple (that will be treated as the new row columns).
    """
    assert isinstance(preprocessing_fns, list)

    print('Processing accomodations dataframe...')
    # load and preprocess the original item_metadata.csv
    accomodations_df = data.accomodations_original_df()

    tqdm.pandas()
    for preprfnc in preprocessing_fns:
        accomodations_df = accomodations_df.progress_apply(preprfnc, axis=1, result_type='broadcast')

    print(f'Saving preprocessed accomodations dataframe to {data.ITEMS_PATH}...', end=' ', flush=True)
    accomodations_df.to_csv(data.ITEMS_PATH, index=False)
    print('Done!')
```

```python id="FmYplmU49ZRe"
class Dataset:
    def __init__(self):
        pass

    def _create_csvs():
        print('creating CSV...')

        # create no_cluster/full
        path = 'dataset/preprocessed/no_cluster'
        full = data.full_df()
        train_len = data.read_config()[data.TRAIN_LEN_KEY]

        train = full.iloc[0:train_len]
        test = full.iloc[train_len:len(full)]
        target_indices = get_target_indices(test)

        check_folder('dataset/preprocessed/no_cluster/full')
        train.to_csv(os.path.join(path, 'full/train.csv'))
        test.to_csv(os.path.join(path, 'full/test.csv'))
        np.save(os.path.join(path, 'full/train_indices'), train.index)
        np.save(os.path.join(path, 'full/test_indices'), test.index)
        np.save(os.path.join(path, 'full/target_indices'), target_indices)

        no_of_rows_in_small = int(input('How many rows do you want in small.csv? '))
        train_small = get_small_dataset(train, maximum_rows=no_of_rows_in_small)
        check_folder('dataset/preprocessed/no_cluster/small')
        split(train_small, os.path.join(path, 'small'))

        check_folder('dataset/preprocessed/no_cluster/local')
        split(train, os.path.join(path, 'local'))

        # create item_metadata in preprocess folder
        original_item_metadata = data.accomodations_original_df()
        original_item_metadata.to_csv(data.ITEMS_PATH)

        # append missing accomodations to item metadata
        append_missing_accomodations('full')

def _preprocess_item_metadata():
    # interactively enable preprocessing function
    labels = ['Remove \'From n stars\' attributes']
    pre_processing_f = [ remove_from_stars_features ]
    menu_title = 'Choose the preprocessing function(s) to apply to the accomodations.\nPress numbers to enable/disable the options, press X to confirm.'
    activated_prefns = menu.options(pre_processing_f, labels, title=menu_title, custom_exit_label='Confirm')

    # preprocess accomodations dataframe
    preprocess_accomodations_df(activated_prefns)

def _create_urm_session_aware():
    """
    NOTE: CHANGE THE PARAMETERS OF THE SEQUENCE AWARE URM HERE !!!!
    """
    create_urm.urm_session_aware(mode, cluster, time_weight='lin')
def _create_urm_clickout():
    """
    NOTE: CHANGE THE PARAMETERS OF THE CLICKOUT_ONLY URM HERE !!!!
    """
    create_urm.urm(mode, cluster, clickout_score=5, impressions_score=1)

def _merge_sessions():
    print("Merging similar sessions (same user_id and city)")
    print("Loading full_df")
    full_df = data.full_df()
    print("Sorting, grouping, and other awesome things")
    grouped = full_df.sort_values(["user_id", "timestamp"], ascending=[True, True]).groupby(["user_id", "city"])
    new_col = np.array(["" for _ in range(len(full_df))], dtype=object)
    print("Now I'm really merging...")
    for name, g in tqdm(grouped):
        s_id = g.iloc[0]["session_id"]
        new_col[g.index.values] = s_id
    print("Writing on the df")
    full_df["unified_session_id"] = pd.Series(new_col)
    print("Saving new df to file")
    with open(data.FULL_PATH, 'w', encoding='utf-8') as f:
        full_df.to_csv(f)
    data.refresh_full_df()

print("Hello buddy... Copenaghen is waiting...")
print()

# create full_df.csv
# pick your custom preprocessing function

# original
# funct = no_custom_preprocess_function

# unroll
funct = unroll_custom_preprocess_function

check_folder(data.FULL_PATH)
if os.path.isfile(data.FULL_PATH):
    menu.yesno_choice('An old full dataframe has been found. Do you want to delete it and create again?', \
        callback_yes=(lambda: create_full_df(funct)))
else:
    print('The full dataframe (index master) is missing. Creating it...', end=' ', flush=True)
    create_full_df(funct)
    print('Done!')


# create CSV files
menu.yesno_choice(title='Do you want to merge similar sessions (adding unified_session_id)?', callback_yes=_merge_sessions)

# create CSV files
menu.yesno_choice(title='Do you want to create the CSV files?', callback_yes=_create_csvs)

# preprocess item_metadata
menu.yesno_choice(title='Do you want to preprocess the item metadata?', callback_yes=_preprocess_item_metadata)

# create ICM
menu.yesno_choice(title='Do you want to create the ICM matrix files?', callback_yes=create_icm.create_ICM)

# create URM
lbls = ['Create URM from LOCAL dataset', 'Create URM from FULL dataset', 'Create URM from SMALL dataset', 'Skip URM creation' ]
callbacks = [lambda: 'local', lambda:'full', lambda: 'small', lambda: 0]
res = menu.single_choice(title='What do you want to do?', labels=lbls, callbacks=callbacks)

if res is None:
    exit(0)

if res != 0:
    # initialize the train and test dataframes
    mode = res

    # get the cluster
    print('for which cluster do you want to create the URM ???')
    cluster = input()
    callbacks = [_create_urm_session_aware, _create_urm_clickout]
    menu.single_choice(title='Which URM do you want create buddy?', labels=['Sequence-aware URM', 'Clickout URM'], callbacks=callbacks)
```
