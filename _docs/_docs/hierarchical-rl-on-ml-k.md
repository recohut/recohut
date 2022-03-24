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

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1178, "status": "ok", "timestamp": 1634823542909, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="gkk6sWhyxOAw" outputId="586451a6-07b5-4ce5-f5b2-bc8f2b25fece"
!git clone https://github.com/zoulixin93/FMCTS.git
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6673, "status": "ok", "timestamp": 1634823549570, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="S_E-icWdxTHO" outputId="b0fec979-3f7a-4317-cae4-66121eda8067"
!apt-get install tree
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 41, "status": "ok", "timestamp": 1634823549572, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="qOuKnYRixUVp" outputId="2f771309-709a-4221-aaf5-e4aebc826bff"
!tree --du -h ./FMCTS
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 748, "status": "ok", "timestamp": 1634823550290, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="Vw6Z7xYhyM-N" outputId="4feaf452-ac2f-4c05-d40b-89498cc0601a"
%tensorflow_version 1.x
```

```python id="z-F_rzTpyA6c"
!pip install ipdb
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1634823557360, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="WNICg6wQxX-2" outputId="451b2886-c968-4110-898e-093bf42c78ff"
%cd FMCTS
```

```python executionInfo={"elapsed": 851, "status": "ok", "timestamp": 1634823558192, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="a72hj2ENyXl-"
!mkdir -p saved_model
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 459, "status": "ok", "timestamp": 1634823638371, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}, "user_tz": -330} id="4emCRnb2xgcl" outputId="22c70e59-38c2-4257-eb43-b1ea6209b07f"
%%writefile run.sh

m100k(){
    python ./train_file.py -model $1 -log_path ./log/ -data_name m100k \
    -root_path ./data/ml-100k/ -rating_path rat.dat -cat_path cat.dat -item_num 1683 \
    -user_num 944 -cat_n1_num 20 -cat_n2_num 20 -c_puct $2 -n_playout $3 -temperature $4 \
    -update_frequency $5 -batch_size $6 -epoch $7 -memory_capacity $8 -learning_rate $9 \
    -optimizer_name ${10} -evaluate_num ${11} -latent_factor ${12} -delete_previous ${13} \
    -job_ports ${14} -task ${15} -evaluate_num 500
}

name="m100k_hmcts"
(m100k $name 20.0 50 20.0 1 60 10000 1000 0.005 sgd 100 20 False [40001] train) &
(m100k $name 20.0 50 20.0 1 60 10000 1000 0.005 sgd 100 20 False [40001] evaluate) &
```

```python colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="dI1lPuwXxoa5"
!sh run.sh
```

```python id="_OFOmfGN80eJ"

```
