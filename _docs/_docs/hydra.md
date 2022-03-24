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

<!-- #region id="nBRHan-DeXBV" -->
# Hydra Config
<!-- #endregion -->

<!-- #region id="jHVXYJ7MQYwd" -->
## Setup
<!-- #endregion -->

```python id="BPzpURI2Kp9l"
!pip install hydra-core --upgrade
```

```python colab={"base_uri": "https://localhost:8080/"} id="71jYiqAmM2SI" executionInfo={"status": "ok", "timestamp": 1632040631374, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="ac7d2a11-eb8c-4256-d368-62325f7d21c3"
!mkdir -p myproject/src/conf
%cd myproject
```

<!-- #region id="4zugqsPaQasm" -->
### Simple example
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="J2mmqaHRMyE6" executionInfo={"status": "ok", "timestamp": 1632040670527, "user_tz": -330, "elapsed": 582, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6e6b933e-4666-4453-c712-c18a49a58c0d"
%%writefile src/conf/config.yaml
db:
  driver: mysql
  user: omry
  pass: secret
```

```python colab={"base_uri": "https://localhost:8080/"} id="mh2ek6xINFAh" executionInfo={"status": "ok", "timestamp": 1632040711122, "user_tz": -330, "elapsed": 426, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="dd206dc7-14fc-427c-bf44-9131d7581fa4"
%%writefile src/myapp.py
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Wynjkp-3NO-O" executionInfo={"status": "ok", "timestamp": 1632040731123, "user_tz": -330, "elapsed": 557, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a6b12e5b-29b1-494a-eda6-7f91a830dbbe"
!python src/myapp.py
```

<!-- #region id="KMoYFqCeNTo2" -->
## Overriding with CLI
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="bd4xmNwLQf6Q" executionInfo={"status": "ok", "timestamp": 1632041586777, "user_tz": -330, "elapsed": 1210, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="766d3b88-0f63-4726-f95e-e16eaf177070"
!python src/myapp.py db.user=root
```

<!-- #region id="1aRv4kg3QkkE" -->
## Composition
<!-- #endregion -->

```python id="z25SD81jQwvv"
# ├── conf
# │   ├── config.yaml
# │   ├── db
# │   │   ├── mysql.yaml
# │   │   └── postgresql.yaml
# │   └── __init__.py
# └── my_app.py
```

```python id="Wn-emgSiQvPJ"
!mkdir -p src/conf/db
```

```python colab={"base_uri": "https://localhost:8080/"} id="ELp5mvIZQ0ti" executionInfo={"status": "ok", "timestamp": 1632041735034, "user_tz": -330, "elapsed": 664, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="91bff666-3bee-41c0-bbd8-3543c7606057"
%%writefile src/conf/db/mysql.yaml
driver: mysql
user: omry
pass: secret
```

```python colab={"base_uri": "https://localhost:8080/"} id="wtblOWVuRBz0" executionInfo={"status": "ok", "timestamp": 1632041735765, "user_tz": -330, "elapsed": 9, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="059fdf5d-708b-4fbd-b0ba-1ffce98d6226"
%%writefile src/conf/db/postgresql.yaml
driver: postgresql
user: root
pass: password
```

```python colab={"base_uri": "https://localhost:8080/"} id="W989d2l7QpUL" executionInfo={"status": "ok", "timestamp": 1632041747156, "user_tz": -330, "elapsed": 705, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="69bc3dbe-7da3-4eec-d7ab-b1c0118ec18b"
%%writefile src/conf/config.yaml
defaults:
  - db: mysql
```

```python colab={"base_uri": "https://localhost:8080/"} id="5ZKFdsXvRL1s" executionInfo={"status": "ok", "timestamp": 1632041816763, "user_tz": -330, "elapsed": 1378, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1873b8bb-d287-44f2-c328-63f0c5c0eb1e"
!python src/myapp.py db=postgresql
```

<!-- #region id="HzKORc9hRY_8" -->
## Multirun
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LVAzTXItR078" executionInfo={"status": "ok", "timestamp": 1632041935946, "user_tz": -330, "elapsed": 1730, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="377c6c5d-8730-475a-8187-78869b8c6140"
!python src/myapp.py --multirun db=mysql,postgresql
```
