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

<!-- #region id="0NhM3Zp3wKZw" -->
## Setup
<!-- #endregion -->

```python id="dHHhW3u6psfi" executionInfo={"status": "ok", "timestamp": 1631026362015, "user_tz": -330, "elapsed": 920, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
project_name = "reco-tut-ml"; branch = "pytest"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="691kvF2Qpsfn" executionInfo={"status": "ok", "timestamp": 1631026363012, "user_tz": -330, "elapsed": 26, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6583bc72-3d36-490e-913e-67c1e9dd3ace"
if not os.path.exists(project_path):
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout "{branch}"
else:
    %cd "{project_path}"
```

```python id="mV5kT_1cpsfo"
!git status
```

```python id="sJxFMdTspsfr"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="ZzcOxcN8pzvu"
# !wget https://media.pragprog.com/titles/bopytest/code/bopytest-code.zip -O /content/code.zip
# !cd /content && unzip code.zip
```

```python id="40B3lIi-sD3D" executionInfo={"status": "ok", "timestamp": 1631026363015, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
# !pip install ipytest
import ipytest
ipytest.autoconfig()
```

<!-- #region id="w3uc20hswIuV" -->
## Development
<!-- #endregion -->

```python id="8FCz8uLnwRKO" executionInfo={"status": "ok", "timestamp": 1631026488002, "user_tz": -330, "elapsed": 500, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
from collections import namedtuple

Task = namedtuple('Task', ['summary', 'owner', 'done', 'id'])
Task.__new__.__defaults__ = (None, None, False, None)
```

<!-- #region id="PNhKMarawNlN" -->
## Testing
<!-- #endregion -->

<!-- #region id="SjTpU83_zc2A" -->
### Simple tests
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="10T9dvnwrImq" executionInfo={"status": "ok", "timestamp": 1631027327498, "user_tz": -330, "elapsed": 509, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4aee49d0-ba9a-4b5a-c77c-a62446b52d7e"
%%ipytest

"""Test the Task data type."""


def test_defaults():
    """Using no parameters should invoke defaults."""
    t1 = Task()
    t2 = Task(None, None, False, None)
    assert t1 == t2


def test_member_access():
    """Check .field functionality of namedtuple."""
    t = Task('buy milk', 'brian')
    assert t.summary == 'buy milk'
    assert t.owner == 'brian'
    assert (t.done, t.id) == (False, None)


def test_asdict():
    """_asdict() should return a dictionary."""
    t_task = Task('do something', 'okken', True, 21)
    t_dict = t_task._asdict()
    expected = {'summary': 'do something',
                'owner': 'okken',
                'done': True,
                'id': 21}
    assert t_dict == expected


def test_replace():
    """replace() should change passed in fields."""
    t_before = Task('finish book', 'brian', False)
    t_after = t_before._replace(id=10, done=True)
    t_expected = Task('finish book', 'brian', True, 10)
    assert t_after == t_expected
```

<!-- #region id="NiTZL-gZzi2e" -->
### Failing tests
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dx1VGv_HsOO4" executionInfo={"status": "ok", "timestamp": 1631027364528, "user_tz": -330, "elapsed": 512, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="17b0a872-65fd-4e99-e77a-01179986a0a5"
%%ipytest

def test_task_equality():
    """Different tasks should not be equal."""
    t1 = Task('sit there', 'brian')
    t2 = Task('do something', 'okken')
    assert t1 == t2


def test_dict_equality():
    """Different tasks compared as dicts should not be equal."""
    t1_dict = Task('make sandwich', 'okken')._asdict()
    t2_dict = Task('make sandwich', 'okkem')._asdict()
    assert t1_dict == t2_dict
```

<!-- #region id="gOTIJPS1zn3I" -->
### Fixtures
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="rglRzAyB0DMO" executionInfo={"status": "ok", "timestamp": 1631027615729, "user_tz": -330, "elapsed": 489, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7eed8ece-3317-413d-8653-f8edd903161e"
%%ipytest

import pytest

@pytest.fixture()
def some_data():
    """Return answer to ultimate question."""
    return 42


def test_some_data(some_data):
    """Use fixture return value in a test."""
    assert some_data == 42


@pytest.fixture()
def some_other_data():
    """Raise an exception from fixture."""
    x = 43
    assert x == 42
    return x


def test_other_data(some_other_data):
    """Try to use failing fixture."""
    assert some_data == 42
```

```python id="bsOHQ6zM0iJI"

```
