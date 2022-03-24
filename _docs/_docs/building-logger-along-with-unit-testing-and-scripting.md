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

```python id="xCK5u4y_DYZZ"
import os
project_name = "reco-tut-chef"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name); nightly=True
```

```python colab={"base_uri": "https://localhost:8080/"} id="z9zsaMZ5MAV2" executionInfo={"status": "ok", "timestamp": 1630346131297, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="17c978f8-20a6-4eb7-c015-2c86581ab11b"
if nightly:
    %cd /content
    !rm -r "{project_path}"
```

```python id="94qocxz_CtFT" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630346158254, "user_tz": -330, "elapsed": 2543, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="da77bfbc-2b7e-46d3-b3b5-ba92a07c3f87"
if not os.path.exists(project_path):
    !cp /content/drive/MyDrive/mykeys.py /content
    import mykeys
    !rm /content/mykeys.py
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    import sys; sys.path.append(path)
    !git config --global user.email "chef@recohut.com"
    !git config --global user.name  "chef"
    !git init
    !git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="KXWYJNCiGQGt"
!git status
```

```python id="zmx9Re9UGRB_"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python id="jGxfumJqzBDm"
!make setup
```

<!-- #region id="3vZFo1lhGT6d" -->
---
<!-- #endregion -->

```python id="x6cOTe1G-TwP"
!rm -r /content/reco-tut-chef/extras/logs/ml-100k/MF/*
```

```python id="o8RNNNhHySPQ"
import sys
import os
import logging
```

```python id="89uj1HcC3bG7"
class Logger(object):
    """`Logger` is a simple encapsulation of python logger.
    This class can show a message on standard output and write it into the
    file named `filename` simultaneously. This is convenient for observing
    and saving training results.
    """

    def __init__(self, filename):
        """Initializes a new `Logger` instance.
        Args:
            filename (str): File name to create. The directory component of this
                file will be created automatically if it is not existing.
        """
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(levelname)s: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # write into file
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # show on console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        # add to Handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        self.logger.critical(message)
        self._flush()
```

```python id="s5y5_J-iyoDM"
import time
import os

from src.config import Configurator
```

```python id="f9Al0htR2nD3" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630344953026, "user_tz": -330, "elapsed": 721, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="abfdbbdb-d356-40fb-9258-d457fe7313a7"
conf = Configurator("config.properties", default_section="hyperparameters")

timestamp = time.time()
data_name = 'ml-100k'
model_name = conf["recommender"]

param_str = "%s_%s" % (data_name, conf.params_str())
run_id = "%s_%.8f" % (param_str[:150], timestamp)

log_dir = os.path.join("extras", "logs", data_name, model_name)
logger_name = os.path.join(log_dir, run_id + ".log")
logger = Logger(logger_name)

print("log file is:\t", logger_name)
logger.info(data_name)
logger.warning("a random message")
```

```python colab={"base_uri": "https://localhost:8080/"} id="3OGRDFQQ074b" executionInfo={"status": "ok", "timestamp": 1630344953747, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4766a3e3-2056-4e54-8dbc-a6436e7e53eb"
!cat "{logger_name}"
```

<!-- #region id="nSwOXp9z1nAd" -->
## TDD
<!-- #endregion -->

<!-- #region id="v_0GeiHy1_P5" -->
Features
- Create the log file if not exist
- Store the messages into the file
- Store different types of messages - info, warning, error etc.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ApS4yAXA2A9t" executionInfo={"status": "ok", "timestamp": 1630344850421, "user_tz": -330, "elapsed": 907, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7e25ce6f-1db8-4540-b3a0-5f6d58081dd7"
import unittest
import tempfile


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger_name = tempfile.NamedTemporaryFile(suffix='.log').name
        self.logger = Logger(self.logger_name)
        self.logger.info('Unittest Message 1')
        self.logger.warning('Unittest Message 2')
        with open(self.logger_name, 'r') as l:
            self.msg = l.readlines()

    def testLogFileCreated(self):
        self.assertTrue(os.path.exists(self.logger_name))

    def testLogsWritten(self):
        self.assertIn('Unittest Message 1',self.msg[0])
        self.assertIn('Unittest Message 2',self.msg[1])
    
    def testLogTypes(self):
        self.assertIn('INFO:',self.msg[0])
        self.assertIn('WARNING:',self.msg[1])

unittest.main(argv=[''], verbosity=2, exit=False)
```

<!-- #region id="7RpoZsc2EQWm" -->
## Packaging
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Rp6Kr_TSIjvo" executionInfo={"status": "ok", "timestamp": 1630345862019, "user_tz": -330, "elapsed": 993, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="73f0e9d1-f0e5-4198-e563-821d123e56a2"
%%writefile ./src/logger.py
import sys
import os
import logging


class Logger(object):
    """`Logger` is a simple encapsulation of python logger.
    This class can show a message on standard output and write it into the
    file named `filename` simultaneously. This is convenient for observing
    and saving training results.
    """

    def __init__(self, filename):
        """Initializes a new `Logger` instance.
        Args:
            filename (str): File name to create. The directory component of this
                file will be created automatically if it is not existing.
        """
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(levelname)s: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # write into file
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # show on console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        # add to Handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        self.logger.critical(message)
        self._flush()
```

```python colab={"base_uri": "https://localhost:8080/"} id="V0SW9DZmIzke" executionInfo={"status": "ok", "timestamp": 1630345862984, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7e4d30e8-5cde-4268-88bd-635728fbeb54"
%%writefile ./tests/test_logger.py
import unittest
import tempfile
import os

from src.logger import Logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger_name = tempfile.NamedTemporaryFile(suffix='.log').name
        self.logger = Logger(self.logger_name)
        self.logger.info('Unittest Message 1')
        self.logger.warning('Unittest Message 2')
        with open(self.logger_name, 'r') as l:
            self.msg = l.readlines()

    def testLogFileCreated(self):
        self.assertTrue(os.path.exists(self.logger_name))

    def testLogsWritten(self):
        self.assertIn('Unittest Message 1',self.msg[0])
        self.assertIn('Unittest Message 2',self.msg[1])
    
    def testLogTypes(self):
        self.assertIn('INFO:',self.msg[0])
        self.assertIn('WARNING:',self.msg[1])
```

```python colab={"base_uri": "https://localhost:8080/"} id="YK_CFD-zJAko" executionInfo={"status": "ok", "timestamp": 1630346174968, "user_tz": -330, "elapsed": 1798, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="4ddee82a-74aa-48c5-d4fe-7faeb78c3afd"
!make setup
```

```python colab={"base_uri": "https://localhost:8080/"} id="w9o7jB9lJIK2" executionInfo={"status": "ok", "timestamp": 1630346176346, "user_tz": -330, "elapsed": 730, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9664c3e4-e2d1-49eb-a224-505e3943b962"
!make test
```
