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

```python colab={"base_uri": "https://localhost:8080/"} id="Lvd7aFDdFL7_" executionInfo={"status": "ok", "timestamp": 1630595863946, "user_tz": -330, "elapsed": 663, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1b222770-056c-483c-f4cf-c99f01ca4e9f"
%cd /content/reco-tut-base/
```

```python id="wgFEO13zkuFj"
import os
project_name = "reco-tut-ml"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Rpm6rqDekuFr" executionInfo={"status": "ok", "timestamp": 1630595804677, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2c558fd8-524b-41bb-cdc7-e0ec29318aa0"
if not os.path.exists(project_path):
    !cp -r /content/drive/MyDrive/git_credentials/. ~
    path = "/content/" + project_name; 
    !mkdir "{path}"
    %cd "{path}"
    !git init
    !git remote add origin https://github.com/"{account}"/"{project_name}".git
    !git pull origin "{branch}"
    !git checkout main
else:
    %cd "{project_path}"
```

```python id="KE-yW12wkuFs" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630595866345, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="51485bce-53bc-49fd-f203-5f481e81449b"
!git status
```

```python id="u8p0jc2DkuFt" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630595870619, "user_tz": -330, "elapsed": 1171, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="76f5e9a8-034b-4904-86b1-5fe6922b72eb"
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

```python colab={"base_uri": "https://localhost:8080/"} id="rQ6Lazhi6Sfb" executionInfo={"status": "ok", "timestamp": 1630594177613, "user_tz": -330, "elapsed": 441, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="978ecbdd-cfc2-4fe6-a699-071fb8aa4e83"
%%writefile setup.py
from setuptools import setup, find_packages

setup(
    name='recotut',
    version='0.1.0',
    packages=find_packages(include=['src', 'src.*'])
)
```

```python colab={"base_uri": "https://localhost:8080/"} id="UivX288f87Ma" executionInfo={"status": "ok", "timestamp": 1630594604246, "user_tz": -330, "elapsed": 478, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="15cb3684-3e42-40f4-a808-0fb0b7fa7bd3"
%%writefile setup.py
from setuptools import setup, find_packages

setup(
    name='recotut',
    version='0.0.1',
    description='RecSys package',
    author='Sparsh Agarwal',
    author_email='recohut@gmail.com',
    url='https://github.com/sparsh-ai',
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        'PyYAML',
        'pandas>=0.23.3',
        'numpy>=1.14.5'
    ],
    extras_require={'plotting': ['matplotlib>=2.2.0', 'jupyter']},
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Education',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Topic :: Recommendation Systems',
      ]
)
```

```python id="L-a8dtTfAHP5"

```

```python id="8qcb8gKOt2RB"
!make setup
```

```python colab={"base_uri": "https://localhost:8080/"} id="q-nrnWXLwa30" executionInfo={"status": "ok", "timestamp": 1630594650593, "user_tz": -330, "elapsed": 10492, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="1dc3b8f8-5693-41e3-ec89-8d2fde4c92d4"
!pip install -e .
```

```python colab={"base_uri": "https://localhost:8080/"} id="3-yPWUCKv62l" executionInfo={"status": "ok", "timestamp": 1630594665803, "user_tz": -330, "elapsed": 1104, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2b241115-4fdc-44cc-d920-d29feb84894a"
%%writefile /content/x.py
from src.utils.gdrive import GoogleDriveHandler
```

```python colab={"base_uri": "https://localhost:8080/"} id="Kf7Wnrv3v35i" executionInfo={"status": "ok", "timestamp": 1630594665805, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="3a99ffcd-7509-4c85-fa2e-4b3b9be10367"
%cd /content
```

```python id="9FOnXsuSwPMe"
!python x.py
```

```python id="2C07UgXa4oBw"
import
```

```python colab={"base_uri": "https://localhost:8080/"} id="3VRiBvSNwWiA" executionInfo={"status": "ok", "timestamp": 1630590540685, "user_tz": -330, "elapsed": 5455, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b8616d2c-17dc-4be2-f71f-68176c47deef"
!python3 -m pip install --upgrade pip
```

```python colab={"base_uri": "https://localhost:8080/"} id="uLwizGeHxQOO" executionInfo={"status": "ok", "timestamp": 1630591724277, "user_tz": -330, "elapsed": 1133, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ca5005e1-083e-4381-cf07-98cf3be87c09"
!mkdir /content/recotut_lib
%cd /content/recotut_lib

!mkdir -p src/ep
!touch src/__init__.py
!touch src/ep/__init__.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="hI-EO3Lfxcjz" executionInfo={"status": "ok", "timestamp": 1630591733628, "user_tz": -330, "elapsed": 755, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="dff8aabc-2443-4e5f-ce4d-2f640fc8439b"
%%writefile src/ep/example.py
def add_one(number):
    return number + 1
```

```python colab={"base_uri": "https://localhost:8080/"} id="H9I3JpCIx1BV" executionInfo={"status": "ok", "timestamp": 1630591733629, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8d89c019-f3b4-4511-bb7e-f2cfbca4c19d"
%%writefile pyproject.toml
[build-system]
requires = [
    "setuptools>=42",
    "wheel"
]
build-backend = "setuptools.build_meta"
```

```python colab={"base_uri": "https://localhost:8080/"} id="rr3g-6wXx8lm" executionInfo={"status": "ok", "timestamp": 1630591735212, "user_tz": -330, "elapsed": 7, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="f906bf55-35f3-4af8-e064-5958dc652653"
%%writefile setup.cfg
[metadata]
name = recotut
version = 0.0.1
author = Example Author
author_email = author@example.com
description = A small example package
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/pypa/sampleproject
project_urls =
    Bug Tracker = https://github.com/pypa/sampleproject/issues
classifiers =
    Programming Language :: Python :: 3
    License :: OSI Approved :: MIT License
    Operating System :: OS Independent

[options]
package_dir =
    = src
packages = find:
python_requires = >=3.6

[options.packages.find]
where = src
```

```python colab={"base_uri": "https://localhost:8080/"} id="BfyYqBRwyFAz" executionInfo={"status": "ok", "timestamp": 1630591737113, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="31ca3d27-aebd-489c-943a-536d976e8bb2"
%%writefile README.md
# Example Package

This is a simple example package. You can use
[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.
```

```python colab={"base_uri": "https://localhost:8080/"} id="5ve8uNYWyMs9" executionInfo={"status": "ok", "timestamp": 1630591738585, "user_tz": -330, "elapsed": 8, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a52a4562-885b-405e-acd7-3453c8155e7a"
%%writefile LICENSE
Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```python colab={"base_uri": "https://localhost:8080/"} id="HgqxAGL2yVyF" executionInfo={"status": "ok", "timestamp": 1630591749022, "user_tz": -330, "elapsed": 4684, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="62b05b36-7eb7-4e97-c2c9-27d48a9bed96"
!python3 -m pip install --upgrade build
```

```python colab={"base_uri": "https://localhost:8080/"} id="MuCsbbdnyxnG" executionInfo={"status": "ok", "timestamp": 1630591749023, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2be3ca1e-b94e-49fa-861e-2925ed0fad11"
!python --version
```

```python colab={"base_uri": "https://localhost:8080/"} id="aKSY1TVxymZl" executionInfo={"status": "ok", "timestamp": 1630592549704, "user_tz": -330, "elapsed": 8083, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="d5aef604-5a49-40d0-882f-245cde013cdb"
!sudo apt-get install python3.7-dev python3.7-venv
```

```python colab={"base_uri": "https://localhost:8080/"} id="lCHqPnekyWWD" executionInfo={"status": "ok", "timestamp": 1630591768290, "user_tz": -330, "elapsed": 12696, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="17eab107-b2ba-4ed4-81a4-dc207999a7e7"
!python3 -m build
```

```python colab={"base_uri": "https://localhost:8080/"} id="Fs3toEPIydqq" executionInfo={"status": "ok", "timestamp": 1630592553710, "user_tz": -330, "elapsed": 1417, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a231b6c5-f4d6-4f95-ca7c-ac12cfcf2e07"
!python3 -m pip install -e .
```

```python colab={"base_uri": "https://localhost:8080/"} id="01jjjCcBzn0y" executionInfo={"status": "ok", "timestamp": 1630591778797, "user_tz": -330, "elapsed": 20, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="9f64d10a-c2a0-4916-907c-b596a15b9813"
%cd /content
```

```python id="GMSD4utt1V6b"
from ep.example import add_one
```

```python colab={"base_uri": "https://localhost:8080/"} id="jlGRl-qc1YhI" executionInfo={"status": "ok", "timestamp": 1630591624148, "user_tz": -330, "elapsed": 544, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="6d8fa0ff-138a-4c4b-96fe-a8b91ce4c463"
add_one(10)
```

```python id="bS3XEFww1Z8q"
import src.
```
