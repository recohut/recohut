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

<!-- #region id="nIh7bl4NsDe2" -->
# Sphinx
<!-- #endregion -->

```python id="cMhgW-nzdD5H" executionInfo={"status": "ok", "timestamp": 1621143185436, "user_tz": -330, "elapsed": 9326, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ec3510fe-9551-4961-dd37-28671982e04a" colab={"base_uri": "https://localhost:8080/", "height": 887}
!pip install -U Sphinx
```

```python id="UKF95PEurHqy"
# import sys
# sys.path.append("/content/drive/MyDrive")
# import mykeys

# project_name = "4CED0278"
# path = "/content/" + project_name
# !mkdir "{path}"
# %cd "{path}"

# import sys
# sys.path.append(path)

# !git config --global user.email "<email>"
# !git config --global user.name  "sparsh-ai"

# !git init
# !git remote add origin2 https://"{mykeys.git_token}":x-oauth-basic@github.com/sparsh-ai/"{project_name}".git

# !git pull origin2 master
```

```python colab={"base_uri": "https://localhost:8080/"} id="Vlmc5BV3tHGs" executionInfo={"status": "ok", "timestamp": 1610473730740, "user_tz": -330, "elapsed": 119372, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="e46dc2c2-3f06-47f5-ef02-df8726d33b7b"
!sphinx-quickstart
```

```python id="EsYbN5n2t4QC"
!pip install -q sphinx-rtd-theme
```

```python colab={"base_uri": "https://localhost:8080/"} id="BA0krP7CuO-A" executionInfo={"status": "ok", "timestamp": 1610480876265, "user_tz": -330, "elapsed": 2229, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aafc42d6-271b-4590-cdd4-c5b7933ec062"
%%writefile conf.py

project = 'MovieLens Recommender System'
copyright = '2021, Sparsh Agarwal'
author = 'Sparsh Agarwal'

release = '1.0.0'
extensions = ['sphinx.ext.autodoc',
              'nbsphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.githubpages',
              'IPython.sphinxext.ipython_console_highlighting',
              'rst2pdf.pdfbuilder'
]
source_suffix = ['.rst', '.ipynb']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
master_doc = 'index'
pdf_documents = [('index', u'rst2pdf', u'Sample rst2pdf doc', u'Sparsh Agarwal'),]

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
```

```python id="1wZvoYtkufpV"
!pip install sphinx
!pip install nbconvert
!pip install pandoc
!pip install latex
!pip install nbsphinx
```

```python id="WxYGwS2ru3tO"
!cp "/content/drive/MyDrive/Colab Notebooks/tutorial_temp.ipynb" .
```

```python id="N4ijhJmK6SwO"
# !mkdir documentation
!rm -r ./documentation
```

```python colab={"base_uri": "https://localhost:8080/"} id="YjVl0I4EwHcu" executionInfo={"status": "ok", "timestamp": 1610474487407, "user_tz": -330, "elapsed": 2350, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="bdcfc001-60b4-4d9b-a718-9fcd2e7d5c08"
%%writefile index.rst

.. MovieLens Recommender System documentation master file, created by
   sphinx-quickstart on Tue Jan 12 17:48:52 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MovieLens Recommender System's documentation!
========================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial_temp



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

```

```python colab={"base_uri": "https://localhost:8080/"} id="pecM_NGywj-8" executionInfo={"status": "ok", "timestamp": 1610480484655, "user_tz": -330, "elapsed": 1246, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="13a9f355-e387-48f0-a229-de24e366cf46"
%%writefile readthedocs.yml

# python version
python:
  version: 3.8
  method: pip
  install:
    - requirements: requirements.txt

# build a PDF
formats:
  - none

sphinx:
  configuration: conf.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="WnQb7X7HHgKJ" executionInfo={"status": "ok", "timestamp": 1610480550730, "user_tz": -330, "elapsed": 1830, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="cf41ca6c-2020-4a8c-8f12-7cdaf3754f5f"
%%writefile requirements.txt

python==3.8
pandoc
nbformat
jupyter_client
ipython
nbconvert
sphinx>=1.5.1
ipykernel
sphinx_rtd_theme
nbsphinx
```

```python id="67BgFQa3xmkR"
# !sphinx-build -b pdf . build/pdf
# !make html
```

```python colab={"base_uri": "https://localhost:8080/"} id="NUjA3E7kykTr" executionInfo={"status": "ok", "timestamp": 1610480561416, "user_tz": -330, "elapsed": 5003, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="2a68730f-3960-4c9c-cc8f-eca6068fe50e"
# !git checkout -b sphinx
!git add .
!git commit -m 'commit'
!git push origin2 sphinx
```

```python colab={"base_uri": "https://localhost:8080/"} id="d55898QQ31aL" executionInfo={"status": "ok", "timestamp": 1610480819345, "user_tz": -330, "elapsed": 6444, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="282f6ae9-edbe-4ec8-878e-7b318186eaf9"
!pip install -q rst2pdf
```

```python id="PXyQs2DaIwL0"

```
