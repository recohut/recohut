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

<!-- #region id="VTHU63hYgwzT" -->
summary: In this tutorial, we will learn how to create your own codelab on cloud for free
id: how-to-create-your-own-codelab
categories: bootstrap
tags: resources, extras, codelab
status: Published 
authors: Sparsh A.
Feedback Link: https://form.jotform.com/211377288388469
<!-- #endregion -->

<!-- #region id="uC6AkZdChapT" -->
# How to create your own codelab
<!-- #endregion -->

<!-- #region id="-67Oh2k3uCIW" -->
<!-- ------------------------ -->
## Prerequisite
Duration: 5
<!-- #endregion -->

<!-- #region id="BfEICnBOqpR4" -->
### Open Colab

We will use colab notebook for building and deploying our codelabs and landing page. Go to [this](https://colab.research.google.com/) link and create a new notebook.
<!-- #endregion -->

<!-- #region id="SBa3rJ7SsMfU" -->
### Create Tutorial in colab
Write down your desired tutorial in colab as per [these](https://github.com/googlecodelabs/tools/blob/master/FORMAT-GUIDE.md) markdown instructions. You can copy/check [this](https://colab.research.google.com/gist/sparsh-ai/813507efeead92c86b8ca8b0d734e25e/codelabs-template.ipynb) example colab notebook for reference.

After creating the tutorial, save the colab by clicking on ```File -> Save``` or use keyboard shortcut ```ctrl+s```. This notebook will automatically be saved in your account's ```gdrive -> colab notebooks``` folder.
<!-- #endregion -->

<!-- #region id="2BNE6As_wsUt" -->
<!-- ------------------------ -->
## Create codelab
Duration: 10
<!-- #endregion -->

<!-- #region id="ngWbrFa_tiMw" -->
### Open a new colab notebook

You can open a new colab notebook or copy this pre-built colab, containing the same code.
<!-- #endregion -->

<!-- #region id="SlGLGtEUwAgA" -->
### Attach the google drive

To get the tutorial notebook from gdrive, we have to attach the drive to our colab runtime environment. 

Click on the third button (with google drive logo) and follow the process to connect:

![gdrive_connect](https://github.com/sparsh-ai/static/blob/main/images/gdrive_connect.png?raw=true)

The button will be changed like this after the connection:

![gdrive_connected](https://github.com/sparsh-ai/static/blob/main/images/gdrive_connected.png?raw=true)
<!-- #endregion -->

<!-- #region id="_GMzTr5OvT_C" -->
### Update these parameters
<!-- #endregion -->

```python id="HICqNEIdv1ei"
filename = 'codelabs-how-to-create-your-own-codelab-shared'
codelab_id = 'how-to-create-your-own-codelab'
target_base = "spar-data.github.io"
target_site = "my-codelabs"
git_username = "spar-data"
git_email = "spar-data@gmail.com"
```

<!-- #region id="U6ObGObhyWBH" -->
### Install go and claat command-line toolkits
<!-- #endregion -->

```python id="orwsW1edqi-8"
!add-apt-repository ppa:longsleep/golang-backports -y
!apt update
!apt install golang-go
%env GOPATH=/root/go
!go get github.com/googlecodelabs/tools/claat
!cp ~/go/bin/claat /usr/bin/
```

<!-- #region id="Ju7wQD96r5ab" -->
### Convert into codelab format
<!-- #endregion -->

```python id="YR3jQRitr7eT"
%cd /content
!cp /content/drive/MyDrive/Colab\ Notebooks/$filename'.ipynb' .
!jupyter nbconvert --to markdown $filename
!claat export $filename'.md'
```

<!-- #region id="I4Ns6eE-zN5C" -->
### Verify the codelab format
<!-- #endregion -->

```python id="aTHjEMJQzSv5"
from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(9090)"))
!cd $codelab_id && claat serve
```

<!-- #region id="HSJvChZgzjqi" -->
### Apply redirect patch
<!-- #endregion -->

```python id="ru0BhFdtzl8-"
%cd /content/$codelab_id
!mkdir -p scripts && \
cd scripts && \
rm codelab-elements.js && \
wget -q https://raw.githubusercontent.com/sparsh-ai/static/main/javascripts/codelab-elements.js
!grep -rl "<a href=\"'+hc(mc(a))+'\" id=\"arrow-back\">" ./ | xargs sed -i "s/<a href=\"'+hc(mc(a))+'\" id=\"arrow-back\">/<a href=\"'+hc(mc(a))+'\/\/\/\/"{target_base}"\/"{target_site}"\/\" id=\"arrow-back\">/g"
!grep -rl "https:\/\/storage.googleapis.com\/codelab-elements\/codelab-elements.js" ./ | xargs sed -i "s/https:\/\/storage.googleapis.com\/codelab-elements\/codelab-elements.js/scripts\/codelab-elements.js/g"
```

<!-- #region id="pvfmD-ShxA1T" -->
<!-- ------------------------ -->
## Create landing page
Duration: 10
<!-- #endregion -->

<!-- #region id="0daQqKJUyZXx" -->
### Fork the base repo
<!-- #endregion -->

<!-- #region id="QZ-oOISxDJQT" -->
Fork [this](https://github.com/sparsh-ai/codelab-tutorials) repo.

Rename the repo if you want. Update the parameter value ```target_site``` in this case.
<!-- #endregion -->

<!-- #region id="uw3WfYxS0Hkx" -->
### Setup git cli
<!-- #endregion -->

<!-- #region id="nK9F66p0DRfA" -->
There are different ways to setup git cli in colab. I will follow this process:
1. Generate personal access token
2. Store this token in ```creds.py``` python file in the following format:
```python
git_access_token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
3. Access this token using the following commands:
<!-- #endregion -->

```python id="3ZbRyBcC0M9a"
import sys
sys.path.append("/content/drive/MyDrive")
import creds
git_access_token = creds.git_access_token
```

<!-- #region id="m6YhEayUEXs3" -->
4. Basic configuration
<!-- #endregion -->

```python id="Jf-cOIwj1k0F"
%cd /content
!mkdir $target_site
%cd $target_site
!git init
!git config --global user.email $git_username
!git config --global user.name $git_email
```

<!-- #region id="eEwc4GLc1mU-" -->
### Pull the repo
<!-- #endregion -->

```python id="UYZsqoCj0wDn"
!git remote add origin https://"{git_access_token}":x-oauth-basic@github.com/"{git_username}"/"{target_site}".git
!git pull origin master
```

<!-- #region id="wjjtUOjx1oV1" -->
### Add your codelab and push the changes to feature branch
<!-- #endregion -->

```python id="vnaL8Hg71chN"
%cd /content/$target_site
!git checkout master
!git checkout -b $codelab_id
!cp -r /content/$codelab_id ./site/codelabs
!git add .
!git commit -m  'add codelab: '{codelab_id}
!git status
!git push -f origin $codelab_id
```

<!-- #region id="z8X50eOH1yCS" -->
### Review and merge

Go to your github repo and review the changes. Verify the changes and create pull request to merge this feature branch into master. If you want to skip this step, you can directly push the changes to master branch.
<!-- #endregion -->

<!-- #region id="pVUuRDzPxKVm" -->
<!-- ------------------------ -->
## Deploy the codelab
Duration: 10
<!-- #endregion -->

<!-- #region id="hISY2ynS2yzD" -->
### Pull the updated repo
<!-- #endregion -->

<!-- #region id="BAwKnwleFCWM" -->
We can rebase the repo that we pulled and updated in last few steps but to make these steps modular and independent of each other, so that we can skip steps if required, we will remove and repull the whole repo.
<!-- #endregion -->

```python id="jLUKqSIi2YmH"
%cd /content
!rm -r $target_site
!mkdir $target_site
%cd $target_site
!git init
!git remote add origin https://"{git_access_token}":x-oauth-basic@github.com/"{git_username}"/"{target_site}".git
!git pull origin master
```

<!-- #region id="Q8czEzOl2Uz6" -->
### Build the site
<!-- #endregion -->

<!-- #region id="3-gyovecGFe4" -->
This code installs node package manager and use gulp to build the static version of the whole codelab site.
<!-- #endregion -->

```python id="upBiPRdO_yAd"
%cd site
!npm install
!npm install -g npm
!npm install -g gulp-cli
!gulp dist
```

<!-- #region id="9MhrvH73GPO2" -->
We will copy the assets in a temporary folder and then paste in the branch of our repo.
<!-- #endregion -->

```python id="ksjrhXnT_122"
import shutil
shutil.copytree(f'/content/{target_site}/site/dist', '/content/temp/site')
!mv /content/temp/site/codelabs /content/temp
```

<!-- #region id="E3pt_Yx-Fscn" -->
The following code is a patch to add our site name to the base paths so that git pages can correctly pull all the local reference files.
<!-- #endregion -->

```python id="odD4kipW_10L"
!cd /content/temp/site && grep -rl '"\/[a-zA-Z0-9]' ./ | xargs sed -i 's/"\//"\/'{target_site}'\//g'
```

<!-- #region id="-gVug739wjIv" -->
### Push the changes to ```artifacts``` branch
<!-- #endregion -->

```python id="PvJL-FeFAC5R"
%cd /content/$target_site
!git reset --hard
!git checkout --orphan artifacts
!git rm -rf .
!cp -r /content/temp/site/* .
!cp -r /content/temp/codelabs .
!rm -r ./site
!git add .
!git commit -m 'build feature: '{codelab_id}
!git push -f origin artifacts
```

<!-- #region id="kye6sOaiwU-c" -->
### Attach github pages to artifacts branch
<!-- #endregion -->

<!-- #region id="D4sUMXewwtxy" -->
1. Go to your repo's settings
2. Go to 'pages' and select 'artifacts (root)' as your gh-pages
3. Go to your gh-pages url to access your codelabs
<!-- #endregion -->

<!-- #region id="PM341vSUGcOq" -->
<!-- ------------------------ -->
## Conclusion
Duration: 5
<!-- #endregion -->

<!-- #region id="6vZ-Cmw6Gknw" -->
### Verify
Go to the github pages of your repo to access the codelab site. Verify the functionality and modify/enhance the process as per requirements.

### Iterate
Add more codelabs easily by using [this](https://colab.research.google.com/gist/sparsh-ai/813507efeead92c86b8ca8b0d734e25e/codelabs-template.ipynb#scrollTo=-67Oh2k3uCIW) colab that we share in the begninning.

### In future
- Automate some steps using *Github Actions*.
- Enhance the design of codelab landing page.
- Enhance the design of codelab tutorials.
- change google analytics id to our own.
<!-- #endregion -->
