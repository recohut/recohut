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

<!-- #region id="050oL2C6c8o6" -->
Who Do I Talk To?

Using Natural Language Processing to Recommend Therapists
<!-- #endregion -->

<!-- #region id="-NsQpfzedCD7" -->
<!-- #endregion -->

<!-- #region id="MgJMY2HTdXCU" -->
## Data Collection

Profile data for 4062 therapists in the Denver Metro Area from GoodTherapy.com. Each profile contains a section where the therapist can describe their approach and their practice. This is the section is outlined in red below and is the focus for this project.
<!-- #endregion -->

<!-- #region id="TL9Bjo8ZdfiB" -->
<!-- #endregion -->

```python id="-p01QlS3eGbT" executionInfo={"status": "ok", "timestamp": 1626575778783, "user_tz": -330, "elapsed": 4511, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!pip install -q watermark
```

```python id="JI1keZEcd61H" executionInfo={"status": "ok", "timestamp": 1626578248377, "user_tz": -330, "elapsed": 519, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import os
import re
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
```

```python id="lcZJQG7cnFsL"
warnings.filterwarnings("ignore")

tqdm.pandas()
%reload_ext autoreload
%autoreload 2
%reload_ext google.colab.data_table
%config InlineBackend.figure_format = 'retina' #'svg'
%reload_ext google.colab.data_table

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-notebook')

%reload_ext watermark
%watermark -m -iv

seed = 13
random.seed(seed)
np.random.seed(seed)
```

```python id="u-P1ih3mlwcN" executionInfo={"status": "ok", "timestamp": 1626578264923, "user_tz": -330, "elapsed": 610, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
def dec_check_none(func):
        def func_wrapper(*args, **kwargs):
            try:
                val = func(*args, **kwargs)
                return val
            except Exception as e:
                if isinstance(e, AttributeError):
                    return None
        return func_wrapper

class GoodTherapySoupScraper(object):
    
    def __init__(self, starting_url):
        self.starting_url = starting_url
        self.escape_chars = ['/','\n','/n','\r', '\t']

    def get_soup(self):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'}
        page = requests.get(self.starting_url, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        soup.prettify()
        return soup

    def clean_string(self, string: str) -> str:
        string = self.clean_escapes(string)
        #string = clean_punctuation(string)
        
        return string.strip()

    def clean_escapes(self, string: str):
        for esc in self.escape_chars:
            string = string.replace(esc,'')
            #string = string.replace('/n','')
            #string = string.replace('\n','')
            #string = string.replace('\r','')

        return string

    def clean_punctuation(self, string: str, keep_emo_punc=False) -> str:
        string = string.replace('&amp;', '&')

    def convert_html_list(self, li: list):
        clean_li = []
        for elem in li:
            if elem not in self.escape_chars:
                try:
                    string = ''
                    if isinstance(elem, str):
                        string = elem.strip()
                    else:
                        string = elem.text.strip()
                    
                    #check if empty
                    if string:
                        clean_li.append(string)
                except Exception as e:
                    print(f'Error caught: {e}')
                    continue
                
        return clean_li
        #return [tag.text for tag in li if type(tag) == 'li']
        
    def get_all_data(self, soup: BeautifulSoup) -> dict:
        all_data = {}
        all_data['name'] = self.get_name(soup)
        all_data['writing_sample'] = self.get_writing_sample(soup)
        all_data['issues'] = self.get_tx_issues(soup)
        all_data['orientations'] = self.get_orientations(soup)
        all_data['services'] = self.get_services(soup)
        all_data['ages'] = self.get_client_ages(soup)
        all_data['professions'] = self.get_professions(soup)
        all_data['credential'] = self.get_primary_credential(soup)
        all_data['license_status'] = self.get_license_status(soup)
        all_data['website'] = self.get_website(soup)
        all_data['address'] = self.get_address(soup)
        all_data['phone'] = self.get_phone(soup)
        all_data['verified'] = self.get_verification(soup)
        
        return all_data

    @dec_check_none
    def get_name(self, soup: BeautifulSoup) -> str:
        name = soup.find('h1', id='profileTitle_id').contents[1].get_text()
        return self.clean_escapes(name)

    @dec_check_none
    def get_writing_sample(self, soup: BeautifulSoup) -> str:
        desc = soup.find_all('div', class_='profileBottomLeft')
        all_text = desc[0].find_all('div', class_='text')
        good_stuff = []
        for txt in all_text:
            for child in txt.children:
                if(child.name == 'p'):
                    good_stuff.append(child.get_text())

        good_stuff_st = ''.join(good_stuff)
        return good_stuff_st

    @dec_check_none
    def get_tx_issues(self, soup: BeautifulSoup)-> list:
        issues_html = soup.find_all('ul', id='issuesData')
        issues_list = list(issues_html[0].children)
        
        ##if want to return string instead
        # issues_str = issues_html[0].get_text()
        # clean_str = clean_string(issues_str)
        list_text = self.convert_html_list(issues_list)
        return list_text

    @dec_check_none
    def get_orientations(self, soup: BeautifulSoup)-> list:
        orientations_html = soup.find_all('ul', id='modelsData')
        orientations_list = list(orientations_html[0].children)
        
        ##if want to return string instead
        # issues_str = issues_html[0].get_text()
        # clean_str = clean_string(issues_str)
        list_text = self.convert_html_list(orientations_list)
        return list_text

    @dec_check_none
    def get_services(self, soup: BeautifulSoup)-> list:
        services_html = soup.find_all('ul', id='servicesprovidedData')
        services_list = list(services_html[0].children)
        
        list_text = self.convert_html_list(services_list)
        return list_text

    @dec_check_none
    def get_client_ages(self, soup: BeautifulSoup) -> list:
        ages_html = soup.find_all('ul', id='agesData')
        ages_list = list(ages_html[0].children)
        
        list_text = self.convert_html_list(ages_list)
        return list_text

    @dec_check_none
    def get_professions(self, soup: BeautifulSoup) -> list:
        profs_str = soup.find('span', id='professionsDefined').get_text()
        profs_list = profs_str.split(',')
        
        return [prof.strip() for prof in profs_list]

    @dec_check_none
    def get_primary_credential(self, soup: BeautifulSoup) -> str:
        credential = soup.find('span', id='licenceinfo1').get_text()
        
        return self.clean_escapes(credential)

    @dec_check_none
    def get_license_status(self, soup: BeautifulSoup) -> str:
        license_status = soup.find('span', id='license_status_id').get_text()
        
        return self.clean_escapes(license_status)

    @dec_check_none
    def get_website(self, soup: BeautifulSoup) -> str:
        try:
            website = soup.find('a', id='edit_website')['href']
        except:
            website = 'None'
        return website

    def get_address(self, soup: BeautifulSoup) -> dict:
        #office = soup.find('div', id='editOffice1')
        address = {}
        
        address['street'] = self.sub_get_street(soup)
        address['city'] = self.sub_get_city(soup)
        address['state'] = self.sub_get_state(soup)
        address['zip'] = self.sub_get_zip(soup)
        
        return address

    @dec_check_none
    def sub_get_street(self, soup: BeautifulSoup) -> str:
            return soup.find('span', itemprop='streetAddress').get_text()

    @dec_check_none      
    def sub_get_city(self, soup: BeautifulSoup) -> str:
        return soup.find('span', itemprop='addressLocality').get_text()

    @dec_check_none
    def sub_get_state(self, soup: BeautifulSoup) -> str:
        return soup.find('span', itemprop='addressRegion').get_text()

    @dec_check_none            
    def sub_get_zip(self, soup: BeautifulSoup) -> str:
        return soup.find('span', itemprop='postalCode').get_text()

    @dec_check_none
    def get_phone(self, soup: BeautifulSoup) -> str:
        phone  =soup.find('span', {'class':'profilePhone'}).text
        #phone = soup.find('span', class='profilePhone').contents[1].contents[0].get_text()

        return self.clean_string(phone)

    @dec_check_none
    def get_verification(self, soup: BeautifulSoup) -> bool:
        verified  = soup.find('div', {'class':'profileVer'}).text

        return self.clean_string(verified) == 'Verified'
```

<!-- #region id="QHHAQBffo1D_" -->
<!-- #endregion -->

```python id="Yi8O7AkBl741" executionInfo={"status": "ok", "timestamp": 1626578382223, "user_tz": -330, "elapsed": 487, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
start_urls = 'https://www.goodtherapy.org/therapists/profile/nicole-nakamura-20190823'
good_scraper = GoodTherapySoupScraper(starting_url=start_urls)
```

```python id="gkafQxeWnrbI" executionInfo={"status": "ok", "timestamp": 1626578384652, "user_tz": -330, "elapsed": 678, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
soup = good_scraper.get_soup()
```

```python colab={"base_uri": "https://localhost:8080/"} id="lXYkc6UuntLx" executionInfo={"status": "ok", "timestamp": 1626578386731, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ae620783-cc6e-422c-e60e-b4f8293548e4"
profs = good_scraper.get_professions(soup)
profs
```

```python colab={"base_uri": "https://localhost:8080/"} id="oQwq_TPinugF" executionInfo={"status": "ok", "timestamp": 1626578445813, "user_tz": -330, "elapsed": 557, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="20981af0-cd5b-4c86-93d2-e9e680679344"
issues = good_scraper.get_tx_issues(soup)
issues
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZwJIjipmoZKQ" executionInfo={"status": "ok", "timestamp": 1626578473453, "user_tz": -330, "elapsed": 457, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="89826b0f-8b61-4ccc-f89d-f4db2a93416f"
data_dict = good_scraper.get_all_data(soup)
data_dict
```

```python colab={"base_uri": "https://localhost:8080/"} id="wyThmmp6oZGT" executionInfo={"status": "ok", "timestamp": 1626578502742, "user_tz": -330, "elapsed": 443, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="edb13e74-78be-4294-bb27-e6f1dc1d5a87"
for k, v in data_dict.items():
    print(f'{k.upper()} : {v}')
```

```python colab={"base_uri": "https://localhost:8080/"} id="eXHjDF66oZCs" executionInfo={"status": "ok", "timestamp": 1626578608422, "user_tz": -330, "elapsed": 524, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="a40a4cf6-649e-4098-e480-541b9ce5cb6d"
good_scraper.get_all_data(soup)
```

```python id="Q4iAxlcBpqKH"
# Install postgresql server
!sudo apt-get -y -qq update
!sudo apt-get -y -qq install postgresql
!sudo service postgresql start

# Setup a password `postgres` for username `postgres`
!sudo -u postgres psql -U postgres -c "ALTER USER postgres PASSWORD 'postgres';"

# Setup a database with name `therapist_predictor` to be used
!sudo -u postgres psql -U postgres -c 'DROP DATABASE IF EXISTS therapist_predictor;'
!sudo -u postgres psql -U postgres -c 'CREATE DATABASE therapist_predictor;'
```

```python colab={"base_uri": "https://localhost:8080/"} id="_bu073ZfqNmj" executionInfo={"status": "ok", "timestamp": 1626579081911, "user_tz": -330, "elapsed": 429, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="ac59f1aa-89cc-44d2-bffd-9788b0e59990"
%env DATABASE_NAME=therapist_predictor
%env DATABASE_HOST=localhost
%env DATABASE_PORT=5432
%env DATABASE_USER=postgres
%env DATABASE_PASS=postgres
```

```python id="s6QfKk46pWmJ" executionInfo={"status": "ok", "timestamp": 1626578729491, "user_tz": -330, "elapsed": 1765, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import psycopg2
```

```python id="Qs6AnNHqpd_o" executionInfo={"status": "ok", "timestamp": 1626579142519, "user_tz": -330, "elapsed": 440, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
conn = psycopg2.connect(dbname=os.environ['DATABASE_NAME'],
                        user=os.environ['DATABASE_USER'],
                        host=os.environ['DATABASE_HOST'],
                        password=os.environ['DATABASE_PASS'])
cur = conn.cursor()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 231} id="xF-EK0Myo9d7" executionInfo={"status": "error", "timestamp": 1626579284324, "user_tz": -330, "elapsed": 1232, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="b794a463-04cb-4507-aca0-21d0c0b4b8fc"
start_url = 'https://www.goodtherapy.org/therapists/profile/nicole-nakamura-20190823'
good_scraper = GoodTherapySoupScraper(starting_url=start_url)
soup = good_scraper.get_soup()

all_data = good_scraper.get_all_data(soup)

#therapist_id = cur.execute("""DROP TABLE IF EXISTS quotes""")
services = good_scraper.get_services(soup)
print(all_data['full_name'])
print(all_data['first_name'])
print(all_data['last_name'])
print(all_data['phone'])

# conn.close()
```

```python colab={"base_uri": "https://localhost:8080/"} id="0Ad6UIarrinw" executionInfo={"status": "ok", "timestamp": 1626579320878, "user_tz": -330, "elapsed": 477, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="8015979b-00d4-4d0f-a7e0-7acc14f5c9a0"
all_data
```

```python id="NBNBqHH2rruE"

```
