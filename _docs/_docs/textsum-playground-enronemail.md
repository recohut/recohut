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

```python id="WslXEGMqm4dF" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 922} outputId="91d1a01b-c8f0-4ab6-ccb8-cacbaa8100d2" executionInfo={"status": "ok", "timestamp": 1587068898255, "user_tz": -330, "elapsed": 23344, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
!pip install mail-parser
!pip install talon
```

```python id="iqnYRN2vmUI8" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 71} outputId="1f9bcc0f-719a-4264-f758-5a861b4681e1" executionInfo={"status": "ok", "timestamp": 1587068899901, "user_tz": -330, "elapsed": 24789, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}}
import sys
from os import listdir
from os.path import isfile, join
import configparser
from sqlalchemy import create_engine

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import email
import mailparser
import xml.etree.ElementTree as ET
from talon.signature.bruteforce import extract_signature
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re

import dask.dataframe as dd
from distributed import Client
import multiprocessing as mp
```

```python id="eLybM5WKm1vJ" colab_type="code" colab={}
# !wget https://github.com/dailykirt/ML_Enron_email_summary/blob/master/data/enron_mail_20150507.tar.gz?raw=true
# !tar -xf /content/enron_mail_20150507.tar.gz?raw=true
mail_dir = '/content/maildir/'
```

<!-- #region id="3PabXGqnoRyJ" colab_type="text" -->
## 2. Data Input 
### A. Enron Email Dataset
The raw enron email dataset contains a maildir directory that contains folders seperated by employee which contain the emails. The following processes the raw text of each email into a dask dataframe with the following columns: 

Employee: The username of the email owner. <br>
Body: Cleaned body of the email. <br>
Subject: The title of the email. <br>
From: The original sender of the email <br>
Message-ID: Used to remove duplicate emails, as each email has a unique ID. <br>
Chain: The parsed out email chain from a email that was forwarded. <br>
Signature: The extracted signature from the body.<br>
Date: Time the email was sent. <br>

All of the Enron emails were sent using the Multipurpose Internet Mail Extensions 1.0 (MIME) format. Keeping this in mind helps find the correct libraries and methods to clean the emails in a standardized fashion. 
<!-- #endregion -->

```python id="m-jg5SyMoNMz" colab_type="code" colab={}
def process_email(index):
    '''
    This function splits a raw email into constituent parts that can be used as features.
    '''
    email_path = index[0]
    employee = index[1]
    folder = index[2]
    
    mail = mailparser.parse_from_file(email_path)
    full_body = email.message_from_string(mail.body)
    
    #Only retrieve the body of the email. 
    if full_body.is_multipart():
        return
    else:
        mail_body = full_body.get_payload()    
    
    split_body = clean_body(mail_body)
    headers = mail.headers
    #Reformating date to be more pandas readable
    date_time = process_date(headers.get('Date'))

    email_dict = {
                "employee" : employee,
                "email_folder": folder,
                "message_id": headers.get('Message-ID'),
                "date" : date_time,
                "from" : headers.get('From'),
                "subject": headers.get('Subject'),
                "body" : split_body['body'],
                "chain" : split_body['chain'],
                "signature": split_body['signature'],
                "full_email_path" : email_path #for debug purposes. 
    }
    
    #Append row to dataframe. 
    return email_dict
```

```python id="nKNuNJpQolgt" colab_type="code" colab={}
def clean_body(mail_body):
    '''
    This extracts both the email signature, and the forwarding email chain if it exists. 
    '''
    delimiters = ["-----Original Message-----","To:","From"]
    
    #Trying to split string by biggest delimiter. 
    old_len = sys.maxsize
    
    for delimiter in delimiters:
        split_body = mail_body.split(delimiter,1)
        new_len = len(split_body[0])
        if new_len <= old_len:
            old_len = new_len
            final_split = split_body
            
    #Then pull chain message
    if (len(final_split) == 1):
        mail_chain = None
    else:
        mail_chain = final_split[1] 
    
    #The following uses Talon to try to get a clean body, and seperate out the rest of the email. 
    clean_body, sig = extract_signature(final_split[0])
    
    return {'body': clean_body, 'chain' : mail_chain, 'signature': sig}
```

```python id="ysBpNTEmopSi" colab_type="code" colab={}
def process_date(date_time):
    '''
    Converts the MIME date format to a more pandas friendly type. 
    '''
    try:
        date_time = email.utils.format_datetime(email.utils.parsedate_to_datetime(date_time))
    except:
        date_time = None
    return date_time
```

```python id="WbNXr6KFowj0" colab_type="code" colab={}
def generate_email_paths(mail_dir):
    '''
    Given a mail directory, this will generate the file paths to each email in each inbox. 
    '''
    mailboxes = listdir(mail_dir)
    for mailbox in mailboxes:
        inbox = listdir(mail_dir + mailbox)
        for folder in inbox:
            path = mail_dir + mailbox + "/" + folder
            emails = listdir(path)
            for single_email in emails:
                full_path = path + "/" + single_email
                if isfile(full_path): #Skip directories.
                    yield (full_path, mailbox, folder)
```

```python id="88rXFkpxo2ox" colab_type="code" colab={}
#bug-patch
# !rm maildir/lokay-m/1.
# !rm maildir/scholtes-d/1.
# !rm maildir/baughman-d/1.
# !rm maildir/corman-s/1.
# !rm maildir/shively-h/2.
# !rm maildir/shively-h/1.
# !rm maildir/richey-c/1.
```

```python id="fYFTqXsmoz0K" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="9dc7a0e7-0b54-4fc3-decc-fcae32df2db9"
#Use multiprocessing to speed up initial data load and processing. Also helps partition DASK dataframe. 
# try:
#     cpus = mp.cpu_count()
# except NotImplementedError:
#     cpus = 2
cpus = 8
pool = mp.Pool(processes=cpus)
print("CPUS: " + str(cpus))

indexes = generate_email_paths(mail_dir)
enron_email_df = pool.map(process_email,indexes)
#Remove Nones from the list
enron_email_df = [i for i in enron_email_df if i]
enron_email_df = pd.DataFrame(enron_email_df)
```

```python id="PUP5A-BasTX4" colab_type="code" colab={}
enron_email_df.describe()
```

<!-- #region id="TsQhe_Z3vTsC" colab_type="text" -->
## BC3 corpus
This dataset is split into two xml files. One contains the original emails split line by line, and the other contains the summarizations created by the annotators. Each email may contain several summarizations from different annotators and summarizations may also be over several emails. This will create a data frame for both xml files, then join them together using the thread number in combination of the email number for a single final dataframe. 

The first dataframe will contain the wrangled original emails containing the following information:

Listno: Thread identifier <br>
Email_num: Email in thread sequence <br>
From: The original sender of the email <br>
To: The recipient of the email. <br>
Recieved: Time email was recieved. <br>
Subject: Title of email. <br>
Body: Original body. <br>
<!-- #endregion -->

```python id="QpvMS8JhvVyc" colab_type="code" colab={}
def parse_bc3_emails(root):
    '''
    This adds every BC3 email to a newly created dataframe. 
    '''
    BC3_email_list = []
    #The emails are seperated by threads.
    for thread in root:
        email_num = 0
        #Iterate through the thread elements <name, listno, Doc>
        for thread_element in thread:
            #Getting the listno allows us to link the summaries to the correct emails
            if thread_element.tag == "listno":
                listno = thread_element.text
            #Each Doc element is a single email
            if thread_element.tag == "DOC":
                email_num += 1
                email_metadata = []
                for email_attribute in thread_element:
                    #If the email_attri is text, then each child contains a line from the body of the email
                    if email_attribute.tag == "Text":
                        email_body = ""
                        for sentence in email_attribute:
                            email_body += sentence.text
                    else:
                        #The attributes of the Email <Recieved, From, To, Subject, Text> appends in this order. 
                        email_metadata.append(email_attribute.text)
                        
                #Use same enron cleaning methods on the body of the email
                split_body = clean_body(email_body)
                    
                email_dict = {
                    "listno" : listno,
                    "date" : process_date(email_metadata[0]),
                    "from" : email_metadata[1],
                    "to" : email_metadata[2],
                    "subject" : email_metadata[3],
                    "body" : split_body['body'],
                    "email_num": email_num
                }
                
                BC3_email_list.append(email_dict)           
    return pd.DataFrame(BC3_email_list)
```

```python id="jhvJw3WavZ7k" colab_type="code" colab={}
#load BC3 Email Corpus. Much smaller dataset has no need for parallel processing. 
parsedXML = ET.parse( "/content/ML_Enron_email_summary/data/BC3_Email_Corpus/corpus.xml" )
root = parsedXML.getroot()

#Clean up BC3 emails the same way as the Enron emails. 
bc3_email_df = parse_bc3_emails(root)
```

```python id="FsQqvnn5vZ5u" colab_type="code" colab={}
bc3_email_df.head(3)
```

<!-- #region id="3CPb_hORvtS-" colab_type="text" -->
The second dataframe contains the summarizations of each email:

Annotator: Person who created summarization. <br>
Email_num: Email in thread sequence. <br>
Listno: Thread identifier. <br>
Summary: Human summarization of the email. <br>
<!-- #endregion -->

```python id="t0d5VvBxvZ48" colab_type="code" colab={}
def parse_bc3_summaries(root):
    '''
    This parses every BC3 Human summary that is contained in the dataset. 
    '''
    BC3_summary_list = []
    for thread in root:
        #Iterate through the thread elements <listno, name, annotation>
        for thread_element in thread:
            if thread_element.tag == "listno":
                listno = thread_element.text
            #Each Doc element is a single email
            if thread_element.tag == "annotation":
                for annotation in thread_element:
                #If the email_attri is summary, then each child contains a summarization line
                    if annotation.tag == "summary":
                        summary_dict = {}
                        for summary in annotation:
                            #Generate the set of emails the summary sentence belongs to (often a single email)
                            email_nums = summary.attrib['link'].split(',')
                            s = set()
                            for num in email_nums:
                                s.add(num.split('.')[0].strip()) 
                            #Remove empty strings, since they summarize whole threads instead of emails. 
                            s = [x for x in set(s) if x]
                            for email_num in s:
                                if email_num in summary_dict:
                                    summary_dict[email_num] += ' ' + summary.text
                                else:
                                    summary_dict[email_num] = summary.text
                    #get annotator description
                    elif annotation.tag == "desc":
                        annotator = annotation.text
                #For each email summarizaiton create an entry
                for email_num, summary in summary_dict.items():
                    email_dict = {
                        "listno" : listno,
                        "annotator" : annotator,
                        "email_num" : email_num,
                        "summary" : summary
                    }      
                    BC3_summary_list.append(email_dict)
    return pd.DataFrame(BC3_summary_list)
```

```python id="5K9AzTzVvxpu" colab_type="code" colab={}
#Load summaries and process
parsedXML = ET.parse( "/content/ML_Enron_email_summary/data/BC3_Email_Corpus/annotation.xml" )
root = parsedXML.getroot()
bc3_summary_df = parse_bc3_summaries(root)
bc3_summary_df['email_num'] = bc3_summary_df['email_num'].astype(int)
```

```python id="YCP2ifmIvxop" colab_type="code" colab={}
bc3_summary_df.info()
```

```python id="hMMpN_Jyvxno" colab_type="code" colab={}
#merge the dataframes together
bc3_df = pd.merge(bc3_email_df, 
                  bc3_summary_df[['annotator', 'email_num', 'listno', 'summary']],
                 on=['email_num', 'listno'])
bc3_df.head()
```
