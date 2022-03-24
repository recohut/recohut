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

<!-- #region id="Sr6mAFvh00wO" -->
### Repo Setup
<!-- #endregion -->

```python id="jfz93Q8hzGq7"
project_name = "reco-tut-esd"; branch = "master"; account = "sparsh-ai"
```

```python colab={"base_uri": "https://localhost:8080/"} id="t592Rm2vzGrB" executionInfo={"status": "ok", "timestamp": 1627319630219, "user_tz": -330, "elapsed": 1303, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="aa131f89-f65a-4920-eeff-983c4c6bd8de"
!cp /content/drive/MyDrive/mykeys.py /content
import mykeys
!rm /content/mykeys.py
path = "/content/" + project_name;
!mkdir "{path}"
%cd "{path}"
import sys; sys.path.append(path)
!git config --global user.email "nb@recohut.com"
!git config --global user.name  "colab-sparsh"
!git init
!git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git
!git pull origin "{branch}"
```

```python id="9LsmBBp1zwFc"

```

```python id="ybIZfydPzGrD"
!git status
!git add . && git commit -m 'commit' && git push origin "{branch}"
```

<!-- #region id="iy-efDcbxkZ-" -->
## Data Extraction
<!-- #endregion -->

```python id="Riyj1rNVx8as"
!pip install selenium
!apt-get update # to update ubuntu to correctly run apt install
!apt install chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin
import sys
sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
```

```python id="4ssEZa5wxy2f"
from logging import error
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
import time
import json
# from tqdm import tqdm
import re
import time
# import geopy
```

```python id="jo3n47UG0TSF"
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
```

<!-- #region id="eWj81WiC1KX2" -->
### ESD Scraping
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hjuNy4sgwz9O" cellView="form" executionInfo={"status": "ok", "timestamp": 1627319420667, "user_tz": -330, "elapsed": 99167, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="7b139b12-4c9a-4de2-8643-38f575409767"
#@markdown
driver = webdriver.Chrome("chromedriver", options=options)
driver.get('https://esd.sutd.edu.sg/academics/undergraduate-programme/courses/')

links = driver.find_elements_by_css_selector('article > div > div.fusion-flexslider.flexslider.fusion-post-slideshow > ul.slides > li > div > a')
count=0
esd_courses = [] 
for link in links:
    link_name = link.get_attribute('href')
    # print(link_name)
    esd_courses.append(link_name)
    count+=1
print(f'number of courses:{count}')

courses = []
errors = []
for idx,link_name in enumerate(esd_courses):
    course={}
    print('\n')
    print(f'navigating to {link_name}')
    driver.get(link_name)
    time.sleep(0.5)
    '''
    do the scraping within each website
    '''
    print(f'course count {idx+1}')
    course_name = driver.find_element_by_class_name('fusion-post-title')
    print(f'course_name:{course_name.text}')
    course['name'] = course_name.text

    course_description = driver.find_element_by_xpath("//*[@class='post-content']/p[1]")
    print(f'course description: {course_description.text}')
    course['description'] = course_description.text
    print('\n') 
    try:
        pre_requisites = driver.find_element_by_xpath("//b[contains(.,'equisite')]/following-sibling::a \
            | //b[contains(.,'equisite')]/../following-sibling::ul \
            | //b[contains(.,'equisite')]/../following-sibling::ol \
            | //strong[contains(.,'equisite')]/following-sibling::a \
            | //strong[contains(.,'equisite')]/../following-sibling::ul \
            | //strong[contains(.,'equisite')]/../following-sibling::ol \
            | //h4[contains(.,'equisite')]/following-sibling::ul \
            | //h4[contains(.,'equisite')]/following-sibling::ol")
        print(f'pre-requisites: {pre_requisites.text}')
        course['pre_requisite'] = [pre_requisites.text.split('\n')]
    except Exception as e:
        print('no pre-requisites')
        errors.append((course_name.text, e))
        course['pre_requisite'] = []
    print('\n')
    learning_objectives = driver.find_element_by_xpath("//h4[contains(.,'Learning Objective')]/following-sibling::ol \
            | //h4[contains(.,'Learning Objective')]/following-sibling::ul")
    print(f'learning objectives:{learning_objectives.text}')
    course['learning_objectives'] = [learning_objectives.text.split('\n')]
    print('\n')
    measurable_outcomes = driver.find_element_by_xpath("//h4[contains(.,'Outcome')]/following-sibling::ol \
            | //h4[contains(.,'Outcome')]/following-sibling::ul")
    print(f'measuerable outcomes:{measurable_outcomes.text}')
    course['measurable_outcomes'] = [measurable_outcomes.text.split('\n')]
    print('\n')
    time.sleep(0.5)
    courses.append(course)
    
with open('esd_courses.json','w') as file:
    json.dump(courses, file)

driver.quit()

print('Errors')
for i in errors:
    print(i)
```

<!-- #region id="z9BxX8pH1Fbk" -->
### ISTD Scraping
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xhl0_CDhyTIi" cellView="form" executionInfo={"status": "ok", "timestamp": 1627319997874, "user_tz": -330, "elapsed": 116016, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="93c91757-b8ac-4c8a-e79a-b352eea576cb"
#@markdown
driver = webdriver.Chrome("chromedriver", options=options)
driver.get('https://istd.sutd.edu.sg/education/undergraduate/course-catalogue/')

# links = driver.find_element_by_xpath("//div[@id='activity_filter_chosen']")'
# //*[@id="blog-1-post-6150"]/div/div[1]/ul[1]/li/div/a'
# //*[@id="blog-1-post-6153"]/div/div[1]/ul[1]/li/div/a

# link = driver.find_element_by_tag_name('a')
# print(link)
links = driver.find_elements_by_css_selector('article > div > div.fusion-flexslider.flexslider.fusion-post-slideshow > ul.slides > li > div > a')
count=0
courses = [] 
for link in links:
    link_name = link.get_attribute('href')
    # print(link_name)
    courses.append(link_name)
    count+=1
print(f'number of courses:{count}')

istd = 'https://istd.sutd.edu.sg'
epd = 'https://epd.sutd.edu.sg'
esd = 'https://esd.sutd.edu.sg'
istd_courses = [course for course in courses if istd in course]
epd_courses = [course for course in courses if epd in course]
esd_courses = [course for course in courses if esd in course]
# print(istd_courses)
# print(esd_courses)
# print(epd_courses)
# print(f'number of istd courses: {len(istd_courses)}')
# print(f'number of esd courses: {len(esd_courses)}')
# print(f'number of epd courses: {len(epd_courses)}')

# settling istd courses

# create table for istd courses
# df = pd.DataFrame(columns =['Name','Course Description, Prerequisites, Learning Objectives, Measurable Outcomes, Topics Covered'])
courses = []
errors = []

for idx,link_name in enumerate(istd_courses):
    course={}
    print('\n')
    print(f'navigating to {link_name}')
    driver.get(link_name)
    time.sleep(0.5)
    '''
    do the scraping within each website
    '''
    print(f'course count {idx+1}')
    course_name = driver.find_element_by_class_name('entry-title')
    print(f'course_name:{course_name.text}')
    course['name'] = course_name.text
    try:
        course_description = driver.find_element_by_xpath("//h4[contains(.,'Course Description')]/following-sibling::p")
        print(f'course description: {course_description.text}')
        print('\n') 
        course['description'] = course_description.text
    except Exception as e:
        errors.append((course_name.text,e))
        print(f'no course description')
        course['description'] = None
    try:
        pre_requisites = driver.find_element_by_xpath("//h4[contains(.,'requisite')]/following-sibling::ol \
            | //h4[contains(.,'requisite')]/following-sibling::ul") # finding folowing sibling with any tag name
        print(f'pre-requsities: {pre_requisites.text}')
        course['pre_requisite'] = [pre_requisites.text.split('\n')]
    except Exception as e:
        errors.append((course_name.text,e))
        print(f'no pre-requisite')
        course['pre_requisite'] = []
    print('\n')
    try: 
        learning_objectives = driver.find_element_by_xpath("//h4[contains(.,'Learning Objective')]/following-sibling::ol \
            | //h4[contains(.,'Learning Objective')]/following-sibling::ul")  # finding folowing sibling with any tag name
        print(f'learning objectives:{learning_objectives.text}')
        course['learning_objectives'] = [learning_objectives.text.split('\n')]
    except Exception as e:
        errors.append((course_name.text,e))
        print(f'no learning objectives')
        course['learning_objectives'] = []
    print('\n')
    try: 
        measureable_outcomes = driver.find_element_by_xpath("//h4[contains(.,'Measurable Outcome')]/following-sibling::ol \
            | //h4[contains(.,'Measurable Outcome')]/following-sibling::ul")
        print(f'measuerable outcomes:{measureable_outcomes.text}')
        course['measurable_outcomes'] = [measureable_outcomes.text.split('\n')]
    except Exception as e:
        errors.append((course_name.text,e))
        print('no measurable outcomes')
        course['measurable_outcomes'] = []
    print('\n') 
    try:
        topics_covered = driver.find_element_by_xpath("//h4[contains(.,'Topics Covered')]/following-sibling::ol \
            | //h4[contains(.,'Topics Covered')]/following-sibling::ul\
            | //h5[contains(.,'Topics Covered')]/following-sibling::ol \
            | //h5[contains(.,'Topics Covered')]/following-sibling::ul")
        print(f'topics covered:{topics_covered.text}')
        course['topics_covered'] = [topics_covered.text.split('\n')]
    except Exception as e:
        errors.append((course_name.text,e))
        print('no topics covered')
        course['topics_covered'] = []
    print('\n')          
    time.sleep(0.5)
    courses.append(course)

print(type(courses))

with open('istd_courses.json','w') as file:
    json.dump(courses, file)

print('Errors: ')
for i in errors:
    print(i)

driver.quit()


# # settling esd courses
# for link_name in esd_courses:
#     print(f'navigating to {link_name}')
#     driver.get(link_name)
#     time.sleep(0.5)
#     '''
#     do the scraping within each website
#     '''
#     driver.back()
#     time.sleep(0.5)

# # settling epd courses
# for link_name in epd_courses:
#     print(f'navigating to {link_name}')
#     driver.get(link_name)
#     time.sleep(0.5)
#     '''
#     do the scraping within each website
#     '''
#     driver.back()
#     time.sleep(0.5)
```

<!-- #region id="Ps4xHOe91DKm" -->
### Merging
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="RFEHB7GT0em_" cellView="form" executionInfo={"status": "ok", "timestamp": 1627320168376, "user_tz": -330, "elapsed": 537, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "", "userId": "13037694610922482904"}} outputId="c4dc946a-974f-4d5e-ecc6-ad61e4eba109"
#@markdown
import json

input_file_istd = 'istd_courses.json'
input_file_esd = 'esd_courses.json'
output_filed_merged = 'course_info.json'

# open json file
with open(input_file_istd) as f:
  istd_courses = json.load(f)

with open(input_file_esd) as f:
  esd_courses = json.load(f)

merged_courses =  istd_courses + esd_courses

istd_electives = ['50.006 User Interface Design and Implementation',
       '50.007 Machine Learning', '50.012 Networks',
       '50.017 Graphics and Visualisation', '50.020 Network Security',
       '50.021 Artificial Intelligence',
       '50.033 Foundations of Game Design and Development',
       '50.035 Computer Vision',
       '50.036 Foundations of Distributed Autonomous Systems',
       '50.037 Blockchain Technology', '50.038 Computational Data Science',
       '50.039 Theory and Practice of Deep Learning',
       '50.040 Natural Language Processing',
       '50.041 Distributed Systems and Computing',
       '50.042 Foundations of Cybersecurity',
       '50.043 Database Systems',
       '50.044 System Security', '50.045 Information Retrieval',
       '50.046 Cloud Computing and Internet of Things',
       '50.047 Mobile Robotics', '50.048 Computational Fabrication',
       'Service Design Studio', '01.116 AI for Healthcare (Term 7)',
       '01.117 Brain-Inspired Computing and its Applications (Term 8)',
       '01.102 Energy Systems and Management',
       '01.104 Networked Life', 
       '01.107 Urban Transportation']

esd_electives = ['40.230 Sustainable Engineering',
       '40.232 Water Resources Management',
       '40.240 Investment Science',
       '40.242 Derivative Pricing and Risk Management',
       '40.260 Supply Chain Management',
       '40.302 Advanced Topics in Optimisation#',
       '40.305 Advanced Topics in Stochastic Modelling#',
       '40.316 Game Theory', '40.317 Financial Systems Design',
       '40.318 Supply Chain Digitalisation and Design',
       '40.319 Statistical and Machine Learning',
       '40.320 Airport Systems Planning and Design',
       '40.321 Airport Systems Modelling and Simulation',
       '40.323 Equity Valuation', 
       '40.324 Fundamentals of Investing']

electives = istd_electives + esd_electives
elective_courses = []

for i in merged_courses:
  name = i['name']
  if name in electives:
    elective_courses.append(i)

print(f'total number of courses: {len(elective_courses)}')
with open(output_filed_merged,'w') as file:
    json.dump(elective_courses, file) # comment out this if want to include the cores

    #json.dump(elective_courses, file) # uncomment if want to include the cores
```

```python id="jyYbcgna1V66"

```
