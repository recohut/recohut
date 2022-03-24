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
    language: python
    name: python3
---

<!-- #region id="Mo2UCxDW7s07" -->
# Using Online Job Postings to Improve your Data Science Resume
<!-- #endregion -->

<!-- #region id="MJuHmxDbzaqP" -->
We’re ready to expand our data science career. Six months from now, we’ll apply for a new job. In preparation, we begin to draft our resume. The early draft is rough and incomplete. It doesn’t yet cover our career goals or education.

Our resume draft is far from perfect. It’s possible that certain vital data science skills are not yet represented. If so, what are those missing skills? We decide to find out analytically. After all, we are data scientists! We fill in gaps in knowledge using rigorous analysis, so why shouldn’t we apply that rigorous analysis to ourselves?

First we need some data. We go online and visit a popular job-search site. The website offers millions of searchable job listings, posted by understaffed employers. A built-in search engine allows us to filter the jobs by keyword, such as *analyst* or *data scientist*. Additionally, the search engine can match jobs to uploaded documents. This feature is intended to search postings based on resume content. Unfortunately, our resume is still a work in progress. So instead, we search on the table of contents of a book! We copy and paste the first 15 listed sections of the table of contents into a text file.

Next, we upload the file to the job-search site. Material is compared against millions of job listings, and thousands of job postings are returned. Some of these postings may be more relevant than others; we can’t vouch for the search engine’s overall quality, but the data is appreciated. We download the HTML from every posting.

Our goal is to extract common data science skills from the downloaded data. We’ll then compare these skills to our resume to determine which skills are missing. To reach our goal, we’ll proceed like this:

1. Parse out all the text from the downloaded HTML files.
2. Explore the parsed output to learn how job skills are commonly described in online postings. Perhaps specific HTML tags are more commonly used to underscore job skills.
3. Try to filter out any irrelevant job postings from our dataset. The search engine isn’t perfect. Perhaps some irrelevant postings were erroneously downloaded. We can evaluate relevance by comparing the postings with our resume and the table of contents.
4. Cluster the job skills within the relevant postings, and visualize the clusters.
5. Compare the clustered skills to our resume content. We’ll then make plans to update our resume with any missing data science skills.

<aside>
📌 To address the problem at hand, we need to know how to do the following: 1) Measure similarity between texts. 2) Efficiently cluster large text datasets. 3) Visually display multiple text clusters. 4) Parse HTML files for text content.

</aside>

Our rough draft of the resume is stored in the file resume.txt. The full text of that draft is as follows:

```
Experience
1. Developed probability simulations using NumPy
2. Assessed online ad clicks for statistical significance using permutation testing
3. Analyzed disease outbreaks using common clustering algorithms
Additional Skills
1. Data visualization using Matplotlib
2. Statistical analysis using SciPy
3. Processing structured tables using Pandas
4. Executing K-means clustering and DBSCAN clustering using scikit-learn
5. Extracting locations from text using GeoNamesCache
6. Location analysis and visualization using GeoNamesCache and Cartopy
7. Dimensionality reduction with PCA and SVD using scikit-learn
8. NLP analysis and text topic detection using scikit-learn
```

Our preliminary draft is short and incomplete. To compensate for any missing material, we also use the partial table of contents of the book, which is stored in the file table_of_contents.txt. It covers the first 15 sections of the book, as well as all the top-level subsection headers. The table of contents file has been utilized to search for thousands of relevant job postings that were downloaded and stored in a job_postings directory. Each file in the directory is an HTML file associated with an individual posting. These files can be viewed locally in a web browser.
<!-- #endregion -->

```sh id="nlg3nSBURvMd"
wget -q --show-progress https://github.com/sparsh-ai/general-recsys/raw/T426474/bookcamp_code.zip
unzip bookcamp_code.zip
unzip bookcamp_code/Case_Study4.zip
unzip Case_Study4/job_postings.zip 
mv Case_Study4/resume.txt .
mv Case_Study4/table_of_contents.txt .
```

```python id="HkJZwoshTXkK"
import warnings
warnings.filterwarnings('ignore')
```

```python id="ECXzq1XMLzoT" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637477971667, "user_tz": -330, "elapsed": 573, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="2e3e3199-3fdd-493f-8c29-b0c7bb8516c3"
import glob
html_contents = []

for file_name in sorted(glob.glob('job_postings/*.html')):
    try:
        with open(file_name, 'r') as f:
            html_contents.append(f.read())
    except:
        None
            
print(f"We've loaded {len(html_contents)} HTML files.")
```

```python id="oWvHcpHsLzoT"
from bs4 import BeautifulSoup as bs

soup_objects = []
for html in html_contents:
    soup = bs(html)
    assert soup.title is not None
    assert soup.body is not None
    soup_objects.append(soup)
```

```python id="BiIJ1QNQLzoU" colab={"base_uri": "https://localhost:8080/", "height": 175} executionInfo={"status": "ok", "timestamp": 1637478038990, "user_tz": -330, "elapsed": 699, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b46e09b6-39e3-4c3b-f932-82856e98d9cb"
import pandas as pd
html_dict = {'Title': [], 'Body': []}

for soup in soup_objects:
    title = soup.find('title').text
    body = soup.find('body').text
    html_dict['Title'].append(title)
    html_dict['Body'].append(body)

df_jobs = pd.DataFrame(html_dict)
summary = df_jobs.describe()
summary
```

```python id="PZ-vRlcrLzoW" colab={"base_uri": "https://localhost:8080/", "height": 508} executionInfo={"status": "ok", "timestamp": 1637478054240, "user_tz": -330, "elapsed": 555, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8f2e774f-e590-49e8-9bd5-84fc405df6a2"
from IPython.core.display import display, HTML
assert len(set(html_contents)) == len(html_contents)
display(HTML(html_contents[0]))
```

```python id="cBeuVIwILzoY"
df_jobs['Bullets'] = [[bullet.text.strip()
                      for bullet in soup.find_all('li')]
                      for soup in soup_objects]
```

```python id="xGwaTSzpLzoY" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478074247, "user_tz": -330, "elapsed": 512, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="580de398-5de7-4294-b20a-cb2fb14a9527"
bulleted_post_count = 0
for bullet_list in df_jobs.Bullets:
    if bullet_list:
        bulleted_post_count += 1

percent_bulleted = 100 * bulleted_post_count / df_jobs.shape[0]
print(f"{percent_bulleted:.2f}% of the postings contain bullets")
```

```python id="L26ODP82LzoZ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478154008, "user_tz": -330, "elapsed": 1479, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d024f9e4-d5d2-422a-ab87-6c271f143c93"
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def rank_words(text_list):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_list).toarray()
    df = pd.DataFrame({'Words': vectorizer.get_feature_names(),
                       'Summed TFIDF': tfidf_matrix.sum(axis=0)})
    sorted_df = df.sort_values('Summed TFIDF', ascending=False)
    return sorted_df

all_bullets = []
for bullet_list in df_jobs.Bullets:
    all_bullets.extend(bullet_list)

sorted_df = rank_words(all_bullets)
print(sorted_df[:5].to_string(index=False))
```

```python id="DkRPDQhULzoa" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478154803, "user_tz": -330, "elapsed": 802, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="d20595f0-3441-45c5-aad7-23b890531eaf"
non_bullets = []
for soup in soup_objects:
    body = soup.body
    for tag in body.find_all('li'):
        tag.decompose()

    non_bullets.append(body.text)

sorted_df = rank_words(non_bullets)
print(sorted_df[:5].to_string(index=False))
```

```python id="Yo6EXnxiLzoa" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478163335, "user_tz": -330, "elapsed": 1892, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b725bccd-b7fa-4e12-f02a-18092609e0d6"
regex = r'Data Scien(ce|tist)'
df_non_ds_jobs = df_jobs[~df_jobs.Title.str.contains(regex, case=False)]

percent_non_ds = 100 * df_non_ds_jobs.shape[0] / df_jobs.shape[0]
print(f"{percent_non_ds:.2f}% of the job posting titles do not mention a "
       "data science position. Below is a sample of such titles:\n")

for title in df_non_ds_jobs.Title[:10]:
    print(title)
```

```python id="iIdUpNilLzob"
resume = open('resume.txt', 'r').read()
table_of_contents = open('table_of_contents.txt', 'r').read()
existing_skills = resume + table_of_contents
```

```python id="yH74jtOTLzob"
text_list = df_jobs.Body.values.tolist() + [existing_skills]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(text_list).toarray()
cosine_similarities = tfidf_matrix[:-1] @ tfidf_matrix[-1]
```

```python id="B88lsXoGLzob" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478261652, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f7ab2000-f0f8-4f37-95fe-bd1202b0615d"
df_jobs['Relevance'] = cosine_similarities
sorted_df_jobs = df_jobs.sort_values('Relevance', ascending=False)
for title in sorted_df_jobs[-20:].Title:
    print(title)
```

```python id="dyt39tr6Lzoc" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478270362, "user_tz": -330, "elapsed": 1220, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7c4d100a-0162-4290-b79b-754a41fb0a7f"
for title in sorted_df_jobs[:20].Title:
    print(title)
```

```python id="9wx5G7KTLzoc" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637478272732, "user_tz": -330, "elapsed": 1079, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="daa99d34-7f7f-43cd-eb75-353ca04a156d"
import matplotlib.pyplot as plt
plt.plot(range(df_jobs.shape[0]), sorted_df_jobs.Relevance.values)
plt.xlabel('Index')
plt.ylabel('Relevance')
plt.show()
```

```python id="NffsS-A_Lzod" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637478276750, "user_tz": -330, "elapsed": 775, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5812c926-7820-48c2-8591-2c3b5f0361f6"
plt.plot(range(df_jobs.shape[0]), sorted_df_jobs.Relevance.values)
plt.xlabel('Index')
plt.ylabel('Relevance')
plt.axvline(60, c='k')
plt.show()
```

```python id="BUaU2Y_ALzod" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478277518, "user_tz": -330, "elapsed": 19, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8ce0256c-b1a3-411a-ad77-6b56142df040"
for title in sorted_df_jobs[40: 60].Title.values:
    print(title)
```

```python id="DYdd7RY5Lzog" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478281211, "user_tz": -330, "elapsed": 16, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="7bbacb4b-a990-4cce-f457-0fa90931f65d"
for title in sorted_df_jobs[60: 80].Title.values:
    print(title)
```

```python id="tbz5Vf9uLzoh" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478284698, "user_tz": -330, "elapsed": 629, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f5907de1-4ac9-4f25-fe53-21a68679e2a9"
import re
def percent_relevant_titles(df):
    regex_relevant = re.compile(r'Data (Scien|Analy)',
                                flags=re.IGNORECASE)
    regex_irrelevant = re.compile(r'\b(Manage)',
                                  flags=re.IGNORECASE)
    match_count = len([title for title in df.Title
                       if regex_relevant.search(title)
                       and not regex_irrelevant.search(title)])
    percent = 100 * match_count / df.shape[0]
    return percent

percent = percent_relevant_titles(sorted_df_jobs[60: 80])
print(f"Approximately {percent:.2f}% of job titles between indices "
       "60 - 80 are relevant")
```

```python id="wLX3Mk_GLzoh" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478287020, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="f13440ec-39f7-409a-a0de-c337bcfea4d2"
percent = percent_relevant_titles(sorted_df_jobs[80: 100])
print(f"Approximately {percent:.2f}% of job titles between indices "
       "80 - 100 are relevant")
```

```python id="PP8v4MUwLzoi" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637478289701, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8e4d446a-4c92-46f6-b5bd-64de38b08114"
def relevant_title_plot(index_range=20):
    percentages = []
    start_indices = range(df_jobs.shape[0] - index_range)
    for i in start_indices:
        df_slice = sorted_df_jobs[i: i + index_range]
        percent = percent_relevant_titles(df_slice)
        percentages.append(percent)

    plt.plot(start_indices, percentages)
    plt.axhline(50, c='k')
    plt.xlabel('Index')
    plt.ylabel('% Relevant Titles')

relevant_title_plot()
plt.show()
```

```python id="OQRJc4P_Lzoi" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637478294537, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="856625ea-aca1-4718-a4b4-96f2ab11d988"
relevant_title_plot(index_range=40)
plt.axvline(700, c='k')
plt.show()
```

```python id="4GXerbkhLzoi" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478298289, "user_tz": -330, "elapsed": 15, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="b931aa5d-e922-4c60-8c69-68a2d7826986"
total_bullets = []
for bullets in sorted_df_jobs[:60].Bullets:
    total_bullets.extend(bullets)
df_bullets = pd.DataFrame({'Bullet': total_bullets})
print(df_bullets.describe())
```

```python id="u4Ukjy5HLzoj" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478299982, "user_tz": -330, "elapsed": 13, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c76f6536-0a63-453f-d613-8f996df6cc2a"
total_bullets = sorted(set(total_bullets))
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(total_bullets)
num_rows, num_columns = tfidf_matrix.shape
print(f"Our matrix has {num_rows} rows and {num_columns} columns")
```

```python id="phJZwutsLzoj"
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
np.random.seed(0)

def shrink_matrix(tfidf_matrix):
    svd_object = TruncatedSVD(n_components=100)
    shrunk_matrix = svd_object.fit_transform(tfidf_matrix)
    return normalize(shrunk_matrix)

shrunk_norm_matrix = shrink_matrix(tfidf_matrix)
```

```python id="E20uivrlLzoj" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637478309957, "user_tz": -330, "elapsed": 8514, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="5cead196-74ee-4c78-b3cd-08f910346490"
np.random.seed(0)
from sklearn.cluster import MiniBatchKMeans
def generate_elbow_plot(matrix):
    k_values = range(1, 61)
    inertia_values = [MiniBatchKMeans(k).fit(matrix).inertia_
                      for k in k_values]
    plt.plot(k_values, inertia_values)
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

generate_elbow_plot(shrunk_norm_matrix)
```

```python id="Pudb-riULzok"
np.random.seed(0)
from sklearn.cluster import KMeans

def compute_cluster_groups(shrunk_norm_matrix, k=15,
                           bullets=total_bullets):
    cluster_model = KMeans(n_clusters=k)
    clusters = cluster_model.fit_predict(shrunk_norm_matrix)
    df = pd.DataFrame({'Index': range(clusters.size), 'Cluster': clusters,
                       'Bullet': bullets})
    return [df_cluster for  _, df_cluster in df.groupby('Cluster')]

cluster_groups = compute_cluster_groups(shrunk_norm_matrix)
```

```python id="JETVZnzNLzok" colab={"base_uri": "https://localhost:8080/", "height": 219} executionInfo={"status": "ok", "timestamp": 1637478321677, "user_tz": -330, "elapsed": 24, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="a6964418-ffd6-4fee-81ca-0f9374cc8887"
from wordcloud import WordCloud
np.random.seed(0)

def cluster_to_image(df_cluster, max_words=10, tfidf_matrix=tfidf_matrix,
                     vectorizer=vectorizer):
    indices = df_cluster.Index.values
    summed_tfidf = np.asarray(tfidf_matrix[indices].sum(axis=0))[0]
    data = {'Word': vectorizer.get_feature_names(),'Summed TFIDF': summed_tfidf}
    df_ranked_words = pd.DataFrame(data).sort_values('Summed TFIDF', ascending=False)
    words_to_score = {word: score
                     for word, score in df_ranked_words[:max_words].values
                     if score != 0}
    cloud_generator = WordCloud(background_color='white',
                                color_func=_color_func,
                                random_state=1)
    wordcloud_image = cloud_generator.fit_words(words_to_score)
    return wordcloud_image

def _color_func(*args, **kwargs):
    return np.random.choice(['black', 'blue', 'teal', 'purple', 'brown'])

wordcloud_image = cluster_to_image(cluster_groups[0])
plt.imshow(wordcloud_image, interpolation="bilinear")
plt.show()
```

```python id="cOfckTrfLzok" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478328887, "user_tz": -330, "elapsed": 1178, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="4e244e63-9968-4467-bd99-7d9a26d6af56"
np.random.seed(1)
def print_cluster_sample(cluster_id):
    df_cluster = cluster_groups[cluster_id]
    for bullet in np.random.choice(df_cluster.Bullet.values, 5,
                                   replace=False):
        print(bullet)

print_cluster_sample(0)
```

```python id="QWwE3IIQLzom" colab={"base_uri": "https://localhost:8080/", "height": 754} executionInfo={"status": "ok", "timestamp": 1637478335645, "user_tz": -330, "elapsed": 5608, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="1481e23c-981b-4241-9250-e85e98b510a0"
def plot_wordcloud_grid(cluster_groups, num_rows=5, num_columns=3,
                        **kwargs):
    figure, axes = plt.subplots(num_rows, num_columns, figsize=(20, 15))
    cluster_groups_copy = cluster_groups[:]
    for r in range(num_rows):
        for c in range(num_columns):
            if not cluster_groups_copy:
                break

            df_cluster = cluster_groups_copy.pop(0)
            wordcloud_image = cluster_to_image(df_cluster, **kwargs)
            ax = axes[r][c]
            ax.imshow(wordcloud_image,
            interpolation="bilinear")
            ax.set_title(f"Cluster {df_cluster.Cluster.iloc[0]}")
            ax.set_xticks([])
            ax.set_yticks([])

plot_wordcloud_grid(cluster_groups)
plt.show()
```

```python id="GiTUmCgQLzop" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478337398, "user_tz": -330, "elapsed": 12, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="af8e96cf-d128-4900-bd55-841c4076d71e"
np.random.seed(1)
print_cluster_sample(7)
```

```python id="7wpViQZOLzoq"
def compute_bullet_similarity(bullet_texts):
    bullet_vectorizer = TfidfVectorizer(stop_words='english')
    matrix = bullet_vectorizer.fit_transform(bullet_texts + [resume])
    matrix = matrix.toarray()
    return matrix[:-1] @ matrix[-1]

bullet_cosine_similarities = compute_bullet_similarity(total_bullets)
```

```python id="kGD2lQPiLzor" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478349919, "user_tz": -330, "elapsed": 545, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="8f1f710e-d2f5-466b-db52-dba92c4795f8"
def compute_mean_similarity(df_cluster):
    indices = df_cluster.Index.values
    return bullet_cosine_similarities[indices].mean()

tech_mean = compute_mean_similarity(cluster_groups[13])
soft_mean =  compute_mean_similarity(cluster_groups[6])
print(f"Technical cluster 13 has a mean similarity of {tech_mean:.3f}")
print(f"Soft-skill cluster 6 has a mean similarity of {soft_mean:.3f}")
```

```python id="TiuVdm38Lzos" colab={"base_uri": "https://localhost:8080/", "height": 754} executionInfo={"status": "ok", "timestamp": 1637478355184, "user_tz": -330, "elapsed": 4582, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="092acafc-4369-4cb0-ad27-9083bb053cca"
def sort_cluster_groups(cluster_groups):
    mean_similarities = [compute_mean_similarity(df_cluster)
                         for df_cluster in cluster_groups]

    sorted_indices = sorted(range(len(cluster_groups)),
                            key=lambda i: mean_similarities[i],
                            reverse=True)
    return [cluster_groups[i] for i in sorted_indices]

sorted_cluster_groups = sort_cluster_groups(cluster_groups)
plot_wordcloud_grid(sorted_cluster_groups)
plt.show()
```

```python id="cqcc93t7Lzot" colab={"base_uri": "https://localhost:8080/", "height": 741} executionInfo={"status": "ok", "timestamp": 1637478357194, "user_tz": -330, "elapsed": 26, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="53c19ee7-a7c8-4b79-c049-3cd98ae043fc"
plot_wordcloud_grid(sorted_cluster_groups[:6], num_rows=3, num_columns=2)
plt.show()
```

```python id="4R8qu-U_Lzov" colab={"base_uri": "https://localhost:8080/", "height": 741} executionInfo={"status": "ok", "timestamp": 1637478359834, "user_tz": -330, "elapsed": 2657, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="05854b2c-6674-4f67-e13d-7b7bfa9745e7"
plot_wordcloud_grid(sorted_cluster_groups[:6], num_rows=3, num_columns=2)
plt.show()
```

```python id="yE7AAOt8Lzow" colab={"base_uri": "https://localhost:8080/", "height": 686} executionInfo={"status": "ok", "timestamp": 1637478364499, "user_tz": -330, "elapsed": 4680, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="e9ab1fb0-9298-4b82-a578-f7b454ea61f7"
np.random.seed(0)
cluster_groups = compute_cluster_groups(shrunk_norm_matrix, k=25)
sorted_cluster_groups = sort_cluster_groups(cluster_groups)
plot_wordcloud_grid(sorted_cluster_groups, num_rows=5, num_columns=5)
plt.show()
```

```python id="5FGJL7mRLzoy" colab={"base_uri": "https://localhost:8080/", "height": 656} executionInfo={"status": "ok", "timestamp": 1637478368297, "user_tz": -330, "elapsed": 3810, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="6d50f679-b359-4fad-e651-563936e489d5"
np.random.seed(0)
cluster_groups = compute_cluster_groups(shrunk_norm_matrix, k=20)
sorted_cluster_groups = sort_cluster_groups(cluster_groups)
plot_wordcloud_grid(sorted_cluster_groups, num_rows=4, num_columns=5)
plt.show()
```

```python id="JY-9O8jBLzoz" colab={"base_uri": "https://localhost:8080/", "height": 656} executionInfo={"status": "ok", "timestamp": 1637478371191, "user_tz": -330, "elapsed": 2911, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="44cd2be5-1ae4-4687-95ed-d4cfea14abf3"
np.random.seed(0)
cluster_groups = compute_cluster_groups(shrunk_norm_matrix, k=20)
sorted_cluster_groups = sort_cluster_groups(cluster_groups)
plot_wordcloud_grid(sorted_cluster_groups, num_rows=4, num_columns=5)
plt.show()
```

<!-- #region id="cnySYM-qLzo0" -->
**Analysing 700 postings**
<!-- #endregion -->

```python id="E8BOGn9zLzo0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1637478372201, "user_tz": -330, "elapsed": 1040, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="9e674501-2dc1-4048-caa0-39b6f77df3ff"
np.random.seed(0)
total_bullets_700 = set()
for bullets in sorted_df_jobs[:700].Bullets:
    total_bullets_700.update([bullet.strip()
                              for bullet in bullets])

total_bullets_700 = sorted(total_bullets_700)
vectorizer_700 = TfidfVectorizer(stop_words='english')
tfidf_matrix_700 = vectorizer_700.fit_transform(total_bullets_700)
shrunk_norm_matrix_700 = shrink_matrix(tfidf_matrix_700)
print(f"We've vectorized {shrunk_norm_matrix_700.shape[0]} bullets")
```

```python id="mR1agt1ELzo1" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1637478393830, "user_tz": -330, "elapsed": 17815, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="c6635f50-4306-4b8b-91ed-f6294f5f89a6"
np.random.seed(0)
generate_elbow_plot(shrunk_norm_matrix_700)
plt.show()
```

```python id="kj6D9nnaLzo5" colab={"base_uri": "https://localhost:8080/", "height": 656} executionInfo={"status": "ok", "timestamp": 1637478399345, "user_tz": -330, "elapsed": 5534, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="80a9ee73-6562-4aec-a7dd-c50727494b1b"
np.random.seed(0)
cluster_groups_700 = compute_cluster_groups(shrunk_norm_matrix_700, k=20,
                                            bullets=total_bullets_700)
bullet_cosine_similarities = compute_bullet_similarity(total_bullets_700)
sorted_cluster_groups_700 = sort_cluster_groups(cluster_groups_700)
plot_wordcloud_grid(sorted_cluster_groups_700, num_rows=4, num_columns=5,
                    vectorizer=vectorizer_700,
                    tfidf_matrix=tfidf_matrix_700)
```

<!-- #region id="ADxBKJj9UdPT" -->
---
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="xiJyIAByUdLS" executionInfo={"status": "ok", "timestamp": 1637478711279, "user_tz": -330, "elapsed": 2797, "user": {"displayName": "Sparsh Agarwal", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "13037694610922482904"}} outputId="24bbc250-50b3-4439-f90a-fd9cbf221efe"
!pip install -q watermark
%reload_ext watermark
%watermark -a "Sparsh A." -m -iv -u -t -d
```

<!-- #region id="BuZA9e02VnHh" -->
---
<!-- #endregion -->
