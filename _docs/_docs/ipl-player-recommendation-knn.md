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

<!-- #region id="2bKTNnLrM8zW" -->
# IPL Player Recommendation System
> Build a Player Recommendation System For Cricket Using K-Nearest Neighbor Algorithm

- toc: true
- badges: true
- comments: true
- categories: [KNN, Sports]
- image:
<!-- #endregion -->

<!-- #region id="_kAJs_cKJMN5" -->
<!-- #endregion -->

<!-- #region id="kjVRbTYQHajp" -->
We will build a cricket player recommendation system that will suggest a list of batsmen for the team based on the statistics of players that have been playing for the team in the past.
<!-- #endregion -->

<!-- #region id="7i4vOS_EKTbj" -->
## Setup
<!-- #endregion -->

```python id="CbBEwSrZIWBo"
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx

import warnings
warnings.filterwarnings('ignore')
```

```python id="PLPP9Ui0O-wC"
NUMBER_OF_SEASONS = 9
NUM_RECOMMENDATION = 15
```

<!-- #region id="uKuGGWZnJkgA" -->
## Data loading
<!-- #endregion -->

<!-- #region id="yNewKt4UJq7B" -->
<!-- #endregion -->

```python id="1A2gQ2oOJytD"
!pip install -q -U kaggle
!pip install --upgrade --force-reinstall --no-deps kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d harsha547/indian-premier-league-csv-dataset
```

```python id="SRNqzxP2J0gX"
!unzip indian-premier-league-csv-dataset.zip
```

```python id="Ri-q29mYKeuC"
def read_data():
  '''
  Function to read all the CSV files
  '''
  folder_path = '/content/'
  ball_by_ball = pd.read_csv(folder_path + 'Ball_by_Ball.csv')
  match = pd.read_csv(folder_path + 'Match.csv')
  player = pd.read_csv(folder_path + 'Player.csv')
  player_match = pd.read_csv(folder_path + 'Player_Match.csv')
  return ball_by_ball, match, player, player_match
```

```python id="w_UW3YJQFpwW"
ball_by_ball, match, player, player_match = read_data()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="OsaqVSE2Kwby" outputId="05894286-d012-43fc-e9e5-d1292cfc734f"
ball_by_ball.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 377} id="ZcXFcpGGKyeF" outputId="a6fd62e5-9696-4e59-e0d3-ba476905d803"
match.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="YGl4DMiGKyZY" outputId="3e6d7567-e904-41b9-bef2-98c3be7e62a5"
player.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="Ueu7d7brKyBH" outputId="3c436951-dd13-49e1-9af2-bd27f8120a78"
player_match.head()
```

<!-- #region id="bNgW5kXpK5hs" -->
## Preprocessing
<!-- #endregion -->

```python id="KTFzeo1gInau"
def season_match_association():
  '''
  Function to create a dataframe for range of Match_Id per season
  '''
  season_info = pd.DataFrame(columns = ['Season', 'Match_Id_start', 'Match_Id_end'])
  for season in range(1, NUMBER_OF_SEASONS + 1):
    match_info = match.loc[match['Season_Id'] == season]['Match_Id']
    season_info = season_info.append({
        'Season' : season,
        'Match_Id_start' : match_info.min(), 
        'Match_Id_end' : match_info.max()
    }, ignore_index=True)
  return season_info

def add_features():
  '''
  Function to add features for every season 
  that can be used to calculate Batting Statistics
  '''
  for season in range(1, NUMBER_OF_SEASONS + 1):
    player_data['Runs_Scored_Season_'+ str(season)] = 0
    player_data['Innings_Season_'+str(season)] = 0
    player_data['NotOut_Innings_Season_'+str(season)] = 0
    player_data['Balls_Faced_Season_'+str(season)] = 0
    player_data['No4_Season_'+str(season)] = 0
    player_data['No6_Season_'+str(season)] = 0

  prev_match_id = prev_innings_id = prev_striker_id = prev_non_striker_id = 0

  NUM_OF_BALLS = len(ball_by_ball)
  for i in range(NUM_OF_BALLS):
    data = ball_by_ball.loc[i]
    current_match_id = int(data['Match_Id'])
    current_innings_id = int(data['Innings_Id'])
    current_season = str(get_season_from_match_id(current_match_id))
    current_striker_id = data['Striker_Id']
    current_non_striker_id = data['Non_Striker_Id']
    player_dismissed = 0 if data['Player_dissimal_Id'] == ' ' else int(data['Player_dissimal_Id'])
    run_scored = 0 if data['Batsman_Scored'] == 'Do_nothing' or data['Batsman_Scored'] == ' ' else int(data['Batsman_Scored'])

    if run_scored == 4:
      # Striker scored a 4 
      player_data.loc[current_striker_id - 1, 'No4_Season_' + current_season] += 1

    elif run_scored == 6: 
      # Striker scored a 6 
      player_data.loc[current_striker_id - 1, 'No6_Season_' + current_season] += 1

    # Increase the runs scored and balls faced for the current striker
    player_data.loc[current_striker_id - 1, 'Runs_Scored_Season_' + current_season] += run_scored 
    player_data.loc[current_striker_id - 1, 'Balls_Faced_Season_' + current_season] += 1 

    #update innings
    if prev_match_id != current_match_id: 
      '''
      match id of the previous ball is different
      i.e current ball is of new match
      so new innings for striker and non striker both
      '''
      increment_innings(current_striker_id, current_season)
      increment_innings(current_non_striker_id, current_season)

    else: # match id is same
      if prev_innings_id != current_innings_id:
        '''
        match id of the previous ball is same
        but innings id has changed
        so new innings for striker and non striker both
        '''
        increment_innings(current_striker_id, current_season)
        increment_innings(current_non_striker_id, current_season)

      else:
        if current_striker_id != prev_striker_id and current_striker_id != prev_non_striker_id:
          '''
          current striker was not the striker or non striker
          on the previous ball
          so new innings for striker
          '''
          increment_innings(current_striker_id, current_season)

        if current_non_striker_id != prev_striker_id and current_non_striker_id != prev_non_striker_id:
          '''
          current non striker was not the striker or non striker
          on the previous ball
          so new innings for non striker
          '''
          increment_innings(current_non_striker_id, current_season)

    if player_dismissed != 0: 
      '''
      If player was dismissed here, 
      decrement the not out innings for that player
      '''
      player_data.loc[player_dismissed - 1, 'NotOut_Innings_Season_' + str(current_season)] -= 1

    # Update the prev_match_id, prev_innings_id, prev_striker_id and prev_non_striker_id
    prev_match_id = current_match_id
    prev_innings_id = current_innings_id
    prev_striker_id = current_striker_id
    prev_non_striker_id = current_non_striker_id

def get_season_from_match_id(match_id):
  '''
  Function to return the season according to match id
  '''
  season = season_info.loc[(season_info['Match_Id_start'] <= match_id) & 
                           (season_info['Match_Id_end'] >= match_id)] ['Season']
  # Return the integer value of the season else return -1 if season is not found   
  return season.item() if not season.empty else -1

def increment_innings(player_id, season):
  '''
  Function to increment innings for the player 
  '''
  player_data.loc[player_id - 1, 'Innings_Season_' + season] += 1
  player_data.loc[player_id - 1, 'NotOut_Innings_Season_' + season] += 1
```

```python id="LUb8i-JQIFyL"
player_data = player.drop(["Is_Umpire", "Unnamed: 7"], axis = 1)
season_info = season_match_association()
add_features()
```

```python id="tdAzgvS1LgFw"
def compute_statistics():
  '''
  Function to compute batsmen statistics like BA, BSR and BRPI
  '''
  player_data['DOB_Convert'] = pd.to_datetime(player_data['DOB'])
  player_metrics = player_data[player_data.columns[:5]]
  player_metrics['Age'] = 2021 - player_data['DOB_Convert'].dt.year
  player_metrics['Age'].fillna(player_metrics['Age'].median(), inplace=True)
  # Adjust age calculation where year is ambiguous
  player_metrics['Age'] = np.where(player_metrics['Age'] < 0, 1000 - player_metrics['Age'], player_metrics['Age'])
  X = []
  NUM_OF_PLAYERS = len(player_data)
  for i in range(NUM_OF_PLAYERS):
    data = player_data.loc[i]  
    player_metrics.loc[i, 'Domestic'] = 1 if data['Country'] == 'India' else 0
    age = player_metrics.loc[i, 'Age']
    if age < 41: # only calculate metrics for eligible players
      batting_avg_list = []
      batting_sr_list = []
      brpi_list = []
      season_count = 0
      for j in range(1, NUMBER_OF_SEASONS + 1):
        runs_scored = int(data['Runs_Scored_Season_'+str(j)])
        innings_played = int(data['Innings_Season_'+str(j)])
        innings_not_out = int(data['NotOut_Innings_Season_'+str(j)])
        balls_faced = int(data['Balls_Faced_Season_'+str(j)])
        num_fours = int(data['No4_Season_'+str(j)])
        num_sixes = int(data['No6_Season_'+str(j)])
        if innings_played > 0:
          season_count += 1 
          batting_strike_rate = 0 if balls_faced == 0 else runs_scored * 100/ balls_faced
          brpi = 0 if innings_played == 0 else (4 * num_fours   + 6 * num_sixes) / innings_played
          try:
            batting_avg = runs_scored / (innings_played - innings_not_out)
          except ArithmeticError:
            batting_avg = batting_strike_rate
          batting_avg_list.append(batting_avg)
          batting_sr_list.append(batting_strike_rate)
          brpi_list.append(brpi)
      player_metrics.loc[i, 'Batting_Average'] = sum(batting_avg_list)/float(season_count) if season_count > 0 else 0.0
      player_metrics.loc[i, 'Batting_Strike_Rate'] = sum(batting_sr_list)/float(season_count) if season_count > 0 else 0.0
      player_metrics.loc[i, 'BRPI'] = sum(brpi_list)/float(season_count) if season_count > 0 else 0.0
      X.append([player_metrics.loc[i, 'Batting_Average'], player_metrics.loc[i, 'Batting_Strike_Rate'], player_metrics.loc[i, 'BRPI'], player_metrics.loc[i, 'Age'],player_metrics.loc[i, 'Domestic']])
    elif age >= 41:
      player_metrics = player_metrics.drop([i], axis = 0)
    
  player_metrics = player_metrics.drop(["DOB","Bowling_Skill"], axis = 1)
  player_metrics = player_metrics.reset_index().drop(["index"], axis = 1)
  return player_metrics, X
```

```python id="CA8jzdPYEFk4"
player_metrics, X = compute_statistics()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="TN_SQFwA_lsN" outputId="3c70e91f-a496-4530-9bb3-e9bb4febc275"
player_metrics.head()
```

<!-- #region id="eEGx05WqMouf" -->
## Model fitting and Inference
<!-- #endregion -->

```python id="ZDxXvXYKL1p-"
def fetch_existing_players():
  '''
  Function to fetch players of current season for a particular team
  This has been programmed to fetch the players for Mumbai Indians only
  The team code for MI is 7
  '''
  # Fetch all matches in season
  matches_in_season = list(match.loc[match['Season_Id'] == 9]["Match_Id"])
  # Fetch all players who played in the season
  all_players = player_match.loc[player_match['Match_Id'].isin(matches_in_season)]
  # Filter players by team
  players = list(set(all_players.loc[all_players['Team_Id'] == 7]["Player_Id"]))
  return players

def sort_by_strike_rate(player_name):
  '''
  Custom sorting key to sort by strike rate
  '''
  return list(player_metrics.loc[player_metrics['Player_Name'] == player_name]['Batting_Strike_Rate'])[0]
```

```python id="FQrIZT59Op6z" colab={"base_uri": "https://localhost:8080/", "height": 548} outputId="64045010-8088-4099-8115-70775769ae36"
X = np.array(X)
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
nbrs.kneighbors_graph(X).toarray()

current_players = fetch_existing_players()

recommendations = set()
for current_p_id in current_players:
  current_player = player_metrics.loc[player_metrics['Player_Id'] == current_p_id]
  if current_player.empty:
    continue
  player_index = list(current_player.index)[0]
  neighbors = indices[player_index]
  for recomm in neighbors:
    recommendations.add(list(player_metrics.iloc[[recomm]]['Player_Name'])[0])
  
sorted_recommendations = sorted(recommendations, key = sort_by_strike_rate, reverse = True)
result = pd.DataFrame(columns = player_metrics.columns)
for player in sorted_recommendations[:NUM_RECOMMENDATION]:
  details = player_metrics.loc[player_metrics['Player_Name'] == player]
  result = result.append(details, ignore_index=True)
print("-"*35,"Batsmen Recommendation for Team","-"*35,"\n")
result.head(NUM_RECOMMENDATION)
```

<!-- #region id="9TjBVnArNSQE" -->
## Graph visualization
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wE6BB7VPGiy2" outputId="5b91bc92-9479-45ca-87b2-5b88e8908634"
adj_mat = nbrs.kneighbors_graph(X).toarray()
adj_mat
```

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="3lFOAs1xHpsU" outputId="5ee8ed8f-999b-416a-d275-c2abc2ffe9fd"
adj_mat = adj_mat[:5]
rows, cols = np.where(adj_mat == 1)
edges = zip(rows.tolist(), cols.tolist())
gr = nx.Graph()
gr.add_edges_from(edges)
nx.draw(gr, node_size=300, with_labels=True)
plt.show()
```
