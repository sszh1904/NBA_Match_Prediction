import pandas as pd
import datetime
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder

from bs4 import BeautifulSoup
import requests
import json

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score,roc_curve,auc,recall_score,f1_score,precision_score,classification_report,confusion_matrix,auc

# Getting yesterday's games data, processing them, merging them with games prediction data, then adding it to "season_history.csv"
def extract_ytd_games():
    """
    Retrieve yesterday's games data (US timing) and clean it.

    :return: Yesterday's cleaned games data
    :rtype: df
    """
    yesterday = datetime.datetime.now() - datetime.timedelta(hours=36)  # based on US timezone
    ytd_date = yesterday.date().strftime("%Y-%m-%d")   # convert to string format'
    
    nba_teams = pd.DataFrame(teams.get_teams())
    team_ids = nba_teams['id'].unique()

    df = pd.DataFrame()
    for team_id in team_ids:
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
        games = gamefinder.get_data_frames()[0]
        games = games[games['GAME_DATE'] == ytd_date]
        df = df.append(games)
    
    df['length'] = df['MATCHUP'].str.len()
    df.sort_values('length', inplace=True)
    df.drop(columns=['length'], inplace=True)
    df_combined = df.merge(df, on='GAME_ID')
    df_combined = df_combined.drop(df_combined[df_combined['TEAM_ID_x'] == df_combined['TEAM_ID_y']].index)
    df_combined = df_combined.iloc[1:].iloc[::2]
    df_combined.reset_index(drop=True, inplace=True)
    df_combined.drop(columns=["SEASON_ID_x", "TEAM_ID_x", "SEASON_ID_y", "TEAM_ID_y", "MATCHUP_x", "MATCHUP_y"],inplace=True)
    df_combined = df_combined.replace(['W','L'], [int(1), int(0)]) # win = 1, lose = 0
    
    return df_combined

def update_team_stats(game_result, results_df):
    """
    Update team stats based on retrieved game results.

    :return: True/False for succesful/unsuccessful update status 
    :rtype: boolean
    """
    with open("data/team_stats.json", 'r') as jsonFile:
        nba_teams = json.load(jsonFile)
    
    team_x = game_result["TEAM_ABBREVIATION_x"]
    team_y = game_result["TEAM_ABBREVIATION_y"]
    
    nba_teams[team_x]['GAME_NO'] += 1
    nba_teams[team_x]['cPTS'] += game_result['PTS_x']
    nba_teams[team_x]['cAST'] += game_result['AST_x']
    nba_teams[team_x]['cOREB'] += game_result['OREB_x']
    nba_teams[team_x]['cDREB'] += game_result['DREB_x']
    nba_teams[team_x]['cFGA'] += game_result['FGA_x']
    nba_teams[team_x]['cTO'] += game_result['TOV_x']
    nba_teams[team_x]['cFTA'] += game_result['FTA_x']
    nba_teams[team_x]['cPTS_ALLOWED'] += game_result['PTS_y']

    nba_teams[team_x]['AVG_PTS'] = nba_teams[team_x]['cPTS'] /nba_teams[team_x]["GAME_NO"]
    nba_teams[team_x]['AVG_AST'] = nba_teams[team_x]['cAST']/nba_teams[team_x]["GAME_NO"]
    nba_teams[team_x]['AVG_OREB'] = nba_teams[team_x]['cOREB']/nba_teams[team_x]["GAME_NO"]
    nba_teams[team_x]['AVG_DREB'] = nba_teams[team_x]['cDREB']/nba_teams[team_x]["GAME_NO"]

    nba_teams[team_y]['GAME_NO'] += 1
    nba_teams[team_y]['cPTS'] += game_result['PTS_y']
    nba_teams[team_y]['cAST'] += game_result['AST_y']
    nba_teams[team_y]['cOREB'] += game_result['OREB_y']
    nba_teams[team_y]['cDREB'] += game_result['DREB_y']
    nba_teams[team_y]['cFGA'] += game_result['FGA_y']
    nba_teams[team_y]['cTO'] += game_result['TOV_y']
    nba_teams[team_y]['cFTA'] += game_result['FTA_y']
    nba_teams[team_y]['cPTS_ALLOWED'] += game_result['PTS_x']

    nba_teams[team_y]['AVG_PTS'] = nba_teams[team_y]['cPTS'] /nba_teams[team_y]["GAME_NO"]
    nba_teams[team_y]['AVG_AST'] = nba_teams[team_y]['cAST']/nba_teams[team_y]["GAME_NO"]
    nba_teams[team_y]['AVG_OREB'] = nba_teams[team_y]['cOREB']/nba_teams[team_y]["GAME_NO"]
    nba_teams[team_y]['AVG_DREB'] = nba_teams[team_y]['cDREB']/nba_teams[team_y]["GAME_NO"]

#       update OFF DEF ratings of both teams
    nba_teams[team_x]['OFF_EFF'] = round(nba_teams[team_x]["cPTS"] / (nba_teams[team_x]["cFGA"] - nba_teams[team_x]["cOREB"] + nba_teams[team_x]["cTO"] + (0.4 * nba_teams[team_x]["cFTA"])) * 100, 2)
    nba_teams[team_y]['OFF_EFF'] = round(nba_teams[team_y]["cPTS"] / (nba_teams[team_y]["cFGA"] - nba_teams[team_y]["cOREB"] + nba_teams[team_y]["cTO"] + (0.4 * nba_teams[team_y]["cFTA"])) * 100, 2)
    nba_teams[team_x]['DEF_EFF'] = round(nba_teams[team_x]["cPTS_ALLOWED"] / (nba_teams[team_x]["cFGA"] - nba_teams[team_x]["cOREB"] + nba_teams[team_x]["cTO"] + (0.4 * nba_teams[team_x]["cFTA"])) * 100, 2)
    nba_teams[team_y]['DEF_EFF'] = round(nba_teams[team_y]["cPTS_ALLOWED"] / (nba_teams[team_y]["cFGA"] - nba_teams[team_y]["cOREB"] + nba_teams[team_y]["cTO"] + (0.4 * nba_teams[team_y]["cFTA"])) * 100, 2)

#       update ELO of both teams
    K_FACTOR = 20       # constant value for multiplier

    P_team = 1/(1 + 10 ** ((nba_teams[team_y]['ELO'] - nba_teams[team_x]['ELO'])/400))      # probability of team winning

    if game_result['WL_x'] == 1:
        elo_change = K_FACTOR * (1 - P_team)        # formula for change in elo if team 1 wins
    else:
        elo_change = K_FACTOR * (0 - P_team)        # formula for change in elo if team 1 loses

    nba_teams[team_x]['ELO'] += elo_change
    nba_teams[team_y]['ELO'] -= elo_change
    
#       add game_no columns for both team
    results_df['GAME_NO_x'] = nba_teams[team_x]['GAME_NO']
    results_df['GAME_NO_y'] = nba_teams[team_y]['GAME_NO']
    return

def merge_prediction_results(results_df, predictions_df):
    """
    Merge game results data with game prediction data.

    :return: Merged game data
    :rtype: df
    """
    predictions_df.rename(columns = {"HOME_TEAM":"TEAM_ABBREVIATION_x"}, inplace=True)
    predictions_df.drop(columns = ["AWAY_TEAM", "GAME_DATE_y", "TEAM_NAME_x", "TEAM_NAME_y"], inplace=True)
    merged_df = pd.merge(results_df, predictions_df, on="TEAM_ABBREVIATION_x")
    merged_df.rename(columns = {"GAME_DATE_x": "GAME_DATE"}, inplace=True)
    return merged_df

def store_game_df(file, game_df):
    """
    Store game_df in either "upcoming_games" or "games_history" depending on file parameter.

    :return: True/False for successful/unsuccessful storing status
    :rtype: boolean
    """
    df = pd.read_csv(file)
    df = pd.concat([df,game_df], ignore_index=True)
    df.to_csv(file,index=False)
    return

def process_ytd_games():
    """
    Full processing of yesterday's games:
    1. Extract game data
    2. Update team stats
    3. Merge with prediction
    4. Add to season history csv file

    :return: True/False for successful/unsuccessful processing status
    :rtype: boolean
    """
    results_df = extract_ytd_games()
    for index, row in results_df.iterrows():
        update_team_stats(row, results_df)
    predictions_df = pd.read_csv("data/upcoming_games.csv")
    merged_df = merge_prediction_results(results_df, predictions_df)
    store_game_df("data/season_history.csv",merged_df)
    return

# Getting next day matchups, processing them and storing it in "upcoming_games.csv"
def get_matchups():
    """
    Retrieves all the matchups from ESPN website for next day games.
    The next day logic is implemented by checking if a gameday is found,
    if found, the for loop breaks.

    :return: a list containing matchup strings
    :rtype: list
    """
    matchup_list = []
    game_date = ''
    game_matchup = ''
    url = 'https://www.espn.com.sg/nba/fixtures'
    r = requests.get(url)
    soup = BeautifulSoup(r.text,parser='html.parser',features="lxml")
    game_containers = soup.findAll('table', {'class':'schedule has-team-logos align-left'})
    counter = 0
    for game in game_containers:
        try:
            if game.thead == None:
                return matchup_list, 'No upcoming games.'
            elif 'time' in game.thead.text:
                game_matchup = game.tbody
                game_date = soup.findAll('div', {'id':'sched-container'})[0].findAll('h2')[counter].text
        except AttributeError:
            continue
        counter += 1

        if game_date != '':
            break
  
    teams_playing = game_matchup.findAll('a', {'class':'team-name'})
    
    error_name = {
            "GS":"GSW",
            "SA":"SAS",
            "WSH":"WAS",
            "NO":"NOP",
            "UTAH":"UTA",
            "NY":"NYK"
        }
    
    for i in range(0,len(teams_playing),2):
        away = teams_playing[i].text.split()[-1]
        home = teams_playing[i+1].text.split()[-1]
        if away in error_name:
            away = error_name[away]
        if home in error_name:
            home = error_name[home]
        matchup_string = '{} (away) vs. {} (home)'.format(away, home)
        matchup_list.append(matchup_string)
    return matchup_list, game_date

def get_team_stats(home_team, away_team):
    """
    Retrieves both teams stats from "team_stats.json".
    Calculate disparities between teams stats.

    :return: match prediction inputs
    :rtype: df
    """
    with open("data/team_stats.json", "r") as jsonFile:
        data = json.load(jsonFile)

    df = pd.DataFrame()
    
    df["HOME_TEAM"] = [home_team]
    df["AVG_PTS_x"] = [data[home_team]["AVG_PTS"]]
    df["AVG_AST_x"] = [data[home_team]["AVG_AST"]]
    df["AVG_OREB_x"] = [data[home_team]["AVG_OREB"]]
    df["AVG_DREB_x"] = [data[home_team]["AVG_DREB"]]
    df["OFF_EFF_x"] = [data[home_team]["OFF_EFF"]]
    df["DEF_EFF_x"] = [data[home_team]["DEF_EFF"]]
    df["ELO_x"] = [data[home_team]["ELO"]]
    
    df["AWAY_TEAM"] = [away_team]
    df["AVG_PTS_y"] = [data[away_team]["AVG_PTS"]]
    df["AVG_AST_y"] = [data[away_team]["AVG_AST"]]
    df["AVG_OREB_y"] = [data[away_team]["AVG_OREB"]]
    df["AVG_DREB_y"] = [data[away_team]["AVG_DREB"]]
    df["OFF_EFF_y"] = [data[away_team]["OFF_EFF"]]
    df["DEF_EFF_y"] = [data[away_team]["DEF_EFF"]]
    df["ELO_y"] = [data[away_team]["ELO"]]
    
    df["DIS_PTS"] = [df["AVG_PTS_x"][0] - df["AVG_PTS_y"][0]]
    df["DIS_AST"] = [df["AVG_AST_x"][0] - df["AVG_AST_y"][0]]
    df["DIS_OREB"] = [df["AVG_OREB_x"][0] - df["AVG_OREB_y"][0]]
    df["DIS_DREB"] = [df["AVG_DREB_x"][0] - df["AVG_DREB_y"][0]]
    df["DIS_OFF_EFF"] = [df["OFF_EFF_x"][0] - df["OFF_EFF_y"][0]]
    df["DIS_DEF_EFF"] = [df["DEF_EFF_x"][0] - df["DEF_EFF_y"][0]]
    df["DIS_ELO"] = [df["ELO_x"][0] - df["ELO_y"][0]]
        
    return df

def predict(df, game_df):
    """
    Predict outcome of game for home team based on 8 features.

    :return: Predicted outcome - 1/0
    :rtype: integer
    """
    features_list = ['DIS_ELO', 'DIS_OFF_EFF', 'DIS_DEF_EFF']
            
    models_dict = {
            'Linear Regression': LinearRegression(),
            'Logistic Regression':LogisticRegression(),
            'Naive Bayes':GaussianNB(),
            'SVM linear': svm.SVC(kernel='linear'),
            'SVM rbf': svm.SVC(kernel='rbf'),
    }
    
    X_train = df[features_list]
    X_test = game_df[features_list]

    prediction_data = {} # store prediction for each model 

    for model_name in models_dict:
        y_train = df['WL_x']

        m = models_dict[model_name]

        if model_name == 'Linear Regression':
            y_train = df['PLUS_MINUS_x']

        m.fit(X_train, y_train)
        prediction = m.predict(X_test)

        if model_name == 'Linear Regression':
            if prediction[0] > 0:
                prediction[0] = 1
            else:
                prediction[0] = 0

        prediction_data[model_name] = prediction[0]

    final_prediction = 0
    for k, v in prediction_data.items():
        final_prediction += v

    final_prediction = round(final_prediction / 5)
    return final_prediction

def process_upcoming_games():
    """
    Full processing of upcoming games:
    1. Get matchups
    2. Get team stats
    3. Make predictions
    4. Store predictions

    :return: True/False for successful/unsuccessful processing status
    :rtype: boolean
    """
    matchups, game_date = get_matchups()
    df = pd.read_csv("data/season_history.csv")
    df.drop(df[(df['GAME_NO_x'] == 1) | (df['GAME_NO_y'] == 1 )].index, inplace=True) # omit first games of all teams

    for game in matchups:
        away = game[0:3]
        home = game[15:18]
        game_df = get_team_stats(home, away)
        prediction = predict(df, game_df)
        game_df['PREDICTION'] = [prediction]
        store_game_df('data/upcoming_games.csv', game_df)
    return 


# ------------------------------------ DRIVERS ------------------------------------------------------------------------
process_ytd_games()
# process_upcoming_games()