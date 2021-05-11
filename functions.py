from bs4 import BeautifulSoup
import requests
import pandas as pd
import json

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score,roc_curve,auc,recall_score,f1_score,precision_score,classification_report,confusion_matrix,auc


def update_team_stats(team, game_result):
    """
    Update team stats based on retrieved game results.

    :return: True/False for succesful/unsuccessful update status 
    :rtype: boolean
    """
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
            if 'time' in game.thead.text:
                game_matchup = game.tbody
                game_date = soup.findAll('div', {'id':'sched-container'})[0].findAll('h2')[counter].text
        except AttributeError:
            continue
        counter += 1

        if game_date != '':
            break
    if game_matchup == '':
        game_date = 'No upcoming games.'
        return matchup_list,game_date
        
    teams_playing = game_matchup.findAll('a', {'class':'team-name'})

    # Not needed for our web app, but just filling it in here incase we need it
    time_playing = game_matchup.findAll('td', {'data-behavior':'date_time'})

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
    df["OFFRATE_x"] = [data[home_team]["OFFRATE"]]
    df["DEFRATE_x"] = [data[home_team]["DEFRATE"]]
    df["ELO_x"] = [data[home_team]["ELO"]]
    
    df["AWAY_TEAM"] = [away_team]
    df["AVG_PTS_y"] = [data[away_team]["AVG_PTS"]]
    df["AVG_AST_y"] = [data[away_team]["AVG_AST"]]
    df["AVG_OREB_y"] = [data[away_team]["AVG_OREB"]]
    df["AVG_DREB_y"] = [data[away_team]["AVG_DREB"]]
    df["OFFRATE_y"] = [data[away_team]["OFFRATE"]]
    df["DEFRATE_y"] = [data[away_team]["DEFRATE"]]
    df["ELO_y"] = [data[away_team]["ELO"]]
    
    df["DIS_PTS"] = [df["AVG_PTS_x"][0] - df["AVG_PTS_y"][0]]
    df["DIS_AST"] = [df["AVG_AST_x"][0] - df["AVG_AST_y"][0]]
    df["DIS_OREB"] = [df["AVG_OREB_x"][0] - df["AVG_OREB_y"][0]]
    df["DIS_DREB"] = [df["AVG_DREB_x"][0] - df["AVG_DREB_y"][0]]
    df["DIS_OFFRATE"] = [df["OFFRATE_x"][0] - df["OFFRATE_y"][0]]
    df["DIS_DEFRATE"] = [df["DEFRATE_x"][0] - df["DEFRATE_y"][0]]
    df["DIS_ELO"] = [df["ELO_x"][0] - df["ELO_y"][0]]
        
    return df

def predict(df, game_df):
    """
    Predict outcome of game for home team based on 8 features.

    :return: Predicted outcome - 1/0
    :rtype: integer
    """
    features_list = ['DIS_ELO', 'DIS_OFFRATE', 'DIS_DEFRATE', 'DIS_PTS', 'DIS_AST', 'DIS_OREB', 'DIS_DREB']
    target = 'WL_x'

    # Creating our independent and dependent variables
    x = df[features_list]
    y = df['PLUS_MINUS_x']

    model = sm.OLS(y,x)
    results = model.fit()

    features_list = []
    for i in range(len(x.keys())):
        if results.pvalues[i] <= 0.05:
            features_list.append(model.exog_names[i])
            
    models_dict = {
            'Linear Regression': LinearRegression(),
            'Logistic Regression':LogisticRegression(),
            'Naive Bayes':GaussianNB(),
            'SVM linear': svm.SVC(kernel='linear'),
            'SVM rbf': svm.SVC(kernel='rbf'),
    }

    prediction_data = {} # store prediction for each model 

    for model_name in models_dict:
        X_train = df[features_list]
        X_test = game_df[features_list]
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
    for game in matchups:
        away = game[0:3]
        home = game[15:18]
        game_df = get_team_stats(home, away)
        prediction = predict(df, game_df)
        game_df['Prediction'] = [prediction]
        store_game_df('data/upcoming_games.csv', game_df)
    return 

process_upcoming_games()