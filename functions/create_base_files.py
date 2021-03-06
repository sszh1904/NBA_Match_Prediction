import json
import pandas as pd
import numpy as np
import datetime
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

S2020_START_DATE = datetime.datetime(2020, 12, 22)
NBA_SEASONS = {
    '2014-15': {'start_date': '2014-10-28', 'end_date': '2015-04-15'},
    '2015-16': {'start_date': '2015-10-27', 'end_date': '2016-04-13'},
    '2016-17': {'start_date': '2016-10-25', 'end_date': '2017-04-12'},
    '2017-18': {'start_date': '2017-10-17', 'end_date': '2018-04-11'},
    '2018-19': {'start_date': '2018-10-16', 'end_date': '2019-04-10'},
    '2019-20': {'start_date': '2019-10-22', 'end_date': '2020-03-11'},
    '2020-21': {'start_date': '2020-12-22', 'end_date': '2021-05-16'}
}
K_FACTOR = 20
PRED_MODELS = {
            'Linear Regression': LinearRegression(),
            'Logistic Regression':LogisticRegression(),
            'Naive Bayes':GaussianNB(),
            'SVM linear': svm.SVC(kernel='linear'),
            'SVM rbf': svm.SVC(kernel='rbf'),
    }

def get_nba_teams():
    print("Retrieving NBA teams...")
    nba_teams = pd.DataFrame(teams.get_teams())
    nba_team_abbr = nba_teams['abbreviation'].tolist()
    print("Succesfully retrieved NBA teams.")
    return nba_team_abbr

def populate_team_stats(nba_teams):
    print("Populating team stats...")
    team_stats = {}
    for team in nba_teams:
        team_stats[team] = {
            "GAME_NO": 0,
            "AVG_PTS": 0,
            "AVG_AST": 0,
            "AVG_OREB": 0,
            "AVG_DREB": 0,
            "cPTS": 0,
            "cAST": 0,
            "cOREB": 0,
            "cDREB": 0,
            "cFGA": 0,
            "cTO": 0,
            "cFTA":0,
            "cPTS_ALLOWED": 0,
            "OFF_EFF": 0,
            "DEF_EFF": 0,
            "ELO": 1500
        }
    print("Successfully populated team stats.")
    return team_stats

def create_team_stats_json():
    print("Creating json file...")
    nba_teams = get_nba_teams()
    team_stats = populate_team_stats(nba_teams)
    with open('data/team_stats.json', 'w') as json_file:
        json.dump(team_stats, json_file, indent=4)
    print("Successfully created json file.")

def create_upcoming_games_csv():
    print("Creating csv file...")
    df = pd.DataFrame(columns=["HOME_TEAM", "AVG_PTS_x", "AVG_AST_x", "AVG_OREB_x", "AVG_DREB_x", "OFF_EFF_x", "DEF_EFF_x", "ELO_x", "AWAY_TEAM", "AVG_PTS_y", "AVG_AST_y", "AVG_OREB_y", "AVG_DREB_y", "OFF_EFF_y", "DEF_EFF_y", "ELO_y", "DIS_PTS", "DIS_AST", "DIS_OREB", "DIS_DREB", "DIS_OFF_EFF", "DIS_DEF_EFF", "DIS_ELO"])
    df.reset_index(drop=True, inplace=True)
    df.to_csv("data/upcoming_games.csv", index=False)
    print("Successfully created csv file.")
    return

def create_season_history_csv():
    print("Creating season history csv file...")
    # games = extract_nba_api()
    # games.to_csv('data/s2020_raw_data.csv', index=False)
    # games = pd.read_csv('data/s2020_raw_data.csv')
    games = pd.read_csv('data/annual_data/annual_nba_data.csv')
    games = games[games['GAME_DATE'].between(NBA_SEASONS['2014-15']['start_date'], NBA_SEASONS['2014-15']['end_date'])]
    games = clean_api_data(games)

    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'], format='%Y-%m-%d')
    # games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'], format='%d/%m/%y')
    games = games.sort_values(by = ["GAME_DATE", "GAME_ID"], ascending = True)
    games.reset_index(drop=True, inplace=True)

    game_date = start_date = games['GAME_DATE'].min()
    # mid_season = datetime.datetime(2021, 2, 23)
    mid_season = datetime.datetime(2015, 1, 18)
    end_date = games['GAME_DATE'].max()
    hist = pd.DataFrame()

    while game_date <= end_date:
        upcoming = games[games['GAME_DATE'] == game_date][["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION_x", "TEAM_ABBREVIATION_y"]]
        if len(upcoming) <= 0:
            game_date = game_date + datetime.timedelta(days=1)
            continue

        with open("data/team_stats.json", 'r') as jsonFile:
            team_stats = json.load(jsonFile)

        update_pregame_stats(team_stats, upcoming)
        calc_team_disparity(team_stats, upcoming)

        if game_date >= mid_season:
            upcoming['PREDICTION'] = predict_outcome(hist, upcoming)
        else:
            upcoming['PREDICTION'] = np.nan
        # prediction = []
        # for i, row in upcoming.iterrows():
        #     if (row['GAME_NO_x'] > 41) or (row['GAME_NO_y'] > 41):
        #         p = predict_outcome(hist, upcoming)
        #         prediction += p
        #     else:
        #         prediction.append(np.nan)
        
        post_game_stats = games[games['GAME_DATE'] == game_date]
        merged_df = pd.merge(upcoming, post_game_stats, on=["GAME_DATE", "TEAM_ABBREVIATION_x", "TEAM_ABBREVIATION_y"], how='left')
        hist = hist.append(merged_df, ignore_index=True)

        team_stats = update_team_stats(team_stats, post_game_stats)

        with open("data/team_stats.json", 'w') as jsonFile:
            json.dump(team_stats, jsonFile, indent=4)
        
        print(f'{game_date} done!')
        game_date = game_date + datetime.timedelta(days=1)

    hist = rearrange_columns(hist)
    # hist.to_csv("data/season_history.csv", index=False)
    hist.to_csv("data/prediction/s2014_history.csv", index=False)

def extract_nba_api():
    print("Extracting nba api data...")
    nba_teams = pd.DataFrame(teams.get_teams())
    team_ids = nba_teams['id'].unique()
    
    df = pd.DataFrame()
    for team_id in team_ids:
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
        games = gamefinder.get_data_frames()[0]
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        games = games[(games['GAME_DATE'] >= S2020_START_DATE) & (games['WL'].isnull() == False)]
        df = df.append(games)
    
    df = df.sort_values('GAME_DATE',ascending=False)
    df.reset_index(drop=True, inplace=True)
    print("Successfully extracted nba api data.")
    return df

def clean_api_data(df):
    print("Cleaning nba api data...")
    
    df_combined = df.merge(df, on='GAME_ID')
    df_combined = df_combined.drop(df_combined[df_combined['TEAM_ID_x'] == df_combined['TEAM_ID_y']].index)
    df_combined = df_combined.drop(df_combined[df_combined['MATCHUP_x'].str.contains("@")].index)
    df_combined.reset_index(drop=True, inplace=True)
    df_combined.insert(0, 'GAME_DATE', df_combined['GAME_DATE_x'])
    df_combined.drop(columns=["SEASON_ID_x", "TEAM_ID_x", "SEASON_ID_y", "TEAM_ID_y", "MATCHUP_x", "MATCHUP_y", "GAME_DATE_x", "GAME_DATE_y"], inplace=True)
    df_combined = df_combined.replace(['W','L'], [1, 0]) # win = 1, lose = 0
    
    print("Successfully cleaned data.")
    return df_combined

def update_pregame_stats(team_stats, upcoming):
    upcoming[['GAME_NO_x', 'AVG_PTS_x', 'AVG_AST_x', 'AVG_OREB_x', 'AVG_DREB_x', 'OFF_EFF_x', 'DEF_EFF_x', 'ELO_x']] = upcoming.apply(lambda x: add_stats(team_stats, x['TEAM_ABBREVIATION_x']), axis=1, result_type='expand')
    upcoming[['GAME_NO_y', 'AVG_PTS_y', 'AVG_AST_y', 'AVG_OREB_y', 'AVG_DREB_y', 'OFF_EFF_y', 'DEF_EFF_y', 'ELO_y']] = upcoming.apply(lambda x: add_stats(team_stats, x['TEAM_ABBREVIATION_y']), axis=1, result_type='expand')

def add_stats(team_stats, team):
    game_no = team_stats[team]['GAME_NO'] + 1
    avg_pts = team_stats[team]['AVG_PTS']
    avg_ast = team_stats[team]['AVG_AST']
    avg_oreb = team_stats[team]['AVG_OREB']
    avg_dreb = team_stats[team]['AVG_DREB']
    off_eff = team_stats[team]['OFF_EFF']
    def_eff = team_stats[team]['DEF_EFF']
    elo = team_stats[team]['ELO']
    return game_no, avg_pts, avg_ast, avg_oreb, avg_dreb, off_eff, def_eff, elo

def calc_team_disparity(team_stats, upcoming):
    upcoming[['DIS_PTS', 'DIS_AST', 'DIS_OREB', 'DIS_DREB', 'DIS_OFF_EFF', 'DIS_DEF_EFF', 'DIS_ELO']] = upcoming.apply(lambda x: calc_disparity(team_stats, x['TEAM_ABBREVIATION_x'], x['TEAM_ABBREVIATION_y']), axis=1, result_type='expand')

def calc_disparity(team_stats, team_x, team_y):
    dis_pts = team_stats[team_x]['AVG_PTS'] - team_stats[team_y]['AVG_PTS']
    dis_ast = team_stats[team_x]['AVG_AST'] - team_stats[team_y]['AVG_AST']
    dis_oreb = team_stats[team_x]['AVG_OREB'] - team_stats[team_y]['AVG_OREB']
    dis_dreb = team_stats[team_x]['AVG_DREB'] - team_stats[team_y]['AVG_DREB']
    dis_off_eff = team_stats[team_x]['OFF_EFF'] - team_stats[team_y]['OFF_EFF']
    dis_def_eff = team_stats[team_x]['DEF_EFF'] - team_stats[team_y]['DEF_EFF']
    dis_elo = team_stats[team_x]['ELO'] - team_stats[team_y]['ELO']
    return dis_pts, dis_ast, dis_oreb, dis_dreb, dis_off_eff, dis_def_eff, dis_elo

def predict_outcome(hist, upcoming):
    hist.drop(hist[(hist['GAME_NO_x'] == 1) | (hist['GAME_NO_y'] == 1 )].index, inplace=True)

    features_list = ['DIS_ELO', 'DIS_OFF_EFF', 'DIS_DEF_EFF']

    prediction_data = {} # store prediction for each model 

    for model_name in PRED_MODELS:
        X_train = hist[features_list]
        y_train = hist['WL_x']
        y_train_lm = hist['PLUS_MINUS_x']
        X_test = upcoming[features_list]
        
        m = PRED_MODELS[model_name]

        if model_name == 'Linear Regression':
            m.fit(X_train, y_train_lm)
            prediction = m.predict(X_test)
            prediction = [1 if p > 0 else 0 for p in prediction]
        else:
            m.fit(X_train, y_train)
            prediction = m.predict(X_test)

        prediction_data[model_name] = prediction

    pred_df = pd.DataFrame(prediction_data)
    pred_df['final_pred'] = round(pred_df.sum(axis=1) / len(PRED_MODELS))
    
    final_prediction = pred_df['final_pred'].tolist()
    return final_prediction

def update_team_stats(team_stats, post_game_stats):
    for i, row in post_game_stats.iterrows():
        team_x = row['TEAM_ABBREVIATION_x']
        team_y = row['TEAM_ABBREVIATION_y']

        team_stats[team_x]['GAME_NO'] += 1
        team_stats[team_x]['cPTS'] += row['PTS_x']
        team_stats[team_x]['cAST'] += row['AST_x']
        team_stats[team_x]['cOREB'] += row['OREB_x']
        team_stats[team_x]['cDREB'] += row['DREB_x']
        team_stats[team_x]['cFGA'] += row['FGA_x']
        team_stats[team_x]['cTO'] += row['TOV_x']
        team_stats[team_x]['cFTA'] += row['FTA_x']
        team_stats[team_x]['cPTS_ALLOWED'] += row['PTS_y']
        
        team_stats[team_x]['AVG_PTS'] = team_stats[team_x]['cPTS'] /team_stats[team_x]["GAME_NO"]
        team_stats[team_x]['AVG_AST'] = team_stats[team_x]['cAST']/team_stats[team_x]["GAME_NO"]
        team_stats[team_x]['AVG_OREB'] = team_stats[team_x]['cOREB']/team_stats[team_x]["GAME_NO"]
        team_stats[team_x]['AVG_DREB'] = team_stats[team_x]['cDREB']/team_stats[team_x]["GAME_NO"]
        
        team_stats[team_y]['GAME_NO'] += 1
        team_stats[team_y]['cPTS'] += row['PTS_y']
        team_stats[team_y]['cAST'] += row['AST_y']
        team_stats[team_y]['cOREB'] += row['OREB_y']
        team_stats[team_y]['cDREB'] += row['DREB_y']
        team_stats[team_y]['cFGA'] += row['FGA_y']
        team_stats[team_y]['cTO'] += row['TOV_y']
        team_stats[team_y]['cFTA'] += row['FTA_y']
        team_stats[team_y]['cPTS_ALLOWED'] += row['PTS_x']
        
        team_stats[team_y]['AVG_PTS'] = team_stats[team_y]['cPTS'] /team_stats[team_y]["GAME_NO"]
        team_stats[team_y]['AVG_AST'] = team_stats[team_y]['cAST']/team_stats[team_y]["GAME_NO"]
        team_stats[team_y]['AVG_OREB'] = team_stats[team_y]['cOREB']/team_stats[team_y]["GAME_NO"]
        team_stats[team_y]['AVG_DREB'] = team_stats[team_y]['cDREB']/team_stats[team_y]["GAME_NO"]

        team_stats[team_x]['OFF_EFF'] = round(team_stats[team_x]["cPTS"] / (team_stats[team_x]["cFGA"] - team_stats[team_x]["cOREB"] + team_stats[team_x]["cTO"] + (0.4 * team_stats[team_x]["cFTA"])) * 100, 4)
        team_stats[team_y]['OFF_EFF'] = round(team_stats[team_y]["cPTS"] / (team_stats[team_y]["cFGA"] - team_stats[team_y]["cOREB"] + team_stats[team_y]["cTO"] + (0.4 * team_stats[team_y]["cFTA"])) * 100, 4)
        team_stats[team_x]['DEF_EFF'] = round(team_stats[team_x]["cPTS_ALLOWED"] / (team_stats[team_x]["cFGA"] - team_stats[team_x]["cOREB"] + team_stats[team_x]["cTO"] + (0.4 * team_stats[team_x]["cFTA"])) * 100, 4)
        team_stats[team_y]['DEF_EFF'] = round(team_stats[team_y]["cPTS_ALLOWED"] / (team_stats[team_y]["cFGA"] - team_stats[team_y]["cOREB"] + team_stats[team_y]["cTO"] + (0.4 * team_stats[team_y]["cFTA"])) * 100, 4)

        P_team = 1/(1 + 10 ** ((team_stats[team_y]['ELO'] - team_stats[team_x]['ELO'])/400)) 

        if row['WL_x'] == 1:
            elo_change = round(K_FACTOR * (1 - P_team), 4) 
        else:
            elo_change = round(K_FACTOR * (0 - P_team), 4) 
            
        team_stats[team_x]['ELO'] += elo_change
        team_stats[team_y]['ELO'] -= elo_change
    return team_stats

def rearrange_columns(hist):
    hist = hist[['GAME_ID_x', 'GAME_DATE', 'TEAM_ABBREVIATION_x', 'GAME_NO_x', 'AVG_PTS_x', 'AVG_AST_x', 'AVG_OREB_x', 'AVG_DREB_x', 'OFF_EFF_x', 'DEF_EFF_x', 'ELO_x', 'TEAM_ABBREVIATION_y', 'GAME_NO_y', 'AVG_PTS_y', 'AVG_AST_y', 'AVG_OREB_y', 'AVG_DREB_y', 'OFF_EFF_y', 'DEF_EFF_y', 'ELO_y', 'DIS_PTS', 'DIS_AST', 'DIS_OREB', 'DIS_DREB', 'DIS_OFF_EFF', 'DIS_DEF_EFF', 'DIS_ELO', 'PREDICTION', 'WL_x', 'MIN_x', 'PTS_x', 'FGM_x', 'FGA_x', 'FG_PCT_x', 'FG3M_x', 'FG3A_x', 'FG3_PCT_x', 'FTM_x', 'FTA_x', 'FT_PCT_x', 'OREB_x', 'DREB_x', 'REB_x', 'AST_x', 'STL_x', 'BLK_x', 'TOV_x', 'PF_x', 'PLUS_MINUS_x', 'WL_y', 'MIN_y', 'PTS_y', 'FGM_y', 'FGA_y', 'FG_PCT_y', 'FG3M_y', 'FG3A_y', 'FG3_PCT_y', 'FTM_y', 'FTA_y', 'FT_PCT_y', 'OREB_y', 'DREB_y', 'REB_y', 'AST_y', 'STL_y', 'BLK_y', 'TOV_y', 'PF_y', 'PLUS_MINUS_y']]
    hist.rename(columns={'GAME_ID_x': 'GAME_ID'}, inplace=True)
    return hist  



# ------------------------------------ DRIVERS ------------------------------------------------------------------------

create_team_stats_json()
create_upcoming_games_csv()
create_season_history_csv()

# df = pd.read_csv("data/s2020_raw_data.csv")
# df.drop(columns="index", inplace=True)
# df.to_csv("data/s2020_raw_data.csv", index=False)