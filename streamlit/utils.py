from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

NBA_SEASONS = {
    '2014-15': {'start_date': '2014-10-28', 'end_date': '2015-04-15'},
    '2015-16': {'start_date': '2015-10-27', 'end_date': '2016-04-13'},
    '2016-17': {'start_date': '2016-10-25', 'end_date': '2017-04-12'},
    '2017-18': {'start_date': '2017-10-17', 'end_date': '2018-04-11'},
    '2018-19': {'start_date': '2018-10-16', 'end_date': '2019-04-10'},
    '2019-20': {'start_date': '2019-10-22', 'end_date': '2020-03-11'},
    '2020-21': {'start_date': '2020-12-22', 'end_date': '2021-05-16'}
}

NBA_TEAMS = ['Atlanta Hawks (ATL)','Boston Celtics (BOS)','Brooklyn Nets (BKN)','Charlotte Hornets (CHA)','Chicago Bulls (CHI)','Cleveland Cavaliers (CLE)','Dallas Mavericks (DAL)','Denver Nuggets (DEN)','Detroit Pistons (DET)','Golden State Warriors (GSW)','Houston Rockets (HOU)','Indiana Pacers (IND)','Los Angeles Clippers (LAC)','Los Angeles Lakers (LAL)','Memphis Grizzlies (MEM)','Miami Heat (MIA)','Milwaukee Bucks (MIL)','Minnesota Timberwolves (MIN)','New Orleans Pelicans (NOP)','New York Knicks (NYK)','Oklahoma City Thunder (OKC)','Orlando Magic (ORL)','Philadelphia 76ers (PHI)','Phoenix Suns (PHX)','Portland Trail Blazers (POR)','Sacramento Kings (SAC)','San Antonio Spurs (SAS)','Toronto Raptors (TOR)','Utah Jazz (UTA)','Washington Wizards (WAS)']

K_FACTOR = 20
PRED_MODELS = {
            'Linear Regression': LinearRegression(),
            'Logistic Regression':LogisticRegression(),
            'Naive Bayes':GaussianNB(),
            'SVM linear': svm.SVC(kernel='linear'),
            'SVM rbf': svm.SVC(kernel='rbf'),
    }

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
