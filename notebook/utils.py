import json
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder

team_stats = {}

def get_nba_teams():
    print("Retrieving NBA teams...")
    nba_teams = pd.DataFrame(teams.get_teams())
    nba_team_abbr = nba_teams['abbreviation'].tolist()
    print("Succesfully retrieved NBA teams.")
    return nba_team_abbr

def populate_team_stats(nba_teams):
    print("Populating team stats...")
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
            "OFFRATE": 0,
            "DEFRATE": 0,
            "ELO": 1500
        }
    print("Successfully populated team stats.")
    return

def create_team_stats_json():
    print("Creating json file,,,")
    nba_teams = get_nba_teams()
    populate_team_stats(nba_teams)
    with open('data/team_stats.json', 'w') as json_file:
        json.dump(team_stats, json_file, indent=4)
    print("Successfully created json file.")
    return

def create_upcoming_games_csv():
    print("Creating csv file...")
    df = pd.DataFrame(columns=["HOME_TEAM", "AVG_PTS_x", "AVG_AST_x", "AVG_OREB_x", "AVG_DREB_x", "OFFRATE_x", "DEFRATE_x", "ELO_x", "AWAY_TEAM", "AVG_PTS_y", "AVG_AST_y", "AVG_OREB_y", "AVG_DREB_y", "OFFRATE_y", "DEFRATE_y", "ELO_y", "DIS_PTS", "DIS_AST", "DIS_OREB", "DIS_DREB", "DIS_OFFRATE", "DIS_DEFRATE", "DIS_ELO"])
    df.reset_index(drop=True,inplace=True)
    df.to_csv("data/upcoming_games.csv", index=False)
    print("Successfully created csv file.")
    return

def extract_nba_api():
    print("Extracting nba api data...")
    nba_teams = pd.DataFrame(teams.get_teams())
    team_ids = nba_teams['id'].unique()
    
    main_df = pd.DataFrame()
    for team_id in team_ids:
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
        games = gamefinder.get_data_frames()[0]
        games = games[(games['GAME_DATE'] >= '2020-12-22') & (games['WL'].isnull() == False)]
        main_df = main_df.append(games)
    
    main_df = main_df.sort_values('GAME_DATE',ascending=False)
    print("Successfully extracted nba api data.")
    return main_df

def clean_api_data(df):
    print("Cleaning nba api data...")
    df['length'] = df['MATCHUP'].str.len()
    df.sort_values('length', inplace=True)
    df.drop(columns=['length'], inplace=True)
    
    df_combined = df.merge(df, on='GAME_ID')
    df_combined = df_combined.drop(df_combined[df_combined['TEAM_ID_x'] == df_combined['TEAM_ID_y']].index)
    df_combined = df_combined.iloc[1:].iloc[::2]
    df_combined.reset_index(drop=True,inplace=True)
    df_combined.drop(columns=["SEASON_ID_x", "TEAM_ID_x", "SEASON_ID_y", "TEAM_ID_y", "MATCHUP_x", "MATCHUP_y"],inplace=True)
    df_combined = df_combined.replace(['W','L'], [int(1), int(0)]) # win = 1, lose = 0
    
    print("Successfully cleaned data.")
    return df_combined

def create_season_history_csv():
    print("Creating season history csv file...")
    # df = extract_nba_api()
    df = pd.read_csv("data/nba_api_raw.csv")
    print(df.head())
    df = clean_api_data(df)
    print(df.head())
    df['GAME_NO_x'] = 0
    df['GAME_NO_y'] = 0
    df['DIS_PTS'] = 0
    df['DIS_AST'] = 0
    df['DIS_OREB'] = 0
    df['DIS_DREB'] = 0
    df['DIS_OFFRATE'] = 0
    df['DIS_DEFRATE'] = 0
    df['DIS_ELO'] = 0
    
    df['GAME_DATE_x'] = pd.to_datetime(df['GAME_DATE_x']) # change GAME_DATE to datetime type
    df = df.sort_values(by = "GAME_DATE_x", ascending = True)
    
    with open("data/team_stats.json", 'r') as jsonFile:
        nba_teams = json.load(jsonFile)
    
    for i, row in df.iterrows():
    #         get the name of both teams
        team_1 = row['TEAM_ABBREVIATION_x']
        team_2 = row['TEAM_ABBREVIATION_y']

    #         add pre-game stats to row
        nba_teams[team_1]['GAME_NO'] += 1
        nba_teams[team_2]['GAME_NO'] += 1
        df.loc[i,'GAME_NO_x'] = nba_teams[team_1]['GAME_NO']
        df.loc[i,'GAME_NO_y'] = nba_teams[team_2]['GAME_NO']
        
        df.loc[i,'AVG_PTS_x'] = nba_teams[team_1]['AVG_PTS']
        df.loc[i,'AVG_PTS_y'] = nba_teams[team_2]['AVG_PTS']
        df.loc[i,'AVG_AST_x'] = nba_teams[team_1]['AVG_AST']
        df.loc[i,'AVG_AST_y'] = nba_teams[team_2]['AVG_AST']
        df.loc[i,'AVG_OREB_x'] = nba_teams[team_1]['AVG_OREB']
        df.loc[i,'AVG_OREB_y'] = nba_teams[team_2]['AVG_OREB']
        df.loc[i,'AVG_DREB_x'] = nba_teams[team_1]['AVG_DREB']
        df.loc[i,'AVG_DREB_y'] = nba_teams[team_2]['AVG_DREB']
        df.loc[i,'OFFRATE_x'] = nba_teams[team_1]['OFFRATE']
        df.loc[i,'OFFRATE_y'] = nba_teams[team_2]['OFFRATE']
        df.loc[i,'DEFRATE_x'] = nba_teams[team_1]['DEFRATE']
        df.loc[i,'DEFRATE_y'] = nba_teams[team_2]['DEFRATE']
        df.loc[i,'ELO_x'] = nba_teams[team_1]['ELO']
        df.loc[i,'ELO_y'] = nba_teams[team_2]['ELO']
        
        df.loc[i,'DIS_PTS'] = nba_teams[team_1]['AVG_PTS'] - nba_teams[team_2]['AVG_PTS']
        df.loc[i,'DIS_AST'] = nba_teams[team_1]['AVG_AST'] - nba_teams[team_2]['AVG_AST']
        df.loc[i,'DIS_OREB'] = nba_teams[team_1]['AVG_OREB'] - nba_teams[team_2]['AVG_OREB']
        df.loc[i,'DIS_DREB'] = nba_teams[team_1]['AVG_DREB'] - nba_teams[team_2]['AVG_DREB']
        df.loc[i,'DIS_OFFRATE'] = nba_teams[team_1]['OFFRATE'] - nba_teams[team_2]['OFFRATE']
        df.loc[i,'DIS_DEFRATE'] = nba_teams[team_1]['DEFRATE'] - nba_teams[team_2]['DEFRATE']    
        df.loc[i,'DIS_ELO'] = nba_teams[team_1]['ELO'] - nba_teams[team_2]['ELO']    

    #       update stats of both teams
        nba_teams[team_1]['cPTS'] += row['PTS_x']
        nba_teams[team_1]['cAST'] += row['AST_x']
        nba_teams[team_1]['cOREB'] += row['OREB_x']
        nba_teams[team_1]['cDREB'] += row['DREB_x']
        nba_teams[team_1]['cFGA'] += row['FGA_x']
        nba_teams[team_1]['cTO'] += row['TOV_x']
        nba_teams[team_1]['cFTA'] += row['FTA_x']
        
        nba_teams[team_1]['AVG_PTS'] = nba_teams[team_1]['cPTS'] /nba_teams[team_1]["GAME_NO"]
        nba_teams[team_1]['AVG_AST'] = nba_teams[team_1]['cAST']/nba_teams[team_1]["GAME_NO"]
        nba_teams[team_1]['AVG_OREB'] = nba_teams[team_1]['cOREB']/nba_teams[team_1]["GAME_NO"]
        nba_teams[team_1]['AVG_DREB'] = nba_teams[team_1]['cDREB']/nba_teams[team_1]["GAME_NO"]
        
        nba_teams[team_2]['cPTS'] += row['PTS_y']
        nba_teams[team_2]['cAST'] += row['AST_y']
        nba_teams[team_2]['cOREB'] += row['OREB_y']
        nba_teams[team_2]['cDREB'] += row['DREB_y']
        nba_teams[team_2]['cFGA'] += row['FGA_y']
        nba_teams[team_2]['cTO'] += row['TOV_y']
        nba_teams[team_2]['cFTA'] += row['FTA_y']
        
        nba_teams[team_2]['AVG_PTS'] = nba_teams[team_2]['cPTS'] /nba_teams[team_2]["GAME_NO"]
        nba_teams[team_2]['AVG_AST'] = nba_teams[team_2]['cAST']/nba_teams[team_2]["GAME_NO"]
        nba_teams[team_2]['AVG_OREB'] = nba_teams[team_2]['cOREB']/nba_teams[team_2]["GAME_NO"]
        nba_teams[team_2]['AVG_DREB'] = nba_teams[team_2]['cDREB']/nba_teams[team_2]["GAME_NO"]

    #       update OFF DEF ratings of both teams
        tot_pos_1 = nba_teams[team_1]['cFGA'] - nba_teams[team_1]['cOREB'] + nba_teams[team_1]['cTO'] +(0.4* nba_teams[team_1]['cFTA'])
        off_ratings_1 = nba_teams[team_1]['cPTS']/tot_pos_1
        nba_teams[team_1]['OFFRATE'] = off_ratings_1
        def_ratings_1 = nba_teams[team_2]['cPTS']/tot_pos_1
        nba_teams[team_1]['DEFRATE'] = def_ratings_1

        tot_pos_2 = nba_teams[team_2]['cFGA'] - nba_teams[team_2]['cOREB'] + nba_teams[team_2]['cTO'] +(0.4* nba_teams[team_2]['cFTA'])
        off_ratings_2 = nba_teams[team_2]['cPTS']/tot_pos_2
        nba_teams[team_2]['OFFRATE'] = off_ratings_2
        def_ratings_2 = nba_teams[team_1]['cPTS']/tot_pos_2
        nba_teams[team_2]['DEFRATE'] = def_ratings_2

    #       update ELO of both teams
        K_FACTOR = 20       # constant value for multiplier

        P_team = 1/(1 + 10 ** ((nba_teams[team_2]['ELO'] - nba_teams[team_1]['ELO'])/400))      # probability of team winning

        if row['WL_x'] == 1:
            elo_change = K_FACTOR * (1 - P_team)        # formula for change in elo if team 1 wins
        else:
            elo_change = K_FACTOR * (0 - P_team)        # formula for change in elo if team 1 loses

        nba_teams[team_1]['ELO'] += elo_change
        nba_teams[team_2]['ELO'] -= elo_change
        
    df.drop(df[(df['GAME_NO_x'] == 1) | (df['GAME_NO_y'] == 1 )].index, inplace=True) # omit first games of all teams
    
    with open("data/team_stats.json", 'w') as jsonFile:
        json.dump(nba_teams, jsonFile, indent=4)
    
    df.to_csv("data/season_history.csv", index=False)
    print("Successfully created season history csv file.")
    return


# ------------------------------------ DRIVERS ------------------------------------------------------------------------

# create_team_stats_json()
# create_upcoming_games_csv()
# create_season_history_csv()

stats = ["GAME_NO","AVG_PTS","AVG_AST","AVG_OREB","AVG_DREB","cPTS","cAST","cOREB","cDREB","cFGM","cFG3M","cFGA","cTO","cFTA","cPTS_ALLOWED","OFF_EFF","DEF_EFF","EFG","ELO"]

def populate_team_stats(nba_teams):
    team_stats ={}
    print("Populating team stats...")
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
            "cFGM": 0,
            "cFG3M": 0,
            "cFGA": 0,
            "cTO": 0,
            "cFTA": 0,
            "cPTS_ALLOWED": 0,
            "OFF_EFF": 0,
            "DEF_EFF": 0,
            "EFG": 0,
            "ELO": 1500
        }
    print("Successfully populated team stats.")
    return team_stats

def update_GAME_NO(team_stats, team_x, team_y):
    team_stats[team_x]['GAME_NO'] += 1
    team_stats[team_y]['GAME_NO'] += 1
    return team_stats[team_x]['GAME_NO'], team_stats[team_y]['GAME_NO']

def update_cPTS(team_stats, team_x, team_y, pts_x, pts_y):
    team_stats[team_x]['cPTS'] += pts_x
    team_stats[team_y]['cPTS'] += pts_y
    return team_stats[team_x]['cPTS'], team_stats[team_y]['cPTS']

def update_cAST(team_stats, team_x, team_y, ast_x, ast_y):
    team_stats[team_x]['cAST'] += ast_x
    team_stats[team_y]['cAST'] += ast_y
    return team_stats[team_x]['cAST'], team_stats[team_y]['cAST']

def update_cOREB(team_stats, team_x, team_y, oreb_x, oreb_y):
    team_stats[team_x]['cOREB'] += oreb_x
    team_stats[team_y]['cOREB'] += oreb_y
    return team_stats[team_x]['cOREB'], team_stats[team_y]['cOREB']

def update_cDREB(team_stats, team_x, team_y, dreb_x, dreb_y):
    team_stats[team_x]['cDREB'] += dreb_x
    team_stats[team_y]['cDREB'] += dreb_y
    return team_stats[team_x]['cDREB'], team_stats[team_y]['cDREB']

def update_cFGM(team_stats, team_x, team_y, fgm_x, fgm_y):
    team_stats[team_x]['cFGM'] += fgm_x
    team_stats[team_y]['cFGM'] += fgm_y
    return team_stats[team_x]['cFGM'], team_stats[team_y]['cFGM']

def update_cFG3M(team_stats, team_x, team_y, fg3m_x, fg3m_y):
    team_stats[team_x]['cFG3M'] += fg3m_x
    team_stats[team_y]['cFG3M'] += fg3m_y
    return team_stats[team_x]['cFG3M'], team_stats[team_y]['cFG3M']

def update_cFGA(team_stats, team_x, team_y, fga_x, fga_y):
    team_stats[team_x]['cFGA'] += fga_x
    team_stats[team_y]['cFGA'] += fga_y
    return team_stats[team_x]['cFGA'], team_stats[team_y]['cFGA']

def update_cTO(team_stats, team_x, team_y, to_x, to_y):
    team_stats[team_x]['cTO'] += to_x
    team_stats[team_y]['cTO'] += to_y
    return team_stats[team_x]['cTO'], team_stats[team_y]['cTO']

def update_cFTA(team_stats, team_x, team_y, fta_x, fta_y):
    team_stats[team_x]['cFTA'] += fta_x
    team_stats[team_y]['cFTA'] += fta_y
    return team_stats[team_x]['cFTA'], team_stats[team_y]['cFTA']

def update_cPTS_ALLOWED(team_stats, team_x, team_y, pts_y, pts_x):
    team_stats[team_x]['cPTS_ALLOWED'] += pts_y
    team_stats[team_y]['cPTS_ALLOWED'] += pts_x
    return team_stats[team_x]['cPTS_ALLOWED'], team_stats[team_y]['cPTS_ALLOWED']

def update_AVG_PTS(team_stats, team_x, team_y, GAME_NO_x, GAME_NO_y, cPTS_x, cPTS_y):
    dis = update_DIS_PTS(team_stats, team_x, team_y)
    team_stats[team_x]['AVG_PTS'] = round(cPTS_x / GAME_NO_x, 2)
    team_stats[team_y]['AVG_PTS'] = round(cPTS_y / GAME_NO_y, 2)
    return dis, team_stats[team_x]['AVG_PTS'], team_stats[team_y]['AVG_PTS']

def update_AVG_AST(team_stats, team_x, team_y, GAME_NO_x, GAME_NO_y, cAST_x, cAST_y):
    dis = update_DIS_AST(team_stats, team_x, team_y)
    team_stats[team_x]['AVG_AST'] = round(cAST_x / GAME_NO_x, 2)
    team_stats[team_y]['AVG_AST'] = round(cAST_y / GAME_NO_y, 2)
    return dis, team_stats[team_x]['AVG_AST'], team_stats[team_y]['AVG_AST']

def update_AVG_OREB(team_stats, team_x, team_y, GAME_NO_x, GAME_NO_y, cOREB_x, cOREB_y):
    dis = update_DIS_OREB(team_stats, team_x, team_y)
    team_stats[team_x]['AVG_OREB'] = round(cOREB_x/ GAME_NO_x, 2)
    team_stats[team_y]['AVG_OREB'] = round(cOREB_y/ GAME_NO_y, 2)
    return dis, team_stats[team_x]['AVG_OREB'], team_stats[team_y]['AVG_OREB']

def update_AVG_DREB(team_stats, team_x, team_y, GAME_NO_x, GAME_NO_y, cDREB_x, cDREB_y):
    dis = update_DIS_DREB(team_stats, team_x, team_y)
    team_stats[team_x]['AVG_DREB'] = round(cDREB_x / GAME_NO_x, 2)
    team_stats[team_y]['AVG_DREB'] = round(cDREB_y / GAME_NO_y, 2)
    return dis, team_stats[team_x]['AVG_DREB'], team_stats[team_y]['AVG_DREB']

def update_OFF_EFF(team_stats, team_x, team_y, cPTS_x, cPTS_y, cFGA_x, cFGA_y, cOREB_x, cOREB_y, cTO_x, cTO_y, cFTA_x, cFTA_y):
    dis = update_DIS_OFF_EFF(team_stats, team_x, team_y)
    team_stats[team_x]['OFF_EFF'] = round(cPTS_x / (cFGA_x - cOREB_x + cTO_x + (0.4 * cFTA_x)) * 100, 2)
    team_stats[team_y]['OFF_EFF'] = round(cPTS_y / (cFGA_y - cOREB_y + cTO_y + (0.4 * cFTA_y)) * 100, 2)
    return dis, team_stats[team_x]['OFF_EFF'], team_stats[team_y]['OFF_EFF']

def update_DEF_EFF(team_stats, team_x, team_y, cPTS_ALLOWED_x, cPTS_ALLOWED_y, cFGA_x, cFGA_y, cOREB_x, cOREB_y, cTO_x, cTO_y, cFTA_x, cFTA_y):
    dis = update_DIS_DEF_EFF(team_stats, team_x, team_y)
    team_stats[team_x]['DEF_EFF'] = round(cPTS_ALLOWED_x / (cFGA_x - cOREB_x + cTO_x + (0.4 * cFTA_x)) * 100, 2)
    team_stats[team_y]['DEF_EFF'] = round(cPTS_ALLOWED_y / (cFGA_y - cOREB_y + cTO_y + (0.4 * cFTA_y)) * 100, 2)
    return dis, team_stats[team_x]['DEF_EFF'], team_stats[team_y]['DEF_EFF']

def update_EFG(team_stats, team_x, team_y, cFGM_x, cFGM_y, cFG3M_x, cFG3M_y, cFGA_x, cFGA_y):
    team_stats[team_x]['EFG'] = round((cFGM_x + 0.5 * cFG3M_x) / cFGA_x, 2)
    team_stats[team_y]['EFG'] = round((cFGM_y + 0.5 * cFG3M_y) / cFGA_y, 2)
    return team_stats[team_x]['EFG'], team_stats[team_y]['EFG']

def update_ELO(team_stats, team_x, team_y, WL_x, K_FACTOR=20): 
    dis = update_DIS_ELO(team_stats, team_x, team_y)

    # K is constant value for multiplier, affects the sensitivity to recent games
    P_team = 1 / (1 + 10 ** ((team_stats[team_y]['ELO'] - team_stats[team_x]['ELO'])/400)) # probability of team winning
    if WL_x == 1:
        elo_change = round(K_FACTOR * (1 - P_team), 3) # formula for change in elo if team 1 wins
    else:
        elo_change = round(K_FACTOR * (0 - P_team), 3) # formula for change in elo if team 1 loses
    
    team_stats[team_x]['ELO'] += elo_change
    team_stats[team_y]['ELO'] -= elo_change

    return dis, team_stats[team_x]['ELO'], team_stats[team_y]['ELO']

def update_DIS_PTS(team_stats, team_x, team_y):
    dis = team_stats[team_x]['AVG_PTS'] - team_stats[team_y]['AVG_PTS']
    return dis

def update_DIS_AST(team_stats, team_x, team_y):
    dis = team_stats[team_x]['AVG_AST'] - team_stats[team_y]['AVG_AST']
    return dis

def update_DIS_OREB(team_stats, team_x, team_y):
    dis = team_stats[team_x]['AVG_OREB'] - team_stats[team_y]['AVG_OREB']
    return dis

def update_DIS_DREB(team_stats, team_x, team_y):
    dis = team_stats[team_x]['AVG_DREB'] - team_stats[team_y]['AVG_DREB']
    return dis

def update_DIS_OFF_EFF(team_stats, team_x, team_y):
    dis = team_stats[team_x]['OFF_EFF'] - team_stats[team_y]['OFF_EFF']
    return dis

def update_DIS_DEF_EFF(team_stats, team_x, team_y):
    dis = team_stats[team_x]['DEF_EFF'] - team_stats[team_y]['DEF_EFF']
    return dis

def update_DIS_ELO(team_stats, team_x, team_y):
    dis = team_stats[team_x]['ELO'] - team_stats[team_y]['ELO']
    return dis

def update_HOME_COURT(matchup):
    if '@' in matchup:
        return 0, 1
    return 1, 0