import json
import pandas as pd
from nba_api.stats.static import teams

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
            "GAME_PLAYED": 0,
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

def create_json_file():
    print("Creating json file,,,")
    nba_teams = get_nba_teams()
    populate_team_stats(nba_teams)
    with open('team_stats.json', 'w') as json_file:
        json.dump(team_stats, json_file, indent=4)
    print("Successfully created json file.")
    return

create_json_file()