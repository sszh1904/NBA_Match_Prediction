from bs4 import BeautifulSoup
import requests
import pandas as pd
import json



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

def predict(game_df):
    """
    Predict outcome of game for home team based on 8 features.

    :return: Predicted outcome - 1/0
    :rtype: integer
    """
    
    return prediction

def store_game_df(file, game_df):
    """
    Store game_df in either "upcoming_games" or "games_history" depending on file parameter.

    :return: True/False for successful/unsuccessful storing status
    :rtype: boolean
    """
    df = pd.read_csv(file)
    print(df)
    df = pd.concat([df,game_df], ignore_index=True)
    print("file_df:",df)
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
    for game in matchups:
        away = game[0:3]
        home = game[15:18]
        game_df = get_team_stats(home, away)
        print(game_df)
        prediction = predict(game_inputs)
        game_df['Prediction'] = [prediction]
        store_game_df('data/upcoming_games.csv', game_df)
    return 

# print(get_team_stats('IND', 'GSW'))
# process_upcoming_games()