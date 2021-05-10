from bs4 import BeautifulSoup
import requests
import pandas as pd

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

def update_team_stats(team, game_result):
    """
    Update team stats based on retrieved game results.

    :return: True/False for succesful/unsuccessful update status 
    :rtype: boolean
    """
    
    return