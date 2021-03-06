import streamlit as st
import pandas as pd
import datetime
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from streamlit_app import utils

def app():
    # Page Title
    st.title('Live Prediction! :basketball:') 

    # Description
    st.write('''
    ### Welcome to live prediction!
    This is where you can pick any NBA regular season game and use our model to predict the outcome of your selected game!

    #### How to use?
    * You need to know the start date and end date of a NBA regular season. This is so that our model can retrieve the data required to train our prediction model for that season.
    * Select the date where the game of your interest will be held
    * Select the game from the dropdown 
    * Click on "PREDICT!" to get our model's prediction

    #### Things to note
    * We can only predict outcome for games that are 16 hours ahead! Teams should have played at least 5 games in the season. 
    * Game date should be in Singapore Time
    * Start and end data of a NBA regular season needs to be correct to ensure the right data is retrieved
    * Prediction on games after the mid season is highly recommended as the model would produce significantly better results
    * There is a limit to the number of request to retrieve data from the API endpoint, thus you may have to wait for it to reset if you were to resubmit the game details. Hence, please ensure that the selected dates are correct! 

    ### **Disclaimer!!!**
    This project is for academic purposes. Betting using our prediction model is strongly discouraged. The team is NOT responsible for any loss!\n 
    **PLAY AT YOUR OWN RISK!!!**
    ''')
    st.write('')
    st.write('')

    # User Inputs
    st.header("Get Your Prediction Here!")
    input1, input2 = st.beta_columns(2)
    input3, input4, input5 = st.beta_columns(3)

    start_date = input1.date_input('Enter Season Start Date', key='start')
    end_date = input2.date_input('Enter Season End Date', key='end')
    game_date = input3.date_input('Enter Game Date', key='game')
    home_team = input4.selectbox('Select Home Team', utils.NBA_TEAMS, key='home')
    away_team = input5.selectbox('Select Away Team', utils.NBA_TEAMS, key='away')

    def get_api_data(start_date, end_date):
        nba_teams = pd.DataFrame(teams.get_teams())
        team_ids = nba_teams['id'].unique()
        
        df = pd.DataFrame()
        for team_id in team_ids:
            try:
                gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
            except:
                return df 

            games = gamefinder.get_data_frames()[0]
            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
            start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
            end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())
            games = games[(games['GAME_DATE'] >= start_date) & (games['GAME_DATE'] <= end_date) & (games['WL'].isnull() == False)]
            df = df.append(games)
        
        df = df.sort_values('GAME_DATE', ascending=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @st.cache
    def get_prediction(df, game_date, home_team, away_team):
        games = clean_api_data(df)
        games.to_csv('test2.csv')
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'], format='%Y-%m-%d')
        games = games.sort_values(by = ["GAME_DATE", "GAME_ID"], ascending = True)
        games.reset_index(drop=True, inplace=True)

        curr_date = games['GAME_DATE'].min()
        game_date = datetime.datetime.combine(game_date, datetime.datetime.min.time())
        end_date = games['GAME_DATE'].max()
        hist = pd.DataFrame()

        team_stats = {}
        for team in NBA_TEAMS:
            team_stats[team[-4:-1]] = {
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

        day = 1
        while curr_date <= end_date:
            upcoming = games[games['GAME_DATE'] == curr_date][["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION_x", "TEAM_ABBREVIATION_y"]]
            if len(upcoming) <= 0:
                curr_date = curr_date + datetime.timedelta(days=1)
                continue

            update_pregame_stats(team_stats, upcoming)
            calc_team_disparity(team_stats, upcoming)
            
            if day > 5:
                # print(hist)
                upcoming['PREDICTION'] = predict_outcome(hist, upcoming)

            post_game_stats = games[games['GAME_DATE'] == curr_date]
            merged_df = pd.merge(upcoming, post_game_stats, on=["GAME_DATE", "TEAM_ABBREVIATION_x", "TEAM_ABBREVIATION_y"], how='left')
            hist = hist.append(merged_df, ignore_index=True)

            team_stats = update_team_stats(team_stats, post_game_stats)

            if game_date == curr_date:
                prediction = hist[(hist['GAME_DATE'] == game_date) & (hist['TEAM_ABBREVIATION_x'] == home_team[-4:-1])]['PREDICTION']
                break
            
            curr_date = curr_date + datetime.timedelta(days=1)
            day += 1

        return int(prediction)


    # Predict Button
    if st.button('PREDICT!'):
        st.info("Please wait while get your prediction...")
        df = get_api_data(start_date, end_date)
        # df = pd.read_csv('test.csv')
        if (len(df) <= 0):
            st.error('Connection timed out. Please try again later!')
        else:
            prediction = get_prediction(df, game_date, home_team, away_team)
            if prediction == 1:
                st.success(f'{home_team} is predicted to win!')
            else:
                st.success(f'{away_team} is predicted to win!')
        

