import streamlit as st
import pandas as pd

NBA_SEASONS = {
    '2014-15': {'start_date': '2014-10-28', 'end_date': '2015-04-15'},
    '2015-16': {'start_date': '2015-10-27', 'end_date': '2016-04-13'},
    '2016-17': {'start_date': '2016-10-25', 'end_date': '2017-04-12'},
    '2017-18': {'start_date': '2017-10-17', 'end_date': '2018-04-11'},
    '2018-19': {'start_date': '2018-10-16', 'end_date': '2019-04-10'},
    '2019-20': {'start_date': '2019-10-22', 'end_date': '2020-03-11'},
    '2020-21': {'start_date': '2020-12-22', 'end_date': '2021-05-16'}
}

def app():
    # Page Title
    st.title("Data Preparation :basketball:")
    st.write('''
    This page consists of 2 sections: 1) **Features Creation**,  2) **EDA**
    ''')

    # Section 1: Features Creation
    st.header("Features Creation")

    # Description
    st.markdown('''
    To train our prediction model, we created new features in our dataset using the original teams' statistics provided from the NBA API. Formula for each new features created will be shown below.\n
    Features created are as follows:
    * [**Offense Efficiency**](https://www.sportsrec.com/calculate-teams-offensive-defensive-efficiencies-7775395.html)
    * [**Defense Efficiency**](https://www.sportsrec.com/calculate-teams-offensive-defensive-efficiencies-7775395.html)
    * [**ELO**](https://www.geeksforgeeks.org/elo-rating-algorithm/)
    * **Stats Disparity Between Teams**
    <br>

    *Note: Team's cumulative stats are used to calculate each new feature.\n
    ** Visit _Dataset_ page for information on dataset variables.
    ''', True)

    # Section 1: Features Creation - Off & Def Eff
    st.subheader("Offensive Efficiency (OEff) & Defensive Efficiency (DEff)")
    st.latex(r'Total\_Possession = cFGA - cOREB + cTOV + (0.4 \times cFTA)')
    st.latex(r'OEff = \frac{cPts} {Total\_Possessions}')
    st.latex(r'DEff = \frac{cPts\_Allowed} {Total\_Possessions}')

    # Section 1: ELO
    st.subheader('ELO Calculation')
    st.write('''
    We drew inspirations from [FiveThirtyEight](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/) for ELO rating feature. 

    Every team will start with the same ELO score (eg. 1500) and their elo score will be adjusted according to the game result and their respective opponent for the game. 

    Each team's probability of winning the game will first be calculated, and the amount elo adjusted will be based on the _indiviual team's probability of winning_ that game and a constant _K_. 

    The higher the value of K, the more sensitive elo rating is to recent games. 
    ''')
    st.latex('Team \space 1 \space ELO: ELO1, \space\space\space Team \space 2 \space ELO: ELO2')
    st.latex('P1: \space Probability \space of \space Team \space 1 \space winning \space against \space Team \space 2')
    st.latex("P1 = \\frac {1} {1+10^ \\frac {ELO2-ELO1} {400}}")
    st.latex('P2: \space Probability \space of \space Team \space 2 \space winning \space against \space Team \space 1')
    st.latex("P2 = \\frac {1} {1+10^ \\frac {ELO1-ELO2} {400}}")

    st.write('If Team 1 wins:')
    st.latex(r'ELO1 = ELO1 + K \times (1 - P1)')
    st.latex(r'ELO2 = ELO2 + K \times (0 - P2)')

    st.write('If Team 2 wins:')
    st.latex(r'ELO1 = ELO1 + K \times (0 - P1)')
    st.latex(r'ELO2 = ELO2 + K \times (1 - P2)')

    # Section 1: Disparity
    st.subheader("Stats Disparity")
    st.write('''
    Offensive Efficiency, Defensive Efficiency and Elo Rating were calculated for each team after each game. However, instead of using these respective features by each team, we decided to calculate the disparity of the key basketball statistics for each team prior to the start of their next game to predict the outcome. 

    Disparity were calculated on these features:
    * **Points (PTS)**
    * **Assists (AST)**
    * **Offensive Rebounds (OREB)**
    * **Defensive Rebounds (DREB)**
    * **Offensive Efficiency (OEff)**
    * **Defensive Efficiency (DEff)**
    * **Elo Rating (ELO)**

    These disparity values for each feature will then be used in our model training.
    ''')
    st.write("")
    st.write("For each feature, the disparity formula is as follow:")
    st.latex(r'DIS = HTeam\_Stats - ATeam\_Stats')
    st.latex(r'Eg. \space\space DIS\_PTS = HTeam\_PTS - ATeam\_PTS')
    st.write("*HTeam = Home Team, ATeam = Away Team")


    # Section 2: EDA
    st.header("EDA")
    st.write('''
    Below shows the new dataframe with the new calculated features for each season.\n
    Since there are 2 records for the same game (each record displaying the statistic for a team in a game), we merge and extended the record to include statistics for both team playing the same game into 1 record.
    ''')

    # Show new dataframe
    st.subheader('Choose a season to view the dataset')
    st.write('''
    * Data shown only includes regular season data
    * Columns with '**_x**' refers to Home Team, '**_y**' refers to Away Team
    ''')
    season_years = list(NBA_SEASONS.keys())
    season = st.selectbox('', season_years, key='season')

    @st.cache
    def read_season_df(season):
        season_df = pd.read_csv(f'data/annual_data/season_2{season[:-3]}_data.csv')
        return season_df

    season_df = read_season_df(season)
    st.dataframe(season_df)
