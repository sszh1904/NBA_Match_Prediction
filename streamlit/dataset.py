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
    st.title("Dataset :basketball:")

    # Description
    st.write('''
    Data is retrieved from an API client for [www.nba.com](https://www.nba.com). More info about this API can be found in the [github](https://github.com/swar/nba_api). 
    * Data is collected from NBA Season 2014-15 to 2020-21 for model training. 
    * Each record is a team's statistics for the game played. 
    ''')

    # Choose data for a specific season
    st.subheader('Choose a season to view the dataset')
    st.write("* Data shown only includes regular season data")
    season_years = list(NBA_SEASONS.keys())
    season = st.selectbox('', season_years)

    @st.cache
    def read_season_df(season):
        all_data = pd.read_csv('data/annual_data/annual_nba_data.csv')
        season_df = all_data[all_data['GAME_DATE'].between(NBA_SEASONS[season]['start_date'], NBA_SEASONS[season]['end_date'])]
        return season_df

    season_df = read_season_df(season)
    st.dataframe(season_df)

    # Separation
    st.markdown('''<br>''', True)

    # Data Dictionary
    st.subheader("Data Dictionary")
    st.write("* Description of data variables in the dataset above")
    data_dict = pd.read_csv('data/annual_data/data_dict.csv')
    st.table(data_dict)