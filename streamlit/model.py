import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import plotly.graph_objects as go
from utils import *

def app():
    # Page Title
    st.title("Model Training & Performance :basketball:")

    # Description
    st.write('''
    After the Data Preparation phase, we performed correlation & covariance check and features importance selection on the selected features listed on Data Preparation page. \n
    In the end, only 3 features are significant:
    * **Offensive Efficiency (OEff)**
    * **Defensive Efficiency (DEff)**
    * **Elo Rating (ELO)**

    **Interesting Fact: Home court advantage has also been factored in our model even though it is not one of the feature. On the Data Preparation page, disparity between teams were calculated by using Home Team's stats - Away Team's stats. As such, we are essentially modelling what is the disparity range that will more likely result in predicting the Home Team winning the game. 
    ''')

    st.write('''
    ### Model Training & Prediction:\n
    * Data will be split into respective NBA season
    * First half of the season will be used as training data, while second half used as testing data
    * Different classifiers were used to find the best performing one
    * Accuracy and F1 score were used to evaluate the model performance
    * Top 5 models were picked out to conduct majority voting 
    * Final prediction will be based on the majority voting
    ''')
    st.write('')
    st.write('''
    After many rounds of model testing, the top 5 classifiers identified are:
    * Logistic Regression
    * Naives Bayes
    * Support Vector Machine (Linear Kernel)
    * Support Vector Machine (RBF Kernal)
    * Linear Regression

    ** Linear Regresssion: Since this is a binary classification problem, we used linear regression model to predict the score difference between home and away team. If the score difference is > 0, it indicates that Home Team will win the game and vice versa.
    ''')

    # # Model Performance
    # st.header("Model Performance")

    # @st.cache
    # def model_performance():
    #     df = pd.read_csv('notebook/model_performance.csv')
    #     df.rename(columns={
    #         'Unnamed: 0': 'Season Year', 
    #         'Unnamed: 1': 'Classifier',
    #         'accuracy': 'Accuracy',
    #         'precision': 'Precision',
    #         'recall': 'Recall',
    #         'f1': 'F1'
    #     }, inplace=True)
    #     return df

    # model_performance_df = model_performance()
    # st.dataframe(model_performance_df)


    # Model Prediction
    st.header('Model Prediction & Performance')
    st.subheader('Choose a season to view prediction & performance')
    st.write("* Data shown only includes regular season data")
    season_years = list(NBA_SEASONS.keys())
    season = st.selectbox('', season_years)

    @st.cache
    def read_prediction(season):
        df = pd.read_csv(f'../data/prediction/s{season[:-3]}_history.csv')
        predictions = df.dropna(subset=['PREDICTION'])

        df.fillna('NA', inplace=True)
        df_final = df[['GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION_x', 'TEAM_ABBREVIATION_y', 'WL_x', 'PREDICTION']]
        df_final.rename(columns={
            'GAME_ID': 'Game ID',
            'GAME_DATE': 'Game Date',
            'TEAM_ABBREVIATION_x': 'Home Team',
            'TEAM_ABBREVIATION_y': 'Away Team',
            'WL_x': 'WL (Home)',
            'PREDICTION': 'Prediction (Home)'
        }, inplace=True)

        metrics = {}
        metrics['Accuracy'] = accuracy_score(predictions['WL_x'], predictions['PREDICTION'])
        metrics['Precision'] = precision_score(predictions['WL_x'], predictions['PREDICTION'])
        metrics['Recall'] = recall_score(predictions['WL_x'], predictions['PREDICTION'])
        metrics['F1'] = f1_score(predictions['WL_x'], predictions['PREDICTION'])
        evaluation = pd.DataFrame(metrics, index=['Score'])

        return df_final, evaluation

    df, evaluation = read_prediction(season)
    st.write('Performance')
    st.table(evaluation)
    st.write('Predictions')
    st.dataframe(df)


    # Compare with actual higher WL-record winning percentage
    st.header('Our Evaluation Benchmark')
    st.write('''
    Other than just looking at the different evaluation metric scores, we also compare our accuracy with the actual winning percentage of the team with higher Win-Lose record.

    Justification: 
    * The most straighforward method of predicting which team is going win in a match up will be to choose the team with the higher W-L record
    * If our model can outperform the above method, then we will consider our model a success

    Below is the comparison of the accuracy of our model versus choosing the higher W-L record team. 
    ''')

    # Plotly barchart
    comparison_df = pd.read_csv('../data/model_performance/higher_WL_comparison.csv')
    seasons = comparison_df['Season']
    y1 = comparison_df['Higher WL record winning % (> midseason)']
    y2 = comparison_df['Accuracy']

    fig = go.Figure(data=[
        go.Bar(name='Higher WL record (> midseason)', x=seasons, y=y1),
        go.Bar(name='Model Accuracy', x=seasons, y=y2)
    ])

    fig.update_layout(barmode='group', template='ggplot2', title='Model Accuracy vs Winning Percentage of Higher W-L record team', xaxis_title='Season', yaxis_title='Accuracy')
    st.plotly_chart(fig)

    # Conclusion
    st.subheader('Overall Result')
    st.write('''
    From the bar chart above, we can see that our model is performing better than just simply picking the team with higher W-L record to win the game! :smile:
    ''')