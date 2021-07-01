# NBA-Match-Predictor

This project is part of Singapore Management University (SMU) Business Intelligence and Analytics (BIA) Club's Data Associate Program (DAP). 

During the program, the team tries to create a prediction model to predict the outcome of a NBA game during the regular season. 

---
### Our App
We have made an interactive web application using Streamlit! Visit our app [here](https://share.streamlit.io/cjianyu98/nba_prediction/main/streamlit-app.py) to explore more!

---
### Table of Contents
- [Description](#Description)
- [Dataset](#Dataset)
- [Data Preparation](#Data-preparation)
- [Model](#Model)
- [Code Navigation](#Code-Navigation)
- [Contributors](#contributors)

---
### Description
Being a group of basketball enthusiasts, the team is interested in prediciting the outcome of NBA games using Machine Learning techniques. We explored many different models made by others online to get inspirations and ideas. We know apply our own domain knowledge on top of the useful information we searched online and developed our own model. 

---
### Dataset
Data is retrieved from an NBA API client for [www.nba.com](https://www.nba.com). More information about the API can be found [here](https://github.com/swar/nba_api).

We retrieved data from NBA Season 2014-15 to 2020-21. 

[Back To The Top](#nba-match-predictor)

---
### Data Preparation
To train our prediction model, we created new features in our dataset using the original teams' statistics retrieved from the NBA API.

Features created:
1. Offensive Efficiency
2. Defensive Efficiency
3. ELO
4. Stats disparity between 2 teams (for every game before predicting)

More details on our feature engineering process can be found on our [web application](https://share.streamlit.io/cjianyu98/nba_prediction/main/streamlit-app.py)!

[Back To The Top](#nba-match-predictor)

---
### Model 
Through various features selection methods, only 3 features are found to be significant;
1. Offensive Efficiency
2. Defensive Efficiency
3. ELO

#### Model training and prediction
* Data will first be split into the respective NBA seasons
* First half of the season will be used as training data, while second half is used as testing data
* Different classifiers were used to find the best performing one
* Accuracy and F1 score were used to evaluate the model performance
* Top 5 models were picked out to conduct majority voting 
* Final prediction will be based on the outcome of majority voting

#### Model Perfomance
Other than just looking at the different evaluation metric scores, we also compared our accuracy against the percentage of the team with the better Win-Lose record winning the game.

Justification: 
* The most straighforward method of predicting the winner of a match up will be simply choosing the team with the better W-L record
* If our model can outperform the above method, then we can consider our model a success

**Performance**
* Our model consistently achieved about 65-70% accuracy for every season.
* The winning percentage of a team with higher W-L record is consistently below 65% for every season. 

[Back To The Top](#nba-match-predictor)

---
### Code Navigation
1. [Dataset (including raw and processed data)](https://github.com/CJianYu98/NBA_Prediction/tree/main/data)
2. [Data Processing Notebook](https://github.com/CJianYu98/NBA_Prediction/blob/main/notebook/data_process.ipynb)
3. [Modelling Notebook](https://github.com/CJianYu98/NBA_Prediction/blob/main/notebook/model.ipynb)
4. [Streamlit App scripts](https://github.com/CJianYu98/NBA_Prediction/tree/main/streamlit_app)

[Back To The Top](#nba-match-predictor)

---
### Contributors

1. Brandon Tan Jun Da ([Linkedin](https://www.linkedin.com/in/brandon-tan-jun-da/), [Github](https://github.com/brandontjd))
2. Chen Jian Yu ([Linkedin](https://www.linkedin.com/in/chen-jian-yu/), [Github](https://github.com/CJianYu98))
3. Leonard Siah Tian Long ([Linkedin](https://www.linkedin.com/in/leonard-siah-0679631a1/), [Github](https://github.com/leoking98))
1. Samuel Sim Zhi Han ([Linkedin](https://www.linkedin.com/in/samuel-sim-7368241aa/), [Github](https://github.com/sszh1904))

[Back To The Top](#nba-match-predictor)