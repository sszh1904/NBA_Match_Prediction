import streamlit as st

def app():
    # Page Title
    st.title('NBA Predictor by Team Sportnalytics :basketball:')

    # Welcome Message
    st.header("Welcome!")
    st.write('''
    Being a group of sports enthusiasts, the team is interested in prediciting the outcome of NBA games using Machine Learning techniques.

    You may navigate to different pages using the sidebar to get a better understanding of our project and prediction model.

    * **Dataset** - Explore our dataset here!
    * **Data Preparation** - Find out how we processed our data and conducted our feature engineering!
    * **Our Model** - Where we showcase our final model and it's performance!
    ''')

    # Project Summary
    st.header("Project Summary")
    # st.image("data_pipeline.png")

    # Contributions
    st.header("Contributions")
    st.write("This was a project as part of the Data Associate Programme by SMU BIA. In the team, we have [Brandon](https://www.linkedin.com/in/brandon-tan-jun-da/), [Jian Yu](https://www.linkedin.com/in/chen-jian-yu/), [Samuel](https://www.linkedin.com/in/samuel-sim-7368241aa/) and [Leonard](https://www.linkedin.com/in/leonard-siah-0679631a1/). Thank you for checking us out and have a nice day!")