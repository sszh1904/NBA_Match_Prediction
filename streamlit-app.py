import streamlit as st
from streamlit_app import multipage
from streamlit_app import home, dataset, features, model, live_prediction

app = multipage.MultiPageApp()


# Adding your projects
app.add_app("Home", home.app)
app.add_app("Dataset", dataset.app)
app.add_app("Data Preparation", features.app)
app.add_app("Our Model", model.app)
app.add_app("Live Prediction", live_prediction.app)

app.run()