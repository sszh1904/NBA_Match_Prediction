import streamlit as st
from multipage import MultiPageApp
from streamlit import dataset, features, model, home

app = MultiPageApp()


# Adding your projects
app.add_app("Home", home.app)
app.add_app("Dataset", dataset.app)
app.add_app("Data Preparation", features.app)
app.add_app("Our Model", model.app)

app.run()