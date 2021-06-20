import streamlit as st
from multipage import MultiPageApp
import home, dataset, features, model

app = MultiPageApp()


# Adding your projects
app.add_app("Home", home.app)
app.add_app("Dataset", dataset.app)
app.add_app("Data Preparation", features.app)
app.add_app("Our Model", model.app)

app.run()