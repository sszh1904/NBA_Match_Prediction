import streamlit as st

class MultiPageApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        st.sidebar.header("Navigation")
        app = st.sidebar.radio(
            '',
            self.apps,
            format_func=lambda app: app['title'])
        
        st.sidebar.write('''
        ### Source Code: 
        https://github.com/sszh1904/NBA-Match-Predictor
        ''')

        st.sidebar.write('''
        ### SMU BIA Website: 
        https://www.smubia.org/
        ''')

        app['function']()