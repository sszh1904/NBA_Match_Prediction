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
        [Github](https://github.com/CJianYu98/NBA_Prediction)
        ''')

        st.sidebar.write('''
        ### SMU BIA Website: 
        https://www.smubia.org/
        ''')

        app['function']()