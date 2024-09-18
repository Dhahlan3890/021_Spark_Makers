import streamlit as st

st.set_page_config(page_title="Main Page", page_icon="🏠")

st.title("Welcome to the Main Page")

st.write("""
This is the main page of our Election helper app. Use the sidebar to navigate to other pages:
- Manifesto Comparator
- Win Predictor
- AI Bot
""")

st.sidebar.success("Select a page above.")
