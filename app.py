"""Sports Props Hub - NBA, MLB and NFL in one app.

    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Sports Props Hub", page_icon="🏆", layout="wide")

page = st.navigation(
    [
        st.Page("sports/top_picks.py", title="Top Picks", icon="🔥", default=True),
        st.Page("sports/nba.py", title="NBA", icon="🏀"),
        st.Page("sports/mlb.py", title="MLB", icon="⚾"),
        st.Page("sports/nfl_props.py", title="NFL Props", icon="🏈"),
        st.Page("sports/nfl.py", title="NFL Research", icon="📋"),
        st.Page("sports/odds_import.py", title="Odds CSV", icon="📄"),
    ],
    position="sidebar",
)
page.run()
