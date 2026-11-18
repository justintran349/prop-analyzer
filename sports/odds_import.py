import pandas as pd
import streamlit as st

from scrapers.common import ODDS_CSV_PATH
from scrapers.csv_import import read_backend_odds_csv


st.title("Backend Odds CSV")
st.caption(
    "Player searches read this backend file directly. No sportsbook website is "
    "queried, and there is no browser upload step."
)

st.code(str(ODDS_CSV_PATH), language=None)
st.download_button(
    "Download CSV header template",
    data=(
        "sport,league,player_name,team,opponent,game_start_time,stat_type,line,"
        "over_odds,under_odds,source,captured_at\n"
    ),
    file_name="odds.csv",
    mime="text/csv",
)

try:
    odds, row_errors = read_backend_odds_csv()
except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
    st.error(f"Could not read the backend odds CSV: {error}")
else:
    if not ODDS_CSV_PATH.exists():
        st.warning("The backend odds.csv file has not been created yet.")
    else:
        st.metric("Valid prop rows", len(odds))
        if odds.empty:
            st.info("Add permitted, current prop rows to odds.csv. Player searches read it directly.")
        else:
            st.dataframe(odds, use_container_width=True, hide_index=True)
    if row_errors:
        st.warning(f"{len(row_errors)} invalid row(s) are ignored by player searches.")
        st.dataframe(
            pd.DataFrame(row_errors),
            use_container_width=True,
            hide_index=True,
        )

st.subheader("CSV fields")
st.markdown(
    "Required: `sport`, `league`, `player_name`, `team`, `opponent`, "
    "`game_start_time`, `stat_type`, `line`, and `source`. Use `MLB`, `NBA`, "
    "or `NFL` for `sport`. `over_odds`, `under_odds`, and `captured_at` are "
    "optional; missing capture times use the CSV file's modified time. Keep "
    "different captures as separate rows to preserve line movement."
)