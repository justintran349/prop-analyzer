"""Results: how the picks have done by day / week / month, the most consistent
players, and a day-by-day search of the history."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from props import history as h

st.title("📈 Results")

SOURCES = {"live": "Top Picks (logged live)", "backtest": "Backtest: this season",
           "tracked": "My tracked bets"}
PCT = st.column_config.NumberColumn(format="percent")
UNITS = st.column_config.NumberColumn(format="%+.2f u")


@st.cache_data(ttl=600, show_spinner="Grading picks against the box scores...")
def load(mtimes):
    h.grade_history()
    return h.load_all()


def _mt(p):
    return p.stat().st_mtime if p.exists() else 0


from props import config  # noqa: E402

df_all = load((_mt(h.HISTORY_CSV), _mt(h.BACKTEST_CSV), _mt(config.PICKS_CSV), _mt(config.GAMELOGS_CSV)))

# --------------------------------------------------------------- filters
with st.container(border=True):
    c = st.columns([2.2, 1, 1.6, 1.4, 1.2])
    have = [s for s in SOURCES if (df_all["source"] == s).any()]
    default = ["live", "tracked"] if any(s in have for s in ("live", "tracked")) and \
        (df_all["source"].isin(["live", "tracked"]) & df_all["result"].isin(["win", "loss"])).sum() >= 20 \
        else have
    src = c[0].multiselect("Picks", have, default=[s for s in default if s in have],
                           format_func=SOURCES.get, key="res_src")
    sports = sorted(df_all["sport"].dropna().unique())
    sport = c[1].multiselect("Sport", sports, default=sports, key="res_sport")
    stats = sorted(df_all["stat_label"].dropna().unique())
    stat_f = c[2].multiselect("Stats", stats, placeholder="All stats", key="res_stats")
    conf = c[3].slider("Min confidence", 0.50, 0.80, 0.60, 0.01, format="%.2f", key="res_conf")
    top_n = c[4].selectbox("Picks per day", ["Top 10", "Top 5", "Top 20", "All"], key="res_top")

df = df_all[df_all["source"].isin(src) & df_all["sport"].isin(sport) & (df_all["confidence"].fillna(1) >= conf)]
if stat_f:
    df = df[df["stat_label"].isin(stat_f)]
n = None if top_n == "All" else int(top_n.split()[1])
df_conf = df.copy()          # every pick above the confidence filter (for player rankings)
df = pd.concat([h.top_per_day(df[df["source"] != "tracked"], n), df[df["source"] == "tracked"]])

if df_all.empty or not src:
    st.info("No picks yet. Every time Top Picks runs it's logged here and graded once the games are in "
            "the data. To see this season re-played, run `python scripts/backtest.py`.")
    st.stop()

if "backtest" in src:
    st.caption("⚠️ **Backtest** picks are graded against *estimated* lines (the player's last-10 average "
               "rounded to x.5) at -110, not real sportsbook lines. Books set sharper lines, so treat these "
               "numbers as an upper bound. The models were trained only on games before this season, so "
               "it is out-of-sample. Injury inputs use who actually sat.")

# ------------------------------------------------------------------ KPIs
r = h.record(df)
k = st.columns(6)
k[0].metric("Graded picks", f"{r['picks']:,}", f"{r['pending']} pending" if r["pending"] else None,
            delta_color="off")
k[1].metric("Record", f"{r['wins']}-{r['losses']}" + (f"-{r['pushes']}" if r["pushes"] else ""))
k[2].metric("Hit rate", f"{r['hit_rate']:.1%}" if pd.notna(r["hit_rate"]) else "-",
            f"{r['hit_rate'] - h.BREAK_EVEN:+.1%} vs break-even" if pd.notna(r["hit_rate"]) else None)
k[3].metric("Units (1u bets)", f"{r['units']:+.1f}")
k[4].metric("ROI", f"{r['roi']:+.1%}" if pd.notna(r["roi"]) else "-")
k[5].metric("Days", df.loc[df["result"].isin(["win", "loss", "push"]), "date"].nunique())

VIEWS = ["Daily", "Weekly", "Monthly", "Players", "Search by day", "Confidence & stats"]
view = st.segmented_control("View", VIEWS, default="Daily", key="res_view",
                            label_visibility="collapsed") or "Daily"


def period_view(freq, label):
    t = h.by_period(df, freq)
    if t.empty:
        st.info("Nothing graded yet for these filters.")
        return
    fmt = {"D": "%b %d", "W": "Week of %b %d", "M": "%B %Y"}[freq]
    t["label"] = t["period"].dt.strftime(fmt)
    fig = go.Figure()
    fig.add_bar(x=t["label"], y=t["hit_rate"], name="Hit rate",
                marker_color=np.where(t["hit_rate"] >= h.BREAK_EVEN, "#2e9e5b", "#c8553d"),
                customdata=np.stack([t["wins"], t["losses"], t["units"]], axis=1),
                hovertemplate="<b>%{x}</b><br>Hit rate %{y:.1%}<br>%{customdata[0]}-%{customdata[1]}"
                              "<br>%{customdata[2]:+.2f} units<extra></extra>")
    fig.add_scatter(x=t["label"], y=t["cum_units"], name="Running units", yaxis="y2",
                    mode="lines+markers", line=dict(color="#4c8bf5"))
    fig.add_hline(y=h.BREAK_EVEN, line_dash="dash", line_color="gray",
                  annotation_text="break-even 52.4%", annotation_position="top left")
    fig.update_layout(yaxis=dict(title="Hit rate", tickformat=".0%", range=[0, 1]),
                      yaxis2=dict(title="Running units", overlaying="y", side="right", showgrid=False),
                      xaxis=dict(type="category"), legend=dict(orientation="h", y=1.12),
                      margin=dict(t=30, b=10), height=380)
    st.plotly_chart(fig, width="stretch")
    show = t[["label", "picks", "wins", "losses", "hit_rate", "units", "roi", "cum_units"]].iloc[::-1]
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"label": label, "hit_rate": PCT, "roi": PCT, "units": UNITS,
                                "cum_units": st.column_config.NumberColumn("running units", format="%+.2f u")})
    best = t.loc[t["hit_rate"].idxmax()]
    worst = t.loc[t["hit_rate"].idxmin()]
    st.caption(f"Best {label.lower()}: **{best['label']}** ({best['wins']}-{best['losses']}, "
               f"{best['hit_rate']:.0%}) · Worst: **{worst['label']}** ({worst['wins']}-{worst['losses']}, "
               f"{worst['hit_rate']:.0%})")


if view == "Daily":
    period_view("D", "Day")
elif view == "Weekly":
    period_view("W", "Week")
elif view == "Monthly":
    period_view("M", "Month")

elif view == "Players":
    c1, c2 = st.columns([1, 3])
    min_p = c1.number_input("Min picks", 3, 100, 8, key="res_minp",
                            help="Players need at least this many graded picks to be ranked.")
    pl = h.players(df_conf, int(min_p))
    st.caption(f"Player rankings use **every** pick at {conf:.0%}+ confidence, not just the top "
               f"{top_n.lower()}, so each player has enough picks to judge.")
    if pl.empty:
        st.info("Not enough graded picks per player yet. Lower 'Min picks'.")
    else:
        c2.caption("**Consistency** = the lowest hit rate that's still believable given how many picks he "
                   "has (lower edge of a 95% range). It ranks a 30-10 record above a lucky 5-0.")
        cfg = {"hit_rate": PCT, "consistency": PCT, "units": UNITS,
               "last_pick": st.column_config.DateColumn(format="MMM D"),
               "longest_win_streak": "longest W streak", "current_streak": "now"}
        cols = ["player", "sport", "picks", "wins", "losses", "hit_rate", "consistency", "units",
                "longest_win_streak", "current_streak", "best_stat", "last_pick"]
        a, b = st.columns(2)
        a.markdown("#### ✅ Most consistent")
        a.dataframe(pl.head(15)[cols], hide_index=True, width="stretch", column_config=cfg)
        b.markdown("#### 💰 Most profitable")
        b.dataframe(pl.sort_values("units", ascending=False).head(15)[cols], hide_index=True,
                    width="stretch", column_config=cfg)
        st.markdown("#### 🚫 Least reliable (consider fading or skipping)")
        st.dataframe(pl.sort_values("hit_rate").head(10)[cols], hide_index=True, width="stretch",
                     column_config=cfg)
        who = st.selectbox("Look at one player's history", pl["player"].tolist(), index=None,
                           placeholder="Pick a player...", key="res_player")
        if who:
            v = df_conf[(df_conf["player"] == who) & df["result"].isin(["win", "loss", "push"])].sort_values("date",
                                                                                               ascending=False)
            st.dataframe(v[["date", "stat_label", "pick", "line", "actual", "result", "confidence", "game",
                            "source"]], hide_index=True, width="stretch",
                         column_config={"date": st.column_config.DateColumn(format="ddd MMM D"),
                                        "confidence": PCT, "stat_label": "stat"})

elif view == "Search by day":
    days = sorted(df["date"].dropna().unique())
    if not days:
        st.info("No picks for these filters.")
    else:
        c1, c2 = st.columns([1, 2])
        day = c1.date_input("Day", value=pd.Timestamp(days[-1]).date(), min_value=pd.Timestamp(days[0]).date(),
                            max_value=pd.Timestamp(days[-1]).date(), key="res_day")
        name = c2.text_input("Player (optional)", placeholder="e.g. Brunson", key="res_name")
        d = df[df["date"] == pd.Timestamp(day)]
        if name:
            d = d[d["player"].str.contains(name, case=False, na=False)]
        if d.empty:
            st.info(f"No picks on {day:%a %b %d} for these filters. Days with picks: "
                    + ", ".join(pd.Timestamp(x).strftime("%b %d") for x in days[-7:]) + "...")
        else:
            rr = h.record(d)
            st.markdown(f"### {day:%A, %B %d}: {rr['wins']}-{rr['losses']}"
                        + (f" ({rr['hit_rate']:.0%})" if pd.notna(rr["hit_rate"]) else "")
                        + f" · {rr['units']:+.2f} units"
                        + (f" · {rr['pending']} pending" if rr["pending"] else ""))
            icon = {"win": "✅", "loss": "❌", "push": "➖", "void": "🚫 DNP", "pending": "⏳"}
            d = d.sort_values("confidence", ascending=False).assign(
                outcome=lambda x: x["result"].map(icon).fillna(x["result"]))
            st.dataframe(d[["outcome", "player", "game", "stat_label", "pick", "line", "actual", "confidence",
                            "odds", "book", "source"]],
                         hide_index=True, width="stretch",
                         column_config={"confidence": PCT, "stat_label": "stat",
                                        "source": st.column_config.TextColumn("from")})

elif view == "Confidence & stats":
    b = h.by_confidence(df)
    if b.empty:
        st.info("Nothing graded yet.")
    else:
        st.markdown("#### Does confidence mean anything?")
        fig = go.Figure()
        fig.add_bar(x=b["bucket"].astype(str), y=b["hit_rate"], name="Actual hit rate", marker_color="#4c8bf5",
                    text=[f"{x:.0%}" for x in b["hit_rate"]], textposition="outside")
        fig.add_scatter(x=b["bucket"].astype(str), y=b["model_said"], name="Model said", mode="markers+lines",
                        line=dict(color="#f0a202", dash="dot"))
        fig.add_hline(y=h.BREAK_EVEN, line_dash="dash", line_color="gray")
        fig.update_layout(yaxis=dict(tickformat=".0%", range=[0.4, 1]), height=340, margin=dict(t=20),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        st.caption("If the bars sit on the dotted line, the confidence numbers are honest. Bars below the "
                   "line mean the model is overconfident in that range.")
        st.dataframe(b, hide_index=True, width="stretch",
                     column_config={"hit_rate": PCT, "model_said": PCT, "units": UNITS})
    st.markdown("#### By stat")
    g = df[df["result"].isin(["win", "loss", "push"])]
    if not g.empty:
        rows = [{"stat": s, **h.record(v)} for s, v in g.groupby("stat_label")]
        t = pd.DataFrame(rows).sort_values("hit_rate", ascending=False)
        st.dataframe(t[["stat", "picks", "wins", "losses", "hit_rate", "units", "roi"]], hide_index=True,
                     width="stretch", column_config={"hit_rate": PCT, "roi": PCT, "units": UNITS})
        rows = [{"pick": s, **h.record(v)} for s, v in g.groupby("pick")]
        st.dataframe(pd.DataFrame(rows)[["pick", "picks", "wins", "losses", "hit_rate", "units"]],
                     hide_index=True, width="stretch",
                     column_config={"hit_rate": PCT, "units": UNITS})
