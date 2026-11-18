"""NFL Props page: weather-aware over/under model for the upcoming week."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from props import nfl_data as nd, nfl_model as nm, nfl_slate, nfl_weather as nw, odds, tracker

PCT = st.column_config.NumberColumn(format="percent")
ONE = st.column_config.NumberColumn(format="%.1f")
LABEL = nd.LABEL.get


@st.cache_resource(show_spinner="Loading NFL data...")
def load():
    return nfl_slate.NFLLeague()


@st.cache_data(show_spinner="Scoring lines...", ttl=1800)
def score(_lg, lines, key):
    return _lg.score_lines(lines)


@st.cache_data(show_spinner="Projecting this week's players...", ttl=1800)
def projections(_lg, stats, key):
    return _lg.projections(stats)


@st.cache_data(show_spinner=False)
def weather_table(_lg, key):
    return nm.weather_impact(_lg.pg, _lg.games)


st.title("🏈 NFL Props")
try:
    lg = load()
except RuntimeError as e:
    st.warning(f"{e}\n\nThen run `python scripts/train_nfl_model.py` and refresh.")
    st.stop()
pg, games = lg.pg, lg.games
key = f"{len(pg)}-{pg['date'].max()}"
week = lg.week

with st.sidebar:
    if not week.empty:
        st.caption(f"{int(week['season'].iloc[0])} season · Week {int(week['week'].iloc[0])} · "
                   f"data through {pg['date'].max():%b %d}")
    if st.button("Reload NFL data", key="nfl_reload"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    st.caption("Refresh stats, injuries & weather: `python scripts/update_nfl_data.py`")

tabs = st.tabs(["This week", "Player analyzer", "Props board", "Weather impact",
                "Defense vs position", "Model"])
t_week, t_player, t_board, t_wx, t_dvp, t_model = tabs


# ------------------------------------------------------------- this week
with t_week:
    if week.empty:
        st.info("No upcoming games found (off-season?).")
    else:
        st.subheader(f"Week {int(week['week'].iloc[0])}")
        cols = st.columns(3)
        for i, g in enumerate(week.itertuples()):
            gd = g._asdict()
            with cols[i % 3].container(border=True):
                final = pd.notna(gd["home_score"])
                st.markdown(f"**{gd['away_team']} @ {gd['home_team']}**  \n"
                            f"{gd['date']:%a %b %d} · {int(gd['kick_hour']) if pd.notna(gd['kick_hour']) else '?'}:00 ET"
                            + (f" · Final {int(gd['away_score'])}-{int(gd['home_score'])}" if final else ""))
                if pd.notna(gd["spread_line"]):
                    fav = gd["home_team"] if gd["spread_line"] > 0 else gd["away_team"]
                    home_imp = gd["total_line"] / 2 + gd["spread_line"] / 2
                    st.caption(f"{fav} -{abs(gd['spread_line']):.1f} · O/U {gd['total_line']:.1f} · "
                               f"implied {gd['away_team']} {gd['total_line'] - home_imp:.1f}, "
                               f"{gd['home_team']} {home_imp:.1f}")
                wx = nw.describe(gd)
                (st.warning if nw.bad_weather(gd) else st.caption)(wx)
                for t in (gd["away_team"], gd["home_team"]):
                    m = lg.missing(t)
                    if m["names"]:
                        st.caption(f"{t} out: {', '.join(m['names'][:5])}" + (" · **starting QB out**" if m["qb_out"] else ""))
        st.subheader("Model projections")
        ps = st.multiselect("Stats", list(nd.STATS), ["passing_yards", "rushing_yards", "receiving_yards", "receptions"],
                            format_func=LABEL, key="nfl_ps")
        if st.button("Project every regular in this week's games"):
            st.session_state["nfl_proj"] = projections(lg, tuple(ps), key)
        pj = st.session_state.get("nfl_proj")
        if pj is not None and not pj.empty:
            st.caption("'Line' = the player's last-5 average rounded to x.5. P(over) = model's chance of beating it.")
            st.dataframe(pj.assign(stat=pj["stat"].map(LABEL)), hide_index=True, width="stretch",
                         column_config={"prob_over": PCT, "avg_last5": ONE, "proj": ONE, "implied_pts": ONE,
                                        "dvp_factor": st.column_config.NumberColumn(format="%.2f")})


# -------------------------------------------------------- player analyzer
with t_player:
    c1, c2, c3 = st.columns([2, 1, 1])
    player = c1.selectbox("Player", sorted(lg.name_to_id), index=None, placeholder="Type a name...", key="nfl_p")
    stat = c2.selectbox("Stat", list(nd.STATS), format_func=LABEL, key="nfl_s")
    line = c3.number_input("Line", min_value=0.0, value=0.5 if stat == "anytime_td" else 49.5, step=0.5, key="nfl_l")
    if player:
        pid = lg.name_to_id[player]
        cur = lg.current.loc[pid]
        if cur["position"] not in nd.STATS[stat][1]:
            st.caption(f"Note: {LABEL(stat)} is usually only posted for {', '.join(nd.STATS[stat][1])}.")
        g = lg.game_for_team(cur["team"])
        hist = pg[pg["player_id"] == pid].merge(
            games[["game_id", "game_temp", "game_wind", "wx_precip", "wx_snow", "indoor", "home_team"]],
            on="game_id", how="left").sort_values("date")
        r = None
        if g is None:
            st.info(f"No upcoming game found for {cur['team']}.")
        else:
            gd = g.to_dict()
            opp = gd["away_team"] if gd["home_team"] == cur["team"] else gd["home_team"]
            st.info(f"{cur['position']} · {cur['team']} {'vs' if gd['home_team'] == cur['team'] else '@'} {opp} · "
                    f"{gd['date']:%a %b %d} · {nw.describe(gd)}")
            if nm.model_exists(stat):
                try:
                    r = lg.predict(pid, stat, line)
                except Exception as e:
                    st.caption(f"Model couldn't score this: {e}")
        m = st.columns(5)
        m[0].metric("Model: chance of OVER", f"{r['prob_over']:.0%}" if r is not None and pd.notna(r["prob_over"]) else "—")
        m[1].metric("Last 5 avg", f"{r['avg_last5']:.1f}" if r is not None else "—")
        m[2].metric("Team implied pts", f"{r['implied_pts']:.1f}" if r is not None and pd.notna(r["implied_pts"]) else "—",
                    help="From the Vegas total and spread. Higher = more scoring expected.")
        m[3].metric("Opp. vs position", f"{(r['dvp_factor'] - 1) * 100:+.0f}%" if r is not None else "—",
                    help="How much more (+) or less (-) than average this defense allows to his position.")
        m[4].metric("Snap share (last 3)", f"{r['snap_last3']:.0%}" if r is not None and pd.notna(r["snap_last3"]) else "—")
        if r is not None:
            notes = []
            if r["windy"]:
                notes.append("💨 15+ mph wind: passing and kicking usually drop")
            if r["wet"]:
                notes.append("🌧️ rain/snow: fewer deep passes, more runs")
            if r["cold"]:
                notes.append("🥶 below 35°F")
            if r["qb_out"]:
                notes.append("⚠️ starting QB out")
            if r["miss_targets"] > 0:
                notes.append(f"{r['miss_targets']:.1f} targets/game from injured teammates up for grabs")
            if notes:
                st.markdown(" · ".join(notes))

        bad = hist.apply(nw.bad_weather, axis=1) if not hist.empty else pd.Series(dtype=bool)
        cur_season = hist[hist["season"] == hist["season"].max()]

        def hr(s):
            s = s.dropna()
            return {"Games": len(s), "Overs": int((s > line).sum()),
                    "Hit rate": (s > line).mean() if len(s) else None, "Average": s.mean() if len(s) else None}
        splits = {"This season": hr(cur_season[stat]), "Last 5": hr(hist[stat].tail(5)),
                  "Home": hr(hist[hist["team"] == hist["home_team"]][stat]),
                  "Away": hr(hist[hist["team"] != hist["home_team"]][stat]),
                  "Indoors": hr(hist[hist["indoor"] == 1][stat]),
                  "Bad weather (wind/rain/snow/freezing)": hr(hist[bad][stat]) if len(bad) else hr(pd.Series(dtype=float)),
                  "Good weather outdoors": hr(hist[(hist["indoor"] == 0) & ~bad][stat]) if len(bad) else hr(pd.Series(dtype=float))}
        if g is not None:
            splits[f"vs {opp}"] = hr(hist[hist["opp"] == opp][stat])
        st.dataframe(pd.DataFrame(splits).T.reset_index().rename(columns={"index": "Split"}), hide_index=True,
                     width="stretch", column_config={"Hit rate": st.column_config.ProgressColumn(
                         format="%.0f%%", min_value=0, max_value=1), "Average": ONE})
        h = hist.tail(17).copy()
        if not h.empty:
            h["label"] = h["season"].astype(str).str[-2:] + " W" + h["week"].astype(str) + " " + h["opp"]
            h["result"] = (h[stat] > line).map({True: "Over", False: "Under"})
            h["conditions"] = h.apply(lambda x: nw.describe(x), axis=1)
            from props.onoff import teammates_out
            h["out"] = teammates_out("NFL", h)
            fig = px.bar(h, x="label", y=stat, color="result", custom_data=["conditions", "out"],
                         category_orders={"label": h["label"].tolist()},
                         color_discrete_map={"Over": "#2e9e5b", "Under": "#c8553d"}, labels={"label": "", stat: LABEL(stat)})
            fig.update_traces(hovertemplate=f"<b>%{{x}}</b><br>{LABEL(stat)}: %{{y}}<br>%{{customdata[0]}}"
                                            "<br><br><b>Teammates out:</b><br>%{customdata[1]}<extra></extra>")
            fig.add_hline(y=line, line_dash="dash", annotation_text=f"Line {line}")
            fig.update_layout(height=330, margin=dict(t=20, b=10), legend_title_text="")
            st.plotly_chart(fig, width="stretch")
        if r is not None and pd.notna(r["prob_over"]):
            side = "OVER" if r["prob_over"] >= 0.5 else "UNDER"
            c_a, c_b = st.columns(2)
            if c_a.button(f"Track {side} {line}", key="nfl_track"):
                tracker.add(g["date"], player, pid, stat, line, side, -110,
                            r["prob_over"] if side == "OVER" else 1 - r["prob_over"], source="nfl")
                st.success("Tracked (Tracker tab on the NBA page shows all picks).")
            if c_b.button("Save line to data/nfl_props/lines.csv", key="nfl_save"):
                new = pd.DataFrame([{"date": g["date"], "player": player, "stat": stat, "line": line, "book": "manual"}])
                new.to_csv(nd.LINES_CSV, mode="a", header=not nd.LINES_CSV.exists(), index=False)
                st.success("Saved.")


# -------------------------------------------------------------- props board
with t_board:
    st.write("Lines from `data/nfl_props/lines.csv` and the last Odds API pull, scored by the model. "
             "**Edge** = model chance minus the chance the odds imply.")
    if odds.odds_api_available():
        if st.button("Pull NFL prop lines (uses API credits)"):
            with st.spinner("Fetching from The Odds API..."):
                try:
                    df, rem = nfl_slate.fetch_odds_api_lines()
                    st.success(f"{len(df)} lines. Credits left: {rem}")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Odds API error: {e}")
    else:
        st.caption("Add `ODDS_API_KEY` to `.env` for live lines from DraftKings, FanDuel, etc.")
    lines = nfl_slate.load_lines()
    if lines.empty:
        st.info("No lines yet. Save some from the Player tab, add rows to data/nfl_props/lines.csv "
                "(date,player,stat,line,over_odds,under_odds), or pull them from The Odds API.")
    else:
        board = score(lg, lines, key + str(len(lines)) + str(lines["line"].sum()))
        if board.empty:
            st.info("Couldn't match those lines to players with an upcoming game.")
        else:
            f1, f2 = st.columns(2)
            fs = f1.multiselect("Stat", sorted(board["stat"].unique()), format_func=LABEL, key="nfl_fs")
            fe = f2.slider("Minimum edge", -0.2, 0.3, -0.2, 0.01, key="nfl_fe")
            v = board[board["edge"].fillna(-1) >= fe]
            if fs:
                v = v[v["stat"].isin(fs)]
            st.dataframe(v.drop(columns=["player_id", "pick_odds", "date"]).assign(stat=lambda d: d["stat"].map(LABEL)),
                         hide_index=True, width="stretch",
                         column_config={"prob_over": PCT, "prob_pick": PCT, "edge": PCT, "l5_hit": PCT,
                                        "implied_pts": ONE, "avg_last5": ONE,
                                        "dvp_factor": st.column_config.NumberColumn(format="%.2f")})
            bl = odds.best_lines(lines.assign(date=lines["date"].dt.normalize()))
            if not bl.empty and bl["books"].max() >= 2:
                st.subheader("Line shopping")
                bl = bl[bl["books"] >= 2].sort_values("line_range", ascending=False)
                st.dataframe(bl.assign(stat=bl["stat"].map(LABEL)).drop(columns="date"), hide_index=True, width="stretch")


# ----------------------------------------------------------- weather impact
with t_wx:
    st.write("How each stat changes with conditions, across every game since 2023 "
             "(players who played at least half the snaps). The model uses temperature, wind, gusts, "
             "rain, snow, indoors and turf directly, so these effects are part of every prediction.")
    wt = weather_table(lg, key)
    s = st.selectbox("Stat", list(nd.STATS), format_func=LABEL, key="wx_stat")
    sub = wt[(wt["stat"] == s)].sort_values("vs_normal")
    fig = px.bar(sub, x="condition", y="vs_normal", text=sub["avg"].round(1),
                 labels={"vs_normal": "vs normal outdoor games", "condition": ""}, color="vs_normal",
                 color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
    fig.update_layout(yaxis_tickformat="+.0%", height=380, coloraxis_showscale=False, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Bar labels = average per player-game. Small samples (few games) are noisy. Check the 'games' column.")
    st.dataframe(sub[["condition", "avg", "vs_normal", "games"]], hide_index=True,
                 column_config={"vs_normal": st.column_config.NumberColumn(format="%+.1%"), "avg": ONE})


# ------------------------------------------------------- defense vs position
with t_dvp:
    s = st.selectbox("Stat ", list(nd.STATS), format_func=LABEL, key="nfl_dvp")
    season = int(pg["season"].max())
    cur = pg[(pg["season"] == season)]
    per = cur.groupby(["opp", "position", "game_id"])[s].sum().groupby(["opp", "position"]).mean().unstack()
    per = per[[p for p in nd.STATS[s][1] if p in per.columns]]
    rel = (per / per.mean() - 1) * 100
    fig = go.Figure(go.Heatmap(z=rel.values, x=rel.columns, y=rel.index, colorscale="RdYlGn", zmid=0,
                               text=per.round(1).values, texttemplate="%{text}", colorbar=dict(title="% vs avg")))
    fig.update_layout(height=820, yaxis=dict(autorange="reversed"), margin=dict(t=10))
    st.caption(f"{LABEL(s)} allowed per game to each position, {season} season. Green = gives up more (good for overs).")
    st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------------- model
with t_model:
    st.write("One model per stat, tested on the most recent games it never saw. "
             "At -110 you need **52.4%** to break even.")
    rows = []
    for s in nd.STATS:
        m = nm.load_metrics(s)
        if m:
            rows.append({"Stat": LABEL(s), "key": s, "Test games": m["test_games"], "Accuracy": m["accuracy"],
                         "AUC": m["auc"], "Brier": m["brier"], "Baseline Brier": m["baseline_brier"],
                         "≥60% sure": m["confident_accuracy"], "Bad-weather accuracy": m["bad_weather_accuracy"],
                         "Bad-weather games": m["bad_weather_test_games"], "Real lines": m["real_line_rows"]})
    if rows:
        mt = pd.DataFrame(rows)
        st.dataframe(mt.drop(columns="key"), hide_index=True, width="stretch",
                     column_config={"Accuracy": PCT, "≥60% sure": PCT, "Bad-weather accuracy": PCT})
        st.warning("Measured against ESTIMATED lines (last-5 average). Sportsbooks are sharper, "
                   "so real accuracy will be lower until real lines are collected.")
        cs = st.selectbox("Calibration chart for", mt["key"], format_func=LABEL, key="nfl_cal")
        cal = pd.DataFrame(nm.load_metrics(cs).get("calibration", []))
        if not cal.empty:
            fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash", color="gray")),
                             go.Scatter(x=cal["predicted"], y=cal["actual"], mode="lines+markers", name="Model")])
            fig.update_layout(height=340, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                              xaxis_title="Model said", yaxis_title="Actually went over", margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("No NFL models yet - run `python scripts/train_nfl_model.py`.")
