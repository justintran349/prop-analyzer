"""NBA page of the Sports Props Hub (opened from app.py)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from props import alerts, analysis, config, context, model, odds, slate, tracker
from props.teams import TEAMS

PCT = st.column_config.NumberColumn(format="percent")
ONE = st.column_config.NumberColumn(format="%.1f")


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner="Loading NBA data...")
def load_league():
    return slate.League()


@st.cache_data(show_spinner="Scoring lines with the model...", ttl=1800)
def score(_lg, lines: pd.DataFrame, key: str):
    return slate.score_lines(_lg, lines)


@st.cache_data(show_spinner="Projecting tonight's players...", ttl=1800)
def projections(_lg, games: pd.DataFrame, stats: tuple, key: str):
    return slate.projections_for_games(_lg, games, stats)


@st.cache_data(show_spinner=False, ttl=1800)
def predict_one(_lg, pid, stat, line, opp, is_home, date, team, key):
    mine, theirs = _lg.missing(team), _lg.missing(opp)
    miss = {k: mine[k] for k in slate.MISS_KEYS} | {"opp_miss_min": theirs["miss_min"]}
    req = pd.DataFrame([{"player_id": pid, "stat": stat, "line": float(line), "opp": opp,
                         "is_home": bool(is_home), "team": team,
                         "date": pd.Timestamp(date or pd.Timestamp.today().normalize()), **miss}])
    r = _lg.predict(req).iloc[0]
    if pd.isna(r.get("prob_over")):
        raise RuntimeError("Not enough history to score this player.")
    return {"prob_over": float(r["prob_over"]), "pred_min": float(r.get("pred_min", np.nan)),
            "proj": float(r.get("proj", np.nan)), "onoff": r.get("onoff"),
            "prob_over_model": r.get("prob_over_model"),
            "features": {k: r.get(k) for k in model.FEATURES}}


st.title("🏀 NBA Props")
try:
    lg = load_league()
except RuntimeError as e:
    st.warning(f"{e}\n\nIn the VS Code terminal run `python scripts/update_data.py`, "
               "then `python scripts/train_model.py`, and refresh this page.")
    st.stop()
logs = lg.logs
data_key = f"{len(logs)}-{logs['date'].max()}"
from props.nfl_data import LABEL as NFL_LABEL  # noqa: E402


def STAT_LABEL(s):
    return config.STATS.get(s) or NFL_LABEL.get(s, s)

with st.sidebar:
    st.caption(f"Season {config.SEASON - 1}-{str(config.SEASON)[-2:]} · {len(logs):,} player-games · "
               f"latest game {logs['date'].max():%b %d}")
    if st.button("Reload data from disk"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    if st.button("Refresh injury report (ESPN)"):
        try:
            context.fetch_injuries()
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Couldn't reach ESPN: {e}")
    st.caption("Fresh game logs: `python scripts/update_data.py` · everything: `python scripts/daily.py`")
    from props import scraped_lines as _sl
    st.caption(_sl.status())


def all_lines() -> pd.DataFrame:
    frames = [odds.load_manual_lines(), odds.load_saved_api_lines()]
    if "live_lines" in st.session_state:
        frames.append(st.session_state["live_lines"])
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=odds.LINE_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[df["date"] >= pd.Timestamp.today().normalize()]
    return df.drop_duplicates(["date", "player", "stat", "book", "line"], keep="last")


def track_button(rows: pd.DataFrame, key: str, label="Track selected picks"):
    if st.button(label, key=key, disabled=rows.empty):
        added = sum(tracker.add(r.date, r.player, r.player_id, r.stat, r.line, r.pick,
                                r.pick_odds, r.prob_pick, r.book) for r in rows.itertuples())
        st.success(f"Tracked {added} pick(s). See the Tracker tab.")


tabs = st.tabs(["Tonight", "Player analyzer", "Props board", "Defense vs position",
                "Matchups", "Tracker", "Model"])
tab_tonight, tab_player, tab_board, tab_dvp, tab_match, tab_track, tab_model = tabs


# --------------------------------------------------------------- tonight
with tab_tonight:
    day = st.date_input("Date", pd.Timestamp.today().date(), key="slate_day")
    games = slate.todays_games(lg, day)
    if games.empty:
        st.info("No games on this date.")
    else:
        cols = st.columns(min(4, len(games)))
        for i, g in enumerate(games.itertuples()):
            v = lg.vegas(g.home, g.date)
            with cols[i % len(cols)].container(border=True):
                st.markdown(f"**{g.away} @ {g.home}**")
                if v:
                    st.caption(f"Books: total {v['total']:.1f} · {g.home} {v['spread']:+.1f}")
                pgm = slate.projected_game(lg, g.home, g.away)
                if pgm:
                    fav = g.home if pgm["home_margin"] >= 0 else g.away
                    st.caption(f"Model: {fav} by {abs(pgm['home_margin']):.1f} · total {pgm['total']:.0f}")
                from props import scraped_lines
                if scraped_lines.available():
                    with st.popover("Alt spreads"):
                        alt = pd.concat([scraped_lines.alternate_spreads("NBA", TEAMS[t]) for t in (g.home, g.away)],
                                        ignore_index=True)
                        if alt.empty:
                            st.caption("None posted.")
                        else:
                            alt = alt[alt["team"].isin([g.home, g.away])].sort_values(["team", "point"])
                            st.dataframe(alt[["team", "point", "price", "bookmaker"]], hide_index=True)
                            if pgm:
                                st.caption(f"Model margin: {g.home} {pgm['home_margin']:+.1f}. An alt spread "
                                           "'covers' in the model's view when team margin + point > 0.")
                for t in (g.away, g.home):
                    names = lg.missing(t)["names"]
                    if names:
                        st.caption(f"{t} out: {', '.join(names)}")
        lines = all_lines()
        day_lines = lines[lines["date"] == pd.Timestamp(day)] if not lines.empty else lines
        st.subheader("Best edges")
        if day_lines.empty:
            st.caption("No sportsbook lines for this date yet. Pull them on the Props board, "
                       "add them to data/lines.csv, or run `python scripts/daily.py`.")
        else:
            board = score(lg, day_lines, data_key + str(len(day_lines)))
            if not board.empty:
                best = board[board["edge"] > 0].drop_duplicates(["player_id", "stat"]).head(25)
                st.dataframe(best[["player", "game", "stat", "line", "pick", "prob_pick", "edge",
                                   "book", "proj", "pred_min", "teammates_out"]]
                             .assign(stat=lambda d: d["stat"].map(STAT_LABEL)),
                             hide_index=True, width="stretch",
                             column_config={"prob_pick": PCT, "edge": PCT, "proj": ONE, "pred_min": ONE})
        st.subheader("Model projections (no lines needed)")
        pstats = st.multiselect("Stats", list(config.STATS), ["pts", "trb", "ast"],
                                format_func=STAT_LABEL, key="proj_stats")
        if st.button("Project every rotation player in these games"):
            st.session_state["proj"] = projections(lg, games, tuple(pstats), data_key + str(day))
        if "proj" in st.session_state and not st.session_state["proj"].empty:
            pj = st.session_state["proj"]
            st.caption("'Line' = the player's recent average rounded to x.5. "
                       "P(over) is the model's chance of beating it.")
            st.dataframe(pj.assign(stat=pj["stat"].map(STAT_LABEL)), hide_index=True, width="stretch",
                         column_config={"prob_over": PCT, "proj": ONE, "pred_min": ONE,
                                        "dvp_factor": st.column_config.NumberColumn(format="%.2f"),
                                        "dvp_recent": st.column_config.NumberColumn(format="%.2f"),
                                        "proj_total": st.column_config.NumberColumn(format="%.0f")})


# ------------------------------------------------------- player analyzer
with tab_player:
    c1, c2, c3 = st.columns([2, 1, 1])
    player = c1.selectbox("Player", sorted(lg.name_to_id), index=None, placeholder="Type a name...")
    stat = c2.selectbox("Stat", list(config.STATS), format_func=STAT_LABEL)
    line = c3.number_input("Line", min_value=0.0, value=20.5, step=0.5)

    if player:
        pid = lg.name_to_id[player]
        team = lg.current.loc[pid, "team"]
        nxt = lg.game_for(team)
        c4, c5, c6 = st.columns([1, 1, 2])
        teams = sorted(TEAMS)
        opp = c4.selectbox("Opponent", teams, index=teams.index(nxt[0]) if nxt else 0,
                           format_func=lambda t: f"{t} – {TEAMS[t]}")
        is_home = c5.radio("Venue", ["Home", "Away"], horizontal=True,
                           index=0 if (nxt[1] if nxt else True) else 1) == "Home"
        game_date = nxt[2] if nxt and nxt[0] == opp else pd.Timestamp.today().normalize()
        if nxt:
            c6.info(f"Next game: {'vs' if nxt[1] else '@'} {nxt[0]} on {nxt[2]:%a %b %d}")

        rep = analysis.player_report(logs, lg.dvp, pid, stat, line, opp, is_home)
        rec = analysis.dvp_lookup(lg.dvp_recent, opp, rep["pos"], stat) if rep["pos"] else {}
        ml = None
        if model.model_exists(stat):
            try:
                ml = predict_one(lg, pid, stat, float(line), opp, is_home, game_date, team, data_key)
            except Exception as e:
                st.caption(f"Model couldn't score this one: {e}")
        has_proj = ml is not None and not np.isnan(ml["proj"])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Model: chance of OVER", f"{ml['prob_over']:.0%}" if ml else "—")
        m2.metric("Projection", f"{ml['proj']:.1f}" if has_proj else "—",
                  delta=f"{ml['proj'] - line:+.1f} vs line" if has_proj else None,
                  help="Per-minute rate (last 10) × projected minutes.")
        m3.metric("Projected minutes", f"{ml['pred_min']:.1f}" if ml else "—",
                  help="From the minutes model: rest, injuries, blowout risk, role.")
        m4.metric(f"{opp} vs {rep['pos']} (season)",
                  f"#{int(rep['dvp_rank'])}" if rep["dvp_rank"] else "—",
                  delta=f"{(rep['dvp_factor'] - 1) * 100:+.0f}% vs avg",
                  help="#1 = gives up the MOST of this stat to this position.")
        m5.metric("Last 15 games", f"#{int(rec['rank'])}" if rec.get("rank") else "—",
                  delta=f"{(rec.get('factor', 1) - 1) * 100:+.0f}% vs avg")

        if ml and isinstance(ml.get("onoff"), dict):
            from props import onoff as _oo
            st.info(_oo.explain(ml["onoff"], config.STATS.get(stat, stat), player,
                                ml["prob_over_model"], ml["prob_over"]))
        mine, theirs = lg.missing(team), lg.missing(opp)
        v = lg.vegas(team, game_date)
        info = []
        if mine["names"]:
            info.append(f"**{team} out:** {', '.join(mine['names'])} "
                        f"({mine['miss_pts']:.0f} pts/g, {mine['miss_ast']:.0f} ast/g missing)")
        if theirs["names"]:
            info.append(f"**{opp} out:** {', '.join(theirs['names'])}")
        if v:
            info.append(f"**Vegas:** total {v['total']:.1f}, {team} {v['spread']:+.1f}")
        elif ml and pd.notna(ml.get("features", {}).get("proj_total")):
            info.append(f"**Projected game total (from scoring):** {ml['features']['proj_total']:.0f}")
        if info:
            st.markdown(" · ".join(info))

        split_df = pd.DataFrame([{"Split": k, "Games": s["games"], "Overs": s["overs"],
                                  "Hit rate": s["hit_rate"], "Average": s["avg"]}
                                 for k, s in rep["splits"].items()])
        st.dataframe(split_df, hide_index=True, width="stretch", column_config={
            "Hit rate": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=1),
            "Average": ONE})

        g = rep["games"].tail(25).copy()
        g["label"] = g["date"].dt.strftime("%m/%d") + " " + g["home"].map({1: "vs", 0: "@"}) + " " + g["opp"]
        g["result"] = (g[stat] > line).map({True: "Over", False: "Under"})
        from props.onoff import teammates_out
        g["out"] = teammates_out("NBA", g.assign(player_id=pid) if "player_id" not in g else g)
        fig = px.bar(g, x="label", y=stat, color="result", custom_data=["min", "out"],
                     color_discrete_map={"Over": "#2e9e5b", "Under": "#c8553d"},
                     category_orders={"label": g["label"].tolist()},
                     labels={"label": "", stat: STAT_LABEL(stat)})
        fig.update_traces(hovertemplate=f"<b>%{{x}}</b><br>{STAT_LABEL(stat)}: %{{y}} in %{{customdata[0]}} min"
                                        "<br><br><b>Teammates out:</b><br>%{customdata[1]}<extra></extra>")
        fig.add_hline(y=line, line_dash="dash", annotation_text=f"Line {line}")
        fig.update_layout(height=340, margin=dict(t=20, b=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")

        with st.expander("Likely defender (NBA.com tracking data)"):
            if st.button("Look up who usually guards him", key="ld"):
                try:
                    from props.matchups import likely_defender
                    with st.spinner("Asking NBA.com..."):
                        d = likely_defender(player, opp)
                    if not d:
                        st.info("No matchup data for this pair yet.")
                    else:
                        st.write(f"**{d['DEF_PLAYER_NAME']}** guarded him for "
                                 f"{d.get('PARTIAL_POSS', 0):.0f} possessions in {d['season'] - 1}-"
                                 f"{str(d['season'])[-2:]}: {d.get('pts_per100', float('nan')):.1f} pts "
                                 f"per 100 possessions ({d.get('pts_vs_overall', float('nan')):+.1f} "
                                 "vs his overall).")
                except Exception as e:
                    st.error(f"Lookup failed: {e}")

        b1, b2, _ = st.columns(3)
        if b1.button("Save line to data/lines.csv"):
            odds.add_manual_line(game_date, player, stat, line)
            st.success("Saved.")
        if ml:
            side = "OVER" if ml["prob_over"] >= 0.5 else "UNDER"
            if b2.button(f"Track {side} {line}"):
                tracker.add(game_date, player, pid, stat, line, side, -110,
                            ml["prob_over"] if side == "OVER" else 1 - ml["prob_over"])
                st.success("Tracked.")


# ------------------------------------------------------------ props board
with tab_board:
    st.write("Lines from `data/lines.csv`, the last Odds API pull, and live pulls. "
             "**Edge** = model chance minus the chance the odds imply.")
    b1, _ = st.columns(2)
    if odds.odds_api_available():
        if b1.button("Pull live lines + game totals (uses API credits)"):
            with st.spinner("Fetching from The Odds API..."):
                try:
                    odds.fetch_game_lines()
                    live, remaining = odds.fetch_odds_api_lines()
                    st.session_state["live_lines"] = live
                    st.cache_resource.clear()
                    st.success(f"{len(live)} lines. API credits remaining: {remaining}")
                except Exception as e:
                    st.error(f"Odds API error: {e}")
    else:
        b1.caption("Add `ODDS_API_KEY=...` to `.env` for live lines from every sportsbook.")
    lines = all_lines()
    if lines.empty:
        st.info("No upcoming lines yet. Add rows to data/lines.csv or save one from the Player tab.")
    else:
        board = score(lg, lines, data_key + str(len(lines)) + str(lines["line"].sum()))
        if board.empty:
            st.info("Couldn't match those lines to players in the data.")
        else:
            f1, f2, f3 = st.columns(3)
            fstat = f1.multiselect("Stat", sorted(board["stat"].unique()), format_func=STAT_LABEL)
            fmin = f2.slider("Minimum edge", -0.2, 0.3, -0.2, 0.01, format="%.2f")
            fbook = f3.multiselect("Book", sorted(board["book"].dropna().unique()))
            view = board[board["edge"].fillna(-1) >= fmin]
            if fstat:
                view = view[view["stat"].isin(fstat)]
            if fbook:
                view = view[view["book"].isin(fbook)]
            view = view.reset_index(drop=True)
            event = st.dataframe(
                view[["player", "game", "stat", "line", "book", "over_odds", "under_odds", "proj",
                      "pred_min", "l10_hit", "dvp_rank", "vegas_total", "prob_over", "pick", "edge"]]
                .assign(stat=lambda d: d["stat"].map(STAT_LABEL)),
                hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row",
                column_config={"proj": ONE, "pred_min": ONE, "l10_hit": PCT, "prob_over": PCT,
                               "edge": PCT, "vegas_total": ONE}, key="board_tbl")
            sel = list(event.selection.rows) if event and event.selection else []
            picked = view.iloc[sel]
            st.caption("Select rows (checkbox on the left) to track them or build a parlay.")
            track_button(picked, "track_board")

            st.subheader("Parlay builder")
            if len(picked) < 2:
                st.caption("Select 2+ rows above.")
            else:
                prob = float(np.prod(picked["prob_pick"].fillna(0.5)))
                fair = odds.to_american(prob)
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Model chance all hit", f"{prob:.1%}")
                pc2.metric("Fair odds", f"{fair:+d}")
                book_odds = pc3.number_input("Book's parlay odds", value=264 if len(picked) == 2 else 600,
                                             step=10)
                ev = prob * odds.payout(book_odds) - (1 - prob)
                st.metric("Expected value per 1 unit", f"{ev:+.2f}u")
                games_in = picked["date"].astype(str) + picked[["team", "opp"]].apply(
                    lambda r: "-".join(sorted(r)), axis=1)
                if picked["player_id"].duplicated().any():
                    st.warning("Two legs on the same player are strongly correlated - the real "
                               "chance is NOT the product shown. Books usually block these.")
                elif games_in.duplicated().any():
                    st.warning("Legs from the same game are correlated (a high-scoring game lifts "
                               "everyone). The combined chance is only an estimate.")

            st.subheader("Line shopping")
            bl = odds.best_lines(lines)
            if bl.empty or bl["books"].max() < 2:
                st.caption("Needs lines from 2+ sportsbooks (Odds API pull).")
            else:
                bl = bl[bl["books"] >= 2].sort_values("line_range", ascending=False)
                st.dataframe(bl.assign(stat=bl["stat"].map(STAT_LABEL)).drop(columns="date"),
                             hide_index=True, width="stretch")
                st.caption("Bet overs at the lowest line and unders at the highest - "
                           "a half point is worth roughly 2-3% in win rate.")


# ---------------------------------------------------- defense vs position
with tab_dvp:
    d1, d2 = st.columns(2)
    s = d1.selectbox("Stat ", list(config.STATS), format_func=STAT_LABEL, key="dvp_stat")
    window = d2.radio("Window", ["Full season", "Last 15 games"], horizontal=True)
    src = lg.dvp if window == "Full season" else lg.dvp_recent
    d = src[src["stat"] == s].pivot(index="opp", columns="pos", values="factor")
    d = (d[[p for p in analysis.POSITIONS if p in d.columns]] - 1) * 100
    fig = go.Figure(go.Heatmap(z=d.values, x=d.columns, y=d.index, colorscale="RdYlGn", zmid=0,
                               text=d.round(0).values, texttemplate="%{text:+.0f}%",
                               colorbar=dict(title="% vs avg")))
    fig.update_layout(height=820, yaxis=dict(autorange="reversed"), margin=dict(t=10))
    st.caption("Green = this defense gives up MORE than league average to that position "
               "(good for overs). Per 36 minutes.")
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------- matchups
with tab_match:
    st.caption("Who guards a player, from NBA.com tracking data. Results are cached for a day "
               "in data/matchups.csv. NBA.com is sometimes slow or blocks requests.")
    m1, m2 = st.columns(2)
    mp = m1.selectbox("Player ", sorted(lg.name_to_id), index=None, key="m_player")
    mo = m2.selectbox("Opponent ", sorted(TEAMS), key="m_opp")
    if mp and st.button("Look up matchups"):
        try:
            from props.matchups import cached_matchups
            with st.spinner("Asking NBA.com..."):
                mt = cached_matchups(mp, mo)
            if mt.empty:
                st.info("No matchup data for that pair this season yet.")
            else:
                st.dataframe(mt, hide_index=True, width="stretch")
                st.caption("pts_vs_overall > 0: he scores MORE per 100 possessions against "
                           "this defender than against everyone else.")
        except Exception as e:
            st.error(f"Matchup lookup failed: {e}")


# ---------------------------------------------------------------- tracker
with tab_track:
    t1, t2 = st.columns([1, 3])
    if t1.button("Grade picks now"):
        n = tracker.grade(logs)
        try:
            from props import nfl_data
            n += tracker.grade_nfl(nfl_data.load()[1])
        except Exception:
            pass
        st.success(f"{n} pick(s) graded.")
    picks = tracker.load()
    if picks.empty:
        st.info("No picks yet. Track them from the Player tab or the Props board. "
                "`scripts/daily.py` also auto-tracks the model's best picks every day.")
    else:
        src = t2.radio("Show", ["All", "My picks", "Auto (model)"], horizontal=True)
        if src == "My picks":
            picks = picks[picks["source"] != "auto"]
        elif src == "Auto (model)":
            picks = picks[picks["source"] == "auto"]
        sm = tracker.summary(picks)
        k = st.columns(5)
        k[0].metric("Record", f"{sm['wins']}-{sm['losses']}")
        k[1].metric("Win rate", f"{sm['win_rate']:.1%}" if sm["win_rate"] is not None else "—",
                    help="Need 52.4% at -110 to break even.")
        k[2].metric("Profit", f"{sm['profit']:+.2f}u")
        k[3].metric("ROI", f"{sm['roi']:+.1%}" if sm["roi"] is not None else "—")
        k[4].metric("Pending", sm["pending"])
        done = picks[picks["result"].isin(["win", "loss", "push"])].sort_values(["date", "created"])
        if not done.empty:
            done = done.assign(cum=done["profit"].cumsum())
            fig = px.line(done, x="date", y="cum", markers=True,
                          labels={"cum": "Profit (units)", "date": ""})
            fig.add_hline(y=0, line_color="gray")
            fig.update_layout(height=280, margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
            g1, g2 = st.columns(2)
            g1.caption("By stat")
            g1.dataframe(tracker.by_group(picks, "stat").assign(stat=lambda x: x["stat"].map(STAT_LABEL)),
                         hide_index=True, column_config={"win_rate": PCT, "roi": PCT})
            g2.caption("By side")
            g2.dataframe(tracker.by_group(picks, "pick"), hide_index=True,
                         column_config={"win_rate": PCT, "roi": PCT})
            cal = done[done["model_prob"].notna() & (done["result"] != "push")].copy()
            if len(cal) >= 20:
                cal["bucket"] = pd.cut(cal["model_prob"], [0, .55, .6, .65, .7, 1])
                c = cal.groupby("bucket", observed=True).agg(
                    predicted=("model_prob", "mean"),
                    actual=("result", lambda r: (r == "win").mean()),
                    n=("result", "size")).reset_index()
                st.caption("Your tracked picks: model confidence vs how often they actually won")
                st.dataframe(c.assign(bucket=c["bucket"].astype(str)), hide_index=True,
                             column_config={"predicted": PCT, "actual": PCT})
        st.dataframe(picks.sort_values("date", ascending=False)
                     .assign(stat=lambda x: x["stat"].map(STAT_LABEL)),
                     hide_index=True, width="stretch",
                     column_config={"model_prob": PCT,
                                    "profit": st.column_config.NumberColumn(format="%+.2f")})


# ------------------------------------------------------------------ model
with tab_model:
    st.write("One model per stat plus a minutes model. Each is tested on the most recent games "
             "it never saw. At -110 odds you need to win **52.4%** to break even.")
    mm = model.load_metrics("minutes")
    if mm:
        st.caption(f"Minutes model: off by {mm['mae']:.1f} min/game on average "
                   f"(a simple last-10 average is off by {mm['baseline_mae_last10_avg']:.1f}).")
    rows = []
    for s in config.STATS:
        m = model.load_metrics(s)
        if m:
            rows.append({"Stat": STAT_LABEL(s), "key": s, "Test games": m["test_games"],
                         "Accuracy": m["accuracy"], "AUC": m["auc"], "Brier": m["brier"],
                         "Baseline Brier": m["baseline_brier"],
                         "Accuracy when ≥60% sure": m["confident_accuracy"],
                         "# confident": m["confident_picks"], "Real lines": m["real_line_rows"]})
    if rows:
        mt = pd.DataFrame(rows)
        st.dataframe(mt.drop(columns="key"), hide_index=True, width="stretch",
                     column_config={"Accuracy": PCT, "Accuracy when ≥60% sure": PCT})
        if mt["Real lines"].max() < 1000:
            st.warning("These results are against ESTIMATED lines (each player's recent average). "
                       "Sportsbooks set sharper lines, so expect lower accuracy on real bets. "
                       "The Tracker tab shows real results.")
        cs = st.selectbox("Calibration chart for", mt["key"], format_func=STAT_LABEL)
        cal = pd.DataFrame(model.load_metrics(cs).get("calibration", []))
        if not cal.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                                     line=dict(dash="dash", color="gray")))
            fig.add_trace(go.Scatter(x=cal["predicted"], y=cal["actual"], mode="lines+markers",
                                     name="Model", text=cal["n"],
                                     hovertemplate="said %{x:.0%}<br>actual %{y:.0%}<br>%{text} games"))
            fig.update_layout(height=380, xaxis_title="Model said", yaxis_title="Actually went over",
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%", margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("On the dashed line = when the model says 65%, it happens 65% of the time.")
    else:
        st.info("No models trained yet.")
    if st.button("Retrain all models (takes several minutes)"):
        lines_hist = (pd.read_csv(config.LINE_HISTORY_CSV, parse_dates=["date"])
                      if config.LINE_HISTORY_CSV.exists() else None)
        with st.spinner("Training..."):
            model.train_all(logs, lines=lines_hist, log=lambda *_: None)
        st.cache_data.clear()
        st.success("Done.")
        st.rerun()
    if alerts.enabled() and st.button("Send a test phone alert"):
        st.write("Sent!" if alerts.send("Test", "NBA props alerts are working.") else "Failed.")
