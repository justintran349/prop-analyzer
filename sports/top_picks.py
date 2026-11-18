"""Top Picks: the models' most confident NBA + NFL picks, shown as soon as you open the app.

Picks are built automatically (and cached in data/top_picks.csv). The page re-checks
every 15 minutes and rebuilds when the data, lines or models change, or every 3 hours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from props import top_picks as tp, tracker

PCT = st.column_config.NumberColumn(format="percent")

st.title("🔥 Top Picks")


@st.cache_data(show_spinner=False, ttl=3600)
def games_for(sport, player):
    return tp.player_games(sport, player)


@st.cache_resource(show_spinner="Loading model...")
def league_for(sport):
    if sport == "NBA":
        from props import slate
        return slate.League()
    from props import nfl_slate
    return nfl_slate.NFLLeague()


@st.cache_data(show_spinner=False, ttl=3600)
def explain_pick(r_key: str, r: dict):
    from props import explain
    try:
        row = explain.model_row(r["sport"], league_for(r["sport"]), r)
    except Exception:
        row = None
    return explain.breakdown(r["sport"], r, row)


@st.cache_data(show_spinner=False, ttl=3600)
def custom_pick(r_key: str, r: dict, line: float, over_odds, under_odds):
    """Re-score at a line you typed + the full explanation at that line."""
    from props import explain
    new, row = tp.rescore_pick(r["sport"], league_for(r["sport"]), r, line, over_odds, under_odds)
    if new is None:
        return None, [], pd.DataFrame()
    steps, effects = explain.breakdown(r["sport"], new, row)
    return new, steps, effects


def _clean(r: dict) -> dict:
    return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in r.items()}


def _valid_odds(x):
    return x is not None and (x <= -100 or x >= 100)


@st.dialog("Why this pick", width="large")
def show_hit_rates(r: dict):
    opp = tp.opponent_of(r["sport"], r["team"], r["game"])
    base_key = f"{r['sport']}|{r['player']}|{r['stat']}|{r['line']}|{r.get('generated_at')}"

    st.markdown("**Try a different line** (e.g. what your book actually shows)")
    c = st.columns([1, 1, 1, 2])
    new_line = c[0].number_input("Line", value=float(r["line"]), min_value=0.0, step=0.5, format="%.1f",
                                 key=f"line|{base_key}")
    oo = c[1].number_input("Over odds", value=None, step=5, format="%d", placeholder="e.g. -115",
                           key=f"oo|{base_key}", help="Optional: American odds, like -115 or +105.")
    uo = c[2].number_input("Under odds", value=None, step=5, format="%d", placeholder="e.g. -105",
                           key=f"uo|{base_key}")
    oo = int(oo) if _valid_odds(oo) else None
    uo = int(uo) if _valid_odds(uo) else None
    changed = float(new_line) != float(r["line"]) or oo is not None or uo is not None
    orig = r
    if changed:
        with st.spinner(f"Re-running the model at {new_line:g}..."):
            new, steps, effects = custom_pick(f"{base_key}|{new_line}|{oo}|{uo}", _clean(r),
                                              float(new_line), oo, uo)
        if new is None:
            st.error("The model couldn't score this line.")
            return
        r = new
        p0, p1 = orig["prob_over"], r["prob_over"]
        c[3].metric(f"Chance of OVER {new_line:g}", f"{p1:.1%}",
                    f"{p1 - p0:+.1%} vs {orig['line']:g}", delta_color="off")
        e_o, e_u = r["edge_over"], r["edge_under"]
        st.caption(f"Edge at these prices: OVER {e_o:+.1%} · UNDER {e_u:+.1%} "
                   + ("(no odds entered, so -110 is assumed: 52.4% break-even)" if oo is None and uo is None
                      else ""))
        if r["pick"] != orig["pick"]:
            st.warning(f"At {new_line:g} the pick flips to **{r['pick']}**.")
    side = "⬆️ OVER" if r["pick"] == "OVER" else "⬇️ UNDER"
    st.markdown(f"### {r['player']} · {side} {r['line']:g} {r['stat_label']}")
    st.caption(f"{r['team']} {r['game']} · model confidence {r['confidence']:.0%}"
               + (f" · tested hit rate {r['tested_hit']:.0%}" if pd.notna(r.get("tested_hit")) else ""))

    tab_why, tab_hr = st.tabs(["🧠 The reasoning", "📊 Hit rates"])
    with tab_why:
        if not changed:
            with st.spinner("Working through the numbers..."):
                steps, effects = explain_pick(base_key, _clean(r))
        for i, s_ in enumerate(steps, 1):
            st.markdown(f"{i}. {s_}")
        if not effects.empty:
            st.markdown("**What moved the chance the most**")
            e = effects.copy()
            toward = "OVER" if r["pick"] == "OVER" else "UNDER"
            e["pushes"] = e["effect_on_over"].map(lambda v: "toward OVER" if v > 0.005 else
                                                  "toward UNDER" if v < -0.005 else "no real effect")
            e["helps this pick"] = ((e["effect_on_over"] > 0) == (toward == "OVER")) & (e["effect_on_over"].abs() > 0.005)
            fig = px.bar(e.iloc[::-1], x="effect_on_over", y="factor", orientation="h",
                         color="helps this pick", color_discrete_map={True: "#2e9e5b", False: "#c8553d"},
                         labels={"effect_on_over": "change in chance of OVER", "factor": ""})
            fig.update_layout(height=60 + 34 * len(e), margin=dict(t=10, b=10), showlegend=False,
                              xaxis_tickformat="+.0%")
            st.plotly_chart(fig, width="stretch")
            st.caption("How this is measured: each group of factors is reset to a neutral value (e.g. an "
                       "average defense, no injuries, indoor weather) and the model is re-run. The bar is how "
                       "much the chance of the over changed. Green bars support this pick, red bars work "
                       "against it. Groups overlap, so they don't add up exactly to the final number.")

    with tab_hr:
        _hit_rate_section(r, opp)


def _hit_rate_section(r: dict, opp: str):
    g = games_for(r["sport"], r["player"])
    hr = tp.hit_rates(g, r["stat"], r["line"], r["pick"], opp)
    if hr.empty:
        st.info("No game history found for this player.")
        return
    cols = st.columns(len(hr))
    for c, (_, row) in zip(cols, hr.iterrows()):
        c.metric(row["Window"].replace("Head to head", "H2H"),
                 f"{row['Hit rate']:.0%}" if row["Games"] else "—",
                 f"{row['Hits']}/{row['Games']}" if row["Games"] else "no games", delta_color="off")
    st.dataframe(hr, hide_index=True, width="stretch", column_config={
        "Hit rate": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=1),
        "Average": st.column_config.NumberColumn(format="%.1f")})
    last = g.tail(15).copy()
    hit = (last[r["stat"]] > r["line"]) if r["pick"] == "OVER" else (last[r["stat"]] < r["line"])
    last["result"] = hit.map({True: "Hit", False: "Miss"})
    fig = px.bar(last, x="label", y=r["stat"], color="result",
                 color_discrete_map={"Hit": "#2e9e5b", "Miss": "#c8553d"},
                 labels={"label": "", r["stat"]: r["stat_label"]}, title="Last 15 games")
    fig.add_hline(y=r["line"], line_dash="dash", annotation_text=f"Line {r['line']:g}")
    fig.update_layout(height=320, margin=dict(t=40, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    h2h = g[g["opp"] == opp].tail(10)
    if not h2h.empty:
        st.caption(f"Every game vs {opp}: " + ", ".join(
            f"{lbl.split(' ')[0]} {v:g}" for lbl, v in zip(h2h["label"], h2h[r["stat"]])))

    st.markdown("**Lines at every book** (from oddsscraper)")
    from props import scraped_lines
    if not scraped_lines.available():
        st.caption("Start your scraper (`cd ~/Desktop/oddsscraper && oddsscraper run`) to compare "
                   "DraftKings, FanDuel, Bovada, PrizePicks and Underdog here.")
    else:
        try:
            cmp = tp.compare_books(r["sport"], league_for(r["sport"]), r["player"], r["stat"])
        except Exception as e:
            cmp = pd.DataFrame()
            st.caption(f"Couldn't compare books: {e}")
        if cmp.empty:
            st.caption("No book has this prop posted right now.")
        else:
            st.dataframe(cmp, hide_index=True, width="stretch", column_config={
                "model_over": st.column_config.NumberColumn("Model P(over)", format="percent"),
                "best_edge": st.column_config.NumberColumn("Edge", format="percent"),
                "best_side": "Best side"})
            st.caption("Edge = model's chance minus the chance the book's price implies. "
                       "PrizePicks/Underdog have no per-pick odds, so a -110 break-even (52.4%) is assumed. "
                       "Their real break-even depends on your entry type.")


def rebuild(previous: pd.DataFrame | None = None):
    with st.status("Scoring every player in the next NBA slate and NFL week (about a minute)...",
                   expanded=False) as s:
        df = tp.build(log=s.write)
        s.update(label=f"Updated: {len(df):,} picks ranked", state="error" if tp.LAST_ERRORS else "complete")
    for sport, err in tp.LAST_ERRORS.items():
        st.error(f"**{sport} picks couldn't be built:** {err}\n\nIf this mentions scikit-learn or a model "
                 "file, run `python scripts/train_model.py` and `python scripts/train_nfl_model.py` in the "
                 "VS Code terminal to rebuild the models for your setup.")
    if df.empty and previous is not None and not previous.empty:
        st.warning("Showing the last picks that built successfully.")
        return previous
    return df


BOOK_NAMES = {"draftkings": "DraftKings", "fanduel": "FanDuel", "bovada": "Bovada",
              "prizepicks": "PrizePicks", "underdog": "Underdog", "betmgm": "BetMGM", "caesars": "Caesars"}


def lines_bar(df):
    """Where the lines come from + a button to pull fresh ones from oddsscraper."""
    from props import scraped_lines as sl
    n_real = int((df["line_type"] == "sportsbook").sum()) if not df.empty else 0
    c1, c2 = st.columns([3, 1])
    if n_real == 0:
        c1.warning("**All lines below are estimated** (the player's recent average), not real sportsbook "
                   "lines. " + ("Click **Pull fresh lines** to scrape DraftKings, FanDuel, Bovada, PrizePicks "
                                "and Underdog with your oddsscraper." if sl.installed() else
                                "Your oddsscraper isn't set up in this project yet. In the VS Code terminal run "
                                "`bash scripts/setup_oddsscraper.sh`."))
    else:
        c1.caption(f"📗 {n_real:,} picks use real sportsbook/DFS lines · {sl.status()}")
    books = sl.books_available()
    if books:
        current = [b for b in sl.allowed_books() if b in books] or books
        chosen = st.multiselect(
            "Sportsbooks to use", books, current, format_func=lambda b: BOOK_NAMES.get(b, b.title()),
            help="Picks, line shopping and the props boards only use lines from these books. "
                 "Players with no line at your books fall back to an estimated line.")
        if chosen and sorted(chosen) != sorted(current):
            sl.set_allowed_books([] if sorted(chosen) == books else chosen)
            rebuild(df)
            st.rerun()
    if c2.button("📥 Pull fresh lines", disabled=not sl.installed(),
                 help="Runs one oddsscraper pass (a few minutes), then rescores every pick."):
        with st.spinner("Scraping books and DFS apps (a few minutes)..."):
            ok, msg = sl.run_scrape()
        (st.success if ok else st.error)(("Lines updated. " if ok else "Scrape failed: ") + msg)
        bad = [r for r in sl.scrape_health() if not r["ok"]]
        if bad:
            st.caption("Sources with errors: " + "; ".join(
                f"{r['source']} {r['sport_key'] or ''}: {(r['error'] or '')[:60]}" for r in bad[:6]))
        rebuild(df)
        st.rerun()


def player_search(df: pd.DataFrame):
    """Find any player on tonight's NBA slate / this week's NFL games and see every prop."""
    opts = (df.drop_duplicates(["sport", "player"])
            .assign(label=lambda d: d["player"] + "  ·  "
                    + np.where(d["sport"] == "NBA", d["team"].astype(str) + " ", "") + d["game"].astype(str)
                    + "  (" + d["sport"] + ")")
            .sort_values("player"))
    choice = st.selectbox("🔍 Search a player", opts["label"].tolist(), index=None,
                          placeholder="Type a name, e.g. Luka or Brunson...", key="player_search")
    if not choice:
        return
    sport, player = opts.loc[opts["label"] == choice, ["sport", "player"]].iloc[0]
    p = (df[(df["sport"] == sport) & (df["player"] == player)]
         .sort_values("confidence", ascending=False).reset_index(drop=True))
    with st.container(border=True):
        r0 = p.iloc[0]
        st.markdown(f"#### {player} · " + (f"{r0['team']} {r0['game']}" if sport == "NBA" else r0["game"]))
        best = p.iloc[0]
        c = st.columns(3)
        c[0].metric("Best pick", f"{best['pick']} {best['line']:g} {best['stat_label']}")
        c[1].metric("Confidence", f"{best['confidence']:.0%}")
        c[2].metric("Props found", len(p))
        if isinstance(r0.get("injury_short"), str) and r0["injury_short"]:
            st.caption(f"🩹 {r0['injury_short']}")
        st.caption("Every prop for him, ignoring the filters below. Click a row for the full reasoning, "
                   "hit rates and **Try a different line**.")
        ev = st.dataframe(p[["stat_label", "pick", "line", "line_type", "confidence", "tested_hit",
                             "avg_season", "book", "edge", "reason"]],
                          hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                          key=f"search_tbl_{sport}_{player}",
                          column_config={"stat_label": "Stat", "confidence": PCT, "tested_hit": PCT,
                                         "edge": PCT, "avg_season": st.column_config.NumberColumn(format="%.1f")})
        if ev and ev.selection.rows:
            show_hit_rates(p.iloc[ev.selection.rows[0]].to_dict())
        missing = sorted(set(df.loc[df["sport"] == sport, "stat_label"].dropna()) - set(p["stat_label"]))
        if missing:
            st.caption("Not listed: " + ", ".join(missing) + " (no realistic line, e.g. he rarely gets them). "
                       + ("Check any line in the NBA page's Player tab." if sport == "NBA" else
                          "Check any line in the NFL Props page's Player analyzer."))
    st.caption("Only players in upcoming games are searchable here. For anyone else, use the NBA or "
               "NFL Props page.")


@st.fragment(run_every="15m")
def picks_view():
    df, fresh = tp.load_cached()
    top_bar = st.columns([3, 1])
    if df.empty or not fresh:
        if not df.empty:
            top_bar[0].caption(f"Showing picks from {df['generated_at'].max():%a %I:%M %p} while updating...")
        df = rebuild(df)
    if top_bar[1].button("🔄 Refresh now"):
        df = rebuild(df)
    lines_bar(df)
    if df.empty:
        st.info("No upcoming games found for NBA or NFL.")
        return
    top_bar[0].caption(f"Updated {df['generated_at'].max():%a %b %d, %I:%M %p} · refreshes automatically")

    player_search(df)

    with st.expander("Filters", expanded=False):
        f = st.columns(4)
        sports = f[0].multiselect("Sport", ["NBA", "NFL"], ["NBA", "NFL"])
        min_conf = f[1].slider("Minimum confidence", 0.5, 0.95, 0.6, 0.01, format="%.2f")
        trusted = f[2].toggle("Only models that beat the baseline", True,
                              help="Hides stats whose model didn't beat a simple guess in testing "
                                   "(e.g. NBA rebounds, NFL passing TDs).")
        two = f[3].toggle("Max 2 picks per player", True)
        g = st.columns(3)
        stats = g[0].multiselect("Stats", sorted(df["stat_label"].dropna().unique()))
        books_only = g[1].toggle("Sportsbook lines only", False,
                                 help="Only picks against real lines you entered or pulled from The Odds API.")
        n_cards = g[2].slider("Cards per sport", 3, 24, 9, 3)

    v = df[df["sport"].isin(sports) & (df["confidence"] >= min_conf)]
    if trusted:
        v = v[v["trusted_model"]]
    if two:
        v = v[v["player_rank"] <= 2]
    if stats:
        v = v[v["stat_label"].isin(stats)]
    if books_only:
        v = v[v["line_type"] == "sportsbook"]

    if (v["line_type"] == "estimated").any():
        st.caption("**Estimated line** = the player's recent average rounded to x.5, used when there's no "
                   "sportsbook line. Check the real line before betting: if it's different, so is the pick. "
                   "**Tested hit rate** = how often picks at this confidence won when the model was tested "
                   "on games it never saw.")

    for sport in ("NBA", "NFL"):
        s = v[v["sport"] == sport]
        if s.empty:
            continue
        day = s["date"].min()
        label = f"tonight ({day:%a %b %d})" if sport == "NBA" else f"this week (from {day:%a %b %d})"
        st.subheader(f"{'🏀' if sport == 'NBA' else '🏈'} {sport} {label}")
        top = s.head(n_cards)
        cols = st.columns(3)
        for i, r in enumerate(top.itertuples()):
            with cols[i % 3].container(border=True):
                arrow = "⬆️" if r.pick == "OVER" else "⬇️"
                st.markdown(f"**{r.player}** · {r.team} {r.game}")
                st.markdown(f"### {arrow} {r.pick} {r.line:g} {r.stat_label}")
                c1, c2 = st.columns(2)
                c1.metric("Confidence", f"{r.confidence:.0%}")
                c2.metric("Tested hit rate", f"{r.tested_hit:.0%}" if pd.notna(r.tested_hit) else "—")
                badge = (f"📗 {r.book} line" + (f" · edge {r.edge:+.0%}" if pd.notna(r.edge) else "")
                         if r.line_type == "sportsbook" else "📝 estimated line")
                avg = f" · season avg {r.avg_season:.1f}" if pd.notna(r.avg_season) else ""
                st.caption(f"{badge}{avg}")
                if isinstance(r.notes, str) and r.notes.strip(" ·"):
                    st.caption(r.notes.strip(" ·"))
                if isinstance(getattr(r, "reason", None), str) and r.reason:
                    st.markdown(f"<small>💡 {r.reason}</small>", unsafe_allow_html=True)
                if st.button("🧠 Why + hit rates", key=f"hr_{sport}_{i}", width="stretch"):
                    show_hit_rates(r._asdict())
        with st.expander(f"All {len(s):,} {sport} picks (click a row for the reasoning and hit rates)"):
            s = s.reset_index(drop=True)
            ev = st.dataframe(s[["player", "team", "game", "stat_label", "line", "line_type", "pick", "confidence",
                                 "tested_hit", "avg_season", "book", "edge", "reason", "notes"]],
                              hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                              key=f"tbl_{sport}",
                              column_config={"confidence": PCT, "tested_hit": PCT, "edge": PCT,
                                             "avg_season": st.column_config.NumberColumn(format="%.1f")})
            if ev and ev.selection.rows:
                show_hit_rates(s.iloc[ev.selection.rows[0]].to_dict())
        if st.button(f"Track the top {len(top)} {sport} picks", key=f"track_{sport}"):
            added = 0
            if sport == "NBA":
                from props import slate
                lookup = slate.League().player_id
            else:
                from props import nfl_slate
                lookup = nfl_slate.NFLLeague().player_id
            for r in top.itertuples():
                pid = lookup(r.player)
                if pid:
                    added += tracker.add(r.date, r.player, pid, r.stat, r.line, r.pick, -110, r.confidence,
                                         r.book if isinstance(r.book, str) else "estimated",
                                         source="top" if sport == "NBA" else "nfl-top")
            st.success(f"Tracked {added} picks. Results appear in the NBA page's Tracker tab once graded.")

    if v.empty:
        st.info("No picks match these filters.")


picks_view()
st.caption("Models estimate chances. They don't guarantee results. At -110 odds you need to win 52.4% to break even.")
