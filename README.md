# 🏆 Sports Props Hub

One Streamlit app for player props:

- **NBA**: game logs from NBA.com, positions and schedule from Basketball-Reference, injuries from ESPN, lines from The Odds API. Machine-learning models estimate the chance a player goes **over** on 12 stats.
- **NFL Props**: weather-aware models for 11 NFL props (passing / rushing / receiving yards, receptions, TDs, ...) using nflverse stats, Vegas lines, injuries and Open-Meteo game-time weather.
- **MLB / NFL Research + Odds CSV**: your sports_stats_db pages (latest Desktop version). Each player search shows "Current imported player props" with hit rates over the last 5, 10 or 15 games or all games. Lines come from `data/odds.csv` **and** your oddsscraper database.

## Setup (once)

Open this folder in VS Code (**File → Open Folder…**), then in the terminal (**Ctrl+`**):

```bash
mv vscode .vscode                  # turns on VS Code run buttons + Python settings
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # optional: Odds API key, phone alerts
```

Models in `models/` are trained with Python 3.9 + scikit-learn 1.6.1 (pinned in requirements.txt), matching this Mac. If you ever change Python or scikit-learn versions, retrain with `python scripts/train_model.py` and `python scripts/train_nfl_model.py`.

## Everyday use

```bash
streamlit run app.py               # the dashboard (NBA / MLB / NFL in the sidebar)
python scripts/daily.py            # refresh everything + score today + alerts
bash scripts/install_daily_job.sh  # run daily.py automatically every morning at 9:07
```

NFL: `python scripts/update_nfl_data.py` (stats, injuries, weather, ~2 min) ·
`python scripts/train_nfl_model.py` (~2 min). The daily job runs both automatically.

Individual NBA steps: `python scripts/update_data.py` (game logs, ~1 min) ·
`python scripts/train_model.py` (models, ~10 min) · `pytest` (tests).
Data through **Nov 13, 2026** and trained models are already included.

## 🔥 Top Picks (opens first)

When you open the app, it shows the models' **most confident picks** for the next NBA slate and the upcoming NFL week. There's no searching: every regular player is scored on every stat automatically.

- Each card shows the pick (e.g. ⬇️ UNDER 24.5 Points), the model's **confidence**, and the **tested hit rate** (how often picks at that confidence won in testing), plus whether the line is a real sportsbook line or an estimated one.
- **Auto-updates:** the page re-checks every 15 minutes and rebuilds when data, lines or models change (or every 3 hours). The daily job also builds it each morning, so it opens instantly.
- **Filters:** sport, minimum confidence, stats, sportsbook lines only, and max 2 picks per player.
- Unrealistic lines are filtered out (like a center going under 0.5 threes), and so are stats whose model didn't beat the baseline in testing (NBA rebounds, NFL passing TDs, NFL anytime TD). You can turn that filter off.
- **💡 Reasoning on every card:** one or two sentences with the main reasons, e.g. *"Line 24.5 is above his season avg (22.1) and last 10 avg (21.4); under hit in 7 of his last 10; model projects 21.8 in 33 min; opponent allows 8% less than average to his position."*
- **🧠 Why + hit rates** opens the full breakdown in plain English:
  1. averages vs the line, and how often he's cleared it lately
  2. the projection math: per-minute rate × projected minutes (NBA), or role, Vegas implied points and weather (NFL)
  3. matchup, injuries and game context, with the actual numbers
  4. the model's raw %, the calibrated %, and the tested hit rate at that confidence
  5. break-even math: e.g. *"At -115 you risk $115 to win $100, so you need 115 / (115 + 100) = 53.5%. The model says 61.0%, so the edge is +7.5%."*
  6. **What moved the chance most:** each group of factors is switched to neutral and the model is re-run, so you see how many points each one added or removed
- **📊 Hit rates tab** (same pop-up): click the button on any card, or a row in the full table, to see how often that exact pick hit **head to head** vs this opponent, over the **last 5, 10 and 15** games, and **this season**, with a chart of the last 15 games against the line.
- **Track the top N** logs the picks so the Tracker shows how they actually do.

## NBA tabs

| Tab | What it does |
|---|---|
| **Tonight** | Today's games with Vegas total/spread and who's out, the best edges from your lines, and model projections for every rotation player (no lines needed). |
| **Player analyzer** | Model chance of the over, projection, projected minutes, defense vs position (season and last 15 games), teammates out, hit rates, a game-log chart, the likely defender, and buttons to save or track the pick. |
| **Props board** | Every line scored and sorted by edge, with filters. Select rows to **track** them or build a **parlay** (combined chance, fair odds, expected value, correlation warnings). **Line shopping** shows the best book for each over and under. |
| **Defense vs position** | Heatmap of what each team allows to each position, for the full season or the last 15 games. |
| **Matchups** | Who guards a player (NBA.com tracking data, cached for a day). |
| **Tracker** | Record, win rate, profit, ROI, a profit chart, results by stat and side, and how your picks' model confidence compares with actual results. |
| **Model** | Test results for all 12 stats plus the minutes model, a calibration chart, and a retrain button. |

## NFL Props page

| Tab | What it does |
|---|---|
| **This week** | Every game with kickoff, spread, total, **implied team totals**, a **weather forecast** (flagged when windy, wet or freezing) and key injuries, including a starting QB being out. Model projections for every regular. |
| **Player analyzer** | Model chance of the over, last-5 average, team implied points, opponent vs position, snap share, weather notes, and hit rates split by home/away, indoors, **bad vs good weather**, and vs this opponent. Game log chart with each game's conditions. |
| **Props board** | Your lines plus Odds API lines, scored and sorted by edge, with line shopping. |
| **Weather impact** | How every stat changes in wind, rain/snow, freezing temperatures and indoors, across all games since 2023. |
| **Defense vs position** | Yards, receptions and TDs each defense allows to QBs, RBs, WRs and TEs. |
| **Model** | Accuracy per stat, including **accuracy in bad-weather games**, plus calibration charts. |

**What the NFL model uses** (`props/nfl_model.py`):
- **Weather:** temperature, wind, gusts, rain, snow, indoors/dome, turf, and cold/windy/wet flags. Past games use the official readings plus the Open-Meteo archive; upcoming games use the 16-day forecast, refreshed each update.
- **Vegas:** game total, team spread, implied team points (game script: trailing teams throw more, leading teams run more)
- **Game:** home/away, rest days, short week, division game, prime time
- **Player role:** snap share, targets, target share, air-yards share, carries, pass attempts, yards per opportunity, recent and season averages
- **Team and defense:** team pass/run volume, points per game, opponent points allowed, opponent vs position (season and last 4 games)
- **Injuries:** targets and carries of injured teammates, starting QB out, opponent injuries (from the weekly NFL injury report)

## What the NBA model looks at (`props/model.py`)

Every feature uses only information known before tip-off:

- **Player form:** season, last 5 and last 10 averages, consistency, per-minute rate, **usage rate**, shot attempts
- **Minutes model:** a separate model projects tonight's minutes (rest, back-to-backs, role, teammates out, blowout risk)
- **Game context:** home/away, rest days, **projected game total and margin** (pace and blowout risk)
- **Teammates out:** minutes, points, assists and rebounds of missing rotation players. For past games this comes from box scores; for upcoming games, from the ESPN injury report plus `data/injuries_manual.csv`
- **Defense:** opponent vs the player's position (season and last 15 games), opponent overall, opponent's own injuries
- **History** against this specific opponent, and the **line** itself (including last-10 hit rate at that line)

**Training lines:** real sportsbook lines (`data/line_history.csv`) are used whenever they exist. Otherwise the model uses estimated lines, with extra copies above and below so it learns how the line's distance from a player's average matters.

**Calibration:** raw tree models get overconfident at the extremes. In testing, a raw "17%" for turnovers really happened 32% of the time. Each model is therefore followed by a calibration step (Platt scaling) fit on games it never saw, so a displayed 65% means about 65%. All accuracy numbers are measured after calibration, cross-fitted so the calibration never grades its own homework.

**Read the results honestly:** until you've collected real lines, test accuracy is measured against estimated lines, and real-world accuracy will be lower. At -110 odds you need to win **52.4%** to break even. The Tracker tab is the real scoreboard.

## Real lines from your oddsscraper

The app reads sportsbook and DFS lines from your **oddsscraper** project (`~/Desktop/oddsscraper`), which scrapes DraftKings, FanDuel, Bovada, PrizePicks and Underdog into a local database.

```bash
# once, inside this project's .venv (installs oddsscraper here + runs a first scrape):
bash scripts/setup_oddsscraper.sh
```

Keep lines fresh with the **📥 Pull fresh lines** button on Top Picks, `python scripts/scrape_lines.py`,
the daily job (it scrapes before scoring), or `oddsscraper run` in a second terminal (every hour).
oddsscraper's `.env` limits it to NBA, NFL and MLB.

Then, automatically:
- **Top Picks, the NBA Props board and the NFL Props board** use the real lines (`client.player_props`) instead of estimated ones, with the book name and your **edge**.
- **Hit rates pop-up → "Lines at every book":** every book's line for that prop side by side, with the model's chance and edge at each line, so you can see where to bet it.
- **NBA Tonight:** the books' spread and total next to the model's projected margin and total, plus **alternate spreads** (`client.alternate_spreads`).
- **The daily job** saves every scraped line to the line history, so the models can retrain on real lines.

The database is found at `~/Desktop/oddsscraper/oddsscraper.db`. Set `ODDSSCRAPER_DB=/path/to/file.db` in `.env` if yours is elsewhere. Without `pip install -e`, props are still read directly from the database, but game lines and alt spreads need the package.

## Optional services

| Feature | Setup in `.env` |
|---|---|
| Live lines from every book (NBA and NFL), game totals, line shopping, line-move alerts | `ODDS_API_KEY` from https://the-odds-api.com. The free tier has 500 credits a month, and each pull costs about 1 credit per market per game: a full NFL week (11 markets × ~14 games) is about 150 credits, so the daily job only pulls NFL lines on game days. |
| Phone alerts for big edges and line moves | Install the **ntfy** app, subscribe to a hard-to-guess topic, set `NTFY_TOPIC` |
| Your own injury calls | `data/injuries_manual.csv` with columns `team,player,status` (status `Out` or `Doubtful` counts as out) |

## Daily job (`scripts/daily.py`)

Each run: refresh data → grade yesterday's picks → pull the injury report → pull lines and game totals (saved for training) → score today's slate (`data/slates/`) → **auto-track** the model's best picks (paper trading, so the tracker fills itself) → phone alerts → retrain on Mondays.

macOS: background jobs can't read `~/Desktop` without permission. If `data/daily.log` says "Operation not permitted", move the project to e.g. `~/Projects` and re-run the install script.

## Put it online

See **DEPLOY.md** for free hosting on Streamlit Community Cloud, so you can check it from your phone.

## Project layout

```
app.py              navigation: NBA / MLB / NFL
sports/nba.py       NBA dashboard
sports/mlb.py       MLB research page (from sports_stats_db)
sports/nfl.py       NFL research page (from sports_stats_db)
sports/odds_import.py  Odds CSV status page (from sports_stats_db)
scrapers/           sports_stats_db helpers: odds.csv reader, prop hit-rate analysis,
                    plus the oddsscraper reader for the research pages
sports/nfl_props.py NFL props dashboard (weather-aware model)
props/
  config.py         settings: season, stats, file paths, alert thresholds
  nba_com.py        NBA.com game logs + name matching
  scraper.py        Basketball-Reference (positions, schedule, fallback game logs)
  context.py        teammates out, pace/totals, usage, ESPN injuries
  analysis.py       hit rates, defense vs position
  model.py          features, minutes model, over/under models, batch scoring
  slate.py          scores lines and projects tonight's games
  odds.py           manual lines, Odds API, line shopping, line moves
  tracker.py        pick tracking and grading
  alerts.py         ntfy phone notifications
  matchups.py       NBA.com defender matchups
  teams.py          team names/abbreviations
  nfl_data.py       nflverse downloads: schedules + Vegas, stats, snaps, injuries
  nfl_weather.py    Open-Meteo archive + forecasts, stadium locations
  nfl_model.py      NFL features, models, weather impact
  nfl_slate.py      NFL week scoring + Odds API NFL props
scripts/            update_data, train_model, daily, install_daily_job.sh
data/ models/       data files and trained models
```

## Not included (yet)

- **First-basket props:** these depend on tip-off and play-by-play, which this data doesn't have.
- **Matchup data in the model:** NBA.com's matchup numbers are season totals that include the game being predicted, so feeding them to the model would let it peek at the answer. They're shown for research only.
- **Potential assists / tracking stats:** NBA.com has them, but they're a separate set of requests. They'd be a good next feature.
- **MLB** doesn't have a model yet.
- **NFL:** precipitation history starts in 2023, and bad-weather games are a small share of the data, so weather moves predictions by a few percent rather than a lot. Each Tuesday's retrain adds the new games.

## Try a different line
In any pick's **Why + hit rates** window, type the line (and optionally the over/under odds) your
book shows. The model re-runs at that line: chance of over, edge on each side, the full
explanation and the hit rates all update, and it warns you if the pick flips.

## Injuries: with vs without (NBA)
When a teammate who plays more minutes or uses more possessions is **out** tonight
(ESPN injury report), each player's projection uses this season's games without him
(`props/onoff.py`):
1. minutes, usage rate and per-minute production in games **with** vs **without** him
2. small samples are shrunk: weight = games / (games + 4)
3. starting from the player's last 10 games, move that weight toward his "without" numbers
   → boost vs recent form (comparing to recent form avoids double counting when the
   star has already been out for most of those games)
4. model projection × boost; the difference, divided by his game-to-game spread, moves
   the probability on the normal curve (capped at 0.75 standard deviations)
Every NBA pick's reasoning shows these numbers in plain English. Keep the data fresh with
`python scripts/update_data.py` so recent absences are counted.
