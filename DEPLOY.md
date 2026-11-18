# Put the app online (free) with Streamlit Community Cloud

Then you can open it from your phone. The app reads the data and models
committed to the repo, so run `python scripts/daily.py` (or update + train) on
your Mac and push to refresh it.

1. **Make a GitHub repo** (private is fine):
   ```bash
   git init && git add . && git commit -m "Sports props hub"
   gh repo create sports-props-hub --private --source . --push
   ```
   (No `gh`? Create the repo on github.com, then follow its "push an existing repository" steps.)
2. Go to **https://share.streamlit.io** → sign in with GitHub → **Create app**.
3. Pick the repo, branch `main`, main file **`app.py`** → **Deploy**.
4. In the app's **Settings → Secrets**, paste your keys (same names as `.env`):
   ```toml
   ODDS_API_KEY = "your-key"
   NTFY_TOPIC = "your-topic"
   ```
5. To refresh the online app: run the daily job on your Mac, then
   `git add data models && git commit -m "data" && git push`. The app reloads by itself.

Notes
- Private apps on the free tier can only be seen by you (and people you invite).
- Don't commit `.env`. It's already in `.gitignore`.
- The free tier sleeps after a few days without visits. Opening the link wakes it in about 30 seconds.
