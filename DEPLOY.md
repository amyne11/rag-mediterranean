# Deploying to Streamlit Cloud

This walks through getting the demo onto a public URL like
`https://your-app-name.streamlit.app`. Time: ~15 minutes.

## Prerequisites

Before you start:

1. The pipeline runs locally — `streamlit run app/app.py` works on your laptop.
2. You've built the index at least once, so `artifacts/chunks_overlapping.json`
   and `artifacts/embeddings_BAAI__bge-small-en-v1.5.pkl` exist.
3. You have a GitHub account and your `GEMINI_API_KEY` handy.

## Step 1 — push to GitHub

Streamlit Cloud deploys from a public GitHub repo, so the code needs to be
there.

```bash
# From the rag-mediterranean directory
git init
git add .
git commit -m "Initial commit — RAG culinary demo"

# Create a new repo on github.com (call it whatever you like, e.g. rag-mediterranean)
# Then connect and push:
git remote add origin https://github.com/YOUR_USERNAME/rag-mediterranean.git
git branch -M main
git push -u origin main
```

**Verify the artifacts got committed.** Look at the repo on github.com — you
should see an `artifacts/` folder with two files inside (a `.json` and a
`.pkl`). If `artifacts/` isn't there, the demo won't work in the cloud.
If you see it locally but not on GitHub, your old `.gitignore` may have
excluded it before this update — run:

```bash
git add -f artifacts/
git commit -m "Add prebuilt index artifacts"
git push
```

The data files (`data/Background_Corpus_All.zip`, `data/qa_benchmark_dataset.json`)
are intentionally NOT pushed — Streamlit Cloud doesn't need them, only the
prebuilt artifacts.

## Step 2 — connect Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/rag-mediterranean`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
   - **App URL** (optional): pick something memorable, e.g. `amine-rag-mediterranean`
5. Click **Advanced settings** -> **Secrets** and paste:

   ```toml
   GEMINI_API_KEY = "AIza_your_actual_key_here"
   ```

   (Replace with your real key. The quotes are required.)

6. Click **Deploy**.

## Step 3 — wait, then verify

The first deploy installs every dependency from scratch. Expect:

- ~3-5 minutes for the build
- ~30 seconds for the first request after deploy (the embedding model
  downloads on cold start)
- Subsequent requests are fast

If the build succeeds and the app loads, paste a question. The first
generation call should return in 1-2 seconds.

## Common deployment issues

**Build fails with "ModuleNotFoundError: No module named 'rag_culinary'".**
Streamlit Cloud needs to know your code is a package. The `pyproject.toml`
handles this — make sure it's in the repo root.

**Build succeeds but app shows "Artifacts not found".**
The `artifacts/` folder didn't get committed to GitHub. See the verify step above.

**App loads but says "GEMINI_API_KEY not found".**
You missed the Secrets step. Go to the app's settings page on
share.streamlit.io and add it under Secrets.

**App OOMs during model load.**
The free tier has a 1 GB memory limit. The BGE embedding model uses ~250 MB,
which leaves enough headroom — but if it bites you, the workaround is
switching to query-side embedding via an API rather than loading a local
encoder. Tell me if this happens.

**Rate-limited.**
Free Gemini tier is 1500 requests/day. If the demo gets popular and hits
this, the answer goes through but errors out. Easy fix: add a Groq key as a
fallback (set both `GEMINI_API_KEY` and `GROQ_API_KEY` in secrets and switch
backends in `config.yaml`).

## Once it's live

Add the URL to your CV / portfolio. Recruiters can click and play with it
without setting anything up. That's the whole point of this exercise.
