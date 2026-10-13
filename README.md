# 🚚 Logistics Radar

A free background tool that reads Indian and global logistics news every morning, uses a free
LLM to pick out what matters for you, and sends it to your phone or inbox.

- **Daily, 08:00 IST:** only high-importance items (score ≥ 8), such as competitor moves, funding, launches and regulation
- **Weekly, Monday 09:00 IST:** everything notable (score ≥ 6), grouped by category, with an AI "Trends & signals + Actions" summary

```
GitHub Actions cron → 44 RSS / Google News feeds → dedupe → Gemini (or Groq) scores & explains
                    → Telegram and/or Email → state saved back to the repo (data/)
```

**Cost: ₹0.** GitHub Actions free minutes (~2–3 min per run), Gemini free tier, Telegram and Gmail.

---

## Setup (about 20 minutes)

### 1. Get a free LLM key (at least one; both = automatic fallback)
- **Gemini** (recommended): https://aistudio.google.com/apikey → *Create API key*
- **Groq** (backup): https://console.groq.com/keys → *Create API key*

### 2. Pick delivery (one or both)

**Telegram** (recommended; instant on your phone)
1. In Telegram, open **@BotFather** → `/newbot` → pick a name → copy the **bot token**.
2. Open your new bot and send it any message (e.g. "hi").
3. Get your chat ID:
   ```bash
   TELEGRAM_BOT_TOKEN=<your-token> .venv/bin/python radar.py telegram-chat-id
   ```
   (For a team, add the bot to a Telegram group, send a message there, and use the group's negative chat ID.)

**Email (Gmail)**
1. Turn on 2-Step Verification on your Google account.
2. Create an **App Password**: https://myaccount.google.com/apppasswords
3. You'll use your Gmail address plus that 16-character app password (not your normal password).

### 3. Put it on GitHub
Create a **private** repo (e.g. `logistics-radar`) and push this folder:
```bash
git init && git add . && git commit -m "Logistics Radar" && gh repo create logistics-radar --private --source . --push
```

### 4. Add secrets
Repo → **Settings → Secrets and variables → Actions → New repository secret**. Add only the ones you use:

| Secret | Value |
|---|---|
| `GEMINI_API_KEY` | from step 1 |
| `GROQ_API_KEY` | optional backup |
| `TELEGRAM_BOT_TOKEN` | from BotFather |
| `TELEGRAM_CHAT_ID` | from step 2 |
| `EMAIL_ADDRESS` | your Gmail address |
| `EMAIL_APP_PASSWORD` | Gmail app password |
| `EMAIL_TO` | optional; comma-separated recipients (defaults to yourself) |

### 5. Test it
Repo → **Actions → Logistics Radar → Run workflow**:
- choose `test-send` to check delivery
- then `daily`, and later `weekly`

After that it runs on its own. The first real weekly digest has more content once a few daily runs have built up history.

---

## Customising: edit `sources.yaml` only

> The bundled `sources.yaml` is a generic example. To keep your real profile and goals private on a public repo, paste the full YAML into a repo secret named `RADAR_CONFIG`; it overrides the file at runtime.

- **`profile`**: describe your company and interests. This controls what the LLM treats as important.
- **`google_news`**: add any search, e.g. a new competitor, a big customer (`Flipkart logistics`), or a theme (`drone delivery India`). Google operators work: `OR`, `"exact phrase"`.
- **`feeds`**: add any RSS URL. Set `filter: true` for broad feeds so only keyword matches reach the LLM.
- **`settings`**: thresholds (`daily_min_score`, `weekly_min_score`), item caps, and `send_empty_daily`.
- **Schedule**: edit the cron lines in `.github/workflows/radar.yml` (times are UTC; IST = UTC + 5:30).

## Running locally
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=...          # optional; without it, a crude keyword score is used
.venv/bin/python radar.py daily --dry-run    # prints the digest, sends/saves nothing
.venv/bin/python radar.py weekly --dry-run
```

## Notes
- **Quota:** each run scores up to 200 new items in batches of 20, so about 10 LLM calls, which is well inside Gemini's free daily limit. Items from a failed batch are retried on the next run.
- **Keep-alive:** GitHub pauses scheduled workflows after 60 days without repo activity. The bot commits `data/` on every run, which counts as activity.
- **Cron delay:** GitHub may start scheduled runs 5–30 minutes late at busy times. That's normal.
- **Model names:** override with `GEMINI_MODEL` / `GROQ_MODEL` env vars if Google or Groq rename their free models.
