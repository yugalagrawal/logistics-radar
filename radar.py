#!/usr/bin/env python3
"""Logistics Radar: fetch industry news, score it with a free LLM, send a digest.

Usage:
  python radar.py daily            # fetch + score new items, send high-importance alerts
  python radar.py weekly           # fetch + score, then send the weekly digest
  python radar.py daily --dry-run  # print instead of sending, don't save state
  python radar.py test-send        # send a test message to every configured channel
  python radar.py telegram-chat-id # print chat IDs that have messaged your bot
"""
import argparse
import hashlib
import html
import json
import os
import re
import smtplib
import sys
import time
from calendar import timegm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import feedparser
import requests
import yaml

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SEEN_FILE = DATA / "seen.json"
ITEMS_FILE = DATA / "items.json"

# RADAR_CONFIG (full YAML, e.g. from a GitHub secret) overrides the bundled example config.
CONFIG = yaml.safe_load(os.environ.get("RADAR_CONFIG") or (ROOT / "sources.yaml").read_text())
S = CONFIG.get("settings", {})
TZ = ZoneInfo(S.get("timezone", "Asia/Kolkata"))
CATEGORIES = CONFIG.get("categories", ["Other"])
GOALS = CONFIG.get("growth_goals", {})  # key -> {label, subtitle, meaning}
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
NOW = datetime.now(timezone.utc)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- storage

def load_json(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, obj):
    DATA.mkdir(exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- fetching

def google_news_url(query):
    q = quote_plus(f"{query} when:2d")
    return f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"


def feed_list():
    feeds = [{"name": f"Google News: {q}", "url": google_news_url(q), "google": True}
             for q in CONFIG.get("google_news", [])]
    feeds += CONFIG.get("feeds", [])
    return feeds


def clean_text(s, limit=300):
    s = html.unescape(re.sub(r"<[^>]+>", " ", s or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def item_key(title):
    # Google News titles end with " - Publisher"; strip it so the same story
    # from different feeds dedupes to one key.
    t = re.sub(r"\s+-\s+[^-]{2,60}$", "", title).lower()
    t = re.sub(r"[^a-z0-9]+", "", t)
    return hashlib.sha1(t.encode()).hexdigest()[:16]


KEYWORD_RES = [
    re.compile(r"\b" + re.escape(k.lower()) + (r"\b" if len(k) <= 3 else ""))
    for k in CONFIG.get("keywords", [])
]


def matches_keywords(text):
    text = text.lower()
    return any(r.search(text) for r in KEYWORD_RES)


EXCLUDE_RES = [re.compile(p, re.I) for p in CONFIG.get("exclude_patterns", [])]


def is_excluded(title):
    return any(r.search(title) for r in EXCLUDE_RES)


def fetch_feed(feed):
    try:
        r = requests.get(feed["url"], headers={"User-Agent": UA}, timeout=25)
        r.raise_for_status()
        parsed = feedparser.parse(r.content)
    except Exception as e:
        log(f"  ! {feed['name']}: {e}")
        return []

    cutoff = NOW - timedelta(hours=S.get("lookback_hours", 36))
    out = []
    for e in parsed.entries:
        title = clean_text(e.get("title"), 250)
        link = e.get("link")
        if not title or not link:
            continue
        ts = e.get("published_parsed") or e.get("updated_parsed")
        published = datetime.fromtimestamp(timegm(ts), timezone.utc) if ts else None
        if published and published < cutoff:
            continue
        if is_excluded(title):
            continue
        summary = "" if feed.get("google") else clean_text(e.get("summary"))
        if feed.get("filter") and not matches_keywords(f"{title} {summary}"):
            continue
        source = (e.get("source") or {}).get("title") if feed.get("google") else feed["name"]
        out.append({
            "key": item_key(title),
            "title": title,
            "link": link,
            "source": source or feed["name"],
            "summary": summary,
            "published": (published or NOW).isoformat(),
        })
    return out


def fetch_all():
    feeds = feed_list()
    log(f"Fetching {len(feeds)} feeds...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch_feed, feeds))
    items = {}
    for batch in results:
        for it in batch:
            items.setdefault(it["key"], it)
    log(f"  {len(items)} unique recent items")
    return list(items.values())


# ---------------------------------------------------------------- LLM

def with_retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code in (429, 500, 502, 503, 504) and i < tries - 1:
                wait = 15 * (i + 1)
                log(f"  LLM HTTP {code}, retrying in {wait}s")
                time.sleep(wait)
                continue
            raise


def gemini(prompt, json_mode):
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    cfg = {"temperature": 0.2}
    if json_mode:
        cfg["responseMimeType"] = "application/json"

    def call():
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
            json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg},
            timeout=180,
        )
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    return with_retry(call)


GROQ_MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]


def groq(prompt, json_mode):
    # Groq retires/re-tiers models often; on a 400/404 fall through to the next attempt.
    # Strict JSON mode sometimes fails validation, so each model gets a second, lenient
    # try without it (parse_json extracts the object from plain text).
    models = [os.environ["GROQ_MODEL"]] if os.getenv("GROQ_MODEL") else GROQ_MODELS
    attempts = [(m, strict) for m in models for strict in ([True, False] if json_mode else [False])]
    last_err = None
    for model, strict in attempts:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        if strict:
            body["response_format"] = {"type": "json_object"}

        def call():
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
                json=body, timeout=180,
            )
            if r.status_code in (400, 404):
                raise RuntimeError(f"{model}: HTTP {r.status_code} {r.text[:200]}")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        try:
            return with_retry(call)
        except (RuntimeError, requests.HTTPError) as e:
            log(f"  groq attempt failed ({'strict JSON' if strict else 'plain'}), trying next: {e}")
            last_err = e
    raise last_err


def llm_available():
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY"))


def call_llm(prompt, json_mode=True):
    """Try Gemini first, fall back to Groq."""
    errors = []
    for env, fn in (("GEMINI_API_KEY", gemini), ("GROQ_API_KEY", groq)):
        if os.getenv(env):
            try:
                return fn(prompt, json_mode)
            except Exception as e:
                errors.append(f"{fn.__name__}: {e}")
    raise RuntimeError("; ".join(errors) or "no LLM key configured")


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


SCORE_PROMPT = """You are a sharp logistics-industry analyst writing a news radar for one reader.

READER PROFILE:
{profile}

For EACH news item below, return:
- "id": the item's id
- "score": integer 1-10 importance for this reader. Be strict: judge against the reader's
  IN SCOPE / OUT OF SCOPE lists above. Out-of-scope items score 1-3 even if they are big news.
    9-10 = in scope AND directly relevant: a new business model, VAS, product or tech in B2B
           trucking/freight; a direct competitor's launch, funding, expansion or pricing move;
           major road-freight regulation
    7-8  = in scope, clearly useful for a trucking/freight product manager
    5-6  = in scope but minor or general
    1-4  = out of scope, generic, stock-price, press-release fluff, or only loosely related
- "category": exactly one of {categories}
- "section": "innovation" if the item is about something NEW in the reader's space - a new
    business model, value-added service, product/feature launch, technology, AI, automation,
    EV fleet model, patent or pilot; otherwise "industry" (company/market news, funding,
    results, deals, customers, policy, prices)
- "region": "india" if the item is mainly about India or Indian companies, else "global"
- "theme": which of the reader's growth goals this item gives info, ideas, peer moves or market
    signals for - one of {theme_keys}, or "none". Use "none" unless it clearly helps that goal:
{themes}
- "why": at most 15 words, for a product manager at the reader's company: a concrete takeaway
    or idea for us (not a restatement of the headline). If theme is not "none", frame it as a
    move for that goal.

Return JSON only: {{"items": [{{"id": ..., "score": ..., "category": ..., "section": ..., "region": ..., "theme": ..., "why": ...}}, ...]}}

ITEMS:
{items}"""


def score_batch(batch):
    payload = [{"id": i, "title": it["title"], "source": it["source"], "snippet": it["summary"][:200]}
               for i, it in enumerate(batch)]
    prompt = SCORE_PROMPT.format(profile=CONFIG.get("profile", "").strip(),
                                 categories=json.dumps(CATEGORIES),
                                 theme_keys=json.dumps(list(GOALS)),
                                 themes="\n".join(f'    "{k}" = {g["meaning"].strip()}'
                                                  for k, g in GOALS.items()),
                                 items=json.dumps(payload, ensure_ascii=False, indent=0))
    result = parse_json(call_llm(prompt))
    by_id = {int(r["id"]): r for r in result.get("items", []) if str(r.get("id", "")).isdigit()}
    scored = []
    for i, it in enumerate(batch):
        r = by_id.get(i)
        if not r:
            continue
        cat = r.get("category") if r.get("category") in CATEGORIES else "Other"
        item = {**it, "score": int(r.get("score", 0)), "category": cat,
                "section": r.get("section"), "region": r.get("region"),
                "theme": r.get("theme") if r.get("theme") in GOALS else "none",
                "why": str(r.get("why", "")).strip()}
        item["section"], item["region"] = section_of(item), region_of(item)
        scored.append(item)
    return scored


INNOVATION_CATEGORIES = {"Product & VAS", "Tech & Automation", "Patent & R&D", "Tech & Innovation"}
INDIA_RE = re.compile(r"\b(india|indian|bharat|rs|crore|lakh|delhi|ncr|mumbai|bengaluru|bangalore|"
                      r"chennai|hyderabad|kolkata|pune|ahmedabad|gujarat|maharashtra|karnataka|"
                      r"tamil nadu|assam|nse|bse|sebi|morth|niti)\b|₹", re.I)


def section_of(it):
    if it.get("section") in ("industry", "innovation"):
        return it["section"]
    return "innovation" if it.get("category") in INNOVATION_CATEGORIES else "industry"


def region_of(it):
    if it.get("region") in ("india", "global"):
        return it["region"]
    return "india" if INDIA_RE.search(f"{it['title']} {it.get('source', '')}") else "global"


def keyword_score(items):
    """Crude fallback when no LLM key is configured."""
    out = []
    for it in items:
        score = 6 if matches_keywords(it["title"]) else 3
        item = {**it, "score": score, "category": "Other", "why": ""}
        item["section"], item["region"] = section_of(item), region_of(item)
        out.append(item)
    return out


def score_items(items):
    if not llm_available():
        log("No GEMINI_API_KEY / GROQ_API_KEY set - using keyword scoring fallback.")
        return keyword_score(items)
    size = S.get("batch_size", 20)
    scored = []
    batches = [items[i:i + size] for i in range(0, len(items), size)]
    for n, batch in enumerate(batches, 1):
        log(f"  scoring batch {n}/{len(batches)}")
        try:
            scored += score_batch(batch)
        except Exception as e:
            # Big batches sometimes come back as broken JSON; retry as two halves.
            log(f"  ! batch {n} failed ({e}); retrying in halves")
            for half in (batch[:len(batch) // 2], batch[len(batch) // 2:]):
                try:
                    scored += score_batch(half)
                except Exception as e2:
                    # Unscored items stay unseen, so they get retried on the next run.
                    log(f"  ! half-batch failed: {e2}")
        if n < len(batches):
            time.sleep(5)  # stay under free-tier requests-per-minute limits
    return scored


# ---------------------------------------------------------------- pipeline

def collect(dry_run):
    """Fetch, dedupe against history, score, and store. Returns newly scored items."""
    seen = load_json(SEEN_FILE, {})
    stored = load_json(ITEMS_FILE, [])

    new = [it for it in fetch_all() if it["key"] not in seen]
    new.sort(key=lambda it: it["published"], reverse=True)
    new = new[:S.get("max_items_per_run", 200)]
    log(f"  {len(new)} new items to score")

    scored = score_items(new) if new else []

    now_iso = NOW.isoformat()
    for it in scored:
        seen[it["key"]] = now_iso
    stored += [{**it, "scored_at": now_iso} for it in scored if it["score"] >= 4]

    # prune history
    seen_cutoff = (NOW - timedelta(days=45)).isoformat()
    seen = {k: v for k, v in seen.items() if v >= seen_cutoff}
    keep_cutoff = (NOW - timedelta(days=S.get("keep_days", 30))).isoformat()
    stored = [it for it in stored if it["scored_at"] >= keep_cutoff]

    if not dry_run:
        save_json(SEEN_FILE, seen)
        save_json(ITEMS_FILE, stored)
    return scored, stored


def rescore_recent(dry_run, days=7):
    """Re-score stored, unsent items from the last `days` with the current prompt and goals
    (e.g. after changing the profile or growth_goals). Keeps scored_at and sent flags."""
    stored = load_json(ITEMS_FILE, [])
    cutoff = (NOW - timedelta(days=days)).isoformat()
    todo = [it for it in stored if it["scored_at"] >= cutoff and not it.get("sent")]
    log(f"Re-scoring {len(todo)} unsent items from the last {days} days")
    fresh = {it["key"]: it for it in score_items(todo)}
    for i, it in enumerate(stored):
        if it["key"] in fresh:
            stored[i] = {**fresh[it["key"]], "scored_at": it["scored_at"]}
    if not dry_run:
        save_json(ITEMS_FILE, stored)
    log(f"  updated {len(fresh)} items")


STOPWORDS = set("a an the to in into of for on by and or with as at is its it from after over "
                "report reports says plays up new india india's indian market markets forecast size "
                "growth share trends outlook analysis logistics vehicle vehicles".split())


def title_words(title):
    t = re.sub(r"\s+-\s+[^-]{2,60}$", "", title).lower()
    return {w for w in re.findall(r"[a-z0-9]+", t)
            if w not in STOPWORDS and not re.fullmatch(r"(19|20)\d\d", w)}


def dedupe_similar(items):
    """Drop items that are the same story from another outlet. Expects items sorted best-first."""
    kept = []
    for it in items:
        words = title_words(it["title"])
        dup = False
        for k in kept:
            common = words & k["_words"]
            if len(common) >= 3 and len(common) / max(1, min(len(words), len(k["_words"]))) >= 0.5:
                dup = True
                break
        if not dup:
            kept.append({**it, "_words": words})
    return [{k: v for k, v in it.items() if k != "_words"} for it in kept]


def local_date(iso=None):
    dt = datetime.fromisoformat(iso) if iso else NOW
    return dt.astimezone(TZ).strftime("%d %b")


def esc(s):
    return html.escape(s or "", quote=True)


DIVIDER = "━━━━━━━━━━━━━━━━━━"
LEGEND = f"{DIVIDER}\n<i>🔥 must-read · ⭐ notable · ▫️ on the radar</i>"

def theme_match(goal):
    return lambda it: it.get("theme") == goal


# Display order. (key, title, subtitle, [(subsection key, label, filter)])
SECTIONS = [
    ("innovation", "💡 INNOVATION RADAR", "New business models · VAS · tech", [
        ("innovation", None, lambda it: it["section"] == "innovation"),
    ]),
    ("growth", "🎯 GROWTH PLAYBOOK", "Signals for your growth goals", [
        (k, f"{g['label']}\n<i>{g['subtitle']}</i>", theme_match(k)) for k, g in GOALS.items()
    ]),
    ("market", "📰 MARKET PULSE", "Trucking · B2B freight · competitors · policy", [
        ("india", "🇮🇳 India", lambda it: it["section"] == "industry" and it["region"] == "india"),
        ("global", "🌍 Global", lambda it: it["section"] == "industry" and it["region"] == "global"),
    ]),
]
# Pick order: goal-specific subsections claim their stories first, so nothing repeats.
PICK_ORDER = ["growth", "innovation", "market"]


def importance_icon(score):
    return "🔥" if score >= 9 else "⭐" if score >= 7 else "▫️"


def render_item(n, it):
    title = it["title"]
    if it.get("source") and title.endswith(f" - {it['source']}"):
        title = title[: -len(it["source"]) - 3]
    head = (f"<b>{n}.</b> {importance_icon(it['score'])} "
            f"<a href=\"{esc(it['link'])}\">{esc(title)}</a> <i>· {esc(it['source'])}</i>")
    return f"{head}\n↳ {esc(it['why'])}" if it.get("why") else head


CLUSTER_PROMPT = """Below are news headlines as "id: headline". Some report the SAME underlying
story (same event, deal or announcement) in different words. Group those together.
Return JSON only: {{"groups": [[id, id, ...], ...]}} listing only groups with 2 or more ids.

{items}"""


def drop_same_story(items):
    """Ask the LLM which headlines are the same story; keep the first (best-scored) of each group."""
    if len(items) < 2 or not llm_available():
        return items
    lines = "\n".join(f"{i}: {it['title']}" for i, it in enumerate(items))
    try:
        groups = parse_json(call_llm(CLUSTER_PROMPT.format(items=lines))).get("groups", [])
    except Exception as e:
        log(f"  ! same-story check failed: {e}")
        return items
    drop = set()
    for g in groups:
        ids = sorted(int(x) for x in g if str(x).isdigit() and int(x) < len(items))
        drop.update(ids[1:])
    log(f"  same-story check dropped {len(drop)} items")
    return [it for i, it in enumerate(items) if i not in drop]


def pick_sections(pool, cfg):
    """Shortlist generously, drop same-story duplicates with the LLM, then pick for real."""
    shortlist = pick_sections_once(pool, cfg, slack=5)
    seen, cands = set(), []
    for it in sorted((it for items in shortlist.values() for it in items),
                     key=lambda it: it["score"], reverse=True):
        if it["key"] not in seen:
            seen.add(it["key"])
            cands.append(it)
    return pick_sections_once(drop_same_story(cands), cfg)


def pick_sections_once(pool, cfg, slack=0):
    """Choose items per subsection: all must-reads (up to max), topped up to min with the best rest.

    Items with "_recent": False (older unsent stories) only qualify for the Growth Playbook.
    `slack` widens each subsection's max, for building a shortlist.
    """
    pool = dedupe_similar(sorted(
        ({**it, "section": section_of(it), "region": region_of(it)} for it in pool
         if it["score"] >= cfg.get("filler_min_score", 4) and not is_excluded(it["title"])),
        key=lambda it: it["score"], reverse=True))
    # variety: at most 2 stories led by the same company/word ("Tesla ...", "Amazon ...")
    per_lead, varied = {}, []
    for it in pool:
        lead = next(iter(re.findall(r"[A-Za-z0-9]+", it["title"])), "").lower()
        per_lead[lead] = per_lead.get(lead, 0) + 1
        if per_lead[lead] <= 2 or lead in STOPWORDS:
            varied.append(it)
    pool = varied
    picked, used = {}, set()
    by_key = {s[0]: s for s in SECTIONS}
    for sec in PICK_ORDER:
        for key, _, match in by_key[sec][3]:
            lim = cfg["sections"].get(key, {"min": 2, "max": 3})
            cands = [it for it in pool if it["key"] not in used and match(it)
                     and (sec == "growth" or it.get("_recent", True))]
            must = sum(1 for it in cands if it["score"] >= cfg.get("must_read_score", 8))
            picked[key] = cands[:min(lim["max"] + slack, max(lim["min"] + slack, must))]
            used.update(it["key"] for it in picked[key])
    return picked


def render_sections(picked):
    blocks = []
    for _, title, subtitle, subs in SECTIONS:
        blocks.append(f"{DIVIDER}\n<b>{title}</b>\n<i>{subtitle}</i>")
        for key, label, _ in subs:
            items = picked[key]
            body = [render_item(n, it) for n, it in enumerate(items, 1)] or ["<i>Nothing notable today.</i>"]
            if label:
                body[0] = f"<b>{label}</b>\n\n{body[0]}"
            blocks += body
    blocks.append(LEGEND)
    return blocks


DAILY_PROMPT = """You are briefing a product manager. Reader profile:
{profile}

The company's growth goals:
{themes}

Below are today's selected logistics news items (section/region/theme, score, headline, takeaway).
Return JSON only:
{{"idea": "one concrete, specific product / VAS / business-model idea for our team, inspired by
           today's items and ideally serving one of the growth goals; name the item it builds on;
           under 30 words"}}

ITEMS:
{items}"""


def item_lines(picked, limit=40):
    rows = [it for items in picked.values() for it in items]
    return "\n".join(f"[{it['section']}/{it['region']}/{it.get('theme', 'none')}] [{it['score']}] "
                     f"{it['title']} — {it.get('why', '')}" for it in rows[:limit])


def build_daily(scored, stored):
    cfg = S["daily"]
    recent = (NOW - timedelta(hours=48)).isoformat()
    week = (NOW - timedelta(days=7)).isoformat()
    new_keys = {it["key"] for it in scored}
    # today's items, plus earlier unsent ones: last 48h for every section, last 7 days
    # for the Growth Playbook (goal-specific news is rarer and stays useful longer)
    pool = [{**it, "_recent": True} for it in scored] + [
        {**it, "_recent": it["scored_at"] >= recent} for it in stored
        if it["key"] not in new_keys and not it.get("sent") and it["scored_at"] >= week]
    picked = pick_sections(pool, cfg)
    header = f"<b>🚚 LOGISTICS RADAR</b>\n<i>Daily brief · {NOW.astimezone(TZ):%a, %d %b %Y}</i>"
    if not any(picked.values()):
        return header, [esc(S.get("empty_daily_message", "No news today."))], []

    blocks = []
    if llm_available():
        try:
            r = parse_json(call_llm(DAILY_PROMPT.format(
                profile=CONFIG.get("profile", "").strip(), items=item_lines(picked),
                themes="\n".join(f"- {k}: {g['meaning'].strip()}" for k, g in GOALS.items()))))
            if r.get("idea"):
                blocks.append(f"💭 <b>Product thought:</b> {esc(r['idea'])}")
        except Exception as e:
            log(f"  ! daily summary failed: {e}")
    blocks += render_sections(picked)
    return header, blocks, [it["key"] for items in picked.values() for it in items]


TRENDS_PROMPT = """You are a logistics-industry strategy analyst briefing a product manager. Reader profile:
{profile}

Below are this week's most important news items (section, score, headline, why it matters).
Write, in plain text (no markdown headings, no bold):
- 3-4 bullet lines starting with "- " on the key trends in B2B trucking / freight this week,
  especially new business models, VAS and innovation
- then a line "Product actions:" followed by 2 bullet lines starting with "- " suggesting concrete
  product, VAS or business-model moves the reader's team could consider

Be specific, name companies, keep each bullet under 20 words.

ITEMS:
{items}"""


def build_weekly(stored):
    cfg = S["weekly"]
    week_cutoff = (NOW - timedelta(days=7)).isoformat()
    picked = pick_sections([it for it in stored if it["scored_at"] >= week_cutoff], cfg)
    header = (f"<b>📊 LOGISTICS RADAR · WEEKLY</b>\n"
              f"<i>{local_date(week_cutoff)} – {local_date()} {NOW.astimezone(TZ):%Y}</i>")
    if not any(picked.values()):
        return header, ["No notable items this week."]

    blocks = []
    if llm_available():
        try:
            trends = call_llm(TRENDS_PROMPT.format(profile=CONFIG.get("profile", "").strip(),
                                                   items=item_lines(picked, 60)), json_mode=False)
            text = re.sub(r"(?m)^\s*[-*•]\s+", "• ", esc(trends.strip()))
            text = text.replace("Product actions:", "\n🎯 <b>Product actions</b>")
            blocks.append(f"<b>🔭 Week in review</b>\n{text}")
        except Exception as e:
            log(f"  ! trends summary failed: {e}")
    return header, blocks + render_sections(picked)


# ---------------------------------------------------------------- delivery

def visible_len(s):
    # Telegram's 4096 limit counts the rendered text, not tags or link URLs
    return len(html.unescape(re.sub(r"<[^>]+>", "", s)))


def chunk_messages(header, blocks, limit=3900):
    msgs, cur = [], header
    for b in blocks:
        if visible_len(cur) + visible_len(b) + 2 > limit:
            msgs.append(cur)
            cur = b
        else:
            cur += "\n\n" + b
    msgs.append(cur)
    return msgs


def send_telegram(header, blocks):
    # TELEGRAM_CHAT_ID may list several chats/channels, comma-separated
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chats = [c.strip() for c in os.getenv("TELEGRAM_CHAT_ID", "").split(",") if c.strip()]
    if not (token and chats):
        return False
    msgs = chunk_messages(header, blocks)
    failed = []
    for chat in chats:
        for msg in msgs:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={
                "chat_id": chat, "text": msg, "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }, timeout=30)
            if not r.ok:
                log(f"  ! telegram {chat}: {r.text}")
                failed.append(chat)
                break
            time.sleep(1)
    if len(failed) == len(chats):
        raise RuntimeError("Telegram delivery failed for every chat")
    return True


def send_email(header, blocks):
    user, pwd = os.getenv("EMAIL_ADDRESS"), os.getenv("EMAIL_APP_PASSWORD")
    if not (user and pwd):
        return False
    to = [a.strip() for a in os.getenv("EMAIL_TO", user).split(",") if a.strip()]
    body = "".join(
        f"<div style='margin:0 0 16px'>{b.replace(chr(10), '<br>')}</div>" for b in blocks)
    page = (f"<div style='font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:680px;"
            f"font-size:14px;line-height:1.5;color:#222'><h2 style='font-size:18px'>"
            f"{header.replace(chr(10), '<br>')}</h2>"
            f"{body}</div>")
    msg = MIMEText(page, "html", "utf-8")
    msg["Subject"] = html.unescape(re.sub(r"<[^>]+>", "", header).replace("\n", " · "))
    msg["From"], msg["To"] = user, ", ".join(to)
    with smtplib.SMTP_SSL(os.getenv("SMTP_HOST", "smtp.gmail.com"), 465, timeout=60) as s:
        s.login(user, pwd)
        s.sendmail(user, to, msg.as_string())
    return True


def deliver(header, blocks, dry_run):
    if dry_run:
        text = "\n\n".join([header] + blocks)
        print(html.unescape(re.sub(r"<[^>]+>", "", text)))
        return
    sent = []
    for name, fn in (("telegram", send_telegram), ("email", send_email)):
        try:
            if fn(header, blocks):
                sent.append(name)
        except Exception as e:
            log(f"  ! {name} delivery failed: {e}")
    if not sent:
        raise SystemExit("No delivery channel succeeded. Set Telegram and/or email secrets.")
    log(f"Delivered via {', '.join(sent)}")


# ---------------------------------------------------------------- CLI

def telegram_chat_id():
    token = os.getenv("TELEGRAM_BOT_TOKEN") or input("Bot token: ").strip()
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=30).json()
    chats = {}
    for u in r.get("result", []):
        # private chats/groups send "message"; channels send "channel_post" or "my_chat_member"
        for kind in ("message", "channel_post", "my_chat_member"):
            if kind in u:
                c = u[kind]["chat"]
                chats[c["id"]] = f"{c['type']}: {c.get('title') or c.get('first_name')}"
    if not chats:
        print("No messages yet. Message your bot (or post in the channel it's admin of), then re-run.")
    for cid, name in chats.items():
        print(f"chat_id={cid}  ({name})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["daily", "weekly", "test-send", "telegram-chat-id", "rescore"])
    ap.add_argument("--dry-run", action="store_true", help="print output, don't send or save")
    ap.add_argument("--preview", action="store_true",
                    help="send only to the first TELEGRAM_CHAT_ID and don't mark items as sent")
    args = ap.parse_args()

    if args.preview:
        os.environ["TELEGRAM_CHAT_ID"] = os.getenv("TELEGRAM_CHAT_ID", "").split(",")[0].strip()

    if args.mode == "telegram-chat-id":
        return telegram_chat_id()
    if args.mode == "test-send":
        return deliver("<b>✅ Logistics Radar test</b>", ["Delivery is working."], args.dry_run)

    if args.mode == "rescore":
        return rescore_recent(args.dry_run)

    scored, stored = collect(args.dry_run)
    if args.mode == "daily":
        header, blocks, sent_keys = build_daily(scored, stored)
        deliver(header, blocks, args.dry_run)
        if sent_keys and not (args.dry_run or args.preview):
            sent_keys = set(sent_keys)
            for it in stored:
                if it["key"] in sent_keys:
                    it["sent"] = True
            save_json(ITEMS_FILE, stored)
    else:
        deliver(*build_weekly(stored), args.dry_run)


if __name__ == "__main__":
    main()
