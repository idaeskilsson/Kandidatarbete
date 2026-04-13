"""
Scraper för Social Blade – hämtar historisk månadsstatistik per kanal.

Använder Social Blades interna tRPC-API (youtube.monthly) via en Selenium-
session för att kringgå Cloudflare. Data hämtas som JSON, ingen HTML-parsing.

Läser:  data/raw/channels_raw.csv
Sparar: data/raw/channel_history.csv
"""

import os
import time
import json
import pandas as pd
import numpy as np
from urllib.parse import quote
import undetected_chromedriver as uc


SOCIALBLADE_PAGE = "https://socialblade.com/youtube/channel/{channel_id}/monthly"
SOCIALBLADE_API  = "https://socialblade.com/api/trpc/youtube.monthly?batch=1&input={input}"

SLEEP_BETWEEN_REQUESTS = 3


# ==============================================================================
# HJÄLPFUNKTIONER
# ==============================================================================

def _make_driver() -> uc.Chrome:
    """Skapar en synlig Chrome-instans (krävs för att kringgå Cloudflare)."""
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return uc.Chrome(options=options, version_main=146)


def _warm_up_session(driver: uc.Chrome, channel_id: str):
    """Laddar sidan en gång så att Cloudflare-cookies sätts."""
    driver.get(SOCIALBLADE_PAGE.format(channel_id=channel_id))
    time.sleep(6)


def _fetch_monthly(driver: uc.Chrome, channel_id: str) -> list[dict]:
    """
    Anropar Social Blades tRPC-API direkt via fetch() i webbläsaren.
    Returnerar lista med månadsposter: {date, subscribers, views, videos}.
    """
    api_input = quote(json.dumps({"0": {"json": {"channelId": channel_id}}}))
    url = SOCIALBLADE_API.format(input=api_input)

    result = driver.execute_async_script(
        f"""
        const resp = await fetch("{url}");
        return await resp.json();
    """
    )

    try:
        data = result[0]["result"]["data"]["json"]["monthly"]
    except (KeyError, IndexError, TypeError):
        return []

    rows = []
    for r in data:
        rows.append({
            "date":        r.get("date"),
            "subscribers": r.get("subscribers", 0),
            "views":       r.get("views", 0),
            "videos":      r.get("videos", 0),
        })
    return rows


# ==============================================================================
# PIPELINE
# ==============================================================================

def run_social_blade_scraping():
    os.makedirs("data/raw", exist_ok=True)

    channels = pd.read_csv("data/raw/channels_raw.csv")
    channels["published_at"] = pd.to_datetime(channels["published_at"], errors="coerce")

    all_rows = []
    driver = None
    session_warmed = False

    for _, ch in channels.iterrows():
        channel_id = ch["channel_id"]

        try:
            if driver is None:
                driver = _make_driver()

            if not session_warmed:
                _warm_up_session(driver, channel_id)
                session_warmed = True

            print(f"Hämtar Social Blade-data för {channel_id}...")

            records = _fetch_monthly(driver, channel_id)

            if not records:
                print(f"  Inga månader hittades för {channel_id}")
                continue

            for rec in records:
                try:
                    month_ts = pd.to_datetime(rec["date"], format="%b %Y")
                except Exception:
                    continue

                age_days = (
                    (month_ts - ch["published_at"]).days
                    if pd.notna(ch["published_at"])
                    else 0
                )

                all_rows.append({
                    "channel_id":       channel_id,
                    "month":            month_ts.strftime("%b %Y"),
                    "subscribers":      rec.get("subscribers", 0),
                    "total_views":      rec.get("views", 0),
                    "video_count":      rec.get("videos", 0),
                    "likes":            0,
                    "comments":         0,
                    "channel_age_days": max(age_days, 0),
                    "country":          ch.get("country", ""),
                })

            print(f"  {len(records)} månader hämtade.")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        except Exception as e:
            print(f"  Misslyckades för {channel_id}: {e}")
            continue

    if driver:
        driver.quit()

    if not all_rows:
        print("\nIngen historikdata kunde hämtas.")
        return

    history_df = pd.DataFrame(all_rows)
    history_df = _merge_video_counts(history_df)

    history_df.to_csv("data/raw/channel_history.csv", index=False, encoding="utf-8-sig")
    print(f"\nSparat {len(history_df)} rader till data/raw/channel_history.csv")


# ==============================================================================
# MERGE VIDEO_COUNT, LIKES OCH COMMENTS FRÅN videos_raw.csv
# ==============================================================================

def _merge_video_counts(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Beräknar månadsvis video_count (kumulativt) och comments från videos_raw.csv
    och mergar in i history_df.
    """
    videos_path = "data/raw/videos_raw.csv"
    if not os.path.exists(videos_path):
        print("  videos_raw.csv saknas – video_count behålls som 0")
        return history_df

    videos = pd.read_csv(videos_path)
    videos["published_at"] = pd.to_datetime(videos["published_at"], errors="coerce")
    videos["month_key"] = videos["published_at"].dt.to_period("M").dt.strftime("%b %Y")

    # Uppladdningar per månad per kanal
    uploads_per_month = (
        videos.groupby(["channel_id", "month_key"])
        .size()
        .reset_index(name="uploads_this_month")
    )

    # Kumulativt video_count per kanal
    uploads_per_month = uploads_per_month.sort_values(["channel_id", "month_key"])
    uploads_per_month["video_count"] = uploads_per_month.groupby("channel_id")[
        "uploads_this_month"
    ].cumsum()

    # Engagement per månad
    engagement_per_month = (
        videos.groupby(["channel_id", "month_key"])
        .agg(likes=("like_count", "sum"), comments=("comment_count", "sum"))
        .reset_index()
    )

    vc = uploads_per_month[["channel_id", "month_key", "video_count"]].rename(
        columns={"month_key": "month"}
    )
    eng = engagement_per_month.rename(columns={"month_key": "month"})

    history_df = history_df.merge(vc,  on=["channel_id", "month"], how="left")
    history_df = history_df.merge(eng, on=["channel_id", "month"], how="left")

    history_df["video_count"] = history_df["video_count_y"].fillna(
        history_df["video_count_x"]
    ).fillna(0).astype(int)
    history_df["likes"]    = history_df["likes_y"].fillna(history_df["likes_x"]).fillna(0).astype(int)
    history_df["comments"] = history_df["comments_y"].fillna(history_df["comments_x"]).fillna(0).astype(int)

    history_df = history_df.drop(
        columns=[c for c in history_df.columns if c.endswith("_x") or c.endswith("_y")],
        errors="ignore"
    )

    print("  video_count, likes och comments mergade från videos_raw.csv")
    return history_df
