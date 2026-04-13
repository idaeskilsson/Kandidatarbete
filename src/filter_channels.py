"""
Förfiltrering av kanaler innan ML-modellerna körs.

Filter 1 – Förvaltningsmöjlighet:
    Utesluter kanaler som ägs av stora etablerade medieaktörer (ej förvärvbara).
    Baseras på matchning mot en blocklista i kanalnamn och beskrivning,
    samt en prenumeranttröskel.

Filter 2 – Tydlig IP:
    Behåller bara kanaler med återkommande karaktärer/värld.
    Mäts genom att räkna hur ofta samma innehållsord återkommer i videotitlarna.
    Hög repetition = konsekvent IP. Generiska sångkanaler faller bort.
"""

import re
import pandas as pd
import numpy as np
from collections import Counter


# ==============================================================================
# KONFIGURATION
# ==============================================================================

# Stora medieaktörer som vi inte är intresserade av att förvärva från
LARGE_OWNER_KEYWORDS = [
    # Globala studios och nätverk
    "nickelodeon", "nick jr", "disney", "cartoon network", "bbc",
    "warner", "warner bros", "hasbro", "mattel", "lego", "dreamworks",
    "pixar", "netflix", "amazon", "hbo", "nbcuniversal", "viacom",
    "sony", "paramount", "turner", "discovery", "nat geo", "pbs kids",
    "sesame street", "fisher-price", "vtech", "spin master",
    "eone", "entertainment one",
    # Nordiska/europeiska storbolag
    "svt", "tv4", "nordic entertainment", "bonnier", "egmont",
    "nrk", "dr ", "yle", "ard ", "zdf ",
    # Stora barnkanalsvarumärken vars ägare är kända storbolag
    "cocomelon", "pinkfong", "baby shark", "super simple songs",
    "little baby bum", "bounce patrol", "blippi",
]

# Prenumerantgräns – kanaler över denna gräns är troligen för stora/dyra
MAX_SUBSCRIBERS = 10_000_000

# Stopord som inte räknas som karaktärsnamn/IP-ord i titlar
STOPWORDS = {
    # Engelska generiska barnvideoord
    "the", "a", "an", "and", "or", "for", "of", "in", "on", "to", "with",
    "is", "are", "be", "it", "its", "this", "that", "by", "at", "as",
    "kids", "children", "baby", "babies", "child", "toddler", "toddlers",
    "song", "songs", "nursery", "rhyme", "rhymes", "lullaby", "lullabies",
    "learn", "learning", "educational", "education", "fun", "funny",
    "video", "videos", "episode", "episodes", "official", "channel",
    "new", "best", "top", "little", "big", "super", "mini", "magic",
    "cartoon", "cartoons", "animation", "animated", "story", "stories",
    "compilation", "part", "collection", "full", "season", "ep",
    "english", "svenska", "norsk", "dansk", "suomi", "german", "french",
    "no", "yes", "my", "your", "our", "their", "more", "all", "very",
    # Svenska generiska ord
    "för", "och", "med", "på", "av", "en", "ett", "de", "det", "den",
    "barn", "sång", "sånger", "visa", "visor", "saga", "sagor",
    "roliga", "kul", "lär", "lära", "svenska",
}

# Minsta andel titlar ett ord måste dyka upp i för att räknas som IP-ord
IP_THRESHOLD = 0.25


# ==============================================================================
# FILTER 1: FÖRVALTNINGSMÖJLIGHET
# ==============================================================================

def _is_large_owner(channel_title: str, description: str,
                    subscriber_count: float) -> bool:
    """Returnerar True om kanalen sannolikt ägs av stor etablerad aktör."""
    text = f"{channel_title} {description}".lower()

    for keyword in LARGE_OWNER_KEYWORDS:
        # Matcha hela ord/fraser (undvik delfragment som "ar" i "warner")
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, text):
            return True

    if pd.notna(subscriber_count) and subscriber_count > MAX_SUBSCRIBERS:
        return True

    return False


def filter_acquirable(df: pd.DataFrame,
                      channels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tar bort kanaler som sannolikt ägs av stora etablerade medieaktörer.

    df           : huvud-dataframe med channel_id
    channels_df  : channels_raw.csv med channel_title, description, subscriber_count
    """
    meta = channels_df[["channel_id", "channel_title",
                         "description", "subscriber_count"]].copy()
    meta["description"] = meta["description"].fillna("")

    meta["large_owner"] = meta.apply(
        lambda r: _is_large_owner(
            r["channel_title"], r["description"], r["subscriber_count"]
        ),
        axis=1,
    )

    large = meta[meta["large_owner"]]["channel_id"]
    filtered = df[~df["channel_id"].isin(large)].copy()

    print(f"  Förvaltningsmöjlighet: {len(large)} kanaler borttagna "
          f"({len(filtered)} kvar)")
    if not large.empty:
        names = meta[meta["large_owner"]]["channel_title"].tolist()
        safe = [n.encode("ascii", "replace").decode() for n in names]
        print(f"    Borttagna: {safe}")

    return filtered


# ==============================================================================
# FILTER 2: TYDLIG IP
# ==============================================================================

def _ip_score(titles: list[str]) -> float:
    """
    Beräknar hur tydlig IP:n är för en kanal baserat på titlarna.
    Returnerar andelen titlar som innehåller det vanligaste IP-ordet (0–1).
    """
    if not titles:
        return 0.0

    word_counts: Counter = Counter()
    for title in titles:
        words = re.findall(r"[a-zåäöA-ZÅÄÖ]{3,}", title.lower())
        unique_in_title = set(words) - STOPWORDS
        word_counts.update(unique_in_title)

    if not word_counts:
        return 0.0

    top_word, top_count = word_counts.most_common(1)[0]
    return top_count / len(titles)


def filter_clear_ip(df: pd.DataFrame,
                    videos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Behåller bara kanaler med tydlig IP – återkommande karaktärer/värld.
    Mäts som: vanligaste innehållsordet förekommer i minst IP_THRESHOLD
    av kanalens videotitlar.

    df        : huvud-dataframe med channel_id
    videos_df : videos_raw.csv med channel_id och title
    """
    scores = (
        videos_df.groupby("channel_id")["title"]
        .apply(lambda titles: _ip_score(titles.tolist()))
        .reset_index(name="ip_score")
    )

    df = df.merge(scores, on="channel_id", how="left")
    df["ip_score"] = df["ip_score"].fillna(0.0)

    channels_with_videos = set(videos_df["channel_id"].unique())

    # Behåll kanaler som antingen:
    # (a) har tillräckligt hög IP-score, eller
    # (b) saknar videor i vår data (okänt – behåll för säkerhets skull)
    has_ip = df["ip_score"] >= IP_THRESHOLD
    no_videos = ~df["channel_id"].isin(channels_with_videos)
    keep = has_ip | no_videos
    filtered = df[keep].copy()

    removed = df[~keep]["channel_id"]
    print(f"  Tydlig IP: {len(removed)} kanaler borttagna "
          f"({len(filtered)} kvar, tröskel={IP_THRESHOLD:.0%})")

    return filtered


# ==============================================================================
# KOMBINERAT FILTER
# ==============================================================================

def apply_filters(df: pd.DataFrame,
                  channels_df: pd.DataFrame,
                  videos_df: pd.DataFrame) -> pd.DataFrame:
    """Kör båda filtren i ordning och returnerar filtrerat dataset."""
    print("Filtrerar kanaler...")
    df = filter_acquirable(df, channels_df)
    df = filter_clear_ip(df, videos_df)
    print(f"  Totalt kvar efter filtrering: {len(df)} kanaler")
    return df
