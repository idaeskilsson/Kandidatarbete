import os
import re
import time
import pandas as pd
from src.youtube_client import get_youtube_client

# ==============================================================================
# 1. SÖKFRÅGOR – fokus på karaktärskanaler med återkommande IP
# ==============================================================================

SEARCH_QUERIES = [
    # Engelska – UK
    ("animated kids character series", "GB"),
    ("children cartoon character show", "GB"),
    ("kids animated series original character", "GB"),
    ("kids character adventure show", "GB"),
    # Svenska
    ("animerad karaktär barn serie", "SE"),
    ("barnprogram karaktär äventyr", "SE"),
    ("tecknad serie barn karaktär", "SE"),
    # Tyska
    ("Zeichentrick Kinderfigur Serie", "DE"),
    ("animierte Kinderserie Figur", "DE"),
    ("Kinderfigur Abenteuer animiert", "DE"),
    # Franska
    ("personnage animé enfants série", "FR"),
    ("dessin animé personnage original enfants", "FR"),
    ("série animée enfants personnage", "FR"),
    # Spanska
    ("serie animada personaje infantil", "ES"),
    ("dibujos animados personaje aventura niños", "ES"),
    ("personaje animado serie niños", "ES"),
    # Norska
    ("animert karakter barn serie", "NO"),
    ("barneprogram figur eventyr", "NO"),
    # Danska
    ("animeret karakter børn serie", "DK"),
    ("børneprogram figur eventyr", "DK"),
    # Polska
    ("postać animowana serial dzieci", "PL"),
    ("bajka animowana postać przygoda", "PL"),
    # Italienska
    ("personaggio animato serie bambini", "IT"),
    ("cartoni animati personaggio avventura", "IT"),
    # Nederländska
    ("animatie personage kinderen serie", "NL"),
    ("tekenfilm karakter kinderen avontuur", "NL"),
    # Finska
    ("animoitu hahmo lapsille sarja", "FI"),
    ("lastenohjelma hahmo seikkailu", "FI"),
    # Portugisiska
    ("personagem animado série crianças", "PT"),
    ("desenho animado personagem aventura", "PT"),
    # Belgien
    ("animatie karakter kinderen serie", "BE"),
    ("personnage animé série enfants", "BE"),
    # Rumänska
    ("personaj animat serial copii", "RO"),
    ("desen animat personaj aventura", "RO"),
    # Ungerska
    ("animált karakter sorozat gyerek", "HU"),
    ("rajzfilm karakter kaland gyerekeknek", "HU"),
]

# YouTube-kategorier att söka videor i för att hitta barnkanaler
# 24 = Entertainment, 27 = Education
CATEGORY_QUERIES = [
    ("animated kids character series",      "GB", 24),
    ("children cartoon character show",     "GB", 27),
    ("Zeichentrick Kinderfigur Serie",       "DE", 24),
    ("serie animada personaje infantil",    "ES", 24),
    ("personnage animé série enfants",      "FR", 24),
    ("postać animowana serial dzieci",      "PL", 24),
    ("personaggio animato serie bambini",   "IT", 24),
    ("animoitu hahmo lapsille sarja",       "FI", 24),
    ("personagem animado série crianças",   "PT", 24),
]

MAX_RESULTS_PER_QUERY = 50

# Seed-expansion: antal seeds att expandera från (kvotbudget: MAX_SEEDS × 100 enheter)
MAX_SEEDS        = 20
SEED_MAX_RESULTS = 10   # kanaler per seed-sökning

# Tidigt filter innan videohämtning
MIN_SUBSCRIBERS  = 1_000
MIN_VIDEO_COUNT  = 5


# ==============================================================================
# HJÄLPFUNKTIONER
# ==============================================================================

def search_channels(query, region_code="GB", max_results=MAX_RESULTS_PER_QUERY):
    """Söker efter kanaler direkt via YouTube search."""
    youtube = get_youtube_client()
    response = youtube.search().list(
        part="snippet",
        q=query,
        type="channel",
        regionCode=region_code,
        maxResults=max_results
    ).execute()

    rows = []
    for item in response.get("items", []):
        rows.append({
            "channel_id":    item["snippet"]["channelId"],
            "channel_title": item["snippet"]["title"],
            "search_query":  query,
            "region_code":   region_code,
        })
    return rows


def search_channels_by_category(query, region_code, category_id,
                                  max_results=MAX_RESULTS_PER_QUERY):
    """
    Söker efter videor i en specifik kategori och returnerar kanalerna bakom dem.
    Kanalsökning stöder inte kategorier direkt – vi söker videor och extraherar channel_id.
    """
    youtube = get_youtube_client()
    response = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        videoCategoryId=str(category_id),
        regionCode=region_code,
        maxResults=max_results
    ).execute()

    rows = []
    for item in response.get("items", []):
        rows.append({
            "channel_id":    item["snippet"]["channelId"],
            "channel_title": item["snippet"]["channelTitle"],
            "search_query":  f"category:{category_id} {query}",
            "region_code":   region_code,
        })
    return rows


def expand_from_seeds(seed_channels_df, max_results=SEED_MAX_RESULTS):
    """
    Tar redan hittade kanaler som seeds och hittar liknande kanaler
    genom att söka på kanalnamnet. Begränsat till MAX_SEEDS seeds.
    """
    youtube = get_youtube_client()
    discovered = []

    for _, row in seed_channels_df.iterrows():
        channel_id   = row["channel_id"]
        channel_name = str(row.get("channel_title", ""))
        if not channel_name:
            continue

        try:
            response = youtube.search().list(
                part="snippet",
                q=channel_name,
                type="channel",
                maxResults=max_results
            ).execute()

            for item in response.get("items", []):
                discovered.append({
                    "channel_id":    item["snippet"]["channelId"],
                    "channel_title": item["snippet"]["title"],
                    "search_query":  f"seed:{channel_id}",
                    "region_code":   "seed",
                })
        except Exception as e:
            print(f"  Seed-expansion misslyckades för {channel_name}: {e}")

        time.sleep(0.3)

    return discovered


def get_featured_channels(channel_ids):
    """
    Hämtar featured channels från brandingSettings för varje kanal.
    Kostar 1 kvottenhet per 50 kanaler – negligibelt.
    Returnerar lista med channel_id för rekommenderade kanaler.
    """
    youtube = get_youtube_client()
    featured_ids = []

    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i:i + 50]
        try:
            response = youtube.channels().list(
                part="brandingSettings",
                id=",".join(batch)
            ).execute()

            for item in response.get("items", []):
                urls = (
                    item.get("brandingSettings", {})
                        .get("channel", {})
                        .get("featuredChannelsUrls", [])
                )
                for url in urls:
                    # Extrahera channel_id från /channel/UCxxxxxx-format
                    match = re.search(r"/channel/(UC[\w-]+)", url)
                    if match:
                        featured_ids.append(match.group(1))
        except Exception as e:
            print(f"  Featured channels misslyckades för batch: {e}")

    return list(set(featured_ids))


def get_channel_details(channel_ids):
    youtube = get_youtube_client()
    rows = []

    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i:i + 50]
        response = youtube.channels().list(
            part="snippet,statistics,contentDetails,status",
            id=",".join(batch)
        ).execute()

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            stats   = item.get("statistics", {})
            content = item.get("contentDetails", {})
            status  = item.get("status", {})

            rows.append({
                "channel_id":          item.get("id"),
                "channel_title":       snippet.get("title"),
                "description":         snippet.get("description"),
                "published_at":        snippet.get("publishedAt"),
                "country":             snippet.get("country"),
                "subscriber_count":    int(stats.get("subscriberCount", 0)) if stats.get("subscriberCount") else 0,
                "view_count":          int(stats.get("viewCount", 0)) if stats.get("viewCount") else 0,
                "video_count":         int(stats.get("videoCount", 0)) if stats.get("videoCount") else 0,
                "uploads_playlist_id": content.get("relatedPlaylists", {}).get("uploads"),
                "made_for_kids":       status.get("madeForKids"),
                "self_declared_made_for_kids": status.get("selfDeclaredMadeForKids"),
            })
    return rows


def get_video_ids_from_playlist(playlist_id, max_videos=60):
    youtube = get_youtube_client()
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_videos:
        response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=min(50, max_videos - len(video_ids)),
            pageToken=next_page_token
        ).execute()

        for item in response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids


def get_video_details(video_ids, channel_id):
    youtube = get_youtube_client()
    rows = []

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        response = youtube.videos().list(
            part="snippet,statistics,contentDetails,status",
            id=",".join(batch)
        ).execute()

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            stats   = item.get("statistics", {})
            content = item.get("contentDetails", {})
            status  = item.get("status", {})

            rows.append({
                "video_id":      item.get("id"),
                "channel_id":    channel_id,
                "title":         snippet.get("title"),
                "published_at":  snippet.get("publishedAt"),
                "category_id":   snippet.get("categoryId"),
                "duration":      content.get("duration"),
                "view_count":    int(stats.get("viewCount", 0)) if stats.get("viewCount") else 0,
                "like_count":    int(stats.get("likeCount", 0)) if stats.get("likeCount") else 0,
                "comment_count": int(stats.get("commentCount", 0)) if stats.get("commentCount") else 0,
                "made_for_kids": status.get("madeForKids"),
                "self_declared_made_for_kids": status.get("selfDeclaredMadeForKids"),
            })
    return rows


# ==============================================================================
# PIPELINE
# ==============================================================================

def run_collection():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/interim", exist_ok=True)

    # --- Steg 1: Direkt kanalsökning ---
    print("Steg 1: Söker kanaler via sökfrågor...")
    discovered = []
    for query, region in SEARCH_QUERIES:
        try:
            results = search_channels(query, region_code=region)
            discovered.extend(results)
            print(f"  '{query}' ({region}): {len(results)} kanaler")
        except Exception as e:
            if "quotaExceeded" in str(e):
                print(f"  Kvot slut vid '{query}' – sparar {len(discovered)} råkandidater och fortsätter.")
                break
            else:
                print(f"  Fel vid '{query}': {e}")
        time.sleep(0.2)

    # --- Steg 2: Kategoribaserad sökning ---
    print("\nSteg 2: Kategoribaserad sökning (video->kanal)...")
    for query, region, cat_id in CATEGORY_QUERIES:
        try:
            results = search_channels_by_category(query, region, cat_id)
            discovered.extend(results)
            print(f"  '{query}' kategori {cat_id} ({region}): {len(results)} kanaler")
        except Exception as e:
            if "quotaExceeded" in str(e):
                print(f"  Kvot slut vid kategoribaserad sökning – fortsätter med det vi har.")
                break
            else:
                print(f"  Fel vid '{query}': {e}")
        time.sleep(0.2)

    if not discovered:
        print("\n  Ingen data hämtad – API-kvoten är troligen slut för idag.")
        print("  Kvoten återställs vid 07:00 svensk tid (midnatt Pacific Time).")
        print("  Tips: skapa ett nytt Google-konto för ett separat projekt med egen kvot.")
        return

    discovered_df = pd.DataFrame(discovered).drop_duplicates(subset=["channel_id"])
    discovered_df.to_csv("data/interim/step1_discovered.csv",
                         index=False, encoding="utf-8-sig")
    print(f"\n  Unika kanaler efter steg 1–2: {len(discovered_df)} (sparat)")

    # --- Steg 3: Hämta kanaldetaljer för seeds ---
    print("\nSteg 3: Hämtar kanaldetaljer för seeds...")
    seed_ids        = discovered_df["channel_id"].tolist()
    seed_details    = get_channel_details(seed_ids)
    seed_details_df = pd.DataFrame(seed_details)
    seed_details_df.to_csv("data/interim/step3_seed_details.csv",
                           index=False, encoding="utf-8-sig")
    print(f"  {len(seed_details_df)} kanaldetaljer hämtade (sparat)")

    # --- Steg 4: Featured channels ---
    print("\nSteg 4: Hämtar featured channels...")
    featured_ids = get_featured_channels(seed_ids)
    new_featured  = [fid for fid in featured_ids
                     if fid not in set(seed_details_df["channel_id"].tolist())]
    print(f"  {len(featured_ids)} featured channel-IDs hittade, "
          f"{len(new_featured)} nya unika")
    featured_rows = [{"channel_id": fid, "channel_title": "",
                      "search_query": "featured", "region_code": "featured"}
                     for fid in new_featured]
    featured_df = pd.DataFrame(featured_rows)

    # --- Steg 5: Seed-expansion ---
    print(f"\nSteg 5: Seed-expansion (topp {MAX_SEEDS} seeds × {SEED_MAX_RESULTS} resultat)...")
    seeds_for_expansion = (
        seed_details_df[seed_details_df["subscriber_count"] >= 1_000]
        .nlargest(MAX_SEEDS, "subscriber_count")
        [["channel_id", "channel_title"]]
    )
    print(f"  Expanderar från {len(seeds_for_expansion)} seeds...")
    expanded    = expand_from_seeds(seeds_for_expansion, max_results=SEED_MAX_RESULTS)
    expanded_df = pd.DataFrame(expanded).drop_duplicates(subset=["channel_id"])

    # --- Slå ihop alla källor ---
    all_discovered = pd.concat(
        [discovered_df, featured_df, expanded_df], ignore_index=True
    ).drop_duplicates(subset=["channel_id"])
    all_discovered.to_csv("data/interim/step5_all_discovered.csv",
                          index=False, encoding="utf-8-sig")
    print(f"  Totalt unika kanaler (alla källor): {len(all_discovered)} (sparat)")

    # --- Steg 6: Hämta detaljer för nya kanaler (featured + expanded) ---
    print("\nSteg 6: Hämtar kanaldetaljer för nya kanaler...")
    already_fetched = set(seed_details_df["channel_id"].tolist())
    new_ids = [cid for cid in all_discovered["channel_id"].tolist()
               if cid not in already_fetched]
    new_details = []
    if new_ids:
        try:
            new_details = get_channel_details(new_ids)
        except Exception as e:
            print(f"  Varning: kunde inte hämta detaljer för {len(new_ids)} nya kanaler "
                  f"(troligen kvot slut): {e}")
            print("  Fortsätter med de kanaler som redan hämtats.")
    channels_df = (
        pd.DataFrame(seed_details + new_details)
        .drop_duplicates(subset=["channel_id"])
    )
    channels_df.to_csv("data/interim/step6_channels_all.csv",
                       index=False, encoding="utf-8-sig")
    print(f"  {len(channels_df)} kanaler med detaljer (sparat)")

    # --- Tidigt filter: ta bort uppenbart irrelevanta kanaler ---
    before = len(channels_df)
    channels_df = channels_df[
        (channels_df["subscriber_count"] >= MIN_SUBSCRIBERS) &
        (channels_df["video_count"]      >= MIN_VIDEO_COUNT)
    ]
    print(f"  Tidig filtrering: {before - len(channels_df)} kanaler borttagna "
          f"(< {MIN_SUBSCRIBERS} subs eller < {MIN_VIDEO_COUNT} videor). "
          f"{len(channels_df)} kvar.")

    # --- Steg 7: Hämta videor ---
    print(f"\nSteg 7: Hämtar videor för {len(channels_df)} kanaler...")
    all_video_rows = []
    quota_exceeded = False
    for i, (_, row) in enumerate(channels_df.iterrows()):
        if quota_exceeded:
            break
        playlist_id = row.get("uploads_playlist_id")
        channel_id  = row.get("channel_id")
        if not playlist_id:
            continue
        try:
            video_ids  = get_video_ids_from_playlist(playlist_id, max_videos=60)
            video_rows = get_video_details(video_ids, channel_id)
            all_video_rows.extend(video_rows)
        except Exception as e:
            if "quotaExceeded" in str(e):
                print(f"  Kvot slut under videohämtning efter {i} kanaler – "
                      f"sparar {len(all_video_rows)} videor hittills.")
                quota_exceeded = True
            else:
                print(f"  Fel för kanal {channel_id}: {e}")
        # Spara checkpoints var 100:e kanal
        if (i + 1) % 100 == 0:
            pd.DataFrame(all_video_rows).to_csv(
                "data/interim/step7_videos_checkpoint.csv",
                index=False, encoding="utf-8-sig"
            )

    videos_df = pd.DataFrame(all_video_rows)

    # --- Spara ---
    all_discovered.drop_duplicates(subset=["channel_id"]).to_csv(
        "data/raw/search_results_raw.csv", index=False, encoding="utf-8-sig"
    )
    channels_df.drop_duplicates(subset=["channel_id"]).to_csv(
        "data/raw/channels_raw.csv", index=False, encoding="utf-8-sig"
    )
    videos_df.drop_duplicates(subset=["video_id"]).to_csv(
        "data/raw/videos_raw.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\nDatainsamling klar.")
    print(f"  Kanaler: {channels_df['channel_id'].nunique()}")
    print(f"  Videor:  {videos_df['video_id'].nunique()}")
