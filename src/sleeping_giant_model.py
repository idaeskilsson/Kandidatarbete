"""
Sleeping Giant – identifierar undervärderade YouTube-barnkanaler.

Metod ActivityBenchmark: XGBoost elitbenchmark (residualanalys).
  Tränas på välpresterande kanaler och predictar log(prenumeranter) för alla.
  Negativ residual = kanalen har färre prenumeranter än sin aktivitetsprofil motiverar.

Metod EngagementCluster: Iterativ K-Means per storlekssegment.
  Grupperar kanaler efter engagemangsprofil och operativ mognad.
  Klustret med starkast engagemangsprofil identifieras som sleeping giants.

Läser från: data/processed/model_dataset.csv
            data/raw/videos_raw.csv
Sparar till: outputs/predictions/sleeping_giant_activity_benchmark.csv
             outputs/predictions/sleeping_giant_engagement_cluster.csv
"""

import os
import re
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


MIN_SUBSCRIBERS      = 5_000
HISTORY_FILE         = "data/raw/channel_history.csv"
TRENDS_CACHE         = "data/raw/search_interest_cache.csv"
YOUTUBE_SEARCH_CACHE = "data/raw/youtube_search_cache.csv"

# Features för ActivityBenchmark (elitbenchmark)
ACTIVITY_BENCHMARK_FEATURES = [
    "log_estimated_monthly_views",  # uppskattade månadsvisningar (sum views 12 mån / 12)
    "log_avg_likes_recent",         # snitt likes/video, senaste 12 mån
    "log_avg_comments_recent",      # snitt kommentarer/video, senaste 12 mån
    "avg_monthly_search",           # Google Trends-sökintresse
    "avg_duration_sec",             # videolängd
    "cpm",                          # marknadsvärde baserat på land
]

# Features för EngagementCluster (K-Means klustring)
ENGAGEMENT_CLUSTER_FEATURES = [
    "like_rate_recent",             # likes per visning, senaste 12 mån
    "comment_rate_recent",          # kommentarer per visning
    "comment_to_like_ratio_recent", # kommentardjup relativt likes
    "search_results_per_sub",       # YouTube-sökresultat för karaktärsnamn / prenumeranter
    "views_cv",                     # variationskoefficient för visningar
    "upload_gap_std",               # oregelbundenhet i uppladdningstakt (std dagar)
    "log_video_count_recent",       # log(antal videos senaste 12 mån)
    "upload_frequency",             # videos per dag sedan kanalstart
]

SUB_TIERS = {
    "mikro": (5_000,    25_000),
    "liten": (25_000,  100_000),
    "medel": (100_000, 500_000),
    "stor":  (500_000, float("inf")),
}


# ==============================================================================
# FEATURE-BERÄKNINGAR
# ==============================================================================

def _parse_duration(iso: str) -> float:
    """Omvandlar ISO 8601-varaktighet (t.ex. PT4M30S) till sekunder."""
    if not isinstance(iso, str):
        return 0.0
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return 0.0
    h, mn, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mn * 60 + s


def _compute_video_features(channel_ids: pd.Series) -> pd.DataFrame:
    """
    Beräknar per kanal från videos_raw.csv:
      avg_duration_sec – genomsnittlig videolängd i sekunder
      views_cv         – variationskoefficient för visningar (hög = viral-beroende)
      upload_gap_std   – std i dagar mellan uppladdningar (hög = oregelbunden)
    """
    fallback = pd.DataFrame({
        "channel_id":       channel_ids.unique(),
        "avg_duration_sec": np.nan,
        "views_cv":         np.nan,
        "upload_gap_std":   np.nan,
    })
    path = "data/raw/videos_raw.csv"
    if not os.path.exists(path):
        return fallback

    videos = pd.read_csv(path)
    videos["duration_sec"] = videos["duration"].apply(_parse_duration)
    videos["published_at"] = pd.to_datetime(videos["published_at"], utc=True)

    records = []
    for ch_id, grp in videos.groupby("channel_id"):
        grp     = grp.sort_values("published_at")
        avg_v   = grp["view_count"].mean()
        records.append({
            "channel_id":       ch_id,
            "avg_duration_sec": grp["duration_sec"].mean(),
            "views_cv":         grp["view_count"].std() / avg_v if avg_v > 0 else np.nan,
            "upload_gap_std":   grp["published_at"].diff().dt.days.dropna().std()
                                if len(grp) >= 2 else np.nan,
        })
    return pd.DataFrame(records)


def _compute_recent_video_averages(channel_ids: pd.Series) -> pd.DataFrame:
    """
    Beräknar per kanal från videos_raw.csv (senaste 12 månader):
      like_rate_recent             – likes / visningar
      comment_rate_recent          – kommentarer / visningar
      comment_to_like_ratio_recent – kommentarer / likes
      log_avg_likes_recent         – log(snitt likes/video)
      log_avg_comments_recent      – log(snitt kommentarer/video)
      log_video_count_recent       – log(antal videos)
    """
    fallback_cols = [
        "like_rate_recent", "comment_rate_recent", "comment_to_like_ratio_recent",
        "log_avg_likes_recent", "log_avg_comments_recent", "log_video_count_recent",
        "log_estimated_monthly_views",
    ]
    fallback = pd.DataFrame({"channel_id": channel_ids.unique()})
    for col in fallback_cols:
        fallback[col] = np.nan

    path = "data/raw/videos_raw.csv"
    if not os.path.exists(path):
        return fallback

    videos = pd.read_csv(path)
    videos["published_at"] = pd.to_datetime(videos["published_at"], utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=12)
    videos = videos[videos["published_at"] >= cutoff]

    if videos.empty:
        return fallback

    records = []
    for ch_id, grp in videos.groupby("channel_id"):
        avg_views    = grp["view_count"].mean()
        avg_likes    = grp["like_count"].mean()
        avg_comments = grp["comment_count"].mean()
        total_views_12m = grp["view_count"].sum()
        records.append({
            "channel_id":                   ch_id,
            "like_rate_recent":             avg_likes    / avg_views if avg_views > 0 else np.nan,
            "comment_rate_recent":          avg_comments / avg_views if avg_views > 0 else np.nan,
            "comment_to_like_ratio_recent": avg_comments / avg_likes if avg_likes > 0 else np.nan,
            "log_avg_likes_recent":         np.log1p(avg_likes),
            "log_avg_comments_recent":      np.log1p(avg_comments),
            "log_video_count_recent":       np.log1p(len(grp)),
            "log_estimated_monthly_views":  np.log1p(total_views_12m / 12),
        })

    result = pd.DataFrame(records)
    all_ids = pd.DataFrame({"channel_id": channel_ids.unique()})
    return all_ids.merge(result, on="channel_id", how="left").fillna(0.0)


def _compute_monthly_features(channel_ids: pd.Series) -> pd.DataFrame:
    """
    Beräknar per kanal från channel_history.csv (senaste 12 månader):
      log_avg_monthly_views    – log(genomsnittliga månatliga visningar)
      log_avg_monthly_likes    – log(genomsnittliga månatliga likes + 1)
      log_avg_monthly_comments – log(genomsnittliga månatliga kommentarer + 1)
    """
    fallback = pd.DataFrame({
        "channel_id":               channel_ids.unique(),
        "log_avg_monthly_views":    np.nan,
        "log_avg_monthly_likes":    np.nan,
        "log_avg_monthly_comments": np.nan,
    })
    if not os.path.exists(HISTORY_FILE):
        return fallback

    hist = pd.read_csv(HISTORY_FILE)
    hist["month_dt"] = pd.to_datetime(hist["month"], format="%b %Y")
    hist = hist.sort_values(["channel_id", "month_dt"])

    records = []
    for ch_id, grp in hist.groupby("channel_id"):
        grp = grp.tail(12)
        records.append({
            "channel_id":               ch_id,
            "log_avg_monthly_views":    np.log1p(grp["total_views"].clip(lower=0).mean()),
            "log_avg_monthly_likes":    np.log1p(grp["likes"].clip(lower=0).mean()),
            "log_avg_monthly_comments": np.log1p(grp["comments"].clip(lower=0).mean()),
        })
    return pd.DataFrame(records)


def _fetch_google_trends(channels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hämtar Google Trends-sökintresse per kanal (kanalnamn som sökterm).
    Returnerar DataFrame med channel_id och avg_monthly_search (0–100).
    Resultat cachas i TRENDS_CACHE.
    """
    import time
    from pytrends.request import TrendReq

    if os.path.exists(TRENDS_CACHE):
        cache = pd.read_csv(TRENDS_CACHE)
    else:
        cache = pd.DataFrame(columns=["channel_id", "avg_monthly_search"])

    cached_ids = set(cache["channel_id"].tolist())
    needed     = channels_df[~channels_df["channel_id"].isin(cached_ids)]

    if len(needed) == 0:
        print("  Google Trends: all data cachad.")
        return cache

    print(f"  Google Trends: hämtar {len(needed)} kanaler...")
    pytrends   = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=2, backoff_factor=0.5)
    new_records = []
    for i, (_, row) in enumerate(needed.iterrows()):
        title = str(row.get("channel_title", ""))[:100]
        try:
            pytrends.build_payload([title], timeframe="today 12-m", geo="")
            iot = pytrends.interest_over_time()
            avg_search = float(iot[title].mean()) if not iot.empty and title in iot.columns else 0.0
        except Exception:
            avg_search = np.nan
        new_records.append({"channel_id": row["channel_id"], "avg_monthly_search": avg_search})
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(needed)} klara...")
        time.sleep(1.2)

    new_df = pd.DataFrame(new_records)
    cache  = pd.concat([cache, new_df], ignore_index=True).drop_duplicates("channel_id")
    os.makedirs("data/raw", exist_ok=True)
    cache.to_csv(TRENDS_CACHE, index=False, encoding="utf-8-sig")
    return cache


# ==============================================================================
# KARAKTÄRSNAMN OCH YOUTUBE-SÖKNING (för search_results_per_sub i EngagementCluster)
# ==============================================================================

# Ord som inte räknas som karaktärsnamn trots stor bokstav
_STOPWORDS = {
    "Der", "Die", "Das", "Den", "Dem", "Des", "Ein", "Eine", "Einen", "Einem", "Einer",
    "Und", "Mit", "Für", "Von", "Bei", "Auf", "Aus", "Zum", "Zur", "Im", "Am",
    "Les", "Des", "Une", "Sur", "Avec", "Pour", "Dans", "Est", "Sont", "Ses",
    "En", "Ett", "Det", "Och", "Med", "För", "Av", "Till", "Från", "Är",
    "Et", "For", "Til", "Av", "Og", "Er", "Vi",
    "Los", "Las", "Del", "Con", "Por", "Para", "Una",
    "Il", "La", "Gli", "Dei", "Con", "Per", "Che",
    "The", "And", "With", "For", "From", "Into",
    "New", "Official", "Kids", "Song", "Songs", "Episode", "Season",
    "Part", "Full", "Baby", "Little", "Big",
}


def _extract_character_names(channel_ids: pd.Series) -> pd.DataFrame:
    """
    Extraherar det mest troliga karaktärsnamnet per kanal från videotitlar.
    Söker efter 2–3-ordsfraser med versaler utan stoppord, minst 3 unika titlar.
    Kanaler utan tydlig karaktär får character_name=NaN.
    """
    from collections import Counter

    path = "data/raw/videos_raw.csv"
    if not os.path.exists(path):
        return pd.DataFrame({"channel_id": channel_ids.unique(), "character_name": np.nan})

    videos = pd.read_csv(path, usecols=["channel_id", "title"])
    videos = videos[videos["channel_id"].isin(channel_ids)]

    cap_phrase = re.compile(
        r'\b([A-ZÁÉÍÓÚÄÖÜÀÈÌÒÙÂÊÎÔÛÑČŠŽ][a-záéíóúäöüàèìòùâêîôûñčšž]+'
        r'(?:\s[A-ZÁÉÍÓÚÄÖÜÀÈÌÒÙÂÊÎÔÛÑČŠŽ][a-záéíóúäöüàèìòùâêîôûñčšž]+){1,2})\b'
    )

    records = []
    for ch_id, grp in videos.groupby("channel_id"):
        phrase_title_count: Counter = Counter()
        for title in grp["title"].dropna():
            seen = set()
            for m in cap_phrase.finditer(title):
                phrase = m.group(1)
                if any(w in _STOPWORDS for w in phrase.split()):
                    continue
                if phrase not in seen:
                    phrase_title_count[phrase] += 1
                    seen.add(phrase)

        candidates = {p: c for p, c in phrase_title_count.items() if c >= 3}
        best = max(candidates, key=candidates.get) if candidates else None
        records.append({"channel_id": ch_id, "character_name": best})

    result = pd.DataFrame(records)

    # Uteslut fraser som är generiska (förekommer i fler än 1 kanal)
    phrase_counts = result["character_name"].dropna().value_counts()
    generic = set(phrase_counts[phrase_counts > 1].index)
    if generic:
        print(f"  Utesluter {len(generic)} generiska fraser: {sorted(generic)}")
    result.loc[result["character_name"].isin(generic), "character_name"] = np.nan

    all_ids = pd.DataFrame({"channel_id": channel_ids.unique()})
    return all_ids.merge(result, on="channel_id", how="left")


def _fetch_youtube_search_results(char_df: pd.DataFrame,
                                   channel_names: dict,
                                   subscriber_counts: dict) -> pd.DataFrame:
    """
    Hämtar antal YouTube-sökresultat per kanal via YouTube Data API.
    Söker på karaktärsnamn om tillgängligt, annars kanalnamn.
    Returnerar search_results_per_sub = totalResults / subscriber_count.
    Resultat cachas i YOUTUBE_SEARCH_CACHE.
    """
    import time
    import requests

    api_key = os.environ.get("YOUTUBE_API_KEY", "")
    if not api_key:
        try:
            with open(".env") as f:
                for line in f:
                    if line.startswith("YOUTUBE_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        break
        except FileNotFoundError:
            pass
    if not api_key:
        print("  VARNING: YOUTUBE_API_KEY saknas – hoppar över YouTube-sökning.")
        result = char_df[["channel_id"]].copy()
        result["search_results_per_sub"] = 0.0
        return result

    df = char_df.copy()
    df["search_term"] = df.apply(
        lambda r: r["character_name"] if pd.notna(r["character_name"])
                  else channel_names.get(r["channel_id"], ""),
        axis=1,
    )
    df = df[df["search_term"] != ""]

    if os.path.exists(YOUTUBE_SEARCH_CACHE):
        cache = pd.read_csv(YOUTUBE_SEARCH_CACHE)
    else:
        cache = pd.DataFrame(columns=["search_term", "total_results"])

    cached_terms = set(cache["search_term"].dropna().tolist())
    needed = [t for t in df["search_term"].unique() if t not in cached_terms]

    if needed:
        print(f"  Hämtar YouTube-sökresultat för {len(needed)} termer...")
        new_records = []
        for i, term in enumerate(needed):
            try:
                resp = requests.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params={"part": "snippet", "q": term, "type": "video",
                            "maxResults": 1, "key": api_key},
                    timeout=10,
                )
                total = int(resp.json().get("pageInfo", {}).get("totalResults", 0))
            except Exception as e:
                print(f"    Fel för '{term}': {e}")
                total = None
            if total is not None:
                new_records.append({"search_term": term, "total_results": total})
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(needed)} klara...")
            time.sleep(0.3)

        if new_records:
            new_df = pd.DataFrame(new_records)
            cache  = pd.concat([cache, new_df], ignore_index=True).drop_duplicates("search_term")
            os.makedirs("data/raw", exist_ok=True)
            cache.to_csv(YOUTUBE_SEARCH_CACHE, index=False, encoding="utf-8-sig")
    else:
        print(f"  YouTube-sökresultat: all data cachad ({len(df['search_term'].unique())} termer).")

    result = df.merge(cache, on="search_term", how="left")
    result["total_results"]       = result["total_results"].fillna(0)
    result["subscriber_count"]    = result["channel_id"].map(subscriber_counts)
    result["search_results_per_sub"] = (
        result["total_results"] / result["subscriber_count"].replace(0, np.nan)
    ).fillna(0)
    return result[["channel_id", "search_results_per_sub"]]


def _detect_franchise_channels(name_lookup: pd.DataFrame) -> set:
    """
    Identifierar kanaler som ingår i flerspråkiga franchise-serier.
    Kanaler vars normaliserade basnamn förekommer i >= 2 kanaler utesluts.
    """
    LANG_RE = re.compile(
        r'\b(en\s+)?(fran[cç]ais|espanol|espa[nñ]ol|deutsch|italiano|'
        r'nederland[s]?|polski|svenska|norsk[a]?|dansk[a]?|english|arabic|'
        r'french|spanish|german|italian|dutch|polish|swedish|norwegian|danish|'
        r'portuguese|portugu[eê]s|suomi|vlaams|belge|belgique|romanian[a]?)\b'
        r'|\b(officiel[le]?|official|officieel|ufficiale|offiziell|oficial[e]?)\b'
        r'|\b(topic|channel|kanal)\b'
        r'|\s*[-–|]\s*(wildBrain|WildBrain|Disney|Nickelodeon|Cartoon\s*Network)\b'
        r'|\s*[-–|]\s*(fr|es|de|it|nl|pl|sv|no|da|at|ch|be|ro)\s*$',
        re.IGNORECASE
    )

    df = name_lookup[["channel_id", "channel_title"]].drop_duplicates("channel_id").copy()
    df["base"] = df["channel_title"].str.lower()
    df["base"] = df["base"].apply(lambda t: LANG_RE.sub(" ", str(t)))
    df["base"] = df["base"].str.replace(r"[^\w\s]", " ", regex=True)
    df["base"] = df["base"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["base"].str.len() > 3]

    base_counts     = df["base"].value_counts()
    franchise_bases = set(base_counts[base_counts >= 2].index)
    franchise_ids   = set(df[df["base"].isin(franchise_bases)]["channel_id"])

    if franchise_ids:
        print(f"  Franchise-kanaler ({len(franchise_ids)} st utesluts):")
        for base in sorted(franchise_bases):
            titles = df[df["base"] == base]["channel_title"].tolist()
            if len(titles) >= 2:
                base_safe   = base.encode("ascii", "replace").decode()
                titles_safe = [t.encode("ascii", "replace").decode() for t in titles]
                print(f"    '{base_safe}': {titles_safe}")

    return franchise_ids


# ==============================================================================
# ACTIVITYBENCHMARK – HJÄLPFUNKTION
# ==============================================================================

def _shap_plots(model, X: pd.DataFrame, prefix: str):
    """Sparar SHAP bar- och beeswarm-plottar för modellförklaring."""
    os.makedirs("outputs", exist_ok=True)
    explainer   = shap.Explainer(model)
    shap_values = explainer(X)

    plt.figure(figsize=(8, 5))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(f"outputs/{prefix}_shap_bar.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(f"outputs/{prefix}_shap_beeswarm.png", dpi=150)
    plt.close()

    print(f"  SHAP-plottar sparade: outputs/{prefix}_shap_bar.png, "
          f"outputs/{prefix}_shap_beeswarm.png")


# ==============================================================================
# ENGAGEMENTCLUSTER: ITERATIV K-MEANS – HJÄLPFUNKTIONER
# ==============================================================================

def _run_kmeans(sub_df, feats, medians, p75, name_lkp, k_range, iter_label):
    """
    Kör K-Means med silhouette-baserat k-val inom k_range.
    Returnerar (scored_df, cents_df, sg_index) eller None om klustring misslyckas.

    Sleeping giant-klustret väljs som det med flest features över medianen,
    med like_rate_recent som tiebreaker.
    """
    X  = sub_df[feats].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    results = []
    for k in k_range:
        if k >= len(sub_df):
            break
        km_try  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl_try = km_try.fit_predict(Xs)
        if len(set(lbl_try)) < 2:
            continue
        results.append((k, silhouette_score(Xs, lbl_try), km_try, lbl_try))

    if not results:
        return None

    best_k, best_sil, km, lbl = max(results, key=lambda x: x[1])
    print(f"  {iter_label} k={best_k} silhouette={best_sil:.3f} ({len(sub_df)} kanaler)")

    sub_df           = sub_df.copy()
    sub_df["cluster"] = lbl

    cents_scaled = km.cluster_centers_
    cents        = pd.DataFrame(sc.inverse_transform(cents_scaled), columns=feats)
    cents["n"]   = pd.Series(lbl).value_counts().sort_index().values

    like_col = "like_rate_recent" if "like_rate_recent" in feats else feats[0]

    cents["above_median"] = cents[feats].apply(
        lambda row: sum(row[f] > medians[f] for f in feats), axis=1
    )
    cents["extreme_count"] = cents[feats].apply(
        lambda row: sum(
            row[f] > (3 * medians[f] if medians[f] > 0 else p75[f])
            for f in feats
            if (3 * medians[f] if medians[f] > 0 else p75[f]) > 0
        ),
        axis=1,
    )

    sg = int(cents.sort_values(["above_median", like_col], ascending=[False, False]).index[0])

    sg_cent = cents_scaled[sg]
    dists   = np.linalg.norm(Xs - sg_cent, axis=1)
    sub_df["sleeping_giant_score"] = MinMaxScaler().fit_transform(
        (1 - dists / dists.max()).reshape(-1, 1)
    )
    return sub_df, cents, sg


def _print_clusters(df_scored, cents, sg, feats, medians, name_lkp):
    """Skriver ut klusterbeskrivning med centroidvärden och exempelkanaler."""
    like_col    = "like_rate_recent" if "like_rate_recent" in feats else feats[0]
    id_to_title = name_lkp.set_index("channel_id")["channel_title"].to_dict()

    print(f"  Sleeping giant = kluster {sg} "
          f"({int(cents.loc[sg, 'above_median'])}/{len(feats)} över median, "
          f"n={int(cents.loc[sg, 'n'])})\n")

    for i in cents.index:
        c      = cents.iloc[i]
        tag_sg  = "  <-- SLEEPING GIANT" if i == sg else ""
        tag_bad = "  [DÅLIGT]"           if (c["above_median"] < 3 and c["extreme_count"] < 2) else ""
        tag_exc = "  [BEHÅLLT extremt]"  if (c["above_median"] < 3 and c["extreme_count"] >= 2) else ""
        print(f"  Kluster {i} (n={int(c['n'])}){tag_sg}{tag_bad}{tag_exc}")
        for f in feats:
            rel = "över" if c[f] > medians[f] else "under"
            print(f"    {f:<38} {c[f]:>10.5f}  ({rel} median {medians[f]:.5f})")
        top5     = df_scored[df_scored["cluster"] == i].nlargest(5, like_col)["channel_id"].map(id_to_title)
        top5_str = ", ".join(str(t).encode("ascii", "replace").decode() for t in top5 if pd.notna(t))
        print(f"    Exempel: {top5_str}\n")


def _kmeans_iterative(sub_df, feats, label, name_lkp,
                       k_range=range(3, 9), bad_threshold=3,
                       target_sg_size=10, max_iter=15):
    """
    Iterativ K-Means i två faser:

    Fas 1: Tar bort dåliga kluster (få features över medianen) och repeterar
           klustring tills SG-klustret har <= target_sg_size kanaler,
           eller tills inga fler dåliga kluster finns.

    Fas 2: Sub-klustrar SG-gruppen iterativt tills <= target_sg_size.
           Om en sub-iteration skulle understicka målet återgås till
           föregående iteration (overshoot-skydd).
    """
    X_full  = sub_df[feats].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    medians = X_full.median()
    p75     = X_full.quantile(0.75)

    current_df  = sub_df.copy()
    removed_ids = set()
    last_result = None

    # ── FAS 1 ─────────────────────────────────────────────────────────────────
    for iteration in range(1, max_iter + 1):
        if len(current_df) < 10:
            print(f"  För få kanaler ({len(current_df)}) – stoppar fas 1.")
            break

        print(f"\n  [{label}] Fas 1, iteration {iteration}:")
        res = _run_kmeans(current_df, feats, medians, p75, name_lkp,
                          k_range, f"  Iter {iteration}:")
        if res is None:
            break
        df_scored, cents, sg = res
        last_result = (df_scored, cents, sg)

        sg_size = int(cents.loc[sg, "n"])
        print(f"  SG-kluster: {sg_size} kanaler (mål <= {target_sg_size})")

        if sg_size <= target_sg_size:
            print(f"  Mål uppnått i fas 1 efter {iteration} iterationer!")
            _print_clusters(df_scored, cents, sg, feats, medians, name_lkp)
            df_removed = sub_df[sub_df["channel_id"].isin(removed_ids)].copy()
            df_removed["cluster"] = -1
            df_removed["sleeping_giant_score"] = 0.0
            return pd.concat([df_scored, df_removed], ignore_index=True)

        bad_clusters = set(
            cents.index[
                (cents["above_median"] < bad_threshold) &
                (cents["extreme_count"] < 2) &
                (cents.index != sg)
            ].tolist()
        )

        if not bad_clusters:
            print(f"  Inga fler dåliga kluster – går till fas 2 (SG={sg_size}).")
            break

        new_bad     = set(df_scored[df_scored["cluster"].isin(bad_clusters)]["channel_id"])
        removed_ids |= new_bad
        current_df  = current_df[~current_df["channel_id"].isin(new_bad)].copy()
        bad_names   = [
            f"kluster {c} (n={int(cents.loc[c,'n'])}, "
            f"{int(cents.loc[c,'above_median'])}/{len(feats)} över median)"
            for c in sorted(bad_clusters)
        ]
        print(f"  Tar bort: {', '.join(bad_names)} → "
              f"{len(new_bad)} kanaler borttagna, {len(current_df)} kvar")
    else:
        print(f"  Max iterationer ({max_iter}) nådda i fas 1.")

    if last_result is None:
        return sub_df.assign(cluster=0, sleeping_giant_score=0.0)

    # ── FAS 2 ─────────────────────────────────────────────────────────────────
    df_main, cents_main, sg_main = last_result
    sg_size = int(cents_main.loc[sg_main, "n"])

    sg_df     = df_main[df_main["cluster"] == sg_main].copy()
    non_sg_df = df_main[df_main["cluster"] != sg_main].copy()
    non_sg_df["sleeping_giant_score"] = 0.0

    last_sub  = None
    sub_phase = 0

    while sg_size > target_sg_size and len(sg_df) >= 6:
        sub_phase  += 1
        k_sub       = range(2, min(7, max(3, len(sg_df) // 5) + 1))
        print(f"\n  [{label}] Fas 2, sub-iteration {sub_phase}: "
              f"klustring av {len(sg_df)} SG-kanaler (k={list(k_sub)}):")

        prev_sg_df = sg_df.copy()  # spara för eventuell overshoot

        res2 = _run_kmeans(sg_df, feats, medians, p75, name_lkp,
                           k_sub, f"  Sub {sub_phase}:")
        if res2 is None:
            break

        sub_scored, sub_cents, sub_sg2 = res2
        new_sg_size = int(sub_cents.loc[sub_sg2, "n"])
        print(f"  Ny SG efter sub-klustring: {new_sg_size} kanaler")

        if new_sg_size < target_sg_size:
            print(f"  SG={new_sg_size} < mål ({target_sg_size}) – "
                  f"återgår till föregående SG ({len(prev_sg_df)} kanaler).")
            sg_df = prev_sg_df
            break

        last_sub = (sub_scored, sub_cents, sub_sg2)
        sg_size  = new_sg_size

        non_sub = sub_scored[sub_scored["cluster"] != sub_sg2].copy()
        non_sub["sleeping_giant_score"] = 0.0
        non_sg_df = pd.concat([non_sg_df, non_sub], ignore_index=True)
        sg_df     = sub_scored[sub_scored["cluster"] == sub_sg2].copy()

        if sg_size <= target_sg_size:
            print(f"  Mål uppnått i fas 2, sub-iteration {sub_phase}!")
            break

    if last_sub is not None:
        _print_clusters(*last_sub, feats, medians, name_lkp)
    else:
        _print_clusters(df_main, cents_main, sg_main, feats, medians, name_lkp)

    df_combined = pd.concat([sg_df, non_sg_df], ignore_index=True)
    df_removed  = sub_df[sub_df["channel_id"].isin(removed_ids)].copy()
    df_removed["cluster"] = -1
    df_removed["sleeping_giant_score"] = 0.0
    return pd.concat([df_combined, df_removed], ignore_index=True)


# ==============================================================================
# PIPELINE-INGÅNGSPUNKT
# ==============================================================================

def run_sleeping_giant():
    os.makedirs("outputs/predictions", exist_ok=True)

    # ── Grunddata ─────────────────────────────────────────────────────────────
    df = pd.read_csv("data/processed/model_dataset.csv")
    df["log_subscriber_count"] = np.log1p(df["subscriber_count"])
    df = df[df["subscriber_count"] >= MIN_SUBSCRIBERS].copy()
    print(f"Sleeping Giant – dataset: {len(df)} kanaler (>= {MIN_SUBSCRIBERS:,} prenumeranter)")

    channels_raw = pd.read_csv("data/raw/channels_raw.csv")
    name_lookup  = channels_raw[["channel_id", "channel_title"]].drop_duplicates()

    # Videobaserade features
    print("Beräknar videobaserade features...")
    vid_df = _compute_video_features(df["channel_id"])
    df = df.merge(vid_df, on="channel_id", how="left")
    for col in ["avg_duration_sec", "views_cv", "upload_gap_std"]:
        df[col] = df[col].fillna(df[col].median())

    # Per-video-snitt senaste 12 månader
    print("Beräknar per-video-snitt (senaste 12 mån)...")
    recent_df = _compute_recent_video_averages(df["channel_id"])
    df = df.merge(recent_df, on="channel_id", how="left")

    # Google Trends
    print("Hämtar Google Trends-data...")
    trends_df = _fetch_google_trends(
        df[["channel_id"]].merge(name_lookup, on="channel_id", how="left")
    )
    df = df.merge(trends_df[["channel_id", "avg_monthly_search"]], on="channel_id", how="left")
    df["avg_monthly_search"] = df["avg_monthly_search"].fillna(0.0)

    # ── ACTIVITYBENCHMARK ─────────────────────────────────────────────────────
    print("\n=== ActivityBenchmark ===")

    feats_ab = [f for f in ACTIVITY_BENCHMARK_FEATURES if f in df.columns]
    print(f"  Features: {feats_ab}")

    df_ab = df.dropna(subset=feats_ab + ["log_subscriber_count"]).copy()
    X_all = df_ab[feats_ab].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

    # Elitsegment: aktiva, regelbundna, höga visningar, stora prenumeranter
    MIN_RECENT_VIDEOS = np.log1p(2)
    mask = (
        df_ab["log_video_count_recent"] >= MIN_RECENT_VIDEOS
        if "log_video_count_recent" in df_ab.columns
        else pd.Series(True, index=df_ab.index)
    )
    if "upload_gap_std" in df_ab.columns:
        mask = mask & (df_ab["upload_gap_std"] <= df_ab.loc[mask, "upload_gap_std"].quantile(0.75))
    elite = df_ab[mask].copy()
    print(f"  Steg 1 (aktiva, regelbundna): {len(elite)} kanaler")

    # Adaptivt elitgolv: 40% av aktiva kanaler, men minst 15 och helst 150.
    # Med litet dataset skalas tröskeln ner automatiskt.
    MIN_ELITE_TARGET = 150
    MIN_ELITE_FLOOR  = 15
    MIN_ELITE = max(MIN_ELITE_FLOOR, min(MIN_ELITE_TARGET, int(len(elite) * 0.40)))
    print(f"  Adaptivt elitmål: {MIN_ELITE} kanaler (pool: {len(elite)})")

    best_elite = pd.DataFrame()
    for pct in [0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]:
        if "log_estimated_monthly_views" in df_ab.columns:
            elite2 = elite[elite["log_estimated_monthly_views"] >= df_ab["log_estimated_monthly_views"].quantile(pct)]
        else:
            elite2 = elite.copy()
        sub_median = elite2["subscriber_count"].median() if len(elite2) > 0 else 0
        elite3 = elite2[elite2["subscriber_count"] >= sub_median]
        if len(best_elite) == 0 or len(elite3) > len(best_elite):
            best_elite = elite3  # spara bästa hittills
        if len(elite3) >= MIN_ELITE:
            elite = elite3
            print(f"  Steg 2+3 (topp {int((1-pct)*100)}% visningar, topp 50% prenumeranter): "
                  f"{len(elite)} kanaler")
            break
    else:
        # Nådde inte målet – använd det bästa vi hittade om det når golvet
        if len(best_elite) >= MIN_ELITE_FLOOR:
            elite = best_elite
            print(f"  Steg 2+3 (bästa tillgängliga): {len(elite)} elitkanaler "
                  f"(nådde inte mål {MIN_ELITE}, men över golv {MIN_ELITE_FLOOR})")
        else:
            print(f"  För få aktiva kanaler ({len(elite)}) – hoppar över ActivityBenchmark.")
            elite = pd.DataFrame()

    ab_scores = None  # används i kombinerat ranking nedan

    if len(elite) >= MIN_ELITE_FLOOR:
        X_elite = elite[feats_ab].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        y_elite = elite["log_subscriber_count"]

        X_train, X_test, y_train, y_test = train_test_split(
            X_elite, y_elite, test_size=0.2, random_state=42
        )

        param_grid = {
            "n_estimators":    [50, 100, 200],
            "max_depth":       [2, 3, 4],
            "learning_rate":   [0.01, 0.05, 0.1],
            "min_child_weight":[3, 5, 10],
            "reg_lambda":      [1.0, 5.0, 10.0],
        }
        cv_folds = min(5, max(2, len(X_train) // 5))  # minst 2, max 5 folds
        grid_search = GridSearchCV(
            XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42),
            param_grid,
            cv=cv_folds,
            scoring="r2",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"  Bästa hyperparametrar: {best_params}")

        model_ab = grid_search.best_estimator_

        train_r2 = r2_score(y_train, model_ab.predict(X_train))
        cv_r2    = grid_search.best_score_
        test_r2  = r2_score(y_test, model_ab.predict(X_test))
        test_mae = mean_absolute_error(y_test, model_ab.predict(X_test))
        print(f"  Träningsset: {len(X_train)} kanaler, testset: {len(X_test)} kanaler")
        print(f"  R² träning={train_r2:.3f}, 5-fold CV (träning)={cv_r2:.3f}")
        print(f"  R² test={test_r2:.3f}, MAE test (log-skala)={test_mae:.3f}")

        # Träna om slutmodellen på hela elitdatan med bästa hyperparametrar
        model_ab = XGBRegressor(
            **best_params,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        model_ab.fit(X_elite, y_elite, verbose=False)

        _shap_plots(model_ab, X_elite, "sleeping_giant_activity_benchmark")

        df_ab["predicted"] = model_ab.predict(X_all)
        df_ab["residual"]  = df_ab["log_subscriber_count"] - df_ab["predicted"]
        df_ab["sleeping_giant_score"] = MinMaxScaler().fit_transform(
            (-df_ab["residual"]).values.reshape(-1, 1)
        )

        out_ab = name_lookup.merge(df_ab, on="channel_id", how="right")
        show   = [c for c in ["channel_title", "subscriber_count"] + feats_ab
                  + ["residual", "sleeping_giant_score"] if c in out_ab.columns]
        print("\nTopp 10 ActivityBenchmark:")
        print(out_ab.nlargest(10, "sleeping_giant_score")[show]
              .to_string(index=False).encode("ascii", "replace").decode())

        out_ab.to_csv("outputs/predictions/sleeping_giant_activity_benchmark.csv",
                      index=False, encoding="utf-8-sig")
        print("Sparat: sleeping_giant_activity_benchmark.csv")

        ab_scores = df_ab[["channel_id", "sleeping_giant_score"]].rename(
            columns={"sleeping_giant_score": "ab_score"}
        )

    # ── ENGAGEMENTCLUSTER: ITERATIV K-MEANS ──────────────────────────────────
    print("\n=== EngagementCluster ===")

    # Karaktärsnamn och YouTube-sökresultat
    print("Extraherar karaktärsnamn från videotitlar...")
    char_df = _extract_character_names(df["channel_id"])
    print(f"  Karaktärsnamn hittade: {char_df['character_name'].notna().sum()}/{len(char_df)} kanaler")

    print("Hämtar YouTube-sökresultat...")
    yt_search = _fetch_youtube_search_results(
        char_df,
        name_lookup.set_index("channel_id")["channel_title"].to_dict(),
        df.set_index("channel_id")["subscriber_count"].to_dict(),
    )
    df = df.merge(yt_search, on="channel_id", how="left")
    df["search_results_per_sub"] = df["search_results_per_sub"].fillna(0.0)

    # Franchise-filter
    print("\nIdentifierar franchise-kanaler...")
    franchise_ids = _detect_franchise_channels(name_lookup)
    df_ec = df[~df["channel_id"].isin(franchise_ids)].copy()
    print(f"  Kvar efter franchise-filter: {len(df_ec)} kanaler (av {len(df)})")

    for col in ["upload_gap_std", "log_video_count_recent", "upload_frequency"]:
        if col in df_ec.columns:
            df_ec[col] = df_ec[col].fillna(df_ec[col].median())

    feats_ec = [f for f in ENGAGEMENT_CLUSTER_FEATURES if f in df_ec.columns]
    print(f"  Features: {feats_ec}")

    all_ec = []
    for tier_label, (lo, hi) in SUB_TIERS.items():
        mask    = df_ec["subscriber_count"].between(lo, hi, inclusive="left")
        tier_df = df_ec[mask]
        hi_str  = f"{int(hi):,}" if hi != float("inf") else "inf"
        print(f"\n=== {tier_label}: {len(tier_df)} kanaler ({lo:,}–{hi_str} subs) ===")
        if len(tier_df) < 10:
            print("  För få kanaler – hoppar över.")
            continue
        tier_scored = _kmeans_iterative(tier_df, feats_ec,
                                        label=f"EC-{tier_label}", name_lkp=name_lookup)
        tier_scored["sub_tier"] = tier_label
        all_ec.append(tier_scored)

    if all_ec:
        df_ec_all = pd.concat(all_ec)
        out_ec    = name_lookup.merge(df_ec_all, on="channel_id", how="right")
        for tier_label in SUB_TIERS:
            tier_out = out_ec[out_ec["sub_tier"] == tier_label].nlargest(10, "sleeping_giant_score")
            if len(tier_out) == 0:
                continue
            show = [c for c in ["channel_title", "subscriber_count"] + feats_ec
                    + ["cluster", "sleeping_giant_score"] if c in tier_out.columns]
            print(f"\nTopp 10 EngagementCluster [{tier_label}]:")
            print(tier_out[show].to_string(index=False).encode("ascii", "replace").decode())
        out_ec.to_csv("outputs/predictions/sleeping_giant_engagement_cluster.csv",
                      index=False, encoding="utf-8-sig")
        print("\nSparat: sleeping_giant_engagement_cluster.csv")

        # ── KOMBINERAT RANKING: Rangbaserat (EC 70% + AB 30%) ────────────────
        if ab_scores is not None:
            print("\n=== Kombinerat ranking (rangbaserat: EC 70% + AB 30%) ===")
            nc = "channel_title_x" if "channel_title_x" in out_ec.columns else "channel_title"

            combined_tiers = []
            for tier_label, (lo, hi) in SUB_TIERS.items():
                # Bara SG-kandidater från EngagementCluster (score > 0)
                tier_sg = out_ec[
                    (out_ec["sub_tier"] == tier_label) &
                    (out_ec["sleeping_giant_score"] > 0)
                ].copy()
                if len(tier_sg) == 0:
                    continue

                # EC-rank: lägst rank = bäst (rank 1 = högst sleeping_giant_score)
                tier_sg["ec_rank"] = tier_sg["sleeping_giant_score"].rank(
                    ascending=False, method="min"
                ).astype(int)

                # Merge AB-score och beräkna AB-rank inom denna tier
                tier_sg = tier_sg.merge(ab_scores, on="channel_id", how="left")
                tier_sg["ab_score"] = tier_sg["ab_score"].fillna(
                    tier_sg["ab_score"].min() if tier_sg["ab_score"].notna().any() else 0.0
                )
                tier_sg["ab_rank"] = tier_sg["ab_score"].rank(
                    ascending=False, method="min"
                ).astype(int)

                # Kombinerad rank: lägre = bättre
                tier_sg["combined_rank"] = (
                    0.7 * tier_sg["ec_rank"] +
                    0.3 * tier_sg["ab_rank"]
                )
                tier_sg["sub_tier"] = tier_label
                combined_tiers.append(tier_sg)

            if combined_tiers:
                df_comb = pd.concat(combined_tiers)
                for tier_label in SUB_TIERS:
                    t = df_comb[df_comb["sub_tier"] == tier_label].nsmallest(10, "combined_rank")
                    if len(t) == 0:
                        continue
                    show = [c for c in [nc, "subscriber_count", "ec_rank",
                                        "ab_rank", "combined_rank"] if c in t.columns]
                    print(f"\nTopp 10 [{tier_label}]:")
                    print(t[show].rename(columns={nc: "kanal"})
                          .to_string(index=False).encode("ascii", "replace").decode())

                export_cols = {
                    nc:                     "Kanal",
                    "country":              "Land",
                    "subscriber_count":     "Prenumeranter",
                    "sub_tier":             "Storlek",
                    "avg_views_per_video":  "Snitt visningar/video",
                    "ec_rank":              "EngagementCluster Rank",
                    "ab_rank":              "ActivityBenchmark Rank",
                    "combined_rank":        "Kombinerad Rank",
                }
                export_cols = {k: v for k, v in export_cols.items() if k in df_comb.columns}
                df_export = (
                    df_comb[list(export_cols.keys())]
                    .rename(columns=export_cols)
                    .sort_values("Kombinerad Rank", ascending=True)
                    .reset_index(drop=True)
                )
                df_export.index += 1
                df_export.index.name = "Rank"
                df_export["Prenumeranter"] = df_export["Prenumeranter"].astype(int)
                df_export["Snitt visningar/video"] = df_export["Snitt visningar/video"].round(0).astype(int)
                df_export["Kombinerad Rank"] = df_export["Kombinerad Rank"].round(1)
                df_export.to_csv("outputs/predictions/sleeping_giant_combined.csv",
                                 encoding="utf-8-sig")
                print("\nSparat: sleeping_giant_combined.csv")


if __name__ == "__main__":
    run_sleeping_giant()
