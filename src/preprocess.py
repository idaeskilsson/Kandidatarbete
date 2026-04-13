import os
import pandas as pd

EUROPEAN_COUNTRIES = {
    "GB", "SE", "NO", "DK", "DE", "FR", "PL", "ES", "IT",
    "NL", "BE", "AT", "FI", "PT", "IE", "CH", "CZ", "HU",
    "RO", "GR", "SK", "HR", "BG", "LT", "LV", "EE",
}

def run_preprocessing():
    os.makedirs("data/interim", exist_ok=True)

    channels = pd.read_csv("data/raw/channels_raw.csv")
    videos = pd.read_csv("data/raw/videos_raw.csv")

    channels = channels.drop_duplicates(subset=["channel_id"]).copy()
    videos = videos.drop_duplicates(subset=["video_id"]).copy()

    channels["published_at"] = pd.to_datetime(channels["published_at"], errors="coerce")
    videos["published_at"] = pd.to_datetime(videos["published_at"], errors="coerce")

    before = len(channels)

    # Behåll bara barnkanaler (True eller ej satt) – ta bort explicit icke-barnkanaler
    channels = channels[channels["made_for_kids"].isin([True, float("nan")]) |
                        channels["made_for_kids"].isna()]

    # Behåll europeiska kanaler + kanaler utan land (kan vara europeiska)
    channels = channels[
        channels["country"].isna() |
        channels["country"].isin(EUROPEAN_COUNTRIES)
    ]

    after = len(channels)
    print(f"  Filtrerade bort {before - after} kanaler "
          f"(ej barn eller utanför Europa). {after} kvar.")

    # Behåll bara videor från kvarvarande kanaler
    videos = videos[videos["channel_id"].isin(channels["channel_id"])]

    channels.to_csv("data/interim/channels_clean.csv", index=False, encoding="utf-8-sig")
    videos.to_csv("data/interim/videos_clean.csv", index=False, encoding="utf-8-sig")

    print("Preprocessing klar.")