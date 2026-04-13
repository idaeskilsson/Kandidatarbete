import os
import pandas as pd

CPM_BY_COUNTRY = {
    "SE": 3.5,
    "GB": 4.5,
    "FR": 3.5,
    "DE": 4.0,
    "ES": 2.5,
    "IT": 2.5,
    "PL": 1.8,
}

DEFAULT_CPM = 2.5
DEFAULT_MONETIZATION_RATE = 0.45

def get_cpm(country_code):
    if pd.isna(country_code):
        return DEFAULT_CPM
    return CPM_BY_COUNTRY.get(country_code, DEFAULT_CPM)

def run_build_target():
    os.makedirs("data/processed", exist_ok=True)
    channels = pd.read_csv("data/interim/channels_clean.csv")
    videos = pd.read_csv("data/interim/videos_clean.csv")

    video_agg = (
        videos.groupby("channel_id")
        .agg(
            sample_video_count=("video_id", "count"),
            avg_views_per_video=("view_count", "mean"),
            avg_likes_per_video=("like_count", "mean"),
            avg_comments_per_video=("comment_count", "mean")
        )
        .reset_index()
    )

    # Behåll bara kanaler med videodata – kanaler utan videor kan inte få RPV
    df = channels.merge(video_agg, on="channel_id", how="inner")

    df["avg_views_per_video"] = df["avg_views_per_video"].fillna(0)
    df["sample_video_count"] = df["sample_video_count"].fillna(0)

    df["cpm"] = df["country"].apply(get_cpm)
    df["monetization_rate"] = DEFAULT_MONETIZATION_RATE

    df["yv"] = (
        df["avg_views_per_video"] *
        df["monetization_rate"] *
        df["cpm"] / 1000
    )

    df.to_csv("data/processed/model_dataset_with_target.csv", index=False, encoding="utf-8-sig")
    print("Target byggd.")