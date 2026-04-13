import os
import pandas as pd
import numpy as np
from src.filter_channels import filter_acquirable

def run_build_features():
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv("data/processed/model_dataset_with_target.csv")

    channels_df = pd.read_csv("data/raw/channels_raw.csv")
    df = filter_acquirable(df, channels_df)

    # Läs datum som UTC så allt får samma tidszon
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

    # Gör "today" också UTC-aware
    today = pd.Timestamp.now(tz="UTC")

    df["channel_age_days"] = (today - df["published_at"]).dt.days
    df["channel_age_days"] = df["channel_age_days"].replace(0, np.nan)

    df["subscriber_count"] = df["subscriber_count"].fillna(0)
    df["video_count"] = df["video_count"].fillna(0)
    df["view_count"] = df["view_count"].fillna(0)
    df["avg_likes_per_video"] = df["avg_likes_per_video"].fillna(0)
    df["avg_comments_per_video"] = df["avg_comments_per_video"].fillna(0)
    df["avg_views_per_video"] = df["avg_views_per_video"].fillna(0)

    df["subscribers_per_video"] = df["subscriber_count"] / df["video_count"].replace(0, np.nan)
    df["views_per_video_reported"] = df["view_count"] / df["video_count"].replace(0, np.nan)
    df["upload_frequency"] = df["video_count"] / df["channel_age_days"]
    df["engagement_rate_like"] = df["avg_likes_per_video"] / df["avg_views_per_video"].replace(0, np.nan)
    df["engagement_rate_comment"] = df["avg_comments_per_video"] / df["avg_views_per_video"].replace(0, np.nan)

    df["is_made_for_kids"] = df["made_for_kids"].fillna(False).astype(int)
    df["is_self_declared_made_for_kids"] = df["self_declared_made_for_kids"].fillna(False).astype(int)

    df = df.fillna(0)

    df.to_csv("data/processed/model_dataset.csv", index=False, encoding="utf-8-sig")
    print("Features byggda.")