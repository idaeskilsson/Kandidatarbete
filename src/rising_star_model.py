"""
Typ 1: Rising Star
Identifierar kanaler på uppgång med momentum-features och XGBoost.
Predictar subscriber-tillväxt om 6 månader baserat på historiskt momentum.

Kräver historisk månadsdata i: data/raw/channel_history.csv
Förväntat format (en rad per kanal per månad):
  channel_id | month | subscribers | total_views | video_count |
  likes | comments | channel_age_days | country

Sparar till: outputs/predictions/rising_star_rankings.csv
"""

import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

HISTORY_FILE = "data/raw/channel_history.csv"

CPM_BY_COUNTRY = {
    "SE": 3.5, "NO": 4.0, "DK": 3.8, "GB": 5.0,
    "DE": 3.2, "FR": 2.8, "PL": 1.2,
    "ES": 2.5, "IT": 2.5,
}
DEFAULT_CPM = 2.0

FEATURE_COLS = [
    "avg_views_growth",
    "avg_sub_growth",
    "views_acceleration",
    "sub_acceleration",
    "avg_engagement",
    "engagement_trend",
    "upload_consistency",
    "viral_ratio",
    "sub_to_views_ratio_trend",
    "log_subscribers",
    "current_age_months",
    "cpm_factor",
]


# ==============================================================================
# DATA OCH MÅNADSBERÄKNINGAR
# ==============================================================================

def _load_and_prepare(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.sort_values(["channel_id", "month"])
    return df


def _compute_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["views_growth"] = df.groupby("channel_id")["total_views"].pct_change()
    df["sub_growth"] = df.groupby("channel_id")["subscribers"].pct_change()
    df["video_delta"] = df.groupby("channel_id")["video_count"].diff()

    df["views_this_month"] = df.groupby("channel_id")["total_views"].diff().clip(lower=0)
    df["engagement_rate"] = (
        (df["likes"] + df["comments"]) /
        df["views_this_month"].replace(0, np.nan)
    )

    df["upload_consistency"] = (
        df.groupby("channel_id")["video_delta"]
          .transform(lambda x: 1 / (x.rolling(3).std() + 1))
    )

    df["cpm"] = df["country"].map(CPM_BY_COUNTRY).fillna(DEFAULT_CPM)
    df["rpv"] = (
        (df["views_this_month"] / df["video_count"].replace(0, np.nan)) *
        0.45 * df["cpm"] / 1000
    )

    return df


# ==============================================================================
# FEATURE ENGINEERING – MOMENTUM
# ==============================================================================

def _build_momentum_features(df: pd.DataFrame,
                               lookback_months: int = 6) -> pd.DataFrame:
    """
    Bygger en observation per kanal per tidpunkt T med:
    - X: momentum-features från de senaste lookback_months månaderna
    - y: subscriber-tillväxt 6 månader framåt
    """
    records = []

    for channel_id, group in df.groupby("channel_id"):
        group = group.reset_index(drop=True)

        if len(group) < lookback_months + 6:
            continue

        for t in range(lookback_months, len(group) - 6):
            window = group.iloc[t - lookback_months:t]
            future = group.iloc[t + 6]
            current = group.iloc[t]

            vg = window["views_growth"].dropna()
            sg = window["sub_growth"].dropna()

            avg_views_growth = vg.mean()
            avg_sub_growth = sg.mean()
            views_acceleration = vg.diff().mean()
            sub_acceleration = sg.diff().mean()

            eng = window["engagement_rate"].dropna()
            avg_engagement = eng.mean()
            engagement_trend = eng.diff().mean()

            avg_consistency = window["upload_consistency"].mean()

            avg_views = window["views_this_month"].mean()
            viral_ratio = (window["views_this_month"] > 2 * avg_views).mean()

            # sub_to_views_ratio_trend: ökar konverteringen tittare→prenumerant?
            svr = (window["sub_growth"] / window["views_growth"].replace(0, np.nan)).dropna()
            sub_to_views_ratio_trend = svr.diff().mean()

            ip_strength = (
                current["views_this_month"] /
                max(current["video_count"], 1) /
                max(current["subscribers"], 1)
            )
            content_gap = current["subscribers"] / max(current["video_count"], 1)

            future_sub_growth = (
                (future["subscribers"] - current["subscribers"]) /
                max(current["subscribers"], 1)
            )

            records.append({
                "channel_id":          channel_id,
                "avg_views_growth":    avg_views_growth,
                "avg_sub_growth":      avg_sub_growth,
                "views_acceleration":  views_acceleration,
                "sub_acceleration":    sub_acceleration,
                "avg_engagement":      avg_engagement,
                "engagement_trend":    engagement_trend,
                "upload_consistency":       avg_consistency,
                "viral_ratio":             viral_ratio,
                "sub_to_views_ratio_trend": sub_to_views_ratio_trend,
                "ip_strength":             ip_strength,
                "content_gap":         content_gap,
                "current_subscribers": current["subscribers"],
                "log_subscribers":     np.log1p(current["subscribers"]),
                "current_age_months":  current["channel_age_days"] / 30,
                "cpm_factor":          current["cpm"],
                "y_future_sub_growth": future_sub_growth,
            })

    return pd.DataFrame(records)


# ==============================================================================
# TRÄNA MODELL
# ==============================================================================

def _train_rising_star_model(feature_df: pd.DataFrame):
    """XGBoost-klassificering: predictar om kanalen tillhör topp 25% i tillväxt
    bland kanaler av sin storlek. Storleksgrupper:
      liten:  1 000 –  9 999 subs
      medel: 10 000 – 49 999 subs
      stor:  50 000+           subs
    """
    clean = feature_df.dropna(subset=["y_future_sub_growth"]).copy()

    # Dela in i storleksgrupper och beräkna 75:e percentilen inom varje grupp
    bins   = [0, 10_000, 50_000, float("inf")]
    labels = ["liten", "medel", "stor"]
    clean["size_group"] = pd.cut(clean["current_subscribers"], bins=bins, labels=labels)

    clean["y"] = clean.groupby("size_group")["y_future_sub_growth"].transform(
        lambda x: (x > x.quantile(0.75)).astype(int)
    )

    for grp in labels:
        sub = clean[clean["size_group"] == grp]
        if len(sub) > 0:
            pct = sub[sub["y"] == 1]["y_future_sub_growth"].min()
            print(f"  Tillväxttröskel ({grp}): {pct:.2%} "
                  f"({len(sub)} kanaler, {sub['y'].sum()} i topp 25%)")

    y = clean["y"]

    X = clean[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    probs = model.predict_proba(X_test)[:, 1]
    print(f"Rising Star modell AUC: {roc_auc_score(y_test, probs):.3f}")

    return model, X_test, y_test


# ==============================================================================
# SHAP-FÖRKLARING
# ==============================================================================

def _explain_model(model, X_test: pd.DataFrame):
    os.makedirs("outputs", exist_ok=True)
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    mean_shap = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": np.abs(shap_values.values).mean(axis=0),
    }).sort_values("importance", ascending=False)

    print("\nFeature importance (SHAP):")
    print(mean_shap.to_string(index=False))

    # Bar-plot
    plt.figure(figsize=(8, 5))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig("outputs/rising_star_shap_bar.png", dpi=150)
    plt.close()

    # Beeswarm-plot
    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig("outputs/rising_star_shap_beeswarm.png", dpi=150)
    plt.close()

    print("SHAP-plottar sparade: outputs/rising_star_shap_bar.png, outputs/rising_star_shap_beeswarm.png")

    return shap_values


# ==============================================================================
# RANKA KANDIDATER
# ==============================================================================

def _score_candidates(model, df_metrics: pd.DataFrame) -> pd.DataFrame:
    feature_df = _build_momentum_features(df_metrics, lookback_months=6)
    latest = feature_df.groupby("channel_id").last().reset_index()

    X_candidates = latest[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    latest["prob_top25pct_growth"] = model.predict_proba(X_candidates)[:, 1]

    return latest[[
        "channel_id",
        "current_subscribers",
        "avg_sub_growth",
        "avg_views_growth",
        "avg_engagement",
        "upload_consistency",
        "prob_top25pct_growth",
    ]].sort_values("prob_top25pct_growth", ascending=False)


# ==============================================================================
# PIPELINE-INGÅNGSPUNKT
# ==============================================================================

def run_rising_star():
    if not os.path.exists(HISTORY_FILE):
        print(
            f"Rising Star hoppas över – historikfil saknas: {HISTORY_FILE}\n"
            "Lägg till månadsvis historikdata per kanal för att aktivera modellen."
        )
        return

    os.makedirs("outputs/predictions", exist_ok=True)

    df_raw = _load_and_prepare(HISTORY_FILE)

    # Behåll bara kanaler som finns i model_dataset (redan filtrerade)
    model_ids = pd.read_csv("data/processed/model_dataset.csv")["channel_id"]
    df_raw = df_raw[df_raw["channel_id"].isin(model_ids)]

    df_metrics = _compute_monthly_metrics(df_raw)

    feature_df = _build_momentum_features(df_metrics, lookback_months=6)
    print(f"Rising Star – {len(feature_df)} observationer från "
          f"{feature_df['channel_id'].nunique()} kanaler")

    model, X_test, y_test = _train_rising_star_model(feature_df)

    _explain_model(model, X_test)

    rankings = _score_candidates(model, df_metrics)

    # Lägg till kanalnamn
    channels_raw = pd.read_csv("data/raw/channels_raw.csv")
    name_lookup  = channels_raw[["channel_id", "channel_title"]].drop_duplicates()
    rankings     = name_lookup.merge(rankings, on="channel_id", how="right")

    # Spara tre separata rankingar baserat på prenumeranttröskel
    tiers = [
        (1_000,  "rising_star_1k.csv"),
        (10_000, "rising_star_10k.csv"),
        (50_000, "rising_star_50k.csv"),
    ]

    for min_subs, filename in tiers:
        tier = rankings[rankings["current_subscribers"] >= min_subs]
        path = f"outputs/predictions/{filename}"
        tier.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\nTopp 10 Rising Stars (>= {min_subs:,} prenumeranter, {len(tier)} kanaler totalt):")
        print(tier.head(10).to_string(index=False).encode("ascii", "replace").decode())

    print("\nSparat: rising_star_1k.csv, rising_star_10k.csv, rising_star_50k.csv")


if __name__ == "__main__":
    run_rising_star()
