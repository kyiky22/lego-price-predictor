"""Feature engineering untuk inference. Harus identik dengan yang dipakai saat training."""
import numpy as np
import pandas as pd

RAW_COLUMNS = ["Set_ID", "Year", "Theme", "Theme_Group", "Category", "Packaging",
               "Num_Instructions", "Availability", "Pieces", "Minifigures", "Owned",
               "Rating", "USD_MSRP", "Total_Quantity"]

NUMERIC_FEATURES = ["Year", "Age", "Set_Variant", "Pieces", "Minifigures", "Owned", "Rating",
                    "Num_Instructions", "USD_MSRP", "MSRP_per_Piece", "Total_Quantity",
                    "Has_Minifig", "Has_Rating"]
CATEGORICAL_FEATURES = ["Theme", "Theme_Group", "Category", "Packaging", "Availability"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _clean_text(v):
    return v.strip() if isinstance(v, str) else np.nan


def prepare_features(raw: pd.DataFrame, current_year: int) -> pd.DataFrame:
    """Raw kolom sets.csv -> fitur siap masuk pipeline model.

    current_year HARUS nilai yang dibekukan saat training (bukan tahun hari ini),
    kalau tidak fitur Age akan bergeser sendiri.
    """
    df = raw.reindex(columns=RAW_COLUMNS).copy()

    for col in ["Set_ID"] + CATEGORICAL_FEATURES:
        df[col] = df[col].map(_clean_text).astype(object)
    for col in ["Packaging", "Availability"]:
        df[col] = df[col].replace({"{Not specified}": "Unknown"})
    df["Theme_Group"] = df["Theme_Group"].fillna("Unknown")

    for col in ["Year", "Pieces", "Minifigures", "Owned", "Rating",
                "Num_Instructions", "USD_MSRP", "Total_Quantity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Pieces"] = df["Pieces"].replace(0, np.nan)
    df["Minifigures"] = df["Minifigures"].fillna(0)

    df["Set_Variant"] = pd.to_numeric(df["Set_ID"].str.split("-").str[-1], errors="coerce")
    df["Age"] = current_year - df["Year"]
    df["Has_Minifig"] = (df["Minifigures"] > 0).astype(int)
    df["Has_Rating"] = (df["Rating"] > 0).astype(int)
    df["MSRP_per_Piece"] = df["USD_MSRP"] / df["Pieces"]

    return df[FEATURE_COLUMNS]
