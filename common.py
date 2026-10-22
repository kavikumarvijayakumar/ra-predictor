"""Shared constants and feature engineering used by training and the app."""
import numpy as np
import pandas as pd

MATERIALS = ["Al6061", "AISI1045", "Ti6Al4V"]
RAW_COLS = ["speed", "feed_per_tooth", "doc", "woc",
            "tool_wear", "tool_dia", "flutes", "material"]
NUM_COLS = ["speed", "feed_per_tooth", "doc", "woc", "tool_wear",
            "tool_dia", "flutes", "cutting_speed", "mrr"]
CAT_COLS = ["material"]
TARGET = "Ra"

# Units: speed [rpm], feed_per_tooth [mm/tooth], doc/woc [mm],
# tool_wear VB [mm], tool_dia [mm], Ra [um]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physically meaningful derived features."""
    d = df.copy()
    d["cutting_speed"] = np.pi * d["tool_dia"] * d["speed"] / 1000.0          # m/min
    d["mrr"] = d["feed_per_tooth"] * d["flutes"] * d["speed"] * d["doc"] * d["woc"] / 1000.0  # cm3/min
    return d


def quality_grade(ra: float) -> str:
    if ra <= 0.8:
        return "Fine finish (N6-N7)"
    if ra <= 1.6:
        return "Good finish (N7-N8)"
    if ra <= 3.2:
        return "Standard machined (N9)"
    if ra <= 6.3:
        return "Rough (N10-N11)"
    return "Very rough"
