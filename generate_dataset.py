"""Generate a synthetic (physics-inspired) end milling dataset.
Replace data/milling_data.csv with your own experimental data for real use
(same column names: see common.RAW_COLS + Ra)."""
import numpy as np
import pandas as pd
from pathlib import Path
from common import MATERIALS

MAT_BASE = {"Al6061": 0.55, "AISI1045": 1.0, "Ti6Al4V": 1.25}


def generate(n=800, seed=42, path="data/milling_data.csv") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "material": rng.choice(MATERIALS, n),
        "tool_dia": rng.choice([6, 8, 10, 12], n),
        "flutes": rng.choice([2, 3, 4], n),
        "speed": rng.uniform(2000, 10000, n).round(0),
        "feed_per_tooth": rng.uniform(0.02, 0.15, n).round(4),
        "doc": rng.uniform(0.5, 4.0, n).round(2),
        "woc": rng.uniform(0.5, 6.0, n).round(2),
        "tool_wear": rng.uniform(0.0, 0.30, n).round(3),
    })
    vc = np.pi * df.tool_dia * df.speed / 1000
    base = df.material.map(MAT_BASE)
    ra = (base
          * (df.feed_per_tooth / 0.05) ** 0.85
          * (vc / 150) ** -0.25
          * (1 + 4.0 * df.tool_wear)
          * (df.doc / 2) ** 0.15
          * (df.woc / 3) ** 0.10
          * (6 / df.tool_dia) ** 0.10)
    # chatter-like bump at high depth + low stiffness (small tools, deep cuts)
    ra *= 1 + 0.25 * np.clip(df.doc / df.tool_dia - 0.3, 0, None) * 5
    ra *= rng.lognormal(0, 0.07, n)          # measurement / process noise
    df["Ra"] = ra.round(4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    d = generate()
    print(d.describe().T[["mean", "min", "max"]])
