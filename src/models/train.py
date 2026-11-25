"""Learn the healthy chiller and the shape of each fault.

1. Baseline: choose the polynomial degree by leaving one healthy test out at
   a time, then fit on all healthy tests.
2. Spread: how far each parameter strays on a healthy test the baseline never
   saw. This is the unit every shift is measured in.
3. Signatures: the direction each fault moves the parameters, in spread units,
   averaged over its tests.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.residuals import PARAMETERS, compute, table
from src.models.baseline import Baseline

TOLERANCE = 0.02  # a higher degree must cut held-out error by more than 2%


def held_out(healthy_points: pd.DataFrame, degree: int) -> pd.DataFrame:
    """Residuals of every healthy test against a baseline fitted without it."""
    frames = []
    for test, points in healthy_points.groupby("test"):
        others = table(healthy_points[healthy_points["test"] != test])
        frames.append(compute(points, Baseline.fit(others, degree)))
    return pd.concat(frames)


def signatures(points: pd.DataFrame, baseline: Baseline, spread: pd.Series) -> dict[str, list[float]]:
    faulty = points[points["fault"] != "none"]
    z = compute(faulty, baseline) / spread
    directions = {}
    for fault, rows in z.groupby(faulty["fault"]):
        mean = rows.mean().to_numpy()
        directions[fault] = (mean / np.linalg.norm(mean)).tolist()
    return directions


def main() -> None:
    points = pd.read_parquet("data/steady.parquet")
    healthy = points[points["fault"] == "none"]

    errors = {degree: held_out(healthy, degree).pow(2).mean().pow(0.5) for degree in range(0, 4)}
    scores = {degree: float((error / errors[0]).mean()) for degree, error in errors.items()}
    degree = 0
    for candidate in range(1, 4):
        if scores[candidate] < scores[degree] * (1 - TOLERANCE):
            degree = candidate

    spread = errors[degree]
    baseline = Baseline.fit(table(healthy), degree)
    Path("models").mkdir(exist_ok=True)
    baseline.save(Path("models/baseline.json"))
    Path("models/signatures.json").write_text(json.dumps({
        "spread": spread.to_dict(), "directions": signatures(points, baseline, spread)}, indent=2))

    print("held-out error, relative to a constant:", {d: round(s, 3) for d, s in scores.items()}, "-> degree", degree)
    for name, unit in PARAMETERS.items():
        print(f"{name:<17} healthy spread {spread[name]:.3f} {unit}")


if __name__ == "__main__":
    main()
