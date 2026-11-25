"""Health in engineering units at each tested severity, against the advisory limit."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.residuals import PARAMETERS, compute
from src.models.baseline import Baseline
from src.monitoring.diagnosis import LIMITS
from src.visualization import style

PANELS = {  # fault: (parameter, title, severity unit)
    "fouling": ("liquid_approach", "Condenser Fouling", "% tubes plugged"),
    "non-condensables": ("subcooling", "Non-Condensables", "% of charge"),
    "overcharge": ("subcooling", "Overcharge", "% over nominal"),
    "leak": ("subcooling", "Refrigerant Leak", "% under nominal"),
    "oil": ("oil_temperature", "Excess Oil", "% over nominal"),
    "condenser flow": ("condenser_flow", "Condenser Flow", "% reduced"),
    "evaporator flow": ("evaporator_flow", "Evaporator Flow", "% reduced"),
}
LABELS = {"liquid_approach": "Liquid approach", "subcooling": "Subcooling", "oil_temperature": "Oil temperature",
          "condenser_flow": "Condenser flow", "evaporator_flow": "Evaporator flow"}


def draw(theme: str, directory: Path) -> None:
    colors = style.apply(theme)
    points = pd.read_parquet("data/steady.parquet")
    residuals = compute(points, Baseline.load(Path("models/baseline.json")))
    residuals[["test", "fault", "severity"]] = points[["test", "fault", "severity"]]
    spread = json.loads(Path("models/signatures.json").read_text())["spread"]
    correct = {r["test"]: r["correct"] for r in json.loads(Path("reports/results.json").read_text())["tests"]}

    figure, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    axes = axes.ravel()
    axes[-1].axis("off")
    for axis, (fault, (parameter, title, unit)) in zip(axes, PANELS.items()):
        tests = residuals[residuals["fault"] == fault].groupby("test").agg(
            severity=("severity", "first"), value=(parameter, "mean")).sort_values("severity")
        x = np.arange(len(tests))
        top = max(abs(tests["value"]).max(), LIMITS[parameter]) * 1.25
        axis.set_xlim(-0.6, len(tests) - 0.4)
        axis.set_ylim(-top if tests["value"].mean() < 0 else -top * 0.12, top if tests["value"].mean() > 0 else top * 0.12)
        sign = 1 if tests["value"].mean() > 0 else -1
        axis.axhspan(-2 * spread[parameter], 2 * spread[parameter], color=colors["hairline"], alpha=0.6, linewidth=0, zorder=0)
        axis.axhline(sign * LIMITS[parameter], color=colors["ink"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
        axis.figure.canvas.draw()
        shades = [colors["strong"] if correct[test] else colors["soft"] for test in tests.index]
        style.rounded_bars(axis, x, tests["value"], shades, width=0.58, radius=0.1)
        axis.axhline(0, color=colors["secondary"], linewidth=0.8, zorder=1)
        axis.set_xticks(x, [f"{s}" for s in tests["severity"]])
        axis.set_xlabel(f"Severity, {unit}")
        axis.set_ylabel(f"{LABELS[parameter]}, {PARAMETERS[parameter]}")
        axis.set_title(title)

    style.legend(figure, [(colors["strong"], "Named correctly"), (colors["soft"], "Missed or misnamed"),
                          (colors["ink"], "Limit"), (colors["hairline"], "Healthy ±2σ")], colors)
    figure.tight_layout(h_pad=2.2, w_pad=2.0)
    style.save(figure, directory, "health", theme)


if __name__ == "__main__":
    for theme in ("light", "dark"):
        draw(theme, Path("reports/figures"))
