"""Each fault at its highest severity, against the healthy chiller at the same conditions.

Before anything is learned, every test point is compared with the five
healthy points nearest to it in load, chilled-water and condenser-water
temperature. The mean difference is the fault's signature; it is coloured
by how many healthy test-to-test spreads it spans.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from src.plant.cycle import states
from src.visualization import style

CONDITIONS = ["load", "evaporator_out", "condenser_in"]
PARAMETERS = {  # column: (label, unit)
    "condenser_approach": ("Condenser\napproach", "K"),
    "liquid_approach": ("Liquid\napproach", "K"),
    "subcooling": ("Subcooling", "K"),
    "discharge_superheat": ("Discharge\nsuperheat", "K"),
    "oil_temperature": ("Oil\ntemperature", "K"),
    "condenser_flow": ("Condenser\nflow", "kg/s"),
    "evaporator_flow": ("Evaporator\nflow", "kg/s"),
}
FAULTS = {"fouling": "Condenser fouling, 45%", "non-condensables": "Non-condensables, 5%",
          "overcharge": "Overcharge, 40%", "leak": "Refrigerant leak, 40%", "oil": "Excess oil, 50%",
          "condenser flow": "Condenser flow, −40%", "evaporator flow": "Evaporator flow, −40%"}


def shifts() -> tuple[pd.DataFrame, pd.Series]:
    points = pd.read_parquet("data/steady.parquet")
    table = pd.concat([points, states(points).drop(columns="subcooling")], axis=1)
    healthy = table[table["fault"] == "none"]
    scale = healthy[CONDITIONS].std()
    reference, values = (healthy[CONDITIONS] / scale).to_numpy(), healthy[list(PARAMETERS)].to_numpy()

    def shift(test: pd.DataFrame) -> pd.Series:
        here = (test[CONDITIONS] / scale).to_numpy()
        nearest = np.argsort(((reference[None] - here[:, None]) ** 2).sum(-1), axis=1)[:, 1:6]
        return pd.Series((test[list(PARAMETERS)].to_numpy() - values[nearest].mean(1)).mean(0), index=list(PARAMETERS))

    per_test = table.groupby("test").apply(lambda t: pd.concat([t[["fault", "severity"]].iloc[0], shift(t)]))
    spread = per_test[per_test["fault"] == "none"][list(PARAMETERS)].astype(float).std()
    worst = per_test[per_test["fault"] != "none"].sort_values("severity").groupby("fault").last()
    return worst.loc[list(FAULTS), list(PARAMETERS)].astype(float), spread


def draw(theme: str, directory: Path) -> None:
    colors = style.apply(theme)
    values, spread = shifts()
    z = (values / spread).abs().clip(upper=12) / 12
    ramp = LinearSegmentedColormap.from_list("", [colors["background"], colors["faint"], colors["base"], colors["strong"]])

    figure, axis = plt.subplots(figsize=(9.6, 5.0))
    for i, fault in enumerate(FAULTS):
        for j, parameter in enumerate(PARAMETERS):
            level = z.loc[fault, parameter]
            axis.add_patch(FancyBboxPatch((j + 0.06, -i - 0.94), 0.88, 0.88, boxstyle="round,pad=0,rounding_size=0.12",
                                          facecolor=ramp(level), edgecolor="none"))
            value = values.loc[fault, parameter]
            ink = "#FFFFFF" if level > 0.55 and theme == "light" else colors["ink"]
            axis.text(j + 0.5, -i - 0.5, f"{value:+.1f}", ha="center", va="center", fontsize=8.5,
                      color=ink if level > 0.15 else colors["secondary"])
    axis.set_xlim(0, len(PARAMETERS))
    axis.set_ylim(-len(FAULTS), 0)
    axis.set_xticks(np.arange(len(PARAMETERS)) + 0.5,
                    [f"{label}, {unit}" for label, unit in PARAMETERS.values()], fontsize=8)
    axis.xaxis.tick_top()
    axis.set_yticks(-np.arange(len(FAULTS)) - 0.5, list(FAULTS.values()), fontsize=8.5)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title("Fault Signatures", pad=46)
    style.save(figure, directory, "signatures", theme)


if __name__ == "__main__":
    for theme in ("light", "dark"):
        draw(theme, Path("reports/figures"))
