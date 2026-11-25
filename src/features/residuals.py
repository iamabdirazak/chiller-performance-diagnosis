"""Measured minus healthy: the parameters each fault moves, and the conditions that move them anyway.

The cycle's operating point is set by three things the operator and the
weather choose: the cooling load, the chilled-water setpoint and the
condenser-water temperature. Every health parameter also moves with them, so
a fault shows only as the gap between a reading and what the healthy chiller
reads at the same three conditions.

The condenser approach is not listed: it is the liquid approach plus the
subcooling, so it adds nothing the two do not already say.
"""

import pandas as pd

from src.plant.cycle import states

CONDITIONS = ("load", "evaporator_out", "condenser_in")
PARAMETERS = {  # column: unit
    "liquid_approach": "K",     # condenser heat transfer
    "subcooling": "K",          # refrigerant held in the condenser
    "oil_temperature": "°C",    # lubrication
    "condenser_flow": "kg/s",
    "evaporator_flow": "kg/s",
}


def table(points: pd.DataFrame) -> pd.DataFrame:
    """Steady points with their cycle states, conditions and parameters side by side."""
    return pd.concat([points, states(points).drop(columns="subcooling")], axis=1)


def compute(points: pd.DataFrame, baseline) -> pd.DataFrame:
    frame = table(points)
    return frame[list(PARAMETERS)] - baseline.predict(frame[list(CONDITIONS)]).to_numpy()
