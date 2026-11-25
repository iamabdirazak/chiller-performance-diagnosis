"""From residuals to advice: limits in engineering units, cause from the fault's shape.

Health is the median residual over the last six steady points, about three
hours of operation, so one odd point cannot move it. An advisory needs a parameter past its limit in two
windows running. Its cause is
the fault whose signature points most nearly the same way as the residuals.
"""

import numpy as np
import pandas as pd

from src.features.residuals import PARAMETERS

WINDOW = 6  # steady points
ON_DELAY = 2  # windows in a row
LIMITS = {  # how far a parameter may move before the work is worth doing
    "liquid_approach": 0.3,    # K, condenser heat transfer
    "subcooling": 1.0,         # K, refrigerant charge either way
    "oil_temperature": 2.0,    # °C
    "condenser_flow": 0.84,    # kg/s, 5% of design flow
    "evaporator_flow": 0.68,   # kg/s, 5% of design flow
}
ADVICE = {
    "fouling": "Condenser fouling. Clean the condenser tubes.",
    "non-condensables": "Non-condensable gas in the condenser. Purge the refrigerant circuit.",
    "overcharge": "Refrigerant overcharge. Recover refrigerant to nominal charge.",
    "leak": "Refrigerant leak. Leak-test and recharge.",
    "oil": "Excess oil in the circuit. Check and correct the oil charge.",
    "condenser flow": "Low condenser water flow. Check the condenser pump, valves and strainer.",
    "evaporator flow": "Low chilled-water flow. Check the chilled-water pump, valves and strainer.",
}


def health(residuals: pd.DataFrame) -> pd.DataFrame:
    return residuals[list(PARAMETERS)].rolling(WINDOW).median()


def cause(window: pd.Series, spread: dict, directions: dict) -> str:
    z = np.array([window[name] / spread[name] for name in PARAMETERS])
    return max(directions, key=lambda fault: float(z @ np.array(directions[fault])))


def advise(residuals: pd.DataFrame, spread: dict, directions: dict) -> pd.Series:
    """Advisory cause, or None, after each steady point."""
    windows = health(residuals)
    over = pd.concat([windows[name].abs() > limit for name, limit in LIMITS.items()], axis=1).any(axis=1)
    held = over.rolling(ON_DELAY).sum() == ON_DELAY
    return pd.Series([cause(row, spread, directions) if flag else None for flag, (_, row) in zip(held, windows.iterrows())],
                     index=residuals.index)
