"""R-134a saturation temperature in closed form, for the edge.

CoolProp is the reference for refrigerant properties, but a property library
cannot run inside an ONNX graph. Over the pressures this chiller sees, the
Antoine form

    T = 1 / (a - b ln p) - c

fits CoolProp's saturation curve to within 0.04 K, which is a fifth of the
healthy spread of the smallest health parameter.
"""

from functools import cache

import numpy as np
from CoolProp.CoolProp import PropsSI

RANGE = (200.0, 1600.0)  # kPa absolute, wider than any test reached


@cache
def coefficients() -> tuple[float, float, float]:
    """(a, b, c) fitted to CoolProp, with c chosen on a fine grid and a, b by least squares."""
    pressure = np.linspace(*RANGE, 400)
    temperature = np.array([PropsSI("T", "P", p * 1000, "Q", 0, "R134a") for p in pressure]) - 273.15
    design = np.column_stack([np.ones_like(pressure), -np.log(pressure)])
    best = None
    for c in np.arange(180.0, 320.0, 0.05):
        (a, b), *_ = np.linalg.lstsq(design, 1 / (temperature + c), rcond=None)
        error = np.abs(1 / (a - b * np.log(pressure)) - c - temperature).max()
        if best is None or error < best[0]:
            best = (error, float(a), float(b), float(c))
    return best[1:]


def temperature(pressure_kpa):
    a, b, c = coefficients()
    return 1 / (a - b * np.log(pressure_kpa)) - c
