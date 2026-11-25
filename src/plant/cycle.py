"""The vapour-compression cycle, as the sensors see it.

Water-side first law gives the heat moved at each heat exchanger. R-134a
saturation properties turn the two refrigerant pressures into condensing and
evaporating temperatures. The Carnot cycle between those temperatures gives
the ideal, and the second-law efficiency says how far the chiller is from it.

The two condenser approaches separate faults that a single one would mix up.
The saturation approach uses the temperature implied by the measured pressure;
the liquid approach uses the liquid's actual temperature. Gas that does not
condense adds its own partial pressure, lifting the first and not the second.
"""

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

CP_WATER = 4.186  # kJ/(kg K)


def saturation_temperature(pressure_kpa) -> np.ndarray:
    """R-134a saturation temperature in °C at an absolute pressure in kPa."""
    return np.array([PropsSI("T", "P", p * 1000, "Q", 0, "R134a") for p in np.atleast_1d(pressure_kpa)]) - 273.15


def states(points: pd.DataFrame) -> pd.DataFrame:
    """Heat duties, efficiencies and the temperature differences that carry each fault."""
    t_evap = saturation_temperature(points["evaporator_pressure"])
    t_cond = saturation_temperature(points["condenser_pressure"])
    q_evap = points["evaporator_flow"] * CP_WATER * (points["evaporator_in"] - points["evaporator_out"])
    q_cond = points["condenser_flow"] * CP_WATER * (points["condenser_out"] - points["condenser_in"])
    cop = q_evap / points["power"]
    carnot = (t_evap + 273.15) / (t_cond - t_evap)
    return pd.DataFrame({
        "load": q_evap,                                             # kW of cooling
        "heat_rejected": q_cond,                                    # kW
        "balance": (q_cond - q_evap - points["power"]) / q_cond,    # first law closure, fraction
        "cop": cop,
        "second_law": cop / carnot,                                 # COP / Carnot COP
        "condenser_approach": t_cond - points["condenser_out"],     # K
        "evaporator_approach": points["evaporator_out"] - t_evap,   # K
        "subcooling": points["subcooling"],                         # K
        "liquid_approach": t_cond - points["subcooling"] - points["condenser_out"],  # K, liquid leaving vs water leaving
        "discharge_superheat": points["discharge_temperature"] - t_cond,  # K
        "condenser_rise": points["condenser_out"] - points["condenser_in"],    # K
        "evaporator_drop": points["evaporator_in"] - points["evaporator_out"], # K
    }, index=points.index)
