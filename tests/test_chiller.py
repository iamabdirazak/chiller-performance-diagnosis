import numpy as np
import pandas as pd
import pytest
from CoolProp.CoolProp import PropsSI

from src.data.prepare import convert, label, steady_points
from src.models.baseline import Baseline
from src.monitoring.diagnosis import advise
from src.plant.cycle import saturation_temperature, states
from src.plant.saturation import temperature


def test_saturation_curve_matches_coolprop():
    pressure = np.linspace(250, 1200, 40)
    assert np.abs(temperature(pressure) - saturation_temperature(pressure)).max() < 0.05


def test_carnot_and_second_law():
    point = pd.DataFrame([{"evaporator_in": 12.0, "evaporator_out": 7.0, "condenser_in": 29.0, "condenser_out": 34.0,
                           "evaporator_flow": 10.0, "condenser_flow": 12.0, "evaporator_pressure": 340.0,
                           "condenser_pressure": 900.0, "subcooling": 3.0, "discharge_temperature": 45.0, "power": 50.0}])
    s = states(point).iloc[0]
    t_e, t_c = saturation_temperature(340.0)[0] + 273.15, saturation_temperature(900.0)[0] + 273.15
    assert s["load"] == pytest.approx(10 * 4.186 * 5)
    assert s["second_law"] == pytest.approx(s["cop"] * (t_c - t_e) / t_e)


def test_labels_and_units():
    assert label("cf45") == ("fouling", 45) and label("normal_r1") == ("none", 0)
    raw = pd.DataFrame({"Time": [0.0], "TWEI": [212.0], "TWEO": [32.0], "TWCI": [50], "TWCO": [50], "FWE": [100.0],
                        "FWC": [0], "PRE": [0.0], "PRC": [0], "TRC_sub": [9.0], "T_suc": [32], "TR_dis": [32],
                        "TO_sump": [32], "PO_net": [1.0], "kW": [10]})
    out = convert(raw).iloc[0]
    assert out["evaporator_in"] == pytest.approx(100) and out["subcooling"] == pytest.approx(5)
    assert out["evaporator_pressure"] == pytest.approx(101.325, abs=0.01)
    assert out["evaporator_flow"] == pytest.approx(6.309, abs=0.001)


def test_steady_points_skip_transients():
    still = {"evaporator_out": 7.0, "condenser_in": 29.0, "power": 60.0}
    rows = [dict(still, time=10 * i) for i in range(60)]
    rows += [dict(still, time=10 * (60 + i), power=60.0 + 10 * (i % 2)) for i in range(20)]
    points = steady_points(pd.DataFrame(rows))
    assert len(points) == 1 and points["power"].iloc[0] == pytest.approx(60.0)


def test_one_odd_point_raises_nothing():
    spread = {name: 1.0 for name in ("liquid_approach", "subcooling", "oil_temperature", "condenser_flow", "evaporator_flow")}
    directions = {"fouling": [1, 0, 0, 0, 0]}
    residuals = pd.DataFrame(0.0, index=range(20), columns=list(spread))
    residuals.loc[10, "liquid_approach"] = 5.0  # one odd point lifts a single six-point window over the limit
    assert advise(residuals, spread, directions).isna().all()
    residuals["liquid_approach"] = 1.0
    assert (advise(residuals, spread, directions).dropna() == "fouling").all()
