import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from src.deployment.export import INPUTS, build
from src.features.residuals import PARAMETERS, compute
from src.models.baseline import Baseline
from src.monitoring.monitor import Monitor


def points(n=8):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "evaporator_in": rng.uniform(11, 15, n), "evaporator_out": rng.uniform(5, 9, n),
        "condenser_in": rng.uniform(20, 30, n), "condenser_out": rng.uniform(25, 35, n),
        "evaporator_flow": rng.uniform(12, 14, n), "condenser_flow": rng.uniform(16, 17, n),
        "evaporator_pressure": rng.uniform(310, 380, n), "condenser_pressure": rng.uniform(550, 1030, n),
        "subcooling": rng.uniform(2, 6, n), "oil_temperature": rng.uniform(40, 50, n), "power": rng.uniform(40, 90, n),
        "discharge_temperature": rng.uniform(40, 55, n),
    })


def baseline():
    rng = np.random.default_rng(2)
    return Baseline(2, [150.0, 7.0, 25.0], [50.0, 2.0, 4.0], rng.normal(0, 0.5, (10, len(PARAMETERS))).tolist())


def test_graph_matches_python_within_the_saturation_fit():
    frame = points()
    session = ort.InferenceSession(build(baseline()).SerializeToString(), providers=["CPUExecutionProvider"])
    residuals, _ = session.run(None, {"readings": frame[list(INPUTS)].to_numpy(np.float32)})
    reference = compute(frame, baseline())[list(PARAMETERS)].to_numpy()
    assert residuals == pytest.approx(reference, abs=0.05)


def test_monitor_turns_steady_stretches_into_points(tmp_path):
    model = tmp_path / "chiller.onnx"
    model.write_bytes(build(baseline()).SerializeToString())
    library = tmp_path / "signatures.json"
    library.write_text('{"spread": {"liquid_approach": 1, "subcooling": 1, "oil_temperature": 1, "condenser_flow": 1, '
                       '"evaporator_flow": 1}, "directions": {"oil": [0, 0, 1, 0, 0]}}')
    monitor = Monitor(model, library)
    for _, row in points(7).iterrows():
        sample = {name: float(row[name]) for name in INPUTS}
        for _ in range(40):
            monitor.step(sample)
        monitor.step(dict(sample, power=sample["power"] + 30))
    assert len(monitor.residuals) == 6 and set(monitor.health) == set(PARAMETERS)
