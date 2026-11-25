"""Check the ONNX graph against the Python pipeline, then time it on one CPU thread.

The Python pipeline takes refrigerant properties from CoolProp and the graph
from its closed-form saturation curve, so the two agree to the accuracy of
that curve, not to float rounding.
"""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.deployment.export import INPUTS
from src.features.residuals import PARAMETERS, compute
from src.models.baseline import Baseline
from src.plant.cycle import states

BUDGET_MS = 50.0


def session(path: str = "models/chiller.onnx") -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def main() -> None:
    model = session()
    points = pd.read_parquet("data/steady.parquet")
    residuals, cycle = model.run(None, {"readings": points[list(INPUTS)].to_numpy(np.float32)})
    reference = compute(points, Baseline.load(Path("models/baseline.json")))[list(PARAMETERS)].to_numpy()
    reference_cycle = states(points)[["load", "cop", "second_law"]].to_numpy()

    one = points[list(INPUTS)].to_numpy(np.float32)[:1]
    for _ in range(200):
        model.run(None, {"readings": one})
    timings = []
    for _ in range(5000):
        start = time.perf_counter()
        model.run(None, {"readings": one})
        timings.append((time.perf_counter() - start) * 1000)

    edge = {
        "size_kb": round(Path("models/chiller.onnx").stat().st_size / 1024, 1),
        "latency_ms_median": round(float(np.median(timings)), 3),
        "latency_ms_p99": round(float(np.percentile(timings, 99)), 3),
        "budget_ms": BUDGET_MS,
        "max_residual_difference": dict(zip(PARAMETERS, np.abs(residuals - reference).max(0).round(4).tolist())),
        "max_second_law_difference": round(float(np.abs(cycle[:, 2] - reference_cycle[:, 2]).max()), 5),
    }
    results = json.loads(Path("reports/results.json").read_text())
    results["edge"] = edge
    Path("reports/results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(edge, indent=2))


if __name__ == "__main__":
    main()
