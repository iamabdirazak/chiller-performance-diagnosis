"""The loop that runs on site: one call per 10-second sample of the chiller's sensors.

It finds steady stretches as they happen, sends each one's mean through the
ONNX graph, keeps the last few residuals and applies the advisory rules.
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.data.prepare import STEADY_LIMITS, STEADY_WINDOW
from src.deployment.export import INPUTS
from src.features.residuals import PARAMETERS
from src.monitoring.diagnosis import ADVICE, LIMITS, ON_DELAY, WINDOW, cause


class Monitor:
    def __init__(self, model: Path = Path("models/chiller.onnx"), library: Path = Path("models/signatures.json")):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        signatures = json.loads(library.read_text())
        self.spread, self.directions = signatures["spread"], signatures["directions"]
        self.recent = deque(maxlen=STEADY_WINDOW)   # samples, for the steadiness test
        self.stretch: list[np.ndarray] = []         # samples of the current steady stretch
        self.residuals = deque(maxlen=WINDOW)       # one per steady point
        self.over = 0
        self.health: dict[str, float] = {}
        self.efficiency: dict[str, float] = {}
        self.advisory: str | None = None

    def step(self, sample: dict[str, float]) -> None:
        """Take one 10-second sample, keyed as in INPUTS (°C, kg/s, kPa absolute, K, kW)."""
        self.recent.append(sample)
        if self._steady():
            # the first steady window counts from its start, as in src/data/prepare.py
            new = self.recent if not self.stretch else [sample]
            self.stretch += [np.array([s[name] for name in INPUTS], dtype=np.float32) for s in new]
        elif self.stretch:
            if len(self.stretch) >= STEADY_WINDOW:
                self._point(np.mean(self.stretch, axis=0))
            self.stretch = []

    def _steady(self) -> bool:
        if len(self.recent) < STEADY_WINDOW or self.recent[-1]["power"] <= 20:
            return False
        frame = pd.DataFrame(self.recent)
        return all(frame[name].std() < limit for name, limit in STEADY_LIMITS.items())

    def _point(self, readings: np.ndarray) -> None:
        residuals, states = self.session.run(None, {"readings": readings[None]})
        self.efficiency = dict(zip(("load", "cop", "second_law"), states[0].astype(float)))
        self.residuals.append(residuals[0].astype(float))
        if len(self.residuals) < WINDOW:
            return
        self.health = dict(zip(PARAMETERS, np.median(self.residuals, axis=0)))
        crossed = any(abs(self.health[name]) > limit for name, limit in LIMITS.items())
        self.over = self.over + 1 if crossed else 0
        self.advisory = ADVICE[cause(pd.Series(self.health), self.spread, self.directions)] if self.over >= ON_DELAY else None
