"""The healthy chiller, learned: each parameter as a function of the three operating conditions.

A polynomial in the standardised conditions, fitted by least squares to the
healthy tests. Its degree is chosen by leaving one healthy test out at a
time, so it is judged on days the fit never saw.
"""

import json
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.residuals import CONDITIONS, PARAMETERS


@dataclass
class Baseline:
    degree: int
    centre: list[float]
    scale: list[float]
    weights: list[list[float]]  # one column per parameter

    def basis(self, conditions: pd.DataFrame) -> np.ndarray:
        z = (conditions[list(CONDITIONS)].to_numpy() - self.centre) / self.scale
        columns = [np.ones(len(z))]
        for order in range(1, self.degree + 1):
            for combination in combinations_with_replacement(range(len(CONDITIONS)), order):
                columns.append(np.prod(z[:, combination], axis=1))
        return np.column_stack(columns)

    @classmethod
    def fit(cls, healthy: pd.DataFrame, degree: int) -> "Baseline":
        conditions = healthy[list(CONDITIONS)]
        model = cls(degree, conditions.mean().tolist(), conditions.std().tolist(), [])
        weights, *_ = np.linalg.lstsq(model.basis(conditions), healthy[list(PARAMETERS)].to_numpy(), rcond=None)
        model.weights = weights.tolist()
        return model

    def predict(self, conditions: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.basis(conditions) @ np.array(self.weights), columns=list(PARAMETERS), index=conditions.index)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Baseline":
        return cls(**json.loads(path.read_text()))
