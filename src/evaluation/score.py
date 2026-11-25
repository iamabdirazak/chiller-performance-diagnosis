"""Score the diagnosis on tests it was not tuned on.

    false advisories   on each healthy test, judged by a baseline fitted without it
    detection          on each fault test, at each severity
    cause              named with signatures learned without the test being judged
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.residuals import compute, table
from src.models.baseline import Baseline
from src.models.train import signatures
from src.monitoring.diagnosis import advise


def main() -> None:
    points = pd.read_parquet("data/steady.parquet")
    baseline = Baseline.load(Path("models/baseline.json"))
    library = json.loads(Path("models/signatures.json").read_text())
    spread = pd.Series(library["spread"])
    healthy = points[points["fault"] == "none"]

    false, hours = 0, 0.0
    for test, rows in healthy.groupby("test"):
        own = Baseline.fit(table(healthy[healthy["test"] != test]), baseline.degree)
        advice = advise(compute(rows, own), library["spread"], library["directions"])
        false += int(advice.notna().any())
        hours += (rows["time"].max() - rows["time"].min()) / 3600

    results = []
    for test, rows in points[points["fault"] != "none"].groupby("test"):
        rest = points[points["test"] != test]
        directions = signatures(rest, baseline, spread)
        advice = advise(compute(rows, baseline), library["spread"], directions).dropna()
        named = advice.mode().iloc[0] if len(advice) else None
        results.append({"test": test, "fault": rows["fault"].iloc[0], "severity": int(rows["severity"].iloc[0]),
                        "detected": bool(len(advice)), "cause": named, "correct": named == rows["fault"].iloc[0]})
    results = pd.DataFrame(results).sort_values(["fault", "severity"])

    summary = {
        "healthy_tests": int(healthy["test"].nunique()),
        "healthy_hours": round(hours, 1),
        "healthy_tests_with_an_advisory": false,
        "fault_tests": len(results),
        "detected": int(results["detected"].sum()),
        "cause_correct_when_detected": int(results["correct"].sum()),
        "tests": results.to_dict(orient="records"),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/results.json").write_text(json.dumps(summary, indent=2))
    print(results.to_string(index=False))
    print({k: v for k, v in summary.items() if k != "tests"})


if __name__ == "__main__":
    main()
