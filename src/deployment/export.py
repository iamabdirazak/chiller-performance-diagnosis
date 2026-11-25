"""Write the per-point path as one ONNX graph: properties, cycle, learned baseline, residuals.

The graph takes one steady operating point, eleven readings, and returns the
five residuals and the cycle's load, COP and second-law efficiency. Finding
steady points, averaging the last few and applying the limits stay in the
host loop: they are state, not computation.

Every operation is arithmetic, so the graph is written directly with ONNX
operators through a small expression builder, as in the pump project.
"""

from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.features.residuals import CONDITIONS, PARAMETERS
from src.models.baseline import Baseline
from src.plant.cycle import CP_WATER
from src.plant.saturation import coefficients

INPUTS = ("evaporator_in", "evaporator_out", "condenser_in", "condenser_out", "evaporator_flow", "condenser_flow",
          "evaporator_pressure", "condenser_pressure", "subcooling", "oil_temperature", "power")
STATES = ("load", "cop", "second_law")


class Graph:
    def __init__(self):
        self.nodes, self.initializers, self.count = [], [], 0

    def name(self, prefix: str) -> str:
        self.count += 1
        return f"{prefix}_{self.count}"

    def constant(self, value) -> "Expression":
        name = self.name("c")
        self.initializers.append(numpy_helper.from_array(np.array(value, dtype=np.float32), name))
        return Expression(self, name)

    def op(self, kind: str, *inputs, **attributes) -> "Expression":
        output = self.name(kind.lower())
        self.nodes.append(helper.make_node(kind, [i.output for i in inputs], [output], **attributes))
        return Expression(self, output)


class Expression:
    """A value in the graph. Arithmetic on it adds ONNX nodes."""

    def __init__(self, graph: Graph, output: str):
        self.graph, self.output = graph, output

    def wrap(self, other):
        return other if isinstance(other, Expression) else self.graph.constant(other)

    def __add__(self, other): return self.graph.op("Add", self, self.wrap(other))
    def __radd__(self, other): return self.graph.op("Add", self.wrap(other), self)
    def __sub__(self, other): return self.graph.op("Sub", self, self.wrap(other))
    def __rsub__(self, other): return self.graph.op("Sub", self.wrap(other), self)
    def __mul__(self, other): return self.graph.op("Mul", self, self.wrap(other))
    def __rmul__(self, other): return self.graph.op("Mul", self.wrap(other), self)
    def __truediv__(self, other): return self.graph.op("Div", self, self.wrap(other))
    def __rtruediv__(self, other): return self.graph.op("Div", self.wrap(other), self)
    def __rpow__(self, base): return self.graph.op("Pow", self.wrap(base), self)

    def __pow__(self, exponent: int):
        result = self
        for _ in range(exponent - 1):
            result = result * self
        return result

    def at_least(self, floor: float): return self.graph.op("Max", self, self.wrap(floor))
    def log(self): return self.graph.op("Log", self)


def build(baseline: Baseline) -> onnx.ModelProto:
    graph = Graph()
    readings = Expression(graph, "readings")
    x = {name: graph.op("Gather", readings, _index(graph, i), axis=1) for i, name in enumerate(INPUTS)}

    # R-134a saturation, as in src/plant/saturation.py
    a, b, c = coefficients()
    t_evap = 1 / (a - b * x["evaporator_pressure"].log()) - c
    t_cond = 1 / (a - b * x["condenser_pressure"].log()) - c

    # The cycle, as in src/plant/cycle.py
    load = x["evaporator_flow"] * CP_WATER * (x["evaporator_in"] - x["evaporator_out"])
    cop = load / x["power"]
    second_law = cop * (t_cond - t_evap) / (t_evap + 273.15)
    parameters = {
        "liquid_approach": t_cond - x["subcooling"] - x["condenser_out"],
        "subcooling": x["subcooling"],
        "oil_temperature": x["oil_temperature"],
        "condenser_flow": x["condenser_flow"],
        "evaporator_flow": x["evaporator_flow"],
    }

    # The learned healthy chiller, as in src/models/baseline.py
    conditions = {"load": load, "evaporator_out": x["evaporator_out"], "condenser_in": x["condenser_in"]}
    z = [(conditions[name] - centre) / scale for name, centre, scale in zip(CONDITIONS, baseline.centre, baseline.scale)]
    terms = [z[0] * 0 + 1]
    for order in range(1, baseline.degree + 1):
        for combination in combinations_with_replacement(range(len(CONDITIONS)), order):
            term = z[combination[0]]
            for i in combination[1:]:
                term = term * z[i]
            terms.append(term)
    basis = _stack(graph, terms, graph.name("basis"))
    weights = graph.constant(np.array(baseline.weights, dtype=np.float32))
    expected = graph.op("MatMul", basis, weights)

    measured = _stack(graph, [parameters[name] for name in PARAMETERS], graph.name("measured"))
    graph.nodes.append(helper.make_node("Sub", [measured.output, expected.output], ["residuals"]))
    _stack(graph, [load, cop, second_law], "states")

    model = helper.make_model(helper.make_graph(
        graph.nodes, "chiller_performance_diagnosis",
        [helper.make_tensor_value_info("readings", TensorProto.FLOAT, ["points", len(INPUTS)])],
        [helper.make_tensor_value_info("residuals", TensorProto.FLOAT, ["points", len(PARAMETERS)]),
         helper.make_tensor_value_info("states", TensorProto.FLOAT, ["points", len(STATES)])],
        graph.initializers,
    ), opset_imports=[helper.make_opsetid("", 17)], producer_name="chiller-performance-diagnosis", ir_version=8)
    model.doc_string = "Eleven readings per steady point in; residuals against the healthy chiller, and its load, COP and second-law efficiency out."
    onnx.checker.check_model(model)
    return model


def _stack(graph: Graph, values, name: str) -> Expression:
    columns = [graph.op("Unsqueeze", v, _axes(graph)) for v in values]
    graph.nodes.append(helper.make_node("Concat", [column.output for column in columns], [name], axis=1))
    return Expression(graph, name)


def _index(graph: Graph, i: int) -> Expression:
    name = graph.name("index")
    graph.initializers.append(numpy_helper.from_array(np.array(i, dtype=np.int64), name))
    return Expression(graph, name)


def _axes(graph: Graph) -> Expression:
    name = graph.name("axes")
    graph.initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name))
    return Expression(graph, name)


if __name__ == "__main__":
    model = build(Baseline.load(Path("models/baseline.json")))
    path = Path("models/chiller.onnx")
    onnx.save(model, path)
    print(f"{path}  {path.stat().st_size / 1024:.1f} KB  {len(model.graph.node)} operators")
