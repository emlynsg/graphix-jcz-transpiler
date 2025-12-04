"""Tests for transpiler from circuit to MBQC patterns via J-∧z decomposition.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

from __future__ import annotations

import logging
from math import pi

import numpy as np
import pytest
from graphix import instruction
from graphix.fundamentals import Plane
from graphix.gflow import find_flow
from graphix.opengraph import OpenGraph
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import Statevec
from graphix.simulator import DefaultMeasureMethod
from graphix.transpiler import Circuit
from numpy.random import PCG64, Generator

from graphix_jcz_transpiler import circuit_to_open_graph, transpile_jcz, transpile_jcz_open_graph

logger = logging.getLogger(__name__)

TEST_BASIC_CIRCUITS = [
    Circuit(1, instr=[instruction.H(0)]),
    Circuit(1, instr=[instruction.S(0)]),
    Circuit(1, instr=[instruction.X(0)]),
    Circuit(1, instr=[instruction.Y(0)]),
    Circuit(1, instr=[instruction.Z(0)]),
    Circuit(1, instr=[instruction.I(0)]),
    Circuit(1, instr=[instruction.RX(0, pi / 4)]),
    Circuit(1, instr=[instruction.RY(0, pi / 4)]),
    Circuit(1, instr=[instruction.RZ(0, pi / 4)]),
    Circuit(2, instr=[instruction.CNOT(0, 1)]),
    Circuit(3, instr=[instruction.CCX(0, (1, 2))]),
    Circuit(2, instr=[instruction.RZZ(0, 1, pi / 4)]),
    Circuit(3, instr=[instruction.CNOT(0, 1), instruction.CCX(0, (1, 2)), instruction.RZ(0, pi / 4)])
]


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation(circuit: Circuit, fx_rng: Generator) -> None:
    """Test circuit transpilation comparing state vector back-end."""
    pattern = transpile_jcz(circuit).pattern
    state = circuit.simulate_statevector().statevec
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_flow(circuit: Circuit) -> None:
    """Test transpiled circuits have flow."""
    pattern = transpile_jcz(circuit).pattern
    og = OpenGraph.from_pattern(pattern)
    f, _layers = find_flow(
        og.inside, set(og.inputs), set(og.outputs), {node: meas.plane for node, meas in og.measurements.items()}
    )
    assert f is not None


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("check", ["simulation", "flow"])
def test_random_circuit(fx_bg: PCG64, jumps: int, check: str) -> None:
    """Test random circuit transpilation."""
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 6
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=True)
    if check == "simulation":
        test_circuit_simulation(circuit, rng)
    elif check == "flow":
        test_circuit_flow(circuit)


def test_measure(fx_rng: Generator) -> None:
    """Test circuit transpilation with measurement.

    Circuits transpiled in JCZ give patterns with causal flow. This test checks that measurement outcomes have probability 1/2 to occur (up to statistical fluctuations).
    """
    circuit = Circuit(2)
    circuit.h(1)
    circuit.cnot(0, 1)
    circuit.m(0, Plane.XY, pi / 4)
    transpiled = transpile_jcz(circuit)
    transpiled.pattern.perform_pauli_measurements()
    transpiled.pattern.minimize_space()

    def simulate_and_measure() -> int:
        measure_method = DefaultMeasureMethod(results=transpiled.pattern.results)
        state = transpiled.pattern.simulate_pattern(rng=fx_rng, measure_method=measure_method)
        measured = measure_method.get_measure_result(transpiled.classical_outputs[0])
        assert isinstance(state, Statevec)
        return measured

    nb_shots = 10000
    count = sum(1 for _ in range(nb_shots) if simulate_and_measure())
    assert abs(count - nb_shots / 2) < nb_shots / 20

@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation_og(circuit: Circuit, fx_rng: Generator) -> None:
    """Test circuit transpilation comparing state vector back-end."""
    pattern = transpile_jcz_open_graph(circuit).pattern
    pattern.perform_pauli_measurements()
    pattern.minimize_space()
    state = circuit.simulate_statevector().statevec
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_flow_og(circuit: Circuit) -> None:
    """Test transpiled circuits have flow."""
    pattern = transpile_jcz_open_graph(circuit).pattern
    og = OpenGraph.from_pattern(pattern)
    f, _layers = find_flow(
        og.inside, set(og.inputs), set(og.outputs), {node: meas.plane for node, meas in og.measurements.items()}
    )
    assert f is not None

@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_og_generation(circuit: Circuit) -> None:
    """Test that open graphs are extracted in the expected way."""
    pattern = transpile_jcz(circuit).pattern
    og = OpenGraph.from_pattern(pattern)
    og_jcz = circuit_to_open_graph(circuit)
    assert og.measurements == og_jcz.measurements
    assert og.inputs == og_jcz.inputs
    assert og.outputs == og_jcz.outputs


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation_compare(circuit: Circuit, fx_rng: Generator) -> None:
    """Test circuit transpilation comparing state vector back-end."""
    pattern = transpile_jcz(circuit).pattern
    pattern_og = transpile_jcz_open_graph(circuit).pattern
    pattern.perform_pauli_measurements()
    pattern.minimize_space()
    pattern_og.perform_pauli_measurements()
    pattern_og.minimize_space()
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    state_mbqc_og = pattern_og.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state_mbqc_og.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("check", ["simulation", "flow"])
def test_random_circuit_og(fx_bg: PCG64, jumps: int, check: str) -> None:
    """Test random circuit transpilation."""
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 6
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=False)
    if check == "simulation":
        test_circuit_simulation_og(circuit, rng)
    elif check == "flow":
        test_circuit_flow_og(circuit)
