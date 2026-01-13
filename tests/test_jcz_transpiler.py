"""Tests for transpiler from circuit to MBQC patterns via J-∧z decomposition.

Copyright (C) 2026, QAT team (ENS-PSL, Inria, CNRS).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from graphix import instruction
from graphix.fundamentals import ANGLE_PI, Axis
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import Statevec
from graphix.simulator import DefaultMeasureMethod
from graphix.transpiler import Circuit
from numpy.random import PCG64, Generator

from graphix_jcz_transpiler import circuit_to_causal_flow, transpile_jcz, transpile_jcz_open_graph

logger = logging.getLogger(__name__)

TEST_BASIC_CIRCUITS = [
    Circuit(1, instr=[instruction.H(0)]),
    Circuit(1, instr=[instruction.S(0)]),
    Circuit(1, instr=[instruction.X(0)]),
    Circuit(1, instr=[instruction.Y(0)]),
    Circuit(1, instr=[instruction.Z(0)]),
    Circuit(1, instr=[instruction.I(0)]),
    Circuit(1, instr=[instruction.RX(0, ANGLE_PI / 4)]),
    Circuit(1, instr=[instruction.RY(0, ANGLE_PI / 4)]),
    Circuit(1, instr=[instruction.RZ(0, ANGLE_PI / 4)]),
    Circuit(2, instr=[instruction.CZ((0, 1))]),
    Circuit(2, instr=[instruction.CNOT(0, 1)]),
    Circuit(3, instr=[instruction.CCX(0, (1, 2))]),
    Circuit(2, instr=[instruction.RZZ(0, 1, ANGLE_PI / 4)]),
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
    og = pattern.extract_opengraph()
    f = og.find_causal_flow()
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


@pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
def test_measure(fx_rng: Generator, axis: Axis) -> None:
    """Test circuit transpilation with measurement.

    Circuits transpiled in JCZ give patterns with causal flow.
    This test checks manual measurements work for the `transpile_jcz` function.
    It also checks that measurements have uniform outcomes.
    """
    circuit = Circuit(2)
    circuit.h(1)
    circuit.cnot(0, 1)
    circuit.m(0, axis)
    transpiled = transpile_jcz(circuit)
    transpiled.pattern.remove_input_nodes()
    # Don't call perform_pauli_measurements() - it would make measurements deterministic
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
    pattern.minimize_space()
    state = circuit.simulate_statevector().statevec
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_flow_og(circuit: Circuit) -> None:
    """Test transpiled circuits have flow."""
    f = circuit_to_causal_flow(circuit)
    jcz = transpile_jcz_open_graph(circuit)
    og = jcz.pattern.extract_opengraph()
    fprime = og.find_causal_flow()
    assert f is not None
    assert fprime is not None


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_og_generation(circuit: Circuit) -> None:
    """Test that open graphs are extracted in the expected way."""
    pattern = transpile_jcz(circuit).pattern
    og = pattern.extract_opengraph()
    causal_flow_jcz = circuit_to_causal_flow(circuit)
    og_jcz = causal_flow_jcz.og
    assert og.isclose(og_jcz)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation_compare(circuit: Circuit, fx_rng: Generator) -> None:
    """Test circuit transpilation comparing state vector back-end."""
    pattern = transpile_jcz(circuit).pattern
    pattern_og = transpile_jcz_open_graph(circuit).pattern
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    pattern.minimize_space()
    pattern_og.remove_input_nodes()
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
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=True)
    if check == "simulation":
        test_circuit_simulation_og(circuit, rng)
    elif check == "flow":
        test_circuit_flow_og(circuit)
