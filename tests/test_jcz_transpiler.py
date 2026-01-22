"""Tests for transpiler from circuit to MBQC patterns via J-∧z decomposition.

Copyright (C) 2026, QAT team (ENS-PSL, Inria, CNRS).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from graphix import Pattern, command, instruction
from graphix.fundamentals import ANGLE_PI, Axis, Plane
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import Statevec
from graphix.simulator import DefaultMeasureMethod
from graphix.transpiler import Circuit
from numpy.random import PCG64, Generator

from graphix_jcz_transpiler import CircuitWithMeasurementError, circuit_to_causal_flow, transpile_jcz, transpile_jcz_cf

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


def check_pattern_equality(pattern1: Pattern, pattern2: Pattern) -> None:
    """Check if two patterns are equal (ignoring order of operations for N and E)."""
    n_set_1: list[int] = []
    e_set_1: list[tuple[int, int]] = []
    m_list_1: list[list[Plane | float | set[int]]] = []
    c_list_1: list[list[int | set[int]]] = []
    n_set_2: list[int] = []
    e_set_2: list[tuple[int, int]] = []
    m_list_2: list[list[Plane | float | set[int]]] = []
    c_list_2: list[list[int | set[int]]] = []
    for elt in pattern1:
        if isinstance(elt, command.N):
            n_set_1.append(elt.node)
        elif isinstance(elt, command.E):
            e_set_1.append(elt.nodes)
        elif isinstance(elt, command.M):
            m_list_1.append([elt.plane, float(elt.angle), elt.s_domain, elt.t_domain])
        else:
            c_list_1.append([elt.node, elt.domain])
    for elt in pattern2:
        if isinstance(elt, command.N):
            n_set_2.append(elt.node)
        elif isinstance(elt, command.E):
            e_set_2.append(elt.nodes)
        elif isinstance(elt, command.M):
            m_list_2.append([elt.plane, float(elt.angle), elt.s_domain, elt.t_domain])
        else:
            c_list_2.append([elt.node, elt.domain])
    assert sorted(n_set_1) == sorted(n_set_2)
    assert sorted(e_set_1) == sorted(e_set_2)
    assert m_list_1 == m_list_2
    assert c_list_1 == c_list_2


def test_fails_with_measure() -> None:
    """Test causal flow circuit transpilation fails with measurement."""
    circuit = Circuit(2)
    circuit.h(1)
    circuit.cnot(0, 1)
    circuit.m(0, Axis.X)
    with pytest.raises(CircuitWithMeasurementError):
        transpile_jcz_cf(circuit)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation(circuit: Circuit, fx_rng: Generator) -> None:
    """Test circuit transpilation simulation compared to direct simulation of the circuit."""
    pattern = transpile_jcz(circuit).pattern
    state = circuit.simulate_statevector().statevec
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_check_circuit_flow(circuit: Circuit) -> None:
    """Test transpiled circuits have flow."""
    pattern = transpile_jcz(circuit).pattern
    og = pattern.extract_opengraph()
    f = og.find_causal_flow()
    assert f is not None


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("check", ["simulation", "flow"])
def test_random_circuit(fx_bg: PCG64, jumps: int, check: str) -> None:
    """Test direct transpilation of random circuit."""
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 6
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=True)
    if check == "simulation":
        test_circuit_simulation(circuit, rng)
    elif check == "flow":
        test_check_circuit_flow(circuit)


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
def test_check_circuit_flow_cf(circuit: Circuit) -> None:
    """Test transpiled circuits have flow."""
    f = circuit_to_causal_flow(circuit)
    assert f is not None


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation_cf(circuit: Circuit, fx_rng: Generator) -> None:
    """Test causal flow transpilation simulation compared to direct simulation of the circuit."""
    pattern = transpile_jcz_cf(circuit).pattern
    pattern.minimize_space()
    state = circuit.simulate_statevector().statevec
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_flow_cf(circuit: Circuit) -> None:
    """Test causal flow transpiled circuits have flow and that they match existing Graphix and direct JCZ transpilation flows."""
    f_cf = circuit_to_causal_flow(circuit)
    f_dir = transpile_jcz(circuit).pattern.extract_causal_flow()
    f_gpx = circuit.transpile().pattern.extract_causal_flow()
    assert f_cf is not None
    assert f_dir is not None
    assert f_gpx is not None
    assert f_cf.og.isclose(f_dir.og)
    # assert f_cf.og.isclose(f_gpx.og)  # This fails due to current Graphix optimisations
    assert f_cf.correction_function == f_dir.correction_function
    # assert f_cf.correction_function == f_gpx.correction_function  # This fails due to current Graphix optimisations
    assert f_cf.partial_order_layers == f_dir.partial_order_layers
    # assert f_cf.partial_order_layers == f_gpx.partial_order_layers  # This fails due to current Graphix optimisations


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_pattern_cf(circuit: Circuit) -> None:
    """Test causal flow transpiled circuits have flow and that they match existing Graphix and direct JCZ transpilation flows."""
    pattern_cf = transpile_jcz_cf(circuit).pattern
    pattern_dir = transpile_jcz(circuit).pattern
    pattern_gpx = circuit.transpile().pattern
    pattern_cf.minimize_space()
    pattern_dir.minimize_space()
    pattern_gpx.minimize_space()
    pattern_cf.standardize()
    pattern_dir.standardize()
    pattern_gpx.standardize()
    pattern_cf.check_runnability()
    pattern_dir.check_runnability()
    check_pattern_equality(
        pattern_cf, pattern_dir
    )  # Doesn't check order, but functionally different order would be captured by other tests
    assert pattern_cf == pattern_dir
    # assert pattern_cf == pattern_gpx  # This fails due to current Graphix optimisations


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_og_generation(circuit: Circuit) -> None:
    """Test that open graphs for direct and causal flow transpilation are equal."""
    og = transpile_jcz(circuit).pattern.extract_opengraph()
    og_jcz = circuit_to_causal_flow(circuit).og
    assert og.isclose(og_jcz)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation_compare_direct(circuit: Circuit, fx_rng: Generator) -> None:
    """Test comparing direct to causal flow transpilation."""
    pattern = transpile_jcz(circuit).pattern
    pattern_cf = transpile_jcz_cf(circuit).pattern
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    pattern.minimize_space()
    pattern_cf.remove_input_nodes()
    pattern_cf.perform_pauli_measurements()
    pattern_cf.minimize_space()
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    state_mbqc_cf = pattern_cf.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state_mbqc_cf.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation_compare_graphix(circuit: Circuit, fx_rng: Generator) -> None:
    """Test comparing Graphix' current transpilation to causal flow transpilation."""
    pattern = circuit.transpile().pattern
    pattern_cf = transpile_jcz_cf(circuit).pattern
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    pattern.minimize_space()
    pattern_cf.remove_input_nodes()
    pattern_cf.perform_pauli_measurements()
    pattern_cf.minimize_space()
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    state_mbqc_cf = pattern_cf.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state_mbqc_cf.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("check", ["simulation", "flow"])
def test_random_circuit_og(fx_bg: PCG64, jumps: int, check: str) -> None:
    """Test random circuit transpilation."""
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 6
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=True)
    if check == "simulation":
        test_circuit_simulation_cf(circuit, rng)
    elif check == "flow":
        test_circuit_flow_cf(circuit)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_random_circuit_compare(fx_bg: PCG64, jumps: int) -> None:
    """Test random circuit transpilation comparing direct and causal flow transpilation."""
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 6
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=True)
    pattern = transpile_jcz(circuit).pattern
    pattern.minimize_space()
    state = pattern.simulate_pattern(rng=rng)
    pattern_og = transpile_jcz_cf(circuit).pattern
    pattern_og.minimize_space()
    state_og = pattern_og.simulate_pattern(rng=rng)
    assert np.abs(np.dot(state.flatten().conjugate(), state_og.flatten())) == pytest.approx(1)
