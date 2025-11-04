"""Graphix Transpiler from circuit to MBQC patterns via J-∧z decomposition.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass
from enum import Enum
from math import pi
from typing import TYPE_CHECKING, ClassVar, Literal

import networkx as nx
from graphix import Pattern, command, instruction
from graphix.fundamentals import Plane
from graphix.generator import _gflow2pattern, _pflow2pattern  # noqa: PLC2701
from graphix.gflow import find_gflow, find_pauliflow
from graphix.instruction import InstructionKind
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.transpiler import Circuit, TranspileResult
from typing_extensions import TypeAlias, assert_never

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix.parameter import ExpressionOrFloat


class JCZInstructionKind(Enum):
    """Tag for instruction kind."""

    CZ = enum.auto()
    J = enum.auto()


@dataclass
class CZ:
    """CZ circuit instruction."""

    targets: tuple[int, int]
    kind: ClassVar[Literal[JCZInstructionKind.CZ]] = dataclasses.field(
        default=JCZInstructionKind.CZ,
        init=False,
    )


@dataclass
class J:
    """J circuit instruction."""

    target: int
    angle: ExpressionOrFloat
    kind: ClassVar[Literal[JCZInstructionKind.J]] = dataclasses.field(
        default=JCZInstructionKind.J,
        init=False,
    )


JCZInstruction: TypeAlias = (
    instruction.CCX
    | instruction.RZZ
    | instruction.CNOT
    | instruction.SWAP
    | instruction.H
    | instruction.S
    | instruction.X
    | instruction.Y
    | instruction.Z
    | instruction.I
    | instruction.RX
    | instruction.RY
    | instruction.RZ
    | CZ
    | J
)


def decompose_ccx(
    instr: instruction.CCX,
) -> list[instruction.H | instruction.CNOT | instruction.RZ]:
    """Return a decomposition of the CCX gate into H, CNOT, T and T-dagger gates.

    This decomposition of the Toffoli gate can be found in
    Michael A. Nielsen and Isaac L. Chuang,
    Quantum Computation and Quantum Information,
    Cambridge University Press, 2000
    (p. 182 in the 10th Anniversary Edition).

    Args:
    ----
        instr: the CCX instruction to decompose.

    Returns:
    -------
        the decomposition.

    """
    return [
        instruction.H(instr.target),
        instruction.CNOT(control=instr.controls[1], target=instr.target),
        instruction.RZ(instr.target, -pi / 4),
        instruction.CNOT(control=instr.controls[0], target=instr.target),
        instruction.RZ(instr.target, pi / 4),
        instruction.CNOT(control=instr.controls[1], target=instr.target),
        instruction.RZ(instr.target, -pi / 4),
        instruction.CNOT(control=instr.controls[0], target=instr.target),
        instruction.RZ(instr.controls[1], pi / 4),
        instruction.RZ(instr.target, pi / 4),
        instruction.CNOT(control=instr.controls[0], target=instr.controls[1]),
        instruction.H(instr.target),
        instruction.RZ(instr.controls[0], pi / 4),
        instruction.RZ(instr.controls[1], -pi / 4),
        instruction.CNOT(control=instr.controls[0], target=instr.controls[1]),
    ]


def decompose_rzz(instr: instruction.RZZ) -> list[instruction.CNOT | instruction.RZ]:
    """Return a decomposition of RZZ(α) gate as CNOT(control, target)·Rz(target, α)·CNOT(control, target).

    Args:
    ----
        instr: the RZZ instruction to decompose.

    Returns:
    -------
        the decomposition.

    """
    return [
        instruction.CNOT(control=instr.control, target=instr.target),
        instruction.RZ(instr.target, instr.angle),
        instruction.CNOT(control=instr.control, target=instr.target),
    ]


def decompose_cnot(instr: instruction.CNOT) -> list[instruction.H | CZ]:
    """Return a decomposition of the CNOT gate as H·∧z·H.

    Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.

    Args:
    ----
        instr: the CNOT instruction to decompose.

    Returns:
    -------
        the decomposition.

    """
    return [
        instruction.H(instr.target),
        CZ(targets=(instr.control, instr.target)),
        instruction.H(instr.target),
    ]


def decompose_swap(instr: instruction.SWAP) -> list[instruction.CNOT]:
    """Return a decomposition of the SWAP gate as CNOT(0, 1)·CNOT(1, 0)·CNOT(0, 1).

    Michael A. Nielsen and Isaac L. Chuang,
    Quantum Computation and Quantum Information,
    Cambridge University Press, 2000
    (p. 23 in the 10th Anniversary Edition).

    Args:
    ----
        instr: the SWAP instruction to decompose.

    Returns:
    -------
        the decomposition.

    """
    return [
        instruction.CNOT(control=instr.targets[0], target=instr.targets[1]),
        instruction.CNOT(control=instr.targets[1], target=instr.targets[0]),
        instruction.CNOT(control=instr.targets[0], target=instr.targets[1]),
    ]


def decompose_y(instr: instruction.Y) -> Iterable[instruction.X | instruction.Z]:
    """Return a decomposition of the Y gate as X·Z.

    Args:
    ----
        instr: the Y instruction to decompose.

    Returns:
    -------
        the decomposition.

    """
    return reversed([instruction.X(instr.target), instruction.Z(instr.target)])


def decompose_rx(instr: instruction.RX) -> list[J]:
    """Return a J decomposition of the RX gate.

    The Rx(α) gate is decomposed into J(α)·H (that is to say, J(α)·J(0)).
    Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.

    Args:
    ----
        instr: the RX instruction to decompose.

    Returns:
    -------
        the decomposition.

    """
    return [J(target=instr.target, angle=angle) for angle in reversed((instr.angle, 0))]


def decompose_ry(instr: instruction.RY) -> list[J]:
    """Return a J decomposition of the RY gate.

    The Ry(α) gate is decomposed into J(0)·J(π/2)·J(α)·J(-π/2).
    Vincent Danos, Elham Kashefi, Prakash Panangaden, Robust and parsimonious realisations of unitaries in the one-way
    model, 2004.

    Args:
    ----
        instr: the RY instruction to decompose.

    Returns:
    -------
        the decomposition.

    """
    return [J(target=instr.target, angle=angle) for angle in reversed((0, pi / 2, instr.angle, -pi / 2))]


def decompose_rz(instr: instruction.RZ) -> Iterable[J]:
    """Return a J decomposition of the RZ gate.

    The Rz(α) gate is decomposed into H·J(α) (that is to say, J(0)·J(α)).
    Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.

    Args:
    ----
        instr: the RZ instruction to decompose.

    Returns:
    -------
        the decomposition.

    """
    return [J(target=instr.target, angle=angle) for angle in reversed((0, instr.angle))]


def instruction_to_jcz(instr: JCZInstruction) -> Iterable[J | CZ]:
    """Return a J-∧z decomposition of the instruction.

    Args:
    ----
        instr: the instruction to decompose.

    Returns:
    -------
        the decomposition.

    """
    # Use == for mypy
    if instr.kind == JCZInstructionKind.J or instr.kind == JCZInstructionKind.CZ:  # noqa: PLR1714
        return [instr]
    if instr.kind == InstructionKind.I:
        return []
    if instr.kind == InstructionKind.H:
        return [J(instr.target, 0)]
    if instr.kind == InstructionKind.S:
        return instruction_to_jcz(instruction.RZ(instr.target, pi / 2))
    if instr.kind == InstructionKind.X:
        return instruction_to_jcz(instruction.RX(instr.target, pi))
    if instr.kind == InstructionKind.Y:
        return instruction_list_to_jcz(decompose_y(instr))
    if instr.kind == InstructionKind.Z:
        return instruction_to_jcz(instruction.RZ(instr.target, pi))
    if instr.kind == InstructionKind.RX:
        return decompose_rx(instr)
    if instr.kind == InstructionKind.RY:
        return decompose_ry(instr)
    if instr.kind == InstructionKind.RZ:
        return decompose_rz(instr)
    if instr.kind == InstructionKind.CCX:
        return instruction_list_to_jcz(decompose_ccx(instr))
    if instr.kind == InstructionKind.RZZ:
        return instruction_list_to_jcz(decompose_rzz(instr))
    if instr.kind == InstructionKind.CNOT:
        return instruction_list_to_jcz(decompose_cnot(instr))
    if instr.kind == InstructionKind.SWAP:
        return instruction_list_to_jcz(decompose_swap(instr))
    assert_never(instr.kind)


def instruction_list_to_jcz(instrs: Iterable[JCZInstruction]) -> list[J | CZ]:
    """Return a J-∧z decomposition of the sequence of instructions.

    Args:
    ----
        instrs: the instruction sequence to decompose.

    Returns:
    -------
        the decomposition.

    """
    return [jcz_instr for instr in instrs for jcz_instr in instruction_to_jcz(instr)]


class IllformedPatternError(Exception):
    """Raised if the circuit is ill-formed."""

    def __init__(self) -> None:
        """Build the exception."""
        super().__init__("Ill-formed pattern")


class InternalInstructionError(Exception):
    """Raised if the circuit contains internal _XC or _ZC instructions."""

    def __init__(self, instr: instruction.Instruction) -> None:
        """Build the exception."""
        super().__init__(f"Internal instruction: {instr}")


def transpile_jcz(circuit: Circuit) -> TranspileResult:
    """Transpile a circuit via a J-∧z decomposition.

    Args:
    ----
        circuit: the circuit to transpile.

    Returns:
    -------
        the result of the transpilation: a pattern and indices for measures.

    Raises:
    ------
        IllformedPatternError: if the pattern is ill-formed.
        InternalInstructionError: if the circuit contains internal _XC or _ZC instructions.

    """
    indices: list[int | None] = list(range(circuit.width))
    n_nodes = circuit.width
    pattern = Pattern(input_nodes=list(range(circuit.width)))
    classical_outputs = []
    for instr in circuit.instruction:
        if instr.kind == InstructionKind.M:
            pattern.add(command.M(instr.target, instr.plane, instr.angle / pi))
            classical_outputs.append(instr.target)
            indices[instr.target] = None
            continue
        # Use == for mypy
        if instr.kind == InstructionKind._XC or instr.kind == InstructionKind._ZC:  # noqa: PLR1714, SLF001
            raise InternalInstructionError(instr)
        for instr_jcz in instruction_to_jcz(instr):
            if instr_jcz.kind == JCZInstructionKind.J:
                target = indices[instr_jcz.target]
                if target is None:
                    raise IllformedPatternError
                ancilla = n_nodes
                n_nodes += 1
                pattern.extend(j_commands(target, ancilla, -instr_jcz.angle))
                indices[instr_jcz.target] = ancilla
                continue
            if instr_jcz.kind == JCZInstructionKind.CZ:
                t0, t1 = instr_jcz.targets
                i0, i1 = indices[t0], indices[t1]
                if i0 is None or i1 is None:
                    raise IllformedPatternError
                pattern.extend([command.E(nodes=(i0, i1))])
                continue
            assert_never(instr_jcz.kind)
    pattern.reorder_output_nodes([i for i in indices if i is not None])
    return TranspileResult(pattern, tuple(classical_outputs))


def j_commands(current_node: int, next_node: int, angle: ExpressionOrFloat) -> list[command.Command]:
    """Return the MBQC pattern commands for a J gate.

    Args:
    ----
        current_node: the current node.
        next_node: the next node.
        angle: the angle of the J gate.
        domain: the domain the X correction is based on.

    Returns:
    -------
        the MBQC pattern commands for a J gate as a list

    """
    return [
            command.N(node=next_node),
            command.E(nodes=(current_node, next_node)),
            command.M(node=current_node, angle=angle / pi),
            command.X(node=next_node, domain={current_node}),
            ]


def circuit_to_open_graph(circuit: Circuit) -> OpenGraph:
    """Transpile a circuit via a J-∧z-like decomposition to an open graph.

    Args:
    ----
        circuit: the circuit to transpile.

    Returns:
    -------
        the result of the transpilation: an open graph.

    Raises:
    ------
        IllformedPatternError: if the pattern is ill-formed (operation on already measured node)
        InternalInstructionError: if the circuit contains internal _XC or _ZC instructions.

    """
    indices: list[int | None] = list(range(circuit.width))
    n_nodes = circuit.width
    measurements: dict[int, Measurement] = {}
    inputs = list(range(n_nodes))
    inside = nx.Graph()  # type: ignore[attr-defined]
    inside.add_nodes_from(inputs)
    for instr in circuit.instruction:
        if instr.kind == InstructionKind.M:
            measurements[instr.target] = Measurement(instr.angle / pi, instr.plane)
            indices[instr.target] = None
            continue
        # Use == for mypy
        if instr.kind == InstructionKind._XC or instr.kind == InstructionKind._ZC:  # noqa: PLR1714, SLF001
            raise InternalInstructionError(instr)
        for instr_jcz in instruction_to_jcz(instr):
            if instr_jcz.kind == JCZInstructionKind.J:
                target = indices[instr_jcz.target]
                if target is None:
                    raise IllformedPatternError
                ancilla = n_nodes
                n_nodes += 1
                inside.add_node(ancilla)
                inside.add_edge(target, ancilla)
                measurements[target] = Measurement(-instr_jcz.angle / pi, plane=Plane.XY)
                indices[instr_jcz.target] = ancilla
                continue
            if instr_jcz.kind == JCZInstructionKind.CZ:
                t0, t1 = instr_jcz.targets
                i0, i1 = indices[t0], indices[t1]
                if i0 is None or i1 is None:
                    raise IllformedPatternError
                inside.add_edge(i0, i1)
                continue
            assert_never(instr_jcz.kind)
    outputs = sorted(set(inside.nodes) - set(measurements.keys()))
    return OpenGraph(inside, measurements, inputs, outputs)


# def open_graph_to_pattern_no_flow(og: OpenGraph) -> Pattern:
#     """Generate a pattern from an open graph using gflow- and pflow-finding algorithms.

#     Parameters
#     ----------
#     og: the open graph to generate the pattern from.

#     Returns
#     -------
#     the pattern generated from the open graph.

#     Raises
#     ------
#     ValueError: if the open graph does not have flow, gflow, or Pauli flow

#     """
#     g = og.inside.copy()
#     inputs = og.inputs
#     outputs = og.outputs
#     meas = og.measurements
#     angles = {node: m.angle for node, m in meas.items()}
#     planes = {node: m.plane for node, m in meas.items()}
#     inputs_set = set(og.inputs)
#     outputs_set = set(og.outputs)
#     measuring_nodes = set(g.nodes) - outputs_set
#     meas_planes = dict.fromkeys(measuring_nodes, Plane.XY) if not planes else dict(planes)
#     # gflow first
#     gflow, l_k = find_gflow(g, inputs_set, outputs_set, meas_planes=meas_planes)
#     if gflow is not None:
#         # gflow found
#         pattern = _gflow2pattern(g, angles, inputs, meas_planes, gflow, l_k)
#         pattern.reorder_output_nodes(outputs)
#         return pattern
#     # then pflow
#     pflow, l_k = find_pauliflow(g, inputs_set, outputs_set, meas_planes=meas_planes, meas_angles=angles)
#     if pflow is not None:
#         # pflow found
#         pattern = _pflow2pattern(g, angles, inputs, meas_planes, pflow, l_k)
#         pattern.reorder_output_nodes(outputs)
#         return pattern
#     msg = "no flow or gflow or pflow found"
#     raise ValueError(msg)


def transpile_jcz_open_graph(circuit: Circuit) -> Pattern:
    """Transpile a circuit via a J-∧z-like decomposition to a pattern.

    Currently fails due to overuse of memory in conversion from open graph to pattern, assumed in the causal flow step.

    Args:
    ----
        circuit: the circuit to transpile.

    Returns:
    -------
        the result of the transpilation: a pattern.

    """
    return circuit_to_open_graph(circuit).to_pattern()
    # return open_graph_to_pattern_no_flow(circuit_to_open_graph(circuit))
