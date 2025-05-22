"""Configuration file for tests for transpiler from circuit to MBQC patterns via J-∧z decomposition.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

import pytest
from numpy.random import PCG64, Generator

SEED = 25


@pytest.fixture
def fx_bg() -> PCG64:
    """Fixture for bit generator.

    Returns
    -------
        Bit generator

    """
    return PCG64(SEED)


@pytest.fixture
def fx_rng(fx_bg: PCG64) -> Generator:
    """Fixture for generator.

    Returns
    -------
        Generator

    """
    return Generator(fx_bg)
