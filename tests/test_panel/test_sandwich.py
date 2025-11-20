import copy

import numpy as np
import pytest

from structures.composites.data.core_props import ROHACELL31A
from structures.composites.data_utils import PanelLoads
from structures.composites.laminate import Laminate
from structures.composites.sandwich import Core, Sandwich
from structures.composites.utils import laminate_builder


@pytest.fixture
def standard_laminate() -> Laminate:
    return laminate_builder([0, 90, 45, -45], True, True, 1)


@pytest.fixture
def standard_sandwich(standard_laminate: Laminate) -> Sandwich:
    core = Core(2.0, properties=ROHACELL31A)
    bot = copy.deepcopy(standard_laminate)
    top = copy.deepcopy(standard_laminate)
    sandwich = Sandwich(bottom_laminate=bot, top_laminate=top, core=core)
    return sandwich


@pytest.fixture
def sandwich_no_core(standard_laminate: Laminate) -> Sandwich:
    core = Core(0.0, properties=ROHACELL31A)
    bot = copy.deepcopy(standard_laminate)
    top = copy.deepcopy(standard_laminate)
    sandwich = Sandwich(bottom_laminate=bot, top_laminate=top, core=core)
    return sandwich


def test_sandwich_initialization(standard_sandwich: Sandwich) -> None:
    sandwich = standard_sandwich
    assert sandwich.bottom_laminate is not None
    assert sandwich.top_laminate is not None
    assert sandwich.core is not None
    assert sandwich.h == sandwich.bottom_laminate.h + sandwich.top_laminate.h + sandwich.core.h


def test_symmetric_B_matrix_allzeros(standard_sandwich: Sandwich) -> None:
    assert np.allclose(
        standard_sandwich.B_matrix, np.zeros((3, 3)), atol=1e-6
    ), " B matrix is not zero for symmetric sandwich"


def test_sandwich_ABD_matrix_no_core(sandwich_no_core: Sandwich) -> None:
    sandwich = sandwich_no_core

    total_sequence = sandwich.bottom_laminate.laminas + sandwich.top_laminate.laminas
    total_sequence = copy.deepcopy(total_sequence)

    laminate = Laminate(total_sequence)

    assert np.allclose(
        laminate.A_matrix, sandwich.A_matrix, rtol=1e-6
    ), " A matrices are not equal"  # comment
    assert np.allclose(
        laminate.D_matrix, sandwich.D_matrix, atol=1e-8
    ), " D matrices are not equal"  # comment
    assert np.allclose(
        laminate.B_matrix, sandwich.B_matrix, atol=1e-8
    ), " B matrices are not equal"  # comment
    assert np.allclose(
        laminate.ABD_matrix, sandwich.ABD_matrix, atol=1e-8
    ), " ABD matrices are not equal"


def test_sandwich_facesheet_strains(standard_sandwich: Sandwich) -> None:
    sandwich = standard_sandwich

    sandwich.loads = PanelLoads(Mx=100)
    sandwich.assign_facesheet_strains()
    Sx_top = sandwich.top_laminate.strains.epsilon_xo
    Sx_bot = sandwich.bottom_laminate.strains.epsilon_xo
    assert Sx_top > 0, "Top facesheet Sx strain should be positive under positive Mx"
    assert Sx_bot < 0, "Bottom facesheet Sx strain should be negative under positive Mx"
