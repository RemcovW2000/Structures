import numpy as np
import pytest

from structures.panel.data.lamina_props import Christos
from structures.panel.data_utils import PanelLoads
from structures.panel.utils import laminate_builder


@pytest.mark.parametrize(
    "angles_list, expected_ABD",
    [
        (
            [0, 0, 0, 0],
            np.array(
                [
                    [114412.0, 2707.22, 0.0, 0.0, 0.0, 0.0],
                    [2707.22, 9024.06, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4000.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 6101.98, 144.385, 0.0],
                    [0.0, 0.0, 0.0, 144.385, 481.283, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 213.333],
                ],
                dtype=float,
            ),
        ),
        (
            [0, 45, 45, 0],
            np.array(
                [
                    [75312.4, 15459.9, 13173.5, 0.0, 0.0, 0.0],
                    [15459.9, 22618.4, 13173.5, 0.0, 0.0, 0.0],
                    [13173.5, 13173.5, 16752.7, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 5580.65, 314.421, 175.647],
                    [0.0, 0.0, 0.0, 314.421, 662.54, 175.647],
                    [0.0, 0.0, 0.0, 175.647, 175.647, 383.37],
                ],
                dtype=float,
            ),
        ),
        (
            [0, 30, 60, 90],
            np.array(
                [
                    [52153.6, 12271.8, 11408.6, 9221.46, 0.0, 552.209],
                    [12271.8, 52153.6, 11408.6, 0.0, -9221.46, -552.209],
                    [11408.6, 11408.6, 13564.5, 552.209, -552.209, 0.0],
                    [9221.46, 0.0, 552.209, 3164.11, 271.912, 152.115],
                    [0.0, -9221.46, -552.209, 271.912, 3164.11, 152.115],
                    [552.209, -552.209, 0.0, 152.115, 152.115, 340.861],
                ],
                dtype=float,
            ),
        ),
    ],
)
def test_ABD(angles_list: list[float], expected_ABD: np.ndarray) -> None:
    laminate = laminate_builder(angles_list, False, False, 1)
    np.testing.assert_allclose(laminate.ABD_matrix, expected_ABD, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "angles_list, expected_engineering_properties",
    [
        ([0, 90, 45, -45], [55105.2, 55105.2, 20940.9, 0.3157]),
        ([10, 20, 30, 40], [45340.3, 13494.2, 9395.95, 0.2275]),
    ],
)
def test_engineering_properties(
    angles_list: list[float], expected_engineering_properties: list[float]
) -> None:
    laminate = laminate_builder(angles_list, True, True, 1, material_props=Christos)
    assert laminate.Ex == pytest.approx(expected_engineering_properties[0], rel=1e-2)
    assert laminate.Ey == pytest.approx(expected_engineering_properties[1], rel=1e-2)
    assert laminate.Gxy == pytest.approx(expected_engineering_properties[2], rel=1e-2)
    assert laminate.vxy == pytest.approx(expected_engineering_properties[3], rel=1e-2)


@pytest.mark.parametrize(
    "angles_list, expected_strength",
    [
        ([0, 0, 0, 0], Christos.failure_properties.R11t),
        ([0, 0, 0, 0], -Christos.failure_properties.R11c),
        ([90, 90, 90, 90], Christos.failure_properties.Yt),
        ([90, 90, 90, 90], -Christos.failure_properties.Yc),
    ],
)
def test_failure(angles_list: list[float], expected_strength: float) -> None:
    """
    Test failure indicator calculations.

    For UD in x direction, the strength should be R11t (tensile)
    Same for transverse, etc.
    """
    laminate = laminate_builder([0, 0, 0, 0, 0], False, True, 1, material_props=Christos)
    laminate.loads = PanelLoads(np.array([Christos.failure_properties.R11t, 0, 0, 0, 0, 0]))
    assert laminate.fi == pytest.approx(1.0, rel=1e-5)

    # Test shear:
    # also tests that failure is recalculated when loads are changed.
    laminate.loads = PanelLoads(np.array([0, 0, Christos.failure_properties.S * 2, 0, 0, 0]))
    assert laminate.fi == pytest.approx(2, rel=1e-5)


def test_fi_dict() -> None:
    laminate = laminate_builder([0, 90, 45, -45], False, True, 1, material_props=Christos)
    laminate.loads = PanelLoads(np.array([Christos.failure_properties.R11t / 2, 0, 0, 0, 0, 0]))
    fi = laminate.fi
    assert fi > 0
    fi_dict = laminate.failure_indicators
    vals = list(fi_dict.values())
    assert all(isinstance(v, float) for v in vals)
    assert all(v >= 0 for v in vals)
    assert len(vals) == 2
