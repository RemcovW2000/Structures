from structures.composites.data.lamina_props import DEFAULT_MATERIAL
from structures.composites.laminate import Laminate
from structures.composites.utils import laminate_builder


def test_laminate_builder() -> None:
    angleslist = [0, 45, -45, 90]
    symmetry = True
    copycenter = False
    multiplicity = 2

    laminate = laminate_builder(
        angleslist=angleslist,
        symmetry=symmetry,
        copycenter=copycenter,
        multiplicity=multiplicity,
        material_props=DEFAULT_MATERIAL,
    )

    assert isinstance(laminate, Laminate)
    assert len(laminate.laminas) == len(angleslist) * 2 * multiplicity - 2
    assert all(lamina.theta in angleslist + angleslist[-2::-1] for lamina in laminate.laminas)
    assert all(lamina.t == DEFAULT_MATERIAL.t for lamina in laminate.laminas)
    assert all(lamina.elastic == DEFAULT_MATERIAL.elastic_properties for lamina in laminate.laminas)
    assert all(lamina.failure == DEFAULT_MATERIAL.failure_properties for lamina in laminate.laminas)
