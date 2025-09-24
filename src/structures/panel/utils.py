from typing import Optional

from structures import Lamina
from structures.panel.data.lamina_props import DEFAULT_MATERIAL
from structures.panel.data_utils import MaterialProperties
from structures.panel.laminate import Laminate


def laminate_builder(
    angleslist: list[float],
    symmetry: bool,
    copycenter: bool,
    multiplicity: int,
    material_props: Optional[MaterialProperties] = None,
) -> Laminate:
    if symmetry:
        if copycenter is True:
            angleslist = angleslist + angleslist[-1::-1]
        elif copycenter is False:
            angleslist = angleslist + angleslist[-2::-1]
    elif not symmetry:
        pass
    angleslist = angleslist * multiplicity

    if not material_props:
        material_props = DEFAULT_MATERIAL

    laminas = []
    for angle in angleslist:
        laminas.append(
            Lamina(
                t=material_props.t,
                theta=angle,
                elastic=material_props.elastic_properties,
                failure=material_props.failure_properties,
            )
        )

    laminate = Laminate(laminas)
    return laminate
