import copy

from structures import Lamina
from structures.panel.laminate import Laminate


def laminate_builder(angleslist, symmetry, copycenter, multiplicity, type=None):
    if symmetry:
        if copycenter is True:
            angleslist = angleslist + angleslist[-1::-1]
        elif copycenter is False:
            angleslist = angleslist + angleslist[-2::-1]
    elif not symmetry:
        angleslist = angleslist
    angleslist = angleslist * multiplicity

    # Define standard lamina:
    if type:
        lamina = Lamina(MP.t, 45, MP.elastic_properties, MP.failure_properties, MP.rho)
    else:
        props = MP.CF[type]
        lamina = Lamina(
            props["t"],
            0,
            props["elastic_properties"],
            props["failure_properties"],
            props["rho"],
        )
    laminas = []

    # populate laminas list:
    for angle in angleslist:
        newlamina = copy.deepcopy(lamina)
        newlamina.theta_ = angle
        newlamina.calculate_QS()
        laminas.append(newlamina)

    laminate = Laminate(laminas)
    return laminate
