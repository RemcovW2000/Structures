import copy

from structures import Lamina
from structures.panel.laminate import Laminate
from structures.panel.data import material_properties as mp


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
        lamina = Lamina(mp.t, 45, mp.elastic_properties, mp.failure_properties, mp.rho)
    else:
        props = mp.CF[type]
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
        newlamina._compute_q_s()
        laminas.append(newlamina)

    laminate = Laminate(laminas)
    return laminate
