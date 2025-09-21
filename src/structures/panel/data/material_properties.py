# ------------------------------------------------------------------
# Standard christos material properties:
# ------------------------------------------------------------------
from structures import ElasticProperties, FailureProperties

E1 = 142000  # From assignment
E2 = 11200  # From assignment
G12 = 5000  # From assignment
v12 = 0.3  # From assignment

elastic_properties = ElasticProperties(E1, E2, G12, v12)

# properties needed for failure analysis
E11f = 230000  # TBD
v21f = 0.2  # TBD
msf = 1.1  # TBD
R11t = 2200  # From assignment
R11c = 1800  # From assignment
yt = 70  # From assignment
yc = 300  # From assignment
S = 100  # From assignment

failure_properties = FailureProperties(E11f, v21f, msf, R11t, R11c, yt, yc, S)

t = 0.2  # mm ply thickness
rho = 0.00161  # 1g/mm3

isotropic = {
    "steel": {
        "t": 1,  # mm ply thickness
        "rho": 7.85,  # 1g/mm3
        "E1": 200e3,  # From assignment
        "E2": 200e3,  # From assignment
        "G12": 76923076923e-6,  # From assignment
        "v12": 0.3,  # From assignment
        "E11f": 230000,  # TBD
        "v21f": 0.2,  # TBD
        "msf": 1.1,  # TBD
        "R11t": 2200,  # From assignment
        "R11c": 1800,  # From assignment
        "yt": 70,  # From assignment
        "yc": 300,  # From assignment
        "S": 100,  # From assignment
        "elastic_properties": [200e3, 200e3, 76923076923e-6, 0.3],
        "failure_properties": [230000, 0.2, 1.1, 2200, 1800, 70, 300, 100],
    }
}

CF = {
    "Christos": {
        "t": 0.2,  # mm ply thickness
        "rho": 0.00161,  # 1g/mm3
        "E1": 142000,  # From assignment
        "E2": 11200,  # From assignment
        "G12": 5000,  # From assignment
        "v12": 0.3,  # From assignment
        "E11f": 230000,  # TBD
        "v21f": 0.2,  # TBD
        "msf": 1.1,  # TBD
        "R11t": 2200,  # From assignment
        "R11c": 1800,  # From assignment
        "yt": 70,  # From assignment
        "yc": 300,  # From assignment
        "S": 100,  # From assignment
        "elastic_properties": [142000, 11200, 5000, 0.3],
        "failure_properties": [230000, 0.2, 1.1, 2200, 1800, 70, 300, 100],
    },
    "ezcomposites_spreadtow": {
        "t": 0.12,  # easycomposites website
        "rho": 0.00161,  # 1g/mm3
        "E1": 73494.3,  # From assignment
        "E2": 73494.3,  # From assignment
        "G12": 5000,  # From assignment
        "v12": 0.046,  # From assignment
        "E11f": 230000,  # TBD
        "v21f": 0.2,  # TBD
        "msf": 1.1,  # TBD
        "R11t": 1285.2,  # Estimated using t700
        "R11c": 793.5,  # Estimated using t700
        "yt": 1285.2,  # From assignment
        "yc": 793.5,  # From assignment
        "S": 100,  # From assignment
        "elastic_properties": [73494.3, 73494.3, 5000, 0.046],
        "failure_properties": [230000, 0.2, 1.1, 1285.2, 793.5, 1285.2, 793.5, 100],
    },
    "T700": {
        "t": 0.12,  # random
        "rho": 0.00161,  # 1g/mm3, random
        "E1": 135000,  # datasheet
        "E2": 11200,  # From assignment Christos
        "G12": 5000,  # From assignment Christos
        "v12": 0.3,  # From assignment Christos
        "E11f": 230000,  # TBD
        "v21f": 0.2,  # TBD
        "msf": 1.1,  # Dimitros slides
        "R11t": 2550,  # Datasheet
        "R11c": 1470,  # Datasheet
        "yt": 69,  # Datasheet
        "yc": 300,  # From assignment Christos
        "S": 100,  # From assignment Christos
        "elastic_properties": [135000, 11200, 5000, 0.3],
        "failure_properties": [230000, 0.2, 1.1, 2550, 1470, 69, 300, 100],
    },
}

corematerials = {
    "HRH128": {
        "Ez": 538,
        "Sxz": 3.31,
        "Gxz": 110,
        "Syz": 1.79,
        "Gyz": 66,
        "Xc": 12.62,
        "rho": 128 / 1000,  # g/mm3
    },
    "HRH144": {
        "Ez": 621,
        "Sxz": 3.55,
        "Gxz": 121,
        "Syz": 2.07,
        "Gyz": 76,
        "Xc": 14.48,
        "rho": 144 / 1000,  # g/mm3
    },
    "ROHACELL31A": {
        "Ez": 32,
        "Sxz": 0.4,
        "Gxz": 13,
        "Syz": 0.4,
        "Gyz": 13,
        "Xc": 0.4,
        "rho": 32 / 1000,  # g/mm3
        "G": 13,  # we set this only for isotropic foams! non isotropic materials should not have this, as it should throw an error if called but not existent
    },
}
