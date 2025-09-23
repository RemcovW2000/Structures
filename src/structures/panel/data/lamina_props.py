from structures.panel.data_utils import ElasticProperties, FailureProperties, MaterialProperties

steel = MaterialProperties(
    t=1.0,
    rho=7.85,  # g/cm3 -> 1e-6 g
    name="steel",
    elastic_properties=ElasticProperties(200e3, 200e3, 76923076923e-6, 0.3),
    failure_properties=FailureProperties(230000, 0.2, 1.1, 2200, 1800, 70, 300, 100),
)
Christos = MaterialProperties(
    name="Christos",
    t=0.2,
    rho=0.00161,
    elastic_properties=ElasticProperties(142000, 11200, 5000, 0.3),
    failure_properties=FailureProperties(230000, 0.2, 1.1, 2200, 1800, 70, 300, 100),
)
ezcomposites_spreadtow = MaterialProperties(
    name="ezcomposites_spreadtow",
    t=0.12,
    rho=0.00161,
    elastic_properties=ElasticProperties(73494.3, 73494.3, 5000, 0.046),
    failure_properties=FailureProperties(230000, 0.2, 1.1, 1285.2, 793.5, 1285.2, 793.5, 100),
)
T700 = MaterialProperties(
    name="T700",
    t=0.12,
    rho=0.00161,
    elastic_properties=ElasticProperties(135000, 11200, 5000, 0.3),
    failure_properties=FailureProperties(230000, 0.2, 1.1, 2550, 1470, 69, 300, 100),
)
