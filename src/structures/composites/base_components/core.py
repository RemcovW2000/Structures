import numpy as np

from structures.composites.data_utils import CoreProperties


class Core:
    def __init__(self, h: float, properties: CoreProperties):
        self.h = h
        self.properties: CoreProperties = properties

    def Gxbarz(self, theta: float) -> float:
        Gxz = self.properties.Gxz
        Gyz = self.properties.Gyz
        sin_theta_squared = np.sin(theta) ** 2
        cos_theta_squared = np.cos(theta) ** 2
        Gxbarz = sin_theta_squared * Gyz + cos_theta_squared * Gxz
        return Gxbarz

    def Gybarz(self, theta: float) -> float:
        Gxz = self.properties.Gxz
        Gyz = self.properties.Gyz
        sin_theta_squared = np.sin(theta) ** 2
        cos_theta_squared = np.cos(theta) ** 2
        Gybarz = cos_theta_squared * Gyz + sin_theta_squared * Gxz
        return Gybarz

    def calculate_weight_per_A(self) -> float:
        """
        Calculates the weight of the core material per unit area based on the density and thickness. Does not take into
        account extra weight per unit area of the core material due to soaking of resin into partially open cells.

        :return:
        """
        return self.h * self.properties.rho
