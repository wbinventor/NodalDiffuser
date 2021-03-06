import numpy as np

class Materializer:

    def __init__(self):

        # Diffusion coefficients (group 1, group 2)
        self.fuel_D = np.array([1.5, 0.4])
        self.coolant_D = np.array([1.5, 0.2])

        # Absorption cross-sections (group 1, group 2)
        self.fuel_abs_xs = np.array([0.005, 0.1])
        self.coolant_abs_xs = np.array([0., 0.01])

        # Nu Fission cross-sections (group 1, group 2)
        self.fuel_nu_fiss_xs = np.array([0.005, 0.15])
        self.coolant_nu_fiss_xs = np.array([0., 0.])

        # Downscatter cross-sections
        self.fuel_scatter12_xs = 0.02
        self.coolant_scatter12_xs = 0.025


    def getDiffusionCoeffs(self):
        return {'fuel': self.fuel_D, 'coolant': self.coolant_D}


    def getAbsorptionXS(self):
        return {'fuel': self.fuel_abs_xs, 'coolant': self.coolant_abs_xs}

    
    def getNuFissionXS(self):
        return {'fuel': self.fuel_nu_fiss_xs, 
                'coolant': self.coolant_nu_fiss_xs}


    def getScatter12XS(self):
        return {'fuel': self.fuel_scatter12_xs, 
                'coolant': self.coolant_scatter12_xs}
