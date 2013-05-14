from materializer import *
from matricizer import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import math


class Solver:


    def __init__(self):

        # Initialize a materializer and matricizer
        self.materializer = Materializer()
        self.matricizer = Matricizer(self.materializer)

        # Initialize keff for each method
        self.cmfd_keff = 1.0
        self.nem4_keff = 1.0
        self.nem2_keff = 1.0

        # Initialize arrays for the x-axis
        self.num_pts = 1000
        self.x = np.linspace(0., 20., self.num_pts)
        self.fuel_indices = self.x <= 10.
        self.coolant_indices = self.x > 10.

        # Initialize arrays for fluxes for each method
        self.cmfd_phi = np.zeros(self.num_pts)
        self.nem4_phi = np.zeros(self.num_pts)
        self.nem2_phi = np.zeros(self.num_pts)


    def solveCMFD(self, D_hat=[0.,0.]):

        self.matricizer.initializeCMFDDestructionMatrix(D_hat)
        self.matricizer.initializeCMFDProductionMatrix()

        M = self.matricizer.getCMFDDestructionMatrix()
        F = self.matricizer.getCMFDProductionMatrix()

        # Guess initial keff and scalar flux
        keff = 1.0
        phi = np.ones(4)

        # Array for phi_res and keff_res
        res = []

        for i in range(10):

            # Solve for the new flux using an Ax=b solve
            phi_new = scipy.linalg.solve(M, (1./keff)*np.dot(F,phi))
	
            # Update keff
            source_new = sum(np.dot(F,phi_new))
            source_old = sum(np.dot(F,phi))
    
            keff_new = source_new / source_old * keff
            
            # Normalize new flux and set it to be the old flux
            phi = phi_new / scipy.linalg.norm(phi_new) 
            keff = keff_new

            # Compute residuals
            res = math.sqrt(scipy.linalg.norm(source_old - source_new) / M.size)

            print ("CMFD: i=%d\tres=%1.5E\tkeff=%1.5f" % (i, res, keff))

            # Check convergence
            if res < 1E-5:
                print ("CMFD converged in %d iters with res=%1E"% (i, res))
                self.cmfd_keff = keff
                self.cmfd_phi[self.fuel_indices] = phi[0]
                self.cmfd_phi[self.coolant_indices] = phi[1]
                break
