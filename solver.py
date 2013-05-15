from materializer import *
from matricizer import *
from nembasis import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import math


class Solver:


    def __init__(self, matricizer):

        self.matricizer = matricizer
        self.materializer = self.matricizer.getMaterializer()

        # Initialize an NEM basis object
        self.nem_basis = NEMBasis()

        # Initialize keff for each method
        self.cmfd_keff = 1.0
        self.nem4_keff = 1.0
        self.nem2_keff = 1.0

        # Initialize arrays for the x-axis
        self.num_pts = 1000
        self.cmfd_x = [0., 5., 15., 20.]
        self.nem_x = np.linspace(0., 20., self.num_pts)

        # Initialize arrays for fluxes (both energy groups) for each method
        self.cmfd_phi = np.zeros((2, 2))
        self.nem4_phi = np.zeros((2, self.num_pts))
        self.nem2_phi = np.zeros((2, self.num_pts))

        # Initialize arrays for NEM 
        self.nem4_a = np.zeros(16)
        self.nem2_a = np.zeros(8)
        self.nem4_phi_bar = np.zeros(16)
        self.nem2_phi_bar = np.zeros(8)
        self.D_hat = np.array([[0.,0.],[0.,0.]])


    ############################################################################
    #########################  SOLVER CMFD AND NEM ROUTINES ####################
    ############################################################################

    def solveCMFD(self, tol=1E-10):

        self.matricizer.initializeCMFDDestructionMatrix(self.D_hat)
        self.matricizer.initializeCMFDProductionMatrix()

        M = self.matricizer.getCMFDDestructionMatrix()
        F = self.matricizer.getCMFDProductionMatrix()

        # Guess initial keff and scalar flux
        keff = 1.0
        phi = np.ones(4)

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

            print ("CMFD: i=%d\tres=%1.5E\t\tkeff=%1.15f" % (i, res, keff))

            # Check convergence
            if res < tol:
                print ("CMFD converged in %d iters with res=%1E"% (i, res))
                self.cmfd_keff = keff
                self.cmfd_phi = phi
                break


    def solveNEM4(self, tol=1E-10):

        D_tilde = self.matricizer.getDiffusionTildes()
    
        # Initial guess for keff
        keff = 1E10

        for i in range(25):

            self.solveCMFD(tol=1E-5)

            self.matricizer.initializeNEM4thOrderCoeffMatrix(keff=self.cmfd_keff)
            nem4_coeff_matrix = self.matricizer.getNEM4thOrderCoeffMatrix()

            self.nem4_phi_bar[14] = -self.cmfd_phi[0] + self.cmfd_phi[2]
            self.nem4_phi_bar[15] = -self.cmfd_phi[1] + self.cmfd_phi[3] 
            
            self.nem4_a = scipy.linalg.solve(nem4_coeff_matrix,self.nem4_phi_bar)

            # Compute current at interface in both groups
            fuel_curr = [self.getNEM4Current(group=1, xi=1.,region=0), \
                             self.getNEM4Current(group=2, xi=1.,region=0)]
            mod_curr = [self.getNEM4Current(group=1, xi=0.,region=1), \
                            self.getNEM4Current(group=2, xi=0.,region=1)]

            print 'fuel current = ' + str(fuel_curr)
            print 'moderator current = ' + str(mod_curr)

            # Back out D_hat from interface currents
            self.D_hat[0][0] = (D_tilde[0] * (self.cmfd_phi[0] - \
                                 self.cmfd_phi[2]) - fuel_curr[0]) / \
                                 (self.cmfd_phi[0] + self.cmfd_phi[2])
            self.D_hat[0][1] = (D_tilde[1] * (self.cmfd_phi[1] - \
                                 self.cmfd_phi[3]) - fuel_curr[1]) / \
                                 (self.cmfd_phi[1] + self.cmfd_phi[3])
            self.D_hat[1][0] = (D_tilde[0] * (self.cmfd_phi[0] - \
                                 self.cmfd_phi[2]) - mod_curr[0]) / \
                                 (self.cmfd_phi[0] + self.cmfd_phi[2])
            self.D_hat[1][1] = (D_tilde[1] * (self.cmfd_phi[1] - \
                                 self.cmfd_phi[3]) - mod_curr[1]) / \
                                 (self.cmfd_phi[1] + self.cmfd_phi[3])

            print 'D_hat = ' + str(self.D_hat)

            # Compute the new residual
            res = abs(self.cmfd_keff - keff)
            keff = self.cmfd_keff

            print ("NEM4: i=%d\tres=%1.5E\t\tkeff=%1.15f" % (i, res, keff))

            if res < tol:
                print ("NEM4 converged in %d iters with res=%1E"% (i, res))
                self.nem4_keff = self.cmfd_keff
                break

        
    def solveNEM2(self, tol=1E-10):

        self.matricizer.initializeNEM2ndOrderCoeffMatrix(self)
        nem2_coeffs = self.matricizer.getNEM2ndOrderCoeffMatrix(self)


    def getNEM4Phi(self, group=1, x=0., region=-1):

        if not isinstance(x, np.ndarray):
            x = np.array([x])

        phis = []
        group -= 1

        for i in range(len(x)):

            # The position is in the fuel
            if x[i] < 10. or (x[i] == 10. and region == 0):
                x[i] /= 10.                     # Project x from [0,10] to [0,1]
                phi = self.cmfd_phi[group]
                phi += self.nem4_a[group*4] * self.nem_basis.P1(x[i])
                phi += self.nem4_a[group*4+1] * self.nem_basis.P2(x[i])
                phi += self.nem4_a[group*4+2] * self.nem_basis.P3(x[i])
                phi += self.nem4_a[group*4+3] * self.nem_basis.P4(x[i])
                phis.append(phi)

            if x[i] > 10. or (x[i] == 0. and region == 1):
                x[i] -= 10.
                x[i] /= 10.                     # Project x from [0,10] to [0,1]
                phi = self.cmfd_phi[group]
                phi += self.nem4_a[8+group*4] * self.nem_basis.P1(x[i])
                phi += self.nem4_a[8+group*4+1] * self.nem_basis.P2(x[i])
                phi += self.nem4_a[8+group*4+2] * self.nem_basis.P3(x[i])
                phi += self.nem4_a[8+group*4+3] * self.nem_basis.P4(x[i])
                phis.append(phi)

        return phis


    def getNEM2Phi(self, group=1, x=0.):

        if not isinstance(x, np.ndarray):
            x = np.array([x])

        phis = []
        group -= 1

        for i in range(len(x)):

            # The position is in the fuel
            if x[i] < 10.:
                x[i] /= 10.                     # Project x from [0,10] to [0,1]
                phi = self.cmfd_phi[group]
                phi += self.nem4_a[group*4] * self.nem_basis.P1(x[i])
                phi += self.nem4_a[group*4+1] * self.nem_basis.P2(x[i])
                phis.append(phi)

            if x[i] > 10.:
                x[i] -= 10.
                x[i] /= 10.                     # Project x from [0,10] to [0,1]
                phi = self.cmfd_phi[group]
                phi += self.nem4_a[8+group*4] * self.nem_basis.P1(x[i])
                phi += self.nem4_a[8+group*4+1] * self.nem_basis.P2(x[i])
                phis.append(phi)

        return phis


    def getNEM4Current(self, group=1, xi=0., region=0):

        group -= 1

        # Build current from the derivatives of the basis functions
        curr = self.nem4_a[region*8+group*4] * self.nem_basis.DP1(xi)
        curr += self.nem4_a[region*8+group*4+1] * self.nem_basis.DP2(xi)
        curr += self.nem4_a[region*8+group*4+2] * self.nem_basis.DP3(xi)
        curr += self.nem4_a[region*8+group*4+3] * self.nem_basis.DP4(xi)

        # Multiply by the diffusion coefficient
        if region == 0:
            curr *= -self.materializer.getDiffusionCoeffs()['fuel'][group]
        else:
            curr *= -self.materializer.getDiffusionCoeffs()['coolant'][group]

        return curr / self.matricizer.getDeltaX()


    def getNEM2Current(self, group=1, x=0.):

        if not isinstance(x, np.ndarray):
            x = np.array([x])

        if x > 10.:
            x -= 10. 

        xi = x/10.

        group -= 1
        region = (xi / (10.+1E-10)).astype(np.int32)

        curr = self.nem4_a[region*8+group*4] * self.nem_basis.DP1(xi)
        curr += self.nem4_a[region*8+group*4+1] * self.nem_basis.DP2(xi)

        if region == 0:
            curr *= -self.materializer.getDiffusionCoeffs()['fuel'][group]
        else:
            curr *= -self.materializer.getDiffusionCoeffs()['coolant'][group]

        return curr[0]


    ############################################################################
    ##################################  PLOTTING  ##############################
    ############################################################################

    def plotCMFDFlux(self):

        self.cmfd_phi[0, 0] = phi[0]      # fuel group 1
        self.cmfd_phi[0, 1] = phi[0]      # fuel group 1
        self.cmfd_phi[1, 0] = phi[1]      # fuel group 2
        self.cmfd_phi[1, 1] = phi[1]      # fuel group 2
        self.cmfd_phi[0, 2] = phi[2]      # coolant group 1
        self.cmfd_phi[0, 3] = phi[2]      # coolant group 1
        self.cmfd_phi[1, 2] = phi[3]      # coolant group 2
        self.cmfd_phi[1, 3] = phi[3]      # coolant group 2

        fig = plt.figure()
        plt.plot(self.cmfd_x, self.cmfd_phi[0, :], linewidth=2)
        plt.plot(self.cmfd_x, self.cmfd_phi[1, :], linewidth=2)
        plt.title('Normalized CMFD Flux')
        plt.xlabel('x [cm]')
        plt.ylabel('Flux')
        plt.legend(['Group 1', 'Group 2'])
        plt.grid()
        plt.savefig('cmfd-flux.png')

    def plotNEM4Flux(self, suffix=''):

        fig = plt.figure()
        plt.plot(self.nem_x, self.nem4_phi[0])
        plt.plot(self.nem_x, self.nem4_phi[1])
        plt.xlabel('x [cm]')
        plt.ylabel('Flux')
        plt.title('Normalized NEM 4th Order Flux')
        plt.legend(['Group 1', 'Group 2'])
        plt.grid()
        plt.savefig('cmfd-flux.png')

#    def plotNEM2Flux(self):

#    def plotAllFluxes(self):
