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
        self.cmfd_phi = np.ones(4)
        self.cmfd_phi_final = np.ones((2,4))
        self.nem4_phi = np.ones((2, self.num_pts))
        self.nem4_phi_final = np.ones((2, self.num_pts))
        self.nem2_phi = np.ones((2, self.num_pts))
        self.nem2_phi_final = np.ones((2, self.num_pts))

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

            print ("CMFD: i=%d\tres=%1.5E\t\tkeff=%1.15f" % (i+1, res, keff))

            # Check convergence
            if res < tol:
                print ("CMFD converged in %d iters with res=%1E"% (i+1, res))
                self.cmfd_keff = keff
                self.cmfd_phi = np.copy(phi)
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

            # Compute the new residual
            res = abs(self.cmfd_keff - keff)
            keff = self.cmfd_keff

            print ("NEM4: i=%d\tres=%1.5E\t\tkeff=%1.15f" % (i, res, keff))

            if res < tol:
                print ("NEM4 converged in %d iters with res=%1E"% (i, res))
                self.nem4_keff = self.cmfd_keff
                break

        
    def solveNEM2(self, tol=1E-10):

        D_tilde = self.matricizer.getDiffusionTildes()
    
        # Initial guess for keff
        keff = 1E10

        for i in range(25):

            self.solveCMFD(tol=1E-5)

            self.matricizer.initializeNEM2ndOrderCoeffMatrix()
            nem2_coeff_matrix = self.matricizer.getNEM2ndOrderCoeffMatrix()

            self.nem2_phi_bar[6] = -self.cmfd_phi[0] + self.cmfd_phi[2]
            self.nem2_phi_bar[7] = -self.cmfd_phi[1] + self.cmfd_phi[3]

            self.nem2_a = scipy.linalg.solve(nem2_coeff_matrix,self.nem2_phi_bar)

            # Compute current at interface in both groups
            fuel_curr = [self.getNEM2Current(group=1, xi=1.,region=0), \
                             self.getNEM2Current(group=2, xi=1.,region=0)]
            mod_curr = [self.getNEM2Current(group=1, xi=0.,region=1), \
                            self.getNEM2Current(group=2, xi=0.,region=1)]

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

            # Compute the new residual
            res = abs(self.cmfd_keff - keff)
            keff = self.cmfd_keff

            print ("NEM2: i=%d\tres=%1.5E\t\tkeff=%1.15f" % (i, res, keff))

            if res < tol:
                print ("NEM2 converged in %d iters with res=%1E"% (i, res))
                self.nem2_keff = self.cmfd_keff
                break


    def getNEM4Flux(self, group=1, xi=0., region=0.):

        group -= 1

        # Build flux from the basis functions
        phi = self.nem4_a[region*8+group*4] * self.nem_basis.P1(xi)
        phi += self.nem4_a[region*8+group*4+1] * self.nem_basis.P2(xi)
        phi += self.nem4_a[region*8+group*4+2] * self.nem_basis.P3(xi)
        phi += self.nem4_a[region*8+group*4+3] * self.nem_basis.P4(xi)

        if region == 0:
            phi += self.cmfd_phi[group]
        else:
            phi += self.cmfd_phi[2+group]

        return phi


    def getNEM2Flux(self, group=1, xi=0., region=0.):

        group -= 1

        # Build flux from the basis functions
        phi = self.nem2_a[region*4+group*2] * self.nem_basis.P1(xi)
        phi += self.nem2_a[region*4+group*2+1] * self.nem_basis.P2(xi)

        if region == 0:
            phi += self.cmfd_phi[group]
        else:
            phi += self.cmfd_phi[2+group]

        return phi


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


    def getNEM2Current(self, group=1, xi=0., region=0):

        group -= 1

        # Build current from the derivatives of the basis functions
        curr = self.nem2_a[region*4+group*2] * self.nem_basis.DP1(xi)
        curr += self.nem2_a[region*4+group*2+1] * self.nem_basis.DP2(xi)

        # Multiply by the diffusion coefficient
        if region == 0:
            curr *= -self.materializer.getDiffusionCoeffs()['fuel'][group]
        else:
            curr *= -self.materializer.getDiffusionCoeffs()['coolant'][group]

        return curr / self.matricizer.getDeltaX()


    def generateCMFDFlux(self, num_pts=1000):

        self.cmfd_phi_final = np.zeros((2,4))
        self.cmfd_phi_final[0, 0] = self.cmfd_phi[0]      # fuel group 1
        self.cmfd_phi_final[0, 1] = self.cmfd_phi[0]      # fuel group 1
        self.cmfd_phi_final[1, 0] = self.cmfd_phi[1]      # fuel group 2
        self.cmfd_phi_final[1, 1] = self.cmfd_phi[1]      # fuel group 2
        self.cmfd_phi_final[0, 2] = self.cmfd_phi[2]      # coolant group 1
        self.cmfd_phi_final[0, 3] = self.cmfd_phi[2]      # coolant group 1
        self.cmfd_phi_final[1, 2] = self.cmfd_phi[3]      # coolant group 2
        self.cmfd_phi_final[1, 3] = self.cmfd_phi[3]      # coolant group 2

    
    def generateNEM4Flux(self, num_pts=1000):

        xi = np.linspace(0,1,500)

        for group in [1,2]:
            for region in [0,1]:
                for i in range(500):
                    self.nem4_phi_final[group-1, i+region*500] = \
                                          self.getNEM4Flux(group, xi[i], region)


    def generateNEM2Flux(self, num_pts=1000):

        xi = np.linspace(0,1,500)

        for group in [1,2]:
            for region in [0,1]:
                for i in range(500):
                    self.nem2_phi_final[group-1, i+region*500] = \
                                          self.getNEM2Flux(group, xi[i], region)


    ############################################################################
    ##################################  PLOTTING  ##############################
    ############################################################################

    def plotCMFDFlux(self):

        fig = plt.figure()
        plt.plot(self.cmfd_x, self.cmfd_phi_final[0, :], linewidth=2)
        plt.plot(self.cmfd_x, self.cmfd_phi_final[1, :], linewidth=2)
        plt.title('Normalized CMFD Flux')
        plt.ylim([0.,1.])
        plt.xlabel('x [cm]')
        plt.ylabel('Flux')
        plt.legend(['Group 1', 'Group 2'])
        plt.grid()
        plt.savefig('cmfd-flux.png')

    def plotNEM4Flux(self):
        
        fig = plt.figure()

        plt.plot(self.nem_x, self.nem4_phi_final[0,:], linewidth=2)
        plt.plot(self.nem_x, self.nem4_phi_final[1,:], linewidth=2)
            
        plt.ylim([0.,1.])
        plt.xlabel('x [cm]')
        plt.ylabel('Flux')
        plt.title('Normalized NEM 4th Order Flux')
        plt.legend(['Group 1', 'Group 2'])
        plt.grid()
        plt.savefig('nem-4-flux.png')


    def plotNEM2Flux(self):

        fig = plt.figure()

        plt.plot(self.nem_x, self.nem2_phi_final[0,:], linewidth=2)
        plt.plot(self.nem_x, self.nem2_phi_final[1,:], linewidth=2)

        plt.ylim([0.,1.])
        plt.xlabel('x [cm]')
        plt.ylabel('Flux')
        plt.title('Normalized NEM 2nd Order Flux')
        plt.legend(['Group 1', 'Group 2'])
        plt.grid()
        plt.savefig('nem-2-flux.png')

    def plotAllFluxes(self, group=1):

        fig = plt.figure()

        plt.plot(self.cmfd_x, self.cmfd_phi_final[group-1,:], linewidth=2)
        plt.plot(self.nem_x, self.nem2_phi_final[group-1,:], linewidth=2)
        plt.plot(self.nem_x, self.nem4_phi_final[group-1,:], linewidth=2)

        plt.ylim([0.,1.])
        plt.xlabel('x [cm]')
        plt.ylabel('Flux')
        plt.title('Normalized Flux in Group ' + str(group))
        if group == 1:
            plt.legend(['CMFD', 'NEM-2', 'NEM-4'], loc=1)
        else:
            plt.legend(['CMFD', 'NEM-2', 'NEM-4'], loc=2)
        plt.grid()
        plt.savefig('group-' + str(group) + '-fluxes.png')
