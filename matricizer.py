import numpy as np
import matplotlib.pyplot as plt


class Matricizer:

    ############################################################################
    ###############################  INITIALIZATION  ###########################
    ############################################################################

    def __init__(self, materializer, delta_x=10.):

        # Set the mesh spacing for this simple two region problem
        self.delta_x = delta_x

        # Get materials data from the materializer
        self.materializer = materializer
        self.D = self.materializer.getDiffusionCoeffs()
        self.abs = self.materializer.getAbsorptionXS()
        self.scatter = self.materializer.getScatter12XS()
        self.fission = self.materializer.getNuFissionXS()

        # Compute diffusion coupling coefficients for CMFD
        self.D_tilde = np.zeros(2)
        self.computeDTilde()

        # Create numpy arrays for CMFD destruction (M) and production (F) arrays
        self.M = np.zeros((4,4))
        self.F = np.zeros((4,4))

        # Create numpy arrays for NEM-4 and NEM-2 coefficients
        self.NEM4_coeffs = np.zeros((16,16))
        self.NEM2_coeffs = np.zeros((8,8))


    def computeDTilde(self):

        # Diffusion coupling coefficients
        self.D_tilde[0] = (2. * self.D['fuel'][0] * self.D['coolant'][0]) / \
                    (self.delta_x * (self.D['fuel'][0] + self.D['coolant'][0]))

        self.D_tilde[1] = (2. * self.D['fuel'][1] * self.D['coolant'][1]) / \
                    (self.delta_x * (self.D['fuel'][1] + self.D['coolant'][1]))


    def initializeCMFDDestructionMatrix(self, D_hat=np.array([[0.,0.],[0.,0.]])):

        # Fuel group 1
        self.M[0,0] = self.D_tilde[0] - D_hat[0,0] + (self.abs['fuel'][0] + \
                      self.scatter['fuel']) * self.delta_x
        self.M[0,2] = -self.D_tilde[0] - D_hat[0,0]

        # Fuel group 2
        self.M[1,1] = self.D_tilde[1] - D_hat[0,1] + self.abs['fuel'][1] * \
                      self.delta_x
        self.M[1,3] = -self.D_tilde[1] - D_hat[0,1]
        self.M[1,0] = -self.scatter['fuel'] * self.delta_x

        # Coolant group 1
        self.M[2,0] = -self.D_tilde[0] + D_hat[1,0]
        self.M[2,2] = self.D_tilde[0] + D_hat[1,0] + (self.abs['coolant'][0] + \
                 self.scatter['coolant']) * self.delta_x

        # Coolant group 2
        self.M[3,1] = -self.D_tilde[1] + D_hat[1,1]
        self.M[3,3] = self.D_tilde[1] + D_hat[1,1] + \
                 self.abs['coolant'][1] * self.delta_x
        self.M[3,2] = -self.scatter['coolant'] * self.delta_x


    def initializeCMFDProductionMatrix(self):

        # Initialize production matrix
        self.F[0,0] = self.fission['fuel'][0] * self.delta_x
        self.F[0,1] = self.fission['fuel'][1] * self.delta_x


    def initializeNEM4thOrderCoeffMatrix(self, keff=1.0):

        # Build 4th order NEM matrix using coefficients found in Mathematica
        self.NEM4_coeffs[0,0] = (-self.fission['fuel'][0] / (3. * keff)) + \
                                (self.abs['fuel'][0]+self.scatter['fuel']) / 3.
        self.NEM4_coeffs[0,2] = (12. * self.D['fuel'][0] / self.delta_x**2) - \
                              (self.fission['fuel'][0] / (5. * float(keff))) + \
                              ((self.abs['fuel'][0] + self.scatter['fuel']) / 5.)
        self.NEM4_coeffs[0,4] = -self.fission['fuel'][1] / (3. * keff)
        self.NEM4_coeffs[0,6] = -self.fission['fuel'][1] / (5. * keff)

        self.NEM4_coeffs[1,0] = -self.scatter['fuel'] / 3.
        self.NEM4_coeffs[1,2] = -self.scatter['fuel'] / 5.
        self.NEM4_coeffs[1,4] = self.abs['fuel'][1] / 3.
        self.NEM4_coeffs[1,6] = (12. * self.D['fuel'][1] / self.delta_x**2) + \
                                (self.abs['fuel'][1] / 5.)

        self.NEM4_coeffs[2,8] = (self.abs['coolant'][0] + \
                                     self.scatter['coolant']) / 3.
        self.NEM4_coeffs[2,10] = (12. * self.D['coolant'][0] / \
                                 self.delta_x**2) + ((self.abs['coolant'][0] + \
                                 self.scatter['coolant']) / 5.)

        self.NEM4_coeffs[3,8] = -self.scatter['coolant'] / 3.
        self.NEM4_coeffs[3,10] = -self.scatter['coolant'] / 5.
        self.NEM4_coeffs[3,12] = self.abs['coolant'][1] / 3.
        self.NEM4_coeffs[3,14] = (12. * self.D['coolant'][1] / \
                                self.delta_x**2) + (self.abs['coolant'][1] / 5.)

        self.NEM4_coeffs[4,1] = (-self.fission['fuel'][0] / (5. * keff)) + \
                                (self.abs['fuel'][0]+self.scatter['fuel']) / 5.
        self.NEM4_coeffs[4,3] = (-12. * self.D['fuel'][0] / self.delta_x**2) + \
                               (3. * self.fission['fuel'][0] / (35. * keff)) - \
                               (3. * (self.abs['fuel'][0] + \
                                      self.scatter['fuel']) / 35.)
        self.NEM4_coeffs[4,5] = -self.fission['fuel'][1] / (5. * keff)
        self.NEM4_coeffs[4,7] = 3. * self.fission['fuel'][1] / (35. * keff)

        self.NEM4_coeffs[5,1] = -self.scatter['fuel'] / 5.
        self.NEM4_coeffs[5,3] = 3. * self.scatter['fuel'] / 35.
        self.NEM4_coeffs[5,5] = self.abs['fuel'][1] / 5.
        self.NEM4_coeffs[5,7] = (-12. * self.D['fuel'][1] / self.delta_x**2) - \
                                      (3. * self.abs['fuel'][1] / 35.)

        self.NEM4_coeffs[6,9] = (self.abs['coolant'][0] + \
                                 self.scatter['coolant']) / 5. 
        self.NEM4_coeffs[6,11] = (-12. * self.D['coolant'][0] / 
                                   self.delta_x**2) - \
                                  (3. * (self.abs['coolant'][0] + \
                                  self.scatter['coolant']) / 35.)

        self.NEM4_coeffs[7,9] = -self.scatter['coolant'] / 5.
        self.NEM4_coeffs[7,11] = 3. * self.scatter['coolant'] / 35.
        self.NEM4_coeffs[7,13] = self.abs['coolant'][1] / 5.
        self.NEM4_coeffs[7,15] = (-12. * self.D['coolant'][1] / 
                                   self.delta_x**2) - \
                                   (3. * self.abs['coolant'][1] / 35.)

        self.NEM4_coeffs[8,0] = -2. * self.D['fuel'][0] / self.delta_x
        self.NEM4_coeffs[8,1] = -6. * self.D['fuel'][0] / self.delta_x
        self.NEM4_coeffs[8,2] = 6. * self.D['fuel'][0] / self.delta_x
        self.NEM4_coeffs[8,3] = -6. * self.D['fuel'][0] / self.delta_x

        self.NEM4_coeffs[9,4] = -2. * self.D['fuel'][1] / self.delta_x
        self.NEM4_coeffs[9,5] = -6. * self.D['fuel'][1] / self.delta_x
        self.NEM4_coeffs[9,6] = 6. * self.D['fuel'][1] / self.delta_x
        self.NEM4_coeffs[9,7] = -6. * self.D['fuel'][1] / self.delta_x

        self.NEM4_coeffs[10,8] = -2. * self.D['coolant'][0] / \
                                  self.delta_x
        self.NEM4_coeffs[10,9] = 6. * self.D['coolant'][0] / self.delta_x
        self.NEM4_coeffs[10,10] = 6. * self.D['coolant'][0] / self.delta_x
        self.NEM4_coeffs[10,11] = 6.* self.D['coolant'][0] / self.delta_x
        
        self.NEM4_coeffs[11,12] = -2. * self.D['coolant'][1] / self.delta_x
        self.NEM4_coeffs[11,13] = 6. * self.D['coolant'][1] / self.delta_x
        self.NEM4_coeffs[11,14] = 6. * self.D['coolant'][1] / self.delta_x
        self.NEM4_coeffs[11,15] = 6. * self.D['coolant'][1] / self.delta_x
        
        self.NEM4_coeffs[12,0] = -2. * self.D['fuel'][0] / self.delta_x
        self.NEM4_coeffs[12,1] = 6.* self.D['fuel'][0] / self.delta_x
        self.NEM4_coeffs[12,2] = 6.* self.D['fuel'][0] / self.delta_x
        self.NEM4_coeffs[12,3] = 6.* self.D['fuel'][0] / self.delta_x
        self.NEM4_coeffs[12,8] = 2.* self.D['coolant'][0] / self.delta_x
        self.NEM4_coeffs[12,9] = 6. * self.D['coolant'][0] / self.delta_x
        self.NEM4_coeffs[12,10] = -6. * self.D['coolant'][0] / self.delta_x
        self.NEM4_coeffs[12,11] = 6. * self.D['coolant'][0] / self.delta_x

        self.NEM4_coeffs[13,4] = -2. * self.D['fuel'][1] / self.delta_x
        self.NEM4_coeffs[13,5] = 6. * self.D['fuel'][1] / self.delta_x
        self.NEM4_coeffs[13,6] = 6. * self.D['fuel'][1] / self.delta_x
        self.NEM4_coeffs[13,7] = 6. * self.D['fuel'][1] / self.delta_x
        self.NEM4_coeffs[13,12] = 2. * self.D['coolant'][1] / self.delta_x
        self.NEM4_coeffs[13,13] = 6.* self.D['coolant'][1] / self.delta_x
        self.NEM4_coeffs[13,14] = -6. * self.D['coolant'][1] / self.delta_x
        self.NEM4_coeffs[13,15] = 6. * self.D['coolant'][1] / self.delta_x

        self.NEM4_coeffs[14,0] = 1.
        self.NEM4_coeffs[14,1] = -1.
        self.NEM4_coeffs[14,8] = 1.
        self.NEM4_coeffs[14,9] = 1.

        self.NEM4_coeffs[15,4] = 1.
        self.NEM4_coeffs[15,5] = -1.
        self.NEM4_coeffs[15,12] = 1.
        self.NEM4_coeffs[15,13] = 1.
        

    def initializeNEM2ndOrderCoeffMatrix(self):

        # Build 2nd order NEM matrix using coefficients found in Mathematica
        self.NEM2_coeffs[0,0] = -2. * self.D['fuel'][0] / self.delta_x
        self.NEM2_coeffs[0,1] = -6. * self.D['fuel'][0] / self.delta_x
        
        self.NEM2_coeffs[1,2] = -2. * self.D['fuel'][1] / self.delta_x
        self.NEM2_coeffs[1,3] = -6. * self.D['fuel'][1] / self.delta_x
        
        self.NEM2_coeffs[2,4] = -2. * self.D['coolant'][0] / self.delta_x
        self.NEM2_coeffs[2,5] = 6. * self.D['coolant'][0] / self.delta_x
        
        self.NEM2_coeffs[3,6] = -2. * self.D['coolant'][1] / self.delta_x
        self.NEM2_coeffs[3,7] = 6. * self.D['coolant'][1] / self.delta_x
        
        self.NEM2_coeffs[4,0] = -2. * self.D['fuel'][0] / self.delta_x
        self.NEM2_coeffs[4,1] = 6.* self.D['fuel'][0] / self.delta_x
        self.NEM2_coeffs[4,4] = 2. * self.D['coolant'][0] / self.delta_x
        self.NEM2_coeffs[4,5] = 6. * self.D['coolant'][0] / self.delta_x
        
        self.NEM2_coeffs[5,2] = -2. * self.D['fuel'][1] / self.delta_x
        self.NEM2_coeffs[5,3] = 6. * self.D['fuel'][1] / self.delta_x
        self.NEM2_coeffs[5,6] = 2. * self.D['coolant'][1] / self.delta_x
        self.NEM2_coeffs[5,7] = 6. * self.D['coolant'][1] / self.delta_x

        self.NEM2_coeffs[6,0] = 1.
        self.NEM2_coeffs[6,1] = -1.
        self.NEM2_coeffs[6,4] = 1.
        self.NEM2_coeffs[6,5] = 1.
        
        self.NEM2_coeffs[7,2] = 1.
        self.NEM2_coeffs[7,3] = -1.
        self.NEM2_coeffs[7,6] = 1.
        self.NEM2_coeffs[7,7] = 1.
        

    ############################################################################
    #############################  GETTERS / SETTERS  ##########################
    ############################################################################

    def getMaterializer(self):
        return self.materializer


    def getDeltaX(self):
        return self.delta_x


    def getDiffusionTildes(self):
        return self.D_tilde


    def getCMFDDestructionMatrix(self):
        return self.M


    def getCMFDProductionMatrix(self):
        return self.F


    def getNEM4thOrderCoeffMatrix(self):
        return self.NEM4_coeffs


    def getNEM2ndOrderCoeffMatrix(self):
        return self.NEM2_coeffs


    ############################################################################
    #################################  PLOTTING  ###############################
    ############################################################################

    def spyCMFDProductionMatrix(self):
        fig = plt.figure()
        plt.spy(self.F, markersize=30)
        plt.title('CMFD Production Matrix')
        plt.show()

    
    def spyCMFDDestructionMatrix(self):
        fig = plt.figure()
        plt.spy(self.M, markersize=30)
        plt.title('CMFD Destruction Matrix')
        plt.show()


    def spyNEM4thOrderCoeffMatrix(self):
        fig = plt.figure()
        plt.spy(self.NEM4_coeffs, markersize=20) 
        plt.title('NEM-4 Coefficient Matrix')
        plt.show()


    def spyNEM2ndOrderCoeffMatrix(self):
        fig = plt.figure()
        plt.spy(self.NEM2_coeffs, markersize=30) 
        plt.title('NEM-2 Coefficient Matrix')
        plt.show()
