import math
import numpy as np
from scipy.linalg import *
import matplotlib.pyplot as plt

# Geometric parameters
delta_x = 10.

# Diffusion coefficients (group 1, group 2)
fuel_D = [1.5, 0.4]
coolant_D = [1.5, 0.2]

# Diffusion coupling coefficients
D_hat = [(2. * fuel_D[0] * coolant_D[0]) / \
         (delta_x * (fuel_D[0] + coolant_D[0])), \
         (2. * fuel_D[1] * coolant_D[1]) / \
         (delta_x * (fuel_D[1] + coolant_D[1]))]

# Diffusion correction terms
D_tilde = [0., 0.]

# Absorption cross-sections (group 1, group 2)
fuel_abs_xs = [0.005, 0.1]
coolant_abs_xs = [0., 0.01]

# Nu Fission cross-sections (group 1, group 2)
fuel_nu_fiss_xs = [0.005, 0.15]

# Downscatter cross-sections
fuel_scatter12_xs = 0.02
coolant_scatter12_xs = 0.025

# Initialize destruction (M) and production (F) matrices
M = np.zeros((4,4))
F = np.zeros((4,4))

# Setup destruction matrix
# Fuel group 1
M[0,0] = D_hat[0] - D_tilde[0] + (fuel_abs_xs[0] + fuel_scatter12_xs)*delta_x
M[0,2] = -D_hat[0] - D_tilde[0]

# Fuel group 2
M[1,1] = D_hat[1] - D_tilde[1] + fuel_abs_xs[1]*delta_x
M[1,3] = -D_hat[1] - D_tilde[1]
M[1,0] = -fuel_scatter12_xs*delta_x

# Coolant group 1
M[2,0] = -D_hat[0] + D_tilde[0]
M[2,2] = D_hat[0] - D_tilde[0] + (coolant_abs_xs[0] + \
                                       coolant_scatter12_xs)*delta_x

# Coolant group 2
M[3,1] = -D_hat[1] + D_tilde[1]
M[3,3] = D_hat[1] - D_tilde[1] + coolant_abs_xs[1]*delta_x
M[3,2] = -coolant_scatter12_xs*delta_x

# Initialize production matrix
F[0,0] = fuel_nu_fiss_xs[0]*delta_x
F[0,1] = fuel_nu_fiss_xs[1]*delta_x

# Guess initial keff and scalar flux
keff = 1.0
phi = np.ones(4)

# Array for phi_res and keff_res
res = []

for i in range(10):

    # Solve for the new flux using an Ax=b solve
    phi_new = solve(M, (1./keff)*np.dot(F,phi))
	
    # Update keff
    source_new = sum(np.dot(F,phi_new))
    source_old = sum(np.dot(F,phi))

    keff_new = source_new / source_old * keff

    # Normalize new flux and set it to be the old flux
    phi = phi_new / norm(phi_new) 
    keff = keff_new

    # Compute residuals
    res = math.sqrt(norm(source_old - source_new) / M.size)

    print ("Power iteration: i=%d\tres=%1.5E\tkeff=%1.5f" % (i, res, keff))

    # Check convergence
    if res < 1E-5:
        print ("Power iteration converged in %1d iters with res=%1E"% (i, res))
        break


nem4_coeffs = np.zeros((16,16))
nem2_coeffs = np.zeros((8,8))

# Build 4th order NEM matrix using coefficients found in Mathematica
nem4_coeffs[0,0] = (fuel_nu_fiss_xs[0] + fuel_scatter12_xs) / 3.
nem4_coeffs[0,2] = (12. * fuel_D[0] / delta_x**2) + \
                   ((fuel_abs_xs[0] + fuel_scatter12_xs) / 5.)

nem4_coeffs[1,0] = -fuel_scatter12_xs / 3.
nem4_coeffs[1,2] = -fuel_scatter12_xs / 5.
nem4_coeffs[1,5] = fuel_abs_xs[0] / 3.
nem4_coeffs[1,7] = (12. * fuel_D[1] / delta_x**2) + (fuel_abs_xs[1] / 5.)

nem4_coeffs[2,8] = (coolant_abs_xs[0] + coolant_scatter12_xs) / 3.
nem4_coeffs[2,10] = (12. * coolant_D[0] / delta_x**2) + \
                   ((coolant_abs_xs[0] + coolant_scatter12_xs) / 5.)

nem4_coeffs[3,8] = -coolant_scatter12_xs / 3.
nem4_coeffs[3,10] = -coolant_scatter12_xs / 5.
nem4_coeffs[3,12] = coolant_abs_xs[1] / 3.
nem4_coeffs[3,14] = (12. * coolant_D[1] / delta_x**2) + (coolant_abs_xs[1] / 5.)

nem4_coeffs[4,1] = (fuel_abs_xs[0] + fuel_scatter12_xs) / 5.
nem4_coeffs[4,3] = (-12. * fuel_D[0] / delta_x**2) - \
                   (3. * (fuel_abs_xs[0] + fuel_scatter12_xs) / 35.)

nem4_coeffs[5,2] = -fuel_scatter12_xs / 5.
nem4_coeffs[5,4] = 3. * fuel_scatter12_xs / 35.
nem4_coeffs[5,6] = fuel_abs_xs[1] / 5.
nem4_coeffs[5,8] = (-12. * fuel_D[1] / delta_x**2) - (3. * fuel_abs_xs[1] / 35.)

nem4_coeffs[6,9] = (coolant_abs_xs[0] + coolant_scatter12_xs) / 5. 
nem4_coeffs[6,11] = (-12. * coolant_D[0] / delta_x**2) - \
                    (3. * (coolant_abs_xs[0] + coolant_scatter12_xs) / 35.)

nem4_coeffs[7,9] = -coolant_scatter12_xs / 5.
nem4_coeffs[7,11] = 3. * coolant_scatter12_xs / 35.
nem4_coeffs[7,13] = coolant_abs_xs[1] / 5.
nem4_coeffs[7,15] = (-12. * coolant_D[1] / delta_x**2) - (3. * coolant_abs_xs[1] / 35.)

nem4_coeffs[8,0] = -2. * fuel_D[0] / delta_x
nem4_coeffs[8,1] = -6. * fuel_D[0] / delta_x
nem4_coeffs[8,2] = 6. * fuel_D[0] / delta_x
nem4_coeffs[8,3] = -6. * fuel_D[0] / delta_x

nem4_coeffs[9,4] = -2. * fuel_D[1] / delta_x
nem4_coeffs[9,5] = -6. * fuel_D[1] / delta_x
nem4_coeffs[9,6] = 6. * fuel_D[1] / delta_x
nem4_coeffs[9,7] = -6. * fuel_D[1] / delta_x

nem4_coeffs[10,8] = -2. * coolant_D[0] / delta_x
nem4_coeffs[10,9] = 6. * coolant_D[0] / delta_x
nem4_coeffs[10,10] = 6. * coolant_D[0] / delta_x
nem4_coeffs[10,11] = 6.* coolant_D[0] / delta_x

nem4_coeffs[11,12] = -2. * coolant_D[1] / delta_x
nem4_coeffs[11,13] = 6. * coolant_D[1] / delta_x
nem4_coeffs[11,14] = 6. * coolant_D[1] / delta_x
nem4_coeffs[11,15] = 6. * coolant_D[1] / delta_x

nem4_coeffs[12,0] = -2. * fuel_D[0] / delta_x
nem4_coeffs[12,1] = 6.* fuel_D[0] / delta_x
nem4_coeffs[12,2] = 6.* fuel_D[0] / delta_x
nem4_coeffs[12,3] = 6.* fuel_D[0] / delta_x
nem4_coeffs[12,8] = 2.* coolant_D[0] / delta_x
nem4_coeffs[12,9] = 6. * coolant_D[0] / delta_x
nem4_coeffs[12,10] = -6. * coolant_D[0] / delta_x
nem4_coeffs[12,11] = 6. * coolant_D[0] / delta_x

nem4_coeffs[13,4] = -2. * fuel_D[1] / delta_x
nem4_coeffs[13,5] = 6. * fuel_D[1] / delta_x
nem4_coeffs[13,6] = 6. * fuel_D[1] / delta_x
nem4_coeffs[13,7] = 6. * fuel_D[1] / delta_x
nem4_coeffs[13,12] = 2. * coolant_D[1] / delta_x
nem4_coeffs[13,13] = 6.* coolant_D[1] / delta_x
nem4_coeffs[13,14] = -6. * coolant_D[1] / delta_x
nem4_coeffs[13,15] = 6. * coolant_D[1] / delta_x

nem4_coeffs[14,0] = 1.
nem4_coeffs[14,1] = -1.
nem4_coeffs[14,8] = 1.
nem4_coeffs[14,9] = 1.

nem4_coeffs[15,4] = 1.
nem4_coeffs[15,5] = -1.
nem4_coeffs[15,12] = 1.
nem4_coeffs[15,13] = 1.


# Build 2nd order NEM matrix using coefficients found in Mathematica
nem2_coeffs[0,0] = -2. * fuel_D[0] / delta_x
nem2_coeffs[0,1] = -6. * fuel_D[0] / delta_x

nem2_coeffs[1,2] = -2. * fuel_D[1] / delta_x
nem2_coeffs[1,3] = -6. * fuel_D[1] / delta_x

nem2_coeffs[2,4] = -2. * coolant_D[0] / delta_x
nem2_coeffs[2,5] = 6. * coolant_D[0] / delta_x

nem2_coeffs[3,6] = -2. * coolant_D[1] / delta_x
nem2_coeffs[3,7] = 6. * coolant_D[1] / delta_x

nem2_coeffs[4,0] = -2. * fuel_D[0] / delta_x
nem2_coeffs[4,1] = 6.* fuel_D[0] / delta_x
nem2_coeffs[4,4] = 2. * coolant_D[0] / delta_x
nem2_coeffs[4,5] = 6. * coolant_D[0] / delta_x

nem2_coeffs[5,2] = -2. * fuel_D[1] / delta_x
nem2_coeffs[5,3] = 6. * fuel_D[1] / delta_x
nem2_coeffs[5,6] = 2. * fuel_D[1] / delta_x
nem2_coeffs[5,7] = 6. * fuel_D[1] / delta_x

nem2_coeffs[6,0] = 1.
nem2_coeffs[6,1] = -1.
nem2_coeffs[6,4] = 1.
nem2_coeffs[6,5] = 1.

nem2_coeffs[7,2] = 1.
nem2_coeffs[7,3] = -1.
nem2_coeffs[7,6] = 1.
nem2_coeffs[7,7] = 1.


fig = plt.figure()
plt.spy(nem4_coeffs) 
plt.title('NEM-4 Coeffs')

fig = plt.figure()
plt.spy(nem2_coeffs) 
plt.title('NEM-2 Coeffs')
plt.show()
