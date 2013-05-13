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
