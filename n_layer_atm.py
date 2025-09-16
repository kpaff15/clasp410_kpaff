#!/usr/bin/env python3
'''
This code allows us to determine the temperature of N atmosphere layers.

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
<tbd>
'''

import numpy as np
import matplotlib.pyplot as plt

# Formulas for reference
# Te = 4th root(((1-a)*s0)/2*sigma)
# Ta = 4th root(((1-a)*s0)/4*sigma)

# Te is temperature of Earth's surface
# Ta is temperature of Atmosphere

# Declare constants
sigma = 5.67E-8 # Units: W/m2/K-4

def n_layer_atmos(nlayers, epsilon=1, albedo=0.33, s0=1350):
    '''
    Doc string

        Parameters
    ----------

    Returns
    -------

    '''

    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # Populate based on our model:
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            A[i, j] = # What math should go here?
            
    b = # What should go here?

    # Invert matrix:
    Ainv = np.linalg.inv(A)
    # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!