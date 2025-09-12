#!/usr/bin/env python3
'''
Test hypothesis that global warming is fully explainable by an increase
in solar forcing using a model and code.
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

# Formulas for reference
# Te = 4th root(((1-a)*s0)/2*sigma)
# Ta = 4th root(((1-a)*s0)/4*sigma)

# Te is temperature of Earth's surface
# Ta is temperature of Atmosphere

# Copy arrays from powerpoint slide
year = np.array([1900, 1950, 2000])
s0 = np.array([1365, 1366.5, 1368])
t_anom = np.array([-.4, 0, .4])

# Set constants from powerpoint slide...
sigma = 5.67E-8 # only sigma is constant because albedo changes

# Create a function for 1 layer Te
def temp_layer1(s0 = s0, a = 0.33):
    '''
    This function returns surface temperature of earth in 1 layer atmosphere.

        Parameters
    ----------
    s0: Numpy array
        An array of solar forcing
    a: floating point, defaults to 0.33
        Albedo value of earth's surface

    Returns
    -------
    te: numpy array
        Temperature of earth's surface

    '''

    te = (((1-a) * s0) / (2 * sigma)) **(1/4)

    return te

# Figure of temp over time
def temp_fig():
    '''
    This function creates a figure that will show surface temp (K) and temperature anomalies over time (Year)
     
     Parameters
    ----------
    Returns
    -------
    figure depicting surface temp (K) and temperature anomalies over time (Year)

    '''

    predicted_te = temp_layer1()
    actual_te = predicted_te[1] + t_anom # Got stuck here and had to peek at L2 Pt 2

    fig, ax = plt.subplots(1, 1, figsize = (10,8))

    ax.plot(year, predicted_te, label="Predicted Temperature")
    ax.plot(year, actual_te, label="Actual Temperature")
    ax.set_title('Does Solar Forcing Account For Climate Change Temperature Anomaly?')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature (K)')
    #ax.grid(True)
    ax.legend(loc='best')