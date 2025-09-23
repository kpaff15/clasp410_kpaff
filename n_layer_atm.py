#!/usr/bin/env python3
'''
This code allows us to determine the temperature of N atmosphere layers.

Collaborators: Alex Veal, Tyler Overbeek

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:

* Run the script in the terminal! *

Question 2:
    n_layer_atmos(3)
Question 3 part 1 (fig 1):
    n_layer_plot(1)
Question 3 part 2 (fig 2):
    n_layer_alt_plot()
Question 4:
    n_layer_atmos(51,1,0.6,2600)
Question 5 (fig 3):
    nuke_at_plot()
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-poster')

# Formulas for reference
# Te = 4th root(((1-a)*s0)/2*sigma)
# Ta = 4th root(((1-a)*s0)/4*sigma)

# Te is temperature of Earth's surface
# Ta is temperature of Atmosphere

# Declare constants
sigma = 5.67E-8 # Units: W/m2/K-4

# Q2: Write a function
# The Model:
def n_layer_atmos(nlayers, epsilon=1.0, albedo=0.33, s0=1350, debug=False):
    '''
    This function allows us to calculate the temperature for N layers of the atmosphere.

    Parameters
    ----------
    nlayers: float
        The number of layers being used to calculate the temperature in Kelvin.
    epsilon: float, defaults to 1.0
        Emissivity value of the atmosphere.
    albedo: float, defaults to 0.33
        Reflectivity value of the earth.
    s0: float, defaults to 1350
        Solar Constant in W/m^2.
    debug: boolean, defaults to False
        When True, it will print the current A matrix.

    Returns
    ----------
    temp: Numpy array
        Fluxes converted into temperature in Kelvin.

    '''

    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1]) # nlayers is the number of layers input into the function
    b = np.zeros(nlayers+1)

    # Populate based on our model:
    for i in range(nlayers+1): # i is for the rows
        for j in range(nlayers+1): # j is for the columns

            if i==j: # put us on the diagonal
                A[i,j] = -2 +(j==0) # diagonals 
            
            else: 
                A[i,j] = (epsilon**(i>0)) * ((1-epsilon)**(np.abs(j - i) - 1))
                # ((1-epsilon)**(np.abs(j - i) - 1)) this handles all other values off diagonal

            if debug:
                print(f'A[i={i},j={j}] = {A[i,j]}')
                print(A)

    b[0] = -s0/4 * (1 - albedo) # solar flux

    # Invert matrix:                            
    A_inv = np.linalg.inv(A)
    
    # Get Flux solution:
    fluxes = np.matmul(A_inv, b)

    # Get Temp Solution:
    temp = np.power((fluxes / sigma / epsilon), 1/4)
    temp[0] = np.power((fluxes[0] / sigma), 1/4) # ground flux, epsilon = 1.

    return temp

# Prepare for Q3:

# Q3 pt1: Plot temp vs emissivity for single layer atm
# Create array for range of emissivities
ems = np.arange(0, 1.01, 0.1)

def n_layer_plot(nlayers, ems=ems, albedo=0.33, s0=1350):
    '''
    This function creates a figure that will show Layer temp (K) versus Emissivity (epsilon)
     
     Parameters
    ----------
    nlayers: float
        The number of layers being used to calculate the temperature in Kelvin.
    ems: float, defaults to range ems
        Range of emissivities from 0 to 1 in 0.1 incerements.
    albedo: float, defaults to 0.33
        Reflectivity value of the earth.
    s0: float, defaults to 1350
        Solar Constant in W/m^2.
    Returns
    -------
    figure depicting Layer temp (K) versus Emissivity (epsilon)

    '''
    # Create empty plot
    fig, ax = plt.subplots(figsize=(10,8))

    # Blank array to store each temp for each emissivity
    temps = []

    # Run through each emissiity and get the temp for each.
    # Store in temps array
    for x in ems:
        nlayer_temps = n_layer_atmos(nlayers,epsilon = x)
        temps.append(nlayer_temps[0]) # Add temps to each emissivity into temps array.

    # Quick debug    
    print(temps)
        
    # Plot statements
    ax.plot(ems, temps, color = 'red')
    ax.set_title('Emissivity vs Temperature for A Single Layer Atmosphere')
    ax.set_xlabel('Emissivity')
    ax.set_ylabel('Temperature (K)')
    ax.grid(True)
    
    return fig

# Q3 Plot 2 (Using nlayers = 5 ~287 K at epsilon = 0.255)
# Plot 2 is how we are answering, what does model predict for emissivity at 288 K?

def n_layer_alt_plot(nlayers = 5, ems=0.255, albedo=0.33, s0=1350):
    '''
    This function creates a figure that will show Layer temp (K) versus Altitude (km)
     
     Parameters
    ----------
    nlayers: float, defaults to 5
        The number of layers being used to calculate the temperature in Kelvin.
        Set to 5 because that temperature is ~287 K
    ems: float, defaults to 0.255
        Emissivity value of the atmosphere.
    albedo: float, defaults to 0.33
        Reflectivity value of the earth.
    s0: float, defaults to 1350
        Solar Constant in W/m^2.
    Returns
    -------
    figure depicting Layer temp (K) versus Altitude (km)

    '''
    # Create empty plot
    fig, ax = plt.subplots(figsize=(10,8))

   # Create array for range of altitudes
    alt = np.arange(0, (nlayers +1) * 10, 10) # Kilometers. Assuming layer thickeness = 10 km
    alt_temps = n_layer_atmos(nlayers,epsilon = ems)
    # Plot statements
    ax.plot(alt_temps, alt, color = 'red')
    ax.set_title('Temperature Vs Altitude for A 5 Layer Atmosphere And Epsilon of 0.255')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Altitude (km)')
    ax.grid(True)
    
    return fig
# Q4: Use model to answer,
# How many atmospheric layers do we expect on Venus?
# Venus surface temp = 700 K
# Venus solar flux = 2600 W/m2
# N layers > 1
# Emissivity = 1
# albedo = 0.6 per https://cneos.jpl.nasa.gov/glossary/albedo.html

# At 51 layers, Venus has a surface temp of ~699 K

# Q5
def nuke_atmos(nlayers, epsilon=1.0, albedo=0, s0=1350, debug=False):
    '''
    Parameters
    ----------
    nlayers: float
        The number of layers being used to calculate the temperature in Kelvin.
    epsilon: float, defaults to 1.0
        Emissivity value of the atmosphere.
    albedo: float, defaults to 0
        Reflectivity value of the earth.
    s0: float, defaults to 1350
        Solar Constant in W/m^2.
    debug: boolean, defaults to False
        When True, it will print the current A matrix.
    Returns
    ----------
    temp: Numpy array
        Fluxes converted into temperature in Kelvin.

    '''

    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1]) # nlayers is the number of layers input into the function
    b = np.zeros(nlayers+1)

    # Populate based on our model:
    for i in range(nlayers+1): # i is for the rows
        for j in range(nlayers+1): # j is for the columns

            if i==j: # put us on the diagonal
                A[i,j] = -2 +(j==0) # diagonals 
            
            else: 
                A[i,j] = (epsilon**(i>0)) * ((1-epsilon)**(np.abs(j - i) - 1))
                # ((1-epsilon)**(np.abs(j - i) - 1)) this handles all other values off diagonal

            if debug:
                print(f'A[i={i},j={j}] = {A[i,j]}')
                print(A)

    b[0] = 0 # surface
    b[-1] = -s0/4 * (1 - albedo) # solar flux top layer. -1 pulls the last row

    # Invert matrix:                            
    A_inv = np.linalg.inv(A)
    
    # Get Flux solution:
    fluxes = np.matmul(A_inv, b)

    # Get Temp Solution:
    temp = np.power((fluxes / sigma / epsilon), 1/4)
    temp[0] = np.power((fluxes[0] / sigma), 1/4) # ground flux, epsilon = 1.

    return temp

# Q5 Plot
# albedo should be zero because flux is completely absorbed by top layer
# nlayers = 5
def nuke_alt_plot(nlayers = 5, ems=0.5, albedo=0, s0=1350):
    '''
    This function creates a figure that will show Layer temp (K) versus Emissivity (epsilon)
     
     Parameters
    ----------
    nlayers: float, defaults to 5
        The number of layers being used to calculate the temperature in Kelvin.
    ems: float, defaults to 0.5
        Emissivity value of the atmosphere.
    albedo: float, defaults to 0
        Reflectivity value of the earth.
        Set to 0 because top layer is absorbing all solar flux.
    s0: float, defaults to 1350
        Solar Constant in W/m^2.
    Returns
    -------
    figure depicting Layer temp (K) versus Altitude (km))

    '''
    # Create empty plot
    fig, ax = plt.subplots(figsize=(10,8))

   # Create array for range of altitudes
    nuke = np.arange(0, (nlayers +1) * 10, 10) # Kilometers. Assuming layer thickeness = 10 km
    nuke_temps = nuke_atmos(nlayers,epsilon = ems)
    # Plot statements
    ax.plot(nuke_temps, nuke, color = 'red')
    ax.set_title('Temperature Vs Altitude for A 5 Layer Atmosphere After A Nuclear Winter') # Make sure these get changed!!!!
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Altitude (km)')
    ax.grid(True)
    
    return fig