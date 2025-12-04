#!/usr/bin/env python3
'''
Lab 5: Snowball Earth.
Create a solver that will model snowball earth.

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
    Figure 1:
        problem1()
    Figure 2:
        problem2()
    Figure 3:
        problem3()
    Figure 4:
        problem4()
'''

import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use('fivethirtyeight')

# Constants
radearth = 6357000  # Earth radius in meters
mxdlyr = 50.      # depth of mixed layer (m)
sigma = 5.67e-8     # Steffan-Boltzmann constant
C = 4.2e6           # Heat capacity of water
rho = 1020          # Density of seawater (kg/m^3)
#lam = 100           # Diffusivity of the ocean (m^2/s)

def gen_grid(npoints=18):
    '''
    Create an evenly spaced latitudinal grid with 'npoints' cell centers.
    Grid will always run from 0 to 180 as the edges of the grid. This
    means that the first grid point will be 'dlat/2' from 0 degrees and the 
    last point will be '180 - dlat/2'.

    Parameters
    ----------
    npoints : int, defaults to 18
        Number of grid points to create.
    
    Returns
    -------
    dlat : float
        Grid spacing in latitude (Degrees)
    lats : numpy array
        Locations of all grid cell centers.
    '''

    dlat = 180 / npoints  # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints)  # Lat cell centers.

    return dlat, lats

def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required.
        0 corresponds to the south pole, 180 to the north.

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Set initial temperature curve
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])

    # Get base grid:
    npoints = T_warm.size
    dlat, lats = gen_grid(npoints)

    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def temp_hot(lats_in):
    '''
    Create a temperature profile for hot earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required.
        0 corresponds to the south pole, 180 to the north.

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Set initial temperature curve
    T_hot = np.array([60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                       60, 60, 60, 60, 60, 60, 60, 60])

    # Get base grid:
    npoints = T_hot.size
    dlat, lats = gen_grid(npoints)

    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_hot, 2)

    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def temp_cold(lats_in):
    '''
    Create a temperature profile for cold earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required.
        0 corresponds to the south pole, 180 to the north.

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Set initial temperature curve
    T_cold = np.array([-60, -60, -60, -60, -60, -60, -60, -60, -60, -60,
                       -60, -60, -60, -60, -60, -60, -60, -60])

    # Get base grid:
    npoints = T_cold.size
    dlat, lats = gen_grid(npoints)

    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_cold, 2)

    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation

def snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=100., emiss=1.0,
                  init_cond = temp_warm, apply_spherecorr=False, albice=0.6,
                  albgnd=0.3, apply_insol=False, solar=1370., gamma=1.0):
    '''
    Solve the snowball earth problem.

    Parameters
    ----------
    nlat : int, defaults to 18
        Number of latitude cells.
    tfinal : int or float, defaults to 10,000
        Time length of simulation in years.
    dt : int or float, defaults to 1.0
        Size of timestep in years.
    lam : float, defaults to 100
        Set ocean diffusivity
    emiss : float, defaults to 1.0
        Set emissivity of Earth/ground.
    init_cond : function, float, or array
        Set the initial condition of the simulation. If a function is given,
        it must take latitudes as input and return temperature as a function
        of lat. Otherwise, the given values are used as-is.
    apply_spherecorr : bool, defaults to False
        Apply spherical correction term
    apply_insol : bool, defaults to False
        Apply insolation term.
    solar : float, defaults to 1370
        Set level of solar forcing in W/m2
    albice, albgnd : float, defaults to .6 and .3
        Set albedo values for ice and ground.
    gamma : float, defaults to 1.0
        Solar multiplier for solar forcing impact.

    Returns
    -------
    lats : Numpy array
        Latitudes represeting cell centers in degrees; 0 is south pole, 180 is north.
    temp : Numpy array
        Temperature as a function of latitude.

    '''

    # Set up grid:
    dlat, lats = gen_grid(nlat)
    # Y-spacing for cells in physical units:
    dy = np.pi * radearth / nlat

    # Set number of time steps:
    nsteps = int(tfinal / dt)

    # Set timestep to seconds:
    dt = dt * 365 * 24 * 3600

    # Create insolation:
    insol = gamma * insolation(solar, lats)

    # Create temp array; set our initial condition
    Temp = np.zeros(nlat)
    if callable(init_cond):
        Temp = init_cond(lats)
    else:
        Temp += init_cond

    # Create our K matrix:
    K = np.zeros((nlat, nlat))
    K[np.arange(nlat), np.arange(nlat)] = -2
    K[np.arange(nlat-1)+1, np.arange(nlat-1)] = 1
    K[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    # Boundary conditions:
    K[0, 1], K[-1, -2] = 2, 2
    # Units!
    K *= 1/dy**2

    # Create our first derivative operator.
    B = np.zeros((nlat, nlat))
    B[np.arange(nlat-1)+1, np.arange(nlat-1)] = -1
    B[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    B[0, :] = B[-1, :] = 0

    # Create area array:
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    # Get derivative of Area:
    dAxz = np.matmul(B, Axz)

    # Create L matrix.
    Linv = np.linalg.inv(np.eye(nlat) - dt * lam * K)

    # Set initial albedo.
    albedo = np.zeros(nlat)
    loc_ice = Temp <= -10  # Sea water freezes at ten below.
    albedo[loc_ice] = albice
    albedo[~loc_ice] = albgnd

    # SOLVE!
    for istep in range(nsteps):
        # Update Albedo:
        loc_ice = Temp <= -10  # Sea water freezes at ten below.
        albedo[loc_ice] = albice
        albedo[~loc_ice] = albgnd

        # Create spherical coordinates correction term
        if apply_spherecorr:
            sphercorr = (lam*dt) / (4*Axz*dy**2) * np.matmul(B, Temp) * dAxz
        else:
            sphercorr = 0

        # Apply radiative/insolation term:
        if apply_insol:
            radiative = (1-albedo)*insol - emiss*sigma*(Temp+273)**4
            Temp += dt * radiative / (rho*C*mxdlyr)

        # Advance solution.
        Temp = np.matmul(Linv, Temp + sphercorr)

    return lats, Temp

def problem1():
    '''
    Create solution figure for Problem 1 (also validate our code qualitatively)
    '''

    # Get warm Earth initial condition.
    dlat, lats = gen_grid()
    temp_init = temp_warm(lats)
    print(temp_init)

    # Get solution after 10K years for each combination of terms:
    lats, temp_diff = snowball_earth()
    lats, temp_sphe = snowball_earth(apply_spherecorr=True)
    lats, temp_alls = snowball_earth(apply_spherecorr=True, apply_insol=True,
                                     albice=.3)

    # Create a fancy plot!
    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    ax.plot(lats-90, temp_init, label='Initial Condition')
    ax.plot(lats-90, temp_diff, label='Diffusion Only')
    ax.plot(lats-90, temp_sphe, label='Diffusion + Spherical Corr.')
    ax.plot(lats-90, temp_alls, label='Diffusion + Spherical Corr. + Radiative')

    # Customize like those annoying insurance commercials
    ax.set_title('Solution after 10,000 Years')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel('Latitude')
    ax.legend(loc='best')
    plt.show()

def problem2():
    '''
    Create solution figure for Problem 2.
    '''

    # Get warm Earth initial condition.
    dlat, lats = gen_grid()
    temp_init = temp_warm(lats)

    # Solution
    lats, temp_alls = snowball_earth(lam=90, emiss=0.70, 
                                    apply_spherecorr=True, apply_insol=True)
    
    # Create a fancy plot!
    fig, ax = plt.subplots(1, 1,figsize=(10,8))
    ax.plot(lats-90, temp_init, label='Initial Condition')
    ax.plot(lats-90, temp_alls, "--",label='Diffusion + Spherical Corr. + Radiative')

    # Customize like those annoying insurance commercials
    ax.set_title('Warm-Earth Equilibrium after 10,000 Years')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel('Latitude')
    ax.legend(loc='best')
    plt.show()

def problem3():
    '''
    Create solution figure for Problem 3.
    '''

    # Get warm Earth initial condition.
    dlat, lats = gen_grid()
    temp_init = temp_warm(lats)

    # Solution
    lats, temp_alls = snowball_earth(lam=90, emiss=0.70, 
                                    apply_spherecorr=True, albgnd=0.6, apply_insol=True)
    lats, cool_temp = snowball_earth(init_cond=temp_cold, lam=90, emiss=0.70,
                                    apply_spherecorr=True, apply_insol=True)
    lats, hot_temp = snowball_earth(init_cond=temp_hot, lam=90, emiss=0.70,
                                    apply_spherecorr=True, apply_insol=True)
    
    print(temp_alls)
    
    # Create a fancy plot!
    fig, ax = plt.subplots(1, 1,figsize=(10,8))
    ax.plot(lats-90, temp_init, label='Warm Conditions',color='black')
    ax.plot(lats-90, cool_temp, label='Cold Conditions',color='blue')
    ax.plot(lats-90, hot_temp, label='Hot Conditions', color='red')
    ax.plot(lats-90, temp_alls, label='Flash Freeze Conditions', color='green')

    # Customize like those annoying insurance commercials
    ax.set_title('Hot and Cold Earth Equilibrium after 10,000 Years')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel('Latitude')
    ax.legend(loc='best')
    plt.show()

def problem4():
    '''
    Create solution for problem 4.
    '''

    # Get cold earth initial condition.
    dlat, lats = gen_grid()
    temp_init = temp_cold(lats)

    # Create ranges for solar forcing
    sol_mult_up = np.arange(0.4, 1.40, 0.05)
    sol_mult_down = np.arange(1.4, 0.4, -0.05)

    # Create arrays to store solutions:
    temp_up = []
    temp_down = []

    # Create arrays to store avg temp:
    avg_temp_up = []
    avg_temp_down = []

    # Set initial condition going up:
    init_cond = temp_cold(lats)
    temp_up.append(init_cond)
    avg_temp_up.append(np.mean(init_cond))

    # solve up:
    for x in sol_mult_up[1:]:
        lats, init_cond = snowball_earth(init_cond=init_cond, lam=90, emiss=0.70,
                                    apply_spherecorr=True, apply_insol=True, gamma=x)
        temp_up.append(init_cond)
        avg_temp_up.append(np.mean(init_cond))

    # Set initial condition going down:
    init_cond = temp_cold(lats)
    temp_down.append(init_cond)
    avg_temp_down.append(np.mean(init_cond))

    # solve down:
    for x in sol_mult_down[1:]:
        lats, init_cond = snowball_earth(init_cond=init_cond, lam=90, emiss=0.70,
                                    apply_spherecorr=True, apply_insol=True, gamma=x)
        temp_down.append(init_cond)
        avg_temp_down.append(np.mean(init_cond))

    # Create a fancy plot!
    fig, ax = plt.subplots(1, 1,figsize=(10,8))
    ax.plot(sol_mult_up, avg_temp_up, label='Increasing Solar Multiplier',color='red')
    ax.plot(sol_mult_down, avg_temp_down, label='Decreasing Solar Multiplier',color='blue')

    # Customize like those annoying insurance commercials
    ax.set_title('Cold Earth Equilibrium after 10,000 Years With Solar Multiplier')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel('Solar Multiplier')
    ax.legend(loc='best')
    plt.show()


def test_functions():
    '''
    Test functions
    '''

    print('Test gen_geid')
    print('For npoints=5')

    dlat_correct, lats_correct = (36.0, np.array([18., 54., 90., 126., 162.]))
    result = gen_grid(5)

    if (result[0] == dlat_correct) and np.all(result[1] == lats_correct):
        print('\tPassed!')
    else:
        print('\tFailed!')
        print(f"expected: {(dlat_correct, lats_correct)}")
        print(f"Got: {gen_grid(5)}")