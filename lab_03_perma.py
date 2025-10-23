#!/usr/bin/env python3
'''
This code calculates and plots permafrost temperature over time, as well as depth.

Collaborators: Tyler Overbeek, Alex Veal

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
    To validate the solver,
        solve_heat(validate=True)
    Figure 1 and 2:
        plot_kanger()
    Figure 3 and 4
        plot_kanger(temp_shift=0.5)
    Figure 5 and 6
        plot_kanger(temp_shift=1.0)
    Figure 7 and 8
        plot_kanger(temp_shift=3.0)
'''

import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use('fivethirtyeight')

# Kangerlussuaq average temperature:
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
10.7, 8.5, 3.1, -6.0, -12.0, -16.9])


def temp_kanger(t,temp_shift=0):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean() + temp_shift

def solve_heat(xstop=1, tstop=0.2, dx=0.2, dt=0.02, c2=1, lowerbound=0, upperbound=0, validate=False):
    '''
    A function for solving the heat equation.
    Apply Neumann boundary conditions such that dU/dx = 0.

    Parameters
    ---------
    xstop : float, default=1
        What number to stop x at.
    tstop : float, default=0.2
        What time to stop at.
    dx : float, default=0.2
        Step through x.
    dt : float, default=0.02
        Step through time.
    c2: float, defaults to 1
        c^2, the square of the diffusion coefficient.
    upperbound, lowerbound : None, scalar, or func, default=0
        Set the lower and upper boundary conditions. If either is set to
        None, then Neumann boundary condtions are used and the boundary value
        is set to be equal to its neighbor, producing zero gradient.
        Otherwise, Dirichlet conditions are used and either a scalar constant
        is provided or a function should be provided that accepts time and
        returns a value.
    validate : bool, default=False
        Set to true to validate solver.
        If True, use the validation initial condition.

    Returns
    --------
    x, t: 1D Numpy arrays
        Space and time values, respectively.
    U: Numpy array
        The solution of the heat equation, size is nSpace x nTime

    '''
    # Check our stability criterion:
    dt_max = dx**2 / (2*c2)
    if dt > dt_max:
        raise ValueError(f'DANGER: dt={dt} > dt_max = {dt_max}.')

    # Get grid sizes
    # The commented out stuff is how you round to a whole integer
    #N = np.floor(tstop / dT)
    #if tsop % dt > 0:
        #raise ValueError('Non-even number of points')
    N = int(tstop / dt) + 1
    M = int(xstop / dx) + 1

    # Set up space in time grid:
    t = np.linspace(0, tstop, N)
    x = np.linspace(0, xstop, M)

    # Create solution matrix; set initial conditions
    U = np.zeros([M, N])
    #U[:,0] = 4*x - 4*x**2

    if validate == True:
        U[:,0] = 4*x - 4*x**2
    else:
        U[:,0] = 0

    # Get our "r" coeff:
    r = c2 * (dt/dx**2)

    # Solve our equation!
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])

        # Apply boundary conditions:
        # Lower boundary
        if lowerbound is None:  # Neumann
            U[0, j+1] = U[1, j+1]
        elif callable(lowerbound):  # Dirichlet/constant
            U[0, j+1] = lowerbound(t[j+1])
        else:
            U[0, j+1] = lowerbound

        # Upper boundary
        if upperbound is None:  # Neumann
            U[-1, j+1] = U[-2, j+1]
        elif callable(upperbound):  # Dirichlet/constant
            U[-1, j+1] = upperbound(t[j+1])
        else:
            U[-1, j+1] = upperbound

    # Return our pretty solution to the caller:
    return t, x, U

def plot_kanger(xstop=100, tstop=27375, dx=1, dt=1, c2=0.0216, temp_shift=0, upperbound=5.0,validate=False, **kwargs):
    '''
    A function to plot the permafrost temperature evolution.

    Parameters
    ---------
    xstop : float, default=100
        What number to stop x at in meters.
    tstop : float, default=27375
        What time to stop at in days.
    dx : float, default=1
        Step through x in meters.
    dt : float, default=1
        Step through time in days.
    c2: float, defaults to 0.0216
        c^2, the square of the diffusion coefficient in m^2/day.
    temp_shift : float, default=0
        A temperature shift to apply to the Kangerlussuaq temperature
    upperbound : float, default=5.0
        The upper boundary condition in degrees C.
    validate : bool, default=False
        Set to true to validate solver.
        If True, use the validation initial condition.
    **kwargs : additional keyword arguments

    '''

    # Define a function to adjust temp_shift for lower bound
    def shift_lowerbound(t):
        return temp_kanger(t, temp_shift=temp_shift)

    # Get solution using your solver:
    time, x, heat = solve_heat(xstop=xstop, tstop=tstop, dx=dx, dt=dt, c2=c2,
                               lowerbound=shift_lowerbound, upperbound=upperbound, validate=validate)

    # Create a figure/axes object
    fig, axes = plt.subplots(1, 1,figsize=(12, 8))

    # Create a color map and add a color bar.
    map = axes.pcolor(time, x, heat, cmap='seismic', vmin=-25, vmax=25)
    axes.invert_yaxis()
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')
    axes.set_xlabel('Time (days)')
    axes.set_ylabel('Depth (m)')
    axes.set_title(f'Temperature Evolution in Permafrost, {temp_shift}($C$) Shift')

    dt = 1 # days

    # Set indexing for the final year of results:
    loc = int(-365/dt) # Final 365 days of the result.

    # Extract the min values over the final year:
    winter = heat[:, loc:].min(axis=1)

    # Extract the max values over the final year:
    summer = heat[:, loc:].max(axis=1)

    # Create a temp profile plot:
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    ax2.plot(winter, x, label='Winter')
    ax2.plot(summer, x, label='Summer')
    ax2.legend()
    ax2.set_xlabel('Temperature ($C$)')
    ax2.set_ylabel('Depth (m)')
    ax2.set_title(f'Minimum Annual Temperature Profile in Permafrost, {temp_shift}($C$) Shift')
    ax2.invert_yaxis()

    plt.show()
