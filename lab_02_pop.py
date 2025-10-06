#!/usr/bin/env python3
'''
This code calculates and plots the Lotka-Volterra competition and predator-prey
equations using Euler's method and Scipy's ODE class and the adaptive step 8th order solver.
It also plots phase diagrams of the two species for each model.

Collaborators: Tyler Overbeek, Alex Veal

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:

* Run the script in the terminal! *
Figure 2:
    comp_pred_plot(0.3,0.6,1,2,1,3)
Figure 3:
    comp_pred_plot(0.3,0.6,1,2,1,3,0.1,0.01)
Figure 4:
    comp_pred_plot(0,1.0,1,3,1,4,0.1)
Figure 5:
    comp_pred_plot(1,0,1,3,1,4,0.1)
Figure 6:
    comp_pred_plot(0,0,1,3,1,4,0.1)
To obtain equilibrium listed in the report:
    equi_state()
Figure 7:
    comp_pred_plot(0.2,0.4,1,2,1,3)
Figure 8:
    comp_pred_plot(0.3,0.6,2,1,1,1)
Figure 9:
    phase_plot(0.3,0.6,1,2,1,3)
Figure 10: WARNING This will compute slowly
    phase_plot(0.3,0.6,1,2,1,3,0.001)
Figure 11: WARNING This will compute slowly
    phase_plot(0.3,0.6,2,1,1,1,0.001)
Figure 12:
    phase_plot(0.3,0.6,1,3,4,1,0.01)
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-poster')

# Function for competition equations
def dNdt_comp(t, N, a=1, b=2, c=1, d=3):

    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
    The current time (not used here).
    N : two-element list
    The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
    The value of the Lotka-Volterra coefficients.

    Returns
    -------
    dN1dt, dN2dt : floats
    The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1] # Eq 1
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0] # Eq 2
    return dN1dt, dN2dt

# Function for Euler's method
def euler_solve(func, N1_init=.3, N2_init=.6, dT=1.0, t_final=100.0):

    '''
    This function solves the Lotka-Volterra competition equations using the 
    Euler method.

    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2. (use dNdt_comp)
    N1_init : float, default=0.3
        Prey initial condition.
    N2_init : float, default=0.6
        Predator initial condition.
    dT : float, default=1.0
        The timestep in years.
    t_final : float, default=100.0
        Final time in years.
    
    Returns
    ---------
    time : Numpy array
        Array of years.
    N1, N2 : Numpy arrays
        Predator and prey population arrays.
    
    '''
    # N1
    time = np.arange(0, t_final, dT) # creating an array of times
    N1 = np.zeros(time.size) # creates array of zeros the size of time
    N1[0] = N1_init # Set first row of N1 to the initial

    # N2
    N2 = np.zeros(time.size) # creates array of zeros the size of time
    N2[0] = N2_init # Set first row of N2 to the initial

    # Eulers equations in a loop
    for i in range(1, time.size):
        dN1, dN2 = func(i, [N1[i-1], N2[i-1]] )
        N1[i] = N1[i-1] + dT * dN1
        N2[i] = N2[i-1] + dT * dN2

    return time, N1, N2

# Function for DOP853
def solve_rk8(func, N1_init=.3, N2_init=.6, dT=1.0, t_final=100.0,
a=1, b=2, c=1, d=3):

    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.
    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init, N2_init : float, defaults to 0.3 and 0.6 respectively
        Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=1.0
        Largest timestep allowed in years.
    t_final : float, default=100
        Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values
    Returns
    -------
    time : Numpy array
        Time elapsed in years.
    N1, N2 : Numpy arrays
        Normalized population density solutions.
    '''
    from scipy.integrate import solve_ivp

    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                        method='DOP853', max_step=dT)

    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]

    # Return values to caller.
    return time, N1, N2

# Function for pred/prey
def dNdt_pred(t, N, a=1, b=2, c=1, d=3):

    '''
    This function calculates the Lotka-Volterra predator-prey equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
    The current time (not used here).
    N : two-element list
    The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
    The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
    The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0] - b*N[0]*N[1] # Eq 5
    dN2dt = -c*N[1] + d*N[1]*N[0] # Eq 6
    return dN1dt, dN2dt

# Create the competition and pred-prey plots
def comp_pred_plot(N1_init, N2_init, a, b, c, d, dTc=1.0, dTp=0.05, t_final=100.0):
    
    '''
    Plot the results of the Lotka-Volterra competition equations
    solved using both Euler's method and Scipy's ODE solver.
    Parameters
    ----------
    N1_init, N2_init : float
        Initial conditions for `N1` and `N2`, ranging from (0,1]
    dTc : float, default=1.0
        Largest timestep allowed in years for competition.
    dTp : float, default=0.05
        Largest timestep allowed in years for predator-prey.
    t_final : float, default=100
        Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values
    Returns
    -------
    Competition Plot
    '''
    # Competition variables
    # Store euler returns in variables
    compeu_time, compeu_N1, compeu_N2 = euler_solve(
        lambda t, N: dNdt_comp(t, N, a, b, c, d), N1_init, N2_init, dTc, t_final)

    # Store rk8 returns in variables
    comprk_time, comprk_N1, comprk_N2 = solve_rk8(
        lambda t, N: dNdt_comp(t,N,a,b,c,d), N1_init, N2_init, dTc, t_final)

    # Predator-Prey variables
    # Store euler returns in variables
    predeu_time, predeu_N1, predeu_N2 = euler_solve(
        lambda t, N: dNdt_pred(t,N,a,b,c,d),N1_init, N2_init, dTp, t_final)
    # Store rk8 returns in variables
    predrk_time, predrk_N1, predrk_N2 = solve_rk8(
        lambda t, N: dNdt_pred(t,N,a,b,c,d),N1_init, N2_init, dTp, t_final)

    # Create figure
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,7))

    # Predator-Prey plots
    # Plot Euler
    ax2.plot(predeu_time, predeu_N1,label='Bunnies (Euler)', color ='blue')
    ax2.plot(predeu_time, predeu_N2,label='Wolves (Euler)', color='red')

    # plot rk8
    ax2.plot(predrk_time, predrk_N1, '--',label='Bunnies (RK8)', color='blue')
    ax2.plot(predrk_time, predrk_N2,'--',label='Wolves (RK8)', color='red')

    ax2.set_xlabel('Time (Years)')
    ax2.set_ylabel('Population/Carrying Capacity')
    ax2.set_title('Lotka-Volterra Predator-Prey Model')
    ax2.legend()
    ax2.grid(True)
    
    # Competition Plots
    # Plot Euler method
    ax1.plot(compeu_time, compeu_N1,label='Bunnies (Euler)', color ='blue')
    ax1.plot(compeu_time, compeu_N2,label='Wolves (Euler)', color='red')  
    # Plot RK8 method
    ax1.plot(comprk_time, comprk_N1, '--',label='Bunnies (RK8)', color='blue')
    ax1.plot(comprk_time, comprk_N2,'--',label='Wolves (RK8)', color='red')   
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Population/Carrying Capacity')
    ax1.set_title('Lotka-Volterra Competition Model')
    ax1.legend()
    ax1.grid(True)

    plt.show()

# def pred_plot(N1_init, N2_init, a, b, c, d, dT=0.05, t_final=100.0):
#    
#    '''
#    Plot the results of the Lotka-Volterra predator-prey equations
#    solved using both Euler's method and Scipy's ODE solver.
#    Parameters
#    ----------
#    N1_init, N2_init : float
#        Initial conditions for `N1` and `N2`, ranging from (0,1]
#    dT : float, default=0.05
#        Largest timestep allowed in years.
#    t_final : float, default=100
#        Integrate until this value is reached, in years.
#    a, b, c, d : float, default=1, 2, 1, 3
#        Lotka-Volterra coefficient values
#    Returns
#    -------
#    Predator Prey Plot
#    '''
#    # Store euler returns in variables
#    predeu_time, predeu_N1, predeu_N2 = euler_solve(
#        lambda t, N: dNdt_pred(t,N,a,b,c,d),N1_init, N2_init, dT, t_final)
#    # Store rk8 returns in variables
#    predrk_time, predrk_N1, predrk_N2 = solve_rk8(
#        lambda t, N: dNdt_pred(t,N,a,b,c,d),N1_init, N2_init, dT, t_final)
#
#    # Create figure
#    fig, ax = plt.subplots(1, 1, figsize=(14,10))
#
#    # Plot Euler
#    ax.plot(predeu_time, predeu_N1,label='Bunnies (Euler)', color ='blue')
#    ax.plot(predeu_time, predeu_N2,label='Wolves (Euler)', color='red')
#
#    # plot rk8
#    ax.plot(predrk_time, predrk_N1, '--',label='Bunnies (RK8)', color='blue')
#    ax.plot(predrk_time, predrk_N2,'--',label='Wolves (RK8)', color='red')
#
#    ax.set_xlabel('Time (Years)')
#    ax.set_ylabel('Population/Carrying Capacity')
#    ax.set_title('Lotka-Volterra Predator-Prey Model')
#    ax.legend()
#    ax.grid(True)
#    plt.show()
    
# Create phase plot
def phase_plot(N1_init, N2_init, a, b, c, d, dT=0.05, t_final=100.0):
    
    '''
    Plot the phase diagram of the Lotka-Volterra predator-prey equations.
    Prey on the x and predator on the y.
    Parameters
    ----------
    N1_init, N2_init : float
        Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=0.05
        Largest timestep allowed in years.
    t_final : float, default=100
        Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values
    Returns
    -------
    Phase Diagram Plot of predator vs prey
    '''
    # Return values from Euler into variables
    predeu_time, predeu_N1, predeu_N2 = euler_solve(
        lambda t, N: dNdt_pred(t,N,a,b,c,d),N1_init,N2_init,dT, t_final)
    # Return values from rk8 into variables
    predrk_time, predrk_N1, predrk_N2 = solve_rk8(
        lambda t, N: dNdt_pred(t,N,a,b,c,d),N1_init,N2_init,dT, t_final)

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,7))

    # Phase Plot
    # Plot euler and rk8 pred vs prey
    ax2.plot(predeu_N1, predeu_N2,label='Euler', color ='blue')
    ax2.plot(predrk_N1, predrk_N2,label='RK8', color='red')

    ax2.set_xlabel('Prey Species')
    ax2.set_ylabel('Predator Species')
    ax2.set_title('Predator-Prey Phase Diagrams')
    ax2.legend()
    ax2.grid(True)
    
    # Predator-Prey plots
    # Plot Euler
    ax1.plot(predeu_time, predeu_N1,label='Bunnies (Euler)', color ='blue')
    ax1.plot(predeu_time, predeu_N2,label='Wolves (Euler)', color='red')

    # plot rk8
    ax1.plot(predrk_time, predrk_N1, '--',label='Bunnies (RK8)', color='blue')
    ax1.plot(predrk_time, predrk_N2,'--',label='Wolves (RK8)', color='red')

    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Population/Carrying Capacity')
    ax1.set_title('Lotka-Volterra Predator-Prey Model')
    ax1.legend()
    ax1.grid(True)

    plt.show()

# Solve for equilibrium
def equi_state(a=1, b=2, c=1, d=3):
    
    '''
    Calculate the equilibrium states for the Lotka-Volterra competition equations.
    Parameters
    ----------
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    Returns
    -------
    N1_eq, N2_eq : floats
        The equilibrium states of `N1` and `N2`.
    '''
    N1_eq = (c * (a - b)) / (c*a - b*d) # Eq 3
    N2_eq = (a * (c - d)) / ((c*a - b*d)) # Eq 4

    return N1_eq, N2_eq