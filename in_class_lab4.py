#!/usr/bin/env python3
'''
The purpose of this code is to explore how fires/infectious diseases can spread utilizing a probalistic solver.

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:

Question 1:
    Create forest for question 1 part 1 (3x3 grid).
        forest = solver(isize=3,jsize=3,nstep=4,pspread=1,pstart=0,pbare=0.0,pfatal=0.0)
    Figure 1:
        plot_environ2d(forest,0)
    Figure 2:
        plot_environ2d(forest,1)
    Figure 3:
        plot_environ2d(forest,2)
    
    Create a forest for question 1 part 2 (10x6 grid).
        forest = solver(isize=6,jsize=10,nstep=4,pspread=1,pstart=0,pbare=0.0,pfatal=0.0)
    Figure 4:
        plot_environ2d(forest,0)
    Figure 5:
        plot_environ2d(forest,1)
    Figure 6:
        plot_environ2d(forest,2)
Question 2:
    Run experiment 1: Vary Pspread from 0-1.
    Figure 7: forest = solver(isize=10,jsize=10,nstep=20,pspread=0.0,pstart=0,pbare=0.0,pfatal=0.0)
        plot_progression(forest)
    Figure 8: forest = solver(isize=10,jsize=10,nstep=20,pspread=0.5,pstart=0,pbare=0.0,pfatal=0.0)
        plot_progression(forest)
    Figure 9: forest = solver(isize=10,jsize=10,nstep=20,pspread=1.0,pstart=0,pbare=0.0,pfatal=0.0)
        plot_progression(forest)

    Run experiment 2: Vary Pbare from 0-100 and pstart
    Figure 10: forest = solver(isize=10,jsize=10,nstep=20,pspread=1.0,pstart=0.25,pbare=0.0,pfatal=0.0)
        plot_progression(forest)
    Figure 11: forest = solver(isize=10,jsize=10,nstep=20,pspread=1.0,pstart=0.25,pbare=0.5,pfatal=0.0)
        plot_progression(forest)
    Figure 12: forest = solver(isize=10,jsize=10,nstep=20,pspread=1.0,pstart=0.25,pbare=1.0,pfatal=0.0)
        plot_progression(forest)
Question 3:
    # Run this before Figs 13-15:
        population = solver(isize=10,jsize=10,nstep=20,pspread=0.75,pstart=0.0,pbare=0.0,pfatal=0.25)
    Figure 13:
        plot_progression(population,healthy_lbl = 'Healthy People',bare_lbl='Infected People',case='Disease')
    Figure 14:
        plot_environ2d(population,itime=0, lbl_0='Deceased', lbl_1='Immune', lbl_2 = 'Healhty', lbl_3='Infected')
    Figure 15:
        plot_environ2d(population,itime=12, lbl_0='Deceased', lbl_1='Immune', lbl_2 = 'Healhty', lbl_3='Infected')
    
    # Run this before Figs 16-18:
        population = solver(isize=10,jsize=10,nstep=20,pspread=0.25,pstart=0.0,pbare=0.0,pfatal=0.75)
    Figure 16:
        plot_progression(population,healthy_lbl = 'Healthy People',bare_lbl='Infected People',case='Disease')
    Figure 17:
        plot_environ2d(population,itime=0, lbl_0='Deceased', lbl_1='Immune', lbl_2 = 'Healhty', lbl_3='Infected')
    Figure 18:
        plot_environ2d(population,itime=3, lbl_0='Deceased', lbl_1='Immune', lbl_2 = 'Healhty', lbl_3='Infected')
    
    # Run this before Figs 19-21:
        population = solver(isize=10,jsize=10,nstep=20,pspread=0.75,pstart=0.50,pbare=0.25,pfatal=0.25)
    Figure 19:
        plot_progression(population,healthy_lbl = 'Healthy People',bare_lbl='Infected People',case='Disease')
    Figure 20:
        plot_environ2d(population,itime=0, lbl_0='Deceased', lbl_1='Immune', lbl_2 = 'Healhty', lbl_3='Infected')
    Figure 21:
        plot_environ2d(population,itime=3, lbl_0='Deceased', lbl_1='Immune', lbl_2 = 'Healhty', lbl_3='Infected')
    
    # Run this before Figs 22-24:
        population = solver(isize=10,jsize=10,nstep=20,pspread=0.25,pstart=0.25,pbare=0.75,pfatal=0.75)
    Figure 22:
        plot_progression(population,healthy_lbl = 'Healthy People',bare_lbl='Infected People',case='Disease')
    Figure 23:
        plot_environ2d(population,itime=0, lbl_0='Deceased', lbl_1='Immune', lbl_2 = 'Healhty', lbl_3='Infected')
    Figure 24:
        plot_environ2d(population,itime=2, lbl_0='Deceased', lbl_1='Immune', lbl_2 = 'Healhty', lbl_3='Infected')
    
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from matplotlib.colors import ListedColormap

# Set plot style:
plt.style.use('fivethirtyeight')

colors = ['darkviolet','tan','forestgreen','crimson']
environ_cmap = ListedColormap(colors)

# Function that can start a fire/disease randomly and spread.
def solver(isize=3, jsize=3, nstep=4, pspread=1.0, pstart=0.0, pbare=0.0, pfatal=0.0):
    '''
    Create a forest fire/disease.

    Parameters
    ----------
    isize, jsize : int, defaults to 3
        Set size of forest in x and y directions, respectively.
    nstep : int, defaults to 4
        Set number of steps to advance solution.
    pspread : float, defaults to 1.0
        Set chance that fire/disease can spread in any direction, from 0 to 1.
        (i.e., 0% chance to spread, 100% chance to spread)
    pstart : float, defaults to 0.0
        Set the chance that a point starts the simulation on fire (or infected)
        from 0 to 1 (0% to 100%).
    pbare : float, defaults to 0.0
        Set the chance that a point starts the simulation as bare land (or immune)
        from 0 to 1 (0% to 100%).
    pfatal : float, defaults to 0.0
        Set the chance that a point starts the simulation as deceased.
        from 0 to 1 (0% to 100%).
    '''

    # create an environment and making all spots have trees/healthy people.
    environ = np.zeros((nstep, isize, jsize))+2

    # Set initial conditions for BURNING/INFECTED and BARE/IMMUNE:
    # Start with BURNING/INFECTED:
    if pstart > 0.0: # Scatter fire/disease randomly
        loc_start = np.zeros((isize, jsize), dtype=bool)
        while loc_start.sum() == 0:
            loc_start = rand(isize, jsize) <= pstart
        print(f"Starting with {loc_start.sum()} points on fire or infected.")
        environ[0, loc_start] = 3
    else:
        # Set initial fire/disease to center:
        environ[0, isize//2, jsize//2] = 3

    # Set bare land/immune people:
    loc_bare = rand(isize, jsize) <= pbare
    environ[0, loc_bare] = 1

    # Loop through time to advance the fire/disease.
    for k in range(nstep-1):
        # Assume the next time step is the same as the current:
        environ[k+1, :, :] = environ[k, :, :]
        # Search every spot that is on fire/diseased and spread fire/disease as needed.
        for i in range(isize):
            for j in range(jsize):
                # Are we on fire/diseased?
                if environ[k, i, j] != 3:
                    continue
                # Ah! it burns. Spread fire/disease in each direction.
                # Spread "up" (i to i-1)
                if (pspread > rand()) and (i>0) and (environ[k, i-1, j] == 2):
                    environ[k+1, i-1, j] = 3
                # Spread "Down" (i to i+1)
                if (pspread > rand()) and (i < isize-1) and (environ[k, i+1, j] == 2):
                    environ[k+1, i+1, j] = 3
                # Spread "East" (j to j-1)
                if (pspread > rand()) and (j>0) and (environ[k, i, j-1] == 2):
                    environ[k+1, i, j-1] = 3
                # Spread "West" (j to j+1)
                if (pspread > rand()) and (j < jsize-1) and (environ[k, i, j+1] == 2):
                    environ[k+1, i, j+1] = 3
                # But did you die?
                if (pfatal > rand()):
                    environ[k+1, i, j] = 0
                    continue
                # Change burning/diseased to burnt/infected:
                environ[k+1, i, j] = 1
                

    return environ

# Function to plot fire/disease over time.
def plot_progression(environ, healthy_lbl='Forested', bare_lbl='Bare/Burnt',case='Forest'):
    '''
    Calculate the time dynamics of a forest fire/disease and plot them.

    Parameters
    ----------
    environ : array
        Pass through environment array generated from solver().
    healthy_lbl : str, defaults to 'Forested'
        Set label for healthy plot.
    bare_lbl : str, defaults to 'Bare/Burnt'
        Set label for Bare/Infected plot.
    case : str, defaults to 'Forest'
        Allows user to set what case is being evaluated into a variable.
    '''
    # Get total number of points:
    ksize, isize, jsize = environ.shape
    npoints = isize * jsize
    
    # Find all spots that have forests (or are healthy people)
    # ... and count them as a function of time
    loc = environ == 2
    healthy = 100 * loc.sum(axis=(1,2)) / npoints

    loc = environ == 1
    bare = 100 * loc.sum(axis=(1,2)) / npoints

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize = (10,9))

    ax.plot(healthy,label= healthy_lbl)
    ax.plot(bare, label = bare_lbl)
    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel(f'Percent Total {case}')
    ax.legend(loc='best')
    ax.set_title(f'Time Progression of {healthy_lbl} vs {bare_lbl}')

# Function to plot grid plots of fire for each time step
def plot_environ2d(environ_in, itime=0, lbl_0='Deceased', lbl_1='Bare/Burnt', lbl_2='Forested', lbl_3='Burning'): # Make sure you set environ = solver() in terminal before plotting
    '''
    Given an environment of size (ntime, nx, ny), plot the itime-th moment
    as a 2D pcolor plot

    Parameters
    ----------
    environ_in : array
        Pass through environment array generated from solver().
    itime : int, defaults to 0
        Set time for timestep (ntime)
    lbl_0 : str, defaults to 'Deceased'
        Set label for status value "0".
    lbl_1 : str, defaults to 'Bare/Burnt'
        Set label for status value "1".
    lbl_2 : str, defaults to 'Forested'
        Set label for status value "2".
    lbl_3 : str, defaults to 'Burning'
        Set label for status value "3".
    '''

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize = (10,9))

    # Add our pcolor plot, save the resulting mappable object.
    map = ax.pcolor(environ_in[itime, :, :], vmin=0, vmax=3, cmap=environ_cmap)

    # Add a colorbar by handing our mappable to the colorbar function.
    cbar = plt.colorbar(map, ax=ax, shrink = .8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels([lbl_0,lbl_1, lbl_2, lbl_3])

    # Flip y-axis (corresponding to the matrix's x direction)
    # And label stuff
    ax.invert_yaxis()
    ax.set_xlabel('Eastward ($km$) $\\longrightarrow$')
    ax.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax.set_title(f'The Current Environment at T={itime:03d}')

    # Return figure object to caller:
    return fig

# Function to create plots for each time step and save in folder
def make_all_2dplots(environ_in, folder='results/'):
    '''
    For every time frame in 'environ_in', create a 2D plot and save the image
    in folder.
    '''

    import os

    # Check to see if folder exists, if not, make it!
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Make a bunch of plots.
    ntime, nx, ny, = environ_in.shape
    for i in range(ntime):
        print(f"\Working on plot #{i:04d}")
        fig = plot_environ2d(environ_in, itime=i)
        fig.savefig(f"{folder}/forest_i{i:04d}.png")
        plt.close('all')