import matplotlib
matplotlib.use('TKAgg')
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
from tqdm import tqdm
from scipy import optimize
from numba import jit

# @jit(nopython=True)

# parameters
a = 0.1
k = 0.1
M = 0.1

# time and space step size 
dt = 2.0
dx = 1.0



def roll(panel,index,axis):
    return np.roll(panel,index, axis = axis)


@jit(nopython=True)
def ini_panel (phi0,size):
    # this function initializes the grid for phi
    noise = 0.1
    return np.random.uniform(-noise + phi0, noise + phi0, (size, size))

#@jit(nopython=True)
def update_mu (panel):

    mu = -a * panel + a * panel**3 - (k / dx**2) * ( np.roll(panel, 1, axis=0) + np.roll(panel, -1, axis=0) + np.roll(panel, 1, axis=1) + np.roll(panel, -1, axis=1) - 4*panel )
    
    return mu

#@jit(nopython=True)
def update (panel):
    # function to update the panel
    mu = update_mu(panel)  
    panel = panel + M * (dt/dx **2) * (roll(mu, 1, axis=0) + roll(mu, -1, axis=0) + roll(mu, 1, axis=1) + roll(mu, -1, axis=1) - 4*mu )

    return panel


def phi_animation (panel, nsteps):

    energies = [] # list to store free energies values

    fig = plt.figure()
    im = plt.imshow(panel, vmax=1, vmin=-1,animated = True, cmap="plasma")
    plt.colorbar()

    for frame in tqdm(range(nsteps)):

        panel = update (panel)

        if frame%1000 == 0 :

            
        
            plt.cla()
            im = plt.imshow(panel,vmax=1, vmin=-1, animated = True, cmap="plasma")
            
            plt.draw()
            plt.pause(0.0001)
        
        if frame%100 == 0 :
            energies.append(free_energy(panel))

        

    plt.show()
    return energies


def free_energy(panel):
    # function to compute free energy of a panel configuration
    free_en = np.sum( -(a/2) * panel**2 + (a/4) * panel**4 + ( 0.5 * k/dx**2 ) * ((np.roll(panel, 1, axis=0) - panel)**2 + (np.roll(panel, 1, axis=1) - panel)**2) )

    return free_en




# initialize program
if __name__ == "__main__":

    if(len(sys.argv) != 4):
        print ("python3 file.py size nsteps phi0")
        sys.exit()

    size =int(sys.argv[1]) # size of the system
    nsteps = int(sys.argv[2]) # number of steps
    phi0 = float(sys.argv[3]) # phi0

    initial_panel = ini_panel(phi0,size)
    energies = phi_animation(initial_panel, nsteps)

    grid = np.arange(0,nsteps, 100)
    plt.plot(grid, energies, "2", color = "red")
    plt.title("Free energy")
    plt.ylabel("Free Energy")
    plt.xlabel("nsteps")
    plt.savefig("free_energy.pdf")
    plt.show()

    energies2 = np.array(energies)

    with open("ch1.txt", "w") as f:
        for i in range(energies2.shape[0]):
                f.write(str(grid[i]) + "   " + str(energies2[i]))
                f.write("\n")
        f.close()
