# imports used 
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
import matplotlib.colors as c

# to initialize the grid (divided in 3 equal sections)
def initial_panel (size ):

    in_panel = np.zeros((size,size))
    in_panel[:size//3]= 0
    in_panel[size//3:2*size//3] = 1
    in_panel[2*size//3:] = -1


    return in_panel


# function that updates the grid following the given rules
def update (panel, size):


    next_panel = panel

    for i in range(size):
        for j in range(size):

            person = panel[i,j]
            # all the neighbours
            look = np.array([((panel[(i+1)%size,j])) , ((panel[(i-1)%size,j]) ), ((panel[i,(j+1)%size]) ) , ((panel[i,(j-1)%size])),((panel[(i+1)%size,(j+1)%size])), ((panel[(i+1)%size,(j-1)%size])),((panel[(i-1)%size,(j-1)%size])), ((panel[(i-1)%size,(j+1)%size]))])
    

            if  (person == 0) : # rock
                papers = look[np.where(look==1)]
                if (papers.size > 2):
                    next_panel[i,j] = 1

            if (person == -1) : # scissors
                rocks = look[np.where(look==0)]
                if (rocks.size > 2):
                    next_panel[i,j] = 0

    
            if (person == 1 ): # paper
                scissors = look[np.where(look==-1)]
                if (scissors.size > 2):
                    next_panel[i,j] = -1



    return next_panel



# function that visualize the grid with an animation   
def do_animation (panel, nsteps,size):

    colours = c.ListedColormap(['deeppink','black','white'])
     
    fig = plt.figure()
    im = plt.imshow(panel, animated = True,vmin = -1, vmax = 1, cmap=colours)

   
    for frame in tqdm(range(nsteps)):

        next_panel = update (panel,size)
        
        plt.cla()
        im = plt.imshow(next_panel, animated = True, vmin = -1, vmax = 1, cmap= colours)
        plt.draw()
        plt.pause(0.1)

        panel = next_panel

    plt.show()
    
 # performs what is asked in part b) of the exam   
def where_rock(panel, nsteps,size):


    count = 0 

    # to store the number of rocks counted over time
    rocks = []

    for frame in tqdm(range(nsteps)):

        next_panel = update (panel,size)

        if (next_panel [ 1,1] == 0):

            count += 1


        rocks.append(count)

        panel = next_panel

    x = np.arange(0,nsteps,1)
    plt.plot(x, rocks)
    plt.ylabel("Counts of Rocks in the middle point of the grid")
    plt.xlabel("sweeps")
    plt.savefig("partb.pdf")
    plt.show()


if __name__ == "__main__":

    if(len(sys.argv) != 4):
        print ("python3 file.py size nsteps  task")
        sys.exit()

    size =int(sys.argv[1]) # size of the system
    nsteps = int(sys.argv[2]) # number of steps for the simulation
    task = str(sys.argv[3]) # part of the exam a, b

    in_panel = initial_panel(size)

    # animation part (part a) of the exam)
    if str(task) == "a": do_animation(in_panel, nsteps, size)

    # to perform part b) of the exam
    if str(task) == "b": where_rock (in_panel, nsteps, size)



