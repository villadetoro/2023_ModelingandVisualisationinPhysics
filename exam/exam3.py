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

# this function is to make np.random.random fasrter using jit
@jit(nopython=True)
def rando():
    return np.random.random()


# updates the grid given the rules and probailities specified
def update (panel, size, p1, p2, p3):

    
    next_panel = panel

    for i in range(size):
        for j in range(size):

            person = panel[i,j]
            look = np.array([((panel[(i+1)%size,j])) , ((panel[(i-1)%size,j]) ), ((panel[i,(j+1)%size]) ) , ((panel[i,(j-1)%size])),((panel[(i+1)%size,(j+1)%size])), ((panel[(i+1)%size,(j-1)%size])),((panel[(i-1)%size,(j-1)%size])), ((panel[(i-1)%size,(j+1)%size]))])
    

            if  (person == 0) : # rock
                if (rando() <= p1) :
                    papers = look[np.where(look==1)]

                    if (papers.size >= 1):
                        next_panel[i,j] = 1

            if (person == -1) : # scissors
                if (rando() <= p3) :
                    rocks = look[np.where(look==0)]
                    if (rocks.size >= 1):
                        next_panel[i,j] = 0

    
            if (person == 1 ): # paper
                if (rando() <= p2) :
                    scissors = look[np.where(look==-1)]
                    if (scissors.size >= 1):
                        next_panel[i,j] = -1




    return next_panel



# function that visualizes the grid with an animation   
def do_animation (panel, nsteps,size, p1, p2, p3):

    colours = c.ListedColormap(['deeppink','black','white'])
     
    fig = plt.figure()
    im = plt.imshow(panel, animated = True,vmin = -1, vmax = 1, cmap=colours)

   
    for frame in tqdm(range(nsteps)):

        next_panel = update (panel,size, p1, p2, p3)
        
        plt.cla()
        im = plt.imshow(next_panel, animated = True, vmin = -1, vmax = 1, cmap= colours)
        plt.draw()
        plt.pause(0.1)

        panel = next_panel

    plt.show()

# function that performs part d) of the exam
def p3_study(panel):

    size = 50
    resolution = 0.001
    nsteps = 550

    p1 = 0.5
    p2 = 0.5

    p3_grid = np.arange(0,0.1 + resolution,resolution)

    N = size*size

    mean_array = np.zeros(p3_grid.shape[0])
    variance_array = np.zeros(p3_grid.shape[0])

    for i in tqdm(range(p3_grid.shape[0])):

        # initializing the panel fopr evey new probability
        panel = np.random.choice([1,0,-1],[size,size])

        # to store the numbe of minorities
        minority = []

        for frame in tqdm(range(nsteps)):

            papers_q = panel[np.where(panel==1)]
            papers = papers_q.size
            rocks_q = panel[np.where(panel==0)]
            rocks = rocks_q.size
            scissors_q = panel[np.where(panel==-1)]
            scissors = scissors_q.size

            counts = np.array([papers, rocks, scissors])
            # in this part what I'm doing is deciding which one is going to be the minority 
            ind = np.argsort(counts)

            next_panel = update (panel,size,  p1, p2, p3_grid[i])
            panel = next_panel

            if frame >= 100: # for stabilization

                papers_q = panel[np.where(panel==1)]
                papers = papers_q.size
                rocks_q = panel[np.where(panel==0)]
                rocks = rocks_q.size
                scissors_q = panel[np.where(panel==-1)]
                scissors = scissors_q.size

                counts = np.array([papers, rocks, scissors])

                # knowing the minority by the found indices before 
                minority.append(counts[ind[0]])

                if np.amin(counts) == 0:
                        break

        mean_array[i] = np.mean(np.array(minority))/N
        variance_array[i] = np.var(np.array(minority))/N

    return p3_grid, mean_array, variance_array

# function that performs part e) of the exam
def heatmap_part():

    size = 50
    resolution = 0.01
    nsteps = 300

    N = size*size

    p2_grid = np.arange(0,0.3 + resolution,resolution)
    p3_grid = np.arange(0,0.3 + resolution,resolution)

    p1 = 0.5

    minority_array = np.zeros((p2_grid.shape[0], p3_grid.shape[0]))

    for i in tqdm(range(p2_grid.shape[0])):

        
        for j in range(p3_grid.shape[0]):

            # initializing the panel fopr evey new probability
            panel = np.random.choice([1,0,-1],[size,size])

            # to store the numbe of minorities
            minority = []
            

            for sweeps in range(nsteps):

                panel = update (panel,size,  p1, p2_grid[i], p3_grid[j])

                papers_q = panel[np.where(panel==1)]
                papers = papers_q.size
                rocks_q = panel[np.where(panel==0)]
                rocks = rocks_q.size
                scissors_q = panel[np.where(panel==-1)]
                scissors = scissors_q.size

                counts = np.array([papers, rocks, scissors])
                # in this part what I'm doing is deciding which one is going to be the minority 
                ind = np.argsort(counts)


                if sweeps >= 50:

        
                    papers_q = panel[np.where(panel==1)]
                    papers = papers_q.size
                    rocks_q = panel[np.where(panel==0)]
                    rocks = rocks_q.size
                    scissors_q = panel[np.where(panel==-1)]
                    scissors = scissors_q.size

                    counts = np.array([papers, rocks, scissors])
                    # knowing the minority by the found indices before 
                    minority.append(counts[ind[0]])
                    
                    if np.amin(counts) == 0:
                        break
          
            

            minorities = np.mean(np.array(minority))
            minority_array[j,i] = minorities/N
            
    # for the axis of the heatmap
    extent = (0, 0.3, 0, 0.3)

    with open('2d_map.txt', 'w') as f:
            for i in range(p2_grid.shape[0]):
                for j in range(p3_grid.shape[0]):
                    f.write(str(p2_grid[i]) + "   " + str(p3_grid[j]) + "   " + str(minority_array[j,i]))
                    f.write("\n")
            f.close()

    #print(infected_array)
    plt.imshow( minority_array, cmap = "plasma", origin = "lower", extent = extent)
    cbar=plt.colorbar()
    plt.title("Minority Average <M>/N")
    plt.savefig("average.pdf")
    plt.xlabel("p2")
    plt.ylabel("p3")
    plt.show()




    


if __name__ == "__main__":

    if(len(sys.argv) != 7):
        print ("python3 file.py size nsteps p1 p2 p3")
        sys.exit()

    size =int(sys.argv[1]) # size of the system
    nsteps = int(sys.argv[2]) # number of steps for the simulation
    task = str(sys.argv[3]) # part of the exam c,d, e

    # the porbabilities are only used for part c) so do not worry about them if you are trying to run part d) or e) 
    p1 = float(sys.argv[4]) # p1
    p2 = float(sys.argv[5]) # p2
    p3 = float(sys.argv[6]) # p3


    in_panel = np.random.choice([1,0,-1],[size,size])


    if str(task) == "c": do_animation(in_panel, nsteps, size, p1, p2, p3)

    if str(task) == "d": 
        x, means, variances = p3_study(in_panel)

        plt.plot(x, means)
        plt.ylabel("<M>/N")
        plt.xlabel("p3")
        plt.savefig("partd_1.pdf")
        plt.show()

        plt.plot(x,variances)
        plt.ylabel("<M^2> - <M>^2/N")
        plt.xlabel("p3")
        plt.savefig("partd_2.pdf")
        plt.show()

    if str(task) == "e": heatmap_part()



