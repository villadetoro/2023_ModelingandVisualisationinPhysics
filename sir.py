# packages and imports
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
import matplotlib.colors as c
from numba import jit


# this function is to make np.random.randit faster using jit
@jit(nopython=True)
def rannum(x,y):
    return np.random.randint(x,y)

# this function is to make np.random.random faster using jit
@jit(nopython=True)
def rando():
    return np.random.random()

@jit(nopython=True)
def sir_go (people, p1, p2, p3, size):
            
    for k in range(size*size):

        #choosing randomly i and j
        i = rannum(0,size )
        j = rannum(0,size )

        person = people[i,j]

        #person = people[i,j]


        if  (person == 0) :
            if (rando() <= p2) :
            # infected to recovered
            #person == 1
                people[i,j] = int(1)

        if (person == -1) :
            if (rando() <= p1):
            #susceptible to infected
         
                if 0 in [((people[(i+1)%size,j])) , ((people[(i-1)%size,j]) ), ((people[i,(j+1)%size]) ) , ((people[i,(j-1)%size]))]:
                    people[i, j] = int(0)

  
        if (person == 1 ):
            if (rando() <= p3) :
            #recovered to suscetible
            #person == -1
                people[i,j] = -int(1)

        #people[i,j] = person



    return people

def sir_animation (people, nsteps, p1, p2, p3, size):

    colours = c.ListedColormap(['deeppink','black','white'])
     
    fig = plt.figure()
    im = plt.imshow(people, animated = True,vmin = -1, vmax = 1, cmap=colours)
    

    for frame in range(nsteps):
        
        next_people = sir_go (people, p1, p2, p3, size)
        
        plt.cla()
        im = plt.imshow(next_people, animated = True, vmin = -1, vmax = 1, cmap= colours)
        plt.draw()
        plt.pause(0.1)

        people = next_people



    
    plt.show()

def task1( size):

    nsteps = 1101
    resolution = 0.05


    p1_grid = np.arange(0,1 + resolution,resolution)
    p3_grid = np.arange(0,1 + resolution,resolution)

    p2 = 0.5
    
    N = size*size


    infected_array = np.zeros((p1_grid.shape[0], p3_grid.shape[0]))



    for i in tqdm(range(p1_grid.shape[0])):

        
        for j in range(p3_grid.shape[0]):

            people = np.random.choice([1,0,-1],[size,size])

            ill = []
            

            for sweeps in range(nsteps):

                people = sir_go (people, p1_grid[i], p2, p3_grid[j], size)

                #people = next_people


                if sweeps >= 99:

                    no_zeros = people[np.where(people==0)]

                    #no_zeros = np.count_nonzero(people == 0)

                    ill.append(no_zeros.size)
                    
                    if no_zeros.size == 0:
                        break
          
            

            infected = np.mean(np.array(ill))
            infected_array[j,i] = infected/N
            

    extent = (0, 1, 0, 1)

    with open('2d_map.txt', 'w') as f:
            for i in range(p1_grid.shape[0]):
                for j in range(p3_grid.shape[0]):
                    f.write(str(p1_grid[i]) + "   " + str(p3_grid[j]) + "   " + str(infected_array[j,i]))
                    f.write("\n")
            f.close()

    #print(infected_array)
    plt.imshow( infected_array, cmap = "plasma", origin = "lower", extent = extent)
    cbar=plt.colorbar()
    plt.title("Infected Average <I>/N")
    plt.savefig("average.pdf")
    plt.xlabel("p1")
    plt.ylabel("p3")
    plt.show()


#@jit(nopython=True)
def error_cal (energy_values, size):

    energy_array = np.array(energy_values)
    random_hc = np.zeros(1000)

    for i in range(1000):
        random_energies = np.random.choice(energy_array, size = 1000)
        random_heatcap = np.var(random_energies)/(size*size)
        random_hc[i] = random_heatcap

    error = np.std (random_hc)

    return error


def task2( size):

    nsteps = 10100
    resolution = 0.01

    p1_grid = np.arange(0.2,0.5 + resolution,resolution)


    p2 = 0.5
    p3 = 0.5
    

    N = size*size


    infected_array = np.zeros(p1_grid.shape[0])
    infected_error = np.zeros(p1_grid.shape[0])


    for i in tqdm(range(p1_grid.shape[0])):

        people = np.random.choice([1,0,-1],[size,size])

        ill = []

        for sweeps in range(nsteps):

            next_people = sir_go (people, p1_grid[i], p2, p3, size)
            people = next_people


            if sweeps >= 100:

                no_zeros = people[np.where(people==0)]

                ill.append(no_zeros.size)

        infected = np.var(np.array(ill))
        infected_array[i] = infected/N
        error = error_cal(np.array(ill), size)

        infected_error [i] = error       

    with open('variance.txt', 'w') as f:
            for i in range(infected_array.shape[0]):
                f.write(str(infected_array[i]) + "   " + str(infected_error[i]))
                f.write("\n")
            f.close()

    
    plt.errorbar(p1_grid,infected_array,yerr = infected_error, marker = "x", linestyle = "")
    #plt.plot(p1_grid, infected_array)
    plt.title("Infected Variance")
    plt.xlabel("p1")
    plt.ylabel("Infected variance")
    plt.savefig("variance.pdf")
    plt.show()






    return infected 

def task3(p1, p2, p3, size):

    people = np.random.choice([1,0,-1],[size,size])

    resolution = 0.005

    fractions =  np.arange(0,1, resolution)

    infected_plot = np.zeros(fractions.shape[0])
    
 

    for i in tqdm(range(fractions.shape[0])):

        nsteps = 10100

        fraction = int(fractions[i] * size*size)

        vaccinated = 0
        
        while vaccinated <= fraction:

            #choosing randomly i and j
            o = rannum(0,size)
            p = rannum(0,size)

            person = people[o,p]

            if person != 2:
                people[o,p] = 2
            
            vaccinated = people[np.where(people==2)].size
      
        next_people = sir_go (people, p1, p2, p3, size)
        people = next_people 

        ill = []

        for sweeps in range(nsteps):

            next_people = sir_go (people, p1, p2, p3, size)

            people = next_people


            if sweeps >= 200:

                no_zeros = people[np.where(next_people==0)]

                ill.append(no_zeros.size)

            
        
        
        infected = np.mean(np.array(ill))


        infected_plot[i] = infected/(size*size)

    with open('vaccine.txt', 'w') as f:
            for i in range(infected_plot.shape[0]):
                f.write(str(infected_plot[i]) )
                f.write("\n")
            f.close()
        
    plt.plot(fractions,infected_plot,"xb")
    plt.title("Infected Average vs Vaccinated fraction")
    plt.ylabel("Fraction")
    plt.xlabel("Infected Average")
    plt.savefig("fractions.pdf")
    plt.show()

            
    


if __name__ == "__main__":

    if(len(sys.argv) != 7):
        print ("python3 file.py size steps p1 p2 p3 task")
        sys.exit()

    size =int(sys.argv[1]) # size of the system
    nsteps = int(sys.argv[2]) # number of steps
    p1 = float(sys.argv[3]) 
    p2 = float(sys.argv[4]) 
    p3 = float(sys.argv[5])
    task = str(sys.argv[6])

    #N =size*size 

    # S -> -1
    # I -> 0
    # R -> 1

    in_people = np.random.choice([1,0,-1],[size,size])

    

    if task == str("animation"):
        sir_animation(in_people, nsteps, p1, p2, p3, size)

    if task == str("task1"):
        task1( size)

    if task == str("task2"):
        infected = task2( size)      

    if task == str("task3"):
        task3(p1, p2, p3, size)

