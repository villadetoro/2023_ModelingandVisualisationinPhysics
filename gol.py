# packages and imports
import matplotlib
matplotlib.use('TKAgg')
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
from tqdm import tqdm
from scipy import optimize

#from numba import jit
#@jit(nopython=True)


# function to create the initial panel: glider, blinker, or random
# it returns the initial panel (numpy array)
def initial_panel (N, seed ):

    # glider matrix
    glider_m = np.array([[0, 0, 1], 
                       [1, 0, 1], 
                       [0, 1, 1]], dtype=np.uint8)
    
    # blinker matrix
    blinker_m = np.array([[0, 1, 0], 
                       [0, 1, 0], 
                       [0, 1, 0]], dtype=np.uint8)   

    if seed == "glider":

        in_panel = np.zeros(shape=(N, N), dtype=np.uint8)
        h, w = glider_m.shape
        #n_gliders = in_panel.size // (9 * 25)
        n_gliders = 1
        for o in range(n_gliders):
            i, j = (5,5,
            )
            in_panel[i : i + h, j : j + w] = glider_m

    if seed == "blinker":

        in_panel = np.zeros(shape=(N, N), dtype=np.uint8)
        h, w = blinker_m.shape
        #n_blinkers = in_panel.size // (9 * 25)
        n_blinkers = 1
        for o in range(n_blinkers):
            i, j = (5,5,
            )
            in_panel[i : i + h, j : j + w] = blinker_m

    
    if seed == "random":
        in_panel = np.random.randint(
            0, 2, size=(N, N), dtype=np.uint8
        )

    return in_panel

# updates the given panel following the "rules" of the GoL
# returns updated panel (numpy array)
def run_gol (panel):

    # Convolution kernel that counts the cells around the center cell
    kernel = np.array([[1, 1, 1], 
                       [1, 0, 1], 
                       [1, 1, 1]], dtype=np.uint8)


    # Run a single 2D convolution on the panel
    convolved_panel = convolve2d(panel, kernel, mode="same", boundary = "wrap") # MODE WRAP OR SAME?
    # The kernel finds the sum of the 8 cells around a given cell
    # To get the next panel following the rules of the game of life:
    next_panel = (
        ((panel == 1) & (convolved_panel > 1) & (convolved_panel < 4))
        | ((panel == 0) & (convolved_panel == 3))
    ).astype(np.uint8)


    return next_panel

# function that performs the GoL and visualize it with an animation   
def game_animation (panel, nsteps):

    fig = plt.figure()
    im = plt.imshow(panel, animated = True, cmap="tab20c")

    for frame in range(nsteps):

        next_panel = run_gol (panel)
        
        plt.cla()
        im = plt.imshow(panel, animated = True, cmap="tab20c")
        plt.draw()
        plt.pause(0.0001)

        panel = next_panel

    plt.show()

# function to analyize the time it takes the panel to stop evolving (equilibration time)
# returns a list of equilibration times

def random_evol (size):

    sweeps = 10000
    nsimulations = 1000 # number of simulations
    eq_time = []

    for simulation in tqdm(range(nsimulations)):

        panel = initial_panel(size, str("random"))
        
        count = 0 
        no_change = 0
        for step in range(sweeps):
            count += 1
            
            alive1 = np.sum(panel)
            next_panel = run_gol (panel)
            
            alive2 = np.sum(next_panel)
            
            if alive1 == alive2:
                no_change += 1

            if alive1 != alive2:
                no_change = 0

            if no_change == 10:
                eq_time.append(count - 10)
                break

            panel = next_panel

    return eq_time

# linear function for later fits
def linear (x, a, b):
    return a*x + b



if __name__ == "__main__":

    if(len(sys.argv) != 5):
        print ("python3 file.py size nsteps panel analysis")
        sys.exit()

    size =int(sys.argv[1]) # size of the system
    nsteps = int(sys.argv[2]) # number of steps for the simulation
    seed = str(sys.argv[3]) # panel evolving to: random, oscillator, glider
    analysis = str(sys.argv[4]) # animation / evoluion / COM 

    in_panel = initial_panel(size, seed) # intialise 

    if analysis == str("animation"):
        game_animation(in_panel, nsteps)

    if analysis == str("evolution"):
        equil_times = random_evol(size)

        # to save data into .txt
        with open('eq_time.txt', 'w') as f:
            for i in range(np.array(equil_times).shape[0]):
                f.write(str(equil_times[i]))
                f.write("\n")
            f.close()

        # plot
        plt.hist(equil_times, bins = 50, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)
        plt.xlabel("Time (sweeps)")
        plt.ylabel("Events")
        plt.title("Time it takes a random panel to reach equilibrium")
        plt.savefig("equil.pdf")
        plt.show()

    if analysis == str("com"):

        panel = in_panel
        r_com_values1 = []
        r_com_values2 = []
        bad = []

        for frame in range(nsteps):
            
            next_panel = run_gol(panel)
            
            alives = np.nonzero(panel)
            alives1, alives2 = np.nonzero(panel)

            # calculation of centre of mass
            x_d = np.abs(np.max(alives1) - np.min(alives1))
            y_d = np.abs(np.max(alives2) - np.min(alives2))

            # when the glider is going through the boundary
            if (x_d >= size/2) or (y_d >= size/2):
                
                panel = next_panel   
                #plt.imshow(panel)
                #plt.show()            

            else:

                com = np.sum(alives, axis = 1)/len(alives[0]) 
                r_com_values1.append(com[0])
                r_com_values2.append(com[1])
                panel = next_panel
        

        x_data = np.arange(50,151,1)

        # fit of the data
        popt1, pcov1 = optimize.curve_fit(linear,x_data, np.array(r_com_values1)[50:151])
        popt2, pcov2 = optimize.curve_fit(linear,x_data, np.array(r_com_values2)[50:151])

        # saving data into .txt
        with open('com.txt', 'w') as f:
            for i in range(np.array(r_com_values1).shape[0]):
                f.write(str(r_com_values1[i]))
                f.write("\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")
            for j in range(np.array(r_com_values2).shape[0]):
                f.write(str(r_com_values2[j]))
                f.write("\n")
            f.write("speeds  " + str(popt1[0]) + " and " + str(popt2[0]))
            f.write("\n")
            f.write(str(np.linalg.norm([popt1[0],popt2[0]])))

            f.close()        
        
        # plots
        plt.plot(r_com_values1, "x")
        plt.xlabel("Time (sweeps)")
        plt.ylabel("COM")
        plt.title("Centre of Mass of a glider as a function of time (x_coord)")
        plt.savefig("r_com_x.pdf")
        plt.show()

        plt.plot(r_com_values2, "x")
        plt.xlabel("Time (sweeps)")
        plt.ylabel("COM")
        plt.title("Centre of Mass of a glider as a function of time (y_coord)")
        plt.savefig("r_com_y.pdf")
        plt.show()







    
 
