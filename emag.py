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

# function used for the linerat fits
def linear (x, a, b):
    return a*x + b

# function to calculate the field from a vector potential
def field(potential, dx, inside ,size):

    field_x = (np.roll(potential,1,axis=0)-np.roll(potential,-1,axis=0))[inside,inside,size//2].ravel()/(-2*dx)
    field_y = (np.roll(potential,1,axis=1)-np.roll(potential,-1,axis=1))[inside,inside,size//2].ravel()/(-2*dx)
    field_z = (np.roll(potential,1,axis=2)-np.roll(potential,-1,axis=2))[inside,inside,size//2].ravel()/(-2*dx)

    return field_x, field_y, field_z

# updating potential with jacobi algorithm
def jacobi_pot (pot, rho, inside_area, npad):

    pot_initial = np.copy(pot)
    
    pot = 1/6 * (np.roll( pot_initial, 1, axis=0 ) + np.roll( pot_initial, -1, axis=0 ) + np.roll( pot_initial, 1, axis=1 ) + np.roll( pot_initial, -1, axis=1 ) + np.roll( pot_initial, 1, axis=2 ) + np.roll( pot_initial, -1, axis=2 ) + rho)
    
    # boundary conditions: potential at the boundary = 0 
    pot_final = np.pad(pot[inside_area],npad)
    pot_error = np.sum(np.absolute(pot_final - pot_initial))

    return pot_final, pot_error

# creating a "checkerboard" for gauss algorithm
def black_white (sizeB):

    white_bool =  np.bool_(np.indices((sizeB, sizeB,sizeB)).sum(axis=0) % 2)
    black_bool = np.invert(white_bool)

    return white_bool, black_bool

# updating potential with gauss algorithm
def gauss_pot(pot, rho, inside_area, sizeB, npad):

    pot_initial = np.copy(pot)

    # definning checkerboard
    black, white = black_white(sizeB)

    # want to update only white positions in the checkerboard 
    pot = 1/6 * (np.roll( pot, 1, axis=0 ) + np.roll( pot, -1, axis=0 ) + np.roll( pot, 1, axis=1 ) + np.roll( pot, -1, axis=1 ) + np.roll( pot, 1, axis=2 ) + np.roll( pot, -1, axis=2 ) + rho)
    
    # not wanting to update blacks so:
    pot[black] = pot_initial[black]

    pot = np.pad(pot[inside_area], npad)
    pot_whites = np.copy(pot)

    # want to update only black positions in the checkerboard 
    pot = 1/6 * (np.roll( pot, 1, axis=0 ) + np.roll( pot, -1, axis=0 ) + np.roll( pot, 1, axis=1 ) + np.roll( pot, -1, axis=1 ) + np.roll( pot, 1, axis=2 ) + np.roll( pot, -1, axis=2 ) + rho)
    
    pot[white] = pot_whites[white]

    pot_final = np.pad(pot[inside_area], npad)
    
    pot_error = np.sum(np.abs(pot_final - pot_initial))
    

    return pot_final, pot_error

# updating potential with 
def sor_pot(pot, rho, inside_area, sizeB, npad, omega):

    pot_initial = np.copy(pot)

    # definning checkerboard
    black, white = black_white(sizeB)

    # want to update only white positions in the checkerboard 
    pot = 1/6 * (np.roll( pot, 1, axis=0 ) + np.roll( pot, -1, axis=0 ) + np.roll( pot, 1, axis=1 ) + np.roll( pot, -1, axis=1 ) + np.roll( pot, 1, axis=2 ) + np.roll( pot, -1, axis=2 ) + rho)
    
    # relaxation
    pot *= omega
    pot += pot_initial * (1 - omega)     

    # not wanting to update blacks so:
    pot[black] = pot_initial[black]

    pot = np.pad(pot[inside_area], npad)
    pot_whites = np.copy(pot)

    # want to update only black positions in the checkerboard 
    pot = 1/6 * (np.roll( pot, 1, axis=0 ) + np.roll( pot, -1, axis=0 ) + np.roll( pot, 1, axis=1 ) + np.roll( pot, -1, axis=1 ) + np.roll( pot, 1, axis=2 ) + np.roll( pot, -1, axis=2 ) + rho)

 
    # relaxation
    pot *= omega
    pot += pot_initial * (1 - omega) 

    pot[white] = pot_whites[white]


    pot_final = np.pad(pot[inside_area], npad)
    
    pot_error = np.sum(np.abs(pot_final - pot_initial))
    

    return pot_final, pot_error    

def plot_pot (pot, size):

    plt.imshow(pot[:,:, (size)//2], cmap='gnuplot')
    plt.colorbar()
    plt.savefig("pot.png")
    plt.show()   

    pot2d = pot[:,:, (size)//2]

    with open("pot.txt", "w") as f:
        for i in range(pot.shape[0]):
            for j in range(pot.shape[1]):
                f.write(str(i) + "   " + str(j) + "   " + str(pot2d[i,j]))
                f.write("\n")
        f.close()

def plot_electric (field_x, field_y, field_z, pot, size):

    position = np.array(np.meshgrid(np.linspace(0,size - 1, size), np.linspace(0,size - 1, size))).T.reshape(-1,2)    
    distance = np.linalg.norm(position - size//2,axis=1)
    potential = pot[:, :, size//2].flatten()

    distance2 = np.log(distance[np.argsort(distance)])
    potential2 = np.log(potential[np.argsort(distance)])



    popt1, pcov1 = optimize.curve_fit(linear, distance2[10:400] , potential2[10:400])
    print(popt1[0])

    plt.plot(distance2[0:500], linear(distance2[0:500], popt1[0],popt1[1]))
    plt.scatter(distance2, potential2, s = 5, color = "red")
    plt.xlabel("log(Distance)")
    plt.title(str(popt1[0]))
    plt.ylabel("log(Potential)")
    plt.savefig("ELECdistancepotentialfit.pdf")
    plt.show()

    field = np.sqrt(field_x**2 + field_y**2 + field_z**2)
    field2 = np.log(field[np.argsort(distance)])

    plt.yscale('linear')
    plt.quiver(position.T[0], position.T[1], field_x/field, field_y/field,  angles='xy', scale_units='xy', scale=1)
    plt.title("Electric Field ")
    plt.savefig("vectorelectric.pdf")
    plt.show()


    popt2, pcov2 = optimize.curve_fit(linear, distance2[50:600] , field2[50:600] )
    print(popt2[0])

    plt.plot(distance2[10:600], linear(distance2[10:600], popt2[0],popt2[1]), color = "red")
    plt.scatter(distance2, field2, marker='+', s=17, c='purple')
    plt.title(str(popt2[0]))
    plt.ylabel("log(Electric field strength)")
    plt.xlabel("log(Distance)")
    plt.savefig("ELECdistancefield2.pdf")
    plt.show()

    with open("electric.txt", "w") as f:
        f.write("position    distance     potential    Ex    Ey   Ez   ")
        f.write("\n")
        for j in range(distance.shape[0]):
            f.write(str(position.T[0,j]) + "   " + str(position.T[1,j]) + "   " + str(distance[j]) + "   " + str(potential[j]) + "   " + str(field_x[j]) + "   " + str(field_y[j]) + "   " + str(field_z[j]) )
            f.write("\n")
        f.close()
    

def plot_magnetic (field_x, field_y, pot, size):
    
    magfield_x = field_y
    magfield_y = -1*field_x
    field = np.sqrt(magfield_x**2 + magfield_y**2)


    position = np.array(np.meshgrid(np.arange(1, size+1), np.arange(1, size+1))).T.reshape(-1, 2)
    distance = np.linalg.norm(position - size//2, axis=1)
    potential = pot[1:size+1, 1:size+1, size//2].ravel()
    
    distance2 = np.log(distance[np.argsort(distance)])
    potential2 = np.log(potential[np.argsort(distance)])


    popt1, pcov1 = optimize.curve_fit(linear, distance2[5:350] , potential2[5:350])
    print(popt1[0])

    plt.plot(distance2[5:350], linear(distance2[5:350], popt1[0],popt1[1]))
    plt.scatter(distance2, potential2, s = 5, color = "red")
    plt.title(str(popt1[0]))
    plt.xlabel("log(Distance)")
    plt.ylabel("log(Potential)")
    plt.savefig("MAGdistancefieldpotentialfit.pdf")
    plt.show()

    field2 = np.log(field[np.argsort(distance)])

    plt.yscale('linear')
    plt.quiver(position.T[0], position.T[1], magfield_x/field, magfield_y/field,  angles='xy', scale_units='xy', scale=1)
    plt.title("Magnetic Field ")
    plt.savefig("vectormagnetic.pdf")
    plt.show()



    popt2, pcov2 = optimize.curve_fit(linear, distance2[100:400] , field2[100:400] )
    print(popt2[0])

    plt.plot(distance2[100:400], linear(distance2[100:400], popt2[0],popt2[1]), color = "red")
    plt.scatter(distance2, field2, marker='+', s=17, c='purple')
    plt.title(str(popt2[0]))
    plt.ylabel("log(Electric field strength)")
    plt.xlabel("log(Distance)")
    plt.savefig("MAGdistancefieldfit.pdf")
    plt.show()

    with open("magnetic.txt", "w") as f:
        f.write("position    distance     potential    Mx    My ")
        f.write("\n")
        for j in range(distance.shape[0]):
            f.write(str(position.T[0,j]) + "   " + str(position.T[1,j]) + "   " + str(distance[j]) + "   " + str(potential[j]) + "   " + str(magfield_x[j]) + "   " + str(magfield_y[j])  )
            f.write("\n")
        f.close()






# initialize program
if __name__ == "__main__":

    if(len(sys.argv) != 5):
        print ("python3 file.py size mode field tolerance")
        sys.exit()

    size =int(sys.argv[1]) # size of the system
    mode = str(sys.argv[2]) # mode of the simulation: gaussian/jacobian/sor
    field_type = str(sys.argv[3]) # type of field magnetic/electric
    tol = float(sys.argv[4]) # tolerance

    # because of boundary conditions when we use roll and padding
    sizeB = size + 2

    # parameters
    pot = np.zeros((sizeB,sizeB,sizeB)) # initial potential 
    rho = np.zeros((sizeB,sizeB,sizeB)) # initial rho
    dx = 1
    nsteps = 3*size**3


    if field_type == str("magnetic"):


        inside = slice(1,sizeB-1)
        inside_area = (inside, inside,)
        npad = ((1, 1), (1, 1), (0, 0)) 
              
        # for rod:
        rho[ size//2, size//2, :] = 1

    if field_type == str("electric"):

        inside = slice(1,sizeB-1)
        inside_area = (inside, inside, inside)
        npad = ((1,1), (1,1),(1, 1))

        # for point charge:
        rho[sizeB//2, sizeB//2, sizeB//2] = 1 



    # updating the potential depending on the mode

    if mode == str("jacobian"):

        for n in tqdm(range(nsteps)):
            pot, error = jacobi_pot(pot, rho, inside_area,  npad)
            if np.isclose(error, 0, atol=tol):
                break 

    if mode == str("gauss"):
        for n in tqdm(range(nsteps)):

            pot, error = gauss_pot(pot, rho, inside_area, sizeB, npad)

            if np.isclose(error, 0, atol=tol):
                break 

    if mode == str("sor"):
        num_steps = []
        sor_grid = np.arange(1.0,2.0, 0.01)

        for omega in tqdm(sor_grid):
            
            pot = np.zeros((sizeB,sizeB,sizeB)) # initial potential 

            for n in range(nsteps):

                pot, error = sor_pot(pot, rho, inside_area, sizeB, npad, omega)

                if np.isclose(error, 0, atol=tol):
                    num_steps.append(n)
                    break 
        
        num_array = np.array(num_steps)
        plt.plot(sor_grid, num_steps, color = "red")
        plt.xlabel("Omega")
        plt.ylabel("nsteps")
        plt.title("SOR")
        plt.savefig("sor.png")
        plt.show()

        with open("sor.txt", "w") as f:
            for i in range(num_array.shape[0]):
                f.write(str(sor_grid[i]) + "  :  " + str(num_array[i]))
                f.write("\n")

            f.close()

    #back to initial shape after padding
    potential = pot[1:-1, 1:-1, 1:-1]

    plot_pot(potential,size)

    field_x, field_y, field_z = field(pot, dx, inside ,size)


    if field_type == str("electric"):
        plot_electric(-1*field_x, -1*field_y, -1*field_z, potential, size)
    if field_type == str("magnetic"):
        plot_magnetic(field_x, field_y, pot, size)






 