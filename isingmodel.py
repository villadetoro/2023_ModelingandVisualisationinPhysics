import matplotlib
matplotlib.use('TKAgg')
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from scipy.stats import bootstrap
from numba import jit
from tqdm import tqdm


if(len(sys.argv) != 3):
    print ("Usage python ising.animation.py N T")
    sys.exit()

N=int(sys.argv[1]) # size of the system
ly=lx=N

J=1.0
nstep= int(10100)



kT=float(sys.argv[2]) 

# user choice for dynamics method
meth = str(input("Dynamics glauber/kawasaki?: "))

# user choice to do temperature analysis or not 
analysis = str(input("With temperature analysis? (yes/no): "))

if analysis == str("yes"):
    temp_in =  float(input("What's the initial temperature for the analysis: "))
    temp_fin =  float(input("What's the final temperature for the analysis: "))
    size_step = float(input("What's the step size?: "))

    temp_grid = np.arange(temp_in, temp_fin + size_step, size_step)

    if meth == str("glauber"):
        
        spin=np.ones((lx,ly),dtype=float)
    
    if meth == str("kawasaki"):

        spin=np.ones((lx,ly),dtype=float)
        spin[:,:int(N/2)] = -1



if analysis == str("no"):
    temp_grid = np.array([kT])

    spin=np.zeros((lx,ly),dtype=float)

    #initialise spins randomly
    for i in range(lx):
        for j in range(ly):
            r=random.random()
            if(r<0.5): spin[i,j]=-1
            if(r>=0.5): spin[i,j]=1

# lists to store computed values for several properties
mag_av = []
sus = []
heat_cap = []
ener_av = []
energy_squared_error = []
energy_all = []
hc_error = []


# for the animation
fig = plt.figure()
im=plt.imshow(spin, animated=True, vmin = -1, vmax = 1)

# this makes the code a bit faster
@jit(nopython=True)
def rannum(x,y):
    return np.random.randint(x,y)

@jit(nopython=True)
def rando():
    return np.random.random()

@jit(nopython=True)
def energy_cal (spin):

    energy_val = 0

    for itrial in range (lx):
        for jtrial in range(ly):

            energy_val += - (spin[itrial,jtrial] * ( spin[(itrial+1)%N,jtrial]+ spin[(itrial-1)%N,jtrial]+ spin[itrial,(jtrial+1)%N]+ spin[itrial,(jtrial-1)%N] ))/2

    return energy_val

@jit(nopython=True)
def error_cal (energy_values):

    energy_array = np.array(energy_values)
    random_hc = np.zeros(1000)

    for i in range(1000):
        random_energies = np.random.choice(energy_array, size = 1000)
        random_heatcap = np.var(random_energies)/(N*N*(kT)**2)
        random_hc[i] = random_heatcap
    
    error = np.std (random_hc)

    return error


# function for Glauber dynamics
def glauber(kT):

    mag_abs = []
    magn = []
    
    energy = []
    

    for n in tqdm(range(nstep)):
        for i in range(lx):
            for j in range(ly):

                itrial = rannum(0,lx)
                jtrial = rannum(0,ly)

                # change in energy
                delta_e = 2 * spin[itrial,jtrial] * ( spin[(itrial+1)%N,jtrial]+ spin[(itrial-1)%N,jtrial]+ spin[itrial,(jtrial+1)%N]+ spin[itrial,(jtrial-1)%N] )

                # change in the spins
                if delta_e <= 0:
                    spin[itrial,jtrial] *= -1       
            
                elif (rando() <= np.exp(-delta_e/kT)):
                    spin[itrial,jtrial] *= -1

                    
        #occasionally plot or update measurements, eg every 10 sweeps
        if(n%10==0) and analysis == str("no"): 
        #   update measurements
        #   show animation
            plt.cla()
            im=plt.imshow(spin, animated=True, vmin = -1, vmax = 1)
            plt.draw()
            plt.pause(0.0001)

        if (n%10==0) and analysis == str("yes"):
            if n >= 100:
            
                mag_abs.append(np.abs(np.sum(spin)))
                magn.append(np.sum(spin))
                #mag_squared.append(np.sum(spin)**2)

           
            
                energy_tot = energy_cal(spin)


                energy.append(energy_tot)
                #energy_squared.append(energy_val**2)

    
    if analysis == str("yes"):

        mag_av.append(np.mean(np.array(mag_abs)))
        sus_val = np.var(magn) / (N*N*kT)
        sus.append(sus_val)

        ener_av.append(np.mean(energy))
        heat_cap_val = np.var(energy)/ (N*N*kT**2)
        heat_cap.append(heat_cap_val)

        hc_error_val = error_cal(energy)
        hc_error.append(hc_error_val)



    
# function for Kawasaki dynamics
def kawasaki(kT):

    energy = []

    for n in tqdm(range(nstep)):

        

        for k in range(N**2):

            p=np.arange(N, dtype= int)+1
            p[-1]=0
            m=np.arange(N, dtype=int)-1
            m[0]=N-1


            i = rannum(0,lx)
            j = rannum(0,ly)
            
            i2=i 
            j2=j


            while (spin[i,j] == spin[i2,j2]):   
                i2 = rannum(0,N)
                j2 = rannum(0,N)
                
            deltaE1=2*spin[i, j]*(spin[p[i], j]+spin[i, p[j]]+spin[m[i], j]+spin[i, m[j]])
            deltaE2=2*spin[i2,j2]*(spin[p[i2], j2]+spin[i2,p[j2]]+spin[m[i2], j2]+spin[i2, m[j2]])
            deltaE=deltaE1+deltaE2
            W=np.exp(-deltaE/kT)
            r=rando()

            if np.linalg.norm(np.subtract([i,j],[i2,j2])%N) == 1:
            # if the difference is == 1 or == 49 then:
                deltaE += 4
            if np.linalg.norm(np.subtract([i,j],[i2,j2])) == 49:
                deltaE += 4

            if r<W:
                itemp=np.copy(spin[i,j])
                spin[i,j]=spin[i2,j2]
                spin[i2,j2]=itemp 


                        
        #occasionally plot or update measurements, eg every 10 sweeps
        if(n%10==0) and analysis == str("no"): 
        #   update measurements
        #   show animation
            plt.cla()
            im=plt.imshow(spin, animated=True, vmin = -1, vmax = 1)
            plt.draw()
            plt.pause(0.0001)

        if (n%10==0) and analysis == str("yes"):
            if n >= 100:
            


                energy_tot = energy_cal(spin)


                energy.append(energy_tot)

    if analysis == str("yes"):
        
        
        ener_av.append(np.mean(energy))
        heat_cap_val = (np.var(energy)) / (N*N*kT**2)
        heat_cap.append(heat_cap_val)

        hc_error_val = error_cal(energy)
        hc_error.append(hc_error_val)


 


if meth == str("glauber"):
    for element in tqdm(temp_grid):
        glauber(float(element))

if meth == str("kawasaki"):
    for element in tqdm(temp_grid):
        kawasaki(float(element))



plt.show();


# plots
if analysis == str("yes"):

    if meth == str("glauber"):

        file = open("glauber_ising_model.txt", "w")

        file.write("magnetisation:\n")
        file.write(str(np.abs(mag_av)))
        file.write("\n")
        file.write("susceptibility:\n")
        file.write(str(sus))
        file.write("\n")
        file.write("Average Energy:\n")
        file.write(str(ener_av))
        file.write("\n")
        file.write("heat capacity:\n")
        file.write(str(heat_cap))
        file.write("\n")
        file.write("heat capacity ERROR:\n")
        file.write(str(hc_error))

        file.close()
        #ener_sample = np.random.choice(energy_all, 1000)
        #heat_cap = np.var(ener_sample)/(N*kT)
        #heat_error = np.std(heat_cap) 

        fig, ax = plt.subplots(2,2)


        ax[0,0].plot(temp_grid, mag_av, "bx")
        ax[0,0].set_title("Magnetisation - Glauber dynamics")
        ax[0,0].set_xlabel("T")
        ax[0,0].set_ylabel("M")
        #ax[0,0].savefig("magnetisation_glauber.pdf")
        

        ax[0,1].plot(temp_grid, sus, "rx")
        ax[0,1].set_title("Susceptibility - Glauber dynamics")
        ax[0,1].set_xlabel("T")
        ax[0,1].set_ylabel("Susceptibility")
        #ax[1,0].savefig("suscep_glauber.pdf")
        

        ax[1,0].plot(temp_grid, ener_av, "bx")
        ax[1,0].set_title("Average energy - Glauber dynamics")
        ax[1,0].set_ylabel("Average Energy")
        ax[1,0].set_xlabel("T")


        #heat_error = bootstrap(energy_all, np.var, confidence_level=0.68)

        #plt.errorbar(temp_grid, heat_cap, yerr = heat_error)
        ax[1,1].errorbar(temp_grid, heat_cap,  yerr = hc_error, marker = "x", linestyle = "")
        ax[1,1].set_title("Heat capacity - Glauber dynamics")
        ax[1,1].set_ylabel("Heat Capacity")
        ax[1,1].set_xlabel("T")
        #ax[0,1].savefig("heat_cap_glauber.pdf")
        #ax[0,1].show();

        fig.tight_layout()
        fig.savefig("glauber_temp_analysis.pdf")
        fig.show();
    


    if meth == str("kawasaki"):

        

        fig, (ax1, ax2) = plt.subplots(1,2)


        

        ax1.plot(temp_grid, ener_av, "bx")
        ax1.set_title("Average energy - Glauber dynamics")
        ax1.set_ylabel("Average Energy")
        ax1.set_xlabel("T")


        #heat_error = bootstrap(energy_all, np.var, confidence_level=0.68)

        #plt.errorbar(temp_grid, heat_cap, yerr = heat_error)
        ax2.errorbar(temp_grid, heat_cap,  yerr = hc_error, marker = "x", linestyle = "")
        ax2.set_title("Heat capacity - Glauber dynamics")
        ax2.set_ylabel("Heat Capacity")
        ax2.set_xlabel("T")
        #ax[0,1].savefig("heat_cap_glauber.pdf")
        #ax[0,1].show();

        fig.tight_layout()
        fig.savefig("kawasaki_temp_analysis.pdf")
        fig.show();

        file = open("kawasaki_ising_model.txt", "w")

        file.write("Average Energy:\n")
        file.write(str(ener_av))
        file.write("\n")
        file.write("heat capacity:\n")
        file.write(str(heat_cap))
        file.write("\n")
        file.write("heat capacity ERROR:\n")
        file.write(str(hc_error))
    

        file.close()




print("End of the code ")



