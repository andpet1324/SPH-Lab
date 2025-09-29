
import numpy as np
import matplotlib.pyplot as plt
from equations import *
from generation import *
from integrator import *

data_file = "~/Lund/FYSN33/Labs/SPH/SPH-Lab/3D/Data/Planet300.dat"
#number_of_particles = int(((data_file.split('/')[-1]).split('.')[0])[6:])

seperation = 3e8
vel = 1e6

#vector_of_states = np.empty(number_of_particles)

class StateVector:
    
    # state vector. Consider putting into its own class
    SV = np.empty(1)

    def __init__(self, state1, state2, n): # Input initial conditions of particle
        #init the state vector for the particle
        SV1 = state1
        SV2 = state2
        self.X = np.array([np.concatenate((SV1[:,0],SV2[:,0])), 
                           np.concatenate((SV1[:,1],SV2[:,1])),
                           np.concatenate((SV1[:,2],SV2[:,2]))])
        self.V = np.array([np.concatenate((SV1[:,3],SV2[:,3])), 
                           np.concatenate((SV1[:,4],SV2[:,4])), 
                           np.concatenate((SV1[:,5],SV2[:,5]))])    
        self.M = np.concatenate((SV1[:,6],SV2[:,6])) # SV[6n:7n]
        self.R = np.concatenate((SV1[:,7],SV2[:,7])) # SV[7n:8n]
        self.P = np.concatenate((SV1[:,8],SV2[:,8])) # SV[8n:9n]

    def getSV(self):
        SV = np.array([self.X[0], self.X[1], self.X[1],
                       self.V[0], self.V[1], self.V[2],
                       self.M,
                       self.R,
                       self.P])
        return SV


    def updateState(self, SV, n, sep):
        # Call equations: internal force calc, then artificial viscous, and external force
        
        new_SV = RK45(SV, n, sep)
        # Unflatten!

        self.X = np.array([new_SV[:n], new_SV[n:2*n], new_SV[2*n:3*n]])
        self.V = np.array([new_SV[3*n:4*n], new_SV[4*n:5*n], new_SV[5*n:6*n]])
        self.M = new_SV[6*n:7*n]
        self.R = new_SV[7*n:8*n]
        self.P = new_SV[8*n:9*n]

        return 1

def plot(state):
    plt.plot(state.X, state.R, label='Density')
    plt.plot(state.X, state.E, label='Energy')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    initial_particles1 = generate_planet_particles(data_file)
    initial_particles2 = generate_planet_particles(data_file)

    initial_particles2 = seperate_particles(initial_particles2, seperation)

    initial_particles1 = edit_velocities(initial_particles1, vel, positive=True)
    initial_particles2 = edit_velocities(initial_particles2, vel, positive=False)

    n = initial_particles1.shape[0] + initial_particles2.shape[0]

    state = StateVector(initial_particles1, initial_particles2, n)

    state.updateState(state, n, seperation)
