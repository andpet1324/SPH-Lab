
import numpy as np
import matplotlib.pyplot as plt
from equations import *
from generation import *
from integrator import *

number_of_particles = 400
dimensions = 1

vector_of_states = np.empty(number_of_particles)

class StateVector:
    
    # state vector. Consider putting into its own class
    SV = np.empty(1)

    def __init__(self, state): # Input initial conditions of particle
        #init the state vector for the particle
        SV = state
        self.SV = SV
        self.X = self.SV[0]
        self.Y = self.SV[1]
        self.Z = self.SV[2]
        self.POS = self.SV[:3]
        self.VX = self.SV[3]
        self.VY = self.SV[4]
        self.VZ = self.SV[5]
        self.VEL = self.SV[2:6]
        self.R = self.SV[6]
        self.P = self.SV[7]
        self.E = self.SV[8]

    def updateState(self, SV):
        # Call equations: internal force calc, then artificial viscous, and external force
        
        new_SV = RK45(SV)

        self.SV = new_SV
        self.X = new_SV[0]
        self.Y = new_SV[1]
        self.Z = new_SV[2]
        self.POS = new_SV[:3]
        self.VX = new_SV[3]
        self.VY = new_SV[4]
        self.VZ = new_SV[5]
        self.VEL = new_SV[2:6]
        self.R = new_SV[6]
        self.P = new_SV[7]
        self.E = new_SV[8]

        return 1

def plot(state):
    plt.plot(state.X, state.R, label='Density')
    plt.plot(state.X, state.E, label='Energy')
    plt.plot(state.X, state.P, label='Pressure')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    initial_particles = generate_tube_particles(number_of_particles)
    
    state = StateVector(initial_particles)

    SV = state.SV

    #plt.scatter(state.X, W_rate[100])
    #plt.show()

    #plt.plot(state.X, density_field)
    #plt.show()
    
    state.updateState(SV)
