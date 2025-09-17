
import numpy as np
import matplotlib.pyplot as plt
import equations as eq

number_of_particles = 400

vector_of_states = np.empty(number_of_particles)

class StateVector:
    
    # state vector. Consider putting into its own class
    SV = np.empty(1)

    def __init__(self, state): # Input initial conditions of particle
        #init the state vector for the particle
        self.SV = state
        self.X = self.SV[0]
        self.Y = self.SV[1]
        self.Z = self.SV[2]
        self.VX = self.SV[3]
        self.VY = self.SV[4]
        self.VZ = self.SV[5]
        self.R = self.SV[6]
        self.P = self.SV[7]
        self.E = self.SV[8]

    def updateState():
        # Call equations
        

        return 1

    def getPosition():
        return 1

    def getVelocity():
        return 1

def generate_tube_particles(): # Should return a vector containing with Particle class particles
    lowrange = np.linspace(-0.6,0.0,321) # should be 320 but we will remove 1 particle
    highrange = np.linspace(0.0,0.6,80)
    # we don't want two particles overlapping at 0.0
    lowrange = lowrange[np.arange(lowrange.size - 1)]
    positions = np.array([np.append(lowrange,highrange), np.zeros(number_of_particles), np.zeros(number_of_particles)])

    velocities = np.array([np.zeros(number_of_particles),np.zeros(number_of_particles),np.zeros(number_of_particles)])

    densities = np.array(np.append(np.ones(320),np.ones(80)-0.75))
    pressures = np.array(np.append(np.ones(320),np.ones(80)-(1-0.1795)))
    energies = np.array(np.append(np.ones(320)+1.5, np.ones(80)+0.795))
    

    particles = np.array([positions[0], positions[1], positions[2], velocities[0], velocities[1], velocities[2], densities, pressures, energies])

    return(particles)

def plot_state(state):
    plt.plot(state.X, state.R, label='Density')
    plt.plot(state.X, state.E, label='Energy')
    plt.plot(state.X, state.P, label='Pressure')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    initial_particles = generate_tube_particles()
    
    state = StateVector(initial_particles)

    plot_state(state)

