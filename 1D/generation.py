import numpy as np

def generate_tube_particles(number_of_particles): # Should return a vector containing with Particle class particles
    lowrange = np.linspace(-0.6,0.0,321) # should be 320 but we will remove 1 particle
    highrange = np.linspace(0.0,0.6,81)
    # we don't want two particles overlapping at 0.0
    lowrange = lowrange[np.arange(lowrange.size - 1)]
    highrange = highrange[1:]
    positions = np.array([np.append(lowrange,highrange), np.zeros(number_of_particles), np.zeros(number_of_particles)])

    velocities = np.array([np.zeros(number_of_particles),np.zeros(number_of_particles),np.zeros(number_of_particles)])

    densities = np.array(np.append(np.ones(320),np.ones(80)-0.75))
    pressures = np.array(np.append(np.ones(320),np.ones(80)-(1-0.1795)))
    energies = np.array(np.append(np.ones(320)+1.5, np.ones(80)+0.795))


    particles = np.array([positions[0], velocities[0], densities, pressures, energies])

    return(particles)
