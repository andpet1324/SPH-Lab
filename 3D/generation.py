import numpy as np
import pandas as pd

def generate_planet_particles(filename):

    df = pd.read_csv(filename, delim_whitespace=True) 
    SV = df.to_numpy()

    return SV

def seperate_particles(data, sep):
    # Make two planets!

    data[:,0] += sep
    data[:,1] += sep
    data[:,2] += sep
    
    return data

def edit_velocities(data, vel, positive=True):

    if positive:
        data[:,3] += vel
        data[:,4] += vel
        data[:,5] += vel
    else:
        data[:,3] -= vel
        data[:,4] -= vel
        data[:,5] -= vel

    return data
