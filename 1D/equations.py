import numpy as np
#np.set_printoptions(threshold=np.inf)

h = 0.01
m = 0.001875

# For a vector, [:, None] gives the i's and [None, :] gives the j's

def nearest_neighbour(SV):
    X = SV[:400]

    dx = (X[:,None] - X[None,:])
    r = np.abs(dx)
    R = r/h
    return r, R, dx

def kernel_matrix(r, R, dx):

    cond1 = (0<=R) & (R<1)
    cond2 = (1<=R) & (R<2)

    W = np.zeros_like(r, dtype=float)
    alpha = 1/h

    W[cond1] = alpha * ((2/3) - R[cond1]**2 + (1/2)*R[cond1]**3)
    W[cond2] = alpha * ((1/6) * (2-R[cond2])**3)

    return W

def kernel_deriv(r, R, dx):

    cond1 = (0<=R) & (R<1)
    cond2 = (1<=R) & (R<2)

    W_prime = np.zeros_like(r, dtype=float)
    alpha = 1/h

    W_prime[cond1] = alpha * (-2 + (3/2)*R[cond1]) * (dx[cond1]/h**2)
    W_prime[cond2] = -alpha * (1/2)*(2-R[cond2])**2 * (dx[cond2]/(h*r[cond2]))

    return W_prime

def smoothing_length(SV): # REMEMBER THAT THERES A FACTOR FOR DIMENSION!!
    eta = 1.3
    dimension = 1
    rho = SV[800:1200]
    h = eta * (m/rho)**(1/dimension)
    return h

def speed_of_sound(SV):
    gamma = 1.4
    E = SV[1600:2000]
    c = np.sqrt((gamma-1)*E)
    return c

def viscosity(SV):
    def average_h(SV):
        h = smoothing_length(SV)
        hij = (1/2) * (h[:,None] + h[None,:])
        #hij = h
        return hij
    def average_c(SV):
        c = speed_of_sound(SV)
        cij = (1/2) * (c[:,None] + c[None,:])
        return cij
    def average_rho(SV):
        rho = SV[800:1200]
        rhoij = (1/2) * (rho[:,None] + rho[None,:])
        return rhoij
    def phi(SV, Xij, Vij):
        hij = h
        small_phi = 0.1 * hij
        phi = (hij * (Vij * Xij)) / (np.abs(Xij)**2 + small_phi**2)
        return phi

    alpha = 1
    beta = 1
    X = SV[:400]
    V = SV[400:800]
    Xij = X[:,None] - X[None,:]
    Vij = V[:,None] - V[None,:]

    Pi = np.zeros_like(Xij, dtype=float) 

    cond = (Vij * Xij) < 0

    Pi[cond] = (-alpha * average_c(SV)[cond] * phi(SV, Xij, Vij)[cond] + beta * phi(SV, Xij, Vij)[cond]**2) / average_rho(SV)[cond]

    return Pi

def density_summation(W):
    rho = m * np.sum(W, axis=1)
    return rho

def velocity_change(SV, W_prime):
    p = SV[1200:1600]
    rho = SV[800:1200]
    v_dt = -1 * np.sum(m * ((p[:,None]/rho[:,None]**2) + (p[None,:]/rho[None,:]**2) + viscosity(SV)) * W_prime, axis=1)
    return v_dt

def energy_change(SV, W_prime):
    v = SV[400:800]
    p = SV[1200:1600]
    rho = SV[800:1200]
    E_dt = (1/2) * np.sum(m * ((p[:,None]/rho[:,None]**2) + (p[None,:]/rho[None,:]**2) + viscosity(SV)) * ((v[:,None] - v[None,:]) * W_prime), axis=1)
    return E_dt

def position_change(SV):
    v = SV[400:800]
    pos_dt = v
    return pos_dt

def pressure(SV):
    E = SV[1600:2000]
    rho = SV[800:1200]
    gamma = 1.4
    dpdt = (gamma-1)*rho*E
    return dpdt
