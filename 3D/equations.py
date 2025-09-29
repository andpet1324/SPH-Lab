import numpy as np
import scipy.constants as constants
#np.set_printoptions(threshold=np.inf)

h = 1e7
# For a vector, [:, None] gives the i's and [None, :] gives the j's

def nearest_neighbour(SV, n):
    X = SV[:3*n]
    X = X.reshape(3, n).T

    dx = (X[:,None] - X[None,:])
    r = np.linalg.norm(dx, axis=2)
    np.fill_diagonal(r, np.inf)
    R = r/h

    return r, R, dx

def kernel_matrix(r, R, dx):

    cond1 = (0<=R) & (R<1)
    cond2 = (1<=R) & (R<2)

    W = np.zeros_like(r, dtype=float)
    alpha = 3/(2*np.pi*h**3)

    W[cond1] = alpha * ((2/3) - R[cond1]**2 + (1/2)*R[cond1]**3)
    W[cond2] = alpha * ((1/6) * (2-R[cond2])**3)
    W[r == 0] = 0.0

    return W

def kernel_deriv(r, R, dx):

    cond1 = (0<=R) & (R<1)
    cond2 = (1<=R) & (R<2)

    W_prime = np.zeros_like(dx, dtype=float)
    alpha = 3/(2*np.pi*h**3)

    W_prime[cond1] = alpha * (-2 + 1.5*R[cond1])[:, None] * (dx[cond1] / h**2)
    W_prime[cond2] = (-alpha * 0.5 * (2-R[cond2])**2)[:, None] * (dx[cond2] / (h * (r[cond2] + 1e-9))[:, None])
    W_prime[r == 0] = 0.0

    return W_prime

def kernel_deriv_gravity(r, R, dx):

    cond1 = (0<=R) & (R<1)
    cond2 = (1<=R) & (R<2)
    cond3 = (R>=2)

    W_prime = np.zeros_like(dx, dtype=float)
    
    W_prime[cond1] = (1/h**2) * ((4/3)*R[cond1][:,None] - ((6/5)*R[cond1]**3)[:,None]+ ((1/2)*R[cond1]**4)[:,None])
    W_prime[cond2] = (1/h**2) * (((8/3)*R[cond2])[:,None] - (3*R[cond2]**2)[:,None] + ((6/5)*R[cond2]**3)[:,None] - ((1/6)*R[cond2]**4)[:,None] - ((1/(15*R[cond2]**2))[:,None]))
    W_prime[cond3] = 1/(r[cond3]**2)[:,None]

    return W_prime

def speed_of_sound(SV, e):
    gamma = 1.4
    c = np.sqrt((gamma-1)*e)
    return c

def viscosity(SV, n, e, rho):
    def average_c(SV, e):
        c = speed_of_sound(SV, e)
        cij = (1/2) * (c[:,None] + c[None,:])
        return cij
    def average_rho(SV, rho):
        #rho = SV[7*n:8*n]
        rhoij = (1/2) * (rho[:,None] + rho[None,:])
        return rhoij
    def phi(SV, Xij, Vij):
        hij = h
        small_phi = 0.1 * hij
        phi = (hij * np.sum(Vij * Xij, axis=2)) / (np.sum(Xij**2, axis=2) + small_phi**2)
        return phi

    alpha = 1
    beta = 1
    X = SV[:3*n]
    X = X.reshape(3, n).T
    V = SV[3*n:6*n]
    V = V.reshape(3,n).T
    Xij = X[:,None] - X[None,:]
    Vij = V[:,None] - V[None,:]

    Pi = np.zeros((n,n)) 

    cond = (np.sum(Vij * Xij,axis=2)) < 0

    Pi[cond] = (-alpha * average_c(SV, e)[cond] * phi(SV, Xij, Vij)[cond] + beta * phi(SV, Xij, Vij)[cond]**2) / (average_rho(SV, rho)[cond] + 1e-9)

    return Pi

def density_summation(SV, W, n):
    m = SV[6*n:7*n]
    rho = np.sum(m * W, axis=1)
    return rho

def velocity_change(SV, W_prime, n, e, rho, p):
    p = SV[8*n:9*n]
    rho = SV[7*n:8*n]
    m = SV[6*n:7*n]
    v_dt = -1 * np.sum(m[:,None] * ((p[:,None]/(rho[:,None]**2 + 1e-9)) + (p[None,:]/(rho[None,:]**2 + 1e-9)) + viscosity(SV, n, e, rho))[:, :, None] * W_prime, axis=1)
    return v_dt

def velocity_change_gravity(SV, phi_prime, r, R, dx, n):
    G = constants.G
    m = SV[6*n:7*n]

    v = -G * np.sum(m[:,None] * phi_prime * (dx / ((r[:,:,None]))),axis=1)

    return v

def energy_change(SV, W_prime, n):
    v = SV[3*n:6*n]
    v = v.reshape(3,n).T
    p = SV[8*n:9*n]
    rho = SV[7*n:8*n]
    m = SV[6*n:7*n]
    E_dt = (1/2) * np.sum(m[:,None] * ((p[:,None]/rho[:,None]**2) + (p[None,:]/rho[None,:]**2)) * np.sum((v[:, None] - v[None, :]) * W_prime, axis=2), axis=1)
    return E_dt

def position_change(SV, n):
    v = SV[3*n:6*n]
    v = v.reshape(3,n).T
    pos_dt = v
    return pos_dt

def pressure(SV, e, n, rho):
    gamma = 1.4
    dpdt = (gamma-1)*rho*e
    return dpdt

def new_e(SV, n, rho):
    p = SV[8*n:9*n]
    gamma = 1.4
    e = p/((gamma-1)*rho + 1e-9)
    return e
