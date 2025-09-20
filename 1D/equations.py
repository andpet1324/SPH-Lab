import numpy as np
#np.set_printoptions(threshold=np.inf)

h = 0.1
m = 0.001875

# For a vector, [:, None] gives the i's and [None, :] gives the j's

def nearest_neighbour(SV):
    X = SV[:400]
    r = (X[:, None] - X[None, :])/h # shape (N,N)
    R = np.abs(r)
    dx = np.sign(r)
    return r, R, dx

def kernel_matrix(SV, r, R, dx):
    W = np.where(
            (0 <= R) & (R < 1),
            ((2/3) - R**2 + 1/2*R**3),
            np.where(
            (1 <= R) & (R < 2),
            ((1/6) * (2-R)**3),
            0
            )
        )
    return W

def kernel_deriv(SV, r, R, dx):
    alpha = 1/h
    W_prime = np.where(
            (0 <= R) & (R < 1),
            alpha * (-2 + 1.5 * R) * (dx / h**2),
            np.where(
            (1 <= R) & (R < 2),
            -alpha * (1/2) * (2 - R)**2 * (dx / (h * r)),
            0
            )
        )
    return W_prime

def density_summation(SV, W):
    # rho_i = sum m_j W_ij, but m_j = m
    densities = m * np.sum(W, axis=1)
    return densities

def density_change(SV, W_prime):
    rho_dt = m * np.sum(W_prime, axis=1)
    return rho_dt

def velocity_change(SV, W_prime, rho_dens):
    p = SV[800:1200]
    v_dt = -m * np.sum(((p/rho_dens**2)[:, None] + (p/rho_dens**2)[None,:]) * W_prime, axis=1)
    return v_dt

def energy_change(SV, W_prime, rho_dens):
    v = SV[400:800]
    p = SV[1200:1600]
    E_dt = (1/2) * m * np.sum(((p/rho_dens**2)[:, None] + (p/rho_dens**2)[None,:]) * (v[:,None] - v[None,:]) * W_prime, axis=1)
    return E_dt

def position_change(SV):
    v = SV[400:800]
    pos_dt = v
    return pos_dt

def pressure_change(SV, rho_dens):
    E = SV[1600:2000]
    gamma = 1.4
    dpdt = (gamma-1)*rho_dens*E
    return dpdt

