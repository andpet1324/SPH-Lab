
import numpy as np
from scipy.integrate import solve_ivp
from equations import *
from video import *
import sys

def progress(t, dt, steps):
    tot_time = dt*steps
    time_left = tot_time-t
    percent = int(100 - 100 * time_left/tot_time)
    if percent % 10 == 0 and percent != 0:
        bar = "█" * int(percent/2) + "-" * int(1.2*steps - percent/2)
        sys.stdout.write(f"\r{percent:3d}%")
        sys.stdout.flush()
        if time_left == tot_time:
            print()  # move to next line when done

def RK45(state, n, sep): # n is number of particles
    empty = np.zeros(n)
    def derivatives(t, SV, dt, steps, n):

        progress(t, dt, steps)

        # Call equations: internal force calc, then artificial viscous, and external force
        #
        # New step, so calculate W, W_prime and rho_dens
        #
        # Calculate the pressure at timestep
        #
        
        r, R, dx = nearest_neighbour(SV, n)
        W = kernel_matrix(r, R, dx)
        W_prime = kernel_deriv(r, R, dx)

        #fig = plt.figure(figsize=(7,6))
        #ax = fig.add_subplot(111, projection='3d')
        #ax.plot_surface(SV[:n], SV[n:2*n], W)
        #plt.show()

        #plt.plot(SV[:400], W[300])
        #plt.plot(SV[:400], W_prime[300])
        #plt.show()

        # Pressure is dependent on rho so needs to be called before!
        rho = density_summation(SV, W, n)
        #SV[7*n:8*n] = rho
        e = new_e(SV, n, rho)
        p = pressure(SV, e, n, rho)
        SV[8*n:9*n] = p

        phi_prime = kernel_deriv_gravity(r, R, dx)
        dvdt = velocity_change(SV, W_prime, n, e, rho, p) + velocity_change_gravity(SV, phi_prime, r, R, dx, n)
        dposdt = position_change(SV, n)

        SV_dt = np.array([dposdt.T[0], dposdt.T[1], dposdt.T[2], 
                          dvdt.T[0], dvdt.T[1], dvdt.T[2],
                          empty,
                          empty,
                          empty])
        SV_dt = SV_dt.flatten()
        return SV_dt

    SV = state.getSV()
    SV = SV.flatten()

    dt = 20
    steps = 800
    t_eval = np.linspace(0, dt*steps, steps+1)
    print("Integrating...")
    new_SV = solve_ivp(
            derivatives,
            (0, dt*steps),
            SV,
            method="RK45",
            rtol=1,
            atol=1e-3,
            max_step=dt,
            t_eval=t_eval,
            args=(dt,steps,n)
    )
    print("Finished!")

    make_video(new_SV, n, sep)
    #make_figures(new_SV)

    return new_SV.y[:,-1]
