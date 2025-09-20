
import numpy as np
from scipy.integrate import solve_ivp
from equations import *
import sys

def RK45(SV):
    def derivatives(t, SV, dt, steps):
        tot_time = dt*steps
        time_left = tot_time-t
        percent = int(100 - 100 * time_left/tot_time)
        if percent % 10 == 0 and percent != 0:
            bar = "█" * int(percent/2) + "-" * int(1.2*steps - percent/2)
            sys.stdout.write(f"\r|{bar}| {percent:3d}%")
            sys.stdout.flush()
            if time_left == tot_time:
                print()  # move to next line when done

        # Call equations: internal force calc, then artificial viscous, and external force
        #
        # New step, so calculate W, W_prime and rho_dens
        r, R, dx = nearest_neighbour(SV)
        W = kernel_matrix(SV, r, R, dx)
        W_prime = kernel_deriv(SV, r, R, dx)
        rho_dens = density_summation(SV, W)

        drhodt = density_change(SV, W_prime)
        dvdt = velocity_change(SV, W_prime, rho_dens)
        dEdt = energy_change(SV, W_prime, rho_dens)
        dposdt = position_change(SV)
        dpdt = pressure_change(SV, rho_dens)
         
        SV_dt = np.array([dposdt, dvdt, drhodt, dpdt, dEdt])
        SV_dt = SV_dt.flatten()

        return SV_dt

    SV = SV.flatten()

    dt = 0.005
    steps = 40
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
            args=(dt,steps),
            vectorized=False # IMPLEMENT THIS!
    )
    print("Finished!")

    print(new_SV.y)

    return new_SV.y
