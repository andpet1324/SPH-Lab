
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
        sys.stdout.write(f"\r|{bar}| {percent:3d}%")
        sys.stdout.flush()
        if time_left == tot_time:
            print()  # move to next line when done

def RK45(SV):
    empty = np.zeros(400)
    def derivatives(t, SV, dt, steps):

        progress(t, dt, steps)

        # Call equations: internal force calc, then artificial viscous, and external force
        #
        # New step, so calculate W, W_prime and rho_dens
        #
        # Calculate the pressure at timestep
        #
        
        r, R, dx = nearest_neighbour(SV)
        W = kernel_matrix(r, R, dx)
        W_prime = kernel_deriv(r, R, dx)

        #plt.plot(SV[:400], W[300])
        #plt.plot(SV[:400], W_prime[300])
        #plt.show()

        # Pressure is dependent on rho so needs to be called before!
        rho = density_summation(W)
        SV[800:1200] = rho
        p = pressure(SV)
        SV[1200:1600] = p

        dvdt = velocity_change(SV, W_prime)
        dEdt = energy_change(SV, W_prime)
        dposdt = position_change(SV)
        

        SV_dt = np.array([dposdt, dvdt, empty, empty, dEdt])
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
            rtol=1e-3,
            atol=1e-6,
            max_step=dt,
            t_eval=t_eval,
            args=(dt,steps),
            vectorized=False # This doesn't work
    )
    print("Finished!")

    print(new_SV)

    make_video(new_SV)
    make_figures(new_SV)

    return new_SV.y[:,-1]
