
import numpy as np
from scipy.integrate import solve_ivp
from equations import *

def RK45(SV):
    def derivatives(t, SV):
        #X = SV[:400]
        #Y = SV[400:800]
        #Z = SV[800:1200]
        #VX = SV[1200:1600]
        #VY = SV[1600:2000]
        #VZ = SV[2000:2400]
        #R = SV[2400:2800]
        #P = SV[2800:3200]
        #E = SV[3200:3600]

        # Call equations: internal force calc, then artificial viscous, and external force
        drhodt = density_change(SV)
        dvdt = velocity_change(SV)
        dEdt = energy_change(SV)
        dposdt = position_change(SV)
        dpdt = pressure_change(SV)
         


        SV_dt = np.array([dposdt, SV[400:800], SV[800:1200], dvdt, SV[1600:2000], SV[2000:2400], drhodt, SV[400:800], dEdt])
        SV_dt = SV_dt.flatten()

        return SV_dt

    t_span = (0,1)
    SV = SV.flatten()
    new_SV = solve_ivp(
            derivatives,
            t_span,
            SV,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
            max_step=1,
            vectorized=False # IMPLEMENT THIS!
    )

    print(new_SV.y)

    return new_SV.y
