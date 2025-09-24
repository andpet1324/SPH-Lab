import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def make_video(new_SV, filename="sph_evolution.mp4", dpi=400, fps=2):
    """
    Make a video using the time evolution stored in new_SV (the solve_ivp result).
    Plots Density (R) and Energy (E) vs Position (X) at each timestep.
    """

    # Each chunk is size 400
    X_all = new_SV.y[0:400, :]       # positions
    R_all = new_SV.y[800:1200, :]    # densities
    E_all = new_SV.y[1600:2000, :]   # energies
    P_all = new_SV.y[1200:1600, :]  # pressures
    V_all = new_SV.y[400:800, :]    # velocities
    times = new_SV.t

    fig, ax = plt.subplots(figsize=(6,4), dpi=dpi)
    line_density, = ax.plot([], [], label="Density")
    line_energy, = ax.plot([], [], label="Energy")
    line_pressure, = ax.plot([], [], label="Pressure")
    line_velocity, = ax.plot([], [], label="Velocity")

    ymax = max(R_all.max(), E_all.max(), P_all.max())
    ax.set_ylim(0, ymax*1.2)
    ax.set_xlim(-0.4,0.4)
    ax.legend()

    def init():
        line_density.set_data([], [])
        line_energy.set_data([], [])
        line_pressure.set_data([], [])
        line_velocity.set_data([], [])
        return line_density, line_energy, line_pressure, line_velocity

    def update(frame):
        X = X_all[:, frame]
        R = R_all[:, frame]
        E = E_all[:, frame]
        P = P_all[:, frame]
        V = V_all[:, frame]
        line_density.set_data(X, R)
        line_energy.set_data(X, E)
        line_pressure.set_data(X, P)
        line_velocity.set_data(X, V)
        ax.set_title(f"t = {times[frame]:.3f}")
        return line_density, line_energy, line_pressure, line_velocity

    ani = animation.FuncAnimation(
        fig, update, frames=X_all.shape[1],
        init_func=init, blit=True, interval=200
    )

    ani.save(filename, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"Video saved to {filename}")

def make_figures(new_SV, dpi=400, xrange=(-0.4, 0.4)):

    X = new_SV.y[0:400,-1]       # positions
    R = new_SV.y[800:1200,-1]    # densities
    E = new_SV.y[1600:2000,-1]   # energies
    P = new_SV.y[1200:1600,-1]   # pressure
    V = new_SV.y[400:800,-1]     # velocity

    mask = (X >= xrange[0]) & (X <= xrange[1])
    X = X[mask]

    y_list = [R, E, P, V]
    y_names = ["Density", "Energy", "Pressure", "Velocity"]
    y_units = [r' [kg/m$^3$]', r' [J/Kg]', r' [N/m$^2$]', r' [m/s]']

    for i,y in enumerate(y_list):

        y = y[mask]

        fig, ax = plt.subplots(figsize=(6,4), dpi=dpi)

        ymax = y.max()
        ymin = y.min()
        ax.set_ylim(ymin*0.8, ymax*1.2)
        ax.set_xlim(-0.4,0.4)
        ylabel = y_names[i] + y_units[i]
        ax.set_ylabel(ylabel)
        ax.set_xlabel("X [m]")

        ax.plot(X,y)
        figname = y_names[i] + ".png"
        fig.savefig(figname, dpi=dpi)
