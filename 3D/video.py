import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def make_video(new_SV, n, sep, filename="sph_evolution.mp4", dpi=400, fps=30, dot_size=5):
    """
    Make a 2D scatter video using the time evolution stored in new_SV.
    X and Y positions are plotted with color representing Density (R).
    """

    n_frames = new_SV.y.shape[1]

    # Extract X, Y positions and density
    X_all = new_SV.y[0*n:1*n, :]  # X positions
    Y_all = new_SV.y[1*n:2*n, :]  # Y positions
    R_all = new_SV.y[7*n:8*n, :]  # Density
    times = new_SV.t

    fig, ax = plt.subplots(figsize=(6,6), dpi=dpi)
    sc = ax.scatter(X_all[:,0], Y_all[:,0], c=R_all[:,0], cmap='viridis', s=dot_size)
    ax.set_xlim(-10e8 + sep/2, 10e8 + sep/2)
    #ax.set_xlim(-10e8, 10e8)
    ax.set_ylim(-10e8 + sep/2, 10e8 + sep/2)
    #ax.set_ylim(-10e8, 10e8)
    #ax.set_xlim(X_all.min(), X_all.max())
    #ax.set_ylim(Y_all.min(), Y_all.max())

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Density (R)")

    def update(frame):
        sc.set_offsets(np.c_[X_all[:, frame], Y_all[:, frame]])
        sc.set_array(R_all[:, frame])
        ax.set_title(f"t = {times[frame]:.2e} s")
        return sc,

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
    ani.save(filename, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"2D scatter video saved to {filename}")

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
