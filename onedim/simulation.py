from onedim.grid import Grid1D
from onedim.constants import *
import onedim.ics as ics
import glob
import matplotlib.animation as animation

from onedim.flux import Flux


from onedim.euler import prim_to_cons_var, flux_var,cons_to_prim

from onedim.reconstruct import weno5_reconstruction
from onedim.bcs import BoundaryConditions

import numpy as np
import matplotlib.pyplot as plt
import os

class Simulation:
    def __init__(self, a_inputs):
        self.inp = a_inputs

        self.grid = Grid1D(self.inp.xlim, self.inp.nx, self.inp.numghosts, NUMQ)
        self.bcs = BoundaryConditions(self.grid, self.inp.bc_lo, self.inp.bc_hi)
        self.flux = Flux(self.grid, self.inp.flux)


        self.applyICS()
        self.bcs.apply_bcs()

        self.t = self.inp.t0
        self.timestepNum = 0

 
        #-1 is no output. Always output ICs if we are outputting.
        if self.inp.output_freq >= 0:
            self.output()

    def run(self):
        # os.makedirs('simulation_frames', exist_ok=True)


        while (self.t < self.inp.t_finish) and self.timestepNum < self.inp.nt:
            print("Timestep: ", self.timestepNum, "  Current time: ", self.t)

            self.bcs.apply_bcs()


            self.grid.assert_variable_type("prim")
            a = np.sqrt(gamma * self.grid.grid[PCOMP] / self.grid.grid[RHOCOMP])
            max_speed = np.max(np.abs(self.grid.grid[UCOMP]) + a)
            dt = min(self.inp.cfl * self.grid.dx / max_speed, self.inp.t_finish - self.t)
            
            U_new = np.ones_like(self.grid.grid) / 0  # np.nans_like lol

            #returns numerical flux and conservative varaibles at interface
            consP, numericalFlux = self.flux.getFlux()

            # Update.
            for i in range(self.grid.Nghost, self.grid.Nx + self.grid.Nghost):
                for icomp in range(NUMQ):
                    U_new[icomp, i] = consP[icomp, i] - (dt / self.grid.dx) * (
                        numericalFlux[icomp, i] - numericalFlux[icomp, i - 1]
                    )

            self.grid.set(U_new)
            self.grid.transform(cons_to_prim, "prim")
            
            self.bcs.apply_bcs()


            self.timestepNum += 1
            self.t += dt

            if self.timestepNum % self.inp.output_freq == 0:
                
                self.output()


            # DEBUG
            for i in range(len(self.grid.grid[0])):
                if self.grid.grid[PCOMP, i] <= 0:
                    print("Bad cell: ", i)
                    assert False
                for icomp in range(NUMQ):
                    if np.isnan(self.grid.grid[icomp, i]):
                        print("Nan cell : ", i)
                        assert False

        if self.inp.make_movie:
            self.generate_movie()
    
        print("SUCCESS!")
        return self.grid

    def plot(self):
        if not os.path.exists(self.inp.output_dir):
            os.makedirs(self.inp.output_dir)

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        axs[0].scatter(self.grid.x, self.grid.grid[RHOCOMP, :], c="black")
        axs[0].set_ylabel("Density")

        axs[1].scatter(self.grid.x, self.grid.grid[UCOMP, :],  c="black")
        axs[1].set_ylabel("Velocity")

        axs[2].scatter(self.grid.x, self.grid.grid[PCOMP, :],  c="black")
        axs[2].set_ylabel("Pressure")

        axs[0].set_title(f"Time: {self.t:.4f}")
        plt.savefig(f"{self.inp.output_dir}/plot_dt{str(self.timestepNum).zfill(6)}")
        plt.close()

    

    def applyICS(self):

        if self.inp.ics == "sodshocktube":
            self.grid.fill_grid(ics.sod_shock_tube)
        elif self.inp.ics == "shuoshershocktube":
            self.grid.fill_grid(ics.shu_osher_shock_tube)
        else:
            raise RuntimeError("[FLUID] ICS not valid.")

        

    def output(self):
        if not os.path.exists(self.inp.output_dir):
            os.makedirs(self.inp.output_dir)
        
        # File naming convention: output_timestepNum.txt
        output_filename = os.path.join(self.inp.output_dir, f"output_{str(self.timestepNum).zfill(6)}.txt")
        
        with open(output_filename, 'w') as f:
            # Write header
            f.write(f"# Time: {self.t:.4f}\n")
            f.write("# x, density, velocity, pressure\n")
            
            # Write data
            for i in range(len(self.grid.x)):
                x = self.grid.x[i]
                density = self.grid.grid[RHOCOMP, i]
                velocity = self.grid.grid[UCOMP, i]
                pressure = self.grid.grid[PCOMP, i]
                f.write(f"{x:.8f}, {density:.8f}, {velocity:.8f}, {pressure:.8f}\n")



    def generate_movie(self):
        # Create a directory for the frames if it doesn't exist
        frames_dir = os.path.join(self.inp.output_dir, "frames")
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        # List all the output files and sort them
        output_files = sorted(glob.glob(os.path.join(self.inp.output_dir, "output_*.txt")))

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        def update_plot(file):
            data = np.loadtxt(file, delimiter=',', skiprows=2)
            x = data[:, 0]
            density = data[:, 1]
            velocity = data[:, 2]
            pressure = data[:, 3]

            with open(file, 'r') as f:
                lines = f.readlines()
                time_line = lines[0]
                time = float(time_line.split(':')[1].strip())

            timestep = int(file.split('_')[-1].split('.')[0])

            axs[0].clear()
            axs[1].clear()
            axs[2].clear()

            axs[0].scatter(x, density, c="black")
            axs[0].set_ylabel("Density")

            axs[1].scatter(x, velocity, c="black")
            axs[1].set_ylabel("Velocity")

            axs[2].scatter(x, pressure, c="black")
            axs[2].set_ylabel("Pressure")

            axs[0].set_title(f"Time: {time:.4f}, Timestep: {timestep}")
        # Create an animation by updating the plot for each output file
        ani = animation.FuncAnimation(fig, update_plot, frames=output_files, repeat=False)

        # Save the animation as a movie file using ffmpeg
        movie_filename = os.path.join(self.inp.output_dir, "simulation_movie.mp4")
        ani.save(movie_filename, writer='ffmpeg', fps=10)

        print(f"Movie saved as {movie_filename}")
