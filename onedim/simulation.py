from constants import *
from grid import Grid1D

class Simulation:

    def __init__(self, a_inputs):
        self.inputs = a_inputs

        self.grid = Grid1D(self.inputs.xlim, self.inputs.nx, self.inputs.numghosts, NUMQ)

        self.grid.fill_grid(ics.sod_shock_tube)
        
        self.grid.apply_zero_gradient_bcs()



    def run(self):


        #os.makedirs('simulation_frames', exist_ok=True)


        while (t < t_finish) and timestepNum < Nt:


            print("Timestep: " , timestepNum, "  Current time: ", t)
            
            self.grid.apply_zero_gradient_bcs()

            if makeMovie:
                fig, axs = plt.subplots(4, 1, figsize=(10, 15))

                axs[0].plot(grid_weno.x, grid_weno.grid[RHOCOMP, :], 'o', label='FD-Prim WENO',c='red')

                axs[0].set_title(f'Time: {t:.2f}')
                axs[0].set_ylabel("Density")
                axs[0].legend()

                axs[1].plot(grid_weno.x, grid_weno.grid[UCOMP, :], 'o', label='FD-Prim WENO',c='red')
                axs[1].set_ylabel("Velocity")
                axs[1].legend()

                axs[2].plot(grid_weno.x, grid_weno.grid[PCOMP, :], 'o', label='FD-Prim WENO',c='red')
                axs[2].set_ylabel("Pressure")
                axs[2].legend()



                axs[3].plot(grid_weno.x, grid_weno.shock_mask, 'o', label='FD-Prim WENO',c='red')
                
                axs[3].set_ylabel("Shock Mask")
                axs[3].legend()


                plt.savefig(f'simulation_frames/frame_{timestepNum:04d}.png')
                plt.close(fig) 

