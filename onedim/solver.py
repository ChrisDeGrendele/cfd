from weno5 import *
from euler import *
from constants import *
import sys

sys.path.append("/Users/chris/Documents/github/gp-recipe/src_new")
from new_driver import GP_recipe1D, GP_recipe2D
import kernels as kern
import grids as old_grid
from riemann import *
import matplotlib.pyplot as plt
import os


def gp_DAS_SE(grid, t0, t_finish, cfl):
    t = t0
    r_gp = 2

    nt = 0

    while t < t_finish:
        print("timestep: ", nt)

        nt += 1

        # Initialize the gradients array with the same shape as the grid
        max_gradients = np.zeros_like(grid.grid[0])

        for icomp in range(NUMQ):
            for i in range(grid.Nghost - 1, grid.Nx + grid.Nghost):
                gradient = np.abs(grid.grid[icomp, i + 1] - grid.grid[icomp, i])
                max_gradients[i] = max(
                    max_gradients[i], gradient
                )  # Keep the maximum gradient across all variables

        # Determine the mask based on the maximum gradients
        threshold = np.mean(max_gradients)  # Define a suitable threshold
        shock_mask = (
            max_gradients > 3 * threshold
        )  # True for non-smooth flow, False for smooth flow

        grid.assert_variable_type("prim")
        a = np.sqrt(gamma * grid.grid[PCOMP] / grid.grid[RHOCOMP])
        max_speed = np.max(np.abs(grid.grid[UCOMP]) + a)
        dt = min(cfl * grid.dx / max_speed, t_finish - t)

        grid.transform(prim_to_cons, "cons")

        # pull out cons variables for update.
        U = grid.grid
        U_new = grid.scratch()

        # back to prim
        grid.transform(cons_to_prim, "prim")

        SE_predict_by_comp = []
        DAS_predict_by_comp = []

        for icomp in range(NUMQ):
            single_comp_grid = old_grid.Grid1D(grid.xlim, grid.Nx, grid.Nghost)
            single_comp_grid.grid = grid.grid[icomp]

            gp_SE = GP_recipe1D(
                single_comp_grid,
                r_gp,
                ell=24 * grid.dx,
                stencil_method="center",
                high_precision=True,
            )
            gp_DAS = GP_recipe1D(
                single_comp_grid, 1, stencil_method="center", high_precision=True
            )

            # we also need 1 ghost flux on each side. So NumGhosts should be 1 more than r_gp
            assert grid.Nghost == r_gp + 1
            x_predict = (
                grid.x[grid.Nghost - 1 : grid.Nx + grid.Nghost + 1] + grid.dx / 2
            )

            assert len(x_predict) == 2 + grid.Nx

            # predict[i] = i+1/2
            # predict[i-1] = i-1/2
            SE_predict = gp_SE.convert_custom(x_predict, kern.SE, kern.SE)
            SE_predict_by_comp.append(SE_predict)
            DAS_predict = gp_DAS.convert_custom(x_predict, kern.AS, kern.AS)
            DAS_predict_by_comp.append(DAS_predict)

        DAS_predict_by_comp = np.array(DAS_predict_by_comp)
        SE_predict_by_comp = np.array(SE_predict_by_comp)

        # Compute SE Flux
        for icomp in range(NUMQ):
            grid.grid[icomp][
                grid.Nghost - 1 : grid.Nx + grid.Nghost + 1
            ] = SE_predict_by_comp[icomp]

        FSE = flux(grid)

        # Compute DAS Flux
        for icomp in range(NUMQ):
            grid.grid[icomp][
                grid.Nghost - 1 : grid.Nx + grid.Nghost + 1
            ] = DAS_predict_by_comp[icomp]

        FDAS = flux(grid)

        for icomp in range(NUMQ):
            for i in range(grid.Nghost, grid.Nx + grid.Nghost):
                if shock_mask[i]:
                    U_new[icomp, i] = U[icomp, i] - (dt / grid.dx) * (
                        FDAS[icomp, i] - FDAS[icomp, i - 1]
                    )

                else:
                    U_new[icomp, i] = U[icomp, i] - (dt / grid.dx) * (
                        FSE[icomp, i] - FSE[icomp, i - 1]
                    )

        grid.set(U_new)
        grid.apply_zero_gradient_bcs()

        grid.transform(cons_to_prim, "prim")
        # grid.grid = char_to_prim(grid.grid,right_eigen_vectors)
        # grid.variables = "prim"

        t += dt
    return grid


def LF(grid, t0, t_finish, cfl):
    t = t0

    while t < t_finish:
        F = flux(grid)

        grid.assert_variable_type("prim")
        a = np.sqrt(gamma * grid.grid[PCOMP] / grid.grid[RHOCOMP])
        max_speed = np.max(np.abs(grid.grid[UCOMP]) + a)
        dt = min(cfl * grid.dx / max_speed, t_finish - t)

        grid.transform(prim_to_cons, "cons")

        # grid.grid, right_eigen_vectors= prim_to_char(grid.grid)
        # grid.variables = "char"

        U = grid.grid
        U_new = grid.scratch()

        # LF
        for icomp in range(NUMQ):
            for i in range(grid.Nghost, grid.Nx + grid.Nghost):
                # LF
                U_new[icomp, i] = 0.5 * (U[icomp, i + 1] + U[icomp, i - 1]) - (
                    dt / (2 * grid.dx)
                ) * (F[icomp, i + 1] - F[icomp, i - 1])

                # for weno.
                # U_new[icomp, i] = U[icomp, i] - (dt / grid.dx) * (F[icomp, i] - F[icomp, i-1])

                # LW
                # U_new[icomp,i] = U[icomp,i+1] - 0.5*dt/grid.dx * (F[icomp,i+1] - F[icomp,i-1])

                # FTCS
                # U_new[icomp, i] = U[icomp,i] + 0.5*dt/grid.dx*(F[icomp,i+1] - F[icomp,i-1])

        # print(np.shape(U_new))

        grid.set(U_new)
        grid.apply_zero_gradient_bcs()

        grid.transform(cons_to_prim, "prim")
        # grid.grid = char_to_prim(grid.grid,right_eigen_vectors)
        # grid.variables = "prim"

        t += dt
    return grid


def weno5solve(grid, t0, t_finish, cfl, nt):
    timestepNum = 0
    t = t0

    os.makedirs("simulation_frames", exist_ok=True)

    while (t < t_finish) and timestepNum < nt:
        print("Timestep: ", timestepNum, "  Current time: ", t)
        # F = flux(grid)

        # Plotting each variable in a separate subplot
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        grid.apply_zero_gradient_bcs()

        axs[0].plot(grid.grid[RHOCOMP, :], label="Density")
        axs[0].set_title(f"Density at Time: {t:.2f}")
        axs[0].legend()

        axs[1].plot(grid.grid[UCOMP, :], label="Velocity")
        axs[1].set_title(f"Velocity at Time: {t:.2f}")
        axs[1].legend()

        axs[2].plot(grid.grid[PCOMP, :], label="Pressure")
        axs[2].set_title(f"Pressure at Time: {t:.2f}")
        axs[2].legend()

        for ax in axs:
            ax.set_xlabel("Grid")
            ax.set_ylabel("Value")

        # Save the figure
        plt.savefig(f"simulation_frames/frame_{timestepNum:04d}.png")
        plt.close(fig)  # Close the figure to free memory
        uPrim = grid.grid
        grid.assert_variable_type("prim")
        a = np.sqrt(gamma * grid.grid[PCOMP] / grid.grid[RHOCOMP])
        max_speed = np.max(np.abs(grid.grid[UCOMP]) + a)
        dt = min(cfl * grid.dx / max_speed, t_finish - t)
        primP = weno5_plus(grid.grid, grid)
        # primM = weno5_minus(grid.grid,grid)

        mass, mom, energy = prim_to_cons_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])
        consP = np.array([mass, mom, energy])
        # consM = flux_var(primM[0], primM[1], primM[2])

        # fL = flux_var(primM[RHOCOMP], primM[UCOMP], primM[PCOMP])
        fR = flux_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])

        # char_vars, right_eigenvectors_list = prim_to_char(grid.grid)
        # assert(np.shape(char_vars) == np.shape(grid.grid))
        # grid.set(char_vars)
        # grid.variables = "char"

        U = grid.grid  # conservative variables
        U_new = np.ones_like(U) / 0  # np.nans_like lol
        LFFlux = np.zeros_like(U_new)

        # Loop through all the cells except for the outermost ghost cells.
        for i in range(1, len(grid.x) - 1):
            for icomp in range(NUMQ):
                # Compute the Lax-Friedrichs flux at i+1/2 and i-1/2 interfaces

                LFFlux[icomp, i] = 0.5 * (
                    fR[icomp, i + 1] + fR[icomp, i]
                ) - 0.5 * max_speed * (consP[icomp, i + 1] - consP[icomp, i])

        # plt.plot(LFFlux[0],'.')
        # plt.plot(LFFlux[1],'.')
        # plt.plot(LFFlux[2],'.')
        # plt.show()

        for i in range(grid.Nghost, grid.Nx + grid.Nghost):
            for icomp in range(NUMQ):
                # # Update the conserved variables using the net flux difference
                U_new[icomp, i] = consP[icomp, i] - (dt / grid.dx) * (
                    LFFlux[icomp, i] - LFFlux[icomp, i - 1]
                )

        grid.set(U_new)
        grid.apply_zero_gradient_bcs()

        grid.transform(cons_to_prim, "prim")
        grid.apply_zero_gradient_bcs()

        # prim_vars = char_to_prim(grid.grid,right_eigenvectors_list)
        # grid.set(prim_vars)
        # grid.variables = "prim"

        for i in range(len(grid.grid[0])):
            if grid.grid[PCOMP, i] <= 0:
                print("Bad cell: ", i)
                assert False

            for icomp in range(NUMQ):
                if np.isnan(grid.grid[icomp, i]):
                    print("Nan cell : ", i)
                    assert False

        t += dt
        timestepNum += 1

    os.system(
        "ffmpeg -r 30 -f image2 -i simulation_frames/frame_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p simulation.mp4"
    )
    return grid


def seSolve(grid, t0, t_finish, cfl, nt):
    timestepNum = 0
    t = t0

    os.makedirs("simulation_frames", exist_ok=True)

    while (t < t_finish) and timestepNum < nt:
        print("Timestep: ", timestepNum, "  Current time: ", t)
        # F = flux(grid)

        # Plotting each variable in a separate subplot
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        grid.apply_zero_gradient_bcs()

        axs[0].plot(grid.grid[RHOCOMP, :], label="Density")
        axs[0].set_title(f"Density at Time: {t:.2f}")
        axs[0].legend()

        axs[1].plot(grid.grid[UCOMP, :], label="Velocity")
        axs[1].set_title(f"Velocity at Time: {t:.2f}")
        axs[1].legend()

        axs[2].plot(grid.grid[PCOMP, :], label="Pressure")
        axs[2].set_title(f"Pressure at Time: {t:.2f}")
        axs[2].legend()

        for ax in axs:
            ax.set_xlabel("Grid")
            ax.set_ylabel("Value")

        # Save the figure
        plt.savefig(f"simulation_frames/frame_{timestepNum:04d}.png")
        plt.close(fig)  # Close the figure to free memory
        uPrim = grid.grid
        grid.assert_variable_type("prim")
        a = np.sqrt(gamma * grid.grid[PCOMP] / grid.grid[RHOCOMP])
        max_speed = np.max(np.abs(grid.grid[UCOMP]) + a)
        dt = min(cfl * grid.dx / max_speed, t_finish - t)
        primP = weno5_plus(grid.grid, grid)
        # primM = weno5_minus(grid.grid,grid)

        mass, mom, energy = prim_to_cons_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])
        consP = np.array([mass, mom, energy])
        # consM = flux_var(primM[0], primM[1], primM[2])

        # fL = flux_var(primM[RHOCOMP], primM[UCOMP], primM[PCOMP])
        fR = flux_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])

        # char_vars, right_eigenvectors_list = prim_to_char(grid.grid)
        # assert(np.shape(char_vars) == np.shape(grid.grid))
        # grid.set(char_vars)
        # grid.variables = "char"

        U = grid.grid  # conservative variables
        U_new = np.ones_like(U) / 0  # np.nans_like lol
        LFFlux = np.zeros_like(U_new)

        # Loop through all the cells except for the outermost ghost cells.
        for i in range(1, len(grid.x) - 1):
            for icomp in range(NUMQ):
                # Compute the Lax-Friedrichs flux at i+1/2 and i-1/2 interfaces

                LFFlux[icomp, i] = 0.5 * (
                    fR[icomp, i + 1] + fR[icomp, i]
                ) - 0.5 * max_speed * (consP[icomp, i + 1] - consP[icomp, i])

        # plt.plot(LFFlux[0],'.')
        # plt.plot(LFFlux[1],'.')
        # plt.plot(LFFlux[2],'.')
        # plt.show()

        for i in range(grid.Nghost, grid.Nx + grid.Nghost):
            for icomp in range(NUMQ):
                # # Update the conserved variables using the net flux difference
                U_new[icomp, i] = consP[icomp, i] - (dt / grid.dx) * (
                    LFFlux[icomp, i] - LFFlux[icomp, i - 1]
                )

        grid.set(U_new)
        grid.apply_zero_gradient_bcs()

        grid.transform(cons_to_prim, "prim")
        grid.apply_zero_gradient_bcs()

        # prim_vars = char_to_prim(grid.grid,right_eigenvectors_list)
        # grid.set(prim_vars)
        # grid.variables = "prim"

        for i in range(len(grid.grid[0])):
            if grid.grid[PCOMP, i] <= 0:
                print("Bad cell: ", i)
                assert False

            for icomp in range(NUMQ):
                if np.isnan(grid.grid[icomp, i]):
                    print("Nan cell : ", i)
                    assert False

        t += dt
        timestepNum += 1

    os.system(
        "ffmpeg -r 30 -f image2 -i simulation_frames/frame_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p simulation.mp4"
    )
    return grid


def LF1(grid, t0, t_finish, cfl, nt):
    timestepNum = 0
    t = t0

    while (t < t_finish) and timestepNum < nt:
        print("Timestep: ", timestepNum)
        # F = flux(grid)

        uPrim = grid.grid
        grid.assert_variable_type("prim")
        a = np.sqrt(gamma * grid.grid[PCOMP] / grid.grid[RHOCOMP])
        max_speed = np.max(np.abs(grid.grid[UCOMP]) + a)
        print("max_speed: ", max_speed)
        dt = min(cfl * grid.dx / max_speed, t_finish - t)
        print("dt:", dt)
        print("dx:", grid.dx)

        primP = grid.grid
        # primM = weno5_minus(grid.grid,grid)

        mass, mom, energy = prim_to_cons_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])
        consP = np.array([mass, mom, energy])
        # consM = flux_var(primM[0], primM[1], primM[2])

        # fL = flux_var(primM[RHOCOMP], primM[UCOMP], primM[PCOMP])
        fR = flux_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])

        # char_vars, right_eigenvectors_list = prim_to_char(grid.grid)
        # assert(np.shape(char_vars) == np.shape(grid.grid))
        # grid.set(char_vars)
        # grid.variables = "char"

        U = grid.grid  # conservative variables
        U_new = np.copy(U)
        LFFlux = np.zeros_like(U_new)

        for i in range(grid.Nghost, grid.Nx + grid.Nghost - 1):
            for icomp in range(NUMQ):
                # Compute the Lax-Friedrichs flux at i+1/2 and i-1/2 interfaces

                LFFlux[icomp, i] = 0.5 * (
                    fR[icomp, i] + fR[icomp, i + 1]
                ) - 0.5 * max_speed * (consP[icomp, i + 1] - consP[icomp, i])

        for i in range(grid.Nghost, grid.Nx + grid.Nghost - 1):
            for icomp in range(NUMQ):
                # # Update the conserved variables using the net flux difference
                U_new[icomp, i] = consP[icomp, i] - (dt / grid.dx) * (
                    LFFlux[icomp, i] - LFFlux[icomp, i - 1]
                )

        grid.set(U_new)
        grid.apply_zero_gradient_bcs()

        grid.transform(cons_to_prim, "prim")
        # prim_vars = char_to_prim(grid.grid,right_eigenvectors_list)
        # grid.set(prim_vars)
        # grid.variables = "prim"
        plt.figure()
        plt.plot(LFFlux[2, :])
        plt.show()
        grid.plot()

        for i in range(len(grid.grid[0])):
            if grid.grid[PCOMP, i] <= 0:
                print("Bad cell: ", i)
                assert False

        t += dt
        timestepNum += 1
    return grid


def LF(grid, t0, t_finish, cfl):
    t = t0

    while t < t_finish:
        F = flux(grid)

        grid.assert_variable_type("prim")
        a = np.sqrt(gamma * grid.grid[PCOMP] / grid.grid[RHOCOMP])
        max_speed = np.max(np.abs(grid.grid[UCOMP]) + a)
        dt = min(cfl * grid.dx / max_speed, t_finish - t)

        grid.transform(prim_to_cons, "cons")

        # grid.grid, right_eigen_vectors= prim_to_char(grid.grid)
        # grid.variables = "char"

        U = grid.grid
        U_new = grid.scratch()

        # LF
        for icomp in range(NUMQ):
            for i in range(grid.Nghost, grid.Nx + grid.Nghost):
                # LF
                U_new[icomp, i] = 0.5 * (U[icomp, i + 1] + U[icomp, i - 1]) - (
                    dt / (2 * grid.dx)
                ) * (F[icomp, i + 1] - F[icomp, i - 1])

                # for weno.
                # U_new[icomp, i] = U[icomp, i] - (dt / grid.dx) * (F[icomp, i] - F[icomp, i-1])

                # LW
                # U_new[icomp,i] = U[icomp,i+1] - 0.5*dt/grid.dx * (F[icomp,i+1] - F[icomp,i-1])

                # FTCS
                # U_new[icomp, i] = U[icomp,i] + 0.5*dt/grid.dx*(F[icomp,i+1] - F[icomp,i-1])

        # print(np.shape(U_new))

        grid.set(U_new)
        grid.apply_zero_gradient_bcs()

        grid.transform(cons_to_prim, "prim")
        # grid.grid = char_to_prim(grid.grid,right_eigen_vectors)
        # grid.variables = "prim"

        t += dt
    return grid


def minmod(a, b, c):
    """
    Minmod function, which returns the argument with the smallest absolute value,
    but only if all arguments have the same sign. Otherwise, it returns zero.
    """
    if a <= 0 and b <= 0 and c <= 0:
        return max(a, b, c)
    elif a >= 0 and b >= 0 and c >= 0:
        return min(a, b, c)
    else:
        return 0


def minmod_limiter(W, i):
    """
    Apply the minmod limiter to the fifth-order interpolations at index i.
    W is the array of cell-centered values.
    """
    # Calculate high-order face values
    WL = (3 * W[i - 2] - 20 * W[i - 1] + 90 * W[i] + 60 * W[i + 1] - 5 * W[i + 2]) / 128
    WR = (3 * W[i + 3] - 20 * W[i + 2] + 90 * W[i + 1] + 60 * W[i] - 5 * W[i - 1]) / 128

    # Compute slopes
    delta_W1 = W[i] - W[i - 1]
    delta_W2 = W[i + 1] - W[i]
    delta_W3 = W[i + 1] - W[i - 1]

    # Apply the minmod limiter to the slopes
    limited_slope = minmod(delta_W1, delta_W2, delta_W3)

    # Adjust the high-order face values based on the limited slope
    WL_limited = W[i] + 0.5 * limited_slope
    WR_limited = W[i + 1] - 0.5 * limited_slope

    return WL_limited, WR_limited


def fdprim(grid, t0, t_finish, cfl, nt):
    """
    A fifth-order finite difference scheme for hyperbolic equations
    on block-adaptive curvilinear grids
    """
    timestepNum = 0
    t = t0

    while (t < t_finish) and timestepNum < nt:
        print("Timestep: ", timestepNum)
        F = flux(grid)

        U_prim = grid.grid

        grid.assert_variable_type("prim")
        a = np.sqrt(gamma * grid.grid[PCOMP] / grid.grid[RHOCOMP])
        max_speed = np.max(np.abs(grid.grid[UCOMP]) + a)
        dt = min(cfl * grid.dx / max_speed, t_finish - t)

        grid.transform(prim_to_cons, "cons")

        U_cons = grid.grid

        U_new = grid.scratch()

        # Applying the limiter on primitive variables
        for icomp in range(NUMQ):
            for i in range(grid.Nghost, grid.Nx + grid.Nghost):
                WL, WR = minmod_limiter(U_prim[icomp, :], i)
                U_new[icomp, i] = 0.5 * (
                    WL + WR
                )  # Updating the primitive variables with limited values

        # Transforming the limited primitive variables back to conservative form
        grid.set(U_new)  # Set the grid with limited primitive variables
        grid.transform(prim_to_cons, "cons")  # Transform back to conservative

        grid.transform(cons_to_prim, "prim")
        # prim_vars = char_to_prim(grid.grid,right_eigenvectors_list)
        # grid.set(prim_vars)
        # grid.variables = "prim"

        for i in range(len(grid.grid[0])):
            assert grid.grid[PCOMP, i] > 0  # assert positive pressures.

        t += dt
        timestepNum += 1
    return grid
