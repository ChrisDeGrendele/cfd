import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 1.4  # Specific heat ratio for air

# Grid and simulation parameters
nx = 100  # number of grid points
x = np.linspace(0, 1, nx)  # spatial domain
dx = x[1] - x[0]

# Initial conditions - Sod shock tube
x_center = 0.5
rho_left = 1.0
u_left = 0.0
p_left = 1.0
rho_right = 0.125
u_right = 0.0
p_right = 0.1

rho = np.where(x < x_center, rho_left, rho_right)
u = np.where(x < x_center, u_left, u_right)
p = np.where(x < x_center, p_left, p_right)
E = p / ((gamma - 1) * rho) + 0.5 * u**2
U = np.array([rho, rho * u, rho * E])


# Function to compute flux
def compute_flux(U):
    rho, m, E = U
    u = m / rho
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    F = np.array([m, m**2 / rho + p, (E + p) * u])
    return F


# Lax-Friedrichs scheme
CFL = 0.5  # Courant number
t = 0.0  # start time
t_end = 0.2  # end time

while t < t_end:
    F = compute_flux(U)

    # Calculate sound speed and determine dt for CFL condition
    a = np.sqrt(gamma * p / rho)
    max_speed = np.max(np.abs(u) + a)
    dt = min(CFL * dx / max_speed, t_end - t)

    # Lax-Friedrichs update
    print("dt: ", dt)
    print("dx: ", dx)

    U[:, 1:-1] = 0.5 * (U[:, 2:] + U[:, :-2]) - dt / (2 * dx) * (F[:, 2:] - F[:, :-2])

    # Boundary conditions
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]

    # Update time and derived quantities
    t += dt
    rho, m, E = U
    u = m / rho
    p = (gamma - 1) * (E - 0.5 * rho * u**2)

    plt.plot(rho, linewidth=10)
    plt.plot(u, ".")
    plt.plot(p)

    plt.show()

    exit()


# Plotting results
plt.figure(figsize=(12, 8))
plt.plot(x, rho, label="Density")
plt.plot(x, u, label="Velocity")
plt.plot(x, p, label="Pressure")
plt.legend()
plt.title(f"Sod Shock Tube Problem at t = {t:.2f} - Lax-Friedrichs Scheme")
plt.xlabel("x")
plt.ylabel("Variables")
plt.show()
