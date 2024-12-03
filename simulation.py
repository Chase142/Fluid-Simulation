import numpy as np
import numba as nb


@nb.njit(fastmath=True, cache=True)
def get_eq(f, rho, u, c, ceq, w, q):
    u2 = (u[0]**2 + u[1]**2)
    for i in range(q):
        cu = (u[0]*c[i, 0] + u[1]*c[i, 1])
        f[i] = w[i]*rho*(ceq[0] + ceq[1]*cu + ceq[2]*cu**2 + ceq[3]*u2)
    return f

@nb.njit(fastmath=True, cache=True)
def set_inlet(f, w, c, ceq, u0, q):
    u2 = u0**2
    for i in range(q):
        cu = u0*c[i, 0]
        f[i, :, 0] = w[i]*(ceq[0] + ceq[1]*cu + ceq[2]*cu**2 + ceq[3]*u2)
    return f

def get_macros(f, rho, u, cT, q, N, Ny, Nx):
    rho = np.sum(f, axis=0)
    f_reshaped = f.reshape(q, N)
    u_reshaped = np.dot(cT, f_reshaped)
    u = u_reshaped.reshape(2, Ny, Nx)
    u /= rho
    return rho, u

def collide(f, f_eq, omega):
    return f_eq * omega[0] + f * omega[1]

def stream(f, f_star, Ny, Nx, indexes, q):
    for i in range(q):
        f[i] = f_star[i].reshape(Ny*Nx)[indexes[i]].reshape(Ny, Nx)
    return f

def interior_boundary(f, f_star, boundary_nodes, opposite_directions, q):
    for i in range(q):
        f[opposite_directions[i]][boundary_nodes[i]] = f_star[i][boundary_nodes[i]]
    return f

def boundary(f, f_star):
    # Bottom Wall
    f[4, -1, :] = f_star[2, -1, :]
    f[8, -1, :] = f_star[6, -1, :]
    f[7, -1, :] = f_star[5, -1, :]

    # Top Wall
    f[2, 0, :] = f_star[4, 0, :]
    f[5, 0, :] = f_star[7, 0, :]
    f[6, 0, :] = f_star[8, 0, :]

    # Right Wall
    f[3, :, -1] = f[3, :, -2]
    f[7, :, -1] = f[7, :, -2]
    f[6, :, -1] = f[6, :, -2]

    return f

def update(f, f_eq, f_star, rho, u, u0, c, cT, ceq, w, q, N, Ny, Nx, omega, indexes, boundary_nodes, opposite_directions, SIMULATION_STEPS):
    for i in range(SIMULATION_STEPS):
        rho, u = get_macros(f, rho, u, cT, q, N, Ny, Nx)
        f_eq = get_eq(f_eq, rho, u, c, ceq, w, q)
        f_star = collide(f, f_eq, omega)
        f_star = set_inlet(f_star, w, c, ceq, u0, q)
        f = stream(f, f_star, Ny, Nx, indexes, q)
        f = interior_boundary(f, f_star, boundary_nodes, opposite_directions, q)
        f = boundary(f, f_star)
    return f, rho, u