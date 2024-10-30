import torch
import torch.nn.functional as F

class Fluid:
    def __init__(self, Nx, Ny, U, L, rho, nu, u_inlet, g=9.8, max_iters=10000, tol=1e-8, alpha=0.1, max_dt=0.001, device='cpu'):
        '''
        rho: desnsity [kg/m^3]
        nu: kinematic viscosity [m^2/s]
        g: gravitational constant [m/s^2]
        u_inlet: horizontal inlet velocity [m/s]
        U: Characteristic velocity [m/s]
        L: Characteristic length [m]
        T: Characteristic time [s]
        P: Characteristic pressure [Pa]

        u_inlet: inlet horizontal velocity

        Re: Reynolds number
        Fr: Froude number

        Ny, Nx: Ny columns, Nx rows (Ny is first index Nx is second)
        dy, dx: dimentionless cell size

        u: padded dimentionless horizontal velocity field (Ny+2, Nx+2)
        v padded dimentionless verticle velocity field (Ny+2, Nx+2)
        p: padded dimentionless pressure field (Ny+2, Nx+2)

        x: dimentionless x-coordinate of each grid cell center (Ny+2, Nx+2)
        y: dimentionless y-coordinate of each grid cell center (Ny+2, Nx+2)
        **for coordinates 0,0 is top left, 1,0 is bottom left, 1,1 is bottom right, 0,1 is top right**

        s: mask of fluid vs object cells
        ** True for fluid False for objects**

        dt: dimentionless time step
        total_time: total dimentionless time the simulation has run for

        device: device for torch. Default is cpu
        max_iters: max number of iterations used for Jacobi iterations pressure solver. Default is 100
        tol: convergence residual tolerance for Jacobi itterations solver. Default is 1e-6
        alpha: safety scale paramter for CFL conditions. Should be between 0 and 1. Default is 1 which is the maximum dt that satisfies the conditions
        max_dt* = max time dimentionless step

        s, s_up, s_down, s_right, s_left: torch.tensor (H, W) ---> boolean masks for non-padding fluid elements. True for non-padding fluid. Flase elsewhere.
        s_obj: torch tensor (H, W) ---> boolean mask for objects. True for obj & obj padding. Flase everywhere else.
        s_obj_boundary: torch tensor (H, W) ---> boolean tensor mask for objects padding. True for obj padding. Flase everywhere else.
        '''
        self.device = torch.device(device)
        self.tol = tol
        self.max_iters = max_iters
        self.alpha = alpha
        self.max_dt = max_dt

        self.rho = rho
        self.nu = nu
        self.g = g

        self.U = U 
        self.L = L 
        self.T = self.L / self.U 
        self.P = rho * (U**2) 

        self.Re = self.U * self.L / self.nu 
        #self.Fr = self.U / ((self.g*L)**(1/2)) 
        self.Fr = 100000

        self.u_inlet = u_inlet

        self.Nx = Nx
        self.Ny = Ny

        self.dx = L / (Nx +2)
        self.dy = L / (Ny + 2)

        self.dx = self.dx / self.L
        self.dy = self.dy / self.L

        self.x = torch.arange(self.Nx+2, device=self.device).unsqueeze(0).expand(self.Ny+2, self.Nx+2) * self.dx + (0.5 * self.dx)
        self.y = torch.arange(self.Ny+2, device=self.device).unsqueeze(1).expand(self.Ny+2, self.Nx+2) * self.dy + (0.5 * self.dy)

        self.u = torch.zeros((self.Ny, self.Nx), device=self.device, dtype=torch.float32)
        self.v = torch.zeros((self.Ny, self.Nx), device=self.device, dtype=torch.float32)

        self.u += torch.empty((self.Nx, self.Ny), device=self.device, dtype=torch.float32).uniform_(-0.1, 0.1)
        self.v += torch.empty((self.Nx, self.Ny), device=self.device, dtype=torch.float32).uniform_(-0.1, 0.1)

        self.u = F.pad(self.u, pad=(1, 1, 1, 1), mode='constant', value=0)
        self.v = F.pad(self.v, pad=(1, 1, 1, 1), mode='constant', value=0)

        self.u = self.u / self.U
        self.v = self.v / self.U

        self.p = torch.zeros((self.Ny, self.Nx), device=self.device, dtype=torch.float32)

        self.p += torch.full((self.Nx, self.Ny), self.P, device=self.device, dtype=torch.float32)

        self.p = F.pad(self.p, pad=(1, 1, 1, 1), mode='constant', value=0)

        self.p = self.p / self.P

        # boolean array mask. True for interior fluid. False for objects & padding 
        self.s = torch.full((self.Ny, self.Nx), 1, device=self.device, dtype=torch.bool)
        self.s = F.pad(self.s, pad=(1,1,1,1), mode='constant', value=0)
        self.s_left = torch.roll(self.s, shifts=-1, dims=1)
        self.s_right = torch.roll(self.s, shifts=1, dims=1)
        self.s_up = torch.roll(self.s, shifts=1, dims=0)
        self.s_down = torch.roll(self.s, shifts=-1, dims=0)

        self.s_obj = torch.zeros_like(self.s, device=self.device, dtype=torch.bool)

        self.s_obj_boundary = torch.zeros_like(self.s, device=self.device, dtype=torch.bool)

        self.total_time = 0
        self.dt = self.cfl_condition()
        self.apply_boundary_conditions()

    def maccormack_advection(self):
        '''
        Updates the velocity field using an MacCormack advection scheme.
        
        u*: torch.tensor (H, W) ---> dimensionless horizontal velocity field
        v*: torch.tensor (H, W) ---> dimensionless vertical velocity field
        dt*: float ---> dimensionless time step
        x*: torch.tensor ((H, W)---> dimensionless grid of the x coordinates for the cneter of each cell
        y*: torch.tensor (H, W) ---> dimensionless grid of the y coordinates for the center of each cell
        boundary_conditions: callable function ---> boundary conditions to be applied to u & v along with any additional arguments
        *args : positional arguments for boundary_conditions
        **kwargs : keyword arguments for boundary_conditions

        **Make sure all tensors are on the same device**

        (Predictor step): Semi-Lagrangian technique to interpolate back in time to the previous location 
        and computed the predicted previous velocity field using bilinearinterpolation
        x_p* = x_i* - u*(j, i) * dt*
        y_p* = y_j* - v*(j, i) * dt*
        u_p* = Interpolate_u*(y_p*, x_p*)
        v_p* = Interpolate_v*(y_p*, x_p*)
        (Corrector step): Now, from the predicted location, step forward in time, again using the Semi-Lagrangian technique, 
        to see where the predicted value would arrive
        x_a* = x_p* + u_p*(j, i) * dt*
        y_a* = y_p* + v_p*(j, i) * dt*
        u_a* = Interpolate_u_p*(y_a*, x_a*)
        v_a* = Interpolate_v_p*(y_a*, x_a*)
        Finally, compute the average of the two predicted velocities:
        u* = 0.5(u_p* + u_a*)
        v* = 0.5(v_p* + v_a*)

        Theis scheme should yeild second-order accuracy in both space and time. However, due to it being an explicit method in is not 
        unconditionally stable. The CFL condition for stability is: dt <= dx/u_max

        Returns:
        u*: torch.tensor (H, W) ---> updated horizontal velocity field (Ny x Nx) (in-place)
        v*: torch.Tensor (H, W) ---> updated vertical velocity field (Ny x Nx) (in-place)
        '''
        # calculating the predicted previous location and then clamping it to force it to be located within the grid
        x_p = (self.x - (self.u*self.dt)).clamp(0,1)
        y_p = (self.y - (self.v*self.dt)).clamp(0,1)

        # normalizing the values of x and y to be between -1 and 1
        x_p_norm = (x_p * 2) - 1
        y_p_norm = (y_p * 2) -1

        # creating the necesary normalized grid for torch's sample grid method. Shape (N=1, H=Ny, W=Nx, 2->(x, y)) 
        xy_p_grid_norm = torch.stack((x_p_norm, y_p_norm), dim=-1).unsqueeze(0)
        
        # stacking the verticle and horizontal velocity fields into shape (N=1, C=2->(u, v), H=Ny, W=Nx) so we can compute all at once 
        uv_stack = torch.stack((self.u, self.v), dim=0).unsqueeze(0)

        # using pytorch's grid sample method to perform bilinear interpolation at predicted points
        uv_p_stack = F.grid_sample(uv_stack, xy_p_grid_norm, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        u_p = uv_p_stack[0]
        v_p = uv_p_stack[1]

        # because this function updates the boundary values need to reset them to their prescibed values before moving forwards
        u_p, v_p = self.velocity_boundary_conditions(u_p, v_p)
        
        # calculating the arival points based on the newly calcualted velocity fields and then clamping it to force it to be located within the grid
        x_a = (x_p + (u_p*self.dt)).clamp(0,1)
        y_a = (y_p + (v_p*self.dt)).clamp(0,1)

        # normalizing the values of x and y to be between -1 and 1
        x_a_norm = (x_a * 2) - 1
        y_a_norm = (y_a * 2) -1
        
        # creating the necesary normalized grid for torch's sample grid method. Shape (N=1, H=Ny, W=Nx, 2->(x, y)) 
        xy_a_grid_norm = torch.stack((x_a_norm, y_a_norm), dim=-1).unsqueeze(0)

        # add an extra dimention back to uv_p_stack such that it meets the required input dimention (N=1, C=2->(u, v), H=Ny, W=Nx)
        uv_p_stack = uv_p_stack.unsqueeze(0)
        
        # using pytorch's grid sample method to perform bilinear interpolation at predicted points
        uv_a_stack = F.grid_sample(uv_p_stack, xy_a_grid_norm, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        u_a = uv_a_stack[0]
        v_a = uv_a_stack[1]

        # because this function updates the boundary values need to reset them to their prescibed values before moving forwards
        u_a, v_a = self.velocity_boundary_conditions(u_a, v_a)

        self.u = 0.5*(u_a + u_p)
        self.v = 0.5*(v_a + v_p)

        return self.u, self.v
    
    def explicit_euler_diffusion(self):
        """
        Applies diffusion to the interior velocity field ignoring edges 
        using an explicit Euler scheme with central differencing used to compute the Laplacian.

        Parameters:
        u*: torch.tensor (H, W) ---> dimensionless horizontal velocity field
        v*: torch.tensor (H, W) ---> dimensionless vertical velocity field
        dt*: float ---> dimensionless time step
        dx*: float ---> dimensionless grid spacing in x
        dy*: float ---> dimensionless grid spacing in y
        Re: float ---> Reynolds number

        Equation:
        u*_new = u*_old + ((dt*/Re) * Laplacian(u))
        v*_new = v*_old + ((dt*/Re) * Laplacian(v))

        This scheme should yeild first order accuracy in time while achieving second order accuaracy in space. Due to it being an explicit 
        scheme it is only conditionally stable. The CFL condition for dt is: dt <= (dx**2 * Re) / (U*L)

        Returns:
        u*: torch.Tensor (H, W) ---> updated horizontal velocity field (Ny x Nx) (in-place)
        v*: torch.Tensor (H, W) ---> updated vertical velocity field (Ny x Nx) (in-place)
        """

        # constant used in calculations
        alpha = self.dt / self.Re

        # squared grid sizes
        dx2 = self.dx**2
        dy2 = self.dy**2
        
        # Compute Laplacian for u using finite differences (only interior points)
        Left = self.u[self.s_left]
        Right = self.u[self.s_right]
        Up = self.u[self.s_up]
        Down = self.u[self.s_down]
        Center = self.u[self.s]
        laplacian_u = ((Right + Left - (2*Center)) / dx2) + ((Up + Down - (2*Center)) / dy2)

        Left = self.v[self.s_left]
        Right = self.v[self.s_right]
        Up = self.v[self.s_up]
        Down = self.v[self.s_down]
        Center = self.v[self.s]
        laplacian_v = ((Right + Left - (2*Center)) / dx2) + ((Up + Down - (2*Center)) / dy2)

        # Update the interior points in place using the explicit Euler method
        self.u[self.s] += alpha * laplacian_u
        self.v[self.s] += alpha * laplacian_v

        return self.u, self.v
    
    def explicit_euler_gravity(self):
        """
        Applies gravity to the interior verticle velocity field ignoring edges using an explicit Euler scheme.

        Parameters:
        v*: torch.tensor (H, W) ---> dimensionless vertical velocity field
        dt*: float ---> dimensionless time step
        Fr: float ---> Fruede number

        Equation:
        v*_new = v*_old + (1/Fr^2) * dt

        This scheme should yeild first order accuracy in time. Due to it being an explicit scheme it is only conditionally stable. 
        The CFL condition for dt is: dt <= Fr

        Returns:
        v*: torch.Tensor (H, W) ---> updated vertical velocity field (Ny x Nx) (in-place)
        """
        # Apply gravity downwards [0][0] is top left and y grows as the first index increases so we want to add
        self.v[self.s] += (1 / (self.Fr**2)) * self.dt

        return self.v
    
    def Jacobi_pressure_solver(self):
        '''
        Iteratively solves the Poisson's equation for pressure using the Jacobi method.

        u*: torch.tensor (H, W) ---> dimensionless horizontal velocity field
        v*: torch.tensor (H, W) ---> dimensionless vertical velocity field
        p*: torch.tensor (H, W) ---> dimentionless pressure field
        dt*: float ---> dimensionless time step
        dx*: float ---> dimensionless grid spacing in x
        dy*: float ---> dimensionless grid spacing in y
        max_iters: int=100 ---> max number of iterations if convergence is not hit
        tol: float=1.0e-6 ---> once the L2 norm between the current and previous pressure field is less than tol it stops
        
        1) Compute divergence of velocity field:
        div*(u*, v*) = (u*_(i+1,j) - u*_(i,j)) / 2*dx* + (v*_(i,j+1) - v*_(i,j)) / 2*dy
        
        2) Update pressure:
        p*_(i,j)^(n+1) = 1 / (2 * (1 / dx*^2 + 1 / dy*^2)) * ((p*_(i+1,j)^n + p*_(i-1,j)^n) / dx*^2 + (p*_(i,j+1)^n + p*_(i,j-1)^n) / dy*^2 - div*(u*, v*))

        3) Loop steps 1 & 2 until we reach convergence or max_iters

        This scheme should yeild first order accuracy in time and second order accuracy in space. 
        Due to it being an semi-implicit scheme it is only conditionally stable. However, the CFL condition for 
        it is linked with that for advection: dt <= dx/u_max

        Returns:
        p*: torch.Tensor (H, W) ---> updated pressure field (Ny x Nx) (in-place)
        '''

        # loop "max_iters" number of times
        for i in range(self.max_iters):
            # calculating the divergence using central differencing
            div = ((self.u[self.s_right] - self.u[self.s_left]) / (2*self.dx)) + ((self.v[self.s_up] - self.v[self.s_down]) / (2*self.dy))

            p_old = self.p.clone()
        
            # updating pressure
            self.p[self.s] = (1 / (2 * ((1 / (self.dx**2)) + (1 / (self.dy**2))))) * ( ((self.p[self.s_right]+self.p[self.s_left])/(self.dx**2)) + ((self.p[self.s_up]+self.p[self.s_down])/(self.dy**2)) - div)

            # apply boundary conditions
            self.p = self.pressure_boundary_conditions(self.p)

            # calculates the L2 norm between current and previous step to check for convergence
            residual = torch.linalg.norm(self.p[self.s] - p_old[self.s])

            # Checks for convergence
            if residual < self.tol:
                break
        # print(div.abs().max())
        return self.p
    
    def pressure_projection(self):
        '''
        Updates the velocity field based on the pressure gradient.
        
        u*: torch.tensor (H, W) ---> dimensionless horizontal velocity field
        v*: torch.tensor (H, W) ---> dimensionless vertical velocity field
        p*: torch.tensor (H, W) ---> dimentionless pressure field
        dt*: float ---> dimensionless time step
        dx*: float ---> dimensionless grid spacing in x
        dy*: float ---> dimensionless grid spacing in y
        
        Update equations:
        u*_(i,j)^(n+1) = u*_(i,j)^n - (dt*/2dx*) * (p*_(i+1,j) - p*_(i-1,j))
        v*_(i,j)^(n+1) = v*_(i,j)^n - (dt*/2dy*) * (p*_(i,j+1) - p*_(i,j-1))

        Returns:
        u*: torch.Tensor (H, W) ---> updated horizontal velocity field (Ny x Nx) (in-place)
        v*: torch.Tensor (H, W) ---> updated vertical velocity field (Ny x Nx) (in-place)
        '''
        # updating velocity fields based on pressure gradient
        self.u[self.s] -= ((self.dt / (2 * self.dx)) * (self.p[self.s_right] - self.p[self.s_left]))
        self.v[self.s] -= ((self.dt / (2 * self.dy)) * (self.p[self.s_up] - self.p[self.s_down]))
        
        return self.u, self.v
    
    def define_object():
        pass
    
    def apply_boundary_conditions(self):
        '''
        Applies boundary conditions to the given velocity and pressure fields.
        
        u*: torch.tensor (H, W) ---> horizontal dimentionless velocity field 
        v*: torch.tensor (H, W) ---> vertical dimentionless velocity field
        p*: torch.tensor (H, W) ---> dimentionless pressure field

        Boundary conditions
        1) Inlet: 
            Velocity: Dirichlet boundary conditions (u=u_inlet, v=0)
            Pressure: Neumann boundary condition (zero pressure gradient)
        2) Outlet: 
            Velocity: Neumann boundary condition (zero velocity gradient)
            Pressure: Dirichlet boundary condition (zero pressure)
        3) Bottom: 
            Velocity: Dirichlet boundary condition (no slip: u=v=0)
            Pressure: Neumann boundary condition (zero pressure gradient)
        4) Top: 
            Velocity: Dirichlet boundary condition (no slip: u=v=0)
            Pressure: Neumann boundary condition (zero pressure gradient)
        5) Obj surface & Interior:
            Velocity: Dirichlet boundary conditions (no slip, u=0, v=0)
            Pressure: Neumann boundary condition (zero pressure gradient)

        Returns:
        u*: torch.Tensor (H, W) ---> updated horizontal velocity field (Ny x Nx) (in-place)
        v*: torch.Tensor (H, W) ---> updated vertical velocity field (Ny x Nx) (in-place)
        p*: torch.Tensor (H, W) ---> updated pressure field (Ny x Nx) (in-place)

        '''
        # Inlet boundary
        self.u[:, 0] = self.u_inlet
        self.v[:, 0] = 0
        self.p[:, 0] = self.p[:, 1] 
        # Top
        self.u[0, :] = 0
        self.v[0, :] = 0
        self.p[0, :] = self.p[1, :]
        # Outlet
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]
        self.p[:, -1] = 0
        # Bottom
        self.u[-1, :] = 0
        self.v[-1, :] = 0
        self.p[-1, :] = self.p[-2, :]

        # Objects
        self.u[self.s_obj] = 0
        self.v[self.s_obj] = 0

        up = torch.roll(self.p, shifts=1, dims=0)
        down = torch.roll(self.p, shifts=-1, dims=0)
        left = torch.roll(self.p, shifts=1, dims=1)
        right = torch.roll(self.p, shifts=-1, dims=1)

        # self.p[self.s_obj_boundary] = up[self.s_obj_boundary]
        # self.p[self.s_obj_boundary] = down[self.s_obj_boundary]
        # self.p[self.s_obj_boundary] = left[self.s_obj_boundary]
        # self.p[self.s_obj_boundary] = right[self.s_obj_boundary]

        self.p[self.s_obj_boundary] = (up[self.s_obj_boundary] + down[self.s_obj_boundary] + left[self.s_obj_boundary] + right[self.s_obj_boundary]) / 4

        return self.u, self.v, self.p
    
    def velocity_boundary_conditions(self, u, v):
        '''
        Applies boundary conditions to the given velocity fields.
        
        u*: torch.tensor (H, W) ---> horizontal dimentionless velocity field 
        v*: torch.tensor (H, W) ---> vertical dimentionless velocity field
        
        Boundary conditions
        1) Inlet: 
            Velocity: Dirichlet boundary condition (no slip: u=v=0)
        2) Outlet: 
            Velocity: Dirichlet boundary condition (no slip: u=v=0)
        3) Bottom: 
            Velocity: Dirichlet boundary condition (no slip: u=v=0)
        4) Top: 
            Velocity: Dirichlet boundary condition (no slip: u=v=0)
        5) Obj surface & Interior:
            Velocity: Dirichlet boundary conditions (no slip, u=0, v=0)

        Returns:
        u*: torch.Tensor (H, W) ---> updated horizontal velocity field (Ny x Nx) (in-place)
        v*: torch.Tensor (H, W) ---> updated vertical velocity field (Ny x Nx) (in-place)
        '''
        # Inlet boundary
        u[:, 0] = self.u_inlet
        v[:, 0] = 0
        # Top
        u[0, :] = 0
        v[0, :] = 0

        # Outlet
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]

        # Bottom
        u[-1, :] = 0
        v[-1, :] = 0

        # Object
        u[self.s_obj] = 0
        v[self.s_obj] = 0

        return u, v
    
    def pressure_boundary_conditions(self, p):
        '''
        Applies boundary conditions to the given pressure field.
        
        p*: torch.tensor (H, W) ---> dimentionless pressure field
        
        Boundary conditions
        1) Inlet: 
            Pressure: Neumann boundary condition (zero pressure gradient)
        2) Outlet: 
            Pressure: Neumann boundary condition (zero pressure gradient)
        3) Bottom: 
            Pressure: Dirichlet boundary condition (1/Fr**2 at bottom)
        4) Top: 
            Pressure: Dirichlet boundary condition (0 at top)
        5) Object:
            Pressure: Neumann boundary condition (zero pressure gradient)

        Returns:
        p*: torch.Tensor (H, W) ---> updated pressure field (Ny x Nx) (in-place)
        '''
        # Inlet boundary
        p[:, 0] = p[:, 1]
        # Top
        p[0, :] = p[1, :]
        # Outlet
        p[:, -1] = 0
        # Bottom
        p[-1, :] = p[-2, :]
        # Object
        up = torch.roll(p, shifts=1, dims=0)
        down = torch.roll(p, shifts=-1, dims=0)
        left = torch.roll(p, shifts=1, dims=1)
        right = torch.roll(p, shifts=-1, dims=1)

        # p[self.s_obj_boundary] = up[self.s_obj_boundary]
        # p[self.s_obj_boundary] = down[self.s_obj_boundary]
        # p[self.s_obj_boundary] = left[self.s_obj_boundary]
        # p[self.s_obj_boundary] = right[self.s_obj_boundary]
        p[self.s_obj_boundary] = (up[self.s_obj_boundary] + down[self.s_obj_boundary] + left[self.s_obj_boundary] + right[self.s_obj_boundary]) / 4

        return p

    
    def cfl_condition(self):
        '''
        Calculates the largest time step that satisfies the Courant-Friedrichs-Lewy (CFL) condition condition 
        for advection, diffusion, and gravty.

        **For maximum efficientcy have u and v be on the cpu as if they aren't data will neeed to be transfered**

        Parameters:
        u*: torch.tensor (H, W) ---> dimensionless horizontal velocity field
        v*: torch.tensor (H, W) ---> dimensionless vertical velocity field
        dt*: float ---> dimensionless time step
        dx*: float ---> dimensionless grid spacing in x 
        dy*: float ---> dimensionless grid spacing in y
        Re: float ---> Reynolds number
        Fr: float ---> Froude number
        alpha: float ---> safety factor used to scale the maximum allowable value down for stability (0,1)

        CFL Conditions:
        (advection): dt <= min(dx / max(|u|), dy / max(|v|))

        (diffusion): dt <= min(dx^2, dy^2) * Re / 4

        (gravity): dt <= Fr
        
        Returns:
        dt*: float ---> updated dt (in place)
            
        '''
        # calculates the max absolute values of the u and v velocity fields
        u_max = self.u.abs().max().item()  
        v_max = self.v.abs().max().item()  

        # calculates the highest allowable dt value during advection
        dt_advection = min(self.dx / u_max, self.dy / v_max)

        # calculates the highest allowable dt value during diffusion
        dx2 = self.dx**2
        dy2 = self.dy**2
        dt_diffusion = min(dx2, dy2) * self.Re / 4

        # chooses the smallest dt of the three such that it satisfies all condition
        dt_gravity = self.Fr

        # scales dt my the specified safety factor    
        self.dt = min(dt_advection, dt_diffusion, dt_gravity, self.max_dt) * self.alpha
        return self.dt
    
    def define_obj(self, x_p, y_p, r):
        '''
        adds a circular object centered at the specified x, y point.

        x_p*: float ---> dimentionless x coordinate of object (will egt rounded to nearest grid cell) [0,1]
        y_p*: float ---> dimentionless y coordinate of object (will egt rounded to nearest grid cell) [0, 1]
        x*: torch.tensor ((H, W)---> dimensionless grid of the x coordinates for the cneter of each cell
        y*: torch.tensor (H, W) ---> dimensionless grid of the y coordinates for the center of each cell
        r*: float ---> radius of object [0, 1]
        s, s_up, s_down, s_right, s_left: torch.tensor (H, W) ---> boolean masks for non-padding fluid elements. True for non-padding fluid. Flase elsewhere.
        s_obj: torch tensor (H, W) ---> boolean tensor mask for objects. True for obj & obj padding. Flase everywhere else.
        s_obj_boundary: torch tensor (H, W) ---> boolean tensor mask for objects padding. True for obj padding. Flase everywhere else.

        returns:
        s: torch.tensor (updated in place), s_obj (updated in place), s_obj_boundary (updated in place)
        '''

        x_p = max(min(x_p, 1), 0)
        y_p = max(min(y_p, 1), 0)

        distance = torch.sqrt((self.x-x_p)**2 + (self.y-y_p)**2)

        obj_mask = distance <= r

        up = torch.roll(obj_mask, shifts=1, dims=0)
        down = torch.roll(obj_mask, shifts=-1, dims=0)
        left = torch.roll(obj_mask, shifts=1, dims=1)
        right = torch.roll(obj_mask, shifts=-1, dims=1)

        boundary_mask = (~obj_mask) & (up | down | left | right)

        self.s[obj_mask] = False
        self.s[boundary_mask] = False
        self.s_obj[obj_mask] = True
        self.s_obj[boundary_mask] = True
        self.s_obj_boundary[boundary_mask] = True

        self.s_left = torch.roll(self.s, shifts=-1, dims=1)
        self.s_right = torch.roll(self.s, shifts=1, dims=1)
        self.s_up = torch.roll(self.s, shifts=1, dims=0)
        self.s_down = torch.roll(self.s, shifts=-1, dims=0)

        return self.s, self.s_obj, self.s_obj_boundary, self.s_left, self.s_right, self.s_up, self.s_down

    
    def update(self):
        '''
        runs a single update sequence

        u*: torch.tensor (H, W) ---> dimensionless horizontal velocity field
        v*: torch.tensor (H, W) ---> dimensionless vertical velocity field
        p*: torch.tensor (H, W) ---> dimentionless pressure field
        dx*: float ---> dimensionless grid spacing in x 
        dy*: float ---> dimensionless grid spacing in y
        Re: float ---> Reynolds number
        Fr: float ---> Froude number
        u_inlet*: float ---> dimentionless inlet velocity
        boundary_conditions: callable function ---> boundary conditions to be applied to u & v
        cfl_conditions: callable function ---> updates dt
        advection: callable function ---> updates u & v based on particle motion
        diffusion: callable function ---> updates u & v based on viscis effects
        gravity: callable function ---> updes u & v based onn graviational effects
        pressure_solver: callable function ---> updates p based on u & v
        projection: updates u & v based on p

        1) Advection: use the explicit MacCormack method to solve du*/dt* = (u* dot del*)u* = 0 with second order error in both space and time<br>
        2) Diffusion: use an explicit Euler scheme to solve du*/dt* = (1/Re) Laplacian*(u*) with second order error in space but first order time
        3) Body force: use an explicit Euler method to solve du*/dt* = (1/Fr^2)g^hat with second order error in space but first order time
        4) Pressure solve: Jacobi itteration method to solve Laplacian*(p*) = div*(u*)
        5) Pressure projection: Use pressure to update velocities
        6) CFL condition on dt & Repeat
        '''
        self.dt = self.cfl_condition()
        self.u, self.v = self.maccormack_advection()
        self.u, self.v = self.velocity_boundary_conditions(self.u, self.v)
        self.u, self.v = self.explicit_euler_diffusion()
        self.u, self.v = self.velocity_boundary_conditions(self.u, self.v)
        self.v = self.explicit_euler_gravity()
        self.u, self.v = self.velocity_boundary_conditions(self.u, self.v)
        self.p = self.Jacobi_pressure_solver()
        self.p = self.pressure_boundary_conditions(self.p)
        self.u, self.v = self.pressure_projection()
        self.u, self.v = self.velocity_boundary_conditions(self.u, self.v)
        self.total_time += self.dt
        return self.u, self.v, self.p, self.dt