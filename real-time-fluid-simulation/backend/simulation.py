import torch
import numpy as np
import torch.nn.functional as F

import fluid
import pvField

class Simulation:
    
    #Facts
    g = 9.8 # gravitational constant [m/s^2]
    
    #Materials
    defaultFluid : fluid.Fluid = None
    field : pvField.PvField = None
    
    #Geometry
    stepSize : np.ndarray = (1, 1)
    dtMax : float = 1
    cellSize : np.ndarray = (1,1)
    nCell : np.ndarray = (1,1)
    simGrid : tuple[torch.tensor, torch.tensor] = (None, None)
    
    #Fake Numbers
    charVelo : float = 1
    charLen  : float = 1
    charTime : float = 1
    charPressure : float = 1
    defReynolds : float = 1
    defFroude   : float = 1
    
    #Util
    torchDevice : torch.device = torch.cpu
    
    #CTORS
    def __init__(self, defFluid : fluid.Fluid, pvInitial : pvField.PvField,
                 dtMax : float, nCell : np.ndarray, cell : np.ndarray,
                 charVelo : float,  torchDevice : torch.device = torch.cpu):
        self.defaultFluid = defFluid
        self.field = pvInitial
        
        self.dtMax = dtMax
        self.cellSize = cell
        self.nCell = nCell
        
        self.charVelo = charVelo
        
        self.torchDevice = torchDevice
        
        self._characterize()
        self._initializeCell()
        
    @classmethod
    def _initializeCell(self):
        self.stepSize = (self.cellSize / self.nCell) / self.
        self.simGrid[0] = torch.arange(self.nCell[0]+2, device=self.device).unsqueeze(0)\
        .expand(self.nCell[1] + 2, self.nCell[0] + 2) * self.nCell[0] + (0.5 * self.nCell[0])
        self.simGrid[1] = torch.arange(self.nCell[1]+2, device=self.device).unsqueeze(0)\
        .expand(self.nCell[1] + 2, self.nCell[0] + 2) * self.nCell[0] + (0.5 * self.nCell[1])
        
    @classmethod
    def _characterize(self):
        self.charTime = self.charLen / self.charVelo # Characteristic time [s]
        self.charPressure = self.defaultFluid.density * (self.charVelo**2) # Characteristic pressure [Pa]

        self.defReynolds = self.charVelo *  self.charLen / self.defaultFluid.viscosity # Reynolds number
        self.defFroude = self.charVelo / ((self.g * self.charLen)**(1/2)) # Froude number
        
        
    #Running Simulation
    def step(self, dt):
        self.maccormack_advection(self, dt, velocity_boundary_conditions, u_inlet)
        self.explicit_euler_diffusion(self, dt)
        self.explicit_euler_gravity(self, dt)
        self.Jacobi_pressure_solver(self, dt)
        self.pressure_projection(self, dt)
    
    def run_for_time(self, t):
        pass
    
    #Math Methods Private
    def _cfl_condition(self, alpha=1):
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
        u_max = self.field.velo[0].abs().max().item()  
        v_max = self.field.velo[0].abs().max().item()

        # calculates the highest allowable dt value during advection
        dt_advection = min(self.stepSize[0] / u_max, self.stepSize[1] / v_max)

        # calculates the highest allowable dt value during diffusion
        dx2 = self.stepSize[0]**2
        dy2 = self.stepSize[1]**2
        dt_diffusion = min(dx2, dy2) * self.defReynolds / 4

        # chooses the smallest dt of the three such that it satisfies all condition
        dt_gravity = self.defFroude

        # scales dt my the specified safety factor    
        dt = min(dt_advection, dt_diffusion, dt_gravity) * alpha

        return dt
    
    def _velocity_boundary_conditions(self, u_inlet):
        '''
        Applies boundary conditions to the given velocity fields.

        u*: torch.tensor (H, W) ---> horizontal dimentionless velocity field 
        v*: torch.tensor (H, W) ---> vecticle dimentionless velocity field
        u_inlet*: float ---> dimentionless inlet velocity

        Boundary conditions
        1) Inlet: 
            Velocity: Dirichlet boundary conditions (u=u_inlet, v=0)
        2) Outlet: 
            Velocity: Neumann boundary condition (zero velocity gradient)
        3) Bottom: 
            Velocity: Dirichlet boundary condition (no slip: u=v=0)
        4) Top: 
            Velocity: Neumann & Dirichlet boundary condition (free slip: zero velocity gradient in u, v=0)
        '''
        # Inlet boundary
        self.field.velo[0][:, 0] = u_inlet
        self.field.velo[1][:, 0] = 0

        # Top
        self.field.velo[0][0, :] = self.field.velo[0][1, :]
        self.field.velo[1][0, :] = 0
    
        # Outlet
        self.field.velo[0][:, -1] = self.field.velo[0][:, -2]
        self.field.velo[1][:, -1] = self.field.velo[1][:, -2]

        # Bottom
        self.field.velo[0][-1, :] = 0
        self.field.velo[1][-1, :] = 0
    
    def _apply_boundary_conditions(self, u_inlet):
        '''
        Applies boundary conditions to the given velocity and pressure fields.

        u*: torch.tensor (H, W) ---> horizontal dimentionless velocity field 
        v*: torch.tensor (H, W) ---> vecticle dimentionless velocity field
        p*: torch.tensor (H, W) ---> dimentionless pressure field
        u_inlet*: float ---> dimentionless inlet velocity

        Boundary conditions
        1) Inlet: 
            Velocity: Dirichlet boundary conditions (u=u_inlet, v=0)
            Pressure: Neumann boundary condition (zero pressure gradient)
        2) Outlet: 
            Velocity: Neumann boundary condition (zero velocity gradient)
            Pressure: Neumann boundary condition (zero pressure gradient)
        3) Bottom: 
            Velocity: Dirichlet boundary condition (no slip: u=v=0)
            Pressure: Neumann boundary condition (zero pressure gradient)
        4) Top: 
            Velocity: Neumann & Dirichlet boundary condition (free slip: zero velocity gradient in u, v=0)
            Pressure: Neumann boundary condition (zero pressure gradient)

        Returns:
        u*: torch.Tensor (H, W) ---> updated horizontal velocity field (Ny x Nx) (in-place)
        v*: torch.Tensor (H, W) ---> updated vertical velocity field (Ny x Nx) (in-place)
        p*: torch.Tensor (H, W) ---> updated pressure field (Ny x Nx) (in-place)

        '''
        # Inlet boundary
        self.field.velo[0][:, 0] = u_inlet
        self.field.velo[1][:, 0] = 0
        self.field.pressure[:, 0] = self.field.pressure[:, 1]
        # Top
        self.field.velo[0][0, :] = self.field.velo[0][1, :]
        self.field.velo[1][0, :] = 0
        self.field.pressure[0, :] = self.field.pressure[1, :]
        # Outlet
        self.field.velo[0][:, -1] = self.field.velo[0][:, -2]
        self.field.velo[1][:, -1] = self.field.velo[1][:, -2]
        self.field.pressure[:, -1] = self.field.pressure[:, -2]
        # Bottom
        self.field.velo[0][-1, :] = 0
        self.field.velo[1][-1, :] = 0
        self.field.pressure[-1, :] = self.field.pressure[-2, :]
    
    def _pressure_projection(self, dt):
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
        '''
        # updating velocity fields based on pressure gradient
        self.field.velo[0][1:-1, 1:-1] = self.field.velo[0][1:-1, 1:-1] - (dt / (2 * self.stepSize[0])) * (self.field.pressure[1:-1, 2:] - self.field.pressure[1:-1, :-2])
        self.field.velo[1][1:-1, 1:-1] = self.field.velo[1][1:-1, 1:-1] - (dt / (2 * self.stepSize[0])) * (self.field.pressure[2:, 1:-1] - self.field.pressure[:-2, 1:-1])
    
    def _explicit_euler_gravity(self, dt):
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
        self.field.velo[1][1:-1, 1:-1] += (1 / (self.defFroude**2)) * dt

        return v
    
    def _explicit_euler_diffusion(self, dt):
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
        alpha = dt / self.defReynolds

        # squared grid sizes
        dx2 = self.cellSize[0]**2
        dy2 = self.cellSize[1]**2

        # Compute Laplacian for u using finite differences (only interior points)
        Left = self.field.velo[0][1:-1, :-2]
        Right = self.field.velo[0][1:-1, 2:]
        Up = self.field.velo[0][:-2, 1:-1]
        Down = self.field.velo[0][2:, 1:-1]
        Center = self.field.velo[0][1:-1, 1:-1]
        laplacian_u = ((Right + Left - (2*Center)) / dx2) + ((Up + Down - (2*Center)) / dy2)

        Left = self.field.velo[1][1:-1, :-2]
        Right = self.field.velo[1][1:-1, 2:]
        Up = self.field.velo[1][:-2, 1:-1]
        Down = self.field.velo[1][2:, 1:-1]
        Center = self.field.velo[1][1:-1, 1:-1]
        laplacian_v = ((Right + Left - (2*Center)) / dx2) + ((Up + Down - (2*Center)) / dy2)



        # Update the interior points in place using the explicit Euler method
        self.field.velo[0][1:-1, 1:-1] += alpha * laplacian_u
        self.field.velo[1][1:-1, 1:-1] += alpha * laplacian_v
    
    def _maccormack_advection(self, dt, boundary_conditions, *args, **kwargs):
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
        x_p = (self.simGrid[0] - (self.field.velo[0]*dt)).clamp(0,1)
        y_p = (self.simGrid[0] - (self.field.velo[1]*dt)).clamp(0,1)

        # normalizing the values of x and y to be between -1 and 1
        x_p_norm = (x_p * 2) - 1
        y_p_norm = (y_p * 2) -1

        # creating the necesary normalized grid for torch's sample grid method. Shape (N=1, H=Ny, W=Nx, 2->(x, y)) 
        xy_p_grid_norm = torch.stack((x_p_norm, y_p_norm), dim=-1).unsqueeze(0)

        # stacking the verticle and horizontal velocity fields into shape (N=1, C=2->(u, v), H=Ny, W=Nx) so we can compute all at once 
        uv_stack = torch.stack(self.field.velo, dim=0).unsqueeze(0)

        # using pytorch's grid sample method to perform bilinear interpolation at predicted points
        uv_p_stack = F.grid_sample(uv_stack, xy_p_grid_norm, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        u_p = uv_p_stack[0]
        v_p = uv_p_stack[1]

        # because this function updates the boundary values need to reset them to their prescibed values before moving forwards
        u_p, v_p = boundary_conditions(u_p, v_p, *args, *kwargs)

        # calculating the arival points based on the newly calcualted velocity fields and then clamping it to force it to be located within the grid
        x_a = (x_p + (u_p*dt)).clamp(0,1)
        y_a = (y_p + (v_p*dt)).clamp(0,1)

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
        u_a, v_a = boundary_conditions(u_a, v_a, *args, *kwargs)

        self.field.velo[0] = 0.5*(u_a + u_p)
        self.field.velo[1] = 0.5*(v_a + v_p)