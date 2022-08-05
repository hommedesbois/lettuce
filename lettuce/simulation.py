"""Lattice Boltzmann Solver"""

from timeit import default_timer as timer
from lettuce import (
    LettuceException, get_default_moment_transform, BGKInitialization, ExperimentalWarning, torch_gradient
)
from lettuce.util import pressure_poisson
import pickle
from copy import deepcopy
import warnings
import torch
import numpy as np

__all__ = ["Simulation"]


class Simulation:
    """High-level API for simulations.

    Attributes
    ----------
    reporters : list
        A list of reporters. Their call functions are invoked after every simulation step (and before the first one).

    """

    def __init__(self, flow, lattice, collision, streaming):
        self.flow = flow
        self.lattice = lattice
        self.collision = collision
        self.streaming = streaming
        self.i = 0

        grid = flow.grid
        p, u = flow.initial_solution(grid)
        assert list(p.shape) == [1] + list(grid[0].shape), \
            LettuceException(f"Wrong dimension of initial pressure field. "
                             f"Expected {[1] + list(grid[0].shape)}, "
                             f"but got {list(p.shape)}.")
        assert list(u.shape) == [lattice.D] + list(grid[0].shape), \
            LettuceException("Wrong dimension of initial velocity field."
                             f"Expected {[lattice.D] + list(grid[0].shape)}, "
                             f"but got {list(u.shape)}.")
        u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        rho = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))
        self.f = lattice.equilibrium(rho, lattice.convert_to_tensor(u))

        self.reporters = []

        # Define masks, where the collision or streaming are not applied
        x = flow.grid
        self.no_collision_mask = lattice.convert_to_tensor(np.zeros_like(x[0], dtype=bool))
        no_stream_mask = lattice.convert_to_tensor(np.zeros(self.f.shape, dtype=bool))

        # Apply boundaries
        self._boundaries = deepcopy(self.flow.boundaries)  # store locally to keep the flow free from the boundary state
        for boundary in self._boundaries:
            if hasattr(boundary, "make_no_collision_mask"):
                self.no_collision_mask = self.no_collision_mask | boundary.make_no_collision_mask(self.f.shape)
            if hasattr(boundary, "make_no_stream_mask"):
                no_stream_mask = no_stream_mask | boundary.make_no_stream_mask(self.f.shape)
        if no_stream_mask.any():
            self.streaming.no_stream_mask = no_stream_mask

    def step(self, num_steps):
        """Take num_steps stream-and-collision steps and return performance in MLUPS."""
        start = timer()
        if self.i == 0:
            self._report()
        for _ in range(num_steps):
            # Perform the collision routine everywhere, expect where the no_collision_mask is true
            if self.i != 0: # After initialization with feq first step should be streaming 
                self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f))
            self.f = self.streaming(self.f)
            for boundary in self._boundaries:
                self.f = boundary(self.f)
            self.i += 1
            self._report()
        end = timer()
        seconds = end - start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups

    def _report(self):
        for reporter in self.reporters:
            reporter(self.i, self.flow.units.convert_time_to_pu(self.i), self.f)

    def initialize(self, max_num_steps=500, tol_pressure=0.001):
        """Iterative initialization to get moments consistent with the initial velocity.

        Using the initialization does not better TGV convergence. Maybe use a better scheme?
        """
        warnings.warn("Iterative initialization does not work well and solutions may diverge. Use with care. "
                      "Use initialize_f_neq instead.",
                      ExperimentalWarning)
        transform = get_default_moment_transform(self.lattice)
        collision = BGKInitialization(self.lattice, self.flow, transform)
        streaming = self.streaming
        p_old = 0
        for i in range(max_num_steps):
            self.f = streaming(self.f)
            self.f = collision(self.f)
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(self.f))
            if (torch.max(torch.abs(p - p_old))) < tol_pressure:
                break
            p_old = deepcopy(p)
        return i

    def initialize_pressure(self, max_num_steps=100000, tol_pressure=1e-6):
        """Reinitialize equilibrium distributions with pressure obtained by a Jacobi solver.
        Note that this method has to be called before initialize_f_neq.
        """
        u = self.lattice.u(self.f)
        rho = pressure_poisson(
            self.flow.units,
            self.lattice.u(self.f),
            self.lattice.rho(self.f),
            tol_abs=tol_pressure,
            max_num_steps=max_num_steps
        )
        self.f = self.lattice.equilibrium(rho, u)

    def initialize_f_neq(self):
        """Initialize the distribution function values. The f^(1) contributions are approximated by finite differences.
        See Krüger et al. (2017).
        """
        rho = self.lattice.rho(self.f)
        u = self.lattice.u(self.f)

        grad_u0 = torch_gradient(u[0], dx=1, order=6)[None, ...]
        grad_u1 = torch_gradient(u[1], dx=1, order=6)[None, ...]
        S = torch.cat([grad_u0, grad_u1]) # this is not the strain rate tensor S 

        if self.lattice.D == 3:
            grad_u2 = torch_gradient(u[2], dx=1, order=6)[None, ...]
            S = torch.cat([S, grad_u2])

        Pi_1 = 1.0 * self.flow.units.relaxation_parameter_lu * rho * S / self.lattice.cs ** 2
        Q = (torch.einsum('ia,ib->iab', [self.lattice.e, self.lattice.e])
             - torch.eye(self.lattice.D, device=self.lattice.device, dtype=self.lattice.dtype) * self.lattice.cs ** 2)
        Pi_1_Q = self.lattice.einsum('ab,iab->i', [Pi_1, Q])
        fneq = self.lattice.einsum('i,i->i', [self.lattice.w, Pi_1_Q])

        feq = self.lattice.equilibrium(rho, u)
        self.f = feq - fneq

    def initialize_f_col_from_macro(self, filename, it):
        
        f = torch.zeros_like(self.f)
        
        # load ns data in lattice units
        if filename == None:
            grid = self.flow.grid
            rho, u = self.flow.analytical_solution(grid, it)
            rho = self.flow.units.convert_density_to_lu(rho)
            u = self.flow.units.convert_velocity_to_lu(u)
        else:
            rho, u = self.flow.load_ns_solution(filename)
        
        rho = self.lattice.convert_to_tensor(rho)
        u = self.lattice.convert_to_tensor(u)

        grad_u0 = torch_gradient(u[0], dx=1, order=4)[None, ...]
        grad_u1 = torch_gradient(u[1], dx=1, order=4)[None, ...]
        
        grad_u = torch.cat([grad_u0, grad_u1])
        S = 0.5 * (grad_u + torch.transpose(grad_u, 0, 1))

        Pi_1 = - 1.0 * self.flow.units.relaxation_parameter_lu * rho * S / self.lattice.cs ** 2
 
        ux = u[0]
        uy = u[1]
        k3 = 2/3 * rho + Pi_1[0,0,:,:] + Pi_1[1,1,:,:] 
        k4 = Pi_1[0,0,:,:] - Pi_1[1,1,:,:] 
        k5 = Pi_1[0,1,:,:]

        f[0] = k3*(ux**2 + uy**2 - 2)/2 - k4*(ux**2 - uy**2)/2 + 4*k5*ux*uy \
             + rho*(ux**2*uy**2 - ux**2 - uy**2 + 1) + rho/9
        f[1] = -k3*(ux**2 + ux + uy**2 - 1)/4 + k4*(ux**2 + ux - uy**2 + 1)/4 \
             - k5*uy*(2*ux + 1) - rho*ux*(ux*uy**2 - ux + uy**2 - 1)/2 - rho/18
        f[2] = -k3*(ux**2 + uy**2 + uy - 1)/4 - k4*(-ux**2 + uy**2 + uy + 1)/4 \
             - k5*ux*(2*uy + 1) - rho*uy*(ux**2*uy + ux**2 - uy - 1)/2 - rho/18
        f[3] = -k3*(ux**2 - ux + uy**2 - 1)/4 + k4*(ux**2 - ux - uy**2 + 1)/4 \
             - k5*uy*(2*ux - 1) - rho*ux*(ux*uy**2 - ux - uy**2 + 1)/2 - rho/18
        f[4] = -k3*(ux**2 + uy**2 - uy - 1)/4 + k4*(ux**2 - uy**2 + uy - 1)/4 \
             - k5*ux*(2*uy - 1) - rho*uy*(ux**2*uy - ux**2 - uy + 1)/2 - rho/18
        f[5] = k3*(ux**2 + ux + uy**2 + uy)/8 - k4*(ux**2 + ux - uy**2 - uy)/8 \
             + k5*(4*ux*uy + 2*ux + 2*uy + 1)/4 + rho*ux*uy*(ux*uy + ux + uy + 1)/4 \
             + rho/36
        f[6] = k3*(ux**2 - ux + uy**2 + uy)/8 + k4*(-ux**2 + ux + uy**2 + uy)/8 \
             + k5*(4*ux*uy + 2*ux - 2*uy - 1)/4 + rho*ux*uy*(ux*uy + ux - uy - 1)/4 \
             + rho/36
        f[7] = k3*(ux**2 - ux + uy**2 - uy)/8 - k4*(ux**2 - ux - uy**2 + uy)/8 \
             + k5*(4*ux*uy - 2*ux - 2*uy + 1)/4 + rho*ux*uy*(ux*uy - ux - uy + 1)/4 \
             + rho/36
        f[8] = k3*(ux**2 + ux + uy**2 - uy)/8 - k4*(ux**2 + ux - uy**2 + uy)/8 \
             + k5*(4*ux*uy - 2*ux + 2*uy - 1)/4 + rho*ux*uy*(ux*uy - ux + uy - 1)/4 \
             + rho/36

        self.f = f



    def save_checkpoint(self, filename):
        """Write f as np.array using pickle module."""
        with open(filename, "wb") as fp:
            pickle.dump(self.f, fp)

    def load_checkpoint(self, filename):
        """Load f as np.array using pickle module."""
        with open(filename, "rb") as fp:
            self.f = pickle.load(fp)
