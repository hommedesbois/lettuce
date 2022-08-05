"""
Vortex in 2D.
Special Inputs & standard value: vortex_radius = 0.1, vortex_strength = 0.1
"""

import numpy as np
from lettuce.unit import UnitConversion
from lettuce.util import pressure_poisson_init



class Vortex2D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice, vortex_radius=0.1, 
                vortex_strength = 0.1, speed_of_sound=343.2, poisson=False):
        self.vortex_radius = vortex_radius
        self.vortex_strength = vortex_strength
        self.speed_of_sound = speed_of_sound
        self.resolution = resolution
        self.mach_number = mach_number
        self.poisson = poisson
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=mach_number*speed_of_sound, characteristic_density_pu=1
        )
        self.lattice = lattice
        self.dx = self.units.characteristic_length_pu / resolution 

    def analytical_solution(self, grid, it):
        """ Analytical solution is only implemented for a vortex translation in x-direction """
        X, Y = grid
        r_sqrd = np.zeros(np.shape(X))
        uy = np.zeros(np.shape(X))
        dt = self.dx/(np.sqrt(3)*self.speed_of_sound)
        T = it * dt
        u_ref = self.units.characteristic_velocity_pu
        mod = (T*u_ref)%self.units.characteristic_length_pu 
        
        if  mod > 0.5:
            center_pu_x = (mod - 0.5) * self.units.characteristic_length_pu
        else:
            center_pu_x = (mod + 0.5) * self.units.characteristic_length_pu 
        center_pu_y = 0.5 * self.units.characteristic_length_pu
        rad = self.vortex_radius
        
        # vortex strength
        epsilon = self.vortex_strength * self.speed_of_sound
        eoc = epsilon / self.speed_of_sound
        eoc_sqrd = eoc * eoc

        for i, x in enumerate(X[:,0]):
            for j, y in enumerate(Y[0,:]):
                if (x-center_pu_x) < -0.5: 
                    center_pu_x = center_pu_x - 1 
                elif (x-center_pu_x) > 0.5: 
                    center_pu_x = center_pu_x + 1 
                r_sqrd[i][j] = (x - center_pu_x)**2 + (y - center_pu_y)**2
                uy[i][j] =  epsilon * (x-center_pu_x)/rad * np.exp(-r_sqrd[i][j]/(2.*rad**2))
        
        # density
        rho_ref = self.units.characteristic_density_pu
        rho = rho_ref * np.exp(-0.5 * eoc_sqrd * np.exp(-r_sqrd/rad**2))
        rho = np.stack([rho])
        
        # velocity
        ux = - epsilon * (Y-center_pu_y)/rad * np.exp(-r_sqrd/(2.*rad**2)) + u_ref
        #uy =   epsilon * (X-center_pu_x)/rad * np.exp(-r_sqrd/(2.*rad**2))
        u = np.stack([ux, uy], axis=0)
        #density correction
        if self.poisson: 
            rho = pressure_poisson_init(self.units, self.lattice, u, rho)
        
        return rho, u
        
        
    def initial_solution_rho(self, grid):
        return self.analytical_solution(grid, it=0)



    def initial_solution(self, grid):   
        rho, u = self.analytical_solution(grid, it=0)
        p = self.units.convert_density_lu_to_pressure_pu(self.units.convert_density_to_lu(rho))
        
        return p, u

    def load_ns_solution(self, filename):

        res = self.resolution

        rho = np.loadtxt(filename, delimiter = " ", skiprows=1, usecols = 0)
        factor = int(np.sqrt(np.shape(rho)[0])/res)
        rho = rho.reshape((res*factor, res*factor))

        u1 = np.loadtxt(filename, delimiter = " ", skiprows=1, usecols = 1).reshape((res*factor, res*factor))
        u2 = np.loadtxt(filename, delimiter = " ", skiprows=1, usecols = 2).reshape((res*factor, res*factor))

        #sxx = np.loadtxt(filename, delimiter = " ", skiprows=1, usecols = 3).reshape((res*factor, res*factor))
        #syy = np.loadtxt(filename, delimiter = " ", skiprows=1, usecols = 4).reshape((res*factor, res*factor))
        #sxy = np.loadtxt(filename, delimiter = " ", skiprows=1, usecols = 5).reshape((res*factor, res*factor))

        rho = rho[::factor, ::factor]
        u1 = u1[::factor, ::factor]
        u2 = u2[::factor, ::factor]

        #sxx = sxx[::factor, ::factor]
        #syy = syy[::factor, ::factor]
        #sxy = sxy[::factor, ::factor]


        #rho = np.stack([rho], axis=0)
        rho = rho[None, ...]
        u = np.stack([u1, u2], axis=0)
        #s = np.stack([sxx, syy, sxy], axis=0)
        return rho, u

    @property
    def grid(self):
        #dt = self.dx/(np.sqrt(3)*self.speed_of_sound)
        #T = it * dt
        #u_ref = self.units.characteristic_velocity_pu
        #delta = T*u_ref
        delta = 0.0
        x = np.linspace(delta + 0. + 0.5 * self.dx, delta + 1. + 0.5 * self.dx, num=self.resolution, endpoint=False)
        y = np.linspace(0. + 0.5 * self.dx, 1. + 0.5 * self.dx, num=self.resolution, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        return []