"""
Vortex in 2D.
Special Inputs & standard value: vortex_radius = 0.1, vortex_strength = 0.1
"""

import numpy as np
from lettuce.unit import UnitConversion


class Vortex2D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice, vortex_radius=0.1,
                 vortex_strength=0.1, speed_of_sound=343.2):
        self.vortex_radius = vortex_radius
        self.vortex_strength = vortex_strength
        self.speed_of_sound = speed_of_sound
        self.resolution = resolution
        self.mach_number = mach_number
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=mach_number*speed_of_sound, characteristic_density_pu=1
        )

    def analytic_solution(self, grid):
        return self.initial_solution(grid)

    def initial_solution(self, grid):
        """ There is an unsually strong initial bang """
        x, y = grid

        gamma = self.vortex_strength
        cs_pu = self.speed_of_sound
        
        center_pu = 0.5 * self.units.characteristic_length_pu
        rad = self.vortex_radius
        r_sqrd = (x-center_pu)**2 + (y-center_pu)**2
        
        u_ref = self.units.characteristic_velocity_pu
        ux = - gamma * cs_pu / rad * (y-center_pu) * np.exp(.5 - (r_sqrd/(2.*rad**2))) + u_ref
        uy =   gamma * cs_pu / rad * (x-center_pu) * np.exp(.5 - (r_sqrd/(2.*rad**2)))
        u = np.stack([ux, uy], axis=0)
        
        rho_ref = self.units.characteristic_density_pu
        rho_corr = -0.5 * rho_ref * gamma**2 / 2 * np.exp(1.0-(r_sqrd/rad**2))
        rho = rho_ref +  rho_corr + 1./2.8 * rho_corr**2
        rho = np.stack([rho])
        p = self.units.convert_density_lu_to_pressure_pu(self.units.convert_density_to_lu(rho))
        return p, u

    @property
    def grid(self):
        x = np.linspace(0., 1., num=self.resolution, endpoint=False)
        y = np.linspace(0., 1., num=self.resolution, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        return []