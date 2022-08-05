"""
Collision models
"""

import torch

from lettuce.equilibrium import *
from lettuce.util import LettuceException

__all__ = [
    "BGKCollision", "KBCCollision2D", "KBCCollision3D", "MRTCollision", "CMCollision", "RegularizedCollision",
    "SmagorinskyCollision", "TRTCollision", "BGKInitialization", "LearnedMRT"
]


class BGKCollision:
    def __init__(self, lattice, tau, force=None):
        self.force = force
        self.lattice = lattice
        self.tau = tau

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u_eq = 0 if self.force is None else self.force.u_eq(f)
        u = self.lattice.u(f, rho=rho) + u_eq
        feq = self.lattice.equilibrium(rho, u)
        Si = 0 if self.force is None else self.force.source_term(u)
        return f - 1.0 / self.tau * (f - feq) + Si


class MRTCollision:
    """Multiple relaxation time collision operator

    This is an MRT operator in the most general sense of the word.
    The transform does not have to be linear and can, e.g., be any moment or cumulant transform.
    """

    def __init__(self, lattice, transform, relaxation_parameters):
        self.lattice = lattice
        self.transform = transform
        self.relaxation_parameters = lattice.convert_to_tensor(relaxation_parameters)

    def __call__(self, f):
        m = self.transform.transform(f)
        meq = self.transform.equilibrium(m)
        m = m - self.lattice.einsum("q,q->q", [1 / self.relaxation_parameters, m - meq])
        f = self.transform.inverse_transform(m)
        return f

class CMCollision:
    """Non-orthogonal central moments relaxing to a discrete equilibrium: \
       A D2Q9 Boltzmann model (De Rosis 2016)
    """

    def __init__(self, lattice, transform, relaxation_parameters):
        self.lattice = lattice
        self.transform = transform
        self.relaxation_parameters = lattice.convert_to_tensor(relaxation_parameters)

    def __call__(self, f):
        m, cm = self.transform.transform(f)
        cmeq = self.transform.equilibrium(cm)
        cm = cm - self.lattice.einsum("q,q->q", [1 / self.relaxation_parameters, cm - cmeq]) # computes k*
        f = self.transform.inverse_transform(m, cm) # I need m only for the velocities ux and uy
        return f

class TRTCollision:
    """Two relaxation time collision model - standard implementation (cf. Krüger 2017)
        """

    def __init__(self, lattice, tau, tau_minus=1.0):
        self.lattice = lattice
        self.tau_plus = tau
        self.tau_minus = tau_minus

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f, rho=rho)
        feq = self.lattice.equilibrium(rho, u)
        f_diff_neq = ((f + f[self.lattice.stencil.opposite]) - (feq + feq[self.lattice.stencil.opposite])) / (
                2.0 * self.tau_plus)
        f_diff_neq += ((f - f[self.lattice.stencil.opposite]) - (feq - feq[self.lattice.stencil.opposite])) / (
                2.0 * self.tau_minus)
        f = f - f_diff_neq
        return f


class RegularizedCollision:
    """Regularized LBM according to Jonas Latt and Bastien Chopard (2006)"""

    def __init__(self, lattice, tau):
        self.lattice = lattice
        self.tau = tau
        self.Q_matrix = torch.zeros([lattice.Q, lattice.D, lattice.D], device=lattice.device, dtype=lattice.dtype)

        for a in range(lattice.Q):
            for b in range(lattice.D):
                for c in range(lattice.D):
                    self.Q_matrix[a, b, c] = lattice.e[a, b] * lattice.e[a, c]
                    if b == c:
                        self.Q_matrix[a, b, c] -= lattice.cs * lattice.cs

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f, rho=rho)
        feq = self.lattice.equilibrium(rho, u)
        pi_neq = self.lattice.shear_tensor(f - feq)
        cs4 = self.lattice.cs ** 4

        pi_neq = self.lattice.einsum("qab,ab->q", [self.Q_matrix, pi_neq])
        pi_neq = self.lattice.einsum("q,q->q", [self.lattice.w, pi_neq])

        fi1 = pi_neq / (2 * cs4)
        f = feq + (1. - 1. / self.tau) * fi1

        return f


class KBCCollision2D:
    """Entropic multi-relaxation time model according to Karlin et al. in two dimensions"""

    def __init__(self, lattice, tau):
        self.lattice = lattice
        assert lattice.Q == 9, LettuceException("KBC2D only realized for D2Q9")
        self.tau = tau
        self.beta = 1. / (2 * tau)

        # Build a matrix that contains the indices
        self.M = torch.zeros([3, 3, 9], device=lattice.device, dtype=lattice.dtype)
        for i in range(3):
            for j in range(3):
                self.M[i, j] = lattice.e[:, 0] ** i * lattice.e[:, 1] ** j

    def kbc_moment_transform(self, f):
        """Transforms the f into the KBC moment representation"""
        m = torch.einsum('abq,qmn', self.M, f)
        rho = m[0, 0]
        m = m / rho
        m[0, 0] = rho

        return m

    def compute_s_seq_from_m(self, f, m):
        s = torch.zeros_like(f)

        T = m[2, 0] + m[0, 2]
        N = m[2, 0] - m[0, 2]

        Pi_xy = m[1, 1]

        s[0] = m[0, 0] * -T
        s[1] = 1. / 2. * m[0, 0] * (0.5 * (T + N))
        s[2] = 1. / 2. * m[0, 0] * (0.5 * (T - N))
        s[3] = 1. / 2. * m[0, 0] * (0.5 * (T + N))
        s[4] = 1. / 2. * m[0, 0] * (0.5 * (T - N))
        s[5] = 1. / 4. * m[0, 0] * (Pi_xy)
        s[6] = -s[5]
        s[7] = 1. / 4 * m[0, 0] * Pi_xy
        s[8] = -s[7]

        return s

    def __call__(self, f):
        # the deletes are not part of the algorithm, they just keep the memory usage lower
        rho = self.lattice.rho(f)
        u = self.lattice.u(f, rho=rho)
        feq = self.lattice.equilibrium(rho, u)
        # k = torch.zeros_like(f)

        m = self.kbc_moment_transform(f)
        delta_s = self.compute_s_seq_from_m(f, m)

        # k[0] = m[0, 0]
        # k[1] = m[0, 0] / 2. * m[1, 0]
        # k[2] = m[0, 0] / 2. * m[0, 1]
        # k[3] = -m[0, 0] / 2. * m[1, 0]
        # k[4] = -m[0, 0] / 2. * m[0, 1]
        # k[5] = 0
        # k[6] = 0
        # k[7] = 0
        # k[8] = 0

        m = self.kbc_moment_transform(feq)

        delta_s -= self.compute_s_seq_from_m(f, m)
        del m
        delta_h = f - feq - delta_s

        sum_s = self.lattice.rho(delta_s * delta_h / feq)
        sum_h = self.lattice.rho(delta_h * delta_h / feq)
        del feq
        gamma_stab = 1. / self.beta - (2 - 1. / self.beta) * sum_s / sum_h
        gamma_stab[gamma_stab < 1E-15] = 2.0
        gamma_stab[torch.isnan(gamma_stab)] = 2.0
        f = f - self.beta * (2 * delta_s + gamma_stab * delta_h)
        return f


class KBCCollision3D:
    """Entropic multi-relaxation time-relaxation time model according to Karlin et al. in three dimensions"""

    def __init__(self, lattice, tau):
        self.lattice = lattice
        assert lattice.Q == 27, LettuceException("KBC only realized for D3Q27")
        self.tau = tau
        self.beta = 1. / (2 * tau)

        # Build a matrix that contains the indices
        self.M = torch.zeros([3, 3, 3, 27], device=lattice.device, dtype=lattice.dtype)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.M[i, j, k] = lattice.e[:, 0] ** i * lattice.e[:, 1] ** j * lattice.e[:, 2] ** k

    def kbc_moment_transform(self, f):
        """Transforms the f into the KBC moment representation"""
        m = torch.einsum('abcq,qmno', self.M, f)
        rho = m[0, 0, 0]
        m = m / rho
        m[0, 0, 0] = rho

        return m

    def compute_s_seq_from_m(self, f, m):
        s = torch.zeros_like(f)

        T = m[2, 0, 0] + m[0, 2, 0] + m[0, 0, 2]
        N_xz = m[2, 0, 0] - m[0, 0, 2]
        N_yz = m[0, 2, 0] - m[0, 0, 2]
        Pi_xy = m[1, 1, 0]
        Pi_xz = m[1, 0, 1]
        Pi_yz = m[0, 1, 1]

        s[0] = m[0, 0, 0] * -T
        s[1] = 1. / 6. * m[0, 0, 0] * (2 * N_xz - N_yz + T)
        s[2] = s[1]
        s[3] = 1. / 6. * m[0, 0, 0] * (2 * N_yz - N_xz + T)
        s[4] = s[3]
        s[5] = 1. / 6. * m[0, 0, 0] * (-N_xz - N_yz + T)
        s[6] = s[5]
        s[7] = 1. / 4 * m[0, 0, 0] * Pi_yz
        s[8] = s[7]
        s[9] = - 1. / 4 * m[0, 0, 0] * Pi_yz
        s[10] = s[9]
        s[11] = 1. / 4 * m[0, 0, 0] * Pi_xz
        s[12] = s[11]
        s[13] = -1. / 4 * m[0, 0, 0] * Pi_xz
        s[14] = s[13]
        s[15] = 1. / 4 * m[0, 0, 0] * Pi_xy
        s[16] = s[15]
        s[17] = -1. / 4 * m[0, 0, 0] * Pi_xy
        s[18] = s[17]

        return s

    def __call__(self, f):
        # the deletes are not part of the algorithm, they just keep the memory usage lower
        feq = self.lattice.equilibrium(self.lattice.rho(f), self.lattice.u(f))
        # k = torch.zeros_like(f)

        m = self.kbc_moment_transform(f)
        delta_s = self.compute_s_seq_from_m(f, m)

        # k[1] = m[0, 0, 0] / 6. * (3. * m[1, 0, 0])
        # k[0] = m[0, 0, 0]
        # k[2] = -k[1]
        # k[3] = m[0, 0, 0] / 6. * (3. * m[0, 1, 0])
        # k[4] = -k[3]
        # k[5] = m[0, 0, 0] / 6. * (3. * m[0, 0, 1])
        # k[6] = -k[5]

        m = self.kbc_moment_transform(feq)

        delta_s -= self.compute_s_seq_from_m(f, m)
        del m
        delta_h = f - feq - delta_s

        sum_s = self.lattice.rho(delta_s * delta_h / feq)
        sum_h = self.lattice.rho(delta_h * delta_h / feq)
        del feq
        gamma_stab = 1. / self.beta - (2 - 1. / self.beta) * sum_s / sum_h
        gamma_stab[gamma_stab < 1E-15] = 2.0
        # Detect NaN
        gamma_stab[torch.isnan(gamma_stab)] = 2.0
        f = f - self.beta * (2 * delta_s + gamma_stab * delta_h)

        return f


class SmagorinskyCollision:
    """Smagorinsky large eddy simulation (LES) collision model with BGK operator."""

    def __init__(self, lattice, tau, smagorinsky_constant=0.17, force=None):
        self.force = force
        self.lattice = lattice
        self.tau = tau
        self.iterations = 2
        self.tau_eff = tau
        self.constant = smagorinsky_constant

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u_eq = 0 if self.force is None else self.force.u_eq(f)
        u = self.lattice.u(f) + u_eq
        feq = self.lattice.equilibrium(rho, u)
        S_shear = self.lattice.shear_tensor(f - feq)
        S_shear /= (2.0 * rho * self.lattice.cs ** 2)
        self.tau_eff = self.tau
        nu = (self.tau - 0.5) / 3.0

        for i in range(self.iterations):
            S = S_shear / self.tau_eff
            S = self.lattice.einsum('ab,ab->', [S, S])
            nu_t = self.constant ** 2 * S
            nu_eff = nu + nu_t
            self.tau_eff = nu_eff * 3.0 + 0.5
        Si = 0 if self.force is None else self.force.source_term(u)
        return f - 1.0 / self.tau_eff * (f - feq) + Si


class BGKInitialization:
    """Keep velocity constant."""

    def __init__(self, lattice, flow, moment_transformation):
        self.lattice = lattice
        self.tau = flow.units.relaxation_parameter_lu
        self.moment_transformation = moment_transformation
        p, u = flow.initial_solution(flow.grid)
        self.u = flow.units.convert_velocity_to_lu(lattice.convert_to_tensor(u))
        self.rho0 = flow.units.characteristic_density_lu 
        #self.equilibrium = FourthOrderEquilibrium(self.lattice)
        self.equilibrium = QuadraticEquilibrium(self.lattice)
        momentum_names = tuple([f"j{x}" for x in "xyz"[:self.lattice.D]])
        self.momentum_indices = moment_transformation[momentum_names]

    def __call__(self, f):
        rho = self.lattice.rho(f)
        feq = self.equilibrium(rho, self.u)
        m = self.moment_transformation.transform(f)
        meq = self.moment_transformation.transform(feq)
        mnew = m - 1.0 / self.tau * (m - meq)
        mnew[0] = m[0] - 1.0 / (self.tau + 1) * (m[0] - meq[0])
        mnew[self.momentum_indices] = rho * self.u
        f = self.moment_transformation.inverse_transform(mnew)
        return f

class LearnedMRT(torch.nn.Module):
    def __init__(self, tau, moment_transform, activation=torch.nn.ReLU()):
        super().__init__()
        self.__name__ = "Learned collision"
        self.tau = tau
        self.trafo = moment_transform  # lt.D2Q9NonOrthoCM
        # 1st net for higher order moment relaxation: Kxx + Kyy
        self.xx_net = torch.nn.Sequential(
            torch.nn.Linear(9,24),
            activation,
            torch.nn.Linear(24,1)
        )
        # 2nd net for higher order moment relaxtion: Kxx - Kyy
        self.yy_net = torch.nn.Sequential(
            torch.nn.Linear(9,24),
            activation,
            torch.nn.Linear(24,1)
        )
         # 2nd net for higher order moment relaxtion: Kxy
        self.xy_net = torch.nn.Sequential(
            torch.nn.Linear(9,24),
            activation,
            torch.nn.Linear(24,1)
        )
    # def flip_xy(self, m):
    #     """flip x and y and moments"""
    #     assert self.trafo.__class__ == D2Q9NonOrthoCM  # other moment sets have different ordering of moments
    #     return m[:,:,[0,2,1,5,4,3,6,8,7]]

    @staticmethod
    def gt_half(a, tau):
        """transform into a value > tau"""
        return tau + 0.025 * torch.relu(a)  
    def __call__(self, f):
        return self.forward(f)
    def forward(self, f):
        qdim, nxdim, nydim = f.shape  # ok
        # transform to moment space
        m, cm = self.trafo.transform(f)
        cmt = cm.permute(1, 2, 0)  # grid dimensions are batch dims for the networks #ok
        assert cmt.shape == (nxdim, nydim, qdim) #ok
        # determine higher-order moment relaxation parameters
        tau_pxx = self.gt_half(self.xx_net.forward(cmt), self.tau)
        tau_pyy = self.gt_half(self.yy_net.forward(cmt), self.tau)
        tau_pxy = self.gt_half(self.xy_net.forward(cmt), self.tau)
        # print(str(torch.min(tau_n))+"   "+str(torch.max(tau_n)))
        # by summing over xy-ordered and yx-ordered, we make tau_n rotation equivariant
        assert tau_pxx.shape == (nxdim, nydim, 1)
        assert tau_pyy.shape == (nxdim, nydim, 1)
        assert tau_pxy.shape == (nxdim, nydim, 1)
        # assign tau to moments
        taus = torch.ones_like(m)
        taus[3] = tau_pxx[...,0]
        taus[4] = tau_pyy[...,0]
        taus[5] = tau_pxy[...,0]
        assert taus.shape == (qdim, nxdim, nydim)
        # relax
        cmeq = self.trafo.equilibrium(cm)
        cm_postcollision = cm - 1./taus * (cm - cmeq)
        
        #print(f"tau_max: {torch.max(tau_pxx):2.8e}")
        return self.trafo.inverse_transform(m, cm_postcollision)