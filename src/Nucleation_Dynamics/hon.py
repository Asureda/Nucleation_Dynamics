import pint
import math
import numpy as np
ureg = pint.UnitRegistry()
ureg.define('atom = mol / 6.0221409e+23 = at')

class HON:
    """
    HyTRANS mass transport class
    Stores component mass transport parameters
    """

    def __init__(self, input, c_type, time):
        self.input = input
        

        self.model = self.input["model"]
        self.atom_volume = ureg.Quantity(self.input["species_atom_volume"], "m**3")
        self.surface_tension = ureg.Quantity(self.input["surface_tension"], "joules/m**2")
        self.solubility = ureg.Quantity(self.input["solubility"], "mol/m**3/Pa**0.5")
        self.pressure = ureg.Quantity(self.input["pressure"], "Pa")
        self.supersaturation_ratio = self.input["supersaturation_ratio"]
        self.diffusivity = ureg.Quantity(self.input["diffusivity"], "m**2/s")
        self.EoS = self.input["EoS"]
        self.BOLTZMANN = 1.3806e-23 * ureg.joule / ureg.kelvin

        self.atom_radius = ureg.Quantity(((3 * self.atom_volume.magnitude) / (4 * math.pi))**(1/3), "m")
        self.atom_surface = (3 * self.atom_volume) / self.atom_radius

    def S0_HON(self, T, M, C):
        N_A =  6.0221409e+23*ureg.Quantity(1.0,"1/mol")
        C_ = self.C*ureg.Quantity(1.0,"mol/m**3")

        molar_mass = M*ureg.Quantity(1.0,"g/mol")
        atom_mass = (molar_mass / N_A).to(ureg.kg)
        if self.supersaturation_ratio > 0:
            A = ((2.0 * self.surface_tension) / (math.pi * atom_mass))**0.5
            S0 = (((self.atom_volume * N_A**2 * self.C**2) / self.supersaturation_ratio) * A).to_base_units()
        else:
            S0 = ureg.Quantity(0.0, "1/m**3/s")
        return S0

    def saturation_driving_force(self, T):
        temperature = ureg.Quantity(T, "kelvin")
    
        if self.supersaturation_ratio > 0:
            Dg = ((-self.BOLTZMANN*temperature * np.log(self.supersaturation_ratio))).to('joule')
        else:
            Dg = ureg.Quantity(0.0, "joule/m**3")
        return Dg

    def critical_DG(self, T):
        if self.Dgvol != 0:
            DG = ((16.0 * np.pi * self.surface_tension**3*self.atom_volume**2) / (3.0 * self.Dgvol**2)).to('joule')
        else:
            DG = ureg.Quantity(0.0, "joule")
        return DG
    def critical_radius(self):
        if self.Dgvol != 0:
            r = ((-2.0 * self.surface_tension*self.atom_volume) / (self.Dgvol)).to('m')
        else:
            r = ureg.Quantity(0.0, "m")
        return r

    def critical_number_atoms(self, T):
        k_b = ureg.boltzmann_constant
        T_k = T*ureg.Quantity(1.0,"K")
        Kb_T = k_b*T_k
        if self.supersaturation_ratio > 0:
            n = ((2.0 * self.DG_c) / (Kb_T * math.log(self.supersaturation_ratio))).to_base_units()
        else:
            n = ureg.Quantity(0.0, "at")
        return n

    def S_HON(self, T, M):
        self.C = self.supersaturation_ratio * self.solubility*self.pressure**0.5
        self.Dgvol = self.saturation_driving_force(T)
        self.DG_c = self.critical_DG(T)
        self.r_c = self.critical_radius()
        self.n_c = self.critical_number_atoms(T)  # at per cluster

        S0_HON = self.S0_HON(T, M)
        k_b = ureg.boltzmann_constant
        T_k = T*ureg.Quantity(1.0,"K")
        Kb_T = k_b*T_k
        EXP = -self.DG_c / (Kb_T)
        S = S0_HON * math.exp(EXP)

        if self.model == "self_consistent":
            O = (self.surface_tension * self.atom_surface) / (Kb_T)
         
            S *= O / self.supersaturation_ratio

        if S.magnitude > 1 and self.r_c > 0.0:
            return S.to_base_units()
        else:
            return ureg.Quantity(0.0, "1/m**3/s")

    def bubble_pressure(self, P, r):
        # Young-Laplace mechanical equilibrium
        return P + (2 * self.surface_tension) / r

    def bubble_radius(self, T, P, n, rho=ureg.Quantity(1000.0, "kg/m**3"), M=ureg.Quantity(10.0, "kg/mol")):
        if self.EoS == "database":
            # EoS
            V = (n * rho) / M
        else:
            # Ideal gas
            R = ureg.gas_constant
            V = (n * R * T) / P

        # then and r = (3*V)/(4*pi))**1/3
        return ((3 * V) / (4 * math.pi))**1/3