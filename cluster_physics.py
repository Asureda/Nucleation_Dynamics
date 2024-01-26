import numpy as np
import pint 

AVOGADRO = 6.022141e23 # 1/mol
BOLTZMANN = 1.3806e-23 # J/K
TEMP_INDEP_DIFFUSIVITY = 2e9 # m^2/s
ACTIVATION_ENERGY = 4.4e5 # J/mol
JUMP_DISTANCE = 4.6e-10 # m
CONSTANT_GAS = 8.314 # J/mol*K
DELTA_S = 40 # J/mol*K
SIGMA = 0.15 # J/m^2
T_M = 1300 # K

ureg = pint.UnitRegistry()

class ClusterPhysics:
    # Constantes de clase
    # Constantes de clase
    BOLTZMANN = 1.3806e-23 * ureg.joule / ureg.kelvin  # J/K
    TEMP_INDEP_DIFFUSIVITY = 2e9 * ureg.meter**2 / ureg.second  # m^2/s
    CONSTANT_GAS = 8.314 * ureg.joule / (ureg.mol * ureg.kelvin)  # J/mol*K

    def __init__(self, params):
        # Asignar valores desde el diccionario params
        self.AVOGADRO =  6.0221409e+23*ureg.Quantity(1.0,"1/mol")
        self.temperature = ureg.Quantity(params['temperature'], "kelvin")
        self.activation_energy = ureg.Quantity(params['activation_energy'], "kelvin")
        self.jump_distance = ureg.Quantity(params['jump_distance'], "m")
        self.molar_mass = ureg.Quantity(params['molar_mass'], "g/mol")
        self.mass_density = ureg.Quantity(params['mass_density'], "g/cm**3")
        self.melting_point = ureg.Quantity(params['melting_point'], "kelvin")
        self.heat_fusion = ureg.Quantity(params['heat_fusion'], "joules/mol")
        self.sigma = ureg.Quantity(params['sigma'], "joules/m**2")
        self.molar_volume = ureg.Quantity(self.molar_mass/self.mass_density, "m**3/mol")
        self.molecular_volume = ureg.Quantity(self.molar_volume / self.AVOGADRO, "m**3")
        self.entropy_fusion = ureg.Quantity(self.heat_fusion / self.melting_point, "joules/kelvin/mol")

    def print_properties(self):
        print(f"Constante de Boltzmann: {self.BOLTZMANN}")
        print(f"Constante de los gases: {self.CONSTANT_GAS}")
        print(f"Temperatura: {self.temperature}")
        print(f"Energía de activación: {self.activation_energy}")
        print(f"Distancia de salto: {self.jump_distance}")
        print(f"Molar mass: {self.molar_mass}")
        print(f"Densidad: {self.mass_density}")
        print(f"Punto de fusión: {self.melting_point}")
        print(f"Calor de fusión: {self.heat_fusion}")
        print(f"Entropía de fusión: {self.entropy_fusion}")
        print(f"Volumen molar: {self.molar_volume}")
        print(f"Volumen molecular: {self.molecular_volume}")
        print(f"Sigma: {self.sigma}")

    def diffusivity(self):
        return self.TEMP_INDEP_DIFFUSIVITY * np.exp(-self.activation_energy/(self.temperature))


def diffusivity(temperature):
    return TEMP_INDEP_DIFFUSIVITY * np.exp(-ACTIVATION_ENERGY/(CONSTANT_GAS*temperature))

def unbiased_jump_rate(temperature):
    return 6 * diffusivity(temperature) / (JUMP_DISTANCE**2)

def number_of_sites_transformation(number_of_molecules):
    return 4 * np.power(number_of_molecules,2.0/3.0)

# Calculate the change in Gibbs free energy per molecule (J/mol)
def delta_g_prime(temperature, delta_s_f, t_m):
    return -delta_s_f / AVOGADRO * (temperature - t_m)

def gibbs_free_energy(number_of_molecules, temperature, sigma, delta_s_f, t_m):
    delta_g = delta_g_prime(temperature, delta_s_f, t_m)
    return number_of_molecules * delta_g + 4 * sigma *number_of_sites_transformation(number_of_molecules)

# Forward rate constant
def forward_rate_constant(number_of_molecules, temperature, sigma, delta_s_f, t_m):
    delta_g_prime_value = delta_g_prime(temperature, delta_s_f, t_m)
    delta_g_n = gibbs_free_energy(number_of_molecules, sigma, delta_g_prime_value)
    delta_g_n_plus_1 = gibbs_free_energy(number_of_molecules + 1, sigma, delta_g_prime_value)
    omega_n = number_of_sites_transformation(number_of_molecules)
    k_plus = omega_n * np.exp(-(delta_g_n_plus_1 - delta_g_n) / (BOLTZMANN * temperature))
    return k_plus

# Backward rate constant
def backward_rate_constant(number_of_molecules, temperature, sigma, delta_s_f, t_m):
    delta_g_prime_value = delta_g_prime(temperature, delta_s_f, t_m)
    delta_g_n = gibbs_free_energy(number_of_molecules, sigma, delta_g_prime_value)
    delta_g_n_plus_1 = gibbs_free_energy(number_of_molecules + 1, sigma, delta_g_prime_value)
    omega_n = number_of_sites_transformation(number_of_molecules)
    k_minus = omega_n * np.exp((delta_g_n_plus_1 - delta_g_n) / (BOLTZMANN * temperature))
    return k_minus

def N_eq(temperature, number_of_molecules, N0):
    print(np.exp(-gibbs_free_energy(number_of_molecules, temperature, SIGMA, DELTA_S, T_M) / (BOLTZMANN * temperature)))
    return N0*AVOGADRO * np.exp(-gibbs_free_energy(number_of_molecules, temperature, SIGMA, DELTA_S, T_M) / (BOLTZMANN * temperature))