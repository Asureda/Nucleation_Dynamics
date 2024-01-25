import numpy as np

AVOGADRO = 6.022141e23 # 1/mol
BOLTZMANN = 1.3806e-23 # J/K
TEMP_INDEP_DIFFUSIVITY = 2e9 # m^2/s
ACTIVATION_ENERGY = 4.4e5 # J/mol
JUMP_DISTANCE = 4.6e-10 # m
CONSTANT_GAS = 8.314 # J/mol*K
DELTA_S = 40 # J/mol*K
SIGMA = 0.15 # J/m^2

def diffusivity(temperature):
    return TEMP_INDEP_DIFFUSIVITY * np.exp(-ACTIVATION_ENERGY/(CONSTANT_GAS*temperature))

def unbiased_jump_rate(temperature):
    return 6 * diffusivity(temperature) / (JUMP_DISTANCE**2)

def number_of_sites_transformation(number_of_molecules):
    return 4 * number_of_molecules ** (2/3)

def gibbs_free_energy_difference(temperature, delta_s_f, t_m):
    delta_g_prime = -delta_s_f / AVOGADRO * (temperature - t_m)
    return delta_g_prime

def gibbs_free_energy(temperature, number_of_molecules, sigma, delta_s_f, t_m):
    delta_g_prime = gibbs_free_energy_difference(temperature, delta_s_f, t_m)
    return number_of_molecules * delta_g_prime + 4 * sigma * number_of_molecules**(2/3)

def rate_constant(temperature, number_of_molecules, sigma, delta_s_f, t_m):
    omega_n = number_of_sites_transformation(number_of_molecules)
    delta_g_n = gibbs_free_energy(temperature, number_of_molecules, sigma, delta_s_f, t_m)
    delta_g_n_plus_1 = gibbs_free_energy(temperature, number_of_molecules + 1, sigma, delta_s_f, t_m)
    delta_g = delta_g_n_plus_1 - delta_g_n
    k_plus = omega_n * np.exp(-delta_g / (2 * BOLTZMANN * temperature))
    k_minus = omega_n * np.exp(delta_g / (2 * BOLTZMANN * temperature))
    return k_plus, k_minus

# Example usage:
sigma = 1.0  # example value for interfacial free energy per molecular site
delta_s_f = 1.0  # example value for entropy difference
t_m = 1.0  # example value for melting temperature
temperature = 750  # Temperature in Kelvin
number_of_molecules = 100  # Number of molecules in the cluster

# Calculate forward and backward rate constants for the cluster
k_plus, k_minus = rate_constant(temperature, number_of_molecules, sigma, delta_s_f, t_m)
print(f"Forward rate constant (k_plus): {k_plus}")
print(f"Backward rate constant (k_minus): {k_minus}")
