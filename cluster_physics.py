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
        self.S = 15

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
    
    def unbiased_jump_rate(self):
        return 6 * self.diffusivity() / (self.jump_distance**2)
    
    def number_of_sites_transformation(self, number_of_molecules):
        return 4 * np.power(number_of_molecules,2.0/3.0)
    
    def bulk_free_energy_melting(self):
        return self.entropy_fusion*(self.temperature - self.melting_point)/self.AVOGADRO
    
    def bulk_free_energy_saturation(self):
        return - ClusterPhysics.BOLTZMANN*self.temperature*np.log(self.S)
    
    def surface_free_energy(self):
        a = (36*np.pi*self.molecular_volume**2)**(1/3)
        return a*self.sigma
    
    def total_free_energy_melting(self, number_of_molecules):
        return self.bulk_free_energy_melting()*number_of_molecules + self.surface_free_energy()*number_of_molecules**(2/3)

    def total_free_energy_saturation(self, number_of_molecules):
        return self.bulk_free_energy_saturation()*number_of_molecules + self.surface_free_energy()*number_of_molecules**(2/3)
  
    def critical_energy_barrier_melting(self):
        return (16*np.pi/3)*self.sigma**3*self.molecular_volume**2/(self.bulk_free_energy_melting()**2)
    
    def critical_energy_barrier_saturation(self):
        return (16*np.pi/3)*self.sigma**3*self.molecular_volume**2/(self.bulk_free_energy_saturation()**2)
    
    def critical_radius_melting(self):
        return -(2*self.sigma/(self.bulk_free_energy_melting()/self.molecular_volume)).to_base_units()

    def critical_radius_saturation(self):
        return -(2*self.sigma/(self.bulk_free_energy_saturation()/self.molecular_volume)).to_base_units()
    
    def critical_number_of_molecules_melting(self):
        return (-2*self.critical_energy_barrier_melting()/self.bulk_free_energy_melting())

    def critical_number_of_molecules_saturation(self):
        return (-2*self.critical_energy_barrier_saturation()/self.bulk_free_energy_saturation())
    
    def attachment_rate_melting(self, number_of_molecules):
        return 4*number_of_molecules**(2/3)*self.unbiased_jump_rate()*np.exp(-(self.total_free_energy_melting(number_of_molecules+1)-self.total_free_energy_melting(number_of_molecules))/(2*ClusterPhysics.BOLTZMANN*self.temperature))

    def attachment_rate_saturation(self, number_of_molecules):
        return 4*number_of_molecules**(2/3)*self.unbiased_jump_rate()*np.exp(-(self.total_free_energy_saturation(number_of_molecules+1)-self.total_free_energy_saturation(number_of_molecules))/(2*ClusterPhysics.BOLTZMANN*self.temperature))

    def detachment_rate_melting(self, number_of_molecules):
        return 4*number_of_molecules**(2/3)*self.unbiased_jump_rate()*np.exp((self.total_free_energy_melting(number_of_molecules+1)-self.total_free_energy_melting(number_of_molecules))/(2*ClusterPhysics.BOLTZMANN*self.temperature))

    def detachment_rate_saturation(self, number_of_molecules):
        return 4*number_of_molecules**(2/3)*self.unbiased_jump_rate()*np.exp((self.total_free_energy_saturation(number_of_molecules+1)-self.total_free_energy_saturation(number_of_molecules))/(2*ClusterPhysics.BOLTZMANN*self.temperature))

    def number_density_equilibrium(self, number_of_molecules, number_of_sites):
        return 
    
    def number_density_equilibrium_saturation(self, number_of_molecules):
        return 
    
def N_eq(temperature, number_of_molecules, N0):
    print(np.exp(-gibbs_free_energy(number_of_molecules, temperature, SIGMA, DELTA_S, T_M) / (BOLTZMANN * temperature)))
    return N0*AVOGADRO * np.exp(-gibbs_free_energy(number_of_molecules, temperature, SIGMA, DELTA_S, T_M) / (BOLTZMANN * temperature))