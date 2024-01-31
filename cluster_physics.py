import numpy as np
import pint 

ureg = pint.UnitRegistry()

class ClusterPhysics:
    # Constantes de clase
    # Constantes de clase
    BOLTZMANN = 1.3806e-23 * ureg.joule / ureg.kelvin  # J/K
    #TEMP_INDEP_DIFFUSIVITY = 2e9 * ureg.meter**2 / ureg.second  # m^2/s
    TEMP_INDEP_DIFFUSIVITY = 1.38 * ureg.meter**2 / ureg.second  # m^2/s
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
        self.S = params['supersaturation_ratio']
        self.method = params['method']
        #print(0.71*self.entropy_fusion*self.temperature/(self.AVOGADRO**(1/3)*self.molar_volume**(2/3)))
        
        
    def print_properties(self):
        properties = [
            ("Constante de Boltzmann", self.BOLTZMANN),
            ("Constante de los gases", self.CONSTANT_GAS),
            ("Temperatura", self.temperature),
            ("Energía de activación", self.activation_energy),
            ("Distancia de salto", self.jump_distance),
            ("Masa molar", self.molar_mass),
            ("Densidad", self.mass_density),
            ("Punto de fusión", self.melting_point),
            ("Calor de fusión", self.heat_fusion),
            ("Entropía de fusión", self.entropy_fusion),
            ("Volumen molar", self.molar_volume),
            ("Volumen molecular", self.molecular_volume),
            ("Sigma", self.sigma),
        ]
    
        for name, value in properties:
                print(f"{name}: {value}")

    def diffusivity(self):
        return self.TEMP_INDEP_DIFFUSIVITY * np.exp(-self.activation_energy/(self.temperature))
    
    def unbiased_jump_rate(self):
        return 6 * self.diffusivity() / (self.jump_distance**2)
    
    def number_of_sites_transformation(self, number_of_molecules):
        return 4 * np.power(number_of_molecules,2.0/3.0)
    
    def bulk_free_energy(self):
        if self.method == 'melting':
            return self.entropy_fusion * (self.temperature - self.melting_point) / self.AVOGADRO
        elif self.method == 'saturation':
            return - ClusterPhysics.BOLTZMANN * self.temperature * np.log(self.S)
        else:
            raise ValueError("Invalid method. Choose 'melting' or 'saturation'.")

    def surface_free_energy(self):
        a = (36*np.pi*self.molecular_volume**2)**(1/3)
        return a*self.sigma

    def total_free_energy(self, number_of_molecules):
        #if number_of_molecules <= 1:
        #    return 0*ureg.joule
        return self.bulk_free_energy() * number_of_molecules + self.surface_free_energy() * number_of_molecules ** (2/3)

    def critical_energy_barrier(self):
        return (16 * np.pi / 3) * self.sigma ** 3 * self.molecular_volume ** 2 / (self.bulk_free_energy() ** 2)

    def critical_radius(self):
        return -(2 * self.sigma / (self.bulk_free_energy() / self.molecular_volume)).to_base_units()

    def critical_number_of_molecules(self):
        return -2 * self.critical_energy_barrier() / self.bulk_free_energy()

    def rate_equation(self, number_of_molecules, attachment=True):
        delta_energy = self.total_free_energy(number_of_molecules + 1) - self.total_free_energy(number_of_molecules)
        if attachment:
            return 4 * number_of_molecules ** (2/3) * self.unbiased_jump_rate() * np.exp(-delta_energy / (2 * ClusterPhysics.BOLTZMANN * self.temperature))
        else:
            return 4 * number_of_molecules ** (2/3) * self.unbiased_jump_rate() * np.exp(delta_energy / (2 * ClusterPhysics.BOLTZMANN * self.temperature))

    def number_density_equilibrium(self, number_of_molecules):
        return self.AVOGADRO * np.exp(-self.total_free_energy(number_of_molecules) / (ClusterPhysics.BOLTZMANN * self.temperature))

    def stationary_rate(self, number_of_molecules, number_of_sites):
        return self.rate_equation(number_of_molecules, attachment=True) * self.number_density_equilibrium(number_of_molecules)
    
    def dr_dt(self, t, r):
        bulk_g = self.bulk_free_energy().magnitude
        return (16 * self.diffusivity().magnitude / self.jump_distance.magnitude**2) * (3 * self.molecular_volume.magnitude / (4 * np.pi))**(1/3) * np.sinh(
            (self.molecular_volume.magnitude / (2 * ClusterPhysics.BOLTZMANN.magnitude * self.temperature.magnitude)) * (bulk_g/self.molecular_volume.magnitude - 2*self.sigma.magnitude/r)
        )


""" AVOGADRO = 6.022141e23 # 1/mol
BOLTZMANN = 1.3806e-23 # J/K
TEMP_INDEP_DIFFUSIVITY = 2e9 # m^2/s
ACTIVATION_ENERGY = 4.4e5 # J/mol
JUMP_DISTANCE = 4.6e-10 # m
CONSTANT_GAS = 8.314 # J/mol*K
DELTA_S = 40 # J/mol*K
SIGMA = 0.15 # J/m^2
T_M = 1300 # K
 """