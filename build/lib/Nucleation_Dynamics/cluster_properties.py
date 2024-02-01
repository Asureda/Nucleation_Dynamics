import numpy as np
import pint
from functools import cache

ureg = pint.UnitRegistry()

class ClusterPhysics:
    def __init__(self, params):
        self.ureg = ureg
        self._initialize_params(params)
        self._precompute_constants()

    def _initialize_params(self, params):
        self._params = {key: ureg.Quantity(value, unit) if key != 'method' and key != 'supersaturation_ratio' else value
                        for key, value, unit in [
                            ('temperature', params['temperature'], "kelvin"),
                            ('activation_energy', params['activation_energy'], "kelvin"),
                            ('diffusivity_factor', params['diffusivity_factor'], 'meter**2 / second'),
                            ('jump_distance', params['jump_distance'], "meter"),
                            ('molar_mass', params['molar_mass'], "gram / mol"),
                            ('mass_density', params['mass_density'], "gram / centimeter**3"),
                            ('melting_point', params['melting_point'], "kelvin"),
                            ('heat_fusion', params['heat_fusion'], "joule / mol"),
                            ('sigma', params['sigma'], "joule / meter**2"),
                            ('supersaturation_ratio', params['supersaturation_ratio'], None),
                            ('method', params['method'], None)
                        ]}

    def _precompute_constants(self):
        self.AVOGADRO =  6.0221409e+23*ureg.Quantity(1.0,"1/mol")

    @property
    def temperature(self):
        return self._params['temperature']

    @property
    def activation_energy(self):
        return self._params['activation_energy']

    @property
    def sigma(self):
        return self._params['sigma']

    @property
    def molar_mass(self):
        return self._params['molar_mass']

    @property
    def diffusivity_factor(self):
        return self._params['diffusivity_factor']

    @property
    def jump_distance(self):
        return self._params['jump_distance']

    @property
    def mass_density(self):
        return self._params['mass_density']

    @property
    def melting_point(self):
        return self._params['melting_point']

    @property
    def heat_fusion(self):
        return self._params['heat_fusion']

    @property
    def supersaturation_ratio(self):
        return self._params['supersaturation_ratio']

    @property
    def method(self):
        return self._params['method']

    @property
    @cache
    def diffusivity(self):
        return self._params['diffusivity_factor'] * np.exp(-self.activation_energy / self.temperature)

    @property
    @cache
    def unbiased_jump_rate(self):
        return 6 * self.diffusivity / (self._params['jump_distance'] ** 2)

    @property
    @cache
    def molar_volume(self):
        return self._params['molar_mass'] / self._params['mass_density']

    @property
    @cache
    def molecular_volume(self):
        return (self.molar_volume / self.AVOGADRO).to_base_units()

    @property
    @cache
    def entropy_fusion(self):
        return self._params['heat_fusion'] / self._params['melting_point']

    @property
    @cache
    def bulk_free_energy(self):
        method = self._params.get('method')
        if method == 'melting':
            return (self.entropy_fusion * (self.temperature - self._params['melting_point']) / self.AVOGADRO).to('joule')
        elif method == 'saturation':
            S = self._params.get('supersaturation_ratio')
            return -(self.boltzmann_constant * self.temperature * np.log(S)).to('joule')
        else:
            raise ValueError("Invalid method. Choose 'melting' or 'saturation'.")

    @property
    @cache
    def surface_free_energy(self):
        a = (36 * np.pi * self.molecular_volume ** 2) ** (1 / 3)
        return a * self._params['sigma']

    @property
    @cache
    def critical_energy_barrier(self):
        return (16 * np.pi / 3) * self._params['sigma'] ** 3 * self.molecular_volume ** 2 / (self.bulk_free_energy ** 2)

    @property
    @cache
    def critical_radius(self):
        return -(2 * self._params['sigma'] / (self.bulk_free_energy / self.molecular_volume)).to_base_units()

    @property
    @cache
    def critical_number_of_molecules(self):
        return -2 * self.critical_energy_barrier / self.bulk_free_energy

    def total_free_energy(self, number_of_molecules):
        # Convertir la entrada en un array de NumPy si aún no lo es
        number_of_molecules = np.array(number_of_molecules, ndmin=1)
        
        # Calcular la energía total para cada valor en el array
        total_energy = np.where(number_of_molecules < 1,
                                0 * ureg.joule,
                                self.bulk_free_energy * number_of_molecules +
                                self.surface_free_energy * number_of_molecules ** (2 / 3))
        
        if total_energy.size == 1:
            return total_energy[0]
        return total_energy
        
    def rate_equation(self, number_of_molecules, attachment=True):
        delta_energy = self.total_free_energy(number_of_molecules + 1) - self.total_free_energy(number_of_molecules)
        if attachment:
            forward_rate = 4 * number_of_molecules ** (2/3) * self.unbiased_jump_rate * np.exp(-delta_energy / (2 * ureg.boltzmann_constant * self.temperature))
            return forward_rate
        else:
            return 4 * number_of_molecules ** (2/3) * self.unbiased_jump_rate * np.exp(delta_energy / (2 * ureg.boltzmann_constant * self.temperature))

    def number_density_equilibrium(self, number_of_molecules):
        return (self.AVOGADRO * np.exp(-self.total_free_energy(number_of_molecules) / (ureg.boltzmann_constant * self.temperature))).to_base_units()

    def stationary_rate(self, number_of_molecules, number_of_sites):
        return self.rate_equation(number_of_molecules, attachment=True) * self.number_density_equilibrium(number_of_molecules)
    
    def dr_dt(self, t, r):
        bulk_g = self.bulk_free_energy().magnitude
        return (16 * self.diffusivity.magnitude / self.jump_distance.magnitude**2) * (3 * self.molecular_volume.magnitude / (4 * np.pi))**(1/3) * np.sinh(
            (self.molecular_volume.magnitude / (2 * ureg.boltzmann_constant.magnitude * self.temperature.magnitude)) * (bulk_g/self.molecular_volume.magnitude - 2*self.sigma.magnitude/r)
        )
