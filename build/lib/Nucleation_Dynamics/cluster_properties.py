import numpy as np
import pint
from functools import cache

ureg = pint.UnitRegistry()

class ClusterPhysics:
    """
    A class representing the physical properties and behaviors of clusters in nucleation dynamics.

    Attributes:
        ureg (pint.UnitRegistry): A unit registry from the pint library to handle physical units.
        _params (dict): A dictionary storing the physical parameters of the system.
    """

    def __init__(self, params):
        """
        Initializes the ClusterPhysics object with given parameters.

        Parameters:
            params (dict): A dictionary containing the physical parameters of the system such as temperature,
                           activation energy, diffusivity factor, jump distance, molar mass, mass density,
                           melting point, heat of fusion, surface tension (sigma), supersaturation ratio, and method.
        """
        self.ureg = ureg
        self._initialize_params(params)
        self._precompute_constants()

    def _initialize_params(self, params):
        """
        Initializes the physical parameters from the provided dictionary.

        Parameters:
            params (dict): A dictionary containing the physical parameters and their units.
        """
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
        """
        Precomputes constants that will be used in calculations, such as Avogadro's number.
        """
        self.AVOGADRO =  6.0221409e+23 * ureg.Quantity(1.0, "1/mol")

    @property
    def temperature(self):
        """Returns the temperature of the system."""
        return self._params['temperature']

    @property
    def activation_energy(self):
        """Returns the activation energy for diffusivity."""
        return self._params['activation_energy']

    @property
    def sigma(self):
        """Returns the surface tension (sigma) of the cluster material."""
        return self._params['sigma']

    @property
    def molar_mass(self):
        """Returns the molar mass of the cluster material."""
        return self._params['molar_mass']

    @property
    def diffusivity_factor(self):
        """Returns the diffusivity factor of the cluster material."""
        return self._params['diffusivity_factor']

    @property
    def jump_distance(self):
        """Returns the average jump distance for atoms or molecules in the cluster."""
        return self._params['jump_distance']

    @property
    def mass_density(self):
        """Returns the mass density of the cluster material."""
        return self._params['mass_density']

    @property
    def melting_point(self):
        """Returns the melting point of the cluster material."""
        return self._params['melting_point']

    @property
    def heat_fusion(self):
        """Returns the heat of fusion of the cluster material."""
        return self._params['heat_fusion']

    @property
    def supersaturation_ratio(self):
        """Returns the supersaturation ratio of the system."""
        return self._params['supersaturation_ratio']

    @property
    def method(self):
        """Returns the method used for calculating bulk free energy ('melting' or 'saturation')."""
        return self._params['method']

    @property
    @cache
    def diffusivity(self):
        """
        Calculates and returns the diffusivity of the cluster material, applying the Arrhenius equation.
        """
        return self._params['diffusivity_factor'] * np.exp(-self.activation_energy / self.temperature)

    @property
    @cache
    def unbiased_jump_rate(self):
        """
        Calculates and returns the unbiased jump rate of atoms or molecules in the cluster.
        """
        return 6 * self.diffusivity / (self._params['jump_distance'] ** 2)

    @property
    @cache
    def molar_volume(self):
        """
        Calculates and returns the molar volume of the cluster material.
        """
        return self._params['molar_mass'] / self._params['mass_density']

    @property
    @cache
    def molecular_volume(self):
        """
        Calculates and returns the molecular volume of the cluster material.
        """
        return (self.molar_volume / self.AVOGADRO).to_base_units()

    @property
    @cache
    def entropy_fusion(self):
        """
        Calculates and returns the entropy of fusion of the cluster material.
        """
        return self._params['heat_fusion'] / self._params['melting_point']

    @property
    @cache
    def bulk_free_energy(self):
        """
        Calculates and returns the bulk free energy of the cluster based on the specified method.
        """
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
        """
        Calculates and returns the surface free energy of the cluster.
        """
        a = (36 * np.pi * self.molecular_volume ** 2) ** (1 / 3)
        return a * self._params['sigma']

    @property
    @cache
    def critical_energy_barrier(self):
        """
        Calculates and returns the critical energy barrier for cluster formation.
        """
        return (16 * np.pi / 3) * self._params['sigma'] ** 3 * self.molecular_volume ** 2 / (self.bulk_free_energy ** 2)

    @property
    @cache
    def critical_radius(self):
        """
        Calculates and returns the critical radius for cluster formation.
        """
        return -(2 * self._params['sigma'] / (self.bulk_free_energy / self.molecular_volume)).to_base_units()

    @property
    @cache
    def critical_number_of_molecules(self):
        """
        Calculates and returns the critical number of molecules for cluster formation.
        """

        return (4/3)*np.pi*self.critical_radius**3/self.molecular_volume #-2 * self.surface_free_energy / self.bulk_free_energy

    def total_free_energy(self, number_of_molecules):
        """
        Calculates the total free energy of a cluster with a given number of molecules.

        Parameters:
            number_of_molecules (int or array-like): The number of molecules in the cluster.

        Returns:
            pint.Quantity: The total free energy of the cluster.
        """
        number_of_molecules = np.array(number_of_molecules, ndmin=1)
        total_energy = np.where(number_of_molecules < 1,
                                0 * ureg.joule,
                                self.bulk_free_energy * number_of_molecules +
                                self.surface_free_energy * number_of_molecules ** (2 / 3))
        if total_energy.size == 1:
            return total_energy[0]
        return total_energy
        
    def rate_equation(self, number_of_molecules, attachment=True):
        """
        Calculates the rate of attachment or detachment of molecules to/from a cluster.

        Parameters:
            number_of_molecules (int): The number of molecules in the cluster.
            attachment (bool): True for attachment rate, False for detachment rate.

        Returns:
            pint.Quantity: The rate of attachment or detachment.
        """
        delta_energy = self.total_free_energy(number_of_molecules + 1) - self.total_free_energy(number_of_molecules)
        if attachment:
            forward_rate = 4 * number_of_molecules ** (2/3) * self.unbiased_jump_rate * np.exp(-delta_energy / (2 * ureg.boltzmann_constant * self.temperature))
            return forward_rate
        else:
            return 4 * number_of_molecules ** (2/3) * self.unbiased_jump_rate * np.exp(delta_energy / (2 * ureg.boltzmann_constant * self.temperature))

    def number_density_equilibrium(self, number_of_molecules):
        """
        Calculates the equilibrium number density of clusters with a given number of molecules.

        Parameters:
            number_of_molecules (int): The number of molecules in the cluster.

        Returns:
            pint.Quantity: The equilibrium number density of the clusters.
        """
        B1 = self.AVOGADRO*np.exp(self.total_free_energy(1) / (ureg.boltzmann_constant * self.temperature))
        return (B1 * np.exp(-self.total_free_energy(number_of_molecules) / (ureg.boltzmann_constant * self.temperature))).to_base_units()

    def stationary_rate(self, number_of_molecules, number_of_sites):
        """
        Calculates the stationary rate of cluster formation or dissolution.

        Parameters:
            number_of_molecules (int): The number of molecules in the cluster.
            number_of_sites (int): The number of available sites for cluster formation.

        Returns:
            pint.Quantity: The stationary rate of cluster formation or dissolution.
        """
        return self.rate_equation(number_of_molecules, attachment=True) * self.number_density_equilibrium(number_of_molecules)
    
    def dr_dt(self, t, r):
        """
        Differential rate equation for cluster growth or shrinkage.

        Parameters:
            t (float): Time variable, not used in the calculation but required for differential equation solvers.
            r (float): The radius of the cluster.

        Returns:
            float: The rate of change of the cluster radius with respect to time.
        """
        bulk_g = self.bulk_free_energy().magnitude
        return (16 * self.diffusivity.magnitude / self.jump_distance.magnitude**2) * (3 * self.molecular_volume.magnitude / (4 * np.pi))**(1/3) * np.sinh(
            (self.molecular_volume.magnitude / (2 * ureg.boltzmann_constant.magnitude * self.temperature.magnitude)) * (bulk_g/self.molecular_volume.magnitude - 2*self.sigma.magnitude/r)
        )
