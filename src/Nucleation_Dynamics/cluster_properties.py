import numpy as np
import pint
import json
from functools import cache

ureg = pint.UnitRegistry()

class ClusterPhysics:
    """
    A class representing the physical properties and behaviors of clusters in nucleation dynamics.

    Attributes:
        ureg (pint.UnitRegistry): A unit registry from the pint library to handle physical units.
        _params (dict): A dictionary storing the physical parameters of the system.
    """

    def __init__(self, json_file_path, method="melting", binary_mixture_model = None, liquid_fraction_A = None, liquidus_temperature = None, 
                 equilibrium_solid_fraction = None, solid_fraction = None, molar_volume_A = None, molar_volume_B = None, entropy_fusion_A = None, entropy_fusion_B = None):
        """
        Initializes the ClusterPhysics object with parameters loaded from a JSON file.

        Parameters:
            json_file_path (str): The path to the JSON file containing the physical parameters of the system.
        """
        self.ureg = ureg
        self._method = method
        params = self._load_params_from_json(json_file_path)  # Cargar parámetros desde un archivo JSON
        self._initialize_params(params)
        if self._method == "binary_mixture":
            self.binary_mixture = binary_mixture_model
            self.liquid_fraction_A = liquid_fraction_A
            self.liquidus_temperature = ureg.Quantity(liquidus_temperature, "kelvin")
            if self.binary_mixture == "full_model":
                self.equilibrium_solid_fraction = equilibrium_solid_fraction
                self.solid_fraction = solid_fraction
                self.molar_volume_A = ureg.Quantity(molar_volume_A, "meter**3/mole")
                self.molar_volume_B = ureg.Quantity(molar_volume_B, "meter**3/mole")
                self.entropy_fusion_A = ureg.Quantity(entropy_fusion_A, "joule/(mole*kelvin)")
                self.entropy_fusion_B = ureg.Quantity(entropy_fusion_B, "joule/(mole*kelvin)")

        self._precompute_constants()

    def _load_params_from_json(self, file_path):
        """
        Loads physical parameters from a JSON file.

        Parameters:
            file_path (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the physical parameters.
        """
        with open(file_path, 'r') as json_file:
            params = json.load(json_file)
        return params

    def _initialize_params(self, params):
        """
        Initializes the physical parameters from the provided dictionary.

        Parameters:
            params (dict): A dictionary containing the physical parameters and their units.
        """
        self._params = {key: ureg.Quantity(value, unit) if key != 'method' and key != 'supersaturation_ratio' else value
                        for key, value, unit in [
                            ('temperature', params['temperature'], "kelvin"),
                            ('activation_energy', params['activation_energy'], "kelvin"),  # Corregido a "joule / mol"
                            ('diffusivity_factor', params['diffusivity_factor'], 'meter**2 / second'),
                            ('jump_distance', params['jump_distance'], "meter"),
                            ('interface_layer', params["interface_layer"], "angstrom"),
                            ('molar_mass', params['molar_mass'], "gram / mol"),
                            ('mass_density', params['mass_density'], "gram / centimeter**3"),
                            ('melting_point', params['melting_point'], "kelvin"),
                            ('heat_fusion', params['heat_fusion'], "joule / mol"),
                            ('sigma', params['sigma'], "joule / meter**2"),
                            ('supersaturation_ratio', params['supersaturation_ratio'], None)
                        ]}
                
    def _precompute_constants(self):
        """
        Precomputes constants that will be used in calculations, such as Avogadro's number.
        """
        self.AVOGADRO =  6.0221409e+23 * ureg.Quantity(1.0, "1/mol")
        self.constant_gas =  8.314 * ureg.Quantity(1.0, "joules/(mol*kelvin)")
    @property
    def temperature(self):
        """Returns the temperature of the system."""
        return self._params['temperature']
    
    @temperature.setter
    def temperature(self, value):
        self._params['temperature'] = ureg.Quantity(value, 'kelvin')

    @property
    def activation_energy(self):
        """Returns the activation energy for diffusivity."""
        return self._params['activation_energy']
    
    @activation_energy.setter
    def activation_energy(self, value):
        self._params['activation_energy'] = self.ureg.Quantity(value, 'kelvin')

    @property
    def sigma(self):
        """Returns the surface tension (sigma) of the cluster material."""
        return self._params['sigma']
    
    @sigma.setter
    def sigma(self, value):
        self._params['sigma'] = self.ureg.Quantity(value, 'joule/meter**2')

    @property
    def molar_mass(self):
        """Returns the molar mass of the cluster material."""
        return self._params['molar_mass']
    
    @molar_mass.setter
    def molar_mass(self, value):
        self._params['molar_mass'] = self.ureg.Quantity(value, 'gram/mol')

    @property
    def diffusivity_factor(self):
        """Returns the diffusivity factor of the cluster material."""
        return self._params['diffusivity_factor']
    
    @diffusivity_factor.setter
    def diffusivity_factor(self, value):
        self._params['diffusivity_factor'] = self.ureg.Quantity(value, 'meter**2/second')

    @property
    def jump_distance(self):
        """Returns the average jump distance for atoms or molecules in the cluster."""
        return self._params['jump_distance']
    
    @jump_distance.setter
    def jump_distance(self, value):
        self._params['jump_distance'] = self.ureg.Quantity(value, 'meter')

    @property
    def interface_layer(self):
        """Returns the average jump distance for atoms or molecules in the cluster."""
        return self._params['interface_layer']

    @property
    def mass_density(self):
        """Returns the mass density of the cluster material."""
        self._params['mass_density']
    
    @mass_density.setter
    def mass_density(self, value):
        self._params['mass_density'] = self.ureg.Quantity(value, 'gram/centimeter**3')

    @property
    def melting_point(self):
        """Returns the melting point of the cluster material."""
        return self._params['melting_point']
    
    @melting_point.setter
    def melting_point(self, value):
        self._params['melting_point'] = self.ureg.Quantity(value, 'kelvin')

    @property
    def heat_fusion(self):
        """Returns the heat of fusion of the cluster material."""
        return self._params['heat_fusion']
    
    @heat_fusion.setter
    def heat_fusion(self, value):
        self._params['heat_fusion'] = self.ureg.Quantity(value, 'joule/mol')

    @property
    def supersaturation_ratio(self):
        """Returns the supersaturation ratio of the system."""
        return self._params['supersaturation_ratio']
    
    @supersaturation_ratio.setter
    def supersaturation_ratio(self, value):
        self._params['supersaturation_ratio'] = value    

    @property
    def method(self):
        """Returns the method used for calculating bulk free energy ('melting' or 'saturation')."""
        return self._method

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
        method = self.method
        if method == 'melting':
            return (self.entropy_fusion * (self.temperature - self._params['melting_point']) / self.AVOGADRO).to('joule')
        elif method == 'saturation':
            S = self._params.get('supersaturation_ratio')
            return -((ureg.boltzmann_constant * self.temperature * np.log(S)).to('joule')).to('joule')
        elif method == "binary_mixture":
            if self.binary_mixture == "dilute":
                driving_force = self.molecular_volume*(self.liquidus_temperature - self.temperature)*(self.entropy_fusion - self.constant_gas*np.log(self.liquid_fraction_A))/self.molar_volume.to("meter**3/mole")
                return -(driving_force).to("joules")
            else:
                self.binary_volume = self.molar_volume_A*self.liquid_fraction_A + self.molar_volume_B*(1-self.liquid_fraction_A)
                self.driving_force_A = self.molecular_volume*((self.liquidus_temperature - self.temperature)*self.entropy_fusion_A + self.constant_gas*self.temperature*np.log(self.liquid_fraction_A/self.solid_fraction) - self.constant_gas*self.liquidus_temperature*np.log(self.liquid_fraction_A/self.equilibrium_solid_fraction))/self.binary_volume.to("meter**3/mole")
                return -(self.driving_force_A).to("joules")
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

    def total_free_energy(self, number_of_molecules, method = None):
        """
        Calculates the total free energy of a cluster with a given number of molecules.

        Parameters:
            number_of_molecules (int or array-like): The number of molecules in the cluster.

        Returns:
            pint.Quantity: The total free energy of the cluster.
        """

        number_of_molecules = np.array(number_of_molecules, ndmin=1)
        if method == "diffuse_interface":
            Rs = (3*number_of_molecules*self.molecular_volume/(4*np.pi))**(1/3)
            print(Rs)
            free_energy = -(4*np.pi/3)*((Rs -  self._params['interface_layer'])**3*self.heat_fusion/self.molar_volume - Rs**3*self.temperature*self.entropy_fusion/self.molar_volume).to('joule')

            total_energy = np.where(number_of_molecules <= 1,
                        0 * ureg.joule,
                        free_energy * number_of_molecules +
                        self.surface_free_energy * number_of_molecules ** (2 / 3))
        else:
            total_energy = np.where(number_of_molecules <= 1,
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
        return (self.AVOGADRO * np.exp(-self.total_free_energy(number_of_molecules) / (ureg.boltzmann_constant * self.temperature))).to_base_units()

    def surface_energy_correlation(self, heat_fusion, melting_points, fractions, type="bcc"):
        """
        Calculates the equilibrium number density of clusters with a given number of molecules.

        Parameters:
            number_of_molecules (int): The number of molecules in the cluster.

        Returns:
            pint.Quantity: The equilibrium number density of the clusters.
        """
        if type == "bcc" : 
            alpha = 0.71 
        elif type == "fcc":
            alpha = 0.86
        heat_fusion_a = ureg.Quantity(heat_fusion[0],"joule/mol")
        heat_fusion_b = ureg.Quantity(heat_fusion[1],"joule/mol")
        melting_point_a = ureg.Quantity(melting_points[0],"kelvin")
        melting_point_b = ureg.Quantity(melting_points[1],"kelvin")
        entropy_a = heat_fusion_a/melting_point_a
        entropy_b = heat_fusion_b/melting_point_b
        entropy = fractions[0]*entropy_a + fractions[1]*entropy_b
        print("Entropy",entropy)
        print("Entropy_a",entropy_a)
        print("Entropy_b",entropy_b)
        
        return (alpha*entropy*self.temperature/(self.AVOGADRO*self.molar_volume**2)**(1/3)).to("joule/meter**2")

    # Define the function to calculate the given formula
    def calculate_delta_mu_A(self, entropy_fusion, melting_point, x_A_i, x_A_s, x_A_s_eq):
        """
        Calculate the change in chemical potential for component A.
        
        Parameters:
        T_i (float): Initial temperature.
        T (float): Final temperature.
        delta_S_A (float): Entropy change for component A.
        R (float): Universal gas constant.
        x_A_i (float): Initial mole fraction of A.
        x_A_s (float): Mole fraction of A at the surface.
        x_A_s_eq (float): Equilibrium mole fraction of A at the surface.
        
        Returns:
        float: Calculated change in chemical potential for component A.
        """
        entropy_fusion_ = ureg.Quantity(entropy_fusion,"joule/(mol*kelvin)")
        melting_point_ = ureg.Quantity(melting_point,"kelvin")
        # Using np.log for natural logarithm
        term1 = (self.temperature - melting_point_) * entropy_fusion_/self.AVOGADRO
        term2 = (ureg.boltzmann_constant * self.temperature * np.log(x_A_i / x_A_s)).to('joule')
        term3 = (ureg.boltzmann_constant * melting_point_ * np.log(x_A_i / x_A_s_eq)).to('joule')

        print(( self.temperature - melting_point_)*(entropy_fusion_/self.AVOGADRO-ureg.boltzmann_constant*np.log(x_A_i)))
        print(term1)
        print(term2)
        print(term3)
        
        return term1+term2+term3
    
    def stationary_rate(self, start, max_number_molecules):
        equilibrium_densities = np.array([self.number_density_equilibrium(i).magnitude for i in range(start, max_number_molecules + 1)])
        forward_rate_array = np.array([self.rate_equation(i, attachment=True).magnitude for i in range(start, max_number_molecules + 1)])
        equilibrium_sum = np.sum((forward_rate_array*equilibrium_densities)**-1)
        return (1/equilibrium_sum)*ureg("1/(mole*second)")
    
    def free_energy_second_derivative(self, number_molecules, method=None):
        """
        Calculates the second derivative of the total free energy with respect to the number of molecules,
        evaluated at the critical number of molecules, considering the discrete nature of molecule numbers.

        Parameters:
            critical_number_of_molecules (int): The critical number of molecules for cluster formation.
            method (str): Optional method argument to pass different total_free_energy computation methods.

        Returns:
            pint.Quantity: The second derivative of the total free energy evaluated at the critical number of molecules.
        """
        # Since the number of molecules is discrete, we use delta = 1
        delta_n = 1
        f_x = self.total_free_energy(number_molecules, method=method)
        f_x_plus_delta = self.total_free_energy(number_molecules + delta_n, method=method)
        f_x_minus_delta = self.total_free_energy(number_molecules - delta_n, method=method)

        # Compute the second derivative using the central difference formula for discrete variables
        second_derivative = (f_x_plus_delta - 2 * f_x + f_x_minus_delta) / delta_n ** 2
        return second_derivative
    
    def zeldovich_factor(self, number_molecules, method=None):
        """
        Calculates the Zeldovich factor for the critical number of molecules.

        Parameters:
            critical_number_of_molecules (int): The critical number of molecules for cluster formation.

        Returns:
            pint.Quantity: The Zeldovich factor for the critical number of molecules.
        """
        return ((np.abs(self.free_energy_second_derivative(number_molecules, method =None)) / (2 * np.pi * ureg.boltzmann_constant * self.temperature)) ** (1 / 2)).to_base_units()

    def turnbull_solution(self, time, number_molecules, max_number_molecules, method=None):
        t = time*ureg.second
        stationary_rate = self.stationary_rate(int(self.critical_number_of_molecules.magnitude + 1), max_number_molecules)
        radius = (3*number_molecules*self.molecular_volume/(4*np.pi))**(1/3)
        critical_rate = self.rate_equation(self.critical_number_of_molecules, attachment=True)
        zeldovich_factor = self.zeldovich_factor(self.critical_number_of_molecules, method=None)
        tau = (2*np.pi*critical_rate*zeldovich_factor**2)**-1
        utf = 2*self.critical_radius*(self.bulk_free_energy*tau/(ureg.boltzmann_constant*self.temperature))**-1
        first_term = (radius/utf).to_base_units()
        second_term =( np.log(6*self.critical_energy_barrier/(ureg.boltzmann_constant*self.temperature))).to_base_units()
        third_term = (np.log(radius/self.critical_radius - 1)).to_base_units()
        t_i = first_term + tau*(second_term - 2 + third_term)
        
        return stationary_rate*np.exp(-np.exp(-(t-t_i)/tau))
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
