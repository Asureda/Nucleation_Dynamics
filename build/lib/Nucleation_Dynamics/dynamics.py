from Nucleation_Dynamics.cluster_properties import ClusterPhysics
import numpy as np
import pint
import time
from numba import njit  # Importa njit de Numba

ureg = pint.UnitRegistry()

class ClusterDynamics:
    def __init__(self, params, time_steps, dt, u, MAX_NUMBER_MOLECULES):
        self.physics_object = ClusterPhysics(params)
        self.temperature = self.physics_object.temperature.magnitude
        self.time_steps = time_steps
        self.u = u
        self.i_max = MAX_NUMBER_MOLECULES
        self.dt = dt
        self.precompute_total_free_energy_array(MAX_NUMBER_MOLECULES)
        self.precompute_rate_equations_array(MAX_NUMBER_MOLECULES)
        self.cluster_array = np.zeros(MAX_NUMBER_MOLECULES)
        self.number_molecules_array = np.arange(1, MAX_NUMBER_MOLECULES + 1)
        equilibrium_densities = np.array([self.physics_object.number_density_equilibrium(i).magnitude for i in range(1, u + 1)])
        self.cluster_array[:u] = equilibrium_densities
        #self.cluster_array[0] = self.physics_object.AVOGADRO.magnitude

    @staticmethod
    @njit
    def update_clusters(cluster_array, forward_rate_array, backward_rate_array, dt, i_max):
        changes = np.zeros(i_max)
        changes[1:-1] = -forward_rate_array[1:-1] * cluster_array[1:-1] - \
                        backward_rate_array[1:-1] * cluster_array[1:-1] + \
                        forward_rate_array[:-2] * cluster_array[:-2] + \
                        backward_rate_array[2:] * cluster_array[2:]
        changes[-1] = -backward_rate_array[-1] * cluster_array[-1] + \
                      forward_rate_array[-2] * cluster_array[-2]
        return cluster_array + dt * changes

    def update_all_clusters(self):
        self.cluster_array = self.update_clusters(self.cluster_array, self.forward_rate_array, self.backward_rate_array, self.dt, self.i_max)

    def precompute_total_free_energy_array(self, max_number_of_molecules):
        self.total_free_energy_array = np.array([self.physics_object.total_free_energy(i).magnitude for i in range(1, max_number_of_molecules + 1)])

    def precompute_rate_equations_array(self, max_number_of_molecules):
        self.forward_rate_array = np.array([self.physics_object.rate_equation(i, attachment=True).magnitude for i in range(1, max_number_of_molecules + 1)])
        self.backward_rate_array = np.array([self.physics_object.rate_equation(i, attachment=False).magnitude for i in range(1, max_number_of_molecules + 1)])

    def simulate(self):
        start_time = time.time()
        for _ in range(self.time_steps):
            self.update_all_clusters()
        end_time = time.time()
        computation_time = end_time - start_time
        print('Computation time: {:.4f} seconds'.format(computation_time))

    def pp(self):
        for i in range(1, self.i_max + 1):
            print("n cluster with", i, "molecules", self.cluster_array[i - 1])
