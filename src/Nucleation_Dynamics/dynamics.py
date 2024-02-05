from Nucleation_Dynamics.cluster_properties import ClusterPhysics
import numpy as np
import pint
import time
from numba import njit

ureg = pint.UnitRegistry()

@njit
def rk4_step(y, dt, dy_dt, forward_rate_array, backward_rate_array, i_max):
    """
    Performs a single RK4 step.

    Parameters:
        y (numpy.ndarray): Current state of the system (cluster sizes).
        dt (float): Time step size.
        dy_dt (function): Function that computes the derivative of the state (rate of change of cluster sizes).
        forward_rate_array (numpy.ndarray): Array of forward rates for each cluster size.
        backward_rate_array (numpy.ndarray): Array of backward rates for each cluster size.
        i_max (int): Maximum number of molecules in a cluster.

    Returns:
        numpy.ndarray: Updated state of the system after the RK4 step.
    """
    k1 = dt * dy_dt(y, forward_rate_array, backward_rate_array, i_max)
    k2 = dt * dy_dt(y + 0.5 * k1, forward_rate_array, backward_rate_array, i_max)
    k3 = dt * dy_dt(y + 0.5 * k2, forward_rate_array, backward_rate_array, i_max)
    k4 = dt * dy_dt(y + k3, forward_rate_array, backward_rate_array, i_max)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

class ClusterDynamics:
    """
    A class to simulate cluster dynamics based on nucleation theory and cluster properties.

    Attributes:
        physics_object (ClusterPhysics): An object to access physical properties and calculations.
        temperature (float): The temperature of the system.
        time_steps (int): Number of time steps for the simulation.
        u (int): Threshold for numerical treatment.
        i_max (int): Maximum number of molecules in a cluster for the system closure.
        dt (float): Time step size.
        cluster_array (numpy.ndarray): Array to store the number of clusters at each size.
        number_molecules_array (numpy.ndarray): Array of cluster sizes.
        total_free_energy_array (numpy.ndarray): Precomputed total free energy for each cluster size.
        forward_rate_array (numpy.ndarray): Precomputed forward rate for each cluster size.
        backward_rate_array (numpy.ndarray): Precomputed backward rate for each cluster size.
    """
    
    def __init__(self, params, time_steps, dt, u, MAX_NUMBER_MOLECULES, record_frequency):
        """
        Initializes the ClusterDynamics object with simulation parameters and precomputes necessary arrays.

        Parameters:
            params (dict): Parameters for the ClusterPhysics object.
            time_steps (int): Number of time steps for the simulation.
            dt (float): Time step size.
            u (int): Threshold for numerical treatment.
            MAX_NUMBER_MOLECULES (int): Maximum number of molecules in a cluster for the system closure.
        """
        self.physics_object = ClusterPhysics(params)
        self.temperature = self.physics_object.temperature.magnitude
        self.time_steps = time_steps
        self.u = u
        self.i_max = MAX_NUMBER_MOLECULES
        self.dt = dt
        self.record_frequency = record_frequency
        self.precompute_total_free_energy_array(MAX_NUMBER_MOLECULES)
        self.precompute_rate_equations_array(MAX_NUMBER_MOLECULES)
        self.cluster_array = np.zeros(MAX_NUMBER_MOLECULES)
        self.number_molecules_array = np.arange(1, MAX_NUMBER_MOLECULES + 1)
        equilibrium_densities = np.array([self.physics_object.number_density_equilibrium(i).magnitude for i in range(1, u + 1)])
        self.cluster_array[:u] = equilibrium_densities
        # Adjust the size of cluster_evolution and rates_evolution arrays based on record_frequency
        self.cluster_evolution = np.zeros((self.i_max, self.time_steps // self.record_frequency + (self.time_steps % self.record_frequency > 0)))
        self.rates_evolution = np.zeros((self.i_max, self.time_steps // self.record_frequency + (self.time_steps % self.record_frequency > 0)))

    @staticmethod
    @njit
    def update_clusters(cluster_array, forward_rate_array, backward_rate_array, i_max):
        """
        Updates the cluster sizes based on forward and backward rates using a JIT-compiled function for speed.

        Parameters:
            cluster_array (numpy.ndarray): Current cluster sizes.
            forward_rate_array (numpy.ndarray): Forward rates for cluster growth.
            backward_rate_array (numpy.ndarray): Backward rates for cluster shrinkage.
            dt (float): Time step size.
            i_max (int): Maximum number of molecules in a cluster.

        Returns:
            numpy.ndarray: Updated cluster sizes.
        """
        changes = np.zeros(i_max)
        changes[1:-1] = -forward_rate_array[1:-1] * cluster_array[1:-1] - \
                        backward_rate_array[1:-1] * cluster_array[1:-1] + \
                        forward_rate_array[:-2] * cluster_array[:-2] + \
                        backward_rate_array[2:] * cluster_array[2:]
        changes[-1] = -backward_rate_array[-1] * cluster_array[-1] + \
                      forward_rate_array[-2] * cluster_array[-2] #- forward_rate_array[-1] * cluster_array[-1]
        return changes

    @staticmethod
    @njit
    def update_clusters_with_boundary_condition_closed(cluster_array, forward_rate_array, backward_rate_array, dt, i_max, initial_N1):
        """
        Updates the cluster sizes based on forward and backward rates using a JIT-compiled function for speed,
        and applies the boundary condition for N1.

        Parameters:
            cluster_array (numpy.ndarray): Current cluster sizes.
            forward_rate_array (numpy.ndarray): Forward rates for cluster growth.
            backward_rate_array (numpy.ndarray): Backward rates for cluster shrinkage.
            dt (float): Time step size.
            i_max (int): Maximum number of molecules in a cluster.
            initial_N1 (float): Initial number of clusters containing one molecule.

        Returns:
            numpy.ndarray: Updated cluster sizes with the boundary condition applied.
        """
        # Update cluster sizes based on rates
        changes = np.zeros(i_max)
        changes[1:-1] = -forward_rate_array[1:-1] * cluster_array[1:-1] - \
                        backward_rate_array[1:-1] * cluster_array[1:-1] + \
                        forward_rate_array[:-2] * cluster_array[:-2] + \
                        backward_rate_array[2:] * cluster_array[2:]
        changes[-1] = -backward_rate_array[-1] * cluster_array[-1] + \
                    forward_rate_array[-2] * cluster_array[-2]
        updated_cluster_array = cluster_array + dt * changes

        # Apply the boundary condition for N1
        sum_iNi = np.sum(np.arange(2, i_max + 1) * updated_cluster_array[1:])
        updated_cluster_array[0] = initial_N1 - sum_iNi

        return updated_cluster_array

    # def update_all_clusters(self):
    #     """
    #     Updates all cluster sizes for the current time step.
    #     """
    #     self.cluster_array = self.update_clusters(self.cluster_array, self.forward_rate_array, self.backward_rate_array, self.dt, self.i_max)

        #initial_N1 = self.cluster_array[0]
        #self.cluster_array = self.update_clusters_with_boundary_condition_closed(
        #        self.cluster_array, self.forward_rate_array, self.backward_rate_array, self.dt, self.i_max, initial_N1
        #    )

    def precompute_total_free_energy_array(self, max_number_of_molecules):
        """
        Precomputes the total free energy for each cluster size up to the maximum number of molecules.

        Parameters:
            max_number_of_molecules (int): Maximum number of molecules to precompute the free energy for.
        """
        self.total_free_energy_array = np.array([self.physics_object.total_free_energy(i).magnitude for i in range(1, max_number_of_molecules + 1)])

    def precompute_rate_equations_array(self, max_number_of_molecules):
        """
        Precomputes the forward and backward rates for each cluster size up to the maximum number of molecules.

        Parameters:
            max_number_of_molecules (int): Maximum number of molecules to precompute rates for.
        """
        self.forward_rate_array = np.array([self.physics_object.rate_equation(i, attachment=True).magnitude for i in range(1, max_number_of_molecules + 1)])
        self.backward_rate_array = np.array([self.physics_object.rate_equation(i, attachment=False).magnitude for i in range(1, max_number_of_molecules + 1)])

    def simulate(self):
        """
        Simulates the cluster dynamics over the specified number of time steps and prints the computation time.
        """
        start_time = time.time()
        record_index = 0
        for step in range(self.time_steps):
            changes = self.update_clusters(self.cluster_array, self.forward_rate_array, self.backward_rate_array, self.i_max)
            self.cluster_array = self.cluster_array + self.dt * changes
            # Record the state at specified intervals
            if step % self.record_frequency == 0:
                self.cluster_evolution[:, record_index] = self.cluster_array
                self.rates_evolution[:, record_index] = changes
                record_index += 1
        end_time = time.time()
        computation_time = end_time - start_time
        print('Computation time: {:.4f} seconds'.format(computation_time))

    def compute_analytical_steady_state(self):
        equilibrium_densities = np.array([self.physics_object.number_density_equilibrium(i).magnitude for i in range(1, self.i_max + 1)])
        equilibrium_sum = np.sum((self.forward_rate_array*equilibrium_densities)**-1)
        return 1/equilibrium_sum
    def pp(self):
        """
        Pretty prints the number of clusters for each size.
        """
        for i in range(1, self.i_max + 1):
            print("n cluster with", i, "molecules", self.cluster_array[i - 1])
