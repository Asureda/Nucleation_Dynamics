import cluster
from cluster_physics import ClusterPhysics
import numpy as np
import pint
ureg = pint.UnitRegistry()
class ClusterSimulation:
    def __init__(self, params, time_steps, dt, u, MAX_NUMBER_MOLECULES):
        self.physics_object = ClusterPhysics(params)
        self.temperature = self.physics_object.temperature.magnitude
        self.time_steps = time_steps
        self.u = u  # Umbral para tratamiento numérico
        self.i_max = MAX_NUMBER_MOLECULES  # Límite superior para cierre del sistema
        self.dt = dt
        self.precompute_total_free_energy_array(MAX_NUMBER_MOLECULES)
        self.precompute_rate_equations_array(MAX_NUMBER_MOLECULES)
        self.total_free_energy_array = self.total_free_energy_array
        self.backward_rate_array = self.backward_rate_array
        self.forward_rate_array = self.forward_rate_array
        self.cluster_array = np.zeros(MAX_NUMBER_MOLECULES)
        self.number_molecules_array = np.arange(1, MAX_NUMBER_MOLECULES+1)
        
        # Inicializar clusters
        for nof_molecules in range(1, MAX_NUMBER_MOLECULES+1):
            start = self.physics_object.number_density_equilibrium(nof_molecules).magnitude if nof_molecules <= u else 0
            self.cluster_array[nof_molecules - 1] = start

    def change_in_clusters_at_number(self, number_molecules):
        current_cluster_array = self.cluster_array[number_molecules - 1]
        if(number_molecules == 1):
            return 0
        if number_molecules>1: 
            prev_cluster_array = self.cluster_array[number_molecules - 2]
        if(number_molecules < self.i_max):
            next_cluster_array = self.cluster_array[number_molecules]
        else:
            next_cluster_array = current_cluster_array

        current_backward_rate = self.backward_rate_array[number_molecules - 1]*current_cluster_array
        current_forward_rate = self.forward_rate_array[number_molecules - 1]*current_cluster_array
        prev_forward_rate = self.forward_rate_array[number_molecules - 2]*prev_cluster_array
        next_backward_rate = self.backward_rate_array[number_molecules-1]*next_cluster_array
        
        return -current_forward_rate-current_backward_rate+prev_forward_rate+next_backward_rate
    
    def update_cluster_at_number(self, number_molecules):
        new_number_clusters_array = self.cluster_array[number_molecules - 1] + self.dt * self.change_in_clusters_at_number(number_molecules)
        self.cluster_array[number_molecules - 1] = new_number_clusters_array

    def update_all_clusters(self):
        for nof_molecules in range(1, self.i_max+1):
            self.update_cluster_at_number(nof_molecules)                
    
    def precompute_total_free_energy_array(self, max_number_of_molecules):
        self.total_free_energy_array = np.zeros(max_number_of_molecules) 
        for i in range(1, max_number_of_molecules+1):
            self.total_free_energy_array[i-1] = self.physics_object.total_free_energy(i).magnitude

    def precompute_rate_equations_array(self, max_number_of_molecules):
        self.forward_rate_array = np.zeros(max_number_of_molecules)
        self.backward_rate_array = np.zeros(max_number_of_molecules)
        for i in range(1, max_number_of_molecules+1):
            self.forward_rate_array[i-1] = self.physics_object.rate_equation(i, attachment=True).magnitude
            self.backward_rate_array[i-1] = self.physics_object.rate_equation(i, attachment=False).magnitude

    def simulate(self):
        accumulated_time = 0.0
        for time in range(self.time_steps):
            self.update_all_clusters()
            accumulated_time += self.dt
        print('accumulated_time', accumulated_time)

    def pp(self):
        for i in range(1, self.i_max+1):
            print("n cluster with", i, "molecules", self.cluster_array[i-1])
    

""" import numpy as np
import cluster
from cluster_physics import ClusterPhysics
from scipy.integrate import solve_ivp

class ClusterSimulation:
    def __init__(self, params, time_steps, dt, u, MAX_NUMBER_MOLECULES):
        self.cluster_array = np.zeros(MAX_NUMBER_MOLECULES)
        self.physics_object = ClusterPhysics(params)
        self.temperature = self.physics_object.temperature.magnitude
        self.time_steps = time_steps
        self.u = u  # Umbral para tratamiento numérico
        self.i_max = MAX_NUMBER_MOLECULES  # Límite superior para cierre del sistema
        self.dt = dt
        # Inicializar clusters
        for nof_molecules in range(1, MAX_NUMBER_MOLECULES + 1):
            start = self.physics_object.number_density_equilibrium(nof_molecules).magnitude if nof_molecules < u else 0
            self.cluster_array[nof_molecules - 1] = start

    def system_of_odes(self, t, y):
        dydt = np.zeros(len(y))
        for i in range(1, len(y) + 1):
            dydt[i - 1] = self.change_in_clusters_at_number(i, y)
        return dydt

    def change_in_clusters_at_number(self, number_molecules, y):
        # Assuming forward_rate and backward_rate are now functions of number_molecules and y
        current_cluster_rate = -y[number_molecules - 1] * (forward_rate(number_molecules, y) + backward_rate(number_molecules, y))
        prev_cluster_rate = y[number_molecules - 2] * forward_rate(number_molecules - 1, y) if number_molecules > 1 else 0
        next_cluster_rate = y[number_molecules] * backward_rate(number_molecules + 1, y) if number_molecules < self.i_max else 0
        return current_cluster_rate + prev_cluster_rate + next_cluster_rate

    def simulate(self):
        y0 = self.cluster_array.copy()
        t_span = (0, self.time_steps * self.dt)
        sol = solve_ivp(self.system_of_odes, t_span, y0, method='RK45', t_eval=np.arange(0, self.time_steps * self.dt, self.dt))

        # Update cluster numbers from the solution
        self.cluster_array = sol.y[:, -1]

        # Print results
        print('accumulated_time', sol.t[-1])
        for i in [1, 2, 23, 40]:
            print(f"n cluster with {i} molecules", self.cluster_array[i - 1])
 """
# Usage example
# params = ...
# simulation = ClusterSimulation(params, time_steps, dt, u, MAX_NUMBER_MOLECULES)
# simulation.simulate()
