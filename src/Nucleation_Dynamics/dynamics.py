from Nucleation_Dynamics.cluster_properties import ClusterPhysics
import numpy as np
import pint
import time
from scipy.integrate import solve_ivp

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

    def dy_dt(self, t, y):
        changes = np.zeros(self.i_max)
        changes[1:-1] = -self.forward_rate_array[1:-1] * y[1:-1] - \
                        self.backward_rate_array[1:-1] * y[1:-1] + \
                        self.forward_rate_array[:-2] * y[:-2] + \
                        self.backward_rate_array[2:] * y[2:]
        changes[-1] = -self.backward_rate_array[-1] * y[-1] + \
                      self.forward_rate_array[-2] * y[-2]
        return changes

    def precompute_total_free_energy_array(self, max_number_of_molecules):
        self.total_free_energy_array = np.array([self.physics_object.total_free_energy(i).magnitude for i in range(1, max_number_of_molecules + 1)])

    def precompute_rate_equations_array(self, max_number_of_molecules):
        self.forward_rate_array = np.array([self.physics_object.rate_equation(i, attachment=True).magnitude for i in range(1, max_number_of_molecules + 1)])
        self.backward_rate_array = np.array([self.physics_object.rate_equation(i, attachment=False).magnitude for i in range(1, max_number_of_molecules + 1)])

    def simulate(self):
        start_time = time.time()
        t_span = [0, self.time_steps * self.dt]
        sol = solve_ivp(self.dy_dt, t_span, self.cluster_array, method='RK45', t_eval=np.linspace(t_span[0], t_span[1], self.time_steps))
        self.cluster_array = sol.y[:, -1]
        end_time = time.time()
        computation_time = end_time - start_time
        print('Computation time: {:.4f} seconds'.format(computation_time))

    def pp(self):
        for i in range(1, self.i_max + 1):
            print("n cluster with", i, "molecules", self.cluster_array[i - 1])
