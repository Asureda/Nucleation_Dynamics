from Nucleation_Dynamics.cluster_properties import ClusterPhysics
import numpy as np
import pint
import time
from diffeqpy import de
from juliacall import Main as jl
ureg = pint.UnitRegistry()

class JuliaClusterDynamics:
    def __init__(self, params, time_steps, dt, u, MAX_NUMBER_MOLECULES, boundary_type='closed'):
        self.physics_object = ClusterPhysics(params)
        self.temperature = self.physics_object.temperature.magnitude
        self.time_steps = time_steps
        self.u = u
        self.i_max = MAX_NUMBER_MOLECULES
        self.dt = dt
        self.boundary_type = boundary_type
        self.precompute_total_free_energy_array(MAX_NUMBER_MOLECULES)
        self.precompute_rate_equations_array(MAX_NUMBER_MOLECULES)
        self.cluster_array = np.zeros(MAX_NUMBER_MOLECULES)
        self.number_molecules_array = np.arange(1, MAX_NUMBER_MOLECULES + 1)
        equilibrium_densities = np.array([self.physics_object.number_density_equilibrium(i).magnitude for i in range(1, u + 1)])
        self.cluster_array[:u] = equilibrium_densities

    def dy_dt(self, t, y, p):
        changes = np.zeros(self.i_max)
        changes[1:-1] = -p[0][1:-1] * y[1:-1] - \
                        p[1][1:-1] * y[1:-1] + \
                        p[0][:-2] * y[:-2] + \
                        p[1][2:] * y[2:]
        changes[-1] = -p[1][-1] * y[-1] + \
                      p[0][-2] * y[-2]
        if self.boundary_type == 'open':
            changes[-1] -= p[0][-1] * y[-1]
        return changes

    def precompute_total_free_energy_array(self, max_number_of_molecules):
        self.total_free_energy_array = np.array([self.physics_object.total_free_energy(i).magnitude for i in range(1, max_number_of_molecules + 1)])

    def precompute_rate_equations_array(self, max_number_of_molecules):
        self.forward_rate_array = np.array([self.physics_object.rate_equation(i, attachment=True).magnitude for i in range(1, max_number_of_molecules + 1)])
        self.backward_rate_array = np.array([self.physics_object.rate_equation(i, attachment=False).magnitude for i in range(1, max_number_of_molecules + 1)])

    def simulate(self, t_span=None, y0=None, t_eval=None, rtol=1e-3, atol=1e-6):
        start_time = time.time()
        if t_span is None:
            t_span = (0.0, self.dt * self.time_steps)
        if y0 is None:
            y0 = self.cluster_array
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 10)

        # Define the differential equation system
        def f(t, y, p):
            return self.dy_dt(t, y, p)

        # Define the parameters for the differential equation system
        p = (self.forward_rate_array, self.backward_rate_array)

        # Define the differential equation problem
        prob = de.ODEProblem(f, y0, t_span, p)

        # Solve the differential equation problem
        sol = de.solve(prob)

        end_time = time.time()
        self.execution_time = end_time - start_time

        print('Computation time: {:.4f} seconds'.format(self.execution_time))
