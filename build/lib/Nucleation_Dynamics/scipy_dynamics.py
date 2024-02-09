from Nucleation_Dynamics.cluster_properties import ClusterPhysics
import numpy as np
import pint
import time
from scipy.integrate import solve_ivp
from numba import njit

ureg = pint.UnitRegistry()

class ScipyClusterDynamics:
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
        equilibrium_densities = np.array(
            [self.physics_object.number_density_equilibrium(i).magnitude for i in range(1, u + 1)])
        self.cluster_array[:u] = equilibrium_densities

    @staticmethod
    @njit
    def dy_dt_closed(t, y, forward_rate_array, backward_rate_array):
        changes = np.zeros(len(y))
        changes[1:-1] = -forward_rate_array[1:-1] * y[1:-1] - \
                        backward_rate_array[1:-1] * y[1:-1] + \
                        forward_rate_array[:-2] * y[:-2] + \
                        backward_rate_array[2:] * y[2:]
        changes[-1] = -backward_rate_array[-1] * y[-1] + \
                      forward_rate_array[-2] * y[-2]
        return changes

    @staticmethod
    @njit
    def dy_dt_open(t, y, forward_rate_array, backward_rate_array):
        changes = np.zeros(len(y))
        changes[1:-1] = -forward_rate_array[1:-1] * y[1:-1] - \
                        backward_rate_array[1:-1] * y[1:-1] + \
                        forward_rate_array[:-2] * y[:-2] + \
                        backward_rate_array[2:] * y[2:]
        changes[-1] = -backward_rate_array[-1] * y[-1] + \
                      forward_rate_array[-2] * y[-2] - \
                      forward_rate_array[-1] * y[-1]
        return changes
    
    def precompute_total_free_energy_array(self, max_number_of_molecules):
        self.total_free_energy_array = np.array([self.physics_object.total_free_energy(i).magnitude for i in range(1, max_number_of_molecules + 1)])

    def precompute_rate_equations_array(self, max_number_of_molecules):
        self.forward_rate_array = np.array([self.physics_object.rate_equation(i, attachment=True).magnitude for i in range(1, max_number_of_molecules + 1)])
        self.backward_rate_array = np.array([self.physics_object.rate_equation(i, attachment=False).magnitude for i in range(1, max_number_of_molecules + 1)])

    def simulate(self, method='RK45', t_span=None, y0=None, t_eval=None, rtol=1e-3, atol=1e-6, **kwargs):
        start_time = time.time()
        if t_span is None:
            t_span = [0, self.dt * self.time_steps]
        if y0 is None:
            y0 = self.cluster_array
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 100)

        # Define una función envoltura que llama a la función dy_dt correcta con parámetros adicionales
        def dy_dt_wrapper(t, y):
            if self.boundary_type == 'closed':
                return ScipyClusterDynamics.dy_dt_closed(t, y, self.forward_rate_array, self.backward_rate_array)
            else:
                return ScipyClusterDynamics.dy_dt_open(t, y, self.forward_rate_array, self.backward_rate_array)

        # Llamar a solve_ivp con la función envoltura
        sol = solve_ivp(fun=dy_dt_wrapper, t_span=t_span, y0=y0, method=method, t_eval=t_eval, rtol=rtol, atol=atol, vectorized=False, **kwargs)

        if sol.success:
            # Almacenar la solución y en cluster_array
            self.time = sol.t
            self.cluster_array = sol.y
            # Inicializar un array para almacenar dy/dt para cada t_eval
            self.dydt_array = np.zeros_like(sol.y)
            # Calcular dy/dt para cada t_eval usando la función envoltura
            for i, t in enumerate(sol.t):
                self.dydt_array[:, i] = dy_dt_wrapper(t, sol.y[:, i])
        else:
            print("La integración no fue exitosa.")

        end_time = time.time()
        self.execution_time = end_time - start_time
        self.success = sol.success
        self.nfev = sol.nfev

        self.results = {
            'execution_time': self.execution_time,
            'success': self.success,
            'nfev': self.nfev,
            # Añadir más métricas según sea necesario
        } 

        print(self.results)   
    def pp(self):
        for i in range(1, self.i_max + 1):
            print("n cluster with", i, "molecules", self.cluster_array[i - 1])