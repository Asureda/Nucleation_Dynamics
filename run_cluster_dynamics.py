from cluster_physics import *
import cluster
from cluster_sim import ClusterSimulation

# Ejemplo de uso de la clase
# Ejemplo de uso de la clase
# Ejemplo de uso de la clase
params = {
    'temperature': 668,
    'activation_energy': 52920.2,
    'jump_distance': 1.17e-10,
    'molar_mass': 7.95,
    'mass_density': 8.2,
    'melting_point': 965.15,
    'heat_fusion': 22.5939 * 1e3,
    'supersaturation_ratio': 2.5,
    'sigma': 1.34, 
    'method': 'saturation'
}

cluster_physics = ClusterPhysics(params)

cluster_physics.print_properties()

MAX_NUMBER_MOLECULES = int(2.5*cluster_physics.critical_number_of_molecules().magnitude)
number_clusters_start = int(0.5*cluster_physics.critical_number_of_molecules().magnitude)
dt = 1e-8/cluster_physics.unbiased_jump_rate().magnitude
dt = dt*1e4

time_step_array = [int(1e3), int(1e5)]

x_array = []
y_array = []
for ts in time_step_array:
    sim = ClusterSimulation(params,ts,dt, number_clusters_start, MAX_NUMBER_MOLECULES)
    sim.simulate()
    x = sim.number_molecules_as_array()
    y = sim.number_clusters_as_array()
    x_array.append(x)
    y_array.append(y)

data = {
    'Time Step': time_step_array,
    'Number of Molecules': x_array,
    'Number of Clusters': y_array
}

df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo CSV
df.to_csv('cluster_simulation_data.csv', index=False)