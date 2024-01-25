import cluster
import cluster_physics as phys

MAX_NUMBER_MOLECULES = 4
T = 750
number_clusters_start = 10
dt = 1e-8/phys.jump_rate(T)
dt = dt * 5e3
time_steps = int(1e5)
 
class ClusterSimulation:
    def __init__(self, temperature, time_steps, u, MAX_NUMBER_MOLECULES):
        self.cluster_dict = {}
        self.temperature = temperature
        self.time_steps = time_steps
        self.u = u  # Umbral para tratamiento numérico
        self.i_max = MAX_NUMBER_MOLECULES  # Límite superior para cierre del sistema

        # Inicializar clusters
        for nof_molecules in range(1, MAX_NUMBER_MOLECULES + 1):
            start = N_eq(temperature, nof_molecules) if nof_molecules < u else 0
            new_cluster = cluster.Cluster(temperature, nof_molecules, start)
            self.cluster_dict[nof_molecules] = new_cluster

    def change_in_clusters_at_number(self, number_molecules):
        # Condiciones de límite
        if number_molecules == 1:
            return 0
        if number_molecules >= self.i_max:
            return 0

        current_cluster = self.cluster_dict[number_molecules]
        prev_cluster = self.cluster_dict.get(number_molecules - 1, None)
        next_cluster = self.cluster_dict.get(number_molecules + 1, None)

        # Calcular cambios de tasa
        rate_change = 0
        if prev_cluster:
            print('prev_cluster', number_molecules)
            rate_change += prev_cluster.forward_rate()
        rate_change -= current_cluster.backward_rate()
        if next_cluster and number_molecules < self.i_max:
            rate_change -= current_cluster.forward_rate()
            rate_change += next_cluster.backward_rate()

        return rate_change

    # ... el resto de la clase permanece igual
    def update_cluster_at_number(self, number_molecules):
        new_number_clusters = self.cluster_dict[number_molecules].get_number_of_clusters() + dt * self.change_in_clusters_at_number(number_molecules)
        self.cluster_dict[number_molecules].set_number_of_clusters(new_number_clusters)

    def update_all_clusters(self):
        for number_of_molecules in self.cluster_dict.keys():
            self.update_cluster_at_number(number_of_molecules)                
    
    def simulate(self):
        accumulated_time = 0.0
        for time in range(self.time_steps):
            self.update_all_clusters()
            accumulated_time += dt
        print('accumulated_time', accumulated_time)

    def pp(self):
        for key in self.cluster_dict.keys():
            print( key, self.cluster_dict[key].get_number_of_clusters())

    def number_molecules_as_array(self):
        return self.cluster_dict.keys()

    def number_clusters_as_array(self):
        return [ self.cluster_dict[key].get_number_of_clusters() for key in self.cluster_dict.keys() ]
    
