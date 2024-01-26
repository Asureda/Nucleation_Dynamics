import cluster
import cluster_physics as phys

MAX_NUMBER_MOLECULES = 4
T = 750
number_clusters_start = 10
dt = 1e-8/phys.jump_rate(T)
dt = dt * 5e3
time_steps = int(1e5)
N0 = 1 
class ClusterSimulation:
    def __init__(self, temperature, time_steps, N0, u, MAX_NUMBER_MOLECULES):
        self.cluster_dict = {}
        self.temperature = temperature
        self.time_steps = time_steps
        self.u = u  # Umbral para tratamiento numérico
        self.i_max = MAX_NUMBER_MOLECULES  # Límite superior para cierre del sistema
        self.N0 = N0

        # Inicializar clusters
        for nof_molecules in range(1, MAX_NUMBER_MOLECULES + 1):
            start = phys.N_eq(self.temperature, nof_molecules, self.N0) if nof_molecules < u else 0
            new_cluster = cluster.Cluster(temperature, nof_molecules, start)
            self.cluster_dict[nof_molecules] = new_cluster

    def change_in_clusters_at_number(self, number_molecules):
        # Handling the lower boundary condition at 'u'
        if number_molecules <= self.u:
            # Use effective forward rate for cluster 'u'
            current_cluster = self.cluster_dict[number_molecules]
            next_cluster = self.cluster_dict.get(number_molecules + 1, None)
            rate_change = -current_cluster.effective_forward_rate()  # Need to define 'effective_forward_rate'
            if next_cluster:
                rate_change += next_cluster.backward_rate()
            return rate_change

        # Handling the upper boundary condition at 'i_max'
        elif number_molecules == self.i_max:
            prev_cluster = self.cluster_dict.get(number_molecules - 1, None)
            # No backward rate from i_max to i_max + 1, effectively setting it to 0
            rate_change = prev_cluster.forward_rate() if prev_cluster else 0
            return rate_change

        # Handling all other cases
        elif 1 < number_molecules < self.i_max:
            current_cluster = self.cluster_dict[number_molecules]
            prev_cluster = self.cluster_dict.get(number_molecules - 1, None)
            next_cluster = self.cluster_dict.get(number_molecules + 1, None)
            rate_change = 0
            if prev_cluster:
                rate_change += prev_cluster.forward_rate()
            rate_change -= current_cluster.forward_rate() + current_cluster.backward_rate()
            if next_cluster:
                rate_change += next_cluster.backward_rate()
            return rate_change
        else:
            return 0    # ... el resto de la clase permanece igual
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
    
