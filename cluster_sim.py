import cluster
from cluster_physics import ClusterPhysics


class ClusterSimulation:
    def __init__(self, params, time_steps, dt, u, MAX_NUMBER_MOLECULES):
        self.cluster_dict = {}
        self.physics_object = ClusterPhysics(params)
        self.temperature = self.physics_object.temperature.magnitude
        self.time_steps = time_steps
        self.u = u  # Umbral para tratamiento numérico
        self.i_max = MAX_NUMBER_MOLECULES  # Límite superior para cierre del sistema
        self.dt = dt
        # Inicializar clusters
        for nof_molecules in range(1, MAX_NUMBER_MOLECULES + 1):
            start = self.physics_object.number_density_equilibrium(nof_molecules).magnitude if nof_molecules < u else 0
            new_cluster = cluster.Cluster(params, nof_molecules, start)
            self.cluster_dict[nof_molecules] = new_cluster

    def change_in_clusters_at_number(self, number_molecules):
        current_cluster = self.cluster_dict[number_molecules]
        first_cluster = self.cluster_dict[1]

        if(number_molecules == 1):
            return 0
        if number_molecules>1: 
            prev_cluster = self.cluster_dict[number_molecules - 1]
        if(number_molecules < self.i_max):
            next_cluster = self.cluster_dict[number_molecules + 1]
        else:
            next_cluster = current_cluster
        return -current_cluster.forward_rate()-current_cluster.backward_rate()+prev_cluster.forward_rate()+next_cluster.backward_rate()

    def update_cluster_at_number(self, number_molecules):
        new_number_clusters = self.cluster_dict[number_molecules].get_number_of_clusters() + self.dt * self.change_in_clusters_at_number(number_molecules)
        self.cluster_dict[number_molecules].set_number_of_clusters(new_number_clusters)

    def update_all_clusters(self):
        for number_of_molecules in self.cluster_dict.keys():
            self.update_cluster_at_number(number_of_molecules)                
    
    def simulate(self):
        accumulated_time = 0.0
        for time in range(self.time_steps):
            self.update_all_clusters()
            accumulated_time += self.dt
            if time % 50 ==0:
                print("time", time, "accumulated_time", accumulated_time)
                print("n cluster with 1 molecule", self.cluster_dict[1].get_number_of_clusters())
                print("n cluster with 2 molecules", self.cluster_dict[2].get_number_of_clusters())
                print("n cluster with 23 molecules", self.cluster_dict[23].get_number_of_clusters())
                print("n cluster with 40 molecules", self.cluster_dict[40].get_number_of_clusters())
        print('accumulated_time', accumulated_time)

    def pp(self):
        for key in self.cluster_dict.keys():
            print( key, self.cluster_dict[key].get_number_of_clusters())

    def number_molecules_as_array(self):
        return self.cluster_dict.keys()

    def number_clusters_as_array(self):
        return [ self.cluster_dict[key].get_number_of_clusters() for key in self.cluster_dict.keys() ]
    
