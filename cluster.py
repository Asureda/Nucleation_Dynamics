import cluster_physics as phys

class Cluster:
    def __init__(self, temperature, number_of_molecules, number_of_clusters):
        self.temperature = temperature
        self.number_of_molecules = number_of_molecules
        self.number_of_clusters = number_of_clusters
        self.sigma = phys.SIGMA
        self.delta_s_f = phys.DELTA_S
        self.t_m = phys.T_M

    def get_number_of_molecules(self):
        return self.number_of_molecules

    def set_number_of_molecules(self, number_of_molecules):
        self.number_of_molecules = number_of_molecules

    def get_number_of_clusters(self):
        return self.number_of_clusters

    def set_number_of_clusters(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters

    def forward_rate(self):
        return phys.forward_rate_constant(self.number_of_molecules, self.temperature, self.sigma, self.delta_s_f, self.t_m) * self.number_of_clusters

    def backward_rate(self):
        return phys.backward_rate_constant(self.number_of_molecules, self.temperature, self.sigma, self.delta_s_f, self.t_m) * self.number_of_clusters
