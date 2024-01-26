from cluster_physics import ClusterPhysics

class Cluster:
    def __init__(self, params, number_of_molecules, number_of_clusters):
        # Inicializar un objeto ClusterPhysics
        self.physics = ClusterPhysics(params)
        
        # Atributos de la clase Cluster
        self.number_of_molecules = number_of_molecules
        self.number_of_clusters = number_of_clusters

    def get_number_of_molecules(self):
        return self.number_of_molecules

    def set_number_of_molecules(self, number_of_molecules):
        self.number_of_molecules = number_of_molecules

    def get_number_of_clusters(self):
        return self.number_of_clusters

    def set_number_of_clusters(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters

    def forward_rate(self):
        return self.physics.attachment_rate_melting(self.number_of_molecules) * self.number_of_clusters

    def backward_rate(self):
        return self.physics.detachment_rate_melting(self.number_of_molecules) * self.number_of_clusters
