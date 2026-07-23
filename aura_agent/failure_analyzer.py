import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

class FailureAnalyzer:
    """
    Discovers failure modes by clustering prediction errors and evaluates
    alignment against ground-truth simulator event tags.
    """
    def __init__(self, n_clusters: int = 6):
        self.n_clusters = n_clusters
        self.failures = []
        
    def add_failure(self, error_vector: np.ndarray, tags: np.ndarray):
        """
        error_vector: (latent_dim,) or similar high-dimensional error signal
        tags: (4,) bool array from sim.state_summary() [near_human, near_forklift, near_door, near_box]
        """
        self.failures.append({
            'error': error_vector,
            'tags': tags
        })
        
    def analyze(self):
        """
        Clusters the prediction errors and computes Normalized Mutual Information (NMI)
        against the ground truth tags to measure how well the agent naturally discovers
        the true failure modes (e.g. human vs door).
        """
        if len(self.failures) < self.n_clusters:
            print("Not enough failures to analyze.")
            return 0.0
            
        errors = np.stack([f['error'] for f in self.failures])
        tags = np.stack([f['tags'] for f in self.failures])
        
        # Convert multi-label tags to a single categorical label for NMI evaluation
        # (e.g., 0=human, 1=forklift, 2=door, 3=box, 4=multiple, 5=none)
        def tags_to_label(tag_row):
            active = np.where(tag_row)[0]
            if len(active) == 1:
                return active[0]
            elif len(active) > 1:
                return 4 # multiple
            return 5 # none
            
        true_labels = np.array([tags_to_label(t) for t in tags])
        
        # Cluster the errors
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        discovered_clusters = kmeans.fit_predict(errors)
        
        # Compute NMI
        nmi = normalized_mutual_info_score(true_labels, discovered_clusters)
        print(f"Failure Analysis Complete. NMI with ground-truth: {nmi:.4f}")
        
        return nmi
