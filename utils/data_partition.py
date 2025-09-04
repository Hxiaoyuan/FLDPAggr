import numpy as np
def create_dirichlet_distribution(alpha: float, num_client: int, num_class: int, seed: int):
    random_number_generator = np.random.default_rng(seed)
    distribution = random_number_generator.dirichlet(np.repeat(alpha, num_client), size=num_class).transpose()
    distribution /= distribution.sum()
    return distribution

def split_by_distribution(targets, distribution):
    num_client, num_class = distribution.shape[0], distribution.shape[1]
    sample_number = np.floor(distribution * len(targets))
    class_idx = {class_label: np.where(targets == class_label)[0] for class_label in range(num_class)}

    idx_start = np.zeros((num_client + 1, num_class), dtype=np.int32)
    for i in range(0, num_client):
        idx_start[i + 1] = idx_start[i] + sample_number[i]

    client_samples = {idx: {} for idx in range(num_client)}
    for client_idx in range(num_client):
        samples_idx = np.array([], dtype=np.int32)
        for class_label in range(num_class):
            start, end = idx_start[client_idx, class_label], idx_start[client_idx + 1, class_label]
            samples_idx = (np.concatenate((samples_idx, class_idx[class_label][start:end].tolist())).astype(np.int32))
        client_samples[client_idx] = samples_idx

    return client_samples