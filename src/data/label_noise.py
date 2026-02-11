import numpy as np

def inject_label_noise(labels, noise_type='symmetric', noise_rate=0.2):
    """
    Args:
        labels: Numpy array of true labels (N,)
        noise_type: 'symmetric' (random) or 'asymmetric' (semantic)
        noise_rate: Percentage of labels to flip (0.0 to 1.0)
    Returns:
        new_labels: The corrupted label array
        noisy_indices: The specific indices that were flipped
    """
    n_samples = len(labels)
    n_noisy = int(n_samples * noise_rate)
    new_labels = labels.copy()
    noisy_indices = []

    if n_noisy == 0:
        return new_labels, np.array([])

    # Randomly choose victims
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    if noise_type == 'symmetric':
        # Flip to any random class (0-9)
        for idx in noisy_indices:
            true_label = labels[idx]
            possible_labels = list(range(10))
            possible_labels.remove(true_label)
            new_labels[idx] = np.random.choice(possible_labels)
            
    elif noise_type == 'asymmetric':
        # CIFAR-10 Mapping (Semantic Flips)
        mapping = {
            9: 1, # Truck -> Auto
            2: 0, # Bird -> Plane
            4: 7, # Deer -> Horse
            3: 5, # Cat -> Dog
            5: 3  # Dog -> Cat
        }
        actual_noisy_indices = []
        for idx in noisy_indices:
            if labels[idx] in mapping:
                new_labels[idx] = mapping[labels[idx]]
                actual_noisy_indices.append(idx)
        noisy_indices = np.array(actual_noisy_indices)

    return new_labels, noisy_indices