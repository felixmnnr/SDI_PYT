# knn_cython.pyx

import numpy as np
cimport numpy as np
from bottleneck import argpartition

# Déclarations de types pour les arguments des fonctions
ctypedef np.float64_t dtype_t

def k_nearest_neighbors(x_train, class_train, x_test, n_neighbors):
    # Compute distances for each row in x_test with respect to x_train
    predicted_classes = []
    for i in range(len(x_test)):
        x = x_test[i]
        distances = [round(np.linalg.norm(x - y), 5) for y in x_train]
        sorted_indices = argpartition(distances, n_neighbors, axis=0)

        # Take only the indices of labels of the n_neighbors nearest neighbors
        k_nearest_labels = [class_train[i] for i in sorted_indices[:n_neighbors]]

        # Use np.bincount to find the mode (most common label) for each row in x_test
        class_pred = np.argmax(np.bincount(k_nearest_labels))
        predicted_classes.append(class_pred)
    return predicted_classes