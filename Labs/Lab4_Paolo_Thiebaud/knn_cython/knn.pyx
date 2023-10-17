# knn_cython.pyx

import numpy as np
cimport numpy as np

# Déclarations de types pour les arguments des fonctions
ctypedef np.float64_t dtype_t

def kNN(np.ndarray[dtype_t, ndim=2] x_train, np.ndarray[dtype_t, ndim=2] x_test, int n_neighbours, np.ndarray[dtype_t, ndim=1] class_train):
    cdef int n_train = x_train.shape[0]
    cdef int n_test = x_test.shape[0]
    cdef int d, i, j

    # 1. Calcul de la matrice des distances
    cdef np.ndarray[dtype_t, ndim=2] mat_distance = np.zeros((n_test, n_train), dtype=np.float64)
    for i in range(n_test):
        for j in range(n_train):
            distance = 0.0
            for d in range(x_test.shape[1]):
                diff = x_test[i, d] - x_train[j, d]
                distance += diff * diff
            mat_distance[i, j] = distance ** 0.5  # Calcul de la norme euclidienne

    # 2. Tri des distances et sélection des indices des plus proches voisins
    cdef np.ndarray[np.int64_t, ndim=2] ordered_distance = np.argsort(mat_distance, axis=1)
    cdef np.ndarray[np.int64_t, ndim=2] id = ordered_distance[:, :n_neighbours]

    # 3. Calcul des labels des plus proches voisins
    cdef np.ndarray[np.int64_t, ndim=2] labels = np.zeros((n_test, n_neighbours), dtype=np.int64)
    for i in range(n_test):
        for j in range(n_neighbours):
            label_indice = id[i, j]
            labels[i, j] = class_train[label_indice]

    # 4. Prédiction des labels
    cdef np.ndarray[np.int64_t, ndim=1] class_pred = np.zeros(n_test, dtype=np.int64)
    for i in range(n_test):
        class_pred[i] = np.bincount(labels[i]).argmax()

    return class_pred
