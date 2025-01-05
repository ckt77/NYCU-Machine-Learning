from PIL import Image
import os
import numpy as np
from scipy.spatial.distance import cdist


def read_pgm(path, resized_factor):
    image_files = sorted(os.listdir(path))
    images = []
    labels = []

    for image_file in image_files:
        with Image.open(os.path.join(path, image_file)) as image:
            width, height = image.size
            width_resized, height_resized = width // resized_factor, height // resized_factor
            image_resized = image.resize((width_resized, height_resized))
            image = np.array(image_resized).flatten() / 255.0  # normalization for kernel calculation

        images.append(image)
        labels.append(int(image_file.split('.')[0][7:9]) - 1)

    return np.array(images), np.array(labels)


def compute_kernel(X, X_new, kernel_type):
    if kernel_type == 'linear':
        K = np.dot(X_new, X.T)

    elif kernel_type == 'polynomial':
        gamma = 5e-3
        coefficient = 1
        degree = 2
        K = (gamma * np.dot(X_new, X.T) + coefficient) ** degree

    elif kernel_type == 'rbf':
        gamma = 1e-3
        distances = cdist(X_new, X, 'sqeuclidean')
        K = np.exp(-gamma * distances)

    return K


def kernel_lda(X, y, n_components, kernel_type):
    N = X.shape[0]  # number of samples
    one_N = np.ones((N, N)) / N
    K = compute_kernel(X, X, kernel_type)  # shape: (N, N), where N is the number of samples
    K_centered = K - one_N @ K - K @ one_N + one_N @ K @ one_N
    unique_classes = np.unique(y)

    # compute SWϕ (within-class scatter matrix)
    S_W_phi = np.zeros((N, N))

    for c in unique_classes:
        indices_c = np.where(y == c)[0]
        N_c = len(indices_c)

        # term1:
        term1 = np.zeros((N, N))
        for i in indices_c:
            term1[i, i] = K[i, i]

        # term2:
        term2 = np.zeros((N, N))
        for i in indices_c:
            for j in indices_c:
                term2[i, j] += K[i, j] / N_c

        # term3
        term3 = np.zeros((N, N))
        for j in indices_c:
            for k in indices_c:
                term3[j, k] += K[j, k] / N_c

        S_W_phi += (term1 - 2 * term2 + term3)
    # print("S_W_phi", S_W_phi)

    # compute SBϕ (between-class scatter matrix)
    S_B_phi = np.zeros((N, N)) 

    for c in unique_classes:
        indices_c = np.where(y == c)[0]
        N_c = len(indices_c)

        # term1
        term1 = np.zeros((N, N))
        for i in indices_c:
            for j in indices_c:
                term1[i, j] = K[i, j] / (N_c ** 2)

        # term2
        term2 = np.zeros((N, N))
        for i in indices_c:
            for j in range(N):
                term2[i, j] = K[i, j] / (N_c * N)

        # term3
        term3 = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                term3[i, j] = K[i, j] / (N * N)

        S_B_phi += N_c * (term1 - 2 * term2 + term3)
    # print("S_B_phi", S_B_phi)

    S_W_phi += np.eye(N) * 1e-6  # pseudo inverse

    eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(S_W_phi) @ S_B_phi)
    sort_index = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, sort_index]

    principal_components = eigenvectors[:, :n_components]
    principal_components = principal_components / np.linalg.norm(principal_components, axis=0)

    return principal_components, K_centered


def project_test(X_train, X_test, principal_components, kernel_type):
    K_test = compute_kernel(X_train, X_test, kernel_type=kernel_type)
    K_train = compute_kernel(X_train, X_train, kernel_type=kernel_type)

    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]
    one_n_test = np.ones((n_samples_test, n_samples_train)) / n_samples_train
    one_n_train = np.ones((n_samples_train, n_samples_train)) / n_samples_train
    
    K_test_centered = (K_test - 
                       np.dot(one_n_test, K_train) - 
                       np.dot(K_test, one_n_train) + 
                       np.dot(one_n_test, np.dot(K_train, one_n_train)))
    
    X_test_projected = np.dot(K_test_centered, principal_components)
    
    return X_test_projected


def performance(train_Z, train_labels, test_Z, test_labels, k=5):
    predictions = np.zeros(test_Z.shape[0])

    for i in range(test_Z.shape[0]):
        distance = np.sum(np.square(test_Z[i] - train_Z), axis=1)
        sort_index = np.argsort(distance)
        nearest_neighbors = train_labels[sort_index[:k]]  # k nearest neighbors
        labels, counts = np.unique(nearest_neighbors, return_counts=True)
        predictions[i] = labels[np.argmax(counts)]

    accuracy = np.sum((test_labels - predictions) == 0) / len(test_labels)

    return accuracy


if __name__ == '__main__':
    train_filepath = os.path.join('./Yale_Face_Database', 'Training')
    train_images, train_labels = read_pgm(train_filepath, 3)
    test_filepath = os.path.join('./Yale_Face_Database', 'Testing')
    test_images, test_labels = read_pgm(test_filepath, 3)

    n_components = 25
    kernel_types = ['linear', 'polynomial', 'rbf']
    
    for kernel_type in kernel_types:
        train_W, K_train_centered = kernel_lda(train_images, train_labels, n_components, kernel_type)
        train_z = np.dot(K_train_centered, train_W)
        test_z = project_test(train_images, test_images, train_W, kernel_type)

        accuracy = performance(train_z, train_labels, test_z, test_labels, k=1)
        print(f'Kernel LDA accuracy ({kernel_type}): {accuracy * 100:.2f}%')
