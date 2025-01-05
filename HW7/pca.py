import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def read_pgm(path, resized_factor):
    image_files = sorted(os.listdir(path))
    images = []
    labels = []
    file_names = []

    for image_file in image_files:
        with Image.open(os.path.join(path, image_file)) as image:
            width, height = image.size
            width_resized, height_resized = width // resized_factor, height // resized_factor
            image_resized = image.resize((width_resized, height_resized))
            image = np.array(image_resized).flatten()

        images.append(image)
        labels.append(int(image_file.split('.')[0][7:9]) - 1)
        file_name_prefix, _ = os.path.splitext(image_file)
        file_names.append(file_name_prefix)

    return np.array(images), np.array(labels), file_names, width_resized, height_resized


def pca(X, n_components):
    X_mean = np.mean(X, axis=0, keepdims=True)  # shape: (1, 5005)
    X_centered = X - X_mean  # shape: (135, 5005)

    S = np.cov(X_centered, rowvar=True, bias=False)  # covariance matrix, shape: (135, 135)
    eigenvalues, eigenvectors = np.linalg.eigh(S)  # eigenvalues are in ascending order
    sort_index = np.argsort(-eigenvalues)  # sort eigenvalues in descending order
    eigenvectors = eigenvectors[:, sort_index]  # shape: (135, 135)

    # calculate eigenvectors of X.T @ X to get the projection matrix W
    projection_matrix = X_centered.T @ eigenvectors  # shape: (5005, 135)
    projection_matrix = projection_matrix[:, :n_components]  # shape: (5005, n_components)
    projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=0)  # normalization

    return projection_matrix, X_mean  # X_mean would be used to reconstruct the image


def show_eigenface(eigenfaces, height, weight, output_dir='eigenfaces'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 10))
    
    for i in range(min(25, eigenfaces.shape[1])):
        plt.subplot(5, 5, i + 1)
        image = eigenfaces[:, i].reshape(height, weight)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
        plt.imsave(f"{output_dir}/eigenface_{i + 1}.png", image, cmap='gray')
    
    plt.tight_layout()
    plt.show()


def show_reconstruction(images, images_reconstructed, height, width, file_names_prefix, n=10, output_dir='pca_reconstructed_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.random.seed(42)
    random_numbers = np.random.choice(images.shape[0], n, replace=False)

    plt.figure(figsize=(15, 6))

    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(images[random_numbers[i]].reshape(height, width), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(images_reconstructed[random_numbers[i]].reshape(height, width), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

        image_filename = f"{output_dir}/{file_names_prefix[random_numbers[i]]}.png"
        plt.imsave(image_filename, images_reconstructed[random_numbers[i]].reshape(height, width), cmap='gray')
        
    plt.tight_layout()
    plt.show()


def performance(train_Z, train_labels, test_Z, test_labels, k=5):
    predictions = np.zeros(test_Z.shape[0])

    for i in range(test_Z.shape[0]):
        distance = np.sum(np.square(test_Z[i] - train_Z), axis=1)  # shape: (135,)
        sort_index = np.argsort(distance)
        nearest_neighbors = train_labels[sort_index[:k]]  # k nearest neighbors
        labels, counts = np.unique(nearest_neighbors, return_counts=True)
        predictions[i] = labels[np.argmax(counts)]

    accuracy = np.mean(test_labels == predictions)

    return accuracy


if __name__ == '__main__':
    train_filepath = os.path.join('./Yale_Face_Database', 'Training')
    train_images, train_labels, train_file_names, width, height = read_pgm(train_filepath, 3)
    
    n_components = 25
    train_W, train_X_mean = pca(train_images, n_components)  # shape: (5005, n_components), (1, 5005)
    show_eigenface(train_W, height, width)

    train_Z = (train_images - train_X_mean) @ train_W  # Z = X_centered @ W, shape: (135, n_components)
    train_images_reconstructed = train_Z @ train_W.T + train_X_mean
    show_reconstruction(train_images, train_images_reconstructed, height, width, train_file_names)

    test_filepath = os.path.join('./Yale_Face_Database', 'Testing')
    test_images, test_labels, _, _, _ = read_pgm(test_filepath, 3)
    
    test_Z = (test_images - train_X_mean) @ train_W  # shape: (30, n_components)
    accuracy = performance(train_Z, train_labels, test_Z, test_labels, k=5)
    print('PCA accuracy: {:.2f}%'.format(accuracy * 100))
