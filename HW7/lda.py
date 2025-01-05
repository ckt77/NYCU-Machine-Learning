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


def lda(X, y, n_components):
    n_classes = len(np.unique(y))  # 15 classes
    # max_components = n_classes - 1

    # if n_components > max_components:
    #     print(f"Warning: n_components reduced from {n_components} to {max_components}")
    #     n_components = max_components

    n_samples, n_features = X.shape
    overall_mean = np.mean(X, axis=0, keepdims=True)  # shape: (1, 5005)
    class_means = np.zeros((n_classes, n_features))
    
    S_W = np.zeros((n_features, n_features))  # shape: (5005, 5005)
    S_B = np.zeros((n_features, n_features))  # shape: (5005, 5005)
    
    for i in range(n_classes):
        x_i = X[y == i]  # samples of class i, shape: (9, 5005)
        class_means[i] = np.mean(x_i, axis=0, keepdims=True)
        within_class_diff = x_i - class_means[i]  # shape: (9, 5005)
        S_W += within_class_diff.T @ within_class_diff

        between_class_diff = (class_means[i] - overall_mean)  # shape: (1, 5005)
        S_B += len(x_i) * (between_class_diff.T @ between_class_diff)
    
    S_W += np.eye(n_features) * 1e-6  # pseudo inverse
    
    eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(S_W) @ S_B)
    sort_index = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, sort_index]

    projection_matrix = eigenvectors[:, :n_components]
    projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=0)
    
    return projection_matrix, overall_mean


def show_fisherface(fisherfaces, height, width, n=25, output_dir='fisherfaces'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(10, 10))
    for i in range(min(n, fisherfaces.shape[1])):
        plt.subplot(5, 5, i + 1)
        image = fisherfaces[:, i].reshape(height, width)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
        plt.imsave(f"{output_dir}/fisherface_{i + 1}.png", image, cmap='gray')
    
    plt.tight_layout()
    plt.show()


def show_reconstruction(images, images_reconstructed, height, width, file_names_prefix, n=10, output_dir='lda_reconstructed_images'):
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
        distance = np.sum(np.square(test_Z[i] - train_Z), axis=1)
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
    train_W, train_X_mean = lda(train_images, train_labels, n_components)
    show_fisherface(train_W, height, width)
    
    train_Z = (train_images - train_X_mean) @ train_W
    train_images_reconstructed = train_Z @ train_W.T + train_X_mean
    show_reconstruction(train_images, train_images_reconstructed, height, width, train_file_names)
    
    test_filepath = os.path.join('./Yale_Face_Database', 'Testing')
    test_images, test_labels, _, _, _ = read_pgm(test_filepath, 3)
    
    test_Z = (test_images - train_X_mean) @ train_W
    accuracy = performance(train_Z, train_labels, test_Z, test_labels, k=5)
    print('LDA accuracy: {:.2f}%'.format(accuracy * 100))
