import numpy as np
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import imageio
import cv2
import time
import os


def custom_kernel(image, gamma_s, gamma_c):
    h, w, c = image.shape  # (100, 100, 3)
    n_points = h * w
    S = np.zeros((n_points, 2))  # (10000, 2)
    C = np.zeros((n_points, c))  # (10000, 3)

    for idx in range(n_points):
        S[idx] = [idx // w, idx % w]
        C[idx] = image[idx // w, idx % w]

    rbf_s = squareform(np.exp(-gamma_s * pdist(S, 'sqeuclidean')))  # (10000, 10000)
    rbf_c = squareform(np.exp(-gamma_c * pdist(C, 'sqeuclidean')))  # (10000, 10000)
    K = rbf_s * rbf_c  # (10000, 10000)

    return K


def eigen_decomposition(L, n_clusters):
    eigenvalues, eigenvectors = np.linalg.eigh(L)  # eigenvalues will be sorted automatically
    return eigenvectors[:, 1:1+n_clusters]  # exclude first eigenvector


def initialization(spectrum, init_method):
    n_points, n_clusters = spectrum.shape

    if init_method == 'random':
        np.random.seed(42)
        centroids = spectrum[np.random.choice(n_points, size=n_clusters, replace=False)]

    elif init_method == 'kmeans++':
        np.random.seed(42)
        centroids = [spectrum[np.random.choice(n_points)]]

        for _ in range(n_clusters - 1):
            min_distances = np.full(n_points, np.inf)  # D(x)^2: record the minimum distance(square) to the existing centers

            for c in centroids:
                distances_to_c = np.linalg.norm(spectrum - c, axis=1)**2
                min_distances = np.minimum(min_distances, distances_to_c)

            probabilities = min_distances / np.sum(min_distances)
            # np.random.seed(42)
            next_centroid = np.random.choice(n_points, p=probabilities)
            centroids.append(spectrum[next_centroid])

        centroids = np.array(centroids)

    else:
        raise ValueError("Unsupported initialization method. Choose 'random' or 'kmeans++'.")
    
    return centroids


def kmeans(spectrum, init_method, image_shape, max_iter=100, tol=10):
    n_points, n_clusters = spectrum.shape
    centroids = initialization(spectrum, init_method)
    clusters = np.zeros(n_points)
    gif_frames = []

    colormap = plt.get_cmap("tab10", n_clusters)  # get a colormap with n_clusters colors, at most 10 colors
    colormap = (colormap(np.arange(n_clusters))[:, :3] * 255).astype(np.uint8)  # convert to RGB format

    for iter in range(max_iter):
        print(f"Iteration: {iter + 1}")
        prev_clusters = clusters.copy()
        distance_to_centroids = np.zeros((n_points, n_clusters))

        for i in range(n_points):
            for j in range(n_clusters):
                distance_to_centroids[i, j] = np.linalg.norm(spectrum[i] - centroids[j])  # Euclidean distance

        clusters = np.argmin(distance_to_centroids, axis=1)
        cluster_result = np.zeros((n_points, 3), dtype=np.uint8)
        for i in range(n_points):
            cluster_result[i] = colormap[clusters[i]]

        cluster_result = cluster_result.reshape(image_shape)
        gif_frames.append(cluster_result)

        clusters_diff = np.sum(clusters != prev_clusters)
        print(f"Cluster changes: {clusters_diff}")

        if clusters_diff < tol:
            print("Converged!")
            break

        else:
            centroids = np.array([spectrum[clusters == i].mean(axis=0) for i in range(n_clusters)])

    return clusters, gif_frames


def eigenspace_visualization(spectrum, init_method, clusters, output_dir, file_name_prefix):
    n_clusters = spectrum.shape[1]
    os.makedirs(output_dir, exist_ok=True)

    if n_clusters == 2:
        plt.figure(figsize=(8, 6))

        for cluster in range(n_clusters):
            cluster_points = spectrum[clusters == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")
        
        plt.title("Clustering in Eigenspace (2D)")
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(output_dir, f"{file_name_prefix}_{init_method}_2D.png")
        plt.savefig(output_path)
        print(f"2D eigenspace clustering visualization saved to: {output_path}")
        plt.close()

    elif n_clusters == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for cluster in range(n_clusters):
            cluster_points = spectrum[clusters == cluster]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f"Cluster {cluster}")
        
        ax.set_title("Clustering in Eigenspace (3D)")
        ax.legend()

        output_path = os.path.join(output_dir, f"{file_name_prefix}_{init_method}_3D.png")
        plt.savefig(output_path)
        print(f"3D eigenspace clustering visualization saved to: {output_path}")
        plt.close()
    
    else:
        print("Eigenspace visualization is only supported for k=2 or k=3 dimensions.")


def save_gif(output_path, frames):
    imageio.mimsave(output_path, frames, format='GIF', duration=0.5)
    print(f".gif saved at {output_path}")


def save_last_frame_as_png(output_path, frame):
    cv2.imwrite(output_path, frame)
    print(f".png saved at {output_path}")


def main(n_clusters, init_method, spectral_method, gamma_s, gamma_c, max_iter, tolerance, image_path):
    file_name_prefix = image_path.split(".")[0]
    image = cv2.imread(image_path)
    start_time = time.time()

    print("Calculating custom kernel...")
    W = custom_kernel(image, gamma_s, gamma_c)
    D = np.diag(np.sum(W, axis=1))  # Degree matrix
    L = D - W  # Laplacian matrix

    if spectral_method == 'ratio':
        print("Performing spectral clustering using the ratio cut method...")
        U = eigen_decomposition(L, n_clusters)
        clusters, gif_frames = kmeans(U, init_method, image.shape, max_iter, tolerance)

        if n_clusters in [2, 3]:
            output_eigen_dir = "./spectral_clustering/ratio_cut/eigenspace"
            eigenspace_visualization(U, init_method, clusters, output_eigen_dir, file_name_prefix)
    
    elif spectral_method == 'normalized':
        print("Performing spectral clustering using the normalized cut method...")  
        D_inv_sqrt = np.diag(1.0 / np.diag(np.sqrt(D)))
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt
        U = eigen_decomposition(L_sym, n_clusters)
        T = U / np.linalg.norm(U, axis=1, keepdims=True)  # normalization
        clusters, gif_frames = kmeans(T, init_method, image.shape, max_iter, tolerance)
        
        if n_clusters in [2, 3]:
            output_eigen_dir = "./spectral_clustering/normalized_cut/eigenspace"
            eigenspace_visualization(T, init_method, clusters, output_eigen_dir, file_name_prefix)
    
    else:
        raise ValueError("Unsupported spectral clustering method. Choose 'ratio' or 'normalized'.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    output_gif_dir = f"./spectral_clustering/{spectral_method}_cut/gif"
    output_png_dir = f"./spectral_clustering/{spectral_method}_cut/png"
    os.makedirs(output_gif_dir, exist_ok=True)
    os.makedirs(output_png_dir, exist_ok=True)

    gif_output_path = os.path.join(output_gif_dir, f"{file_name_prefix}_{init_method}_{n_clusters}clusters.gif")
    save_gif(gif_output_path, gif_frames)

    png_output_path = os.path.join(output_png_dir, f"{file_name_prefix}_{init_method}_{n_clusters}clusters.png")
    save_last_frame_as_png(png_output_path, gif_frames[-1])


if __name__ == "__main__":
    gamma_s, gamma_c = 0.001, 0.001
    max_iter = 100
    tolerance = 10
    mode = '0'  # 0: single test, 1: test all parameter combinations

    if mode == '0':
        n_clusters = 3
        # init_method = 'random'
        init_method = "kmeans++"
        spectral_method = "normalized"
        # spectral_method = "ratio"
        # image_path = "image1.png"
        image_path = "image2.png"
        main(n_clusters, init_method, spectral_method, gamma_s, gamma_c, max_iter, tolerance, image_path)

    else:
        n_clusters_list = [2, 3, 4]
        init_method_list = ["random", "kmeans++"]
        spectral_method_list = ["normalized", "ratio"]
        image_path_list = ["image1.png", "image2.png"]

        for spectral_method in spectral_method_list:
            for image_path in image_path_list:
                for init_method in init_method_list:
                    for n_clusters in n_clusters_list:
                        print(f"\nTesting with n_clusters={n_clusters}, init_method={init_method}, "
                              f"spectral_method={spectral_method}, image_path={image_path}")

                        main(n_clusters, init_method, spectral_method, gamma_s, gamma_c, max_iter, tolerance, image_path)
    