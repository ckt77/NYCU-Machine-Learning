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


def initialization(K, n_clusters, init_method):
    n_points = K.shape[0]

    if init_method == 'random':
        np.random.seed(42)
        clusters = np.random.randint(low=0, high=n_clusters, size=n_points)

    elif init_method == 'kmeans++':
        np.random.seed(42)
        centers = [np.random.randint(low=0, high=n_points)]  # store the indices of the centers

        for _ in range(n_clusters - 1):
            min_distances = 1 - np.max(K[:, centers], axis=1)  # D(x): record the minimum distance to the existing centers
            min_distances = min_distances ** 2  # D(x)^2
            probabilities = min_distances / min_distances.sum()  # p(x) = D(x)^2 / sum(D(x)^2)
            # np.random.seed(42)
            next_center = np.random.choice(n_points, p=probabilities)
            centers.append(next_center)

        clusters = np.full(n_points, -1)
        for i in range(n_points):
            clusters[i] = np.argmax([K[i, centers[j]] for j in range(n_clusters)])

    else:
        raise ValueError("Unsupported initialization method. Choose 'random' or 'kmeans++'.")

    return clusters


def kernel_kmeans(K, n_clusters, init_method, image_shape, max_iter=100, tolerance=10):
    n_points = K.shape[0]
    clusters = initialization(K, n_clusters, init_method)
    gif_frames = []

    colormap = plt.get_cmap("tab10", n_clusters)  # get a colormap with n_clusters colors, at most 10 colors
    colormap = (colormap(np.arange(n_clusters))[:, :3] * 255).astype(np.uint8)  # convert to RGB format

    for iter in range(max_iter):
        print(f"Iteration: {iter + 1}")
        prev_clusters = clusters.copy()
        term_3 = np.zeros(n_clusters)

        for i in range(n_clusters):
            cluster_members = np.where(clusters == i)[0]

            if len(cluster_members) > 0:
                term_3[i] = np.sum(K[np.ix_(cluster_members, cluster_members)]) / (len(cluster_members) ** 2)

        for i in range(n_points):
            distance_to_clusters = np.zeros(n_clusters)

            for j in range(n_clusters):
                cluster_members = np.where(clusters == j)[0]

                if len(cluster_members) > 0:
                    distance_to_clusters[j] += (-2 * np.sum(K[i, cluster_members]) / len(cluster_members))
                    distance_to_clusters[j] += term_3[j]

                else:
                    distance_to_clusters[j] = np.inf

            clusters[i] = np.argmin(distance_to_clusters)

        cluster_result = np.zeros((n_points, 3), dtype=np.uint8)
        for i in range(n_points):
            cluster_result[i] = colormap[clusters[i]]

        cluster_result = cluster_result.reshape(image_shape)
        gif_frames.append(cluster_result)

        clusters_diff = np.sum(clusters != prev_clusters)
        print(f"Cluster changes: {clusters_diff}")

        if clusters_diff < tolerance:
            print("Converged!")
            break

    return clusters, gif_frames


def save_gif(output_path, frames):
    imageio.mimsave(output_path, frames, format='GIF', duration=0.5)
    print(f".gif saved at {output_path}")


def save_last_frame_as_png(output_path, frame):
    cv2.imwrite(output_path, frame)
    print(f".png saved at {output_path}")


def main(n_clusters, init_method, gamma_s, gamma_c, max_iter, tolerance, image_path):
    file_name_prefix = image_path.split(".")[0]
    image = cv2.imread(image_path)

    start_time = time.time()

    print("Calculating custom kernel...")
    K = custom_kernel(image, gamma_s, gamma_c)

    print("Performing kernel k-means clustering...")
    clusters, gif_frames = kernel_kmeans(K, n_clusters, init_method, image.shape, max_iter, tolerance)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    output_gif_dir = "./kernel_kmeans/gif"
    output_png_dir = "./kernel_kmeans/png"
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
    mode = '1'  # 0: single test, 1: test all parameter combinations

    if mode == '0':
        n_clusters = 2
        # init_method = "random"
        init_method = "kmeans++"
        image_path = "image1.png"
        # image_path = "image2.png"
        main(n_clusters, init_method, gamma_s, gamma_c, max_iter, tolerance, image_path)

    else:
        n_clusters_list = [2, 3, 4]
        init_method_list = ["random", "kmeans++"]
        image_path_list = ["image1.png", "image2.png"]

        for image_path in image_path_list:
            for init_method in init_method_list:
                for n_clusters in n_clusters_list:
                    print(f"\nTesting with n_clusters={n_clusters}, init_method={init_method}, image_path={image_path}")

                    main(n_clusters, init_method, gamma_s, gamma_c, max_iter, tolerance, image_path)
