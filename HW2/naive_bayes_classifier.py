import numpy as np
import struct  # to read binary files
from tqdm import tqdm


def read_images(filename):
    with open(filename, 'rb') as f:
        magic_number, num_images, rows, columns = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, columns)

    return images


def read_labels(filename):
    with open(filename, 'rb') as f:
        magic_number, num_labels = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    return labels


def train_naive_bayes(images, labels, mode='discrete'):
    num_classes = 10
    num_samples, num_rows, num_columns = images.shape
    class_counts = np.zeros(num_classes)

    if mode == 'discrete':
        cond_prob = np.zeros((num_classes, num_rows, num_columns, 32))  # conditional probability

        for i in tqdm(range(num_samples)):
            class_counts[labels[i]] += 1
            image = images[i]

            for row in range(num_rows):
                for column in range(num_columns):
                    bin_index = image[row, column] // 8  # Tally the frequency of the values of each pixel into 32 bins.
                    cond_prob[labels[i], row, column, bin_index] += 1

        class_prior = class_counts / num_samples
        cond_prob = (cond_prob + 1) / (class_counts[:, None, None, None] + 32)  # Laplace smoothing

        return class_prior, cond_prob

    elif mode == 'continuous':
        mean = np.zeros((num_classes, num_rows, num_columns), dtype=np.float32)
        variance = np.zeros((num_classes, num_rows, num_columns), dtype=np.float32)
        log_cond_prob = np.zeros((num_classes, num_rows, num_columns, 256))

        for i in range(num_samples):
            class_counts[labels[i]] += 1
            mean[labels[i]] += images[i]

        for i in range(num_classes):
            mean[i] /= class_counts[i]

        for i in range(num_samples):
            variance[labels[i]] += (images[i] - mean[labels[i]])**2

        for i in range(num_classes):
            variance[i] /= class_counts[i]

        class_prior = class_counts / num_samples

        for i in tqdm(range(num_classes)):
            for row in range(num_rows):
                for column in range(num_columns):
                    for j in range(256):
                        if variance[i, row, column] == 0:  # To prevent division by zero
                            variance[i, row, column] = 1e3

                        # Calculate the log of the conditional probability of each pixel value given the class.
                        log_cond_prob[i, row, column, j] = (-(j - mean[i, row, column])**2) / (2 * variance[i, row, column]) - np.log(np.sqrt(variance[i, row, column])) - 0.5 * np.log(2 * np.pi)

        return class_prior, log_cond_prob


def calculate_posterior(test_image, class_prior, cond_prob, mode):
    log_posteriors = np.log(class_prior)

    for label in range(10):
        for row in range(test_image.shape[0]):
            for column in range(test_image.shape[1]):
                if mode == 'discrete':
                    pixel_value = test_image[row, column] // 8
                    log_posteriors[label] += np.log(cond_prob[label, row, column, pixel_value])

                elif mode == 'continuous':
                    pixel_value = test_image[row, column]
                    log_posteriors[label] += cond_prob[label, row, column, pixel_value]

    log_posteriors_normalized = log_posteriors / np.sum(log_posteriors)  # normalization

    return log_posteriors_normalized


def print_digit_imagination(cond_prob, mode='discrete'):
    print("Imagination of numbers in Bayesian classifier:")

    for label in range(10):
        print(f"{label}:")

        if mode == 'discrete':
            for row in range(28):
                row = ''.join(['1 ' if np.argmax(cond_prob[label, row, column]) >= 16 else '0 ' for column in range(28)])
                print(row)

        elif mode == 'continuous':
            for row in range(28):
                row = ''.join(['1 ' if np.argmax(cond_prob[label, row, column]) >= 128 else '0 ' for column in range(28)])
                print(row)

        print()


def main():
    train_images_path = 'train-images.idx3-ubyte_'
    train_labels_path = 'train-labels.idx1-ubyte_'
    test_images_path = 't10k-images.idx3-ubyte_'
    test_labels_path = 't10k-labels.idx1-ubyte_'

    train_images = read_images(train_images_path)
    train_labels = read_labels(train_labels_path)
    test_images = read_images(test_images_path)
    test_labels = read_labels(test_labels_path)

    toggle = input('Toggle option (0: discrete mode / 1: continuous mode): ')
    if toggle == '0':
        mode = 'discrete'
    elif toggle == '1':
        mode = 'continuous'
    else:
        print('Invalid input. Please try again.')
        return

    class_prior, cond_prob = train_naive_bayes(train_images, train_labels, mode=mode)
    error = 0
    total = len(test_labels)

    for i in range(total):
        log_posteriors_normalized = calculate_posterior(test_images[i], class_prior, cond_prob, mode=mode)

        print("Posterior (in log scale):")
        for j in range(10):
            print(f"{j}: {log_posteriors_normalized[j]}")

        predicted_label = np.argmin(log_posteriors_normalized)
        print(f"Prediction: {predicted_label}, Ans: {test_labels[i]}\n")

        if predicted_label != test_labels[i]:
            error += 1

    print_digit_imagination(cond_prob, mode=mode)
    error_rate = error / total
    print(f"Error rate: {error_rate}")


if __name__ == "__main__":
    main()
