import numpy as np
import struct
import matplotlib.pyplot as plt


def read_images(filename):
    with open(filename, 'rb') as f:
        magic_number, num_images, rows, columns = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows * columns)

    return images  # shape: (60000, 28*28)


def read_labels(filename):
    with open(filename, 'rb') as f:
        magic_number, num_labels = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    return labels


def print_digit_imagation(p, mapping, gt=False):  # gt: ground truth
    for i in range(10):
        if gt:
            print('labeled', end=" ")
        print('class {}:'.format(i))

        digit = mapping[i]

        for j in range(28):
            for k in range(28):
                print('1', end=" ") if p[digit][j * 28 + k] >= 0.5 else print('0', end=" ")  # p is the probability of pixel being 1
            print()
        print()


def assign_label(real_labels, w):
    table = np.zeros((10, 10))  # shape: (real_label, cluster)
    mapping = np.zeros(10, dtype='uint32')

    for i in range(len(real_labels)):
        table[real_labels[i], np.argmax(w[i])] += 1

    for _ in range(10):
        index = np.argmax(table)
        cluster = index % 10
        real_label = index // 10
        mapping[real_label] = cluster
        table[real_label, :] = 0
        table[:, cluster] = 0

    return mapping


def print_confusion_matrix(real_labels, w, mapping, iteration):
    confusion_matrix = np.zeros((10, 2, 2))
    correct = 0

    for idx in range(len(real_labels)):
        cluster = np.argmax(w[idx])
        prediction = np.where(mapping == cluster)[0]  # [0]: convert from an 1x1 array to a scalar
        real_label = real_labels[idx]

        for digit in range(10):
            if real_label != digit:
                if prediction == digit:
                    confusion_matrix[digit, 1, 0] += 1  # FP
                else:
                    confusion_matrix[digit, 1, 1] += 1  # TN
            else:
                if prediction == digit:
                    confusion_matrix[digit, 0, 0] += 1  # TP
                else:
                    confusion_matrix[digit, 0, 1] += 1  # FN

    for digit in range(10):
        print('------------------------------------------------------------------------------')
        print()
        print('Confusion Matrix {}'.format(digit))
        print("\t\t\tPredict number {}\tPredict not number {}".format(digit, digit))
        print("Is number {}\t\t    {}\t\t\t{}".format(digit, confusion_matrix[digit, 0, 0], confusion_matrix[digit, 0, 1]))
        print("Isn't number {}\t\t    {}\t\t\t{}".format(digit, confusion_matrix[digit, 1, 0], confusion_matrix[digit, 1, 1]))
        sensitivity = confusion_matrix[digit, 0, 0] / (confusion_matrix[digit, 0, 0] + confusion_matrix[digit, 0, 1])
        specificity = confusion_matrix[digit, 1, 1] / (confusion_matrix[digit, 1, 0] + confusion_matrix[digit, 1, 1])
        correct += confusion_matrix[digit, 0, 0]
        print(f"\nSensitivity (Successfully predict number {digit})\t: {sensitivity}")
        print(f"Specificity (Successfully predict not number {digit}) : {specificity}")
        print()

    print('Total iteration to converge: {}'.format(iteration))
    print('Total error rate: {}'.format(1 - correct / len(real_labels)))


def em_algorithm(images, labels):
    # initialization
    image_length, num_pixels = images.shape  # 60000, 28*28
    lamb = [0.1 for _ in range(10)]
    # strong prior (just like supervised)
    # p = np.zeros((10, num_pixels))
    # for i in range(image_length):
    #     p[labels[i]] += images[i]
    # p /= 6000
    p = np.random.rand(10, num_pixels) / 3  # shape: (10, 28*28), p range from 0 to 0.33
    p[p <= 1e-2] = 1e-2

    # make the center of the image more likely to be 1
    for i in range(10):
        center_start = 4
        center_end = center_start + 20

        for j in range(center_start, center_end):
            p[i, j * 28 + center_start: j * 28 + center_end] += 0.2

    p_prev = np.zeros((10, num_pixels))
    w = np.zeros((image_length, 10))  # shape: (60000, 10)
    iteration = 0
    difference = np.inf
    tolerence = 20

    while iteration < 20 and difference > tolerence:
        iteration += 1

        # E-step: calculate w
        for idx in range(image_length):
            w[idx] = lamb * np.prod(p ** images[idx], axis=1) * np.prod((1 - p) ** (1 - images[idx]), axis=1)
            w[idx] /= np.sum(w[idx])

        # M-step: update p and lamb
        w_sum = np.sum(w, axis=0)  # shape: (10,)
        lamb = w_sum / image_length
        p = np.dot(w.T, images) / (w_sum.reshape(10, 1))  # shape: (10, 28*28)
        # p[p == 0] = 1e-10

        mapping = list(range(10))
        print_digit_imagation(p, mapping, gt=False)
        difference = np.sum(abs(p - p_prev))
        p_prev = p
        print('No. of Iteration: {}, Difference: {}'.format(iteration, difference))
        print('------------------------------------------------------------------------------')
        print()
        print()

    print('------------------------------------------------------------------------------')
    print()
    mapping = assign_label(labels, w)
    print_digit_imagation(p, mapping, gt=True)
    print_confusion_matrix(labels, w, mapping, iteration)


def main():
    train_images_path = 'train-images.idx3-ubyte__'
    train_labels_path = 'train-labels.idx1-ubyte__'
    train_images = read_images(train_images_path)
    train_labels = read_labels(train_labels_path)
    train_images_binary = train_images // 128
    em_algorithm(train_images_binary, train_labels)


if __name__ == '__main__':
    main()
