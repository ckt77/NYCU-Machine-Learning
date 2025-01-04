import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)  # forbid scientific notation


# Box-Muller Transform
def univariate_gaussian_data_generator(m, s):
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)
    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    return m + z0 * np.sqrt(s)


def generate_data(n, m, v):
    return [univariate_gaussian_data_generator(m, v) for _ in range(n)]


def show_result(ax, title, A, y, w):
    predict_class_0 = []
    predict_class_1 = []
    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(A.shape[0]):
        if A[i] @ w < 0:  # 0 means positive, 1 means negative
            predict_class_0.append(A[i, 1:])
            if y[i] == 0:
                tp += 1
            else:
                fp += 1

        else:
            predict_class_1.append(A[i, 1:])
            if y[i] == 1:
                tn += 1
            else:
                fn += 1

    predict_class_0 = np.array(predict_class_0)
    predict_class_1 = np.array(predict_class_1)
    ax.plot(predict_class_0[:, 0], predict_class_0[:, 1], 'ro')
    ax.plot(predict_class_1[:, 0], predict_class_1[:, 1], 'bo')
    ax.set_title(title)

    if title == 'Gradient descent':
        print('Gradient descent:')
    else:
        print('Newton\'s method:')

    print()
    print('w:')
    print('  ', w[0, 0])
    print('  ', w[1, 0])
    print('  ', w[2, 0])
    print()
    print('Confusion Matrix:')
    print("\t\t\tPredict cluster 1\tPredict cluster 2")
    print("Is cluster 1\t\t\t{}\t\t\t{}".format(tp, fn))
    print("Is cluster 2\t\t\t{}\t\t\t{}".format(fp, tn))
    print('\nSensitivity (Successfully predict cluster 1): ', tp / (tp + fn))
    print('Specificity (Successfully predict cluster 2): ', tn / (tn + fp))
    print()


def gradient_descent(A, y, ax1, lr=0.1, epsilon=1e-6, max_iter=1000):
    w_old = np.random.rand(A.shape[1], 1)

    for i in range(max_iter):
        gradient = (A.T) @ (y - (1 / (1 + np.exp(-A @ w_old))))
        w_new = w_old + lr * gradient
        if np.linalg.norm(w_new - w_old) < epsilon:
            break
        w_old = w_new

    show_result(ax1, 'Gradient descent', A, y, w_old)

    return w_old


def newton_method(A, y, ax2, lr=0.1, epsilon=1e-6, max_iter=1000):
    w_old = np.random.rand(A.shape[1], 1)

    for i in range(max_iter):
        D = np.zeros((A.shape[0], A.shape[0]))

        for i in range(A.shape[0]):
            temp = np.dot(A[i], w_old)[0]  # [0] is to convert a 1x1 matrix to a scalar
            D[i, i] = np.exp(-temp) / ((1 + np.exp(-temp)) ** 2)

        H = A.T @ D @ A
        if np.linalg.det(H) == 0:
            gradient = (A.T) @ (y - (1 / (1 + np.exp(-A @ w_old))))
        else:
            H_inv = np.linalg.inv(H)
            gradient = H_inv @ (A.T) @ (y - 1 / (1 + np.exp(-A @ w_old)))

        w_new = w_old + lr * gradient
        if np.linalg.norm(w_new - w_old) < epsilon:
            break
        w_old = w_new

    show_result(ax2, 'Newton\'s method', A, y, w_old)

    return w_old


def main():
    N = 50
    mx1, my1 = 1, 1
    mx2, my2 = 3, 3
    vx1, vy1 = 2, 2
    vx2, vy2 = 4, 4
    
    D1_x = generate_data(N, mx1, vx1)  # shape: (N,)
    D1_y = generate_data(N, my1, vy1)  # shape: (N,)
    D2_x = generate_data(N, mx2, vx2)  # shape: (N,)
    D2_y = generate_data(N, my2, vy2)  # shape: (N,)
    D1 = np.stack((D1_x, D1_y), axis=1)  # shape: (N, 2)
    D2 = np.stack((D2_x, D2_y), axis=1)  # shape: (N, 2)
    D = np.vstack((D1, D2))  # shape: (2N, 2)
    ones_column = np.ones((2 * N, 1))  # shape: (2N, 1)
    A = np.hstack((ones_column, D))  # shape: (2N, 3)
    # print(A)
    y = np.vstack((np.zeros((N, 1)), np.ones((N, 1))))  # y means labels, shape: (2N, 1), 0 for D1, 1 for D2

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
    # ax0: ground truth, ax1: gradient descent, ax2: newton's method
    ax0.plot(D1[:, 0], D1[:, 1], 'ro')
    ax0.plot(D2[:, 0], D2[:, 1], 'bo')
    ax0.set_title('Ground truth')
    w_gradient_descent = gradient_descent(A, y, ax1)
    # print(w_gradient_descent)
    print("---------------------------------------------------------------------")
    w_newton_method = newton_method(A, y, ax2)
    # print(w_newton_method)
    plt.show()


if __name__ == '__main__':
    main()
