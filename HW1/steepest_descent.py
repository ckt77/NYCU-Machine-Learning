import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    return data[:, 0], data[:, 1]  # x, y


def transpose_matrix(A):
    assert len(A.shape) == 2, "Input must be a 2D matrix."
    transpose = np.zeros((A.shape[1], A.shape[0]))

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            transpose[j, i] = A[i, j]

    return transpose


def polynomial_basis(x, degree):  # construct 'A', i.e., Vandermonde matrix
    basis = transpose_matrix(np.array([x ** i for i in range(degree)]))
    # print(basis.shape)  # should be (n, degree), n represents the number of data points

    return basis


def matrix_multiplication(A, B):  # O(n^3), n = A.shape[0]
    assert A.shape[1] == B.shape[0], "A's number of columns must equal B's number of rows for multiplication."
    result = np.zeros((A.shape[0], B.shape[1]))

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i, j] += A[i, k] * B[k, j]

    return result


def compute_gradient(A, y, coefficients, lambd, epsilon):
    coefficients_reshape = coefficients.reshape(-1, 1)  # (degree, ) -> (degree, 1)
    residuals = matrix_multiplication(A, coefficients_reshape).flatten() - y  # Aw - y
    A_transpose = transpose_matrix(A)

    # gradient = 2 * A^T(Aw - y) + λsign(w)
    gradient = 2 * matrix_multiplication(A_transpose, residuals.reshape(-1, 1)).flatten() + lambd * np.sign(coefficients)

    return gradient
# def compute_gradient(A, y, coefficients, lambd, epsilon):
#     coefficients_reshape = coefficients.reshape(-1, 1)  # (degree, ) -> (degree, 1)
#     residuals = matrix_multiplication(A, coefficients_reshape).flatten() - y  # Aw - y
#     A_transpose = transpose_matrix(A)

#     # Gradient of the least squares term
#     gradient_ls = 2 * matrix_multiplication(A_transpose, residuals.reshape(-1, 1)).flatten()

#     # Gradient of the approximated L1 norm (L1,epsilon)
#     gradient_l1_approx = coefficients / np.sqrt(coefficients**2 + epsilon)

#     # Full gradient: 2 * A^T(Aw - y) + λ * gradient of L1 norm approximation
#     gradient = gradient_ls + lambd * gradient_l1_approx

#     return gradient


def steepest_descent(A, y, degree, lambd, learning_rate=1e-4, tol=1e-10, max_iter=10000, epsilon=1e-10):
    coefficients_old = np.zeros(degree)
    coefficients_new = np.zeros(degree)

    for i in range(max_iter):
        gradient = compute_gradient(A, y, coefficients_old, lambd, epsilon)
        coefficients_new = coefficients_old - learning_rate * gradient

        if np.linalg.norm(abs(coefficients_new - coefficients_old)) < tol:
            break

        coefficients_old = coefficients_new

    return coefficients_new


def plot_regression_curve(x, y, coefficients, degree, lambd):
    plt.scatter(x, y, label='Data points', color='blue')
    x_line = np.linspace(min(x) - 1, max(x) + 1, 100)
    y_line = matrix_multiplication(polynomial_basis(x_line, degree), coefficients).flatten()  # (n, 1) -> (n, )
    plt.plot(x_line, y_line, color='red', label='Fitted polynomial curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Polynomial Regression (Degree {degree}) with λ={lambd}')
    plt.show()


def linear_regression_via_steepest_descent(x, y, degree, lambd, epsilon):
    A = polynomial_basis(x, degree)
    coefficients = steepest_descent(A, y, degree, lambd, epsilon)
    coefficients_reshape = coefficients.reshape(-1, 1)  # (degree, ) -> (degree, 1)

    y_pred = matrix_multiplication(A, coefficients_reshape).flatten()  # (n, 1) -> (n, )
    total_error = np.sum((y_pred - y) ** 2)
    equation = "Fitting line: "

    for i, coef in enumerate(coefficients[::-1]):
        if i == 0:  # leading coefficient
            equation += f"{coef:.10f}X^{(degree - 1 - i)}"
        else:
            if i == (degree - 1):
                if coef > 0:
                    equation += f" + {coef:.10f}"
                else:
                    equation += f" - {abs(coef):.10f}"
            else:
                if coef > 0:
                    equation += f" + {coef:.10f}X^{(degree - 1 - i)}"
                else:
                    equation += f" - {abs(coef):.10f}X^{(degree - 1 - i)}"

    print("Steepest descent method:")
    print(equation)
    print(f"Total Error: {total_error:.10f}")

    return coefficients_reshape


file_path = 'testfile.txt'
degree = 3
lambd = 10000
epsilon = 1e-4
x, y = load_data(file_path)
coefficients = linear_regression_via_steepest_descent(x, y, degree, lambd, epsilon)
plot_regression_curve(x, y, coefficients, degree, lambd)
