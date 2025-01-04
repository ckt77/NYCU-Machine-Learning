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


def LU_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    return L, U


def forward_substitution(L, b):  # Ly = b
    n = L.shape[0]
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    return y


def backward_substitution(U, y):  # Ux = y
    n = U.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def solve_via_LU(A, b):  # Ax = b
    L, U = LU_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    return x


def inverse_matrix(A):
    n = A.shape[0]
    inv_A = np.zeros_like(A)
    identity = np.eye(n)

    for i in range(n):
        e_i = identity[:, i]
        inv_A[:, i] = solve_via_LU(A, e_i)

    return inv_A


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


def rLSE(x, y, degree, lambd):
    A = polynomial_basis(x, degree)
    # print(A)
    A_transpose = transpose_matrix(A)
    A_transpose_A = matrix_multiplication(A_transpose, A)
    y_reshape = y.reshape(-1, 1)  # (n, ) -> (n, 1)
    A_transpose_y = matrix_multiplication(A_transpose, y_reshape)

    # add regularization term λI
    reg_matrix = A_transpose_A + lambd * np.eye(A_transpose_A.shape[0])

    # calculate (A^T A + λI)^-1
    inv_reg_matrix = inverse_matrix(reg_matrix)

    # x = ((A^T A + λI)^-1 A^T) b
    coefficients = matrix_multiplication(inv_reg_matrix, A_transpose_y)
    # print(coefficients.shape)

    y_pred = matrix_multiplication(A, coefficients).flatten()  # (n, 1) -> (n, )
    total_error = np.sum((y_pred - y) ** 2)
    equation = "Fitting line: "

    for i, coef in enumerate(coefficients[::-1]):
        coef = coef[0]  # Solve "TypeError: unsupported format string passed to numpy.ndarray.__format__"
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

    print("LSE:")
    print(equation)
    print(f"Total Error: {total_error:.10f}")

    return coefficients


file_path = 'testfile.txt'
degree = 3
lambd = 10000
x, y = load_data(file_path)
coefficients = rLSE(x, y, degree, lambd)
plot_regression_curve(x, y, coefficients, degree, lambd)
