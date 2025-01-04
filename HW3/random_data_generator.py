import numpy as np


# Box-Muller Transform
def univariate_gaussian_data_generator(m, s):
    """
    Generates a data point from a univariate Gaussian distribution N(m, s).
    
    Parameters:
    m (float): Mean of the distribution.
    s (float): Variance of the distribution.
    
    Returns:
    float: A data point from the Gaussian distribution.
    """
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)
    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    return m + z0 * np.sqrt(s)


def polynomial_basis_linear_model_data_generator(n, a, w):
    """
    Generates a data point (x, y) from a polynomial basis linear model.
    
    Parameters:
    n (int): Basis number.
    a (float): Variance of the normal distribution for error term.
    w (list of float): Coefficients of the polynomial.
    
    Returns:
    tuple: A data point (x, y).
    """
    x = np.random.uniform(-1, 1)
    phi = np.array([x**i for i in range(n)])
    e = univariate_gaussian_data_generator(0, a)
    y = np.dot(w, phi) + e
    y_ = float(y)
    # print(y)
    # print(y_)

    return x, y_


if __name__ == "__main__":
    m = 0
    s = 1
    print("Univariate Gaussian Data Point:", univariate_gaussian_data_generator(m, s))

    n = 4
    a = 1
    w = [1, 2, 3, 4]
    print("Polynomial Basis Linear Model Data Point:", polynomial_basis_linear_model_data_generator(n, a, w))
