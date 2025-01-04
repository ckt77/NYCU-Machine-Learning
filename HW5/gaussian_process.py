from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def rational_quadratic_kernel(x1, x2, sigma, alpha, length_scale):
    """
    Computes the Rational Quadratic kernel between two sets of points.

    Parameters:
    1. x1, x2: Input data, can be of different lengths
    2. sigma: The amplitude parameter
    3. alpha: The shape parameter
    4. length_scale: Length scale controls how far apart inputs must be to be considered correlated
    """
    distance = cdist(x1, x2, 'euclidean')
    kernel = (sigma**2) * (1 + distance**2 / (2 * alpha * length_scale**2))**(-alpha)

    return kernel


def gaussian_process_regression(X_train, Y_train, X_pred, sigma=1.0, alpha=1.0, length_scale=1.0, beta=5):
    """
    Perform Gaussian Process Regression to predict the distribution of f at X_pred.

    Parameters:
    1. X_train: Training input data with shape (n, 1)
    2. Y_train: Training output data with shape (n, 1)
    3. X_pred: Prediction points with shape (m, 1)
    4. sigma: Amplitude parameter for kernel
    5. alpha: Shape parameter for kernel
    6. length_scale: Length scale parameter for kernel
    7. beta: Noise variance (inverse of observation noise)
    """
    # Compute covariance matrices
    C = rational_quadratic_kernel(X_train, X_train, sigma, alpha, length_scale) + np.eye(len(X_train)) / beta  # K(X, X) + (β^-1)I
    K_s = rational_quadratic_kernel(X_train, X_pred, sigma, alpha, length_scale)  # K(X, X*)
    K_ss = rational_quadratic_kernel(X_pred, X_pred, sigma, alpha, length_scale) + np.eye(len(X_pred)) / beta  # K(X*, X*) + (β^-1)I

    C_inv = np.linalg.inv(C)

    # Compute the mean and covariance of the posterior distribution
    mu_s = (K_s.T).dot(C_inv).dot(Y_train)  # µ(x*)
    cov_s = K_ss - (K_s.T).dot(C_inv).dot(K_s)  # σ^2(x*)

    return mu_s, cov_s


def negative_log_likelihood(params, X_train, Y_train, beta=5):
    """
    Compute the negative log marginal likelihood for the GP with Rational Quadratic kernel.

    Parameters:
    1. params: List of kernel parameters [sigma, alpha, length_scale]
    2. X_train: Training input data
    3. Y_train: Training output data
    4. beta: Noise variance (inverse of observation noise)
    """
    sigma, alpha, length_scale = params
    K = rational_quadratic_kernel(X_train, X_train, sigma, alpha, length_scale)
    C = K + np.eye(len(X_train)) / beta
    C_inv = np.linalg.inv(C)

    NLL = 0.5 * (np.log(np.linalg.det(C)) + (Y_train.T).dot(C_inv).dot(Y_train) + len(C) * np.log(2 * np.pi))
    # print(NLL.shape) # (1, 1)

    return NLL[0, 0]


# Load the data from the file
data = np.loadtxt('./data/input.data')
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)
# print(X.shape, Y.shape)  # (34, 1) (34, 1)

# Task 1
X_pred = np.linspace(-60, 60, 1000).reshape(-1, 1)
initial_params = [1.0, 1.0, 1.0]  # [sigma, alpha, length_scale]
beta = 5
mu_s, cov_s = gaussian_process_regression(X, Y, X_pred, sigma=initial_params[0], alpha=initial_params[1], 
                                          length_scale=initial_params[2], beta=beta)
# print(mu_s.shape, cov_s.shape)  # (1000, 1) (1000, 1000)
std = np.sqrt(np.diag(cov_s))
print(f"Initial Parameters: σ={initial_params[0]}, α={initial_params[1]}, length_scale={initial_params[2]}")
nll_initial = negative_log_likelihood(initial_params, X, Y, beta)
print(f"Initial Negative Log Likelihood: {nll_initial:.3f}")

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='red', label='Training Data')
plt.plot(X_pred, mu_s, color='blue', label='Mean Prediction')
plt.fill_between(X_pred.flatten(), 
                 mu_s.flatten() - 1.96 * std, 
                 mu_s.flatten() + 1.96 * std, 
                 color='blue', alpha=0.2, label='95% Confidence Interval')
plt.title('Gaussian Process Regression with Rational Quadratic Kernel')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-60, 60)
plt.legend()
plt.show()

# Task 2
result = minimize(negative_log_likelihood, initial_params, args=(X, Y, beta), bounds=[(1e-2, None), (1e-2, None), (1e-2, None)], 
                  options={'maxiter': 1000})
optimized_params = result.x
print(f"Optimized Parameters: σ={optimized_params[0]:.3f}, α={optimized_params[1]:.3f}, length_scale={optimized_params[2]:.3f}")
nll_optimized = negative_log_likelihood(optimized_params, X, Y, beta)
print(f"Optimized Negative Log Likelihood: {nll_optimized:.3f}")
mu_s_optimized, cov_s_optimized = gaussian_process_regression(X, Y, X_pred, 
                                                   sigma=optimized_params[0], 
                                                   alpha=optimized_params[1], 
                                                   length_scale=optimized_params[2], 
                                                   beta=beta)
std_optimized = np.sqrt(np.diag(cov_s_optimized))

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='red', label='Training Data')
plt.plot(X_pred, mu_s_optimized, color='blue', label='Optimized Mean Prediction')
plt.fill_between(X_pred.flatten(), 
                 mu_s_optimized.flatten() - 1.96 * std_optimized, 
                 mu_s_optimized.flatten() + 1.96 * std_optimized, 
                 color='blue', alpha=0.2, label='Optimized 95% Confidence Interval')
plt.title('Optimized Gaussian Process Regression with Rational Quadratic Kernel')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-60, 60)
plt.legend()
plt.show()
