import numpy as np
from random_data_generator import polynomial_basis_linear_model_data_generator
import matplotlib.pyplot as plt


def visualization(ax, a, num_points, x_values, y_values, mean, variance, title):
    t = np.linspace(-2, 2, 500)
    mean_predict = np.zeros(500)
    var_predict = np.zeros(500)
    for i in range(500):
        X = np.asarray([t[i]**k for k in range(n)])
        mean_predict[i] = (X @ mean).item(0)  # item(0): 1x1 array to scalar
        var_predict[i] = (a + X @ variance @ X.T).item(0)

    ax.plot(x_values[0:num_points], y_values[0:num_points], 'bo')
    ax.plot(t, mean_predict, 'k-')
    ax.plot(t, mean_predict + var_predict, 'r-')
    ax.plot(t, mean_predict - var_predict, 'r-')
    # ax.fill_between(t, mean_predict - var_predict, mean_predict + var_predict, color='red', alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-20, 20)
    ax.set_title(title)


def bayes_linear_regression(b, n, a, w, threshold=1e-6):
    phi = lambda x: np.array([x**i for i in range(n)]).reshape(1, -1)  # design matrix
    prior_covariance = np.identity(n) / b  # covariance matrix, initial = (b^(-1))I
    prior_lambda_ = np.linalg.inv(prior_covariance)  # lambda = covariance^(-1)
    prior_mean = np.zeros((n, 1))  # mean vector, initial = 0
    x_values, y_values = [], []
    num_points = 0
    previous_predictive_variance = None

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    while num_points < 1000:
        x, y = polynomial_basis_linear_model_data_generator(n, a, w)
        x_values.append(x)
        y_values.append(y)
        print("Add data point ({}, {}):".format(x, y))
        print()

        phi_x = phi(x)  # design matrix 'A'
        posterior_lambda_ = (1 / a) * (phi_x.T @ phi_x) + prior_lambda_
        posterior_covariance = np.linalg.inv(posterior_lambda_)
        posterior_mean = posterior_covariance @ ((1 / a) * phi_x.T * y + prior_lambda_ @ prior_mean)
        
        print("Posterior mean:")
        print(posterior_mean)
        print()
        print("Posterior variance:")
        print(posterior_covariance)
        print()

        # predictive distribution
        predictive_mean = (phi_x @ posterior_mean).item(0)
        predictive_variance = a + (phi_x @ posterior_covariance @ phi_x.T).item(0)
        # print("phi_x shape:", phi_x.shape)
        print("Predictive distribution ~ N({:.5f}, {:.5f})".format(predictive_mean, predictive_variance))
        print()

        if previous_predictive_variance is not None:
            variance_difference = abs(predictive_variance - previous_predictive_variance)
            # print("Variance difference: {:.5f}".format(variance_difference))

            if variance_difference < threshold:
                # print("Predictive variance difference is below the threshold. Stopping iteration.")
                break

        previous_predictive_variance = predictive_variance

        prior_lambda_ = posterior_lambda_
        prior_covariance = posterior_covariance
        prior_mean = posterior_mean

        if num_points == 9:
            visualization(axs[1, 0], a, num_points + 1, x_values, y_values, posterior_mean, posterior_covariance, f'After {num_points + 1} incomes')
        if num_points == 49:
            visualization(axs[1, 1], a, num_points + 1, x_values, y_values, posterior_mean, posterior_covariance, f'After {num_points + 1} incomes')

        num_points += 1

    visualization(axs[0, 1], a, num_points, x_values, y_values, posterior_mean, posterior_covariance, f'Predict result')
    visualization(axs[0, 0], a, 0, x_values, y_values, w, np.zeros((n,n)), f'Ground truth')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

    print("Total number of iterations: ", num_points)


if __name__ == "__main__":
    b = 1
    n = 3
    a = 3
    w = [1, 2, 3]
    bayes_linear_regression(b, n, a, w)
