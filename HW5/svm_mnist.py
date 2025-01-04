import gc
import numpy as np
from libsvm.svmutil import *
from scipy.spatial.distance import cdist
import pandas as pd


def reset_environment():
    gc.collect()
    np.random.seed(42)


def grid_search(X_train, Y_train, kernel_type, param_grid):
    best_acc = 0
    best_params = None

    for C in param_grid['C']:
        for gamma in param_grid.get('gamma', [None]):
            for degree in param_grid.get('degree', [None]):
                options = f'-t {kernel_type} -c {C}'

                if gamma is not None:
                    options += f' -g {gamma}'
                if degree is not None:
                    options += f' -d {degree}'

                options += ' -v 5 -q'
                print(f"Training with options: {options}")

                if kernel_type == 4:
                    formatted_X_train = np.hstack((np.arange(1, X_train.shape[0] + 1).reshape(-1, 1), custom_kernel(X_train, X_train, gamma)))
                    acc = svm_train(Y_train, formatted_X_train.tolist(), options)
                else:
                    acc = svm_train(Y_train, X_train, options)
                
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'C': C}

                    if gamma is not None:
                        best_params['gamma'] = gamma
                    if degree is not None:
                        best_params['degree'] = degree

    return best_params, best_acc


def custom_kernel(x1, x2, gamma=0.01):
    # print(x1.shape, x2.shape)
    linear_kernel = np.dot(x1, x2.T)
    rbf_kernel = np.exp(-gamma * cdist(x1, x2, 'sqeuclidean'))
    
    return linear_kernel + rbf_kernel


# Load the data
X_train = pd.read_csv('./data/X_train.csv', header=None).values
Y_train = pd.read_csv('./data/Y_train.csv', header=None).values.ravel()
X_test = pd.read_csv('./data/X_test.csv', header=None).values
Y_test = pd.read_csv('./data/Y_test.csv', header=None).values.ravel()

# Task 1
reset_environment()
print("Task 1: Training and evaluating with different kernels")
kernels = {'linear': 0, 'polynomial': 1, 'RBF': 2}

for kernel_name, kernel_type in kernels.items():
    print(f"Training with {kernel_name} kernel...")
    model = svm_train(Y_train, X_train, f'-t {kernel_type} -c 1 -q')
    print(f"Evaluating with {kernel_name} kernel...")
    p_label, p_acc, p_val = svm_predict(Y_test, X_test, model)
    # print(f"{kernel_name} kernel accuracy: {p_acc[0]}%")
    print("----------------------------------------------------------")

# Task 2
reset_environment()
print("\nTask 2: Grid search for best parameters")
param_grid_task2 = {
    'linear': {'C': [0.1, 1, 10]},
    'polynomial': {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1], 'degree': [2, 3]},
    'RBF': {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
}

for kernel_name, kernel_type in kernels.items():
    print(f"Performing grid search for {kernel_name} kernel...")
    best_params_task2, best_acc_task2 = grid_search(X_train, Y_train, kernel_type, param_grid_task2[kernel_name])
    print(f"Best parameters for {kernel_name} kernel: {best_params_task2}")
    print(f"Best 5-fold cross-validation accuracy: {best_acc_task2:.2f}%")
    print("----------------------------------------------------------")

model_task2 = svm_train(Y_train, X_train, f'-t 2 -c 10 -g 0.01 -q')
print(f"Evaluating with RBF kernel...")
p_label, p_acc, p_val = svm_predict(Y_test, X_test, model_task2)
# print(f"RBF kernel accuracy: {p_acc[0]}%")

# Task 3
reset_environment()
print("\nTask 3: Using custom kernel (linear + RBF) with Grid Search")
param_grid_task3 = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}

best_params_task3, best_acc_task3 = grid_search(X_train, Y_train, 4, param_grid_task3)
print(f"Best parameters for custom kernel (linear + RBF): {best_params_task3}")
print(f"Best 5-fold cross-validation accuracy: {best_acc_task3:.2f}%")

formatted_X_train = np.hstack((np.arange(1, X_train.shape[0] + 1).reshape(-1, 1), custom_kernel(X_train, X_train, best_params_task3['gamma'])))
model_task3 = svm_train(Y_train, formatted_X_train.tolist(), f'-t 4 -c {best_params_task3["C"]} -g {best_params_task3["gamma"]} -q')
print(f"Evaluating with custom kernel (linear + RBF)...")
formatted_X_test = np.hstack((np.arange(1, X_test.shape[0] + 1).reshape(-1, 1), custom_kernel(X_test, X_train, best_params_task3['gamma'])))
p_label, p_acc, p_val = svm_predict(Y_test, formatted_X_test.tolist(), model_task3)
# print(f"Custom kernel (linear + RBF) accuracy: {p_acc[0]}%")
