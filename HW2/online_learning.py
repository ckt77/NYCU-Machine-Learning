import numpy as np


def log_factorial(n):
    if n == 0 or n == 1:
        return 0
    return np.sum(np.log(np.arange(1, n + 1)))


def comb(n, k):
    return np.exp(log_factorial(n) - log_factorial(k) - log_factorial(n - k))


def calculate_binomial_likelihood(k, n, p):  # p: probability of success, k: number of success, n: number of trials
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def main(file_path, a, b):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()  # delete '\n'
        n = len(line)
        k = line.count('1')

        p_mle = k / n  # Maximum Likelihood Estimation
        likelihood = calculate_binomial_likelihood(k, n, p_mle)

        print(f"case {i+1}: {line}")
        print(f"Likelihood: {likelihood}")
        print(f"Beta prior: a = {a} b = {b}")

        a_posterior = a + k
        b_posterior = b + (n - k)

        print(f"Beta posterior: a = {a_posterior} b = {b_posterior}\n")

        a, b = a_posterior, b_posterior


file_path = "testfile.txt"
a = 10
b = 1

main(file_path, a, b)
