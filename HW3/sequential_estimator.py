from random_data_generator import univariate_gaussian_data_generator


def sequential_estimation(m, s, threshold=1e-3):
    n = 0  # count for the iterations
    mean = 0
    M2 = 0
    previous_mean = float('inf')
    previous_variance = float('inf')
    print("Data point source function: N({}, {})".format(m, s))
    print()

    while True:
        n += 1
        x = univariate_gaussian_data_generator(m, s)
        delta = x - mean  # X_(n+1) - mean_n
        mean += (delta / n)
        delta2 = x - mean  # X_(n+1) - mean_(n+1)
        M2 += (delta * delta2)
        variance = M2 / n if n > 1 else 0

        print(f"Add data point: {x}")
        print(f"Mean = {mean} Variance = {variance}")

        if abs(mean - previous_mean) < threshold and abs(variance - previous_variance) < threshold:
            break

        previous_mean = mean
        previous_variance = variance

    # print(f"Total iterations: {n}")


if __name__ == "__main__":
    sequential_estimation(3.0, 5.0)
