import numpy as np
import itertools

import pandas as pd


def calculate_bootstrapped_alpha(data_matrix, D_e, num_samples=200, p_value=0.1,
                                 metric=lambda pair: 1 if pair[0] == pair[1] else 0):
    alpha_dict = {}

    N_0 = np.sum(
        [np.count_nonzero(~np.isnan(unit)) * (np.count_nonzero(~np.isnan(unit)) - 1) // 2 for unit in data_matrix]
    )

    N_dot = np.sum([np.count_nonzero(~np.isnan(unit)) if np.count_nonzero(~np.isnan(unit)) >= 2 else 0 for unit in data_matrix])

    pairs = []
    for unit in data_matrix:
        not_nan = unit[~np.isnan(unit)]
        if len(not_nan) < 2:
            continue
        unit_pairs = list(itertools.combinations(not_nan, 2))
        pairs.extend(unit_pairs)

    for _ in range(num_samples):
        alpha = 1.0

        for unit in data_matrix:
            num_observers = np.count_nonzero(~np.isnan(unit))
            num_pairs = num_observers * (num_observers - 1) // 2
            r_pairs = np.random.choice(np.arange(N_0), num_pairs, replace=False)

            for i in range(num_pairs):
                pair = pairs[r_pairs[i]]
                E_r = 2 * metric(pair) / (N_dot * D_e)
                alpha -= E_r / (num_observers - 1)

        alpha_key = int(np.ceil(alpha * 1000))
        if alpha < -1:
            alpha_key = -1000

        if alpha_key in alpha_dict:
            alpha_dict[alpha_key] += 1
        else:
            alpha_dict[alpha_key] = 1

    for key in alpha_dict:
        alpha_dict[key] /= num_samples

    sorted_alpha_dict = dict(sorted(alpha_dict.items()))

    cumulative_sum = 0
    alpha_smallest = None
    for alpha, n_alpha in sorted_alpha_dict.items():
        cumulative_sum += n_alpha
        if cumulative_sum >= p_value / 2:
            alpha_smallest = alpha / 1000
            break

    cumulative_sum = 0
    alpha_largest = None
    for alpha, n_alpha in reversed(sorted_alpha_dict.items()):
        cumulative_sum += n_alpha
        if cumulative_sum >= p_value / 2:
            alpha_largest = alpha / 1000
            break

    return {
        'confidence_interval': (alpha_smallest, alpha_largest),
        'alpha_distribution': alpha_dict
    }


df = pd.read_csv('crowd_labels.tsv', sep='\t', names=['worker', 'task', 'answer'])
data = pd.pivot_table(df, index='task', columns='worker', values='answer')
data_array = data.values

data_test = pd.DataFrame([
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
]).T

result = calculate_bootstrapped_alpha(data_test.values, 0.4421052631578947)
#result = calculate_bootstrapped_alpha(data_array, 0.4796444639525952)
print("confidence interval:", result['confidence_interval'])



