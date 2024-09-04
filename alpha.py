import numpy as np
import pandas as pd


df = pd.read_csv('small.tsvт', sep='\t', names=['worker', 'task', 'answer'])
data_matrix1 = pd.pivot_table(df, index='task', columns='worker', values='answer')


def alpha(data_matrix):
    n_items = len(data_matrix)
    max_rate = int(np.max(data_matrix) + 1)
    coincidence_matrix = np.zeros((max_rate, max_rate))
    for i in range(n_items):
        current_row = data_matrix.iloc[i].dropna()
        valid_indices = current_row.index
        num_valid_raters = len(valid_indices)

        if num_valid_raters < 2:
            continue

        ratings = current_row.values.astype(int)
        row_matrix = np.zeros((max_rate, max_rate))
        for j in range(num_valid_raters):
            for k in range(num_valid_raters):
                if j != k:
                    row_matrix[ratings[j], ratings[k]] += 1

        row_matrix /= (num_valid_raters - 1)
        coincidence_matrix += row_matrix

    n_sums = [np.sum(coincidence_matrix[i]) for i in range(max_rate)]

    D_o = (np.sum(coincidence_matrix) - np.trace(coincidence_matrix)) / np.sum(coincidence_matrix)
    print(f'D_o: {D_o}')
    D_e = 0.0
    for i in range(max_rate):
        for c in range(max_rate):
            if i == c:
                continue
            D_e += n_sums[i] * n_sums[c]
    D_e /= np.sum(coincidence_matrix) * (np.sum(coincidence_matrix) - 1)
    print(f'D_e: {D_e}')

    return 1 - D_o / D_e


data_test = pd.DataFrame([
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
])
# print(alpha(data_test.T))
print(alpha(data_matrix1))
