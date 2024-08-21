import numpy as np
import pandas as pd

df = pd.read_csv('crowd_labels.tsv', sep='\t', names=['worker', 'task', 'answer'])
data_matrix1 = pd.pivot_table(df, index='task', columns='worker', values='answer')

def alpha(data_matrix):
    n_items = len(data_matrix)
    n_raters = len(data_matrix.T)
    max_rate = int(np.max(data_matrix.values) + 1)
    coincidence_matrix = np.zeros((max_rate, max_rate))
    for i in range(n_items):
        sum_nan = 0
        row_matrix = np.zeros((max_rate, max_rate))
        for j in range(n_raters):
            if data_matrix.iloc[i, j] == np.nan:
                sum_nan += 1
            for k in range(n_raters):
                if j == k:
                    continue
                row_matrix[int(data_matrix.iloc[i, j]), int(data_matrix.iloc[i, k])] += 1
        row_matrix /= (n_raters - sum_nan - 1)
        coincidence_matrix += row_matrix

    n_sums = [np.sum(coincidence_matrix[i]) for i in range(max_rate)]

    D_o = (np.sum(coincidence_matrix) - np.trace(coincidence_matrix)) / np.sum(coincidence_matrix)
    print(f'D_o: {D_o}')
    D_e = 0.0
    for i in range(max_rate):
        for j in range(max_rate):
            if i == j:
                continue
            D_e += n_sums[i] * n_sums[j]
    D_e /= np.sum(coincidence_matrix) * (np.sum(coincidence_matrix) - 1)
    print(f'D_e: {D_e}')

    return 1 - D_o / D_e


#data_test = pd.DataFrame([
 #   [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
 #   [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
#])
print(alpha(data_matrix1.T))