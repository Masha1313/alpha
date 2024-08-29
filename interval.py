import numpy as np
import itertools
from tqdm import tqdm
import pandas as pd


def create_task_answer_dict(df):
    units_dict = {}

    for unit, group in df.groupby('task'):
        answers = group['answer'].dropna().tolist()
        if len(answers) >= 2:
            units_dict[unit] = answers

    return units_dict


def calculate_bootstrapped_alpha(units_dict, D_e, num_samples=100, p_value=0.05,
                                 metric=lambda pair: 0 if pair[0] == pair[1] else 1):
    num_dig = len(str(num_samples))
    alpha_dict = {}
    N_dot = sum(len(answers) for answers in units_dict.values())

    for _ in tqdm(range(num_samples), ncols=80, desc='Progress'):
        alpha = 1.0
        for unit, answers in units_dict.items():
            num_observers = len(answers)
            unit_pairs = list(itertools.combinations(answers, 2))
            num_pairs = len(unit_pairs)
            r_pairs_indices = np.random.choice(num_pairs, num_pairs, replace=False)

            for i in range(num_pairs):
                pair = unit_pairs[r_pairs_indices[i]]
                E_r = 2 * metric(pair) / (N_dot * D_e)
                alpha -= E_r / (num_observers - 1)

        alpha_key = int(np.ceil(alpha * (10 ** num_dig)))
        if alpha < -1:
            alpha_key = -10 ** num_dig

        alpha_dict[alpha_key] = alpha_dict.get(alpha_key, 0) + 1

    for key in alpha_dict:
        alpha_dict[key] /= num_samples

    sorted_alpha_dict = dict(sorted(alpha_dict.items()))

    cumulative_sum = 0
    alpha_smallest = None
    for alpha, n_alpha in sorted_alpha_dict.items():
        cumulative_sum += n_alpha
        if cumulative_sum >= p_value / 2:
            alpha_smallest = alpha / (10 ** num_dig)
            break

    cumulative_sum = 0
    alpha_largest = None
    for alpha, n_alpha in reversed(sorted_alpha_dict.items()):
        cumulative_sum += n_alpha
        if cumulative_sum >= p_value / 2:
            alpha_largest = alpha / (10 ** num_dig)
            break

    return {'confidence_interval': (alpha_smallest, alpha_largest)}


df = pd.read_csv('crowd_labels.tsv', sep='\t', names=['worker', 'task', 'answer'])

tasks_dict = create_task_answer_dict(df)

result = calculate_bootstrapped_alpha(tasks_dict, 0.485570550804453)
print("confidence interval:", result['confidence_interval'])
