import numpy as np
import itertools
from tqdm import tqdm
import pandas as pd
from random import sample

#создаем хэш таблицу из файла без нан значений
def create_task_answer_dict(df):
    units_dict = {}
   #task - ключи, answers- значения
    for unit, group in df.groupby('task'):
        answers = group['answer'].dropna().tolist()
        if len(answers) >= 2:
            units_dict[unit] = answers

    return units_dict

def calculate_bootstrapped_alpha(units_dict, D_e, num_samples=100, p_value=0.1,
                                 metric=lambda pair: 0 if pair[0] == pair[1] else 1):
    num_dig = len(str(num_samples))
    #хэш таблица с альфами
    alpha_dict = {}
#кол-во всех пар
    N_0 = np.sum(
        [len(answers) * (len(answers) - 1) // 2 for unit, answers in units_dict.items()]
    )
#кол-во значений в матрице
    N_dot = sum(len(answers) for answers in units_dict.values())

# массив всех возможных пар
    pairs = []
    for unit, answers in units_dict.items():
        unit_pairs = list(itertools.combinations(answers, 2))
        pairs.extend(unit_pairs)

    for _ in tqdm(range(num_samples), ncols=80, desc='Progress'):
        alpha = 1.0
        for unit, answers in units_dict.items():
            num_observers = len(answers)
            num_pairs = num_observers * (num_observers - 1) // 2
            pair_indices = sample(range(N_0), num_pairs)

            for i in range(num_pairs):
                pair = pairs[pair_indices[i]]
                E_r = 2 * metric(pair) / (N_dot * D_e)
                alpha -= E_r / (num_observers - 1)

        alpha_key = int(np.ceil(alpha * (10 ** num_dig)))
        if alpha < -1:
            alpha_key = -10**num_dig

        alpha_dict[alpha_key] = alpha_dict.get(alpha_key, 0) + 1

    for key in alpha_dict:
        alpha_dict[key] /= num_samples

#сортируем хэштаблицу с альфами
    sorted_alpha_dict = dict(sorted(alpha_dict.items()))

#считаем интервал
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
#alpha=0.261920..