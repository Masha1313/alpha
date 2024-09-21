import numpy as np
import itertools
from tqdm import tqdm
import pandas as pd
from random import sample
from typing import Dict, List, Tuple
import jsonl
from jsonl import units_dict

def metric(label1: int, label2: int) -> float:
   return (sum([g for g in range(label1, label2 + 1)]) - (label1 + label2) / 2) ** 2
def calculate_bootstrapped_alpha(
        units_dict: Dict[str, List[int]],
        D_e: float,
        num_samples: int = 200,
       # metric= lambda pair: abs(pair[0]-pair[1]),
        #metric=lambda pair: 0 if pair[0] == pair[1] else 1,
        p_value: float = 0.05) -> Dict[str, Tuple[float, float]]:
    num_dig: int = len(str(num_samples))
    # Хэш таблица с альфами
    alpha_dict: Dict[int, int] = {}
    # Количество всех пар
    N_0: int = np.sum(
        [len(answers) * (len(answers) - 1) // 2 for unit, answers in units_dict.items()]
    )
    # Количество значений в матрице
    N_dot: int = sum(len(answers) for answers in units_dict.values())
    print(N_dot)
    errors_dict: Dict[str, List[float]] = {}
    # Массив всех возможных пар
    pairs: List[Tuple[str, str]] = []

    for unit, answers in units_dict.items():
        unit_pairs = list(itertools.combinations(answers, 2))

        unit_errors: List[float] = [round( 2 * metric(pair[0], pair[1]) / (N_dot * D_e), 3) for pair in unit_pairs]

        sum_errors = round(sum(unit_errors), 3)

        errors_dict[unit] = {'errors': unit_errors, 'sum': sum_errors}

        sorted_errors_dict = dict(sorted(errors_dict.items(), key=lambda item: item[1]['sum'], reverse=True))

    for unit, data in sorted_errors_dict.items():
        print( f"{unit}, Sum: {data['sum']}, Errors: {data['errors']}")

result: Dict[str, Tuple[float, float]] = calculate_bootstrapped_alpha(units_dict,  0.575)
