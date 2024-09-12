import pandas as pd
import nltk
import random
from tqdm import tqdm
import numpy as np
from nltk.metrics.agreement import AnnotationTask

df = pd.read_csv('m-transformed.tsv', sep='\t', names=['worker', 'task', 'answer'])

task = AnnotationTask(data=df.itertuples(index=False))
print(task.alpha())

def bootstrap(data, num_samples=5000, p=0.05):
    bootstrap_alpha = []
    num_rows, num_cols = data.shape
    for _ in tqdm(range(num_samples), ncols=80, desc='Progress'):
        resample_indices = np.random.choice(num_rows, 200, replace=False)
        resample_data = data[resample_indices, :]
        resample_alpha = AnnotationTask(data=resample_data).alpha()
        bootstrap_alpha.append(resample_alpha)
    lower_bound = np.percentile(bootstrap_alpha, 100 * p / 2)
    upper_bound = np.percentile(bootstrap_alpha, 100 * (1 - p / 2))
    return lower_bound, upper_bound


data_array = df[['worker', 'task', 'answer']].to_numpy()
print(bootstrap(data_array))