import pandas as pd

df = pd.read_csv('matrix_1-0.csv', sep=';', header=None)

output_rows = []

for task_number, row in df.iterrows():
    for worker_number, mark in row.items():
        if not pd.isna(mark):
            output_rows.append([worker_number + 1, task_number + 1, mark])

output_df = pd.DataFrame(output_rows)

output_df.to_csv('formatted_ratings.tsv', sep='\t', index=False, header=False)
