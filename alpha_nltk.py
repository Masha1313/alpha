import pandas as pd
import nltk
from nltk.metrics.agreement import AnnotationTask
df1 = pd.read_csv('formatted_ratings.tsv', sep='\t')

print(AnnotationTask(data=df1.itertuples(index=False)).alpha())