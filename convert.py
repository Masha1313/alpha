# Python program to convert .tsv file to .csv file
# importing pandas library
import pandas as pd

tsv_file='crowd_labels.tsv'

# reading given tsv file
csv_table=pd.read_table(tsv_file,sep='\t')

# converting tsv file into csv
csv_table.to_csv('crowd_convert.csv',index=False)

# output
print("Successfully made csv file")
