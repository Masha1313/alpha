import pandas
import json
import pandas as pd
pd.set_option('display.max_row', None)
df_claude = pd.read_json('annotated_train_claude.jsonl', lines=True)
#print(df_claude)
df_claude['llm'] = df_claude['judge'].apply(lambda s: s.get('verdict'))
df_claude.to_json('llm_annotated_train_claude.jsonl', orient='records', lines=True)

with open('llm_annotated_train_claude.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]


columns = []

for item in data:

    columns.append([item['annotation'], item['exact_match'], item['llm']])

annot_matrix = pd.DataFrame(columns, columns=['annotation', 'exact_match', 'llm'])





units_dict = {f"unit{index}": row.tolist() for index, row in annot_matrix.iterrows()}

# Сохранение в JSON
with open('units_dict.json', 'w') as json_file:
    json.dump(units_dict, json_file, indent=4)
print(annot_matrix)


#df_llm = pd.read_json('llm_annotated_train_claude.jsonl', lines=True)
#print(df_claude)

