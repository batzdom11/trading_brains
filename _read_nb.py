import json, sys

with open('temporal_transformers_exploration.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    src = ''.join(cell['source'])
    print(f'===== CELL {i+1} ({ctype}) =====')
    print(src)
    print()
