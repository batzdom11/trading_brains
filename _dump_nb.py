import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('temporal_transformers_exploration.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Only dump key cells: 6 (features), 8 (training), 11 (dataset), 12 (model config), 15-16 (trainer), 18 (prediction)
key_cells = [6, 8, 11, 12, 15, 16, 18, 21, 24]
for i, c in enumerate(nb['cells']):
    cell_num = i + 1
    if cell_num in key_cells:
        print(f"=== CELL {cell_num} ({c['cell_type']}) ===")
        src = ''.join(c['source'])
        # Replace problematic unicode chars
        src = src.replace('\u274c', 'X').replace('\u2705', 'OK')
        print(src)
        print()
