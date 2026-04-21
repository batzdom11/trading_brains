import json
d = json.loads(open(r'c:\Users\batzi\OneDrive\Dokumente\trading_brains\backfill_result.json').read())
print('status:', d.get('status'))
print('updated:', d.get('total_updated'))
print('skipped:', d.get('total_skipped'))
print('total:', d.get('total_rows'))
# Show a few sample details with different last_price values
details = d.get('details', [])
seen = set()
for item in details:
    lp = item.get('new_last_price')
    if lp and lp not in seen:
        seen.add(lp)
        print(f"  ts={item['timestamp']} last_price={lp} pred_60m={item.get('new_pred_60m')}")
    if len(seen) >= 10:
        break
