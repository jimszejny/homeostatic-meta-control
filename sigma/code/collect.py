"""Merge all experiment JSON into results/results.json."""
import json, os
R = '../results/'
out = {}
for k in ('step0', 'e1', 'e2', 'e3', 'e4', 'e5'):
    out[k] = json.load(open(R + k + '.json'))
out['predictions_sha256'] = open(R + 'predictions.sha256').read().split()[0]
json.dump(out, open(R + 'results.json', 'w'), indent=1)
print('results.json written', os.path.getsize(R + 'results.json'), 'bytes')
