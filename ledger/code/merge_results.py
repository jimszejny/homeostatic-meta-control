"""Merge per-step result files into results/results.json."""
import json, os, hashlib
R = os.path.join(os.path.dirname(__file__), '..', 'results')
out = {}
for k in ('step0', 'r1', 'r2', 'r3', 'r4', 'm2_alignment', 'posthoc'):
    p = os.path.join(R, k + '.json')
    if os.path.exists(p):
        out[k] = json.load(open(p))
out['predictions_sha256'] = hashlib.sha256(open(os.path.join(R, 'predictions.md'), 'rb').read()).hexdigest()
out['predictions_sha256_recorded'] = open(os.path.join(R, 'predictions.sha256')).read().split()[0]
json.dump(out, open(os.path.join(R, 'results.json'), 'w'), indent=1)
print('results.json written; predictions hash matches:', out['predictions_sha256'] == out['predictions_sha256_recorded'])
