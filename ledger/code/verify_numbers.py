"""Check every decimal number in paper/sec_deathledger.tex and paper/app_deathledger.tex against
results/*.json (and the fixed model parameters). A number printed with d decimals matches a value v
if |x - v| <= 0.5*10^-d (also tried: -v, 100*v). 'a\\times10^{-k}' is read as a*10^-k with the
precision of a. Writes results/verify_numbers.txt and lists unmatched numbers."""
import json, os, re, glob

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.normpath(os.path.join(HERE, '..', '..'))
RES = os.path.join(HERE, '..', 'results')
FILES = [os.path.join(S, 'paper', f) for f in ('sec_deathledger.tex', 'app_deathledger.tex')]
PARAMS = {'p': 0.05, 'gamma': 0.95, 'psig': 0.02, 'h': 10, 'c_r': 12.4787, 'phi_tier': 0.02,
          'zeta': 0.1, 'xi': 1.0, 'deltas': [0.0, 0.05, 0.2], 'G': [0.5, 0.7, 0.8, 0.95],
          'omegas': [0.0, 0.5, 1.0, 2.0, 4.0], 'drive half': 0.5, 'battery gammas': [0.8, 0.9, 0.95, 0.99],
          'battery d': 0.8}


def leaves(o, path=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f'{path}/{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f'{path}[{i}]')
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield path, float(o)


vals = []
for f in sorted(glob.glob(os.path.join(RES, '*.json'))):
    if os.path.basename(f) == 'results.json':
        continue
    vals += [(os.path.basename(f) + p, v) for p, v in leaves(json.load(open(f)))]
vals = [('param:' + p, v) for p, v in leaves(PARAMS)] + vals   # parameters claim matches first

num = re.compile(r'(\d+\.\d+)(\s*\\times\s*10\^\{(-?\d+)\})?')
out, bad = [], []
for f in FILES:
    for ln, line in enumerate(open(f), 1):
        if line.lstrip().startswith('%') or 'sha256' in line.lower() or 'arraystretch' in line or 'includegraphics' in line:
            continue
        for m in num.finditer(line):
            a = m.group(1)
            d = len(a.split('.')[1])
            exp = int(m.group(3)) if m.group(3) else 0
            x = float(a) * 10 ** exp
            tol = 0.5 * 10 ** (exp - d) * (1 + 1e-9)
            hit = None
            for tag, fn in (('', lambda v: v), (' (negated)', lambda v: -v), (' (x100)', lambda v: 100 * v)):
                for key, v in vals:
                    if abs(fn(v) - x) <= tol:
                        hit = key + tag
                        break
                if hit:
                    break
            rec = f'{os.path.basename(f)}:{ln}: {m.group(0)} -> {hit or "NOT FOUND"}'
            out.append(rec)
            if not hit:
                bad.append(rec)
txt = '\n'.join(out) + f'\n\nchecked {len(out)} numbers; unmatched {len(bad)}\n' + '\n'.join(bad) + '\n'
open(os.path.join(RES, 'verify_numbers.txt'), 'w').write(txt)
print(f'checked {len(out)} numbers; unmatched {len(bad)}')
for b in bad:
    print('  ', b)
