"""Extract every decimal number from paper/sec_sigma.tex and paper/app_sigma.tex and check it against
the numeric leaves of sigma/results/*.json (incl. derived.json). A number with d decimals matches a
leaf v if |x - v| <= 0.5*10^-d (rounding), or |x - 100 v| <= 0.5*10^-d (percent). Scientific forms
a\\times10^{-k} are checked with the precision of the mantissa. Writes results/verify_tex.json."""
import json, re, glob, sys
R = '../results/'
leaves = []
def walk(o):
    if isinstance(o, dict):
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        leaves.append(float(o))
for fn in glob.glob(R + '*.json'):
    if fn.endswith('results.json') or fn.endswith('verify_tex.json'):
        continue
    walk(json.load(open(fn)))
leaves = sorted(set(leaves))
host_quoted = {'2.63', '0.58', '8.36', '14.13', '12.48', '0.42', '0.86'}   # numbers quoted from the host paper (verdicts/A1)
out = {}
for fn in ('../../paper/sec_sigma.tex', '../../paper/app_sigma.tex'):
    txt = open(fn).read()
    txt = re.sub(r'(?m)^\s*%.*$', '', txt)                     # drop comment lines
    txt = re.sub(r'\\label\{[^}]*\}|\\ref\{[^}]*\}|\\cite[pt]?\{[^}]*\}|\\includegraphics(\[[^]]*\])?\{[^}]*\}', '', txt)
    txt = re.sub(r'\\texttt\{[0-9a-f]{64}\}', '', txt)
    txt = re.sub(r'\[leftmargin=[^]]*\]|\\setlength\{[^}]*\}\{[^}]*\}|width=[0-9.]+\\textwidth', '', txt)
    res = []
    for m in re.finditer(r'(\d+(?:\{,\}\d{3})*\.\d+|\d+(?:\{,\}\d{3})+)(?:\s*\\times\s*10\^\{(-?\d+)\})?', txt):
        raw = m.group(1).replace('{,}', '')
        d = len(raw.split('.')[1]) if '.' in raw else 0
        x = float(raw); tol = 0.5 * 10 ** (-d) + 1e-12
        if m.group(2):
            k = int(m.group(2)); x *= 10 ** k; tol *= 10 ** k
        ok = any(abs(x - v) <= tol or abs(x - 100 * v) <= tol for v in leaves)
        ctx = txt[max(0, m.start() - 50):m.end() + 20].replace('\n', ' ')
        if not ok and raw in host_quoted:
            ok = 'host'
        res.append(dict(number=m.group(0), value=x, ok=ok, context=ctx))
    out[fn] = res
    bad = [r for r in res if r['ok'] is False]
    print(f'{fn}: {len(res)} decimals, {sum(r["ok"] is True for r in res)} matched, '
          f'{sum(r["ok"] == "host" for r in res)} host-quoted, {len(bad)} unmatched')
    for r in bad:
        print('   UNMATCHED', r['number'], '|', r['context'])
json.dump(out, open(R + 'verify_tex.json', 'w'), indent=1)
