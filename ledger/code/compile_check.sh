#!/usr/bin/env bash
# Standalone compile of sec_deathledger.tex + app_deathledger.tex with the host preamble
# (lines 1-46 of techrxiv.tex, read-only) and the tiers macros (top of sec_tiers.tex).
set -u
S=/tmp/claude-0/-home-claude/0b0a5afe-fb2d-5015-8d90-aaae8c596e01/scratchpad/synth
B=$S/ledger/verify
mkdir -p $B; cd $B
{ sed -n '1,46p' $S/v22/paper/tex/techrxiv.tex
  grep '^\\newcommand{\\tiers' $S/paper/sec_tiers.tex
  echo '\begin{document}'
  echo '\input{sec_deathledger}'
  echo '\appendix'
  echo '\input{app_deathledger}'
  echo '\end{document}'; } > wrap.tex
export TEXINPUTS=$S/paper//:
pdflatex -interaction=nonstopmode wrap.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode wrap.tex >/dev/null 2>&1
echo "errors:"; grep -E '^!' wrap.log | sort | uniq -c
echo "overfull/underfull:"; grep -E 'Overfull|Underfull' wrap.log
echo "undefined references (host/sibling labels expected):"
grep -oE "Reference \`[^']+' on page" wrap.log | sed "s/Reference \`//; s/' on page//" | sort -u | tr '\n' ' '; echo
echo "undefined citations:"; grep -oE "Citation \`[^']+'" wrap.log | sort -u | tr '\n' ' '; echo
pdfinfo wrap.pdf | grep Pages
