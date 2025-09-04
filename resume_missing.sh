#!/usr/bin/env bash
set -euo pipefail

PY="/home/itartoussi/miniconda3/bin/python"
EPOCHS=200

for d in papaya rice sunflower; do
  data="datasets_merged/$d"
  for v in baseline cbam sam bam c2psa; do
    f="runs/$d/$v/summary_${v}.json"
    if [[ -f "$f" ]]; then
      echo "✔️  Skip $d/$v (found $f)"
    else
      echo "▶️  Train $d/$v ..."
      $PY mobilevitv2_${v}_cv.py --data "$data" --epochs $EPOCHS --out runs/$d
    fi
  done
done
