#!/usr/bin/env bash
set -euo pipefail

cd /home/itartoussi/classification/project

PY="/home/itartoussi/miniconda3/bin/python"
EPOCHS=200
IMG=256
BATCH=32

DATASETS=(banana bean blackgram coconut corn guava papaya rice sunflower)
VARIANTS=(baseline cbam sam bam c2psa)

mkdir -p runs/logs

run_one () {
  local ds="$1" var="$2"
  local data="datasets_merged/${ds}"
  local out="runs/${ds}"
  local summary="${out}/${var}/summary_${var}.json"
  local log="runs/logs/${ds}_${var}.log"

  if [[ ! -d "$data" ]]; then
    echo "⚠️  Skip ${ds}/${var}: data dir not found -> ${data}"
    return
  fi

  if [[ -f "$summary" ]]; then
    echo "✔️  Skip ${ds}/${var}: summary exists -> ${summary}"
    return
  fi

  echo "▶️  Train ${ds}/${var} ..."
  ${PY} mobilevitv2_${var}_cv.py \
    --data "${data}" \
    --epochs ${EPOCHS} \
    --img-size ${IMG} \
    --batch-size ${BATCH} \
    --out "${out}" 2>&1 | tee "${log}"

  # حدّث الجداول بشكل خفيف
  ${PY} make_tables.py >/dev/null 2>&1 || true
}

for ds in "${DATASETS[@]}"; do
  for var in "${VARIANTS[@]}"; do
    run_one "$ds" "$var"
  done
done

# بعد ما يخلص لفّة التدريب: جداول + PDF Matplotlib
${PY} make_tables.py
${PY} make_pdf_matplotlib.py
