#!/usr/bin/env bash
set -euo pipefail

# Run BSEEJ on the 16 minimap2-aligned BAMs in long_read_evaluation_V2.
# Usage (from repo root): bash run_V2.sh

V2="long_read_evaluation_V2"
BAMDIR="$V2/minimap2_alignments"
ACCESSIONS="$V2/samples/accessions_human.txt"

# Ensembl GRCh38.105 annotation (GTF/GFF) for gene spans used by prepare_gene_inputs.py.
GTF="$V2/V2_reference_genome/human/Homo_sapiens.GRCh38.105.gtf.gz"

JUNCDIR="$V2/junctions_from_bam"
PREPARED="$V2/prepared_grch38_105"
RESULTS="$V2/bseej_results_k5"

# Adjust if you want different parallelism.
JOBS="${JOBS:-8}"

# Avoid matplotlib cache permission warnings in some environments.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/.mpl}"

if [[ ! -d "$BAMDIR" ]]; then
  echo "[ERROR] BAM directory not found: $BAMDIR" >&2
  exit 1
fi
if [[ ! -f "$ACCESSIONS" ]]; then
  echo "[ERROR] Accession list not found: $ACCESSIONS" >&2
  exit 1
fi
if [[ ! -f "$GTF" ]]; then
  echo "[ERROR] GTF/GFF not found: $GTF" >&2
  exit 1
fi

mkdir -p "$JUNCDIR" "$PREPARED" "$RESULTS"

echo "[INFO] Extracting junction tables from BAMs..."
while IFS= read -r acc; do
  [[ -z "$acc" ]] && continue
  bam="$BAMDIR/$acc.bam"
  if [[ ! -f "$bam" ]]; then
    echo "[ERROR] Missing BAM: $bam" >&2
    exit 1
  fi
  python build_junctions_from_bam.py "$bam" "$JUNCDIR/$acc"
done < "$ACCESSIONS"

echo "[INFO] Preparing per-gene inputs + running BSEEJ (k=5)..."
python run_bseej_per_gene.py \
  --prepare -p "$JUNCDIR" -a "$GTF" \
  --prepared-dir "$PREPARED" \
  --results-dir "$RESULTS" \
  --jobs "$JOBS" \
  -- -k 5 --long_mode 1

echo "[OK] Done. Results: $RESULTS"
