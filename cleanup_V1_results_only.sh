#!/usr/bin/env bash
set -euo pipefail

# Cleanup helper for long_read_evaluation_V1.
#
# Goal: keep only
#   - BSEEJ inference outputs (e.g., bseej_results/, bseej_*.gtf)
#   - Evaluation outputs (gffcompare outputs, sqanti outputs)
# and delete intermediate artifacts (prepared per-gene inputs, junction tables, logs, etc.).
#
# Default is DRY RUN. To actually delete, pass --yes.
#
# Usage:
#   ./cleanup_V1_results_only.sh          # dry run
#   ./cleanup_V1_results_only.sh --yes    # delete
#

YES=0
if [[ "${1:-}" == "--yes" ]]; then
  YES=1
fi

ROOT="long_read_evaluation_V1/V1_results/human"
if [[ ! -d "$ROOT" ]]; then
  echo "[ERROR] Not found: $ROOT" >&2
  exit 1
fi

declare -a TO_DELETE=()

should_keep_run_dir() {
  local d="$1"
  # Keep any directory that contains recognizable outputs.
  [[ -d "$d/bseej_results" ]] && return 0
  [[ -d "$d/gffcompare" ]] && return 0
  [[ -d "$d/sqanti" ]] && return 0
  compgen -G "$d"'/bseej_*.gtf' >/dev/null && return 0
  compgen -G "$d"'/*.tmap' >/dev/null && return 0
  compgen -G "$d"'/*.refmap' >/dev/null && return 0
  return 1
}

add_if_exists() {
  local p="$1"
  if [[ -e "$p" ]]; then
    TO_DELETE+=("$p")
  fi
}

while IFS= read -r -d '' run_dir; do
  # If a run directory has no outputs at all, delete the whole run directory.
  if ! should_keep_run_dir "$run_dir"; then
    TO_DELETE+=("$run_dir")
    continue
  fi

  # Otherwise, keep the run dir but remove known intermediates.
  # Some runs use suffixes like *_Test_1; remove those too.
  for p in \
    "$run_dir"/prepared "$run_dir"/prepared_* "$run_dir"/prepared-* \
    "$run_dir"/junc "$run_dir"/junc_* "$run_dir"/junc-* \
    "$run_dir"/logs "$run_dir"/logs_* "$run_dir"/logs-* \
    "$run_dir"/_logs "$run_dir"/_logs_* "$run_dir"/_logs-*; do
    add_if_exists "$p"
  done

  # Large or reproducible inputs to evaluation; keep only evaluation outputs.
  for p in "$run_dir"/reference*.gtf "$run_dir"/query_for_sqanti*.gtf "$run_dir"/bams*.tsv; do
    add_if_exists "$p"
  done
done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -print0)

if [[ ${#TO_DELETE[@]} -eq 0 ]]; then
  echo "[OK] Nothing to delete under $ROOT"
  exit 0
fi

echo "[INFO] Targets (${#TO_DELETE[@]}):"
for p in "${TO_DELETE[@]}"; do
  if [[ -d "$p" ]]; then
    sz=$(du -sh "$p" 2>/dev/null | awk '{print $1}')
    echo "  - DIR  $p  (${sz:-?})"
  else
    sz=$(du -h "$p" 2>/dev/null | awk '{print $1}')
    echo "  - FILE $p  (${sz:-?})"
  fi
done

if [[ $YES -ne 1 ]]; then
  echo "[DRY-RUN] Re-run with --yes to delete."
  exit 0
fi

echo "[INFO] Deleting..."
for p in "${TO_DELETE[@]}"; do
  rm -rf -- "$p"
done
echo "[OK] Cleanup complete."
