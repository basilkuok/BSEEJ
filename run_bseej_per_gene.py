#!/usr/bin/env python3
"""
End-to-end Option 1 runner:

  (A) prepare_gene_inputs.py  (split genome-wide junction tables by gene)
  (B) run bseej.py per gene in parallel (process-level), building one interval
      graph per gene across all samples.

This script intentionally shells out to bseej.py in separate processes to keep
memory isolated per gene and to avoid cross-talk between runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple


def _resolve_executable(cmd: str) -> str:
    """
    Resolve an executable name to an absolute path when possible, while keeping
    explicit paths unchanged.
    """
    if not cmd:
        return cmd
    if os.sep in cmd or (os.altsep and os.altsep in cmd):
        return os.path.abspath(cmd)
    found = shutil.which(cmd)
    return found or cmd


def _load_manifest(prepared_dir: str) -> Dict:
    path = os.path.join(prepared_dir, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"manifest.json not found under: {prepared_dir}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _discover_gene_dirs(prepared_dir: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(prepared_dir)):
        if name in (".", "..", "manifest.json"):
            continue
        path = os.path.join(prepared_dir, name)
        if not os.path.isdir(path):
            continue
        # Only run genes that have at least one .junc file.
        has_junc = any(fn.endswith(".junc") or fn.endswith(".junc.gz") for fn in os.listdir(path))
        if has_junc:
            out.append(name)
    return out


def _run_one(
    gene_key: str,
    gene_id: str,
    *,
    prepared_dir: str,
    results_dir: str,
    bseej_script: str,
    python_exe: str,
    extra_args: List[str],
    mplconfigdir: str,
) -> Tuple[str, int]:
    gene_in = os.path.join(prepared_dir, gene_key)
    log_dir = os.path.join(results_dir, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    # gene_id can contain characters that are awkward on some filesystems; keep logs safe.
    safe_gene_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(gene_id))
    safe_gene_id = safe_gene_id.strip("_") or "gene"
    log_path = os.path.join(log_dir, f"{safe_gene_id}.log")

    cmd = [python_exe, bseej_script, "-p", gene_in, "-g", gene_id, "-o", results_dir]
    cmd.extend(extra_args)

    env = os.environ.copy()
    if mplconfigdir:
        env["MPLCONFIGDIR"] = mplconfigdir

    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
    return gene_id, int(proc.returncode)

def _parse_flag_value(args: List[str], *names: str) -> Optional[str]:
    for i, tok in enumerate(args):
        if tok in names:
            if i + 1 < len(args):
                return args[i + 1]
            return None
    return None


def _parse_int_flag(args: List[str], default: int, *names: str) -> int:
    v = _parse_flag_value(args, *names)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _infer_method_label(mode: int) -> str:
    return {1: "gibbs", 2: "cavi", 3: "svi"}.get(int(mode), "cavi")

def _load_gene_filter(*, genes: Optional[List[str]], genes_file: Optional[str]) -> Optional[List[str]]:
    """
    Load an optional list of gene IDs to restrict inference.
    Lines starting with '#' are treated as comments.
    """
    out: List[str] = []
    if genes:
        for g in genes:
            g = (g or "").strip()
            if g:
                out.append(g)
    if genes_file:
        try:
            with open(genes_file, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    out.append(s.split()[0])
        except FileNotFoundError:
            raise SystemExit(f"--genes-file not found: {genes_file}")
    # De-duplicate but keep order
    if not out:
        return None
    seen = set()
    uniq: List[str] = []
    for g in out:
        if g in seen:
            continue
        seen.add(g)
        uniq.append(g)
    return uniq


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare per-gene inputs and run BSEEJ per gene in parallel.")
    ap.add_argument("--prepared-dir", required=True, help="Directory to write/read per-gene inputs (will contain manifest.json).")
    ap.add_argument("--results-dir", required=True, help="BSEEJ results output directory (passed as -o).")
    ap.add_argument("--bseej-script", default=os.path.join(os.path.dirname(__file__), "bseej.py"), help="Path to bseej.py.")
    ap.add_argument("--python", default=sys.executable, help="Python executable to run bseej.py (default: current).")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) // 2), help="Parallel jobs (default: half CPUs).")
    ap.add_argument("--mplconfigdir", default="/tmp/.mpl", help="MPLCONFIGDIR for bseej subprocesses (default: /tmp/.mpl).")
    ap.add_argument("--dry-run", action="store_true", help="Print planned per-gene runs but do not execute bseej.py.")

    ap.add_argument("--prepare", action="store_true", help="Run prepare_gene_inputs.py before running inference.")
    ap.add_argument("-p", "--junc-dir", default=None, help="(with --prepare) directory containing per-sample *.junc/*.jxs.tsv.")
    ap.add_argument("-a", "--annotation", default=None, help="(with --prepare) GTF/GFF annotation file.")
    ap.add_argument("--jxs-dir", default=None, help="(with --prepare) optional separate directory for per-sample *.jxs.tsv.")
    ap.add_argument("--key", choices=["gene_id", "gene_name"], default="gene_id", help="(with --prepare) folder key to use.")
    ap.add_argument("--genes", nargs="*", default=None, help="Restrict to these gene IDs.")
    ap.add_argument("--genes-file", default=None, help="File with one gene ID per line (optional '#'-comments).")
    ap.add_argument("--bin-size", type=int, default=1_000_000, help="(with --prepare) bin size for interval indexing.")
    ap.add_argument("--overwrite-prepared", action="store_true", help="(with --prepare) allow writing into existing prepared-dir.")

    ap.add_argument("bseej_args", nargs=argparse.REMAINDER, help="Extra args passed to bseej.py after '--'.")
    args = ap.parse_args(argv)

    prepared_dir = os.path.abspath(args.prepared_dir)
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(prepared_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    extra_args = list(args.bseej_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    if args.prepare:
        if not args.junc_dir or not args.annotation:
            raise SystemExit("--prepare requires --junc-dir and --annotation")
        # Import locally to keep this runner lightweight.
        from prepare_gene_inputs import main as prepare_main

        prep_argv = [
            "--junc-dir",
            args.junc_dir,
            "--annotation",
            args.annotation,
            "--out-dir",
            prepared_dir,
            "--bin-size",
            str(args.bin_size),
            "--key",
            args.key,
        ]
        if args.jxs_dir:
            prep_argv += ["--jxs-dir", args.jxs_dir]
        if args.genes:
            prep_argv += ["--genes"] + list(args.genes)
        if args.genes_file:
            prep_argv += ["--genes-file", args.genes_file]
        if args.overwrite_prepared:
            prep_argv += ["--overwrite"]

        rc = int(prepare_main(prep_argv))
        if rc != 0:
            return rc

    manifest = _load_manifest(prepared_dir)
    gid_to_key = manifest.get("gene_id_to_key", {}) or {}
    key_to_gid = {v: k for k, v in gid_to_key.items()}

    gene_keys = _discover_gene_dirs(prepared_dir)
    if not gene_keys:
        raise SystemExit(f"No gene input folders with *.junc found under: {prepared_dir}")

    # Map folder keys back to gene IDs (fallback to key itself).
    runs: List[Tuple[str, str]] = []
    for key in gene_keys:
        gid = key_to_gid.get(key, key)
        runs.append((key, gid))

    # Optional inference-time filtering by gene ID list.
    gene_filter = _load_gene_filter(genes=args.genes, genes_file=args.genes_file)
    if gene_filter:
        want = set(gene_filter)
        filtered: List[Tuple[str, str]] = []
        present = set()
        for gene_key, gene_id in runs:
            if gene_id in want or gene_key in want:
                filtered.append((gene_key, gene_id))
                present.add(gene_id)
                present.add(gene_key)
        missing = [g for g in gene_filter if g not in present]
        if missing:
            preview = ", ".join(missing[:10])
            more = "" if len(missing) <= 10 else f" (+{len(missing)-10} more)"
            print(f"[WARN] {len(missing)} requested genes not found under prepared-dir (skipping): {preview}{more}")
        runs = filtered

    method_label = _infer_method_label(_parse_int_flag(extra_args, 2, "-m", "--mode"))
    k = _parse_int_flag(extra_args, 1, "-k", "--n_cluster")
    idx_suffix = str(_parse_flag_value(extra_args, "-identifier", "--idx") or "")

    jobs = max(1, int(args.jobs))
    print(f"[INFO] Running {len(runs)} genes with jobs={jobs}")
    print(f"[INFO] Logs: {os.path.join(results_dir, '_logs')}")

    if args.dry_run:
        py_exe = _resolve_executable(args.python)
        bseej_script = os.path.abspath(args.bseej_script)
        for gene_key, gene_id in runs[:50]:
            gene_in = os.path.join(prepared_dir, gene_key)
            cmd = [py_exe, bseej_script, "-p", gene_in, "-g", gene_id, "-o", results_dir] + extra_args
            print(" ".join(cmd))
        if len(runs) > 50:
            print(f"[INFO] (Showing first 50 of {len(runs)} genes)")
        return 0

    failures: List[Tuple[str, int]] = []

    # Some environments disallow creating POSIX semaphores (used by
    # ProcessPoolExecutor). For jobs=1, run sequentially without
    # multiprocessing to avoid PermissionError on SemLock.
    if jobs == 1:
        for gene_key, gene_id in runs:
            gid, rc = _run_one(
                gene_key,
                gene_id,
                prepared_dir=prepared_dir,
                results_dir=results_dir,
                bseej_script=os.path.abspath(args.bseej_script),
                python_exe=_resolve_executable(args.python),
                extra_args=extra_args,
                mplconfigdir=args.mplconfigdir,
            )
            if rc != 0:
                failures.append((gid, rc))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = []
            for gene_key, gene_id in runs:
                futs.append(
                    ex.submit(
                        _run_one,
                        gene_key,
                        gene_id,
                        prepared_dir=prepared_dir,
                        results_dir=results_dir,
                        bseej_script=os.path.abspath(args.bseej_script),
                        python_exe=_resolve_executable(args.python),
                        extra_args=extra_args,
                        mplconfigdir=args.mplconfigdir,
                    )
                )
            for fut in as_completed(futs):
                gid, rc = fut.result()
                if rc != 0:
                    failures.append((gid, rc))

    if failures:
        print("[ERROR] Some genes failed:")
        for gid, rc in failures[:50]:
            print(f"  - {gid}: exit={rc}")
        return 1

    print("[OK] All genes finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
