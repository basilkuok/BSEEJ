#!/usr/bin/env python3
"""
Prepare per-gene, per-sample junction inputs for BSEEJ (Option 1 workflow):

1) Extract genome-wide junction tables once per sample (e.g. via extract_junctions.sh),
   producing per-sample files:
     - <sample>.junc
     - <sample>.jxs.tsv   (optional, for long-read/multi-junction paths)
2) Split those genome-wide tables into gene-specific folders using a GTF/GFF:
     <out_dir>/<gene_key>/<sample>.junc
     <out_dir>/<gene_key>/<sample>.jxs.tsv   (if any lines map to this gene)

This script does not run inference; it only prepares inputs. Use
`run_bseej_per_gene.py` to run BSEEJ per gene in parallel.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from annotation_utils import load_transcript_introns


def _open_text(path: str):
    """
    Open a UTF-8 text file, supporting optional gzip compression (.gz).
    """
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class GeneInterval:
    chrom: str
    start: int
    end: int
    strand: str
    gene_id: str
    gene_name: str


def _parse_attrs(attr: str) -> Dict[str, str]:
    """
    Parse GTF (key "value";) or GFF3 (key=value;) attributes into a dict.
    """
    attr = attr.strip()
    if not attr:
        return {}
    out: Dict[str, str] = {}
    # Heuristic: GFF3 uses '='.
    if "=" in attr and not re.search(r'\b\w+\s+"', attr):
        for part in attr.split(";"):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
        return out

    # GTF style
    for part in attr.split(";"):
        part = part.strip()
        if not part:
            continue
        # Common formats: key "value" or key value
        m = re.match(r'^(\S+)\s+"([^"]+)"$', part)
        if m:
            out[m.group(1)] = m.group(2)
            continue
        toks = part.split()
        if len(toks) >= 2:
            out[toks[0]] = " ".join(toks[1:]).strip('"')
    return out


def load_gene_intervals(
    gtf_path: str,
    *,
    gene_id_attr: str = "gene_id",
    gene_name_attr: str = "gene_name",
    prefer_gene_feature: bool = True,
) -> Dict[str, GeneInterval]:
    """
    Load gene intervals from a GTF/GFF.

    If `prefer_gene_feature` and "gene" features exist, use those coordinates.
    Otherwise, fall back to min(start), max(end) across exons per gene_id.
    """
    gene_rows: Dict[str, Tuple[str, int, int, str, str]] = {}
    exon_spans: Dict[str, Tuple[str, int, int, str, str]] = {}
    saw_gene_feature = False

    # Support both plain-text and gzipped GTF/GFF.
    with _open_text(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, _source, feature, start_s, end_s, _score, strand, _frame, attr_s = fields[:9]
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            attrs = _parse_attrs(attr_s)
            gid = attrs.get(gene_id_attr) or attrs.get("gene_id") or attrs.get("ID") or ""
            if not gid:
                continue
            gname = attrs.get(gene_name_attr) or attrs.get("gene_name") or attrs.get("Name") or gid
            chrom = str(chrom)
            strand = str(strand) if strand else "."

            if feature == "gene":
                saw_gene_feature = True
                gene_rows[gid] = (chrom, start, end, strand, gname)
            elif feature == "exon":
                prev = exon_spans.get(gid)
                if prev is None:
                    exon_spans[gid] = (chrom, start, end, strand, gname)
                else:
                    pchrom, pstart, pend, pstrand, pgname = prev
                    exon_spans[gid] = (
                        pchrom,
                        min(pstart, start),
                        max(pend, end),
                        pstrand if pstrand != "." else strand,
                        pgname or gname,
                    )

    out: Dict[str, GeneInterval] = {}
    if prefer_gene_feature and saw_gene_feature:
        for gid, (chrom, start, end, strand, gname) in gene_rows.items():
            out[gid] = GeneInterval(chrom=chrom, start=start, end=end, strand=strand, gene_id=gid, gene_name=gname)
        return out

    # Fallback: exon spans
    for gid, (chrom, start, end, strand, gname) in exon_spans.items():
        out[gid] = GeneInterval(chrom=chrom, start=start, end=end, strand=strand, gene_id=gid, gene_name=gname)
    return out


def safe_key(s: str) -> str:
    # Keep it filesystem-friendly and stable.
    s = s.strip()
    s = s.replace(os.sep, "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "gene"


def iter_junc_files(junc_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(junc_dir)):
        if name.endswith(".junc") or name.endswith(".junc.gz"):
            files.append(os.path.join(junc_dir, name))
    return files


def sample_stem_from_junc(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".junc.gz"):
        return name[: -len(".junc.gz")]
    if name.endswith(".junc"):
        return name[: -len(".junc")]
    return os.path.splitext(name)[0]


class _LRUFileCache:
    def __init__(self, max_open: int):
        self.max_open = max(8, int(max_open))
        self._cache: "OrderedDict[str, object]" = OrderedDict()

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_open:
            _key, fh = self._cache.popitem(last=False)
            try:
                fh.close()
            except Exception:
                pass

    def get_append_handle(self, path: str, *, header_line: Optional[str] = None):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        os.makedirs(os.path.dirname(path), exist_ok=True)
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        fh = open(path, "a", encoding="utf-8")
        if is_new and header_line:
            fh.write(header_line.rstrip("\n") + "\n")
        self._cache[path] = fh
        self._evict_if_needed()
        return fh

    def close_all(self) -> None:
        for fh in self._cache.values():
            try:
                fh.close()
            except Exception:
                pass
        self._cache.clear()


def build_bin_index(
    genes: Dict[str, GeneInterval], *, bin_size: int
) -> Dict[str, Dict[int, List[str]]]:
    idx: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for gid, g in genes.items():
        b0 = g.start // bin_size
        b1 = g.end // bin_size
        for b in range(b0, b1 + 1):
            idx[g.chrom][b].append(gid)
    return idx


def candidate_genes_for_interval(
    chrom: str,
    start: int,
    end: int,
    *,
    genes: Dict[str, GeneInterval],
    bin_index: Dict[str, Dict[int, List[str]]],
    bin_size: int,
) -> Iterable[str]:
    if end < start:
        start, end = end, start
    b0 = start // bin_size
    b1 = end // bin_size
    cand: List[str] = []
    chrom_bins = bin_index.get(chrom)
    if not chrom_bins:
        return []
    for b in range(b0, b1 + 1):
        cand.extend(chrom_bins.get(b, []))
    # De-dup while preserving order (small lists, keep it simple).
    seen = set()
    out = []
    for gid in cand:
        if gid in seen:
            continue
        seen.add(gid)
        out.append(gid)
    return out


def introns_from_jx_list(jx_list: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    jx_list = (jx_list or "").strip()
    if not jx_list:
        return out
    for tok in jx_list.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" not in tok:
            continue
        a, b = tok.split("-", 1)
        try:
            s = int(a)
            e = int(b)
        except ValueError:
            continue
        if e < s:
            s, e = e, s
        out.append((s, e))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Split genome-wide per-sample junction tables into per-gene inputs.")
    ap.add_argument("-p", "--junc-dir", required=True, help="Directory containing per-sample *.junc (and optionally *.jxs.tsv).")
    ap.add_argument("--jxs-dir", default=None, help="Optional separate directory for per-sample *.jxs.tsv (defaults to --junc-dir).")
    ap.add_argument("-a", "--annotation", required=True, help="GTF/GFF annotation file (gene_id -> chrom/start/end).")
    ap.add_argument("-o", "--out-dir", required=True, help="Output directory to create per-gene subfolders in.")
    ap.add_argument("--bin-size", type=int, default=1_000_000, help="Bin size for interval indexing (default: 1,000,000).")
    ap.add_argument("--gene-id-attr", default="gene_id", help='Attribute key for gene IDs (default: "gene_id").')
    ap.add_argument("--gene-name-attr", default="gene_name", help='Attribute key for gene names (default: "gene_name").')
    ap.add_argument("--no-gene-feature", action="store_true", help="Do not use 'gene' features even if present; derive spans from exons.")
    ap.add_argument("--genes", nargs="*", default=None, help="Optional list of gene IDs to process (default: all).")
    ap.add_argument("--genes-file", default=None, help="Optional file with one gene ID per line to process.")
    ap.add_argument("--key", choices=["gene_id", "gene_name"], default="gene_id", help="Folder key to use under out-dir.")
    ap.add_argument("--max-open", type=int, default=64, help="Max open output file handles per input sample (default: 64).")
    ap.add_argument("--overwrite", action="store_true", help="Allow writing into an existing --out-dir (files will be appended/created).")
    args = ap.parse_args(argv)

    junc_dir = os.path.abspath(args.junc_dir)
    jxs_dir = os.path.abspath(args.jxs_dir) if args.jxs_dir else junc_dir
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(junc_dir):
        raise SystemExit(f"--junc-dir not found: {junc_dir}")
    if not os.path.isfile(args.annotation):
        raise SystemExit(f"--annotation not found: {args.annotation}")

    if os.path.exists(out_dir) and not args.overwrite:
        # Allow empty dir.
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            raise SystemExit(f"--out-dir exists and is not empty (use --overwrite): {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    genes = load_gene_intervals(
        args.annotation,
        gene_id_attr=args.gene_id_attr,
        gene_name_attr=args.gene_name_attr,
        prefer_gene_feature=not args.no_gene_feature,
    )
    if not genes:
        raise SystemExit(f"No gene intervals loaded from: {args.annotation}")

    wanted: Optional[set] = None
    if args.genes_file:
        with open(args.genes_file, "r") as fh:
            wanted = {ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")}
    if args.genes:
        wanted = set(args.genes) if wanted is None else (wanted & set(args.genes))
    if wanted is not None:
        genes = {gid: g for gid, g in genes.items() if gid in wanted}
        if not genes:
            raise SystemExit("After filtering by --genes/--genes-file, no genes remain.")

    transcript_index = load_transcript_introns(
        args.annotation,
        restrict_gene_ids=genes.keys(),
        gene_id_attr=args.gene_id_attr,
        gene_name_attr=args.gene_name_attr,
    )

    # Map gene_id -> folder key.
    # Ensure folder keys are unique. `gene_id` is usually unique already, but
    # `gene_name` can collide (aliases, case variants, etc.).
    base_key: Dict[str, str] = {}
    for gid, g in genes.items():
        k = g.gene_id if args.key == "gene_id" else g.gene_name
        base_key[gid] = safe_key(k)
    key_counts: Dict[str, int] = defaultdict(int)
    for gid in genes.keys():
        key_counts[base_key[gid]] += 1
    gene_key: Dict[str, str] = {}
    for gid, g in genes.items():
        k = base_key[gid]
        if key_counts.get(k, 0) > 1:
            # Disambiguate deterministically with gene_id.
            k = f"{k}__{safe_key(g.gene_id)}"
        gene_key[gid] = k

    bin_size = int(args.bin_size)
    if bin_size <= 0:
        raise SystemExit("--bin-size must be > 0")
    bin_index = build_bin_index(genes, bin_size=bin_size)

    junc_files = iter_junc_files(junc_dir)
    if not junc_files:
        raise SystemExit(f"No *.junc files found in: {junc_dir}")

    samples = [os.path.basename(sample_stem_from_junc(p)) for p in junc_files]

    # Capture the header line from the first .junc to reuse for outputs.
    header_line = None
    with _open_text(junc_files[0]) as fh:
        header_line = fh.readline().rstrip("\n")
    if not header_line:
        header_line = "chrom\tchromStart\tchromEnd\t.\tscore\tstrand"

    touched_genes = set()
    n_junc_lines = 0
    n_jxs_lines = 0

    for junc_path in junc_files:
        stem = sample_stem_from_junc(junc_path)
        sample_name = os.path.basename(stem)

        cache = _LRUFileCache(max_open=args.max_open)

        # Split .junc
        with _open_text(junc_path) as fh:
            # Skip header
            _ = fh.readline()
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                fields = line.split("\t")
                if len(fields) < 3:
                    continue
                chrom = fields[0]
                try:
                    s = int(fields[1])
                    e = int(fields[2])
                except ValueError:
                    continue
                if e < s:
                    s, e = e, s
                n_junc_lines += 1
                for gid in candidate_genes_for_interval(
                    chrom, s, e, genes=genes, bin_index=bin_index, bin_size=bin_size
                ):
                    g = genes.get(gid)
                    if g is None:
                        continue
                    if g.chrom != chrom:
                        continue
                    if s < g.start or e > g.end:
                        continue
                    out_gene = os.path.join(out_dir, gene_key[gid])
                    out_path = os.path.join(out_gene, f"{sample_name}.junc")
                    out_fh = cache.get_append_handle(out_path, header_line=header_line)
                    out_fh.write(line + "\n")
                    touched_genes.add(gid)

        cache.close_all()

        # Split .jxs.tsv (optional)
        jxs_in = os.path.join(jxs_dir, f"{sample_name}.jxs.tsv")
        if os.path.exists(jxs_in):
            cache = _LRUFileCache(max_open=args.max_open)
            with open(jxs_in, "r", encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    raw = raw.rstrip("\n")
                    if not raw:
                        continue
                    fields = raw.split("\t")
                    if len(fields) < 6:
                        continue
                    chrom1 = fields[0]
                    introns1 = introns_from_jx_list(fields[5])
                    if not introns1:
                        continue

                    chrom2 = None
                    introns2: List[Tuple[int, int]] = []
                    if len(fields) > 12:
                        chrom2 = fields[7]
                        introns2 = introns_from_jx_list(fields[12])

                    # If mates are present and disagree on chrom, treat as chimeric evidence and drop.
                    if chrom2 is not None and chrom2 != chrom1:
                        continue

                    all_introns = introns1 + introns2
                    if not all_introns:
                        continue
                    s_min = min(s for s, _e in all_introns)
                    e_max = max(e for _s, e in all_introns)

                    n_jxs_lines += 1
                    for gid in candidate_genes_for_interval(
                        chrom1, s_min, e_max, genes=genes, bin_index=bin_index, bin_size=bin_size
                    ):
                        g = genes.get(gid)
                        if g is None or g.chrom != chrom1:
                            continue
                        # Keep the line only if *all* introns fall within the gene span.
                        ok = True
                        for s, e in all_introns:
                            if s < g.start or e > g.end:
                                ok = False
                                break
                        if not ok:
                            continue
                        out_gene = os.path.join(out_dir, gene_key[gid])
                        out_path = os.path.join(out_gene, f"{sample_name}.jxs.tsv")
                        out_fh = cache.get_append_handle(out_path, header_line=None)
                        out_fh.write(raw + "\n")
                        touched_genes.add(gid)

            cache.close_all()

    # Write a manifest for reproducibility.
    manifest = {
        "junc_dir": junc_dir,
        "jxs_dir": jxs_dir,
        "samples": samples,
        "n_samples": len(samples),
        "annotation": os.path.abspath(args.annotation),
        "bin_size": bin_size,
        "gene_key_mode": args.key,
        "n_genes_total": len(genes),
        "n_genes_touched": len(touched_genes),
        "n_junc_lines_scanned": n_junc_lines,
        "n_jxs_lines_scanned": n_jxs_lines,
        "gene_id_to_key": {gid: gene_key[gid] for gid in sorted(genes.keys())},
        "genes_touched": sorted(touched_genes),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    for gid in sorted(touched_genes):
        ref_payload = transcript_index.get(
            gid,
            {"gene_name": gid, "chrom": "", "strand": ".", "transcripts": {}},
        )
        gene_dir = os.path.join(out_dir, gene_key[gid])
        os.makedirs(gene_dir, exist_ok=True)
        with open(os.path.join(gene_dir, "reference_introns.json"), "w", encoding="utf-8") as fh:
            json.dump(ref_payload, fh, indent=2, sort_keys=True)

    print(f"[OK] Wrote per-gene inputs under: {out_dir}")
    print(f"[OK] Genes with any evidence: {len(touched_genes)} / {len(genes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
