#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
from typing import Dict, Iterable, Iterator, Optional, Tuple


def _open_text(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="replace")
    return open(path, mode, encoding="utf-8", errors="replace")


_GTF_GENE_ID_RE = re.compile(r'\bgene_id\s+"([^"]+)"')
_GFF_GENE_ID_RE = re.compile(r"(?:^|;)ID=([^;]+)")


def _extract_gene_id(attr_field: str) -> str:
    attr_field = attr_field.strip()
    if not attr_field:
        return ""
    m = _GTF_GENE_ID_RE.search(attr_field)
    if m:
        return m.group(1)
    # Extremely small fallback for GFF3-style attributes.
    m = _GFF_GENE_ID_RE.search(attr_field)
    if m:
        return m.group(1)
    return ""


def load_gene_strand_map(ref_gtf: str) -> Dict[str, str]:
    """
    Build gene_id -> strand (+/-) from a reference GTF.
    Prefer feature == gene; fall back to any feature that carries a gene_id.
    """
    gene_strand: Dict[str, str] = {}
    fallback: Dict[str, str] = {}

    with _open_text(ref_gtf, "rt") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            _chrom, _source, feature, _start, _end, _score, strand, _frame, attrs = fields[:9]
            if strand not in ("+", "-"):
                continue
            gid = _extract_gene_id(attrs)
            if not gid:
                continue
            if feature == "gene":
                gene_strand[gid] = strand
            else:
                if gid not in fallback:
                    fallback[gid] = strand

    # Fill missing from fallback.
    for gid, strand in fallback.items():
        gene_strand.setdefault(gid, strand)
    return gene_strand


def iter_gtf_lines(path: str) -> Iterator[Tuple[str, Optional[list]]]:
    with _open_text(path, "rt") as fh:
        for raw in fh:
            if raw.startswith("#") or raw.strip() == "":
                yield raw, None
                continue
            fields = raw.rstrip("\n").split("\t")
            yield raw, fields


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fill missing strand in a query GTF using gene_id strand from a reference GTF.")
    ap.add_argument("--query-gtf", required=True, help="Query GTF (may be .gz).")
    ap.add_argument("--ref-gtf", required=True, help="Reference GTF (may be .gz).")
    ap.add_argument("--out-gtf", required=True, help="Output GTF path.")
    ap.add_argument("--strict", action="store_true", help="Fail if any feature still lacks strand after fixing.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    gene_strand = load_gene_strand_map(args.ref_gtf)
    if not gene_strand:
        print(f"ERROR: could not load any gene strand from reference: {args.ref_gtf}", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.out_gtf)), exist_ok=True)

    n_total = 0
    n_fixed = 0
    n_missing_gid = 0
    n_unmapped_gid = 0
    n_still_bad = 0

    with open(args.out_gtf, "w", encoding="utf-8") as out_fh:
        for raw, fields in iter_gtf_lines(args.query_gtf):
            if fields is None:
                out_fh.write(raw)
                continue
            if len(fields) < 9:
                out_fh.write(raw)
                continue

            n_total += 1
            strand = fields[6]
            if strand in ("+", "-"):
                out_fh.write("\t".join(fields) + "\n")
                continue

            gid = _extract_gene_id(fields[8])
            if not gid:
                n_missing_gid += 1
                n_still_bad += 1
                out_fh.write("\t".join(fields) + "\n")
                continue

            new_strand = gene_strand.get(gid)
            if new_strand in ("+", "-"):
                fields[6] = new_strand
                n_fixed += 1
                out_fh.write("\t".join(fields) + "\n")
                continue

            n_unmapped_gid += 1
            n_still_bad += 1
            out_fh.write("\t".join(fields) + "\n")

    msg = (
        f"[OK] strand-fix: total_features={n_total} fixed={n_fixed} "
        f"missing_gene_id={n_missing_gid} unmapped_gene_id={n_unmapped_gid} "
        f"still_non_stranded={n_still_bad}"
    )
    print(msg, file=sys.stderr)

    if args.strict and n_still_bad > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

