from __future__ import annotations

import gzip
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


Intron = Tuple[str, int, int]


def open_annotation_text(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_gtf_attrs(attr: str) -> Dict[str, str]:
    attr = attr.strip()
    if not attr:
        return {}
    out: Dict[str, str] = {}
    if "=" in attr and not re.search(r'\b\w+\s+"', attr):
        for part in attr.split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
        return out

    for part in attr.split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r'^(\S+)\s+"([^"]+)"$', part)
        if m:
            out[m.group(1)] = m.group(2)
            continue
        toks = part.split()
        if len(toks) >= 2:
            out[toks[0]] = " ".join(toks[1:]).strip('"')
    return out


def load_transcript_introns(
    gtf_path: str,
    *,
    restrict_gene_ids: Optional[Iterable[str]] = None,
    gene_id_attr: str = "gene_id",
    transcript_id_attr: str = "transcript_id",
    gene_name_attr: str = "gene_name",
) -> Dict[str, Dict[str, object]]:
    wanted = None if restrict_gene_ids is None else {str(g) for g in restrict_gene_ids}
    exon_map: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    meta: Dict[str, Dict[str, str]] = {}

    with open_annotation_text(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, _source, feature, start_s, end_s, _score, strand, _frame, attr_s = fields[:9]
            if feature != "exon":
                continue
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            attrs = parse_gtf_attrs(attr_s)
            gid = attrs.get(gene_id_attr) or attrs.get("gene_id") or attrs.get("ID") or ""
            if not gid:
                continue
            if wanted is not None and gid not in wanted:
                continue
            txid = attrs.get(transcript_id_attr) or attrs.get("transcript_id") or attrs.get("Parent") or ""
            if not txid:
                continue
            gname = attrs.get(gene_name_attr) or attrs.get("gene_name") or attrs.get("Name") or gid
            meta.setdefault(
                gid,
                {
                    "chrom": str(chrom),
                    "strand": str(strand) if strand else ".",
                    "gene_name": str(gname),
                },
            )
            exon_map[gid][txid].append((start, end))

    out: Dict[str, Dict[str, object]] = {}
    for gid, tx_map in exon_map.items():
        transcripts: Dict[str, List[List[object]]] = {}
        for txid, exons in tx_map.items():
            if len(exons) < 2:
                continue
            exons_sorted = sorted(exons, key=lambda x: (x[0], x[1]))
            introns: List[List[object]] = []
            chrom = meta[gid]["chrom"]
            for (_e0s, e0e), (e1s, _e1e) in zip(exons_sorted, exons_sorted[1:]):
                intr_start = int(e0e + 1)
                intr_end = int(e1s - 1)
                if intr_end < intr_start:
                    continue
                introns.append([chrom, intr_start, intr_end])
            if introns:
                transcripts[str(txid)] = introns
        out[gid] = {
            "gene_name": meta.get(gid, {}).get("gene_name", gid),
            "chrom": meta.get(gid, {}).get("chrom", ""),
            "strand": meta.get(gid, {}).get("strand", "."),
            "transcripts": transcripts,
        }
    return out
