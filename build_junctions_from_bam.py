#!/usr/bin/env python

"""
Lightweight junction extractor for JARVIS, implemented in pure Python + samtools.

This script builds the three files JARVIS needs from a single sorted BAM:

  1) <prefix>.all_jxs.tsv
       One line per intron excision (per read), with columns:
         qname, chrom, intron_start, intron_end, strand, CIGAR, is_unique
       This is the same input format expected by
       megadepth-master/junctions/process_jx_output.sh.

  2) <prefix>.all_jxs.tsv.sjout  (via process_jx_output.sh)
       STAR SJ.out-like aggregate per intron.

  3) <prefix>.bam.junc
       Final 6-column junction file used by JARVIS:
         chrom, chromStart, chromEnd, ".", score, "."
       where score = unique_count + multi_count.

  4) <prefix>.jxs.tsv
       Co-occurrence file: one line per multi-junction read with a list of
       "start-end" coordinate pairs in the 6th field. JARVIS uses only this
       6th field (and, for paired-end, an optional 13th field) to build
       multi-junction path nodes and the intron co-occurrence matrix.

Usage
-----
    python build_junctions_from_bam.py \
        /path/to/sample.sorted.bam \
        /path/to/output_prefix

For the Lexogen dataset in this repo you would run, from JARVIS_final:

    python build_junctions_from_bam.py \
        long_read_dataset/Lexogen.SIRVs.Set4.ONT_cDNA.R10.4.sorted.bam \
        long_read_dataset/junc_lexogen/Lexogen.SIRVs.Set4.ONT_cDNA.R10.4.sorted.bam

This script assumes:
  - samtools is available on PATH (installed in the bseej_env);
  - the BAM is coordinate-sorted and indexed (recommended, not strictly required for samtools view);
  - we can ignore multi-mapping distinctions for JARVIS purposes
    (we treat all alignments we keep as "unique" for counting).
"""

import argparse
import os
import re
import subprocess
import sys
from typing import List, Tuple


_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")


def parse_cigar(cigar: str) -> List[Tuple[int, int]]:
    """
    Parse a CIGAR string and return a list of intron intervals
    (start, end) in 1-based reference coordinates.

    We treat 'N' operations as introns. Other operations:
      - M, D, N, =, X consume reference positions
      - I, S, H, P do not.
    """
    introns: List[Tuple[int, int]] = []
    if cigar == "*" or "N" not in cigar:
        return introns

    ref_pos = 0  # We'll set the initial ref_pos in the caller.
    # We don't know the absolute starting POS here, so the caller should
    # pass POS and we add offsets based on these operations.
    # This helper only interprets relative positions, so we return offsets.
    # To keep things simple, we will handle absolute positions in the caller
    # and use this function only to detect the lengths of N operations.
    raise NotImplementedError


def iter_bam_alignments(
    bam_path: str,
    *,
    keep_supplementary: bool = False,
    region: str | None = None,
    max_alignments: int | None = None,
    strip_chr: bool = False,
):
    """
    Stream alignments from samtools view -h <bam_path>.

    Yields (qname, flag, rname, pos, mapq, cigar, strand_char) for primary,
    mapped alignments, using filtering semantics close to Megadepth's
    defaults:
      - drop unmapped (0x4) and secondary (0x100) alignments;
      - drop supplementary (0x800) by default to avoid inflating junction evidence
        from split/chimeric records (use --keep-supplementary to override).
    """
    cmd = ["samtools", "view", "-h", bam_path]
    if region:
        cmd.append(region)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert proc.stdout is not None
    n_seen = 0
    for line in proc.stdout:
        if not line or line.startswith("@"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 6:
            continue
        qname = fields[0]
        try:
            flag = int(fields[1])
        except ValueError:
            continue
        rname = fields[2]
        if strip_chr and rname.startswith("chr"):
            rname = rname[3:]
        try:
            pos = int(fields[3])
        except ValueError:
            continue
        try:
            mapq = int(fields[4])
        except ValueError:
            mapq = 0
        cigar = fields[5]

        # XS:A:<+|-> tag encodes transcript strand for many splice-aware
        # aligners (e.g. STAR). We propagate it so junction lines can carry
        # strand information when available.
        strand_char = "0"
        for tag in fields[11:]:
            if tag.startswith("XS:A:") and len(tag) >= 6:
                xs = tag[5]
                if xs in ("+", "-"):
                    strand_char = xs
                break

        # Skip unmapped and secondary alignments (Megadepth default filter-out).
        if flag & 0x4:
            continue  # unmapped
        if flag & 0x100:
            continue  # secondary
        if (not keep_supplementary) and (flag & 0x800):
            continue  # supplementary
        yield qname, flag, rname, pos, mapq, cigar, strand_char

        n_seen += 1
        if max_alignments is not None and n_seen >= int(max_alignments):
            break

    proc.stdout.close()
    proc.wait()


def extract_introns_from_cigar(
    start_pos: int, cigar: str
) -> List[Tuple[int, int]]:
    """
    Given a 1-based alignment start position and a CIGAR, return
    a list of absolute intron intervals (start, end) in 1-based coordinates.
    """
    introns: List[Tuple[int, int]] = []
    if cigar == "*" or "N" not in cigar:
        return introns

    # Parse CIGAR into (length, op) pairs so we can inspect flanking
    # segments around each N operation.
    ops = _CIGAR_RE.findall(cigar)

    # Regtools-style filters:
    #   - minimum anchor of 6 matched bases (M/= /X) on each side of N
    #   - intron length between 50 and 500000 bp.
    MIN_ANCHOR = 6
    MIN_INTRON_LEN = 50
    MAX_INTRON_LEN = 500000

    ref_pos = start_pos
    for idx, (length_str, op) in enumerate(ops):
        length = int(length_str)
        if op in ("M", "D", "N", "=", "X"):
            if op == "N":
                intron_len = length
                intron_start = ref_pos
                intron_end = ref_pos + intron_len - 1

                if MIN_INTRON_LEN <= intron_len <= MAX_INTRON_LEN:
                    # Left anchor: contiguous M/=/X run immediately before N.
                    left_anchor = 0
                    j = idx - 1
                    while j >= 0:
                        prev_len = int(ops[j][0])
                        prev_op = ops[j][1]
                        if prev_op in ("M", "=", "X"):
                            left_anchor += prev_len
                            j -= 1
                        else:
                            break

                    # Right anchor: contiguous M/=/X run immediately after N.
                    right_anchor = 0
                    j = idx + 1
                    while j < len(ops):
                        next_len = int(ops[j][0])
                        next_op = ops[j][1]
                        if next_op in ("M", "=", "X"):
                            right_anchor += int(ops[j][0])
                            j += 1
                        else:
                            break

                    if left_anchor >= MIN_ANCHOR and right_anchor >= MIN_ANCHOR:
                        introns.append((intron_start, intron_end))

                ref_pos += length
            else:
                ref_pos += length
        # I, S, H, P do not advance ref_pos
    return introns


def build_all_jxs_and_jxs(
    bam_path: str,
    prefix: str,
    *,
    keep_supplementary: bool = False,
    region: str | None = None,
    max_alignments: int | None = None,
    strip_chr: bool = False,
) -> Tuple[str, str]:
    """
    Build <prefix>.all_jxs.tsv and <prefix>.jxs.tsv from a BAM file.

    Returns (all_jxs_path, jxs_path).
    """
    all_jxs_path = prefix + ".all_jxs.tsv"
    jxs_path = prefix + ".jxs.tsv"

    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    with open(all_jxs_path, "w") as f_all, open(jxs_path, "w") as f_jxs:
        for qname, flag, rname, pos, mapq, cigar, strand_char in iter_bam_alignments(
            bam_path,
            keep_supplementary=keep_supplementary,
            region=region,
            max_alignments=max_alignments,
            strip_chr=strip_chr,
        ):
            introns = extract_introns_from_cigar(pos, cigar)
            if not introns:
                continue

            # Approximate Megadepth's default uniqueness heuristic:
            #   unique_aln = (MAPQ >= 10) when no explicit threshold is set.
            # process_jx_output.sh only uses is_unique to split counts into
            # "unique" vs "multi", which JARVIS later sums, but we keep the
            # distinction for fidelity.
            is_unique = "1" if mapq >= 10 else "0"

            # 1) Emit one line per intron for all_jxs.tsv
            for s, e in introns:
                f_all.write(
                    f"{qname}\t{rname}\t{s}\t{e}\t{strand_char}\t{cigar}\t{is_unique}\n"
                )

            # 2) Emit a single co-occurrence line per read with >=2 introns
            if len(introns) >= 2:
                intron_tokens = [f"{s}-{e}" for (s, e) in introns]
                intron_list = ",".join(intron_tokens)
                # Minimal header: chrom, read_start, strand, tlen (0), cigar, intron_list, is_unique
                # JARVIS only uses the 6th field (intron_list).
                f_jxs.write(
                    f"{rname}\t{pos}\t{strand_char}\t0\t{cigar}\t{intron_list}\t{is_unique}\n"
                )

    return all_jxs_path, jxs_path


def build_junc_from_sjout(sjout_path: str, junc_path: str) -> None:
    """
    Convert a STAR-style SJ.out-like file (produced by process_jx_output.sh)
    into the 6-column .junc format expected by JARVIS:

      chrom  chromStart  chromEnd  .  score  strand

    where score = unique_count + multi_count, and strand is set to ".".
    """
    with open(sjout_path, "r") as fh_in, open(junc_path, "w") as fh_out:
        fh_out.write("chrom\tchromStart\tchromEnd\t.\tscore\tstrand\n")
        for line in fh_in:
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 8:
                continue
            chrom = fields[0]
            try:
                chrom_start = int(fields[1])
                chrom_end = int(fields[2])
                uniq = int(fields[6])
                multi = int(fields[7])
            except ValueError:
                continue
            score = uniq + multi
            fh_out.write(
                f"{chrom}\t{chrom_start}\t{chrom_end}\t.\t{score}\t.\n"
            )


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Extract junction tables from a BAM using samtools (no megadepth binary required)."
    )
    ap.add_argument("bam_file", help="Input BAM (recommended: coordinate-sorted).")
    ap.add_argument(
        "output_prefix",
        help=(
            "Output prefix. Files will be written as "
            "<prefix>.all_jxs.tsv, <prefix>.all_jxs.tsv.sjout, <prefix>.jxs.tsv, <prefix>.junc"
        ),
    )
    ap.add_argument(
        "--keep-supplementary",
        action="store_true",
        help="Keep supplementary (0x800) alignments (default: drop to avoid inflated evidence).",
    )
    ap.add_argument(
        "--region",
        default=None,
        help="Optional samtools region (e.g. 'chr1:1-2000000') to limit extraction.",
    )
    ap.add_argument(
        "--max-alignments",
        type=int,
        default=None,
        help="Optional cap on number of alignments processed (for quick smoke tests).",
    )
    ap.add_argument(
        "--strip-chr",
        action="store_true",
        help="Strip a leading 'chr' from reference names in outputs (helps match Ensembl GTF).",
    )
    args = ap.parse_args(argv[1:])

    bam_path = args.bam_file
    prefix = args.output_prefix

    if not os.path.exists(bam_path):
        sys.stderr.write(f"ERROR: BAM file '{bam_path}' not found.\n")
        return 1

    # 1) Build all_jxs and jxs from BAM.
    all_jxs_path, jxs_path = build_all_jxs_and_jxs(
        bam_path,
        prefix,
        keep_supplementary=bool(args.keep_supplementary),
        region=args.region,
        max_alignments=args.max_alignments,
        strip_chr=bool(args.strip_chr),
    )
    sys.stderr.write(f"[INFO] Wrote {all_jxs_path}\n")
    sys.stderr.write(f"[INFO] Wrote {jxs_path}\n")

    # 2) Call the existing process_jx_output.sh to aggregate introns.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    process_script = os.path.join(
        script_dir, "megadepth-master", "junctions", "process_jx_output.sh"
    )
    if not os.path.exists(process_script):
        sys.stderr.write(
            f"ERROR: process_jx_output.sh not found at '{process_script}'.\n"
        )
        return 1

    try:
        subprocess.check_call(["bash", process_script, all_jxs_path])
    except (OSError, subprocess.CalledProcessError) as exc:
        sys.stderr.write(f"ERROR: Failed to run process_jx_output.sh: {exc}\n")
        return 1

    sjout_path = all_jxs_path + ".sjout"
    if not os.path.exists(sjout_path):
        sys.stderr.write(
            f"ERROR: Expected SJ.out-like file '{sjout_path}' not found.\n"
        )
        return 1

    # 3) Build the .junc file from SJ.out-like aggregation.
    junc_path = prefix + ".junc"
    build_junc_from_sjout(sjout_path, junc_path)
    sys.stderr.write(f"[INFO] Wrote {junc_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
