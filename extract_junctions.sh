#!/usr/bin/env bash

set -euo pipefail

# Initialize variables
input=""
output=""
fasta=""
min_cov=0  # Portcullis --min_cov; 0 means "disabled"

# Function to show usage
usage() {
    echo "Usage: $0 -i <input_bam_dir> -o <output_junc_dir> [-f <reference_fasta>] [-m <min_cov>]" 1>&2
    echo "  <input_bam_dir>: directory containing per-gene BAM files" 1>&2
    echo "  <output_junc_dir>: directory to write per-sample .junc, .jxs.tsv, and .all_jxs.tsv files" 1>&2
    echo "  -f <reference_fasta>: reference FASTA for Portcullis (required if -m > 0 and Portcullis is used)" 1>&2
    echo "  -m <min_cov>: Portcullis coverage threshold (default = 0, i.e., Portcullis filtering disabled)" 1>&2
    exit 1
}

# Parse command-line options
while getopts ":i:o:f:m:" opt; do
    case ${opt} in
        i )
            input=$OPTARG
            ;;
        o )
            output=$OPTARG
            ;;
        f )
            fasta=$OPTARG
            ;;
        m )
            min_cov=$OPTARG
            ;;
        \? )
            echo "Invalid Option: -$OPTARG" 1>&2
            usage
            ;;
        : )
            echo "Invalid Option: -$OPTARG requires an argument" 1>&2
            usage
            ;;
    esac
done
shift $((OPTIND -1))

# Check if input and output were provided
if [ -z "$input" ] || [ -z "$output" ]; then
    usage
fi

bam_path=$input
junc_path=$output

# Resolve script directory so we can find megadepth and helper scripts
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allow overriding megadepth binary via MEGADEPTH_BIN, otherwise use repo-local wrapper
megadepth_bin="${MEGADEPTH_BIN:-${script_dir}/megadepth-master/megadepth}"
process_jx_script="${script_dir}/megadepth-master/junctions/process_jx_output.sh"
#
# By default we exclude supplementary alignments (0x800) in addition to the
# default Megadepth filter-out mask (unmapped 0x4 + secondary 0x100).
# Rationale: supplementary (split/chimeric) records can cause the same read to
# contribute junctions multiple times.
megadepth_filter_out_mask="${MEGADEPTH_FILTER_OUT_MASK:-2308}"

if [ ! -x "$megadepth_bin" ]; then
    echo "ERROR: megadepth binary not found or not executable at '$megadepth_bin'." 1>&2
    echo "       Build it with 'cd megadepth-master && ./build_megadepth.sh' or set MEGADEPTH_BIN." 1>&2
    exit 1
fi

if [ ! -x "$process_jx_script" ]; then
    echo "ERROR: helper script '$process_jx_script' not found or not executable." 1>&2
    exit 1
fi

mkdir -p "$junc_path"

total_files=$(find -P "$bam_path" -maxdepth 1 -type f -name '*.bam' | wc -l | awk '{print $1}')
echo "Number of bam files: $total_files"

if [ "$total_files" -eq 0 ]; then
    echo "No BAM files found in '$bam_path'." 1>&2
    exit 1
fi

# Decide whether Portcullis-based filtering is enabled
portcullis_enabled=false
min_cov_num=0
if [ -n "$min_cov" ]; then
    # best-effort numeric cast
    min_cov_num=$(printf "%d" "$min_cov" 2>/dev/null || echo 0)
fi
if [ "$min_cov_num" -gt 0 ]; then
    if command -v portcullis >/dev/null 2>&1 && [ -n "$fasta" ]; then
        portcullis_enabled=true
    else
        echo "Portcullis filtering not installed, min_cov not enforced" 1>&2
        portcullis_enabled=false
        min_cov_num=0
    fi
fi

threads="${PORTCULLIS_THREADS:-4}"

cd "$bam_path"
i=0
for bam_file in *.bam; do
    # Skip if the glob didn't match any files
    [ -f "$bam_file" ] || continue
    i=$((i+1))
    echo "[INFO] Processing (${i}/${total_files}): ${bam_file}"

    # Use the BAM filename (including .bam) as the prefix so outputs stay per-sample
    prefix="${junc_path}/${bam_file}"

    # 1) Run megadepth to get:
    #    - all junctions per read:  <prefix>.all_jxs.tsv
    #    - co-occurring junctions: <prefix>.jxs.tsv
    "$megadepth_bin" "$bam_file" \
        --all-junctions \
        --junctions \
        --filter-out "$megadepth_filter_out_mask" \
        --prefix "$prefix"

    # 2) Aggregate --all-junctions output into a STAR-style SJ.out-like
    #    file with unique/multi counts per intron:
    #      <prefix>.all_jxs.tsv.sjout
    "$process_jx_script" "${prefix}.all_jxs.tsv"

    sjout="${prefix}.all_jxs.tsv.sjout"
    junc="${junc_path}/${bam_file}.junc"

    if [ ! -s "$sjout" ]; then
        # No junctions for this sample; still create a header-only .junc
        {
            printf "chrom\tchromStart\tchromEnd\t.\tscore\tstrand\n"
        } > "$junc"
        echo "[INFO] No junctions found for ${bam_file}; wrote header-only ${junc}"
    else
        # 3) Convert SJ.out-like file into the 6-column .junc format expected by JARVIS:
        #    chrom, chromStart, chromEnd, ".", score, strand
        #    where score = unique_count + multi_count (cols 7+8 in SJ.out),
        #    and strand is left as "." (JARVIS does not use strand downstream).
        {
            printf "chrom\tchromStart\tchromEnd\t.\tscore\tstrand\n"
            awk 'BEGIN{OFS="\t"} {score=$7+$8; strand="."; print $1,$2,$3,".",score,strand}' "$sjout"
        } > "$junc"
    fi

    # 4) Optional Portcullis-based high-stringency filtering:
    #    If enabled, run Portcullis on the BAM and intersect its filtered
    #    junctions with the Megadepth-derived .junc, keeping only junctions
    #    that overlap by at least 90%.
    if $portcullis_enabled && [ -s "$junc" ]; then
        port_base="${junc_path}/${bam_file%.bam}_portcullis"
        prep_dir="${port_base}/1-prep"
        junc_dir="${port_base}/2-junc"
        filt_dir="${port_base}/3-filt"
        mkdir -p "$prep_dir" "$junc_dir" "$filt_dir"

        portcullis prep -t "$threads" -v --force \
            -o "$prep_dir" \
            "$fasta" \
            "${bam_file}"

        portcullis junc -t "$threads" -v \
            -o "${junc_dir}/portcullis_all" \
            --intron_gff "$prep_dir"

        portcullis filt -t "$threads" -v -n \
            --max_length 500000 \
            --min_cov "$min_cov_num" \
            -o "${filt_dir}/portcullis_filtered" \
            --intron_gff "$prep_dir" \
            "${junc_dir}/portcullis_all.junctions.tab"

        port_filt="${filt_dir}/portcullis_filtered.junctions.tab"

        if [ -s "$port_filt" ]; then
            # Intersect Megadepth .junc with Portcullis-filtered junctions.
            # Keep a Megadepth junction (chromStart, chromEnd) if there exists
            # a Portcullis junction on the same chromosome with at least 90%
            # overlap of the Megadepth intron length.
            tmp_junc="${junc}.tmp"
            awk -v OFS="\t" '
                NR==FNR {
                    # Portcullis filtered junctions: assume first 3 columns are chrom,start,end
                    pc_chr[NR] = $1;
                    pc_start[NR] = $2;
                    pc_end[NR] = $3;
                    pc_n = NR;
                    next;
                }
                FNR==1 {
                    print;  # header line of .junc
                    next;
                }
                {
                    chr = $1;
                    s   = $2;
                    e   = $3;
                    len = e - s;
                    if (len <= 0) next;
                    keep = 0;
                    for (i = 1; i <= pc_n; i++) {
                        if (pc_chr[i] != chr) continue;
                        ps = pc_start[i];
                        pe = pc_end[i];
                        # overlap length
                        os = (s > ps ? s : ps);
                        oe = (e < pe ? e : pe);
                        ov = oe - os;
                        if (ov > 0 && ov >= 0.9 * len) {
                            keep = 1;
                            break;
                        }
                    }
                    if (keep) print;
                }
            ' "$port_filt" "$junc" > "$tmp_junc"
            mv "$tmp_junc" "$junc"
        fi
    fi

    echo "[Done] $i : ${bam_file}"
done

echo "Finished extracting junctions with megadepth."
