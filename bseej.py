import argparse
import sys
import shlex
import os
import json

from BSEEJ.gene import Gene
from BSEEJ.model import Model
from utilities import *


def _resolve_annotation_path(main_path: str, explicit_path: str = "") -> str:
    explicit_path = str(explicit_path or "").strip()
    if explicit_path:
        return explicit_path
    try:
        manifest_path = os.path.join(os.path.abspath(os.path.dirname(main_path)), "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            return str(manifest.get("annotation") or "").strip()
    except Exception:
        pass
    return ""

def _export_trivial_single_transcript(gene: Gene, *, method_label: str, k_requested: int, idx_suffix: str) -> None:
    """
    Emit a single transcript per gene by chaining all introns (and any path-node segments)
    in sorted genomic order. This is used for genes deemed "not trainable" (e.g., no
    interval conflicts / min_k < 2) to avoid dropping them from assembly evaluation.
    """
    import numpy as np

    nodes_df = getattr(gene, "nodes_df", None)
    doc = getattr(gene, "document", None)
    if nodes_df is None or getattr(nodes_df, "shape", (0, 0))[0] == 0:
        return
    if doc is None or not hasattr(doc, "shape") or doc.shape[0] < 1:
        return

    V = int(nodes_df.shape[0])
    D = int(doc.shape[0])

    # Build a union intron set from all nodes (including path segments).
    seg_indices = sorted(
        int(col.split("_")[-1])
        for col in nodes_df.columns
        if str(col).startswith("seg_start_")
    )
    max_segments = max(seg_indices) if seg_indices else 1

    def _node_introns(v_idx: int):
        out = []
        if max_segments > 1:
            for seg_i in range(1, max_segments + 1):
                s_col = f"seg_start_{seg_i}"
                e_col = f"seg_end_{seg_i}"
                if s_col not in nodes_df.columns or e_col not in nodes_df.columns:
                    continue
                try:
                    s = int(nodes_df.loc[v_idx, s_col])
                    e = int(nodes_df.loc[v_idx, e_col])
                except Exception:
                    continue
                if e > s and s > 0:
                    out.append((s, e))
        if not out:
            try:
                out = [(int(nodes_df.loc[v_idx, "start"]), int(nodes_df.loc[v_idx, "end"]))]
            except Exception:
                out = []
        return out

    def _is_path_node(v_idx: int) -> bool:
        if max_segments <= 1:
            return False
        segs = _node_introns(v_idx)
        return len(segs) >= 2

    intron_set = set()
    for v in range(V):
        for s, e in _node_introns(v):
            intron_set.add((int(s), int(e)))
    introns = sorted(list(intron_set), key=lambda x: (x[0], x[1]))
    if not introns:
        return

    # Validate non-overlap and basic sanity.
    for s, e in introns:
        if e < s:
            return
    for (s1, e1), (s2, e2) in zip(introns, introns[1:]):
        if e1 >= s2:
            return

    chrom = "."
    if "chrom" in nodes_df.columns:
        chroms = {str(c) for c in nodes_df["chrom"].astype(str).tolist() if str(c) and str(c) != "nan"}
        if len(chroms) == 1:
            chrom = next(iter(chroms))
        elif len(chroms) > 1:
            return

    strand = "."
    if "strand" in nodes_df.columns:
        strands = {str(s) for s in nodes_df["strand"].astype(str).tolist() if str(s) in ("+", "-")}
        if len(strands) == 1:
            strand = next(iter(strands))

    # Create the run-specific output directory consistent with the normal path.
    base_dir = getattr(gene, "result_path", None) or "."
    suffix = f"_{idx_suffix}" if idx_suffix else ""
    run_dirname = f"{gene.name}_{method_label.lower()}_{int(k_requested)}{suffix}"
    run_dir = os.path.join(base_dir, run_dirname)
    os.makedirs(run_dir, exist_ok=True)
    gene.result_path = run_dir

    # Keep the interval graph text export for inspection.
    if hasattr(gene, "_debug_print_interval_graph"):
        try:
            gene._debug_print_interval_graph()
        except Exception:
            pass

    # Infer exons from intron boundaries (1-based inclusive), mirroring utilities exporter.
    exons = []
    first_s, _first_e = introns[0]
    _last_s, last_e = introns[-1]
    exon1_end = max(1, first_s - 1)
    exons.append((exon1_end, exon1_end))
    for (prev_s, prev_e), (next_s, next_e) in zip(introns, introns[1:]):
        ex_s = int(prev_e + 1)
        ex_e = int(next_s - 1)
        if ex_e < ex_s:
            return
        exons.append((ex_s, ex_e))
    exonN_start = int(last_e + 1)
    exons.append((exonN_start, exonN_start))

    # Sample names (for counts header), mirroring utilities exporter.
    sample_files = getattr(gene, "sample_files", None) or []
    sample_names = []
    for path in sample_files:
        base = os.path.basename(str(path))
        for suf in (".junc.gz", ".junc"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        sample_names.append(base)
    if len(sample_names) != D:
        sample_names = [f"sample_{i}" for i in range(D)]

    # Members: all nodes.
    members = np.arange(V, dtype=int)
    members_intron = np.array([i for i in members.tolist() if not _is_path_node(int(i))], dtype=int)
    intron_counts = (doc[:, members_intron].sum(axis=1).tolist() if members_intron.size > 0 else [0] * D)
    aug_counts = doc[:, members].sum(axis=1).tolist()

    def _emit_gtf_line(fh, feature, start, end, attrs):
        fh.write(
            "\t".join(
                [
                    str(chrom),
                    "BSEEJ",
                    str(feature),
                    str(int(start)),
                    str(int(end)),
                    ".",
                    str(strand) if strand in ("+", "-") else ".",
                    ".",
                    attrs,
                ]
            )
            + "\n"
        )

    # Write both b and beta GTFs as the same single transcript.
    tid_b = f"BSEEJ_{gene.name}_C0_b"
    tid_beta = f"BSEEJ_{gene.name}_C0_beta"
    gid = str(gene.name)
    attrs_b = f'gene_id "{gid}"; transcript_id "{tid_b}";'
    attrs_beta = f'gene_id "{gid}"; transcript_id "{tid_beta}";'
    tx_start = min(s for s, _e in exons)
    tx_end = max(e for _s, e in exons)

    gtf_b_path = os.path.join(run_dir, f"{gene.name}_bseej_predicted_b.gtf")
    gtf_beta_path = os.path.join(run_dir, f"{gene.name}_bseej_predicted_beta.gtf")
    with open(gtf_b_path, "w", encoding="utf-8") as fh_b:
        _emit_gtf_line(fh_b, "transcript", tx_start, tx_end, attrs_b)
        for exon_idx, (ex_s, ex_e) in enumerate(exons, start=1):
            _emit_gtf_line(fh_b, "exon", ex_s, ex_e, attrs_b + f' exon_number "{exon_idx}";')
    with open(gtf_beta_path, "w", encoding="utf-8") as fh_beta:
        _emit_gtf_line(fh_beta, "transcript", tx_start, tx_end, attrs_beta)
        for exon_idx, (ex_s, ex_e) in enumerate(exons, start=1):
            _emit_gtf_line(fh_beta, "exon", ex_s, ex_e, attrs_beta + f' exon_number "{exon_idx}";')

    def _write_counts(path, tid, row):
        with open(path, "w", encoding="utf-8") as out_fh:
            out_fh.write("transcript_id\t" + "\t".join(sample_names) + "\n")
            out_fh.write(tid + "\t" + "\t".join(map(str, row)) + "\n")

    _write_counts(os.path.join(run_dir, f"{gene.name}_bseej_counts_b.tsv"), tid_b, intron_counts)
    _write_counts(os.path.join(run_dir, f"{gene.name}_bseej_counts_b_augmented.tsv"), tid_b, aug_counts)
    _write_counts(os.path.join(run_dir, f"{gene.name}_bseej_counts_beta.tsv"), tid_beta, intron_counts)
    _write_counts(os.path.join(run_dir, f"{gene.name}_bseej_counts_beta_augmented.tsv"), tid_beta, aug_counts)

    print(f"[INFO] Trivial export for {gene.name}: wrote 1 transcript (no conflicts / not trainable).")


class Main(object):
    """ Initializes the input values """
    n_cluster = 1
    max_n_iter = 10000
    eta = 0.01     # hyperparameter for bete, |V|-Dirichlet
    alpha = 1
    r = 1
    s = 1
    p = ''
    g = 'A2ML1'
    o = ''
    @classmethod
    def main(cls, cmd_args):
        """
        The main function sets the hyper-parameters values, accordingly initilizes BREM algorithm,
        then makes the model and saves the results.
        """
    
        cls.init(cmd_args)
    
        print('=====================================================')
        print('Gene:', cls.g)
        print('junction path:', cls.p)
    
        print('result path:', cls.o)
    
        print('Number of clusters:', cls.n_cluster)
        print('Maximum number of iterations:', cls.max_n_iter)
    
        print('model parameter, eta:', cls.eta)
        print('model parameter, alpha:', cls.alpha)
        print('model parameter, r:', cls.r)
        print('model parameter, s:', cls.s)
        print('=====================================================')
    
        burn_in = cls.max_n_iter / 2
        convergence_checkpoint_interval = (cls.max_n_iter - burn_in) / 10
        epsilon = 0.000001
    
        # Read gene junction files
        # with zipfile.ZipFile(os.path.join(cls.p, cls.g) + '.zip', 'r') as zip_ref:
        #     zip_ref.extractall(cls.p)
    
        # Make the model and gene objects
        print('training gene', cls.g, 'with k =', cls.n_cluster)
        model = Model(eta=cls.eta, alpha=cls.alpha, epsilon=epsilon, r=cls.r, s=cls.s)
    
        gene = Gene(cls.g, cls.p, cls.o)
    
        # Preprocess the gene
        gene.preprocess()
        
        # Train the gene
        model.train(gene, cls.n_cluster, n_iter=cls.max_n_iter, burn_in=burn_in,
                    convergence_checkpoint_interval=convergence_checkpoint_interval, verbose=True)
        
        # Save all the results, including all the parameters in the model in a pickle file and clusters
        _ = save_results(gene, model)
    
    @classmethod
    def init(cls, cmd_args):
        """ Check the parser for possible inputs and overrides the existing default values if any. """
        parser = Main.get_parser()
        args = parser.parse_args(cmd_args[1:])
        
        cls.n_cluster = int(args.n_cluster)
        cls.max_n_iter = int(args.max_n_iter)
        cls.eta = args.eta
        cls.alpha = args.alpha
        cls.r = args.r
        cls.s = args.s
        cls.p = args.main_path
        cls.g = args.gene_name
        cls.o = args.result_path
    
    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser(description='Implementation of BREM.')
        parser.add_argument("-k", "--n_cluster", help="Number of clusters (integer >= 1, default = 1)",
                            default=1)
        parser.add_argument("-i", "--max_n_iter", help="Max number of iterations (integer) (default = 1000)",
                            default=cls.max_n_iter)
        parser.add_argument("-e", "--eta", required=False, help="eta (default = 0.01)", default=0.01)
        parser.add_argument("-a", "--alpha", required=False, help="alpha (default = 1)", default=1)
        parser.add_argument("-r", "--r", required=False, help="model parameter r (default = 1)", default=1)
        parser.add_argument("-s", "--s", required=False, help="model parameter s (default = 1)", default=1)
        parser.add_argument("-p", "--main_path", required=False, help="Main path (default = A2ML1/)", default='A2ML1/')
        parser.add_argument("-o", "--result_path", required=False, help="result path (default = ./)", default='./results')
        parser.add_argument("-g", "--gene_name", required=False, help="gene_name (default = A2ML1)",
                            default='A2ML1')
        return parser



if __name__ == '__main__':
    Main.main(sys.argv)
