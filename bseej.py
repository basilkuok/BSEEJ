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
    mode = 2  # 1 = Gibbs, 2 = CAVI, 3 = SVI
    # Whether to write the large per-sample result CSV via utilities.save_results.
    # 0 = disabled (default while debugging), 1 = enabled.
    save_result = 0
    # Whether to write diagnostic outputs (ELBO plot, interval graph, phi/zeta exports).
    # 0 = disabled (default), 1 = enabled.
    diagnostics = 0
    # Whether to write preprocessing cache files (.pkl/.sig).
    # 0 = disabled (default), 1 = enabled.
    write_preproc_cache = 0
    # Default minimum junction coverage used when defining nodes.
    # Junctions with total support < min_cov across all samples are dropped.
    min_cov = 30
    idx_suffix = ""
    variant = "current"
    annotation_path = ""
    novel_m = None
    # Sequencing / preprocessing mode:
    #   0 = short-read pipeline  (STAR + Megadepth),
    #   1 = long-read pipeline   (minimap2 + Megadepth).
    # This flag describes how BAMs and junctions were generated upstream;
    # JARVIS itself always consumes the same .junc / .jxs.tsv format.
    long_mode = 1

    @classmethod
    def interactive_main(cls):
        """
        Interactive entry point: show current defaults, allow the user to
        override flags, confirm, then run with the chosen configuration.
        """
        parser = Main.get_parser()

        print("Do you want to specify the configurations?")
        # Parse empty arg list to show defaults
        args_default = parser.parse_args([])
        print("Current default configuration:")
        print(f"  -k (n_cluster): {args_default.n_cluster}")
        print(f"  -i (max_n_iter): {args_default.max_n_iter}")
        print(f"  -e (eta): {args_default.eta}")
        print(f"  -a (alpha): {args_default.alpha}")
        print(f"  -r (r): {args_default.r}")
        print(f"  -s (s): {args_default.s}")
        print(f"  -p (main_path): {args_default.main_path or './'}")
        print(f"  -o (result_path): {args_default.result_path or './'}")
        print(f"  -g (gene_name): {args_default.gene_name}")
        print(f"  -m (inference mode): {args_default.mode} (1=Gibbs, 2=CAVI, 3=SVI)")
        print(f"  -min_cov (junction/node filtering inside JARVIS): {args_default.min_cov}")
        print(f"  -identifier (output suffix): {args_default.idx}")
        print(f"  -variant: {args_default.variant}")
        print(f"  -annotation: {args_default.annotation}")
        print(f"  -novel_m: {args_default.novel_m}")
        print(f"  -long (sequencing mode): {args_default.long_mode} (0=short-read, 1=long-read)")
        print(f"  -save_result (write full result CSVs): {args_default.save_result} (0=disabled, 1=enabled)")
        print("If you want to specify, please type one or more flags.")
        user_line = input("If you wish to proceed without specification, type [n]: ").strip()

        if user_line and user_line.lower() != "n":
            try:
                user_args = shlex.split(user_line)
            except ValueError:
                print("Could not parse the flags you entered; proceeding with defaults.")
                user_args = []
        else:
            user_args = []

        # Re-parse with any user overrides
        args = parser.parse_args(user_args)

        print("Do you wish to proceed with the following configuration:")
        print(f"  -k (n_cluster): {args.n_cluster}")
        print(f"  -i (max_n_iter): {args.max_n_iter}")
        print(f"  -e (eta): {args.eta}")
        print(f"  -a (alpha): {args.alpha}")
        print(f"  -r (r): {args.r}")
        print(f"  -s (s): {args.s}")
        print(f"  -p (main_path): {args.main_path or './'}")
        print(f"  -o (result_path): {args.result_path or './'}")
        print(f"  -g (gene_name): {args.gene_name}")
        print(f"  -m (inference mode): {args.mode} (1=Gibbs, 2=CAVI, 3=SVI)")
        print(f"  -min_cov (junction/node filtering inside JARVIS): {args.min_cov}")
        print(f"  -identifier (output suffix): {args.idx}")
        print(f"  -variant: {args.variant}")
        print(f"  -annotation: {args.annotation}")
        print(f"  -novel_m: {args.novel_m}")
        print(f"  -long (sequencing mode): {args.long_mode} (0=short-read, 1=long-read)")
        print(f"  -save_result (write full result CSVs): {args.save_result} (0=disabled, 1=enabled)")
        confirm = input("[y/n]: ").strip().lower()
        if confirm != "y":
            print("Aborting.")
            return

        # Build a synthetic argv and delegate to the normal main()
        synthetic_argv = [sys.argv[0]] + user_args
        print("Proceeding with configuration:")
        print(f"  -k (n_cluster): {args.n_cluster}")
        print(f"  -i (max_n_iter): {args.max_n_iter}")
        print(f"  -e (eta): {args.eta}")
        print(f"  -a (alpha): {args.alpha}")
        print(f"  -r (r): {args.r}")
        print(f"  -s (s): {args.s}")
        print(f"  -p (main_path): {args.main_path or './'}")
        print(f"  -o (result_path): {args.result_path or './'}")
        print(f"  -g (gene_name): {args.gene_name}")
        print(f"  -m (inference mode): {args.mode} (1=Gibbs, 2=CAVI, 3=SVI)")
        print(f"  -min_cov (Portcullis filtering): {args.min_cov}")
        print(f"  -identifier (output suffix): {args.idx}")
        print(f"  -variant: {args.variant}")
        print(f"  -annotation: {args.annotation}")
        print(f"  -novel_m: {args.novel_m}")
        print(f"  -long (sequencing mode): {args.long_mode} (0=short-read, 1=long-read)")
        cls.main(synthetic_argv)
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
        print('Minimum junction coverage (junction/node filtering inside JARVIS):', cls.min_cov)
        print('Sequencing mode (-long):', cls.long_mode, '(0=short-read, 1=long-read)')
        if cls.idx_suffix:
            print('Output filename suffix:', cls.idx_suffix)
        print('Variant:', cls.variant)
        if cls.annotation_path:
            print('Annotation path:', cls.annotation_path)
        if cls.novel_m is not None:
            print('Novel transcript offset m:', cls.novel_m)
        print('Inference mode (1=Gibbs, 2=CAVI, 3=SVI):', cls.mode)
        print('Save full result CSVs (-save_result):', getattr(cls, 'save_result', 0), '(0=disabled, 1=enabled)')
    
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
        requested_k = int(cls.n_cluster)
        print('training gene', cls.g, 'with requested k =', requested_k)
        model = Model(
            eta=cls.eta,
            alpha=cls.alpha,
            epsilon=epsilon,
            r=cls.r,
            s=cls.s,
            idx_suffix=cls.idx_suffix,
        )
        # Record sequencing / preprocessing mode on the model for provenance.
        setattr(model, "long_mode", cls.long_mode)
        # Control whether to emit the large utilities.save_results CSV.
        setattr(model, "save_results", bool(getattr(cls, "save_result", 0)))

        gene = Gene(
            cls.g,
            cls.p,
            cls.o,
            min_coverage=cls.min_cov,
            idx_suffix=cls.idx_suffix,
            variant=cls.variant,
            annotation_path=cls.annotation_path,
            novel_m=cls.novel_m,
        )
        setattr(gene, "long_mode", cls.long_mode)

        # Disable expensive, non-essential outputs by default. This does not
        # change inference; it only suppresses diagnostic/caching artifacts.
        if not bool(getattr(cls, "diagnostics", 0)):
            def _export_phi_zeta_csv_only(gene, K, method_label, phi, zeta):
                suffix = f"_{model.idx_suffix}" if getattr(model, "idx_suffix", "") else ""
                export_dir = gene.result_path
                os.makedirs(export_dir, exist_ok=True)
                joiner = f"_K_{K}{suffix}" if suffix else f"_K_{K}"

                joint_csv = os.path.join(export_dir, f"{gene.name}_{method_label}_phi_zeta{joiner}.csv")
                phi_flat_cols = phi.shape[1] * phi.shape[2]
                n_cols = max(zeta.shape[1], phi_flat_cols)
                header = "matrix,row_index," + ",".join(f"col_{j}" for j in range(n_cols))
                with open(joint_csv, "w") as fh:
                    fh.write(header + "\n")
                    for i, row in enumerate(zeta):
                        vals = list(map(str, row)) + [""] * (n_cols - len(row))
                        fh.write("zeta (beta)," + str(i) + "," + ",".join(vals) + "\n")
                    for d in range(phi.shape[0]):
                        flattened = phi[d].reshape(-1)
                        vals = list(map(str, flattened)) + [""] * (n_cols - len(flattened))
                        fh.write("phi (z)," + str(d) + "," + ",".join(vals) + "\n")
                print(f"[{str(method_label).upper()}] Wrote combined phi/zeta to {joint_csv}")

            try:
                setattr(model, "_export_phi_zeta_and_excel", _export_phi_zeta_csv_only)
            except Exception:
                pass
            try:
                setattr(model, "_save_elbo_plot", lambda *args, **kwargs: None)
            except Exception:
                pass
            try:
                setattr(gene, "_export_raw_junction_files", lambda *args, **kwargs: None)
            except Exception:
                pass

        os.environ["BSEEJ_WRITE_PREPROC_CACHE"] = "1" if bool(getattr(cls, "write_preproc_cache", 0)) else "0"
    
        # Preprocess the gene
        gene.preprocess()
        effective_k = int(getattr(gene, "effective_k", requested_k) or requested_k)
        print('effective k used for inference =', effective_k)

        # Skip genes with insufficient junction evidence to train a non-trivial model.
        # This commonly happens when a per-gene folder exists but all junctions were
        # filtered out (e.g., min_cov) or the interval graph is trivial.
        try:
            n_v = int(getattr(gene, "n_v", 0) or 0)
        except Exception:
            n_v = 0
        try:
            doc = getattr(gene, "document", None)
            v_doc = int(doc.shape[1]) if doc is not None and hasattr(doc, "shape") else 0
        except Exception:
            v_doc = 0
        if not getattr(gene, "trainable", True) or n_v < 2 or v_doc < 1:
            # For assembly evaluation, avoid dropping "simple" genes that have
            # evidence but no interval conflicts (min_k < 2). Emit a single
            # transcript that chains all introns in order.
            min_k = getattr(gene, "min_k", None)
            has_evidence = getattr(gene, "samples_df", None) is not None and len(getattr(gene, "samples_df", [])) > 0
            # Always export assembly artifacts when possible.
            if has_evidence and (min_k is not None and int(min_k) < 2):
                method = {1: "gibbs", 2: "cavi", 3: "svi"}.get(int(cls.mode), "cavi")
                _export_trivial_single_transcript(
                    gene,
                    method_label=method,
                    k_requested=effective_k,
                    idx_suffix=str(getattr(cls, "idx_suffix", "") or ""),
                )
                return

            print(f"[INFO] Gene {gene.name} is not trainable (n_v={n_v}, V_doc={v_doc}); skipping.")
            return

        # Train the gene based on selected inference mode
        if cls.mode == 1:
            train_gibbs = getattr(model, 'train_gibbs', None)
            if train_gibbs is None:
                raise AttributeError("Mode 1 requested, but Model.train_gibbs is not available.")
            train_gibbs(
                gene,
                effective_k,
                n_iter=cls.max_n_iter,
                burn_in=burn_in,
                convergence_checkpoint_interval=convergence_checkpoint_interval,
                verbose=True,
            )
        elif cls.mode == 2:
            model.train(
                gene,
                effective_k,
                n_iter=cls.max_n_iter,
                burn_in=burn_in,
                convergence_checkpoint_interval=convergence_checkpoint_interval,
                verbose=True,
            )
        elif cls.mode == 3:
            batch_size = max(1, int(gene.document.shape[0] // 10)) if hasattr(gene, "document") else 1
            model.train_svi(
                gene,
                effective_k,
                n_iter=cls.max_n_iter,
                batch_size=batch_size,
                verbose=True,
            )
        else:
            raise ValueError(f"Unsupported inference mode '{cls.mode}'. Expected 1, 2, or 3.")
        
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
        cls.mode = int(args.mode)
        cls.save_result = int(getattr(args, "save_result", 0))
        cls.diagnostics = int(getattr(args, "diagnostics", 0))
        cls.write_preproc_cache = int(getattr(args, "write_preproc_cache", 0))
        cls.min_cov = int(args.min_cov)
        cls.idx_suffix = str(args.idx)
        cls.variant = str(getattr(args, "variant", "current") or "current").strip().lower()
        cls.annotation_path = _resolve_annotation_path(cls.p, str(getattr(args, "annotation", "") or ""))
        novel_m_raw = getattr(args, "novel_m", None)
        cls.novel_m = None if novel_m_raw in (None, "") else int(novel_m_raw)
        if cls.variant not in ("current", "reference", "hybrid"):
            raise ValueError(f"Unsupported variant '{cls.variant}'. Expected current, reference, or hybrid.")
        if cls.variant == "hybrid" and cls.novel_m is None:
            raise ValueError("Variant 'hybrid' requires --novel-m.")
        if cls.variant in ("reference", "hybrid") and not cls.annotation_path:
            raise ValueError(
                f"Variant '{cls.variant}' requires --annotation or a manifest.json beside the prepared gene folders."
            )
        # Sequencing / preprocessing mode: 0=short-read (STAR), 1=long-read (minimap2)
        try:
            lm = int(args.long_mode)
        except (TypeError, ValueError):
            lm = 1
        if lm not in (0, 1):
            print("Warning: -long must be 0 (short-read) or 1 (long-read); defaulting to 1.")
            lm = 1
        cls.long_mode = lm
    
    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser(description='Implementation of BREM.')
        parser.add_argument("-k", "--n_cluster", help="Number of clusters (integer >= 1, default = 1)",
                            default=1)
        parser.add_argument("-i", "--max_n_iter", help="Max number of iterations (integer) (default = 10000)",
                            default=cls.max_n_iter)
        parser.add_argument("-e", "--eta", required=False, help="eta (default = 0.01)", default=cls.eta)
        parser.add_argument("-a", "--alpha", required=False, help="alpha (default = 1)", default=cls.alpha)
        parser.add_argument("-r", "--r", required=False, help="model parameter r (default = 1)", default=cls.r)
        parser.add_argument("-s", "--s", required=False, help="model parameter s (default = 1)", default=cls.s)
        parser.add_argument("-p", "--main_path", required=False, help="Main path (default = ./)", default=cls.p)
        parser.add_argument("-o", "--result_path", required=False, help="result path (default = ./)", default=cls.o)
        parser.add_argument("-g", "--gene_name", required=False, help="gene_name (default = A2ML1)",
                            default=cls.g)
        parser.add_argument("-m", "--mode", required=False,
                            help="Inference mode: 1=Gibbs, 2=CAVI, 3=Stochastic VI (default = 2)",
                            default=cls.mode)
        parser.add_argument(
            "-min_cov",
            dest="min_cov",
            required=False,
            help="Minimum junction coverage for node definition (default = 30; "
                 "junctions with total support < min_cov across all samples are dropped).",
            default=cls.min_cov,
        )
        parser.add_argument("-identifier", dest="idx", required=False,
                            help="Identifier suffix to append to output filenames (default = '')",
                            default=cls.idx_suffix)
        parser.add_argument(
            "--variant",
            dest="variant",
            required=False,
            choices=["current", "reference", "hybrid"],
            help="Structural variant: current, reference, or hybrid.",
            default=cls.variant,
        )
        parser.add_argument(
            "--annotation",
            dest="annotation",
            required=False,
            help="Reference GTF/GFF for reference or hybrid variants. If omitted, BSEEJ will try manifest.json.",
            default=cls.annotation_path,
        )
        parser.add_argument(
            "--novel-m",
            dest="novel_m",
            required=False,
            type=int,
            help="User-specified transcript offset m used only for variant=hybrid.",
            default=cls.novel_m,
        )
        parser.add_argument(
            "-long", "--long_mode", dest="long_mode", required=False, type=int,
            help="Sequencing / preprocessing mode: 0=short-read (STAR + Megadepth), "
                 "1=long-read (minimap2 + Megadepth). Default = 1.",
            default=cls.long_mode,
        )
        parser.add_argument(
            "-save_result",
            dest="save_result",
            required=False,
            type=int,
            help="Whether to write full per-sample result CSVs via utilities.save_results "
                 "(0=disabled, 1=enabled; default=0).",
            default=cls.save_result,
        )
        parser.add_argument(
            "-diagnostics",
            dest="diagnostics",
            required=False,
            type=int,
            help="Whether to write diagnostic outputs (ELBO plots, interval graph, phi/zeta exports) "
                 "(0=disabled, 1=enabled; default=0).",
            default=cls.diagnostics,
        )
        parser.add_argument(
            "-write_preproc_cache",
            dest="write_preproc_cache",
            required=False,
            type=int,
            help="Whether to write preprocessing cache files (.pkl/.sig) "
                 "(0=disabled, 1=enabled; default=0).",
            default=cls.write_preproc_cache,
        )
        return parser



if __name__ == '__main__':
    if len(sys.argv) == 1:
        Main.interactive_main()
    else:
        Main.main(sys.argv)
