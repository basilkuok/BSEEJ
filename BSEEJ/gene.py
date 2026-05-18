from utilities import *
from utilities import _maximum_independent_set_from_intersection
from annotation_utils import load_transcript_introns
import shutil
import json
import pickle


class Gene(object):
    
    def __init__(
        self,
        name,
        gene_list_dir,
        result_path,
        min_coverage=30,
        idx_suffix="",
        variant="current",
        annotation_path="",
        novel_m=None,
    ):
        """Initialize gene instance from the zip file containing the gene .bam files:
        This function computes the gene nodes and the interval graph and minimum number of clusters"""

        self.name = name
        # self.junc_path = gene_list_dir + name + '/'
        self.junc_path = gene_list_dir
        # self.result_path = gene_list_dir + 'results_' + self.name
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        if not os.path.exists(os.path.join(result_path, self.name)):
            os.mkdir(os.path.join(result_path, self.name))
        self.result_path = os.path.join(result_path, self.name)
        self.idx_suffix = idx_suffix
        self.variant = str(variant or "current").strip().lower()
        self.annotation_path = str(annotation_path or "").strip()
        self.novel_m = None if novel_m is None else int(novel_m)
        # BSEEJ no longer applies its own min_coverage threshold; coverage
        # filtering is handled upstream (e.g., by Portcullis). We keep the
        # parameter for backward compatibility but ignore it in preprocessing.
        self.min_coverage = int(min_coverage)
        self.samples_df, self.samples_df_dict = self.get_sample_df()
        self.nodes_df = self.get_junctions()
        # Whether to include chromosome in intron keys ("chr:start-end").
        # We preserve the legacy "start-end" keys when all nodes are on a
        # single chromosome (common for per-gene runs like A2ML1/SIRV).
        self._use_chrom_in_keys = False
        # Structural quantities depending on the interval graph (min_k,
        # trainability, and candidate K values) are computed in preprocess()
        # after the conflict matrix has been built once.
        self.min_k = None
        self.trainable = True
        self.all_n_k = []
        self.intersection = None
        self.overlap_m = None
        self.mvc = None
        self.word_dict = None
        self.document = None
        self.id2w_dict = None
        self.w2id_dict = None
        self.cooc_matrix = None
        self.node_introns = []
        self.reference_transcripts = {}
        self.reference_intron_to_txs = {}
        self.node_reference_txs = []
        self.reference_k = None
        self.effective_k = None
        self.n_w_list = None
        self.n_w = None
        self.n_v = None
        self.n_d = None
        self.document_tr = None
        self.document_te = None
        self.training_idx = None
        self.test_idx = None
        self.mis = None
        self.max_ind_set = None
        # self.samples_df = None
        # self.samples_df_dict = None
        # self.nodes_df = None
        # self.min_k = None

    # --- Helpers for caching expensive preprocessing state -----------------

    def _build_preprocess_signature(self):
        """
        Build a lightweight signature of the junction inputs (junc/jxs/all_jxs
        files under junc_path) so we can detect when cached preprocessing
        state is still valid. We use filenames + size + mtime.
        """
        files = []
        if os.path.isdir(self.junc_path):
            for fname in sorted(os.listdir(self.junc_path)):
                if not (
                    fname.endswith(".junc")
                    or fname.endswith(".jxs.tsv")
                    or fname.endswith(".all_jxs.tsv")
                ):
                    continue
                path = os.path.join(self.junc_path, fname)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                files.append(
                    {
                        "name": fname,
                        "size": int(st.st_size),
                        "mtime": int(st.st_mtime),
                    }
                )
        # JSON is stable and human-readable; order is deterministic due to
        # sorted filenames.
        return json.dumps(files, separators=(",", ":"))

    def get_sample_df(self):
        """Compute the gene's intron excisions from .junc files."""
        # Deterministic ordering is important because downstream outputs use
        # integer sample IDs (row indices). Sort by filename so multi-sample
        # runs are reproducible across OS/filesystems.
        samples_list = [
            os.path.join(self.junc_path, s)
            for s in sorted(os.listdir(self.junc_path))
            if s.endswith('.junc')
        ]
        if not samples_list:
            # No samples available.
            empty = pd.DataFrame(columns=["chrom", "chromStart", "chromEnd", "score", "strand"])
            return empty, {}
        columns = ['chrom', 'chromStart', 'chromEnd', 'junc_id', 'score', 'strand', 'start', 'end', 'f1', 'f2', 'f3',
                   'f4']
        samples_dfs = [pd.read_csv(f, sep='\t', names=columns, skiprows=1) for f in samples_list]
        if not samples_dfs:
            empty = pd.DataFrame(columns=["chrom", "chromStart", "chromEnd", "score", "strand"])
            return empty, {}
        samples_df = pd.concat(samples_dfs, ignore_index=True)
        # samples = []
        # for sample in samples_list:
        #     if '.gz' in sample:
        #         with gzip.open(self.junc_path + sample) as f:
        #             sample = pd.read_csv(f, sep='\t').values.tolist()
        #             samples.extend(sample)
        #     else:
        #         sample = pd.read_csv(self.junc_path + sample, sep='\t').values.tolist()
        #         samples.extend(sample)
        #
        # samples_df = pd.DataFrame(samples, columns=['chrom', 'chromStart', 'chromEnd',
        #                                               'qual', 'score', 'strand'])
        samples_df = samples_df[['chrom', 'chromStart', 'chromEnd', 'score', 'strand']]
        samples_df = samples_df.astype({"chromStart": np.int32, "chromEnd": np.int32, "score": np.int32})
    
        if len(samples_df) == 0:
            return [], []
        else:
            # Aggregate counts per unique intron across all samples.
            # For whole-sample inputs we must keep chromosome (and strand if present)
            # distinct; otherwise introns on different chromosomes with the same
            # start/end would be merged incorrectly.
            group_cols = ['chrom', 'chromEnd', 'chromStart']
            if 'strand' in samples_df.columns:
                group_cols.append('strand')
            samples_df = samples_df.groupby(group_cols, dropna=False)['score'].sum().reset_index()

            # Optional hard coverage filter (e.g. min_coverage = 30) applied
            # once at node-definition time, mirroring the older pipeline.
            min_cov = getattr(self, "min_coverage", 0)
            try:
                min_cov = int(min_cov)
            except (TypeError, ValueError):
                min_cov = 0
            if min_cov > 0:
                samples_df = samples_df[samples_df['score'] >= min_cov].reset_index(drop=True)

            if len(samples_df) == 0:
                empty = pd.DataFrame(columns=["chrom", "chromStart", "chromEnd", "score", "strand"])
                return empty, {}

            samples_df_dict = {}
            for i in range(len(samples_df)):
                samples_df_dict[i] = {}
                for ke in list(samples_df.columns):
                    samples_df_dict[i][ke] = samples_df.loc[i, ke]
            return samples_df, samples_df_dict
    
    def get_junctions(self):
        """Generate an interval graph,
        node_n = number of nodes in the generated graph
        Irange: (integer): The range, in which the intervals fall into"""
        
        junc_num = self.samples_df.shape[0]
        # Keep chromosome/strand metadata so whole-sample runs do not conflate
        # unrelated loci.
        nodes_df = pd.DataFrame(
            data=np.zeros([junc_num, 3]),
            columns=['start', 'length', 'end'],
            index=range(0, junc_num),
        )
        
        nodes_df['start'] = self.samples_df.chromStart
        nodes_df['end'] = self.samples_df.chromEnd
        nodes_df['length'] = nodes_df['end'] - nodes_df['start']
        if 'chrom' in self.samples_df.columns:
            nodes_df['chrom'] = self.samples_df.chrom.astype(str).values
        if 'strand' in self.samples_df.columns:
            nodes_df['strand'] = self.samples_df.strand.astype(str).values
        
        # Sort primarily by chromosome to keep related loci together, then by end.
        sort_cols = ['end']
        if 'chrom' in nodes_df.columns:
            sort_cols = ['chrom', 'end', 'start']
        nodes_df = nodes_df.sort_values(by=sort_cols)
        nodes_df = nodes_df.reset_index(drop=True)
        # nodes_df['label'] = nodes_df.index.values
        graph_labels = []
        node_labels = []
        
        for v in range(nodes_df.shape[0]):
            if 'chrom' in nodes_df.columns:
                graph_labels.append(
                    str(nodes_df.loc[v, 'chrom'])
                    + ':'
                    + str(int(nodes_df.loc[v, 'start']))
                    + '_'
                    + str(int(nodes_df.loc[v, 'end']))
                )
            else:
                graph_labels.append(str(int(nodes_df.loc[v, 'start'])) + '_' + str(int(nodes_df.loc[v, 'end'])))
            node_labels.append(str(v))
        
        nodes_df['graph_labels'] = graph_labels
        nodes_df['node_labels'] = node_labels
        return nodes_df

    def _reference_cache_path(self):
        return os.path.join(self.junc_path, "reference_introns.json")

    def _load_reference_transcripts(self):
        if self.variant not in ("reference", "hybrid"):
            self.reference_transcripts = {}
            self.reference_intron_to_txs = {}
            self.reference_k = None
            return
        ref_payload = None
        cache_path = self._reference_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    ref_payload = json.load(fh)
            except Exception:
                ref_payload = None
        if ref_payload is None:
            if not self.annotation_path:
                raise RuntimeError(
                    f"Variant '{self.variant}' for gene {self.name} requires an annotation GTF/GFF."
                )
            loaded = load_transcript_introns(self.annotation_path, restrict_gene_ids={self.name})
            ref_payload = loaded.get(
                self.name,
                {"gene_name": self.name, "chrom": "", "strand": ".", "transcripts": {}},
            )
            try:
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump(ref_payload, fh, indent=2, sort_keys=True)
            except Exception:
                pass

        tx_map = ref_payload.get("transcripts", {}) if isinstance(ref_payload, dict) else {}
        reference_transcripts = {}
        intron_to_txs = {}
        for txid, introns in tx_map.items():
            normalized = []
            for intr in introns:
                if not isinstance(intr, (list, tuple)) or len(intr) < 3:
                    continue
                chrom = str(intr[0])
                try:
                    start = int(intr[1])
                    end = int(intr[2])
                except (TypeError, ValueError):
                    continue
                if end < start:
                    start, end = end, start
                intron = (chrom, start, end)
                normalized.append(intron)
                intron_to_txs.setdefault(intron, set()).add(str(txid))
            if normalized:
                reference_transcripts[str(txid)] = tuple(normalized)
        self.reference_transcripts = reference_transcripts
        self.reference_intron_to_txs = intron_to_txs
        self.reference_k = max(1, len(reference_transcripts)) if self.variant in ("reference", "hybrid") else None

    def _seed_base_node_introns(self):
        node_introns = []
        if self.nodes_df is None:
            self.node_introns = node_introns
            return
        for row_idx in range(self.nodes_df.shape[0]):
            chrom = str(self.nodes_df.loc[row_idx, "chrom"]) if "chrom" in self.nodes_df.columns else ""
            start = int(self.nodes_df.loc[row_idx, "start"])
            end = int(self.nodes_df.loc[row_idx, "end"])
            node_introns.append(((chrom, start, end),))
        self.node_introns = node_introns

    def _rebuild_token_maps(self):
        self.id2w_dict = {}
        self.w2id_dict = {}
        try:
            chroms = set(str(c) for c in self.nodes_df.get("chrom", []).tolist())
        except Exception:
            chroms = set()
        self._use_chrom_in_keys = len([c for c in chroms if c and c != "nan"]) > 1

        for idx in range(self.nodes_df.shape[0]):
            start = int(self.nodes_df.loc[idx, "start"])
            end = int(self.nodes_df.loc[idx, "end"])
            token = f"{start}-{end}"
            if self._use_chrom_in_keys and "chrom" in self.nodes_df.columns:
                token = f"{self.nodes_df.loc[idx, 'chrom']}:{token}"
            self.id2w_dict[idx] = token
            introns = self.node_introns[idx] if idx < len(self.node_introns) else ()
            if len(introns) == 1:
                chrom, intr_start, intr_end = introns[0]
                intr_token = f"{intr_start}-{intr_end}"
                if self._use_chrom_in_keys and chrom:
                    intr_token = f"{chrom}:{intr_token}"
                self.w2id_dict[intr_token] = idx

    def _node_reference_tx_sets(self):
        tx_sets = []
        for intron_chain in self.node_introns:
            if not intron_chain:
                tx_sets.append(set())
                continue
            chain_txs = None
            for intron in intron_chain:
                txs = self.reference_intron_to_txs.get(intron, set())
                if chain_txs is None:
                    chain_txs = set(txs)
                else:
                    chain_txs &= txs
                if not chain_txs:
                    break
            tx_sets.append(set() if chain_txs is None else chain_txs)
        return tx_sets

    def _apply_node_filter(self, keep_mask):
        keep_idx = np.where(np.asarray(keep_mask, dtype=bool))[0]
        self.nodes_df = self.nodes_df.iloc[keep_idx].reset_index(drop=True)
        self.document = self.document[:, keep_idx]
        self.node_introns = [self.node_introns[i] for i in keep_idx.tolist()]
        self._rebuild_token_maps()

    def _resolve_active_graph_and_k(self):
        overlap_intersection, overlap_m = self.get_conflict()
        if self.variant == "current":
            self.node_reference_txs = [set() for _ in range(self.nodes_df.shape[0])]
            self.effective_k = None
            return overlap_intersection, overlap_m

        self._load_reference_transcripts()
        ref_sets = self._node_reference_tx_sets()

        if self.variant == "reference":
            supported = np.array([bool(s) for s in ref_sets], dtype=bool)
            self._apply_node_filter(supported)
            ref_sets = self._node_reference_tx_sets()
            overlap_intersection, overlap_m = self.get_conflict()

        V = self.nodes_df.shape[0]
        active = overlap_intersection.astype(np.int32, copy=True)
        for v1 in range(V):
            tx1 = ref_sets[v1]
            for v2 in range(v1 + 1, V):
                tx2 = ref_sets[v2]
                if active[v1, v2] == 1:
                    continue
                if tx1 and tx2 and not (tx1 & tx2):
                    active[v1, v2] = 1
                    active[v2, v1] = 1
        self.node_reference_txs = ref_sets

        if self.variant == "reference":
            self.effective_k = int(self.reference_k or 1)
        elif self.variant == "hybrid":
            if self.novel_m is None:
                raise RuntimeError("Variant 'hybrid' requires --novel-m.")
            self.effective_k = int(self.reference_k or 1) + int(self.novel_m)
        else:
            self.effective_k = None

        return active, overlap_m
    
    def get_conflict(self):
        """Find the intervals that have intersection.

        If per-node segment columns (seg_start_i/seg_end_i) are present, we
        treat each node as a union of segments and declare a conflict when
        any pair of segments overlaps. Otherwise, we fall back to a simple
        single-interval overlap check on start/end.
        """
        V = self.nodes_df.shape[0]
        intersection_m = np.zeros([V, V], dtype=np.int32)
        overlap_m = np.zeros([V, V])

        # Detect available segment indices.
        seg_indices = sorted(
            int(col.split('_')[-1])
            for col in self.nodes_df.columns
            if col.startswith('seg_start_')
        )
        use_segments = len(seg_indices) > 0
        has_chrom = 'chrom' in self.nodes_df.columns

        for v1 in range(V):
            for v2 in range(v1 + 1, V):
                has_overlap = False
                if has_chrom:
                    # Different chromosomes never conflict.
                    if str(self.nodes_df.loc[v1, "chrom"]) != str(self.nodes_df.loc[v2, "chrom"]):
                        continue
                if use_segments:
                    # Check all segment pairs between v1 and v2.
                    for si in seg_indices:
                        s1 = int(self.nodes_df.loc[v1, f"seg_start_{si}"])
                        e1 = int(self.nodes_df.loc[v1, f"seg_end_{si}"])
                        if e1 <= s1:
                            continue  # padding / no segment
                        for sj in seg_indices:
                            s2 = int(self.nodes_df.loc[v2, f"seg_start_{sj}"])
                            e2 = int(self.nodes_df.loc[v2, f"seg_end_{sj}"])
                            if e2 <= s2:
                                continue
                            if e1 > s2 and s1 < e2:
                                has_overlap = True
                                # Approximate overlap percentage using the
                                # outermost overlapping segments.
                                overlap_len = min(e1, e2) - max(s1, s2)
                                if overlap_len > 0:
                                    denom = ((e2 - s2) + (e1 - s1)) / 2.0
                                    if denom > 0:
                                        overlap_pct = overlap_len / denom
                                        overlap_m[v1, v2] = overlap_pct
                                        overlap_m[v2, v1] = overlap_pct
                                break
                        if has_overlap:
                            break
                else:
                    # Fallback: single-interval representation.
                    s1 = self.nodes_df.loc[v1, 'start']
                    e1 = self.nodes_df.loc[v1, 'end']
                    s2 = self.nodes_df.loc[v2, 'start']
                    e2 = self.nodes_df.loc[v2, 'end']
                    if e1 > s2 and s1 < e2:
                        has_overlap = True
                        overlap_len = min(e1, e2) - max(s1, s2)
                        if overlap_len > 0:
                            denom = ((e2 - s2) + (e1 - s1)) / 2.0
                            if denom > 0:
                                overlap_pct = overlap_len / denom
                                overlap_m[v1, v2] = overlap_pct
                                overlap_m[v2, v1] = overlap_pct

                if has_overlap:
                    intersection_m[v1, v2] = 1
                    intersection_m[v2, v1] = 1

        return intersection_m, overlap_m
    
    def get_document(self):  # preprocess_gene_opt
        """Extract all samples information from .junc files."""

        columns = ['chrom', 'chromStart', 'chromEnd', 'junc_id', 'score', 'strand', 'start', 'end', 'f1', 'f2', 'f3',
                   'f4']
        # samples_dfs = [pd.read_csv(file, sep='\t', names=columns) for file in samples_list]
        # samples_df = pd.concat(samples_dfs)
        # samples_df = samples_df[samples_df['score'] >= min_coverage].reset_index(drop=True)
        # junc_files_list = os.listdir(self.junc_path)
        gene_word_dict = {self.name: {}}
        # Keep sample ordering deterministic for reproducible multi-sample outputs.
        samples_list = [
            os.path.join(self.junc_path, s)
            for s in sorted(os.listdir(self.junc_path))
            if s.endswith('.junc')
        ]
        valid_samples = []
        for sample in samples_list:
            if '.gz' in sample:
                # with gzip.open(sample) as f:
                sample_df = pd.read_csv(sample, names=columns, sep='\t', skiprows=1, compression='gzip')
                sample_df = sample_df[['chrom', 'chromStart', 'chromEnd', 'score', 'strand']]
            else:
                sample_df = pd.read_csv(sample, names=columns, sep='\t', skiprows=1)
                sample_df = sample_df[['chrom', 'chromStart', 'chromEnd', 'score', 'strand']]

            if sample_df.shape[0] > 0:
                valid_samples.append(sample)
                gene_word_dict[self.name][sample] = {}
                # Preserve chromosome in per-sample aggregation for whole-sample inputs.
                group_cols = ['chromStart', 'chromEnd']
                if 'chrom' in sample_df.columns:
                    group_cols = ['chrom'] + group_cols
                if 'strand' in sample_df.columns:
                    group_cols.append('strand')
                sample_df = sample_df.groupby(group_cols, dropna=False)['score'].sum().reset_index()

                for idx, row in sample_df.iterrows():
                    start_row = row.chromStart
                    end_row = row.chromEnd
                    chrom = str(row.chrom) if 'chrom' in sample_df.columns else ""
                    key = str(start_row) + '-' + str(end_row)
                    if getattr(self, "_use_chrom_in_keys", False) and chrom:
                        key = chrom + ":" + key
                    gene_word_dict[self.name][sample][key] = row.score
        
        # Preserve the per-row sample ordering so we can align additional
        # features (e.g., multi-junction paths) later.
        self.sample_files = list(valid_samples)

        w2id_dict = {}
        id2w_dict = {}

        # Decide whether to include chromosome in intron tokens. If the node set
        # spans multiple chromosomes, the legacy "start-end" key would collide.
        try:
            chroms = set(str(c) for c in self.nodes_df.get("chrom", []).tolist())
        except Exception:
            chroms = set()
        self._use_chrom_in_keys = len([c for c in chroms if c and c != "nan"]) > 1

        for i in range(self.nodes_df.shape[0]):
            word = str(int(self.nodes_df.loc[i, 'start'])) + '-' + str(int(self.nodes_df.loc[i, 'end']))
            if self._use_chrom_in_keys and 'chrom' in self.nodes_df.columns:
                word = str(self.nodes_df.loc[i, 'chrom']) + ":" + word
            id2w_dict[i] = word
            w2id_dict[word] = i
        n_v = self.nodes_df.shape[0]
        document = np.zeros([len(valid_samples), n_v], dtype=np.int32)
        for sample_id in range(len(valid_samples)):
            for key in gene_word_dict[self.name][valid_samples[sample_id]]:
                # Some low-coverage introns may have been filtered out at
                # node-definition time (min_coverage) and therefore are not
                # present in w2id_dict/self.nodes_df. Skip those keys when
                # building the document matrix.
                idx = w2id_dict.get(key)
                if idx is None:
                    continue
                document[sample_id, idx] = gene_word_dict[self.name][valid_samples[sample_id]][key]

        return gene_word_dict, document, id2w_dict, w2id_dict
    
    def is_trainable(self):
        """This function determines if the minimum number of clusters for a gene is less than 2 (trivial case)"""
        # self.samples_df, self.samples_df_dict = self.get_sample_df()
        if len(self.samples_df) == 0:
            return False
        else:
            # self.nodes_df = self.get_junctions()
            # self.min_k = find_min_clusters(self.nodes_df)
            if self.min_k < 2:
                return False
            else:
                return True
    
    def preprocess(self):
        """
        Compute initial properties of the interval graph and document matrix.

        This is computationally heavy (O(V^2) interval graph, long-read path
        augmentation). To avoid repeating it for identical inputs, we cache
        the resulting state on disk keyed by a hash of the junction files.
        """
        # Where to store cache: the per-gene base result directory created
        # in __init__ is stable across runs (before _finalize_after_inference
        # adds run-specific subfolders).
        cache_dir = self.result_path or self.junc_path
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{self.name}_preproc_cache.pkl")
        sig_path = os.path.join(cache_dir, f"{self.name}_preproc_cache.sig")
        sig = self._build_preprocess_signature()

        # Always run the full preprocessing pipeline for each run. We still
        # write a cache (.pkl/.sig) for provenance, but we no longer reload
        # preprocessing state across different runs.
        self.word_dict, self.document, self.id2w_dict, self.w2id_dict = self.get_document()
        self._seed_base_node_introns()
        # Augment the intron-based representation with optional multi-junction
        # path nodes derived from long reads, so that the model can learn
        # directly from both short- and long-read evidence.
        self._augment_with_long_read_paths()
        self._rebuild_token_maps()

        # Interval graph and related quantities are built on the final node set
        # (single introns + any multi-junction paths).
        self.intersection, self.overlap_m = self._resolve_active_graph_and_k()
        if self.variant == "current":
            self.mvc = generalized_min_node_cover(self.intersection, i=2)
        else:
            self.mvc = []

        # Compute min_k and trainability from the precomputed conflict matrix,
        # so we do not rebuild the interval graph inside find_min_clusters().
        if self.min_k is None:
            self.min_k = find_min_clusters_from_intersection(self.intersection)
        if self.variant == "current":
            self.effective_k = None
        elif self.effective_k is None:
            self.effective_k = int(self.reference_k or 1)
        self.trainable = self.is_trainable()
        self.all_n_k = list(range(self.min_k, self.min_k + 19)) if self.trainable else []

        self.n_w_list = list(np.sum(self.document, axis=1))
        self.n_w = np.mean(self.n_w_list)
        self.n_v = self.nodes_df.shape[0]
        self.n_d = self.document.shape[0]
        self.document_tr, self.document_te, self.training_idx, self.test_idx = split_training_test(
            self.document, tr_percentage=95
        )

        # Use CliSAT with the existing conflict matrix to compute a maximum
        # independent set without recomputing the interval graph.
        self.mis, self.max_ind_set = _maximum_independent_set_from_intersection(self.intersection)

        # Optional: compute intron co-occurrence from Megadepth --junctions output, if present.
        # This captures how often introns co-occur in multi-junction reads and can be used by
        # the variational model as a soft prior over clusters.
        self.cooc_matrix = self._compute_cooccurrence_from_jxs()

        # For debugging and provenance, export the raw Megadepth junction
        # outputs (if present) into the gene-specific result directory.
        self._export_raw_junction_files()

        # Persist the heavy-weight state so subsequent runs with the same
        # junction inputs can skip preprocessing.
        state = {
            "word_dict": self.word_dict,
            "document": self.document,
            "id2w_dict": self.id2w_dict,
            "w2id_dict": self.w2id_dict,
            "intersection": self.intersection,
            "overlap_m": self.overlap_m,
            "mvc": self.mvc,
            "cooc_matrix": self.cooc_matrix,
            "node_introns": self.node_introns,
            "reference_k": self.reference_k,
            "effective_k": self.effective_k,
            "variant": self.variant,
            "n_w_list": self.n_w_list,
            "n_w": self.n_w,
            "n_v": self.n_v,
            "n_d": self.n_d,
            "document_tr": self.document_tr,
            "document_te": self.document_te,
            "training_idx": self.training_idx,
            "test_idx": self.test_idx,
            "mis": self.mis,
            "max_ind_set": self.max_ind_set,
            "nodes_df": self.nodes_df,
        }
        write_cache = str(os.environ.get("BSEEJ_WRITE_PREPROC_CACHE", "0")).strip().lower() in ("1", "true", "yes", "y")
        if write_cache:
            try:
                with open(cache_path, "wb") as fh:
                    pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
                with open(sig_path, "w") as fh:
                    fh.write(sig)
                print(f"[DEBUG] Saved preprocessing cache for gene {self.name} to {cache_path}")
            except Exception:
                # Caching is best-effort; ignore failures.
                pass

    def _debug_print_interval_graph(self):
        """
        Write a human-readable view of the interval graph (nodes + conflicts)
        to a text file under the gene-specific result directory.
        """
        import os

        # Determine output path (gene-specific result folder is created in Gene.__init__).
        out_dir = getattr(self, "result_path", None) or getattr(self, "junc_path", ".")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{self.name}_interval_graph.txt")

        V = self.nodes_df.shape[0]
        lines = []
        lines.append(f"Interval graph for gene {self.name}: {V} nodes\n")

        # Detect segment indices, if present.
        seg_indices = sorted(
            int(col.split('_')[-1])
            for col in self.nodes_df.columns
            if col.startswith("seg_start_")
        )
        use_segments = len(seg_indices) > 0

        # Print node intervals / segments.
        for v in range(V):
            label = self.nodes_df.loc[v, "graph_labels"]
            segments = []
            if use_segments:
                for si in seg_indices:
                    s = int(self.nodes_df.loc[v, f"seg_start_{si}"])
                    e = int(self.nodes_df.loc[v, f"seg_end_{si}"])
                    if e > s:
                        segments.append((s, e))
            else:
                s = int(self.nodes_df.loc[v, "start"])
                e = int(self.nodes_df.loc[v, "end"])
                segments.append((s, e))
            lines.append(f"node {v:3d} ({label}): segments={segments}\n")

        # Print adjacency lists from the intersection matrix.
        for v in range(V):
            neighbors = np.where(self.intersection[v] == 1)[0].tolist()
            if neighbors:
                lines.append(f"edges from node {v:3d}: {neighbors}\n")

        try:
            with open(out_path, "w") as fh:
                fh.writelines(lines)
            print(f"[DEBUG] Interval graph written to {out_path}")
        except OSError as exc:
            print(f"[DEBUG] Warning: failed to write interval graph to {out_path} ({exc})")

    def _export_raw_junction_files(self):
        """
        Copy Megadepth --junctions and --all-junctions outputs (*.jxs.tsv,
        *.all_jxs.tsv) from the junction directory into this gene's result
        directory, if available. This makes it easy to inspect the raw long-
        read and intron-level evidence alongside model outputs.
        """
        src_dir = getattr(self, "junc_path", None)
        dst_dir = getattr(self, "result_path", None)
        if src_dir is None or dst_dir is None:
            return
        if not os.path.isdir(src_dir):
            return
        os.makedirs(dst_dir, exist_ok=True)

        for fname in os.listdir(src_dir):
            if fname.endswith(".jxs.tsv") or fname.endswith(".all_jxs.tsv"):
                src = os.path.join(src_dir, fname)
                dst = os.path.join(dst_dir, fname)
                try:
                    shutil.copy2(src, dst)
                except OSError:
                    # Debug export is best-effort; ignore copy failures.
                    continue

    def _compute_cooccurrence_from_jxs(self):
        """
        Build an intron co-occurrence matrix from Megadepth --junctions (*.jxs.tsv) outputs.

        cooc[v,u] encodes how often intron v co-occurs with intron u in multi-junction reads,
        scaled by:
            - how frequently v appears in multi-junction reads, and
            - the fraction of v's total support that comes from such reads.

        Returns
        -------
        np.ndarray or None
            A (V,V) matrix of floats if any co-occurrence is observed; otherwise None.
        """
        # We need a mapping from "start-end" to intron index.
        if self.nodes_df is None or self.nodes_df.shape[0] == 0:
            return None
        if self.w2id_dict is None:
            return None

        # Discover *.jxs.tsv files (Megadepth --junctions outputs) in the same directory as .junc files.
        try:
            jxs_files = [
                os.path.join(self.junc_path, f)
                for f in os.listdir(self.junc_path)
                if f.endswith(".jxs.tsv")
            ]
        except FileNotFoundError:
            return None
        if not jxs_files:
            return None

        V = self.nodes_df.shape[0]
        cooc = np.zeros((V, V), dtype=np.float64)
        long_counts = np.zeros(V, dtype=np.float64)

        for path in jxs_files:
            try:
                with open(path, "r") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        fields = line.split("\t")
                        if len(fields) < 6:
                            continue
                        # Megadepth --junctions output may include 1 mate (6-7 fields)
                        # or 2 mates (14 fields). Each mate has its own chromosome.
                        coord_parts = [(fields[0], fields[5])]
                        if len(fields) > 12:
                            coord_parts.append((fields[7], fields[12]))
                        # If mates disagree on chromosome, treat as chimeric and ignore.
                        if len(coord_parts) > 1 and coord_parts[0][0] != coord_parts[1][0]:
                            continue

                        intron_idx_set = set()
                        for chrom, cstr in coord_parts:
                            if not cstr:
                                continue
                            for tok in cstr.split(","):
                                tok = tok.strip()
                                if not tok:
                                    continue
                                # Megadepth emits "start-end" 1-based coordinates; these
                                # match the keys we use in w2id_dict.
                                key = tok
                                if getattr(self, "_use_chrom_in_keys", False):
                                    key = f"{chrom}:{tok}"
                                if key in self.w2id_dict:
                                    intron_idx_set.add(self.w2id_dict[key])

                        intron_idx_list = sorted(intron_idx_set)
                        if len(intron_idx_list) < 2:
                            # We only care about true multi-junction reads here.
                            continue

                        # Track how often each intron participates in multi-junction reads.
                        for v in intron_idx_list:
                            long_counts[v] += 1.0

                        # Increment pairwise co-occurrence counts.
                        for i in range(len(intron_idx_list)):
                            v1 = intron_idx_list[i]
                            for j in range(i + 1, len(intron_idx_list)):
                                v2 = intron_idx_list[j]
                                cooc[v1, v2] += 1.0
                                cooc[v2, v1] += 1.0
            except OSError:
                continue

        if long_counts.sum() == 0.0 or not np.any(cooc):
            return None

        # Total intron counts across all samples (from the full document).
        total_counts = None
        try:
            if self.document is not None:
                total_counts = np.sum(self.document, axis=0).astype(np.float64)
        except Exception:
            total_counts = None

        # Build a co-occurrence prior matrix:
        #   cooc_prior[v,u] ~= P(u | v, multi-junction) * frac_long_reads(v)
        cooc_prior = np.zeros_like(cooc)
        for v in range(V):
            if long_counts[v] <= 0.0:
                continue
            row = cooc[v, :]
            if not np.any(row):
                continue
            # Conditional distribution of partners u given v in multi-junction reads
            row_norm = row / long_counts[v]
            weight_v = 1.0
            if total_counts is not None and total_counts[v] > 0.0:
                # Fraction of v's total support coming from multi-junction reads
                weight_v = float(long_counts[v] / total_counts[v])
            cooc_prior[v, :] = row_norm * weight_v

        if not np.any(cooc_prior):
            return None

        return cooc_prior

    def _augment_with_long_read_paths(self):
        """
        Augment the intron-level nodes and document matrix with multi-junction
        path nodes derived directly from Megadepth --junctions (*.jxs.tsv)
        outputs. Each path node represents a set of introns observed together
        in a single multi-junction read or read-pair.

        This allows the model to learn directly from long reads, in addition
        to the per-intron counts from short and long reads encoded in .junc.
        """
        # We require a document matrix and a stable mapping from intron
        # coordinates ("start-end") to intron indices.
        if getattr(self, "document", None) is None:
            return
        if getattr(self, "sample_files", None) is None:
            return
        if self.nodes_df is None or self.nodes_df.shape[0] == 0:
            return
        if self.w2id_dict is None:
            return

        from collections import defaultdict

        n_intron = self.nodes_df.shape[0]
        n_samples = self.document.shape[0]

        # Collect path-level counts across all samples:
        #   path_key (tuple of intron indices) -> {sample_idx: count}
        path_counts_by_sample = defaultdict(lambda: defaultdict(int))
        # Keep track of which introns make up each path so we can derive
        # multi-segment intervals per node.
        path_introns = {}     # path_key -> tuple of intron indices
        path_intervals = {}   # path_key -> (start, end)

        # Build a fast view of intron start/end coordinates (single intron nodes).
        intron_starts = self.nodes_df['start'].to_numpy()
        intron_ends = self.nodes_df['end'].to_numpy()

        for sample_idx, junc_path in enumerate(self.sample_files):
            # Derive the corresponding .jxs.tsv path from the .junc filename.
            base, ext = os.path.splitext(junc_path)
            jxs_path = base + ".jxs.tsv"
            if not os.path.exists(jxs_path):
                continue

            try:
                with open(jxs_path, "r") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        fields = line.split("\t")
                        if len(fields) < 6:
                            continue
                        # Megadepth --junctions output may include 1 mate (6-7 fields)
                        # or 2 mates (14 fields). Each mate has its own chromosome.
                        coord_parts = [(fields[0], fields[5])]
                        if len(fields) > 12:
                            coord_parts.append((fields[7], fields[12]))
                        # If mates disagree on chromosome, treat as chimeric and ignore.
                        if len(coord_parts) > 1 and coord_parts[0][0] != coord_parts[1][0]:
                            continue

                        intron_idx_set = set()
                        for chrom, cstr in coord_parts:
                            if not cstr:
                                continue
                            for tok in cstr.split(","):
                                tok = tok.strip()
                                if not tok:
                                    continue
                                # Megadepth emits "start-end" 1-based coordinates;
                                # these match the keys we use in w2id_dict for introns.
                                key = tok
                                if getattr(self, "_use_chrom_in_keys", False):
                                    key = f"{chrom}:{tok}"
                                if key in self.w2id_dict:
                                    intron_idx_set.add(self.w2id_dict[key])

                        if len(intron_idx_set) < 2:
                            # We only create path nodes for true multi-junction reads.
                            continue
                        # Use a sorted tuple of intron indices as a key; the
                        # order of introns within the path does not affect the
                        # independent-set constraint, but the set of introns
                        # does.
                        path_key = tuple(sorted(intron_idx_set))

                        # Cache the intron membership and union interval for this path.
                        if path_key not in path_introns:
                            path_introns[path_key] = path_key
                            idxs = np.array(path_key, dtype=int)
                            s_path = int(intron_starts[idxs].min())
                            e_path = int(intron_ends[idxs].max())
                            path_intervals[path_key] = (s_path, e_path)

                        path_counts_by_sample[path_key][sample_idx] += 1
            except OSError:
                continue

        if not path_intervals:
            # No multi-junction paths observed; nothing to augment.
            return

        # Assign new node indices to each unique path.
        path_keys = sorted(path_introns.keys())
        n_paths = len(path_keys)
        if n_paths == 0:
            return

        # Determine the maximum number of segments in any path so that we can
        # allocate a fixed set of segment columns on nodes_df.
        max_segments = max(len(path_introns[k]) for k in path_keys)
        max_segments = max(1, max_segments)

        # Ensure intron nodes have segment columns (degenerate single segment).
        for seg_idx in range(1, max_segments + 1):
            s_col = f"seg_start_{seg_idx}"
            e_col = f"seg_end_{seg_idx}"
            if s_col not in self.nodes_df.columns:
                self.nodes_df[s_col] = np.zeros(n_intron, dtype=np.int32)
            if e_col not in self.nodes_df.columns:
                self.nodes_df[e_col] = np.zeros(n_intron, dtype=np.int32)

        for row_idx in range(n_intron):
            self.nodes_df.loc[row_idx, "seg_start_1"] = int(self.nodes_df.loc[row_idx, "start"])
            self.nodes_df.loc[row_idx, "seg_end_1"] = int(self.nodes_df.loc[row_idx, "end"])
            for seg_idx in range(2, max_segments + 1):
                self.nodes_df.loc[row_idx, f"seg_start_{seg_idx}"] = 0
                self.nodes_df.loc[row_idx, f"seg_end_{seg_idx}"] = 0

        # Extend nodes_df with new path nodes, including segment columns.
        add_columns = ['start', 'length', 'end', 'graph_labels', 'node_labels']
        if 'chrom' in self.nodes_df.columns:
            add_columns = ['chrom'] + add_columns
        if 'strand' in self.nodes_df.columns:
            add_columns = add_columns + ['strand']
        seg_columns = [f"seg_start_{i}" for i in range(1, max_segments + 1)] + \
                      [f"seg_end_{i}" for i in range(1, max_segments + 1)]
        all_cols = add_columns + seg_columns
        # Use object dtype for chrom/labels columns; int for numeric fields.
        add_df = pd.DataFrame(columns=all_cols)
        for col in all_cols:
            if col in ("chrom", "graph_labels", "node_labels", "strand"):
                add_df[col] = ["" for _ in range(n_paths)]
            else:
                add_df[col] = np.zeros(n_paths, dtype=np.int32)

        graph_labels = []
        node_labels = []
        for offset, path_key in enumerate(path_keys):
            idx = n_intron + offset
            intron_idxs = np.array(path_introns[path_key], dtype=int)
            s_path, e_path = path_intervals[path_key]
            if 'chrom' in self.nodes_df.columns:
                # Path is built from introns within the same read; assume single chrom.
                add_df.loc[offset, 'chrom'] = str(self.nodes_df.loc[int(intron_idxs[0]), 'chrom'])
            add_df.loc[offset, 'start'] = s_path
            add_df.loc[offset, 'end'] = e_path
            add_df.loc[offset, 'length'] = e_path - s_path
            if 'strand' in self.nodes_df.columns:
                add_df.loc[offset, 'strand'] = str(self.nodes_df.loc[int(intron_idxs[0]), 'strand'])

            # Fill segment coordinates sorted by genomic start.
            order = np.argsort(intron_starts[intron_idxs])
            ordered_idxs = intron_idxs[order]
            for seg_pos, intr_idx in enumerate(ordered_idxs):
                s = int(intron_starts[intr_idx])
                e = int(intron_ends[intr_idx])
                add_df.loc[offset, f"seg_start_{seg_pos + 1}"] = s
                add_df.loc[offset, f"seg_end_{seg_pos + 1}"] = e
            # Remaining segments (if any) remain as zeros (padding).

            if 'chrom' in self.nodes_df.columns:
                chrom = str(add_df.loc[offset, 'chrom'])
                graph_labels.append(f"{chrom}:{s_path}_{e_path}")
            else:
                graph_labels.append(f"{s_path}_{e_path}")
            node_labels.append(str(idx))
        add_df['graph_labels'] = graph_labels
        add_df['node_labels'] = node_labels

        # Concatenate intron and path nodes; reset indices.
        self.nodes_df = pd.concat([self.nodes_df, add_df], axis=0, ignore_index=True)

        # Extend the document matrix with per-sample path counts.
        doc_aug = np.zeros((n_samples, n_intron + n_paths), dtype=np.int32)
        doc_aug[:, :n_intron] = self.document
        for path_idx, path_key in enumerate(path_keys):
            col = n_intron + path_idx
            counts_for_path = path_counts_by_sample[path_key]
            for sample_idx, cnt in counts_for_path.items():
                if 0 <= sample_idx < n_samples:
                    doc_aug[sample_idx, col] = int(cnt)
        self.document = doc_aug

        # Extend id2w_dict for interpretability; keep w2id_dict focused on
        # intron nodes so that Megadepth tokens ("start-end") continue to map
        # only to intron indices. For path nodes we store the union interval
        # as "start-end" so downstream utilities remain compatible.
        if self.id2w_dict is None:
            self.id2w_dict = {}
        for path_idx, path_key in enumerate(path_keys):
            idx = n_intron + path_idx
            s_path, e_path = path_intervals[path_key]
            if getattr(self, "_use_chrom_in_keys", False) and 'chrom' in self.nodes_df.columns:
                chrom = str(self.nodes_df.loc[int(path_introns[path_key][0]), 'chrom'])
                self.id2w_dict[idx] = f"{chrom}:{s_path}-{e_path}"
            else:
                self.id2w_dict[idx] = f"{s_path}-{e_path}"

        if not self.node_introns:
            self._seed_base_node_introns()
        for path_key in path_keys:
            ordered_introns = []
            for intr_idx in np.argsort(intron_starts[np.array(path_introns[path_key], dtype=int)]):
                base_idx = int(np.array(path_introns[path_key], dtype=int)[intr_idx])
                intr_chain = self.node_introns[base_idx] if base_idx < len(self.node_introns) else ()
                if intr_chain:
                    ordered_introns.append(intr_chain[0])
            self.node_introns.append(tuple(ordered_introns))
