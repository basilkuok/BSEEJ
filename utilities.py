import gzip
import os
import pickle
import random
import subprocess
import tempfile
from collections import Counter
from copy import deepcopy

try:
    import arviz as az
except Exception:  # optional dependency (often breaks on SciPy/ArviZ mismatch)
    az = None
import numba as nb
import numpy as np
import pandas as pd
from numba import jit


def _detect_clisat_binary():
    """
    Locate a usable CliSAT binary, if available.

    Search order:
      1) Environment variable CLISAT_BIN
      2) Repository-local CliSAT/bin/CliSAT_2024_07_06
      3) Repository-local CliSAT/bin/CliSAT_2024_07_06_ub16.04

    Returns
    -------
    str or None
        Absolute path to the CliSAT binary, or None if not found / not executable.
    """
    # Explicit override via environment variable.
    env_bin = os.environ.get("CLISAT_BIN")
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin

    # Try repo-local binaries (only valid on Linux).
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        here = os.getcwd()
    root_dir = os.path.dirname(here)
    candidate_dirs = [
        os.path.join(here, "CliSAT", "bin"),       # CliSAT inside JARVIS_final
        os.path.join(root_dir, "CliSAT", "bin"),   # CliSAT beside JARVIS_final
    ]
    for clisat_dir in candidate_dirs:
        candidates = [
            os.path.join(clisat_dir, "CliSAT_2024_07_06"),
            os.path.join(clisat_dir, "CliSAT_2024_07_06_ub16.04"),
        ]
        for path in candidates:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
    return None


def _run_clisat_max_clique(n_vertices, edges, time_limit=None, ordering=2, heuristic=1):
    """
    Call the CliSAT binary on a graph given by a vertex count and edge list.

    Parameters
    ----------
    n_vertices : int
        Number of vertices (labeled 0 .. n_vertices-1).
    edges : iterable of (int, int)
        Undirected edges with 0-based endpoints.
    time_limit : int or None
        Time limit in seconds. If None, a default of 3600 is used.
    ordering : int
        CliSAT vertex ordering parameter (1 = DEG-SORT, 2 = COLOR-SORT).
    heuristic : int
        CliSAT heuristic parameter (0 = none, 1 = AMTS 0.05 s).

    Returns
    -------
    (int, list[int]) or (None, None)
        Tuple (omega, clique_vertices) on success, where clique_vertices are 0-based.
        Returns (None, None) if CliSAT is not available or the call fails.
    """
    clisat_bin = _detect_clisat_binary()
    if clisat_bin is None:
        raise RuntimeError(
            "CliSAT binary not found. Please clone the CliSAT repository under "
            "'BSEEJ_final/CliSAT', ensure an executable binary exists in "
            "'CliSAT/bin' (e.g., CliSAT_2024_07_06), or set CLISAT_BIN to the "
            "full path of the CliSAT executable."
        )

    if time_limit is None:
        # Allow user to override via environment; fall back to 3600 s.
        try:
            time_limit = int(os.environ.get("CLISAT_TIMELIMIT", "3600"))
        except ValueError:
            time_limit = 3600

    # Prepare a temporary DIMACS .clq file.
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".clq", delete=False) as tmp:
            tmp_path = tmp.name
            # Minimal DIMACS format: p edge n m followed by undirected edges.
            m_edges = len(edges)
            tmp.write(f"p edge {n_vertices} {m_edges}\n")
            for u, v in edges:
                # DIMACS vertices are 1-based.
                tmp.write(f"e {u + 1} {v + 1}\n")
    except OSError as exc:
        raise RuntimeError(f"Failed to create temporary DIMACS file for CliSAT: {exc}")

    try:
        cmd = [
            clisat_bin,
            tmp_path,
            str(int(time_limit)),
            str(int(ordering)),
            str(int(heuristic)),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Optional debug: dump command and a snippet of CliSAT output
    if os.environ.get("CLISAT_DEBUG", "").strip():
        print("[CliSAT DEBUG] Command:", " ".join(cmd))
        print("[CliSAT DEBUG] Exit code:", result.returncode)
        print("[CliSAT DEBUG] Stdout (first 40 lines):")
        for i, line in enumerate(result.stdout.splitlines()):
            if i >= 40:
                break
            print(line)
        if result.stderr.strip():
            print("[CliSAT DEBUG] Stderr:", result.stderr.strip())

    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(
            f"CliSAT failed with exit code {result.returncode}. "
            f"Stderr: {result.stderr.strip()}"
        )

    # Robust parsing of CliSAT output. Different builds may emit either
    #   omega:K
    # or
    #   w:K ...
    # (sometimes both, e.g. incumbent + final optimum), and usually a final
    # line of the form:
    #   v1 v2 ... vK  [K]
    # where v_i are 1-based vertex IDs and the bracketed K repeats the clique
    # size.  We treat the *last* reported w/omega as the true optimum and the
    # last such bracketed line as the explicit clique.
    import re

    text = result.stdout
    omega = None
    clique_vertices = None

    # 1) Primary path: scan from the bottom for a line of the form
    #       v1 v2 ... vK [K]
    #    and extract the vertex IDs and K from that line only.  This avoids
    #    accidentally mixing integers from multiple lines.
    lines = text.splitlines()
    for raw in reversed(lines):
        line = raw.strip()
        if "[" not in line or "]" not in line:
            continue
        try:
            # Split on the last '[' and take the numeric content up to ']'.
            prefix, bracket = line.rsplit("[", 1)
            inside = bracket.split("]", 1)[0]
            nums_inside = re.findall(r"\d+", inside)
            if not nums_inside:
                continue
            k = int(nums_inside[-1])
        except Exception:
            continue

        # Integers before '[' are interpreted as vertex IDs (1‑based).
        tokens_all = re.findall(r"\d+", prefix)
        if not tokens_all:
            continue
        try:
            verts = [int(t) - 1 for t in tokens_all]
        except Exception:
            continue

        # If there are more integers than K (for any reason), take the last K.
        if len(verts) >= k:
            clique_vertices = verts[-k:]
        else:
            clique_vertices = verts
        omega = k
        break

    # 2) If, for some reason, we did not see such a bracketed clique line, fall
    #    back to parsing the last reported 'omega:K' or 'w:K' from the whole
    #    CliSAT output.  In that case we may not have the explicit vertex set.
    if omega is None:
        omega_matches = re.findall(r"omega\s*[:=]\s*(\d+)", text)
        w_matches = re.findall(r"w\s*[:=]\s*(\d+)", text)
        try:
            if omega_matches:
                omega = int(omega_matches[-1])
            elif w_matches:
                omega = int(w_matches[-1])
        except ValueError:
            omega = None

    # Extra debug: report what we parsed from CliSAT.
    if os.environ.get("CLISAT_DEBUG", "").strip():
        print(f"[CliSAT DEBUG] Parsed omega: {omega}")
        if clique_vertices is not None:
            print(f"[CliSAT DEBUG] Parsed clique size: {len(clique_vertices)}")
        else:
            print("[CliSAT DEBUG] No clique vertices parsed.")

    if omega is None:
        raise RuntimeError(
            "CliSAT did not report a clique size (omega/w). "
            "Please inspect the CliSAT output and configuration."
        )
    if clique_vertices is None:
        clique_vertices = []
    return omega, clique_vertices


def compute_df(n_sample, effective_k, n_introns, result_df, gene_name, z_matrix, starts, ends):
    counter = 0
    for sample_id in range(n_sample):
        for cl in range(effective_k):
            for intr in range(n_introns):
                # start = int(id2w[intr].split('-')[0])
                # end = int(id2w[intr].split('-')[1])
                result_df.iloc[counter, :] = gene_name, cl, intr, starts[intr], ends[intr], sample_id, z_matrix[
                    sample_id, intr, cl]
                counter += 1


def compute_df_vectorized(n_sample, effective_k, n_introns, result_df, gene_name, z_matrix, starts, ends):
    cls, intrs, startss, endss, sample_ids, zs = getvecs(n_sample * effective_k * n_introns, n_sample, effective_k,
                                                         n_introns, starts, ends, z_matrix)
    result_df.gene = gene_name
    result_df.trans_id = cls
    result_df["index"] = intrs
    result_df.start = startss
    result_df.end = endss
    result_df["sample"] = sample_ids
    result_df.FPKM = zs


@nb.jit(nb.types.UniTuple(nb.int32[:], 6)(nb.int32, nb.int32, nb.int32, nb.int32, nb.int32[:], nb.int32[:],
                                          nb.int32[:, :, :]), nopython=True)
def getvecs(overallsize, n_sample, effective_k, n_introns, starts, ends, z_matrix):
    cls = np.zeros(overallsize, dtype=np.int32)
    intrs = np.zeros(overallsize, dtype=np.int32)
    startss = np.zeros(overallsize, dtype=np.int32)
    endss = np.zeros(overallsize, dtype=np.int32)
    sample_ids = np.zeros(overallsize, dtype=np.int32)
    zs = np.zeros(overallsize, dtype=np.int32)
    idx = 0
    for sample_id in range(n_sample):
        for cl in range(effective_k):
            for intr in range(n_introns):
                cls[idx] = cl
                intrs[idx] = intr
                startss[idx] = starts[intr]
                endss[idx] = ends[intr]
                sample_ids[idx] = sample_id
                zs[idx] = z_matrix[sample_id, intr, cl]
                idx += 1
    return cls, intrs, startss, endss, sample_ids, zs


def get_lo(intersection_m):
    lo = np.zeros([intersection_m.shape[0], 1], dtype=np.int32)
    # compute lo
    for node in range(intersection_m.shape[0]):
        
        lo_set = []
        all_adj = np.where(intersection_m[node, :] == 1)[0]
        for adj in all_adj:
            if adj < node:
                lo_set.append(adj)
        
        if len(lo_set) == 0:
            lo_set.append(node)
        
        lo[node] = min(lo_set)
    return lo


def generalized_min_node_cover(intersection_m, i=2):
    """Compute minimum node cover from the generalized min node cover algorithm."""
    lo = get_lo(intersection_m)
    w = np.zeros([intersection_m.shape[0], 1], dtype=np.int32)
    mvc = []
    
    for node in range(intersection_m.shape[0]):
        must = False
        for u in range(int(lo[node]), node + 1):
            w[u] += 1
            if w[u] == i:
                must = True
        if must:
            mvc.append(node)
            for u in range(int(lo[node]), node + 1):
                w[u] -= 1
    return mvc


def find_min_clusters(nodes_df):
    """
    Compute the minimum number of clusters (colors) required for the interval
    graph induced by nodes_df. For an interval graph, this is equal to the
    size of a maximum clique.

    This function **only** uses the CliSAT maximum clique solver. If CliSAT
    is not available or fails, a RuntimeError is raised.
    """
    intersection_m, edges_list = get_conflict_for_plot(nodes_df)
    n_v = intersection_m.shape[0]
    if n_v <= 1:
        # Trivial graphs do not require CliSAT.
        return 1

    # CliSAT returns (omega, clique_vertices); we only need omega here.
    omega, _ = _run_clisat_max_clique(n_v, edges_list)
    return omega


def find_min_clusters_from_intersection(intersection_m):
    """
    Compute the clique number / minimum number of colours from a precomputed
    conflict matrix ``intersection_m``, without rebuilding the interval graph
    from ``nodes_df``.

    Parameters
    ----------
    intersection_m : np.ndarray, shape (V, V)
        Symmetric {0,1} adjacency matrix with zeros on the diagonal.

    Returns
    -------
    int
        The size of a maximum clique in the interval graph, equal to the
        chromatic number for interval graphs.
    """
    V = intersection_m.shape[0]
    if V <= 1:
        # Trivial graphs do not require CliSAT.
        return 1
    edges_list = []
    for v1 in range(V):
        row = intersection_m[v1]
        for v2 in range(v1 + 1, V):
            if row[v2] == 1:
                edges_list.append((v1, v2))
    omega, _ = _run_clisat_max_clique(V, edges_list)
    return omega


def get_conflict_for_plot(nodes_df):
    """Find the intervals that have intersection.

    When per-node segment columns (seg_start_i/seg_end_i) are present, treat
    each node as a union of segments and declare a conflict when any pair of
    segments overlaps. Otherwise, fall back to a single-interval check.
    """
    V = nodes_df.shape[0]
    intersection_m = np.zeros([V, V], dtype=np.int32)
    edges_list = []

    seg_indices = sorted(
        int(col.split('_')[-1])
        for col in nodes_df.columns
        if col.startswith('seg_start_')
    )
    use_segments = len(seg_indices) > 0

    for v1 in range(V):
        for v2 in range(v1 + 1, V):
            has_overlap = False
            if use_segments:
                for si in seg_indices:
                    s1 = int(nodes_df.loc[v1, f"seg_start_{si}"])
                    e1 = int(nodes_df.loc[v1, f"seg_end_{si}"])
                    if e1 <= s1:
                        continue
                    for sj in seg_indices:
                        s2 = int(nodes_df.loc[v2, f"seg_start_{sj}"])
                        e2 = int(nodes_df.loc[v2, f"seg_end_{sj}"])
                        if e2 <= s2:
                            continue
                        if e1 > s2 and s1 < e2:
                            has_overlap = True
                            break
                    if has_overlap:
                        break
            else:
                s1 = nodes_df.loc[v1, 'start']
                e1 = nodes_df.loc[v1, 'end']
                s2 = nodes_df.loc[v2, 'start']
                e2 = nodes_df.loc[v2, 'end']
                if e1 > s2 and s1 < e2:
                    has_overlap = True

            if has_overlap:
                intersection_m[v1, v2] = 1
                intersection_m[v2, v1] = 1
                edges_list.append((v1, v2))

    return intersection_m, edges_list


def split_training_test(document_orig, tr_percentage=95):
    tr_size = int(tr_percentage / 100 * document_orig.shape[0])
    indices = np.random.RandomState(seed=2021).permutation(document_orig.shape[0])
    training_idx, test_idx = indices[:tr_size], indices[tr_size:]
    document = document_orig[training_idx, :]
    document_te = document_orig[test_idx, :]
    return document, document_te, training_idx, test_idx


def find_mis(nodes_df):
    """
    Compute the size of a maximum independent set (MIS) and one such set.

    This function **only** uses CliSAT on the complement graph (where cliques
    correspond to independent sets in the original graph). If CliSAT is not
    available or fails, a RuntimeError is raised.
    """
    intersection_m, _ = get_conflict_for_plot(nodes_df)
    mis_size, max_ind_set = _maximum_independent_set_from_intersection(intersection_m)
    if not max_ind_set:
        raise RuntimeError(
            "CliSAT did not provide a maximum independent set for the conflict graph."
        )
    return mis_size, sorted(max_ind_set)


def _maximum_independent_set_from_intersection(intersection_m, candidate_vertices=None):
    """
    Helper: compute a maximum independent set of the conflict graph whose
    adjacency is given by ``intersection_m``.

    Parameters
    ----------
    intersection_m : np.ndarray, shape (V, V)
        Symmetric {0,1} adjacency matrix with zeros on the diagonal.
    candidate_vertices : optional iterable of int
        If provided, restrict the MIS search to the subgraph induced by this
        vertex subset.

    Returns
    -------
    (int, list[int])
        Size of the maximum independent set and a list of vertex indices in
        the original graph.
    """
    n_v = intersection_m.shape[0]
    if candidate_vertices is None:
        verts = np.arange(n_v, dtype=np.int32)
    else:
        verts = np.array(list(candidate_vertices), dtype=np.int32)

    n_sub = verts.shape[0]
    if n_sub == 0:
        return 0, []

    # Build complement edges for the induced subgraph on verts.
    comp_edges = []
    for i in range(n_sub):
        gi = verts[i]
        row = intersection_m[gi]
        for j in range(i + 1, n_sub):
            gj = verts[j]
            if row[gj] == 0:
                comp_edges.append((i, j))

    mis_size, clique_vertices_local = _run_clisat_max_clique(n_sub, comp_edges)
    if not clique_vertices_local:
        return mis_size, []
    mis_global = [int(verts[int(lv)]) for lv in clique_vertices_local]
    return mis_size, mis_global


def get_initialization(nodes_df, n_k):
    _, edges_list = get_conflict_for_plot(nodes_df)
    gra = generate_interval_graph_nx(nodes_df, edges_list, intervalviz=False)
    all_max_ind_set = []
    while len(all_max_ind_set) < n_k:
        temp = nx.maximal_independent_set(gra)
        if temp not in all_max_ind_set:
            all_max_ind_set.append(temp)
    return all_max_ind_set


def find_initial_nodes(nodes_df, n_k):
    _, edges_list = get_conflict_for_plot(nodes_df)
    gra = generate_interval_graph_nx(nodes_df, edges_list, intervalviz=False)
    all_max_ind_set = []
    for i in range(1000):
        temp = nx.maximal_independent_set(gra)
        if temp not in all_max_ind_set:
            all_max_ind_set.append(temp)
    
    while len(all_max_ind_set) < n_k:
        temp = nx.maximal_independent_set(gra)
        all_max_ind_set.append(temp)
    return all_max_ind_set


def add_node_is_beta(s, gene_intersection, n_v, bet):
    free = set(range(n_v)) - set(s)
    for ss in s:
        neighbor_ss = set(np.where(gene_intersection[ss, :] == 1)[0])
        free = free - neighbor_ss
        if len(free) == 0:
            return []
    add_node = random.choices(list(free), weights=bet[list(free)] / np.sum(bet[list(free)]), k=1)
    return add_node


def del_node_is_beta(s, bet):
    if len(s) <= 1:
        return s
    else: 
        return random.choices(list(s), weights=1 - (bet[s] / np.sum(bet[s])), k=1)


def sample_local_ind_set(gene_intersection, n_v, n_s, b_k, beta_k, mis):
    max_trial = 200
    
    s = list(np.where(b_k)[0])
    s.sort()
    random_clusters = []
    temp2 = deepcopy(s)
    random_clusters.append(temp2)
    trial = 0
    while len(random_clusters) < n_s and trial < max_trial:
        trial += 1
        rnd = np.random.binomial(n=1, p=1 - (len(s) / mis))
        if rnd >= 0.5:
            an = add_node_is_beta(s, gene_intersection, n_v, beta_k)
            if len(an) != 0:
                s.append(an[0])
                s.sort()
                temp = deepcopy(s)
                if temp not in random_clusters:
                    random_clusters.append(temp)
        else:
            dn = del_node_is_beta(s, beta_k)
    
            if len(dn) != 0 and len(s) != 1:
                s.remove(dn[0])
                s.sort()
                temp = deepcopy(s)
                if temp not in random_clusters and len(temp) > 0:
                    random_clusters.append(temp)
    
    return random_clusters


def find_duplicate_clusters(b):
    inputs = map(tuple, b)
    
    freq_dict = Counter(inputs)
    
    duplicated_clusters = [row for row in freq_dict.keys() if freq_dict[row] > 1]
    
    return duplicated_clusters


def merge_suplicate_clusters(b, z):
    dup_cl = find_duplicate_clusters(b)
    
    while len(dup_cl) > 0:
        print('hit:', len(dup_cl))
        dup0 = np.array(dup_cl[0])
        dup0_indices = list(np.where(np.all(b == dup0, axis=1))[0])
        removing_indices = dup0_indices[1:]
        
        for dd in removing_indices[::-1]:
            b = np.delete(b, dd, 0)
        
        z[:, :, dup0_indices[0]] = np.sum(z[:, :, dup0_indices], axis=2)
        
        for dd in removing_indices[::-1]:
            z = np.delete(z, dd, 2)
        
        dup_cl = find_duplicate_clusters(b)
    return b, z


def save_results(gene, model):
    print('Saving the results for gene', gene.name)
    method_label = str(model.run_info.get('inference', 'unknown'))
    idx_suffix = getattr(gene, 'idx_suffix', '') or getattr(model, 'idx_suffix', '') or model.run_info.get('idx_suffix', '')
    suffix = f"_{idx_suffix}" if idx_suffix else ""
    # the comb_name is replaced for VI because filename is too long
    comb_suffix = f"_K_{model.run_info['N_K']}" + suffix
    comb_name = 'gene_' + gene.name + '_alpha_' + str(model.alpha) + '_eta_' + str(model.eta) + '_epsilon_' + \
               str(model.epsilon) + '_rs_' + str(model.r) + comb_suffix + '_method_' + method_label
    #comb_name_VI = 'gene_' + gene.name + '_alpha_' + str(model.alpha) + '_eta_' + '_epsilon_' + \
                #str(model.epsilon) + '_rs_' + str(model.r) + '_K_' + str(model.run_info['N_K'])
    # For VI modes, avoid depending on per-iteration run_info snapshots
    # (which are expensive to store). Prefer the final variational state.
    last_z = model.run_info.get('final_phi', None)
    last_b = model.run_info.get('final_b_mask', None)
    if last_z is None or last_b is None:
        # Backward-compatible fallback: use the last recorded snapshot if present.
        last_run = list(model.run_info['gibbs'])[-1]
        last_z = deepcopy(model.run_info['gibbs'][last_run]['Z'])
        last_b = deepcopy(model.run_info['gibbs'][last_run]['b'])
    new_b, new_z = merge_suplicate_clusters(np.asarray(last_b), np.asarray(last_z))
    model.run_info['new_b'] = deepcopy(new_b)
    model.run_info['new_Z'] = deepcopy(new_z)
    # ensure the per-run result directory exists
    if not os.path.exists(gene.result_path):
        os.mkdir(gene.result_path)
    
    z_matrix = model.run_info['new_Z']
    id2w = model.run_info['id2w_dict']
    n_sample = z_matrix.shape[0]
    n_introns = z_matrix.shape[1]
    effective_k = z_matrix.shape[2]
    gene_name = model.run_info['gene']
    
    def _parse_start_end(token: str):
        # Supports legacy "start-end" and extended "chrom:start-end".
        tok = str(token)
        if ":" in tok:
            tok = tok.split(":", 1)[1]
        parts = tok.split("-")
        if len(parts) < 2:
            raise ValueError(f"Cannot parse intron token '{token}'")
        return int(parts[0]), int(parts[1])

    starts = np.asarray([_parse_start_end(id2w[j])[0] for j in range(n_introns)], np.int32)
    ends = np.asarray([_parse_start_end(id2w[j])[1] for j in range(n_introns)], np.int32)
    
    result_df = pd.DataFrame(data=0, columns=['gene', 'trans_id', 'index', 'start', 'end', 'sample', 'FPKM'],
                             index=range(n_sample * effective_k * n_introns))

    compute_df_vectorized(n_sample, effective_k, n_introns, result_df, gene_name, z_matrix, starts, ends)
    
    csv_suffix = f"_K_{effective_k}" + suffix
    file_name_2 = f"bseej_{gene_name}_{method_label}{csv_suffix}.csv"
    result_df.to_csv(gene.result_path + '/' + file_name_2)
    print(gene.result_path + '/' + file_name_2, 'saved.')

    # NOTE: GTF/count export is triggered post-inference in Model._finalize_after_inference
    # so -save_result only controls this CSV.
    return gene.result_path + '/' + file_name_2


def _export_bseej_transcripts_gtf_and_counts(gene, model):
    """
    Export a per-gene GTF where each BSEEJ cluster is represented as a transcript
    composed from the cluster's introns.

    Notes / rigor:
    - BSEEJ operates on introns; transcript start/end are not identifiable from
      introns alone. To avoid using reference annotations (leakage), we construct
      minimal terminal exons of length 1 bp adjacent to the first/last intron.
    - Internal exons are inferred deterministically from splice sites:
        exon_start = prev_intron_end + 1
        exon_end   = next_intron_start - 1
      using 1-based inclusive coordinates (STAR SJ.out convention).
    - Nodes that represent multi-junction paths (seg_start_*/seg_end_*) are
      currently excluded from this export to keep the mapping unambiguous.
    """
    import os
    import sys
    import numpy as np

    out_dir = getattr(gene, "result_path", None) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Prefer final state; fall back to legacy 'new_b' when present.
    b = model.run_info.get("final_b_mask", None)
    if b is None:
        b = model.run_info.get("new_b", None)
    b = np.asarray(b) if b is not None else None
    if b is None or b.ndim != 2:
        raise ValueError("Expected model.run_info['final_b_mask'] (or legacy 'new_b') to be a (K,V) matrix")
    K, V = int(b.shape[0]), int(b.shape[1])

    # Prefer final variational state (CAVI/SVI).
    doc = np.asarray(getattr(gene, "document", None)) if getattr(gene, "document", None) is not None else None
    phi = model.run_info.get("final_phi", None)
    phi = np.asarray(phi) if phi is not None else None
    if doc is None or doc.ndim != 2:
        raise ValueError("Gene.document missing/invalid for export")
    if phi is None or phi.ndim != 3:
        raise ValueError("model.run_info['final_phi'] missing/invalid for export")
    D = int(doc.shape[0])

    # Beta ("topics over nodes") is represented by zeta; we use its mean to rank nodes.
    zeta = model.run_info.get("final_zeta", None)
    if zeta is None:
        # Try last snapshot (CAVI/SVI history stores 'zeta')
        try:
            last_run = list(model.run_info.get("gibbs", {}))[-1]
            zeta = model.run_info["gibbs"][last_run].get("zeta")
        except Exception:
            zeta = None
    if zeta is not None:
        zeta = np.asarray(zeta, dtype=float)
        # If b was merged elsewhere, zeta may still be unmerged. Keep strictness
        # for now: mismatched shapes likely indicate an inconsistent export state.
        if zeta.shape != (K, V):
            raise ValueError(f"Expected zeta to have shape (K,V)=({K},{V}); got {zeta.shape}")
        zeta_sum = np.sum(zeta, axis=1, keepdims=True)
        zeta_sum = np.where(zeta_sum > 0, zeta_sum, 1.0)
        beta_mean = zeta / zeta_sum  # E[beta_{k,v}] up to normalization
    else:
        beta_mean = None

    nodes_df = getattr(gene, "nodes_df", None)
    if nodes_df is None or nodes_df.shape[0] != V:
        raise ValueError("Gene.nodes_df missing or not aligned with V")
    if "start" not in nodes_df.columns or "end" not in nodes_df.columns:
        raise ValueError("Gene.nodes_df must contain 'start' and 'end'")

    # Detect whether multi-junction path nodes exist (seg_start_i/seg_end_i).
    # In this codebase, single-intron nodes are encoded as segment 1 with
    # seg_start_1/start and seg_end_1/end, while path nodes have multiple
    # non-zero segments.
    seg_start_cols = sorted([c for c in nodes_df.columns if str(c).startswith("seg_start_")])
    seg_end_cols = sorted([c for c in nodes_df.columns if str(c).startswith("seg_end_")])
    max_segments = 0
    if seg_start_cols and seg_end_cols:
        try:
            max_segments = max(
                int(str(c).split("_")[-1])
                for c in seg_start_cols
                if str(c).split("_")[-1].isdigit()
            )
        except Exception:
            max_segments = 0

    def _node_introns(v_idx: int):
        """
        Return a list of intron intervals for node v.
        For path nodes, returns multiple (start,end) segments.
        For intron nodes, returns the single (start,end).
        """
        if max_segments <= 0:
            return [(int(starts[v_idx]), int(ends[v_idx]))]
        out = []
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
            out = [(int(starts[v_idx]), int(ends[v_idx]))]
        return out

    def _is_path_node(v_idx: int) -> bool:
        if max_segments <= 1:
            return False
        segs = _node_introns(v_idx)
        return len(segs) >= 2

    starts = nodes_df["start"].to_numpy().astype(int)
    ends = nodes_df["end"].to_numpy().astype(int)

    if "chrom" in nodes_df.columns:
        chroms = nodes_df["chrom"].astype(str).to_numpy()
    else:
        chroms = np.array(["."] * V, dtype=object)

    if "strand" in nodes_df.columns:
        strands = nodes_df["strand"].astype(str).to_numpy()
    else:
        strands = np.array(["."] * V, dtype=object)

    # Sample names (for the count matrix header).
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
        # Fallback: numeric IDs if we cannot reconcile.
        sample_names = [f"sample_{i}" for i in range(D)]

    gtf_b_path = os.path.join(out_dir, f"{gene.name}_bseej_predicted_b.gtf")
    gtf_beta_path = os.path.join(out_dir, f"{gene.name}_bseej_predicted_beta.gtf")
    counts_b_path = os.path.join(out_dir, f"{gene.name}_bseej_counts_b.tsv")
    counts_beta_path = os.path.join(out_dir, f"{gene.name}_bseej_counts_beta.tsv")

    def _emit_gtf_line(fh, chrom, source, feature, start, end, strand, attrs):
        fh.write(
            "\t".join(
                [
                    str(chrom),
                    str(source),
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

    # Build transcripts and a transcript-by-sample count matrix for two decodings:
    #   (A) b-based (structural mask): members = {v : b[k,v] == 1}
    #   (B) beta-based (soft): members selected from E[beta_{k,v}] without any
    #       post-inference constraint projection. By default we select the smallest
    #       top-mass set reaching BSEEJ_EXPORT_BETA_MASS cumulative mass; an optional
    #       top-fraction-by-rank mode is available via BSEEJ_EXPORT_BETA_TOP_FRACTION.
    #
    # Important: we do NOT apply any additional MWIS/constraint projection at export time.
    # If a chosen set cannot be represented as a valid transcript (overlapping introns,
    # invalid exon inference), we skip emitting that transcript rather than modifying it.
    beta_top_fraction = None
    try:
        beta_top_fraction_raw = os.environ.get("BSEEJ_EXPORT_BETA_TOP_FRACTION", "").strip()
        if beta_top_fraction_raw != "":
            beta_top_fraction = float(beta_top_fraction_raw)
    except Exception:
        beta_top_fraction = None
    if beta_top_fraction is not None:
        beta_top_fraction = min(max(beta_top_fraction, 0.0), 1.0)

    beta_mass = 0.80
    try:
        beta_mass = float(os.environ.get("BSEEJ_EXPORT_BETA_MASS", "0.80"))
    except Exception:
        beta_mass = 0.80
    beta_mass = min(max(beta_mass, 0.0), 1.0)

    # Persist the minimal decoding state needed to (re)construct predicted_b/predicted_beta
    # transcripts later without re-running inference.
    try:
        try:
            flank_len = int(os.environ.get("BSEEJ_EXPORT_FLANK_EXON_LEN", "1"))
        except Exception:
            flank_len = 1
        flank_len = max(1, int(flank_len))

        state_path = os.path.join(out_dir, f"{gene.name}_bseej_decoding_state_K_{K}.npz")
        state = {
            "b_mask": b.astype(np.int8, copy=False),
            "K": np.int32(K),
            "V": np.int32(V),
            # Record export-time parameters for reproducibility.
            "export_beta_mass": np.float32(beta_mass),
            "export_beta_top_fraction": (
                np.float32(beta_top_fraction) if beta_top_fraction is not None else np.float32(-1.0)
            ),
            "export_flank_exon_len": np.int32(flank_len),
        }
        if zeta is not None:
            state["zeta"] = np.asarray(zeta, dtype=np.float32)
        if beta_mean is not None:
            state["beta_mean"] = np.asarray(beta_mean, dtype=np.float32)
        np.savez_compressed(state_path, **state)
    except Exception as exc:
        print(f"[WARN] Failed to write decoding state for {gene.name}: {exc}", file=sys.stderr)

    def _select_members_beta(k_idx: int) -> np.ndarray | None:
        if beta_mean is None:
            return None
        probs_all = beta_mean[k_idx, :].astype(float)
        order = np.argsort(-probs_all)
        if beta_top_fraction is not None:
            n_keep = int(np.ceil(float(V) * float(beta_top_fraction)))
            n_keep = max(1, min(V, n_keep))
            return order[:n_keep].astype(int)

        probs_sorted = probs_all[order]
        total = float(np.sum(probs_sorted))
        if total <= 0.0:
            # Degenerate case: still emit the single top-ranked node.
            return np.array([int(order[0])], dtype=int)
        cum = 0.0
        chosen = []
        for v_idx, p in zip(order.tolist(), probs_sorted.tolist()):
            if p <= 0.0:
                break
            chosen.append(int(v_idx))
            cum += float(p)
            if cum / total >= beta_mass:
                break
        if not chosen:
            chosen = [int(order[0])]
        return np.array(chosen, dtype=int)

    def _counts_for_members(k_idx: int, members: np.ndarray, members_intron: np.ndarray):
        intron_counts = (
            (doc[:, members_intron] * phi[:, members_intron, k_idx]).sum(axis=1).tolist()
            if members_intron.size > 0
            else [0.0] * D
        )
        aug_counts = (doc[:, members] * phi[:, members, k_idx]).sum(axis=1).tolist()
        return intron_counts, aug_counts

    def _emit_one_transcript(
        gtf_fh,
        *,
        k_idx: int,
        members: np.ndarray,
        transcript_suffix: str,
        transcript_ids_out: list,
        counts_intron_out: list,
        counts_aug_out: list,
    ):
        if members.size == 0:
            return

        # Require a single chromosome.
        chrom_set = {str(chroms[i]) for i in members if str(chroms[i]) and str(chroms[i]) != "nan"}
        if len(chrom_set) == 0:
            chrom = "."
        elif len(chrom_set) == 1:
            chrom = next(iter(chrom_set))
        else:
            return

        strand_set = {strands[i] for i in members if strands[i] in ("+", "-")}
        strand = next(iter(strand_set)) if len(strand_set) == 1 else "."

        # Build intron set from selected nodes (including path segments).
        intron_set = set()
        for v_idx in members.tolist():
            for s, e in _node_introns(int(v_idx)):
                intron_set.add((int(s), int(e)))
        introns = sorted(list(intron_set), key=lambda x: (x[0], x[1]))
        if not introns:
            return

        # Validate non-overlap. If violated, skip rather than modifying the selection.
        ok = True
        for (s, e) in introns:
            if e < s:
                ok = False
                break
        if ok:
            for (s1, e1), (s2, e2) in zip(introns, introns[1:]):
                if e1 >= s2:
                    ok = False
                    break
        if not ok:
            print(
                f"[WARN] Skipping export for {gene.name} cluster {k_idx}{transcript_suffix}: overlapping/invalid introns",
                file=sys.stderr,
            )
            return

        # Infer exons from intron boundaries (1-based inclusive).
        exons = []
        first_s, _first_e = introns[0]
        _last_s, last_e = introns[-1]
        # We don't know true transcript boundaries from junctions alone, but we
        # need flanking exons so the first/last introns are representable in GTF.
        # Use short flanks of configurable length (default: 1bp) to anchor the
        # intron chain in a valid GTF exon representation.
        try:
            flank_len = int(os.environ.get("BSEEJ_EXPORT_FLANK_EXON_LEN", "1"))
        except Exception:
            flank_len = 1
        flank_len = max(1, flank_len)

        exon1_end = int(first_s - 1)
        if exon1_end < flank_len:
            print(
                f"[WARN] Skipping export for {gene.name} cluster {k_idx}{transcript_suffix}: cannot place {flank_len}bp 5' flank exon",
                file=sys.stderr,
            )
            return
        exons.append((int(exon1_end - flank_len + 1), int(exon1_end)))
        for (prev_s, prev_e), (next_s, next_e) in zip(introns, introns[1:]):
            ex_s = int(prev_e + 1)
            ex_e = int(next_s - 1)
            if ex_e < ex_s:
                print(
                    f"[WARN] Skipping export for {gene.name} cluster {k_idx}{transcript_suffix}: invalid exon inference",
                    file=sys.stderr,
                )
                return
            exons.append((ex_s, ex_e))
        exonN_start = int(last_e + 1)
        exons.append((exonN_start, int(exonN_start + flank_len - 1)))

        tid = f"BSEEJ_{gene.name}_C{k_idx}{transcript_suffix}"
        gid = str(gene.name)
        attrs_tx = f'gene_id "{gid}"; transcript_id "{tid}";'
        tx_start = min(s for s, _e in exons)
        tx_end = max(e for _s, e in exons)
        _emit_gtf_line(gtf_fh, chrom, "BSEEJ", "transcript", tx_start, tx_end, strand, attrs_tx)
        for exon_idx, (ex_s, ex_e) in enumerate(exons, start=1):
            attrs_ex = attrs_tx + f' exon_number "{exon_idx}";'
            _emit_gtf_line(gtf_fh, chrom, "BSEEJ", "exon", ex_s, ex_e, strand, attrs_ex)

        members_intron = np.array([i for i in members.tolist() if not _is_path_node(int(i))], dtype=int)
        intron_counts, aug_counts = _counts_for_members(k_idx, members, members_intron)
        transcript_ids_out.append(tid)
        counts_intron_out.append(intron_counts)
        counts_aug_out.append(aug_counts)

    # Collect results for both decodings.
    transcript_ids_b = []
    counts_intron_b = []
    counts_aug_b = []
    transcript_ids_beta = []
    counts_intron_beta = []
    counts_aug_beta = []

    with open(gtf_b_path, "w", encoding="utf-8") as gtf_fh_b, open(gtf_beta_path, "w", encoding="utf-8") as gtf_fh_beta:
        for k_idx in range(K):
            members_b = np.where(b[k_idx] > 0)[0]
            if members_b.size > 0:
                _emit_one_transcript(
                    gtf_fh_b,
                    k_idx=k_idx,
                    members=members_b,
                    transcript_suffix="_b",
                    transcript_ids_out=transcript_ids_b,
                    counts_intron_out=counts_intron_b,
                    counts_aug_out=counts_aug_b,
                )

            members_beta = _select_members_beta(k_idx)
            if members_beta is not None and members_beta.size > 0:
                _emit_one_transcript(
                    gtf_fh_beta,
                    k_idx=k_idx,
                    members=members_beta,
                    transcript_suffix="_beta",
                    transcript_ids_out=transcript_ids_beta,
                    counts_intron_out=counts_intron_beta,
                    counts_aug_out=counts_aug_beta,
                )

    # Write count matrix.
    def _write_counts(path, tids, rows):
        with open(path, "w", encoding="utf-8") as out_fh:
            out_fh.write("transcript_id\t" + "\t".join(sample_names) + "\n")
            for tid, row in zip(tids, rows):
                out_fh.write(tid + "\t" + "\t".join(map(str, row)) + "\n")

    _write_counts(counts_b_path, transcript_ids_b, counts_intron_b)
    _write_counts(os.path.join(out_dir, f"{gene.name}_bseej_counts_b_augmented.tsv"), transcript_ids_b, counts_aug_b)
    _write_counts(counts_beta_path, transcript_ids_beta, counts_intron_beta)
    _write_counts(os.path.join(out_dir, f"{gene.name}_bseej_counts_beta_augmented.tsv"), transcript_ids_beta, counts_aug_beta)

    print(f"[INFO] Exported BSEEJ transcript GTF (b): {gtf_b_path}")
    print(f"[INFO] Exported BSEEJ transcript GTF (beta): {gtf_beta_path}")
    print(f"[INFO] Exported BSEEJ transcript counts (b): {counts_b_path}")
    print(f"[INFO] Exported BSEEJ transcript counts (beta): {counts_beta_path}")


def needed_n_k_list(gene):
    if os.path.exists(gene.result_path):
        # done_comb = [dname.split('run_info_')[1].split('.json')[0] for dname in os.listdir(gene.result_path) if
        # '.json' in dname]
        # done_k = [int(comb.split('.json')[0].split('_K_')[1]) for comb in done_comb]
        done_k = []
    else:
        done_k = []
    n_k_v = sorted(gene.all_n_k[::2][:9])
    n_k_v = list(set(n_k_v) - set(done_k))
    return n_k_v


def compute_config_score(sam_df, trans_introns_f, config):
    # config = [1, 3, 7]
    tr_score_list = []
    for trii in trans_introns_f:
        this_tr_score = 0
        for node in config:
            node_interval = list(sam_df.iloc[node, :].values)
            for intr in trans_introns_f[trii]:
                intr_start = trans_introns_f[trii][intr]['start']
                intr_end = trans_introns_f[trii][intr]['end']
                intron_interval = [intr_start, intr_end]
                
                if np.abs(node_interval[0] - intron_interval[0]) <= 12 and np.abs(
                        node_interval[1] - intron_interval[1]) <= 12:
                    this_tr_score += 1
                    break
                # else:
                #     print('start points difference', np.abs(node_interval[0] - intron_interval[0]),
                #           'end points difference', np.abs(node_interval[1] - intron_interval[1]))
        tr_score_list.append(this_tr_score)
    if len(tr_score_list) != 0:
        this_config_score = max(tr_score_list) / len(config)
    else:
        this_config_score = 0
    return this_config_score


def calc_bic(n_d, n_v, n_k, max_l):
    bic_k = n_v + n_k
    return (bic_k * np.log(n_d)) - (2 * max_l)


def calc_bic2(n_d, n_v, n_k, all_n_w, max_l):
    num_theta = n_k - 1
    num_z = all_n_w * (n_k - 1)
    num_beta = n_v * (n_k - 1)
    num_b = n_v * n_k
    num_pi = n_k - 1
    
    bic_k = num_theta + num_z + num_beta + num_b + num_pi
    return (bic_k * np.log(n_d)) - (2 * max_l)


@jit(nopython=True)
def adjust_matrices(mat, eps):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] < eps:
                mat[i, j] = eps
    return mat


@jit
def update_z_loop_numba(beta, theta, n_tr, n_v, n_k, document_tr):
    z = np.array([n_tr, n_v, n_k])
    for doc in range(0, n_tr):
        for v in range(0, n_v):
            ratio_v = np.exp(np.log(theta[doc, :]) + np.log(beta[:, v]))
            ratio_v /= np.sum(ratio_v)
            tempz = np.random.multinomial(1, ratio_v, size=document_tr[doc, v]).argmax(axis=1)
            for k in range(0, n_k):
                z[doc, v, k] = np.count_nonzero(tempz == k)
    return z


def read_run_info(path):
    run_info = 0
    if os.path.getsize(path) == 0:
        run_info = 0
    else:
        if '.gz' in path:
            with gzip.open(path) as handle:
                run_info = pickle.load(handle)
        elif '.json' in path and os.path.getsize(path) > 0:
            with open(path, 'rb') as handle:
                run_info = pickle.load(handle)
        elif '.pkl' in path:
            with gzip.open(path, 'rb') as ifp:
                run_info = pickle.load(ifp)
    
    return run_info


def is_converged_fwsr(likelihood, threshold=0.005):
    # Convergence diagnostic is optional; if ArviZ is unavailable we simply
    # report "not converged" so the sampler/optimizer runs to max iterations.
    if az is None:
        return False
    n0 = int(len(likelihood) / 2)
    this_ess = az.ess(np.array(likelihood[n0:]), method="quantile", prob=0.95)
    indices = range(n0, len(likelihood), int(this_ess))
    if len(indices) < 4:
        return False
    relevant_likelihood = [likelihood[i] for i in indices]
    sigma_hat_g_n = np.std(relevant_likelihood)
    honest_metric = sigma_hat_g_n / np.sqrt(len(indices)) + (1 / len(indices))
    mean_g_n = np.mean(relevant_likelihood)
    conv = honest_metric < np.abs(mean_g_n * threshold)
    return conv


def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)
