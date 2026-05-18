import gzip
import os
import pickle
import random
import subprocess
import tempfile
from collections import Counter
from copy import deepcopy

import arviz as az
import networkx as nx
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
    _, edges_list = get_conflict_for_plot(nodes_df)
    gra = generate_interval_graph_nx(nodes_df, edges_list, intervalviz=False)
    # min_k = nx.graph_clique_number(gra)
    min_k = max(len(clique) for clique in nx.find_cliques(gra))
    # min_k = len(nx.maximal_independent_set(G))
    return min_k


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

def generate_interval_graph_nx(nodes_df, edges_list, intervalviz=True):
    """Generate the graph G=(V,E) using networkx library and visualize"""
    gra = nx.Graph()
    if intervalviz:
        newedges_list = [(nodes_df['graph_labels'][ee[0]], nodes_df['graph_labels'][ee[1]]) for ee in edges_list]
        gra.add_nodes_from(nodes_df['graph_labels'])
    else:
        newedges_list = [(nodes_df['node_labels'][ee[0]], nodes_df['node_labels'][ee[1]]) for ee in edges_list]
        gra.add_nodes_from(nodes_df['node_labels'])
        # newedgesList = edges_list
    
    for e in newedges_list:
        gra.add_edge(*e)
    return gra


def split_training_test(document_orig, tr_percentage=95):
    tr_size = int(tr_percentage / 100 * document_orig.shape[0])
    indices = np.random.RandomState(seed=2021).permutation(document_orig.shape[0])
    training_idx, test_idx = indices[:tr_size], indices[tr_size:]
    document = document_orig[training_idx, :]
    document_te = document_orig[test_idx, :]
    return document, document_te, training_idx, test_idx


def find_mis(nodes_df):
    _, edges_list = get_conflict_for_plot(nodes_df)
    gra = generate_interval_graph_nx(nodes_df, edges_list, intervalviz=False)
    gc = nx.complement(gra)
    # This is for older versions of networkx
    # mis = nx.graph_clique_number(gc)
    mis = max(len(clique) for clique in nx.find_cliques(gc))
    max_ind_set = nx.maximal_independent_set(gra)
    while len(max_ind_set) < mis:
        max_ind_set = nx.maximal_independent_set(gra)
    max_ind_set = [int(n) for n in max_ind_set]
    max_ind_set.sort()
    return mis, max_ind_set


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
    comb_name = 'gene_' + gene.name + '_alpha_' + str(model.alpha) + '_eta_' + str(model.eta) + '_epsilon_' + \
                str(model.epsilon) + '_rs_' + str(model.r) + '_K_' + str(model.run_info['N_K'])
    last_run = list(model.run_info['gibbs'])[-1]
    last_z = deepcopy(model.run_info['gibbs'][last_run]['Z'])
    last_b = deepcopy(model.run_info['gibbs'][last_run]['b'])
    new_b, new_z = merge_suplicate_clusters(last_b, last_z)
    model.run_info['new_b'] = deepcopy(new_b)
    model.run_info['new_Z'] = deepcopy(new_z)
    # save the result
    if not os.path.exists(gene.result_path):
        os.mkdir(gene.result_path)
    # pickle.dump(model.run_info, open(gene.result_path + '/' + 'run_info_' + comb_name + '.json', 'wb'))
    filename = gene.result_path + '/' + 'run_info_' + comb_name + '.pkl'
    file_s = gzip.GzipFile(filename, 'wb')
    pickle.dump(model.run_info, file_s)
    print(filename, 'saved.')
    
    z_matrix = model.run_info['new_Z']
    id2w = model.run_info['id2w_dict']
    n_sample = z_matrix.shape[0]
    n_introns = z_matrix.shape[1]
    effective_k = z_matrix.shape[2]
    gene_name = model.run_info['gene']
    
    starts = np.asarray([int(id2w[j].split('-')[0]) for j in range(n_introns)], np.int32)
    ends = np.asarray([int(id2w[j].split('-')[1]) for j in range(n_introns)], np.int32)
    
    result_df = pd.DataFrame(data=0, columns=['gene', 'trans_id', 'index', 'start', 'end', 'sample', 'FPKM'],
                             index=range(n_sample * effective_k * n_introns))

    compute_df_vectorized(n_sample, effective_k, n_introns, result_df, gene_name, z_matrix, starts, ends)
    
    file_name_2 = 'bseej_' + gene_name + '_K_' + str(effective_k) + '.csv'
    result_df.to_csv(gene.result_path + '/' + file_name_2)
    print(gene.result_path + '/' + file_name_2, 'saved.')
    return gene.result_path + '/' + file_name_2


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
