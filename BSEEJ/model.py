import time

from scipy.special import gammaln, xlogy, psi
from scipy.special import expit
from scipy.stats import beta as sci_beta
from scipy.stats import dirichlet, multinomial
import numpy as np
import bisect
import os
from copy import deepcopy
from typing import Tuple

# Check what does score in the junc.file mean, Professor think it is the number of reads that express that junction in the input file
# Check the preprocessing in BAM file


from utilities import *
from utilities import find_initial_nodes, split_training_test, generalized_min_node_cover
import utilities
from utilities import save_results, compute_df


_eps = 1e-12


class Model(object):
    
    def __init__(self, eta, alpha, epsilon, r, s):
        self.eta = eta
        self.alpha = alpha
        self.eta_prior = eta
        self.alpha_prior = alpha
        # Strength of long-read co-occurrence prior (0 disables it)
        self.cooc_strength = 0.0
        self.cooc_matrix = None
        self.epsilon = epsilon
        self.r = r
        self.s = s
        self.beta = None
        self.b = None
        self.theta = None
        self.pi = None
        self.z = None
        self.init_nodes = None

        self.z = None
        self.beta = None
        self.theta = None
        self.pi = None
        self.b = None

        self.converged = None
        self.z_init = None
        self.run_info = None
        self.alpha_vec = None
        self.gamma = None
        self.phi = None
        self.zeta = None
        self.pi_a = None
        self.pi_b = None
        self.b_probs = None
        self.eta0 = None

        # Interval-graph MWIS state for projecting Bernoulli means
        self._iv_order = None
        self._iv_P = None
        self.conflict_m = None
        self.b_mask = None
        self._mask_ready = False

        # Optional damping factors for stable updates
        self.b_stepsize = 0.3
        self.beta_stepsize = 0.5
        self.idx_suffix = idx_suffix or ""

    def _precompute_interval_dp_structs(self, starts: np.ndarray, ends: np.ndarray):
        """
        Build end-sorted order and predecessor array for weighted interval scheduling.
        """
        V = int(len(starts))
        if V == 0:
            self._iv_order = np.array([], dtype=int)
            self._iv_P = np.array([], dtype=int)
            return
        order = np.argsort(ends)
        s = np.asarray(starts)[order]
        e = np.asarray(ends)[order]
        P = np.empty(V, dtype=int)
        end_list = e.tolist()
        for j in range(V):
            i = bisect.bisect_right(end_list, s[j], 0, j) - 1
            P[j] = i
        self._iv_order = order
        self._iv_P = P

    def _chromosome_safe_coords(self, gene, starts: np.ndarray, ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make interval scheduling chromosome-safe by embedding per-chromosome coordinate
        systems into one non-overlapping axis.

        This matters if ``gene.nodes_df`` contains multiple chromosomes: the overlap
        relation is not purely 1D, and a naive DP on raw start/end would incorrectly
        treat intervals on different chromosomes as conflicting.
        """
        try:
            nodes_df = getattr(gene, "nodes_df", None)
            if nodes_df is None:
                return starts, ends
            if "chrom" not in nodes_df.columns:
                return starts, ends
            chroms = [str(c) for c in nodes_df["chrom"].tolist()]
            uniq = sorted(set(chroms))
            if len(uniq) <= 1:
                return starts, ends
        except Exception:
            return starts, ends

        # Compute an offset per chromosome so coordinate ranges do not overlap.
        # We use a conservative gap based on observed maxima.
        chrom_to_max_end = {}
        for c, e in zip(chroms, ends.tolist()):
            try:
                ee = int(e)
            except Exception:
                continue
            prev = chrom_to_max_end.get(c)
            if prev is None or ee > prev:
                chrom_to_max_end[c] = ee

        offsets = {}
        offset = 0
        # Deterministic chromosome order for reproducible runs.
        for c in sorted(chrom_to_max_end.keys()):
            offsets[c] = offset
            # gap to avoid accidental overlap between chromosomes
            offset += int(chrom_to_max_end[c]) + 1_000_000

        starts_g = np.asarray([int(offsets.get(str(c), 0)) + int(s) for c, s in zip(chroms, starts)], dtype=np.int64)
        ends_g = np.asarray([int(offsets.get(str(c), 0)) + int(e) for c, e in zip(chroms, ends)], dtype=np.int64)
        return starts_g, ends_g

    def _mwis_mask_sorted(self, weights_sorted: np.ndarray) -> np.ndarray:
        """
        Solve the maximum-weight independent set on intervals in sorted order.
        """
        V = int(weights_sorted.shape[0])
        if V == 0:
            return np.zeros(0, dtype=bool)
        if V == 1:
            return np.array([weights_sorted[0] > 0.0], dtype=bool)
        P = self._iv_P
        M = np.zeros(V + 1, dtype=float)
        choose = np.zeros(V + 1, dtype=np.int8)
        for j in range(1, V + 1):
            wj = weights_sorted[j - 1]
            pj = P[j - 1] + 1
            take = wj + M[pj]
            skip = M[j - 1]
            if take > skip:
                M[j] = take
                choose[j] = 1
            else:
                M[j] = skip
                choose[j] = 0
        mask_sorted = np.zeros(V, dtype=bool)
        j = V
        while j > 0:
            if choose[j] == 1:
                mask_sorted[j - 1] = True
                j = P[j - 1] + 1
            else:
                j -= 1
        return mask_sorted

    def _greedy_graph_mask(self, weights: np.ndarray) -> np.ndarray:
        """
        Graph-aware projection used by CAVI/SVI.

        The goal here is semantic correctness with respect to ``gene.intersection``:
        the returned mask is always an independent set of the active conflict graph,
        including long-read multi-segment path nodes and reference-guided variants.
        """
        V = int(weights.shape[0])
        if V <= 0:
            return np.zeros(0, dtype=bool)
        if self.conflict_m is None:
            keep = weights > 0.0
            return keep.astype(bool, copy=False)

        conflict = np.asarray(self.conflict_m, dtype=bool)
        degrees = np.sum(conflict, axis=1)
        positive = [int(v) for v in np.where(weights > 0.0)[0].tolist()]
        if not positive:
            return np.zeros(V, dtype=bool)

        base_order = sorted(positive, key=lambda v: (-float(weights[v]), int(degrees[v]), int(v)))
        candidate_orders = [base_order]
        for seed in base_order[: min(16, len(base_order))]:
            rest = [v for v in base_order if v != seed]
            candidate_orders.append([seed] + rest)

        best_mask = np.zeros(V, dtype=bool)
        best_score = float("-inf")
        for order in candidate_orders:
            keep = np.zeros(V, dtype=bool)
            blocked = np.zeros(V, dtype=bool)
            score = 0.0
            for v in order:
                if blocked[v]:
                    continue
                keep[v] = True
                blocked |= conflict[v]
                blocked[v] = True
                score += float(weights[v])
            if score > best_score:
                best_score = score
                best_mask = keep
        return best_mask

    def _save_elbo_plot(self, gene, history, method_label='cavi', k=None, iterations=None):
        """
        Persist the ELBO trajectory as a PNG in the gene's result directory.
        """
        if not history:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[CAVI] Skipping ELBO plot: matplotlib not installed.")
            return

        output_dir = getattr(gene, "result_path", None)
        if output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"{method_label.lower()}"
        if k is not None:
            suffix += f"_k{k}"
        idx_suffix = getattr(self, "idx_suffix", "") or getattr(gene, "idx_suffix", "")
        if idx_suffix:
            suffix += f"_{idx_suffix}"
        output_path = os.path.join(output_dir, f"{gene.name}_elbo_{suffix}.png")

        plt.figure(figsize=(6, 4))
        x_vals = iterations if iterations is not None else range(len(history))
        plt.plot(list(x_vals), history, linewidth=1.0)
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        plt.title(f"ELBO Progression – {gene.name}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[CAVI] ELBO plot saved to {output_path}")

    def _initialize_cavi_from_gibbs(self, gene, n_k):
        """
        Mirror the Gibbs initializer to seed CAVI with comparable states.
        """
        doc = gene.document
        D, V = doc.shape

        # — Gibbs-style initialization for structural parameters —
        init_nodes = find_initial_nodes_from_intersection(gene.intersection, n_k)
        if len(init_nodes) < n_k:
            raise RuntimeError(f"Unable to find {n_k} initial independent sets for gene {gene.name}.")
        self.init_nodes = list(init_nodes[:n_k])

        z_matrix = np.zeros((D, V, n_k), dtype=np.int32)
        for d in range(D):
            for v in range(V):
                count = int(doc[d, v])
                if count <= 0:
                    continue
                assignments = np.random.randint(0, n_k, size=count)
                for k in range(n_k):
                    z_matrix[d, v, k] = np.count_nonzero(assignments == k)
        self.z_init = z_matrix
        self.z = z_matrix.copy()

        # theta ~ Dirichlet(alpha)
        theta = np.zeros((D, n_k))
        if np.ndim(self.alpha_prior) == 0:
            alpha_dir = np.full(n_k, float(self.alpha_prior))
        else:
            alpha_dir = np.array(self.alpha_prior, dtype=float)
            if alpha_dir.size != n_k:
                alpha_dir = np.resize(alpha_dir, n_k)
        for i in range(D):
            temp_dir = np.array([np.nan])
            while np.isnan(np.sum(temp_dir)):
                temp_dir = np.random.dirichlet(alpha_dir)
            theta[i] = temp_dir
        self.theta = theta

        # pi ~ Beta(r,s)
        self.pi = np.random.beta(self.r, self.s, size=n_k)

        # b initialized from maximal independent sets
        b = np.zeros((n_k, V), dtype=np.int32)
        for k in range(n_k):
            nodes = self.init_nodes[k]
            if nodes is None:
                continue
            idx = np.asarray(list(nodes), dtype=int)
            idx = idx[(idx >= 0) & (idx < V)]
            if idx.size:
                b[k, idx] = 1
        self.b = b

        # beta ~ Dirichlet(eta)
        if np.ndim(self.eta_prior) == 0:
            eta_dir = np.full(V, float(self.eta_prior))
        else:
            eta_vec = np.array(self.eta_prior, dtype=float)
            eta_dir = eta_vec if eta_vec.size == V else np.resize(eta_vec, V)
        beta = np.zeros((n_k, V))
        for k in range(n_k):
            temp_dirb = np.array([np.nan])
            while np.isnan(np.sum(temp_dirb)):
                temp_dirb = np.random.dirichlet(eta_dir)
            beta[k] = temp_dirb
        beta[beta < self.epsilon] = self.epsilon
        self.beta = beta

        # — Variational parameters seeded from Gibbs draws —
        phi = np.zeros((D, V, n_k))
        for d in range(D):
            for v in range(V):
                count = doc[d, v]
                if count > 0:
                    phi[d, v, :] = z_matrix[d, v, :] / count
                else:
                    phi[d, v, :] = 1.0 / n_k
        self.phi = phi

        if np.ndim(self.alpha_prior) == 0:
            self.alpha_vec = np.full(n_k, float(self.alpha_prior))
        else:
            self.alpha_vec = np.array(self.alpha_prior, dtype=float)
        counts_theta = np.einsum('dv,dvk->dk', doc, phi)
        self.gamma = counts_theta + self.alpha_vec[np.newaxis, :]

        self.b_probs = np.where(b == 1, 1.0 - self.epsilon, self.epsilon).astype(float)
        self.b_mask = b.astype(bool)
        self._mask_ready = True

        self.pi_a = self.r + np.sum(self.b_probs, axis=1)
        self.pi_b = self.s + V - np.sum(self.b_probs, axis=1)
        self.pi_a = np.maximum(self.pi_a, self.epsilon)
        self.pi_b = np.maximum(self.pi_b, self.epsilon)

        counts_beta = np.einsum('dv,dvk->kv', doc, phi)
        self.zeta = self.eta0[None, :] * self.b_probs + counts_beta
        self.zeta = np.maximum(self.zeta, self.epsilon)

    
    def make_run_info(self, gene, n_k, burn_in, convergence_checkpoint_interval, n_iter):
        """this function saves all the information in the Gibbs runs"""
        self.run_info = dict()
        self.run_info['N_V'] = gene.n_v
        self.run_info['N_D'] = gene.n_d
        self.run_info['N_K'] = n_k
        self.run_info['r'] = self.r
        self.run_info['s'] = self.s
        self.run_info['alpha'] = self.alpha_prior
        self.run_info['eta'] = self.eta_prior
        self.run_info['epsilon'] = self.epsilon
        self.run_info['min_k'] = gene.min_k
        self.run_info['variant'] = getattr(gene, 'variant', 'current')
        self.run_info['reference_k'] = getattr(gene, 'reference_k', None)
        self.run_info['effective_k'] = getattr(gene, 'effective_k', n_k)
        self.run_info['annotation_path'] = getattr(gene, 'annotation_path', '')
        self.run_info['novel_m'] = getattr(gene, 'novel_m', None)
        self.run_info['gene'] = gene.name
        self.run_info['burn_in'] = burn_in
        self.run_info['convergence_checkpoint_interval'] = convergence_checkpoint_interval
        self.run_info['n_iter'] = convergence_checkpoint_interval
        self.run_info['document']= gene.document
        self.run_info['document_tr'] = gene.document_tr
        self.run_info['document_te']= gene.document_te
        self.run_info['tr_idx']= gene.training_idx
        self.run_info['te_idx']= gene.test_idx




        self.run_info['N_W'] = gene.n_w
        self.run_info['gene_mvc_id'] = gene.mvc
        gene_mvc = [gene.id2w_dict[i] for i in gene.mvc]
        self.run_info['gene_mvc'] = gene_mvc
        self.run_info['samples_df'] = gene.samples_df
        self.run_info['gene_intersection'] = gene.intersection
        self.run_info['w2id_dict'] = gene.w2id_dict
        self.run_info['id2w_dict'] = gene.id2w_dict
        self.run_info['MIS'] = gene.mis
        self.run_info['max_ind_set'] = gene.max_ind_set
        self.run_info['init_nodes'] = self.init_nodes
        self.run_info['convergence_point'] = n_iter
        self.run_info['idx_suffix'] = self.idx_suffix
       
    
    def log_likelihood(self):
        """
        Compute the current ELBO
        L(q) = E_q[ log p(w,z,θ,β,π,b) ] − E_q[ log q(z,θ,β,π,b) ]
        following the closed‐form in Blei et al. (2018).
        Uses self.gamma, self.phi, self.pi_a, self.pi_b, self.b_probs, self.zeta,
         self.alpha_vec, self.eta0, self.r, self.s, and self.run_info['document'] (D×V counts).
        """
        D, V = self.run_info['document'].shape
        K    = self.gamma.shape[1]

        # shorthands
        doc   = self.run_info['document']   # (D, V)
        gamma = self.gamma                  # (D, K)
        phi   = self.phi                    # (D, V, K)
        a     = self.pi_a                   # (K,)
        b_    = self.pi_b                   # (K,)
        eta   = self.b_probs                # (K, V)
        zeta  = self.zeta                   # (K, V)
        alpha = self.alpha_vec              # (K,)
        eta0  = self.eta0                   # (V,)
        r, s  = self.r, self.s

        # 1) E_q[ log p(θ | α) ]
        term = D*(gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)))
        term += np.sum((alpha-1) * (psi(gamma) - psi(np.sum(gamma,axis=1))[:,None]))

        # 2) E_q[ log p(z | θ, β) ]
        # compute digamma terms
        E_log_theta = psi(gamma) - psi(np.sum(gamma,axis=1))[:,None]   # (D, K)
        E_log_beta  = psi(zeta)  - psi(np.sum(zeta,axis=1))[:,None]    # (K, V)

        #    raise dims to (D,1,K) and (1,V,K) before summing ---
        Theta_term = E_log_theta[:, None, :]        # (D, 1, K)
        Beta_term  = E_log_beta.T[None, :, :]       # (1, V, K)
        term += np.sum(doc[:,:,None] * phi * (Theta_term + Beta_term))

        # 3) E_q[ log p(π | r,s) ]
        term += np.sum( gammaln(r+s) - gammaln(r) - gammaln(s)
                     + (r-1)*(psi(a)-psi(a+b_))
                     + (s-1)*(psi(b_)-psi(a+b_)) )

        # 4) E_q[ log p(b | π) ]
        term += np.sum( eta*(psi(a)[:,None]-psi(a+b_)[:,None])
                     + (1-eta)*(psi(b_)[:,None]-psi(a+b_)[:,None]) )

        # 5) E_q[ log p(β | η⁰ ⊙ b) ]
        eta0_mask = self.eta0[None, :]                  # (1,V)
        eta_expect = self.b_probs                      # (K,V)
        eta_tilde = np.clip(eta0_mask * eta_expect, self.epsilon, None)  # (K,V)
        term += np.sum(
            gammaln(np.sum(eta_tilde, axis=1)) - np.sum(gammaln(eta_tilde), axis=1)
        )
        term += np.sum(
            (eta_tilde - 1.0) * (psi(zeta) - psi(np.sum(zeta, axis=1))[:, None])
        )

        # — subtract entropies —
        # a) −E_q[ log q(θ) ]
        psi_sum = psi(np.sum(gamma,axis=1))
        term -= np.sum( gammaln(np.sum(gamma,axis=1)) - np.sum(gammaln(gamma),axis=1)
                     + np.sum((gamma-1)*(psi(gamma)-psi_sum[:,None]),axis=1) )

        # b) −E_q[ log q(z) ]
        term -= np.sum(doc[:,:,None] * phi * np.log(phi + _eps))

        # c) −E_q[ log q(π) ]  (entropy of Beta(a,b))
        term -= np.sum(
            (gammaln(a) + gammaln(b_) - gammaln(a + b_))
            - (a - 1.0) * psi(a)
            - (b_ - 1.0) * psi(b_)
            + (a + b_ - 2.0) * psi(a + b_)
        )

        # d) −E_q[ log q(b) ]
        term -= np.sum( eta*np.log(eta + _eps) + (1-eta)*np.log(1-eta + _eps) )

        # e) −E_q[ log q(β) ]
        term -= np.sum( gammaln(np.sum(zeta,axis=1))
                     - np.sum(gammaln(zeta),axis=1)
                     + np.sum((zeta-1)*(psi(zeta)-psi(np.sum(zeta,axis=1))[:,None]), axis=1) )


        return term


    # — 2. Held‐out ELBO average — 
    def log_likelihood_te(self, document_te):
        """
        Approximate held‐out ELBO by plugging in variational params:
        average_d[ log ∑_k exp( ψ(γ_dk)-ψ(sum γ_d) + ψ(η_kv)-ψ(sum η_k) ) ]
        """
        D_te, V = document_te.shape
        K       = self.gamma.shape[1]

        # digamma terms
        E_log_theta = psi(self.gamma) - psi(np.sum(self.gamma, axis=1))[:, None]  # (D,K)
        E_log_beta  = psi(self.zeta)    - psi(np.sum(self.zeta, axis=1))[:, None] # (K,V)

        total = 0.0
        for d in range(D_te):
            x_d = document_te[d]  # (V,)
            for v in range(V):
                cnt = x_d[v]
                if cnt == 0:
                    continue
                scores = E_log_theta[d] + E_log_beta[:, v]  # (K,)
                # log ∑_k exp(scores)
                total += cnt * np.logaddexp.reduce(scores)
        return total / D_te


    
    def update_z(self):
        """
        Variational E‐step:  φ_{d,v,k} ∝ exp{ E_log θ_{d,k} + E_log β_{k,v} },
        then normalize over k, and set expected counts z = doc*φ.
        """
        doc = self.run_info['document']  # (D,V)
        D, V = doc.shape
        K    = self.phi.shape[2]

        E_log_theta = psi(self.gamma) - psi(np.sum(self.gamma, axis=1))[:, None]  # (D,K)
        E_log_beta  = psi(self.zeta)   - psi(np.sum(self.zeta, axis=1))[:, None]  # (K,V)

        phi = np.zeros((D, V, K))

        # normalization with a constant denominator
        for d in range(D):
            for v in range(V):
                scores = E_log_theta[d] + E_log_beta[:, v]  # (K,)
                shifted = scores - np.max(scores)
                w       = np.exp(shifted)
                phi[d, v, :] = w / (np.sum(w) + _eps)
        self.phi = phi
    
    def update_theta(self):
        """
        Variational M‐step for θ:  γ_{d,k} = α_k + sum_v doc[d,v]*φ[d,v,k].
        """
        doc  = self.run_info['document']  # (D,V)
        phi  = self.phi                   # (D,V,K)
        # counts[d,k] = sum_v doc[d,v] * phi[d,v,k]
        counts = np.einsum('dv,dvk->dk', doc, phi)  # (D,K)

        # ensure alpha is length‑K
        if self.alpha_vec is None or self.alpha_vec.shape[0] != counts.shape[1]:
            base_alpha = np.array(self.alpha_prior, ndmin=1, dtype=float)
            if base_alpha.size == 1:
                self.alpha_vec = np.full(counts.shape[1], float(base_alpha[0]))
            else:
                self.alpha_vec = base_alpha

        self.gamma = counts + self.alpha_vec[np.newaxis, :]



    def update_pi(self):
        """
        Variational M‐step for π: 
          a_k     = r + sum_v ζ_{k,v},
        b'_k    = s + V - sum_v ζ_{k,v}.
        """
        K, V = self.b_probs.shape  # η is (K,V)
        self.pi_a = self.r + np.sum(self.b_probs, axis=1)       # (K,)
        self.pi_b = self.s + V - np.sum(self.b_probs, axis=1)   # (K,)
        self.pi_a = np.maximum(self.pi_a, self.epsilon)
        self.pi_b = np.maximum(self.pi_b, self.epsilon)


    def _apply_interval_projection(self, logits: np.ndarray):
        """
        Project Bernoulli probabilities to respect interval constraints.
        """
        K, V = self.b_probs.shape
        if self.b_mask is None or self.b_mask.shape != (K, V):
            self.b_mask = np.zeros((K, V), dtype=bool)
        for k in range(K):
            keep = self._greedy_graph_mask(logits[k])
            self.b_mask[k, :] = keep
            self.b_probs[k, ~keep] = self.epsilon
        self._mask_ready = True


    def update_b(self):
        """
        Variational update for Bernoulli b_{k,v} with MWIS projection.
        """
        if self.b_probs is None:
            raise RuntimeError("b_probs must be initialized before calling update_b.")
        psi_a  = psi(self.pi_a)[:, None]
        psi_b  = psi(self.pi_b)[:, None]
        psi_e  = psi(self.zeta) - psi(np.sum(self.zeta, axis=1))[:, None]
        logits = (psi_a - psi_b) + psi_e

        new_b = expit(logits)
        new_b = np.clip(new_b, self.epsilon, 1.0 - self.epsilon)
        self.b_probs = (1.0 - self.b_stepsize) * self.b_probs + self.b_stepsize * new_b
        self.b_probs = np.clip(self.b_probs, self.epsilon, 1.0 - self.epsilon)

        self._apply_interval_projection(logits)



    def update_beta(self):
        """
        Variational M‐step for β: 
        η_{k,v} = η0_v + sum_d doc[d,v]*φ[d,v,k].
        """
        doc = self.run_info['document']  # (D,V)
        D, V = doc.shape
        K    = self.zeta.shape[0]        # ζ is (K,V)

        # sum over d: (D,V,1)*(D,V,K) → (D,V,K) → sum over axis=0 → (K,V)
        counts = np.einsum('dv,dvk->kv', doc, self.phi)  # (K,V)
        new_zeta = self.eta0[None, :] * self.b_probs + counts

        # Optional long-read co-occurrence prior: bias ζ_{k,v} so that
        # introns that frequently co-occur with already-included introns
        # receive additional pseudo-counts.
        if getattr(self, "cooc_strength", 0.0) > 0.0 and getattr(self, "cooc_matrix", None) is not None:
            # cooc_matrix is (V,V); b_probs is (K,V) → cooc_term is (K,V)
            cooc_term = self.b_probs @ self.cooc_matrix
            new_zeta = new_zeta + self.cooc_strength * cooc_term

        new_zeta = np.maximum(new_zeta, self.epsilon)
        if self.zeta is None:
            self.zeta = new_zeta
        else:
            self.zeta = (1.0 - self.beta_stepsize) * self.zeta + self.beta_stepsize * new_zeta
            self.zeta = np.maximum(self.zeta, self.epsilon)


    def update_run_info(self, t, gg, burn_in):
        """
        Record ELBO for convergence tracking.

        Heavy per-iteration snapshots (phi/zeta/b) are intentionally omitted:
        downstream exports use the final variational state persisted in
        _finalize_after_inference, which keeps memory usage bounded for large
        whole-genome runs.
        """
        elbo = float(self.log_likelihood())
        self.run_info.setdefault('cavi', {})[t] = {
            'elbo': elbo,
        }
        # alias into 'gibbs' for backward compatibility
        self.run_info['gibbs'] = self.run_info['cavi']

    def _record_svi_snapshot(self, gene, t, tag='svi'):
        """
        Compute a full-data diagnostic snapshot (ELBO + params) without
        disturbing the ongoing SVI run.
        """
        # Refresh φ and γ on the full corpus using current globals
        self.update_z()
        self.update_theta()
        elbo = float(self.log_likelihood())

        from copy import deepcopy
        if self._mask_ready and self.b_mask is not None:
            b_mask = self.b_mask
        else:
            b_mask = self.b_probs > self.epsilon

        self.run_info.setdefault(tag, {})[t] = {
            'elbo':   elbo,
            'Z':      deepcopy(self.phi),
            'b':      deepcopy(b_mask.astype(np.int32)),
            'b_prob': deepcopy(self.b_probs),
            'zeta':   deepcopy(self.zeta),
            'pi_a':   deepcopy(self.pi_a),
            'pi_b':   deepcopy(self.pi_b),
        }
        # Back-compat alias so downstream utilities continue to work
        self.run_info['gibbs'] = self.run_info[tag]
        return elbo


    def get_log_likelihood_vec(self):
        """
        Return list of ELBOs in order of iterations.
        """
        return [
            info['elbo']
            for it, info in sorted(self.run_info.get('cavi', {}).items())
        ]

    def train(self, gene, n_k, n_iter, burn_in, convergence_checkpoint_interval, verbose):
        """
        Replace the Gibbs sampler.  Run CAVI for up to n_iter iterations:
        initialize q(θ,π,b,β) & φ randomly,
        repeat updates {φ,γ} → {a,b} → {η,ζ} → ELBO until convergence.
        """
        
        # ─── 1) Gather document and interval metadata ───────────────────────
        doc = gene.document                  # shape (D, V)
        D, V = doc.shape
        K    = n_k
        self.conflict_m = np.asarray(getattr(gene, "intersection", None), dtype=bool) if getattr(gene, "intersection", None) is not None else None

        if hasattr(gene, "nodes_df") and {'start', 'end'}.issubset(gene.nodes_df.columns):
            starts = np.asarray(gene.nodes_df['start'].to_numpy())
            ends = np.asarray(gene.nodes_df['end'].to_numpy())
        elif hasattr(gene, "intron_starts") and hasattr(gene, "intron_ends"):
            starts = np.asarray(gene.intron_starts)
            ends = np.asarray(gene.intron_ends)
        else:
            raise RuntimeError("Cannot locate intron start/end coordinates on Gene.")
        starts_dp, ends_dp = self._chromosome_safe_coords(gene, starts, ends)
        self._precompute_interval_dp_structs(starts_dp, ends_dp)

        eta0 = np.array(self.eta_prior)
        if eta0.ndim == 0:
            eta0 = np.full(V, eta0)
        self.eta0 = eta0

        # Optional intron co-occurrence prior derived from Megadepth --junctions.
        # If present on the Gene, this is a (V,V) matrix where cooc[v,u] reflects
        # how strongly intron v tends to co-occur with intron u in multi-junction reads.
        if hasattr(gene, "cooc_matrix") and gene.cooc_matrix is not None:
            self.cooc_matrix = np.asarray(gene.cooc_matrix, dtype=float)
        else:
            self.cooc_matrix = None

        # ─── 2) Initialize variational state via Gibbs-style seeding ─────────
        self._initialize_cavi_from_gibbs(gene, n_k)

        # ─── 3) Build run_info and attach the document matrix ───────────────
        self.make_run_info(gene,
                       n_k,
                       burn_in,
                       convergence_checkpoint_interval,
                       n_iter)
        self.run_info['document'] = doc
        self.run_info['gene_intron_starts'] = starts_dp
        self.run_info['gene_intron_ends'] = ends_dp

        # prepare to record ELBO each iteration
        self.run_info['cavi'] = {}

        # ─── 3) Coordinate Ascent Variational Inference ────────────────────────────
        for t in range(int(n_iter)):
            
            self.update_z()
            self.update_theta()
            self.update_b()
            self.update_pi()
            self.update_beta()

            # record ELBO
            self.update_run_info(t, None, None)

            # fetch ELBOs
            curr_elbo = self.run_info['cavi'][t]['elbo']
            if t > 0:
                prev_elbo = self.run_info['cavi'][t-1]['elbo']
                # Converge when the absolute ELBO change between consecutive
                # iterations is smaller than 2e-5.
                converged = abs(curr_elbo - prev_elbo) < 2e-5
            else:
                converged = False

            if verbose:
                print(f"Gene {gene.name}, Iter {t:3d}, ELBO {curr_elbo:.6f}, Converged {converged}")
            if converged:
                if verbose:
                    print(f"[CAVI] exiting at iter={t} (ΔELBO={curr_elbo-prev_elbo:.2e})")
                break


        self._finalize_after_inference(gene, doc, K, history_tag='cavi', method_label='cavi')

        return self




    def _finalize_after_inference(self, gene, doc, K, history_tag='cavi', method_label='cavi'):
        """
        Persist diagnostics, CSV summaries, and run_info patches after inference.
        """
        method_label = (method_label or history_tag or 'cavi').lower()
        self.run_info['inference'] = method_label

         # Per-run output directory: <gene>/<gene>_<method>_<K>[_suffix]
        orig_result_path = gene.result_path
        idx_suffix = getattr(gene, 'idx_suffix', '') or getattr(self, 'idx_suffix', '') or self.run_info.get('idx_suffix', '')
        suffix = f"_{idx_suffix}" if idx_suffix else ""
        run_dirname = f"{gene.name}_{method_label}_{K}{suffix}"
        run_result_path = os.path.join(orig_result_path, run_dirname)
        os.makedirs(run_result_path, exist_ok=True)
        gene.result_path = run_result_path
        self.run_info['result_path'] = run_result_path

        # Debug: write the interval graph and copy raw Megadepth junction
        # outputs into the run-specific output directory.
        if hasattr(gene, "_debug_print_interval_graph"):
            try:
                gene._debug_print_interval_graph()
            except Exception:
                pass
        if hasattr(gene, "_export_raw_junction_files"):
            try:
                gene._export_raw_junction_files()
            except Exception:
                pass

        print(f"Document is shape (N, V): {doc.shape} and content is{doc}")
        print(f"Parameter of THETA: \n gamma shape:(N, K) {self.gamma.shape}, size: {self.gamma.size}, dtype: {self.gamma.dtype}")
        # Print a representative row of gamma without assuming N >= 11.
        try:
            D_gamma = self.gamma.shape[0]
            example_row = min(10, D_gamma - 1)
            print(self.gamma[example_row], "\n")
        except Exception:
            print(self.gamma, "\n")

        print(f"Parameter of Z: \n phi shape: (N, V, K) {self.phi.shape}, size: {self.phi.size}, dtype: {self.phi.dtype}")
        # Print a representative entry of phi without assuming N,V >= 11.
        try:
            D_phi, V_phi, _ = self.phi.shape
            example_d = min(10, D_phi - 1)
            example_v = min(10, V_phi - 1)
            print(self.phi[example_d, example_v], "\n")
        except Exception:
            print(self.phi, "\n")

        if hasattr(self.zeta, "shape"):
            print(f"Parameter of BETA: \n zeta  shape: (K, V) {self.zeta.shape}, size: {self.zeta.size}, dtype: {self.zeta.dtype}")
        else:
            print(f"Parameter of BETA: \n zeta  type: {type(self.zeta)}")
        print(self.zeta, "\n")

        if hasattr(self.b_probs, "shape"):
            print(f"Parameter of B: b_probs shape (K, V): {self.b_probs.shape}, size: {self.b_probs.size}, dtype: {self.b_probs.dtype}")
            # Print a representative entry of b_probs without assuming V >= 11.
            try:
                K_b, V_b = self.b_probs.shape
                example_k = min(K - 1, K_b - 1)
                example_v = min(10, V_b - 1)
                print(self.b_probs[example_k, example_v])
            except Exception:
                print(self.b_probs)
        else:
            print(f"b_probs   type: {type(self.b_probs)}; no shape attribute")

        # Persist the converged variational parameters so downstream analyses (and pickled run_info)
        # can recover the final state without crawling the per-iteration history.
        from copy import deepcopy
        self.run_info['final_gamma'] = deepcopy(self.gamma)
        self.run_info['final_phi'] = deepcopy(self.phi)
        self.run_info['final_pi_a'] = deepcopy(self.pi_a)
        self.run_info['final_pi_b'] = deepcopy(self.pi_b)
        self.run_info['final_b_probs'] = deepcopy(self.b_probs)
        self.run_info['final_zeta'] = deepcopy(self.zeta)
        # Persist final hard mask used by MWIS projection for export/CSV generation.
        if getattr(self, "b_mask", None) is not None:
            self.run_info['final_b_mask'] = deepcopy(self.b_mask.astype(np.int32))
        else:
            self.run_info['final_b_mask'] = deepcopy((self.b_probs > self.epsilon).astype(np.int32))
        self.run_info['idx_suffix'] = self.idx_suffix
        # provide alias so downstream code expecting "eta" can find the Bernoulli means
        self.run_info['final_eta'] = self.run_info['final_b_probs']

        # Export variational φ/ζ snapshots (CSV + optional Excel).
        phi = self.run_info.get('final_phi')
        zeta = self.run_info.get('final_zeta')
        if phi is not None and zeta is not None:
            self._export_phi_zeta_and_excel(gene, K, method_label, phi, zeta)

        # Always export transcript-style GTF + counts for external evaluation
        # (e.g., gffcompare/SQANTI). This export is derived from the final
        # variational state and does not require -save_result.
        try:
            import utilities
            utilities._export_bseej_transcripts_gtf_and_counts(gene, self)
        except Exception as exc:
            print(f"[WARN] Failed to export BSEEJ transcript GTF/counts for {gene.name}: {exc}")

        output_dir = os.path.join('results', gene.name)
        os.makedirs(output_dir, exist_ok=True)

        if np.ndim(self.alpha) == 0:
            hyper_alpha = float(self.alpha)
        else:
            hyper_alpha = float(self.alpha[0])
        hyper_eta = float(np.mean(self.eta0))
        print(f"hyper_eta is {hyper_eta}")
        print(f"hyper_alpha is {hyper_alpha}")

        self.run_info['alpha'] = hyper_alpha
        self.run_info['eta'] = hyper_eta

        orig_alpha, orig_eta = self.alpha, self.eta
        self.alpha = hyper_alpha
        self.eta = hyper_eta

        import utilities
        original_compute_df = utilities.compute_df
        original_vec = getattr(utilities, 'compute_df_vectorized', None)
        original_merge = getattr(utilities, 'merge_suplicate_clusters', None)

        def compute_df_vi(n_sample, effective_k, n_introns, result_df, gene_name, z_matrix, starts, ends):
            result_df['gene'] = result_df['gene'].astype(object)
            result_df['FPKM'] = result_df['FPKM'].astype(float)
            original_compute_df(n_sample, effective_k, n_introns, result_df, gene_name, z_matrix, starts, ends)

        if original_vec is not None and method_label != 'gibbs':
            utilities.compute_df_vectorized = compute_df_vi
        if original_merge is not None and method_label != 'gibbs':
            utilities.merge_suplicate_clusters = lambda b, z: (b, z)

        from utilities import save_results
        # Optionally skip writing the large per-sample result CSV when debugging.
        if getattr(self, "save_results", True):
            save_results(gene, self)

        if original_vec is not None and method_label != 'gibbs':
            utilities.compute_df_vectorized = original_vec
        if original_merge is not None and method_label != 'gibbs':
            utilities.merge_suplicate_clusters = original_merge

        self.alpha, self.eta = orig_alpha, orig_eta

        try:
            if history_tag == 'svi':
                history = [
                    info['elbo']
                    for it, info in sorted(self.run_info.get('svi', {}).items())
                ]
            else:
                history = self.get_log_likelihood_vec()
        except Exception:
            history = []
        items = sorted(self.run_info.get(history_tag, {}).items())
        iterations = [it for it, _ in items]
        self._save_elbo_plot(gene, history, method_label=method_label, k=K, iterations=iterations)

        # Restore the gene-level result_path after writing all run-specific outputs.
        gene.result_path = orig_result_path


    def _export_phi_zeta_and_excel(self, gene, K, method_label, phi, zeta):
        """
        Helper to export φ/ζ matrices to CSV and Excel, shared by CAVI/SVI and Gibbs.
        """
        suffix = f"_{self.idx_suffix}" if getattr(self, 'idx_suffix', "") else ""
        export_dir = gene.result_path
        os.makedirs(export_dir, exist_ok=True)
        joiner = f"_K_{K}{suffix}" if suffix else f"_K_{K}"

        joint_csv = os.path.join(export_dir, f"{gene.name}_{method_label}_phi_zeta{joiner}.csv")
        try:
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
            print(f"[{method_label.upper()}] Wrote combined phi/zeta to {joint_csv}")
        except Exception as exc:
            print(f"[{method_label.upper()}] Warning: failed to write combined CSV ({exc})")

        try:
            import pandas as pd  # optional
            excel_path = os.path.join(
                export_dir,
                f"{gene.name}_{method_label}_variational_params{joiner}.xlsx",
            )
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:  # type: ignore[call-arg]
                pd.DataFrame(zeta).to_excel(writer, sheet_name='zeta', index=False)
                # flatten phi into (D, V*K) to stay rectangular for Excel
                phi_flat = phi.reshape(phi.shape[0], -1)
                pd.DataFrame(phi_flat).to_excel(writer, sheet_name='phi', index=False)
            print(f"[{method_label.upper()}] Variational parameters exported to {excel_path}")
        except ImportError:
            print(f"[{method_label.upper()}] pandas/openpyxl not available; skipping Excel export of variational parameters.")
        except Exception as exc:
            print(f"[{method_label.upper()}] Warning: failed to export Excel variational parameters ({exc})")


################################################################################################################
################################################################################################################
#### Stochastic Variational Inference


    def train_svi(
        self,
        gene,
        n_k,
        n_iter,
        batch_size,
        burn_in=None,
        convergence_checkpoint_interval=None,
        kappa=0.7,
        t0=10.0,
        local_max_iter=50,
        local_tol=1e-4,
        eval_every=10,
        verbose=True,
    ):
        """
        Stochastic Variational Inference (SVI) for BAMIE/BSEEJ.
        Keeps deterministic CAVI untouched; results logged under run_info['svi'].
        """
        doc = gene.document
        D, V = doc.shape
        K = int(n_k)
        self.conflict_m = np.asarray(getattr(gene, "intersection", None), dtype=bool) if getattr(gene, "intersection", None) is not None else None

        # Interval-graph structures for Bernoulli projection
        if hasattr(gene, "nodes_df") and {'start', 'end'}.issubset(gene.nodes_df.columns):
            starts = np.asarray(gene.nodes_df['start'].to_numpy())
            ends = np.asarray(gene.nodes_df['end'].to_numpy())
        elif hasattr(gene, "intron_starts") and hasattr(gene, "intron_ends"):
            starts = np.asarray(gene.intron_starts)
            ends = np.asarray(gene.intron_ends)
        else:
            raise RuntimeError("Cannot locate intron start/end coordinates on Gene.")
        starts_dp, ends_dp = self._chromosome_safe_coords(gene, starts, ends)
        self._precompute_interval_dp_structs(starts_dp, ends_dp)

        # Prior vector for beta
        eta0 = np.array(self.eta_prior)
        if eta0.ndim == 0:
            eta0 = np.full(V, eta0)
        self.eta0 = eta0

        # Optional intron co-occurrence prior derived from Megadepth --junctions.
        # If present on the Gene, this is a (V,V) matrix where cooc[v,u] reflects
        # how strongly intron v tends to co-occur with intron u in multi-junction reads.
        if hasattr(gene, "cooc_matrix") and gene.cooc_matrix is not None:
            self.cooc_matrix = np.asarray(gene.cooc_matrix, dtype=float)
        else:
            self.cooc_matrix = None

        # Seed variational parameters (reuses Gibbs-style initializer)
        self._initialize_cavi_from_gibbs(gene, K)

        # Prepare run_info skeleton (mirrors deterministic train)
        self.make_run_info(
            gene,
            K,
            burn_in if burn_in is not None else max(1, int(n_iter) // 2),
            convergence_checkpoint_interval if convergence_checkpoint_interval is not None else max(1, int(n_iter) // 10),
            n_iter,
        )
        self.run_info['document'] = doc
        self.run_info['gene_intron_starts'] = starts_dp
        self.run_info['gene_intron_ends'] = ends_dp
        self.run_info['svi'] = {}

        # Ensure alpha_vec matches requested K
        if self.alpha_vec is None or self.alpha_vec.shape[0] != K:
            base_alpha = np.array(self.alpha_prior, ndmin=1, dtype=float)
            self.alpha_vec = np.full(K, float(base_alpha[0])) if base_alpha.size == 1 else base_alpha

        rng = np.random.default_rng()
        indices = np.arange(D, dtype=int)

        # ── Convergence tracking via smoothed ELBO deltas ──
        ma_window = max(3, int(np.ceil(50 / max(1, eval_every))))  # ~50 iter span
        ma_tol = 2e-6
        patience_required = 3
        patience = 0
        elbo_trace = []

        for t in range(int(n_iter)):
            rho_t = (t0 + t) ** (-kappa)

            if batch_size <= 0:
                raise ValueError("batch_size must be >= 1")
            batch = rng.choice(indices, size=min(batch_size, D), replace=False)
            M = int(len(batch))
            if M == 0:
                break

            Xb = doc[batch, :]

            phi_b = np.full((M, V, K), 1.0 / K, dtype=float)
            gamma_b = np.tile(self.alpha_vec, (M, 1))

            E_log_beta = psi(self.zeta) - psi(np.sum(self.zeta, axis=1))[:, None]

            last_loc = -np.inf
            for _ in range(int(local_max_iter)):
                E_log_theta_b = psi(gamma_b) - psi(np.sum(gamma_b, axis=1))[:, None]
                for m in range(M):
                    scores = E_log_theta_b[m][:, None] + E_log_beta  # (K,V)
                    shifted = scores - np.max(scores, axis=0, keepdims=True)
                    w = np.exp(shifted)
                    denom = np.sum(w, axis=0, keepdims=True) + _eps
                    phi_b[m] = (w / denom).T

                counts_b = np.einsum('mv,mvk->mk', Xb, phi_b)
                gamma_b = counts_b + self.alpha_vec[None, :]

                E_log_theta_b = psi(gamma_b) - psi(np.sum(gamma_b, axis=1))[:, None]
                approx_loc = 0.0
                for m in range(M):
                    x_m = Xb[m]
                    scores = E_log_theta_b[m][None, :] + E_log_beta.T  # (V,K)
                    approx_loc += float(np.dot(x_m, np.logaddexp.reduce(scores, axis=1)))
                approx_loc /= max(1, M)
                if abs(approx_loc - last_loc) < float(local_tol):
                    break
                last_loc = approx_loc

            scale = float(D) / float(M)
            counts_scaled = scale * np.einsum('mv,mvk->kv', Xb, phi_b)
            hat_zeta = self.eta0[None, :] * self.b_probs + counts_scaled

            # Optional long-read co-occurrence prior: bias ζ_{k,v} with
            # co-occurrence-derived pseudo-counts.
            if getattr(self, "cooc_strength", 0.0) > 0.0 and getattr(self, "cooc_matrix", None) is not None:
                cooc_term = self.b_probs @ self.cooc_matrix
                hat_zeta = hat_zeta + self.cooc_strength * cooc_term

            hat_zeta = np.maximum(hat_zeta, self.epsilon)

            sum_b = np.sum(self.b_probs, axis=1)
            hat_a = self.r + sum_b
            hat_b = self.s + V - sum_b

            self.pi_a = (1.0 - rho_t) * self.pi_a + rho_t * hat_a
            self.pi_b = (1.0 - rho_t) * self.pi_b + rho_t * hat_b
            self.pi_a = np.maximum(self.pi_a, self.epsilon)
            self.pi_b = np.maximum(self.pi_b, self.epsilon)

            self.zeta = (1.0 - rho_t) * self.zeta + rho_t * hat_zeta
            self.zeta = np.maximum(self.zeta, self.epsilon)

            psi_a = psi(self.pi_a)[:, None]
            psi_b = psi(self.pi_b)[:, None]
            psi_e = psi(self.zeta) - psi(np.sum(self.zeta, axis=1))[:, None]
            logits = (psi_a - psi_b) + psi_e
            new_b_probs = expit(logits)
            new_b_probs = np.clip(new_b_probs, self.epsilon, 1.0 - self.epsilon)
            self.b_probs = (1.0 - rho_t) * self.b_probs + rho_t * new_b_probs
            self.b_probs = np.clip(self.b_probs, self.epsilon, 1.0 - self.epsilon)

            self._apply_interval_projection(logits)

            do_eval = (t % max(1, int(eval_every)) == 0) or (t == int(n_iter) - 1)
            if do_eval:
                try:
                    elbo = self._record_svi_snapshot(gene, t, tag='svi')
                    elbo_trace.append(elbo)
                    if len(elbo_trace) >= ma_window:
                        recent = np.array(elbo_trace[-ma_window:])
                        avg_delta = float(np.mean(np.abs(np.diff(recent))))
                        baseline = float(np.mean(np.abs(recent))) + _eps
                        rel_delta = avg_delta / baseline
                        if rel_delta < ma_tol:
                            patience += 1
                        else:
                            patience = 0
                        if patience >= patience_required:
                            if verbose:
                                print(f"[SVI] Gene {gene.name}, iter {t:4d}, moving-average ΔELBO={rel_delta:.3e}; stopping.")
                            break
                except Exception:
                    pass
                if verbose:
                    elbo = self.run_info['svi'].get(t, {}).get('elbo', float('nan'))
                    print(f"[SVI] Gene {gene.name}, iter {t:4d}, rho_t={rho_t:.4f}, ELBO≈{elbo:.6f}")

        self._finalize_after_inference(gene, doc, K, history_tag='svi', method_label='svi')
        return self




################################################################################################################
################################################################################################################
#### Gibbs Sampling



    def initialize_vars_gibbs(self, gene, n_k):
        """This function initializes model parameters and other variables for training (Gibbs)"""
        self.init_nodes = find_initial_nodes(gene.nodes_df, n_k)
        z_matrix = np.zeros([gene.n_d, gene.n_v, n_k], dtype=np.int32)
        for doc in range(0, gene.n_d):
            for v in range(0, gene.n_v):
                tempz = np.random.randint(0, n_k, size=gene.document[doc, v])
                for k in range(0, n_k):
                    z_matrix[doc, v, k] = np.count_nonzero(tempz == k)
        self.z_init = z_matrix
    
        # theta: distribution of the samples over clusters
        theta = np.zeros([gene.n_d, n_k])
    
        for i in range(gene.n_d):
            temp_dir = np.array([np.nan])
            while np.isnan(sum(temp_dir)):
                temp_dir = np.random.dirichlet(self.alpha * np.ones(n_k))
            theta[i] = temp_dir
    
        # pi: distribution initialization
        pi = np.random.beta(self.r, self.s, size=n_k)
    
        # b: distribution initialization
        b = np.zeros([n_k, gene.n_v], dtype=np.int32)
        for k in range(n_k):
            init = [int(node) for node in self.init_nodes[k]]
            b[k, init] = 1
    
        # beta: distribution of the Clusters over intron excisions
        beta = np.zeros([n_k, gene.n_v])
        for k in range(n_k):
            temp_dirb = np.array([np.nan])
            while np.isnan(sum(temp_dirb)):
                temp_dirb = np.random.dirichlet(self.eta * np.ones(gene.n_v))
            beta[k, :] = temp_dirb
    
        beta[beta < self.epsilon] = self.epsilon
    
        self.z = z_matrix
        self.beta = beta
        self.theta = theta
        self.pi = pi
        self.b = b
        self.converged = False

    def make_run_info_gibbs(self, gene, n_k, burn_in, convergence_checkpoint_interval, n_iter):
        """this function saves all the information in the Gibbs runs"""
        self.run_info = dict()
        self.run_info['N_V'] = gene.n_v
        self.run_info['N_D'] = gene.n_d
        self.run_info['N_K'] = n_k
        self.run_info['N_W'] = gene.n_w
        self.run_info['gene_mvc_id'] = gene.mvc
        gene_mvc = [gene.id2w_dict[i] for i in gene.mvc]
        self.run_info['gene_mvc'] = gene_mvc
        self.run_info['r'] = self.r
        self.run_info['s'] = self.s
        self.run_info['alpha'] = self.alpha
        self.run_info['eta'] = self.eta
        self.run_info['epsilon'] = self.epsilon
        self.run_info['min_k'] = gene.min_k
        self.run_info['samples_df'] = gene.samples_df
        self.run_info['gene'] = gene.name
        self.run_info['gene_intersection'] = gene.intersection
        self.run_info['w2id_dict'] = gene.w2id_dict
        self.run_info['id2w_dict'] = gene.id2w_dict
        self.run_info['MIS'] = gene.mis
        self.run_info['max_ind_set'] = gene.max_ind_set
        self.run_info['init_nodes'] = self.init_nodes
        self.run_info['burn_in'] = burn_in
        self.run_info['convergence_checkpoint_interval'] = convergence_checkpoint_interval
        self.run_info['n_iter'] = n_iter
        self.run_info['convergence_point'] = n_iter
        self.run_info['document'] = gene.document      #####
        self.run_info['document_tr'] = gene.document_tr
        self.run_info['document_te'] = gene.document_te
        self.run_info['tr_idx'] = gene.training_idx
        self.run_info['te_idx'] = gene.test_idx
        self.run_info['idx_suffix'] = self.idx_suffix

    def log_likelihood_gibbs(self):
        """Computes log likelihood at the end of each Gibbs iteration"""
        n_d = self.z.shape[0]
        n_v = self.z.shape[1]
        n_k = self.z.shape[2]
        
        beta_rel = self.beta > self.epsilon
        beta_rel = beta_rel.T
        bet = np.repeat(beta_rel[np.newaxis, :, :], n_d, axis=0)
        rel_dim = bet * (self.z > self.epsilon)
        z_matrix_new = self.z.swapaxes(1, 2).reshape(n_d * n_k, n_v)
        
        rel_dim_new = rel_dim.swapaxes(1, 2).reshape(n_d * n_k, n_v)
        
        aa = z_matrix_new * rel_dim_new
        z_cut = aa[~np.all(aa == 0, axis=1)]
        beta_rep = np.repeat(self.beta[np.newaxis, :, :], n_d, axis=0).reshape(n_d * n_k, n_v)
        beta_relevant = beta_rep * rel_dim_new
        beta_cut = beta_relevant[~np.all(aa == 0, axis=1)]
        beta_cut = beta_cut / (np.sum(beta_cut, axis=1).reshape(-1, 1))  # normalize
        multinomial_pmf = gammaln(np.sum(z_cut, axis=1) + 1) + np.sum(xlogy(z_cut, beta_cut) - gammaln(z_cut + 1),
                                                                      axis=-1)
        likelihood = np.sum(multinomial_pmf)
        return likelihood

    def log_likelihood_te_gibbs(self, document_te):
        """Computes log likelihood of test"""
        n_k = self.run_info['N_K']
        likelihood_te = 0
        for i in range(document_te.shape[0]):
            for k in range(n_k):
                x = document_te[i, :]
                relevant_beta = list(np.where(self.beta[k, :] > self.epsilon)[0])
                relevant_lambda = list(np.where(x > self.epsilon)[0])
                relevant_dim = list(set(relevant_beta).intersection(set(relevant_lambda)))
                if len(relevant_dim) > 0:
                    temp_x = list(x[relevant_dim])
                    temp_beta = list(self.beta[k, relevant_dim])
                    likelihood_te += multinomial.logpmf(temp_x, np.sum(temp_x), temp_beta)
        likelihood_te = likelihood_te / (document_te.shape[0] * n_k)
        return likelihood_te

    def update_z_gibbs(self):
        """Update z variable in the model"""
        # Sample from full conditional of Z
        # save for computing relative error
        self.beta = adjust_matrices(self.beta, self.epsilon)
        self.theta = adjust_matrices(self.theta, self.epsilon)
        
        for doc in range(0, self.run_info['document'].shape[0]):
            for v in range(0, self.run_info['N_V']):
                
                ratio_v = np.exp(np.log(self.theta[doc, :]) + np.log(self.beta[:, v]))
                ratio_v /= np.sum(ratio_v)
                
                tempz = np.random.multinomial(1, ratio_v, size=self.run_info['document'][doc, v]).argmax(axis=1)
                
                for k in range(0, self.run_info['N_K']):
                    self.z[doc, v, k] = np.count_nonzero(tempz == k)

    def update_theta(self):
        """Update \theta variable in the model"""
    
        # Sample from full conditional of Theta
        for doc in range(self.run_info['N_D']):
            self.theta[doc, :] = np.random.dirichlet(self.alpha + np.sum(self.z[doc, :, :], axis=0))
        self.theta[self.theta < self.epsilon] = self.epsilon

    def update_pi(self):
        """Update \pi variable in the model"""
    
        # update for pi
        m = np.sum(self.b, axis=1)
        for k in range(self.run_info['N_K']):
            self.pi[k] = np.random.beta(self.r + m[k], self.s + self.run_info['N_V'] - m[k], size=None)
            # pi[k] = np.random.beta(r + np.sum(Z_matrix[:, :, k]), s + np.sum(document) -
            # np.sum(Z_matrix[:, :, k]), size=None)

    def update_b_gibbs(self):
        """Update b variable in the model"""
        n_s = 10
        if not self.converged:
            for k in range(self.run_info['N_K']):
                random_clusters = sample_local_ind_set(self.run_info['gene_intersection'], self.run_info['N_V'], n_s,
                                                       self.b[k, :], self.beta[k, :], self.run_info['MIS'])
                
                unnorm_p_phi = np.zeros([len(random_clusters)])
                for t in range(len(random_clusters)):
                    cluster = random_clusters[t]
                    cluster_neighbor = list(
                        np.where(np.sum(self.run_info['gene_intersection'][cluster, :] != 0, axis=0))[0])
    
                    term1 = sci_beta.logpdf(x=self.pi[k], a=self.r + len(cluster), b=self.s + len(cluster_neighbor),
                                            loc=0,
                                            scale=1)
                    # relevant_indices = list(set(range(self.run_info['N_V'])) - set(cluster_neighbor))
                    # relevant_indices = np.sort(relevant_indices)
                    b_eta = self.eta * self.b[k, :]
                    b_eta_eps = np.array([v + self.epsilon if np.abs(v) < self.epsilon else v for v in list(b_eta)])
                    temp3 = np.array(
                        [v + self.epsilon if np.abs(v) < self.epsilon else v for v in list(self.beta[k, :])])
                    term2 = dirichlet.logpdf(temp3 / np.sum(temp3), b_eta_eps)
                    p_phi = np.exp(term1 + term2)
                    unnorm_p_phi[t] = np.nan_to_num(p_phi)
                
                norm_p_phi = np.nan_to_num(unnorm_p_phi / np.sum(unnorm_p_phi))
                
                pop_no = 10
                pop = []
                for po in range(pop_no):
                    sel_cluster = np.random.multinomial(1, norm_p_phi, size=1)[0]
                    new_cluster_idx = np.where(sel_cluster)[0][0]
                    pop.append(new_cluster_idx)
                new_cluster_idx = max(set(pop), key=pop.count)
                new_cluster = random_clusters[new_cluster_idx]
                temp = np.zeros([self.run_info['N_V']], dtype=np.int32)
                temp[new_cluster] = 1
                self.b[k, :] = deepcopy(temp)

    def update_beta(self):
        """Update \beta variable in the model"""
    
        # Sample from full conditional of Beta
        # Z_matrix[:, v, k] counts the number of times word v is assigned to cluster k throughout the whole corpus
        for k in range(self.run_info['N_K']):
            temp_b = np.array([v + self.epsilon if v == 0 else v for v in list(self.b[k, :])])
            self.beta[k, :] = np.random.dirichlet(temp_b * self.eta + np.sum(self.z[:, :, k], axis=0))

            # Optional long-read co-occurrence prior: bias the Dirichlet
            # concentration for β_k so that introns that co-occur with the
            # currently included introns in cluster k receive additional
            # pseudo-counts.
            if getattr(self, "cooc_strength", 0.0) > 0.0 and getattr(self, "cooc_matrix", None) is not None:
                b_row = self.b[k, :].astype(float)
                cooc_term = b_row @ self.cooc_matrix  # (V,)
                prior_vec = prior_vec + self.cooc_strength * cooc_term

            alpha_vec = prior_vec + np.sum(self.z[:, :, k], axis=0)
            alpha_vec = np.maximum(alpha_vec, self.epsilon)
            self.beta[k, :] = np.random.dirichlet(alpha_vec)

    def update_run_info_gibbs(self, t, gg, burn_in):
        """saves Gibbs iteration info in the data"""
    
        self.run_info['gibbs'][t]['Theta'] = deepcopy(self.theta)
        self.run_info['gibbs'][t]['Beta'] = deepcopy(self.beta)
        self.run_info['gibbs'][t]['b'] = deepcopy(self.b)
        self.run_info['gibbs'][t]['error'] = np.sum(np.abs(self.z - self.z_init)) / (
                self.run_info['N_D'] * self.run_info['N_W'])
        self.run_info['gibbs'][t]['likelihood_i'] = self.log_likelihood()
        self.run_info['gibbs'][t]['likelihood_te'] = self.log_likelihood_te(gg.document_te)
    
        if t == 0:
            self.run_info['gibbs'][t]['relative_error'] = np.sum(np.abs(self.z - self.z_init)) / (
                    self.run_info['N_D'] * self.run_info['N_W'])
        else:
            self.run_info['gibbs'][t]['relative_error'] = np.sum(
                np.abs(self.z - self.run_info['gibbs'][t - 1]['Z'])) / (self.run_info['N_D'] * self.run_info['N_W'])
        
        if self.converged or t >= burn_in:
            self.run_info['gibbs'][t]['Z'] = deepcopy(self.z)
        else:
            self.run_info['gibbs'][t]['Z'] = 0

    def get_log_likelihood_vec(self):
        """extract the values of likelihood from run info dictionary"""
    
        runs_dict = self.run_info['gibbs']
        likelihood = []
        for i in runs_dict.keys():
            likelihood.append(runs_dict[i]['likelihood_i'])
        return likelihood

    def train(self, gene, n_k, n_iter, burn_in, convergence_checkpoint_interval, verbose):
        """Run Gibbs sampling on the data"""
    
        self.initialize_vars(gene, n_k)
        self.make_run_info(gene, n_k, burn_in, convergence_checkpoint_interval, n_iter)
        self.run_info['gibbs'] = {}
    
        startiter = time.time()
        it = 0
        while it <= min(self.run_info['convergence_point'] + 100, n_iter):
        
            self.run_info['gibbs'][it] = {}
            
            self.update_z()
            
            self.update_theta()
            
            self.update_pi()
            
            self.update_b()
            
            self.update_beta()
            
            self.update_run_info(it, gene, burn_in)
            
            if it >= burn_in and it % convergence_checkpoint_interval == 0 and not self.converged:
    
                log_likelihood_vector = self.get_log_likelihood_vec()
                self.converged = is_converged_fwsr(log_likelihood_vector, threshold=0.005)
                
                if self.converged:
                    self.run_info['convergence_point'] = it
            
            if it % 100 == 0 and verbose:
                print('Gene', gene.name, ', Iteration', it, ', Likelihood =',
                      round(self.run_info['gibbs'][it]['likelihood_i'], 4), ', Converged:', self.converged)
            it += 1
        
        self.run_info['duration'] = round(time.time() - startiter, 3)
        self.run_info['duration_per_iter'] = round(self.run_info['duration'] / n_iter, 3)
        self.run_info['error'] = np.sum(np.abs(self.z - self.z_init)) / (self.run_info['N_D'] * self.run_info['N_W'])
        self.run_info['likelihood_te'] = self.log_likelihood_te(gene.document_te)
