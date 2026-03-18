from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors
from scipy.signal import correlate

from surrogates import generate_surrogates_batch
from system import effective_dt
from utils import batch_pearsonr, safe_pearsonr, stable_hash_int

try:
    from scipy.interpolate import UnivariateSpline
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - optional dependency
    UnivariateSpline = None
    _HAVE_SCIPY = False


@dataclass(frozen=True)
class CCMOperator:
    weights: np.ndarray
    neighbor_time: np.ndarray
    manifold_time: np.ndarray


def build_ccm_operator(
    target: np.ndarray,
    Dim: int,
    tau: int,
    exclusion_radius: int = 0,
) -> Optional[CCMOperator]:
    target = np.asarray(target, dtype=float)
    L = target.size
    valid_start = (Dim - 1) * tau
    M = L - valid_start
    if M <= Dim + 1:
        return None

    shadow = np.column_stack([target[i * tau : L - (Dim - 1 - i) * tau] for i in range(Dim)])
    exclusion_radius = max(0, int(exclusion_radius))
    keep = Dim + 1
    row_ids = np.arange(M)

    def _query_neighbors(k_query: int) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
        nbrs = NearestNeighbors(n_neighbors=k_query, algorithm="kd_tree").fit(shadow)
        dist_all, ind_all = nbrs.kneighbors(shadow)
        dist = np.empty((M, keep), dtype=float)
        ind = np.empty((M, keep), dtype=int)

        for row in range(M):
            candidate_idx = ind_all[row]
            candidate_dist = dist_all[row]
            valid = candidate_idx != row_ids[row]
            if exclusion_radius > 0:
                valid &= np.abs(candidate_idx - row_ids[row]) > exclusion_radius

            filtered_idx = candidate_idx[valid]
            filtered_dist = candidate_dist[valid]
            if filtered_idx.size < keep:
                return None, None

            ind[row] = filtered_idx[:keep]
            dist[row] = filtered_dist[:keep]
        return dist, ind

    k_base = Dim + 2
    k_query = min(M, max(k_base, k_base + 2 * exclusion_radius + 8))
    dist, ind = _query_neighbors(k_query)
    if dist is None or ind is None:
        if k_query >= M:
            return None
        dist, ind = _query_neighbors(M)
        if dist is None or ind is None:
            return None

    e = 1e-12
    u = np.exp(-dist / (dist[:, [0]] + e))
    w = u / np.sum(u, axis=1, keepdims=True)

    manifold_time = np.arange(valid_start, L)
    neighbor_time = manifold_time[ind]
    return CCMOperator(weights=w, neighbor_time=neighbor_time, manifold_time=manifold_time)


def ccm_rho(op: Optional[CCMOperator], source: np.ndarray) -> float:
    """Compute CCM prediction correlation using a pre-built operator.

    This is the core CCM measurement: use the target manifold's nearest-neighbour
    weights to predict the source series, then return Pearson r.
    """
    if op is None:
        return 0.0
    source = np.asarray(source, dtype=float).flatten()
    pred = np.sum(op.weights * source[op.neighbor_time], axis=1)
    true_vals = source[op.manifold_time]
    return float(safe_pearsonr(true_vals, pred))


def ccm_asymmetry_score(
    op_fwd: Optional[CCMOperator],
    src_series: np.ndarray,
    op_rev: Optional[CCMOperator],
    tgt_series: np.ndarray,
) -> float:
    """Bidirectional CCM asymmetry: max(0, rho_fwd - rho_rev).

    CCM's core discriminative signal is directional: for a true edge src→tgt,
    the target manifold should predict the source well (high rho_fwd), but the
    source manifold should predict the target poorly (low rho_rev).

    Parameters
    ----------
    op_fwd : CCMOperator built from target embedding (tests src→tgt).
    src_series : source time series.
    op_rev : CCMOperator built from source embedding (tests tgt→src).
    tgt_series : target time series.

    Returns
    -------
    Clamped asymmetry in [0, 1].
    """
    rho_fwd = ccm_rho(op_fwd, src_series)
    rho_rev = ccm_rho(op_rev, tgt_series)
    return float(max(0.0, rho_fwd - rho_rev))


def ccm_scores_for_pair(
    op: CCMOperator,
    source: np.ndarray,
    n_surrogates: int,
    surrogate_methods: List[str], 
    rng: np.random.Generator,
    iaaft_iters: int,
) -> Dict[str, float]:
    """
    Return dict with:
      - Raw: rho_raw
      - For each surrogate base method M:
          M_Z      : (rho0 - mean(rho_s))/std(rho_s)
          M_Delta  : rho0 - mean(rho_s)
          M_P      : empirical one-sided p = P(rho_s >= rho0)
          M_LogP   : -log10(p)
    """
    source = np.asarray(source, dtype=float).flatten()

    # --- Raw
    src_pred = np.sum(op.weights * source[op.neighbor_time], axis=1)
    src_true = source[op.manifold_time]
    rho_raw = float(safe_pearsonr(src_true, src_pred))
    out: Dict[str, float] = {"Raw": rho_raw}

    # --- Surrogate-based scores
    for method in surrogate_methods:
        surrogates = generate_surrogates_batch(
            source, int(n_surrogates), method, rng, iaaft_iters=iaaft_iters
        )

        # batch: [1 + n_surr, T]
        src_batch = np.vstack([source, surrogates])

        # op.weights: [L, K], op.neighbor_time: [L, K]
        pred_batch = np.sum(op.weights[None, :, :] * src_batch[:, op.neighbor_time], axis=2)  # [B, L]
        true_batch = src_batch[:, op.manifold_time]  # [B, L]

        corrs = np.asarray(batch_pearsonr(true_batch, pred_batch), dtype=float)  # [B]
        if corrs.size < 2 or not np.isfinite(corrs[0]):
            out[f"{method}_Z"] = 0.0
            out[f"{method}_Delta"] = 0.0
            out[f"{method}_P"] = 1.0
            out[f"{method}_LogP"] = 0.0
            continue

        rho0 = float(corrs[0])
        rho_s = corrs[1:]
        rho_s = rho_s[np.isfinite(rho_s)]

        if rho_s.size < 5:
            # surrogate 太少/全是 nan -> 给一个保守输出
            out[f"{method}_Z"] = 0.0
            out[f"{method}_Delta"] = 0.0
            out[f"{method}_P"] = 1.0
            out[f"{method}_LogP"] = 0.0
            continue

        mu = float(np.mean(rho_s))
        sigma = float(np.std(rho_s, ddof=1)) if rho_s.size > 1 else 0.0

        z_raw = float((rho0 - mu) / sigma) if sigma > 1e-6 else 0.0
        delta_raw = float(rho0 - mu)

        # Clip to max(0, ...) — non-causal pairs where surrogate matches real
        # should score ~0, not negative.  This prevents negative scores from
        # destroying AUROC ranking (both true and false pairs often get positive
        # raw z, but clipping removes the long negative tail that blurs ranking).
        z = max(0.0, z_raw)
        delta = max(0.0, delta_raw)

        # one-sided empirical p-value: how often surrogate >= observed
        # add +1 smoothing to avoid p=0
        ge = int(np.sum(rho_s >= rho0))
        p = float((ge + 1) / (rho_s.size + 1))
        logp = float(-np.log10(max(p, 1e-12)))

        # Rank: percentile of rho0 within the surrogate distribution [0, 1]
        rank = float(np.mean(rho_s < rho0))

        out[f"{method}_Z"] = z
        out[f"{method}_Delta"] = delta
        out[f"{method}_P"] = p
        out[f"{method}_LogP"] = logp
        out[f"{method}_Rank"] = rank

    return out

def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson r of ranks)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return safe_pearsonr(rx, ry)


def ccm_convergence_score(
    target: np.ndarray,
    source: np.ndarray,
    Dim: int,
    tau: int,
    n_lib_sizes: int = 8,
) -> float:
    """Measure whether CCM ρ increases with library size.

    True causal relationships show monotonic ρ increase as L grows.
    Returns the Spearman rank correlation between library fraction and ρ,
    clamped to [0, 1].  Spearman is more robust than Pearson to outlier ρ
    values at small library sizes.
    """
    target = np.asarray(target, dtype=float).flatten()
    source = np.asarray(source, dtype=float).flatten()
    T = min(target.size, source.size)
    if T < 50:
        return 0.0

    fracs = np.linspace(0.25, 1.0, n_lib_sizes)
    rhos = []
    for frac in fracs:
        L = int(T * frac)
        t_sub = target[:L]
        s_sub = source[:L]
        op = build_ccm_operator(t_sub, Dim=Dim, tau=tau)
        if op is None:
            rhos.append(0.0)
            continue
        rhos.append(ccm_rho(op, s_sub))

    rhos_arr = np.array(rhos, dtype=float)
    if np.std(rhos_arr) < 1e-12:
        return 0.0
    conv = _spearman_r(fracs, rhos_arr)
    return float(max(0.0, conv))


def _make_sporadic_observations(x, dt, obs_prob, jitter_frac, rng, min_obs):
    T = x.shape[0]
    t_full = np.arange(T, dtype=float) * float(dt)
    mask = rng.random(T) < float(obs_prob) if obs_prob < 1.0 else np.ones(T, dtype=bool)
    if int(mask.sum()) < int(min_obs):
        idx = np.linspace(0, T - 1, num=min_obs, dtype=int)
        mask[idx] = True
    t_obs = t_full[mask]
    x_obs = x[mask]
    if jitter_frac > 0.0:
        jf = float(np.clip(jitter_frac, 0.0, 0.49))
        t_obs += rng.uniform(-jf, jf, size=t_obs.size) * float(dt)
    order = np.argsort(t_obs)
    return t_obs[order], x_obs[order]


def latent_embedding_spline_derivative(x, dt, embed_dim, smooth, obs_prob, jitter_frac, rng, min_obs):
    T = x.shape[0]
    t_full = np.arange(T, dtype=float) * dt
    t_obs, x_obs = _make_sporadic_observations(x, dt, obs_prob, jitter_frac, rng, min_obs)

    H = np.zeros((T, embed_dim), dtype=float)
    use_spline = bool(_HAVE_SCIPY and UnivariateSpline is not None and t_obs.size >= 4)

    if use_spline:
        s_val = smooth * t_obs.size * max(float(np.var(x_obs)), 1e-12)
        try:
            sp = UnivariateSpline(t_obs, x_obs, k=3, s=s_val)
            for d in range(embed_dim):
                if d == 0:
                    H[:, d] = sp(t_full)
                else:
                    H[:, d] = sp.derivative(n=d)(t_full)
        except Exception:
            use_spline = False

    if not use_spline:
        H[:, 0] = x
        if embed_dim >= 2:
            H[:, 1] = np.gradient(x, dt)
        for d in range(2, embed_dim):
            H[:, d] = np.gradient(H[:, d - 1], dt)

    H = (H - np.mean(H, axis=0)) / np.maximum(np.std(H, axis=0), 1e-12)
    return np.nan_to_num(H)


def _latent_ccm_corr(source_lat, target_lat, k):
    T = source_lat.shape[0]
    if T < 10:
        return 0.0
    nn = NearestNeighbors(n_neighbors=k + 1).fit(target_lat)
    dist, ind = nn.kneighbors(target_lat)
    ind, dist = ind[:, 1:], dist[:, 1:]
    u = np.exp(-dist / np.maximum(dist[:, [0]], 1e-12))
    w = u / np.sum(u, axis=1, keepdims=True)
    src_pred = np.sum(source_lat[ind] * w[:, :, None], axis=1)
    corrs = [safe_pearsonr(source_lat[:, d], src_pred[:, d]) for d in range(source_lat.shape[1])]
    return float(np.mean(corrs))


def latent_ccm_score(source_lat, target_lat, k, lib0, rng):
    T = min(source_lat.shape[0], target_lat.shape[0])
    source_lat, target_lat = source_lat[:T], target_lat[:T]
    full = _latent_ccm_corr(source_lat, target_lat, k)
    idx = rng.choice(T, size=min(T, lib0), replace=False)
    subset_corr = _latent_ccm_corr(source_lat[idx], target_lat[idx], k)
    return max(0.0, full - subset_corr)


def build_latent_processes_for_trial(data, system_type, cfg, trial_seed):
    n_nodes, T = data.shape
    dt = effective_dt(system_type, cfg)
    latents = np.zeros((n_nodes, T, cfg.latent_ccm.embed_dim))
    ss = np.random.SeedSequence([trial_seed, stable_hash_int("LatentCCM")])
    for i, seed in enumerate(ss.spawn(n_nodes)):
        rng = np.random.default_rng(seed)
        latents[i] = latent_embedding_spline_derivative(
            data[i],
            dt,
            cfg.latent_ccm.embed_dim,
            cfg.latent_ccm.spline_smooth,
            cfg.latent_ccm.obs_prob,
            cfg.latent_ccm.time_jitter_frac,
            rng,
            cfg.latent_ccm.min_obs,
        )
    return latents


class EmbeddingOptimizer:
    def __init__(self, data: np.ndarray, max_lag: int = 50, max_dim: int = 10):
        self.data = np.asarray(data, dtype=float).flatten()
        s = np.std(self.data)
        self.data = (self.data - np.mean(self.data)) / (s if s > 1e-12 else 1.0)
        self.N = len(self.data)
        self.max_lag = max_lag
        self.max_dim = max_dim

    def find_tau_autocorr(self, method: str = "first_zero") -> int:
        acf = correlate(self.data, self.data, mode="full")
        acf = acf[acf.size // 2 :]
        acf /= acf[0]
        if method == "first_zero":
            zeros = np.where(np.diff(np.sign(acf)))[0]
            if zeros.size > 0:
                return int(zeros[0]) + 1
        elif method == "1/e":
            threshold = 1.0 / np.e
            drops = np.where(acf < threshold)[0]
            if drops.size > 0:
                return int(drops[0])
        return 1

    def find_tau_mutual_info(self, plotting: bool = False) -> int:
        mis = []
        lags = range(1, self.max_lag + 1)
        for lag in lags:
            x = self.data[:-lag].reshape(-1, 1)
            y = self.data[lag:]
            mi = mutual_info_regression(x, y, discrete_features=False, n_neighbors=3, random_state=42)[0]
            mis.append(mi)
        mis = np.array(mis)
        for i in range(1, len(mis) - 1):
            if mis[i] < mis[i - 1] and mis[i] < mis[i + 1]:
                tau = lags[i]
                if plotting:
                    plt.figure(figsize=(8, 4))
                    plt.plot(lags, mis, "o-", label="Mutual Information")
                    plt.axvline(tau, color="r", linestyle="--", label=f"First Min (tau={tau})")
                    plt.title("Mutual Information vs Lag")
                    plt.xlabel("Lag (tau)")
                    plt.ylabel("MI (nats)")
                    plt.legend()
                    plt.show()
                return tau
        # No MI local minimum found; fall back to autocorrelation first-zero
        tau_ac = self.find_tau_autocorr(method="first_zero")
        return max(1, min(tau_ac, self.max_lag))

    def _embed(self, dim: int, tau: int) -> np.ndarray:
        N = self.N
        L = N - (dim - 1) * tau
        if L <= 0:
            raise ValueError(f"Data too short for E={dim}, tau={tau}")
        emb = np.column_stack([self.data[i * tau : i * tau + L] for i in range(dim)])
        return emb

    def find_e_simplex(self, tau: int = 1, max_dim: int = 10) -> Tuple[int, float]:
        rhos = []
        dims = range(1, max_dim + 1)
        theiler = max(1, (max_dim - 1) * tau)
        for E in dims:
            lib_emb = self._embed(E, tau)
            if len(lib_emb) < 10:
                rhos.append(0)
                continue
            X = lib_emb[:-1]
            y_true = lib_emb[1:, 0]
            n_pts = len(X)
            # Query extra neighbors to have enough after Theiler exclusion
            k_query = min(n_pts, E + 2 + 2 * theiler)
            nbrs = NearestNeighbors(n_neighbors=k_query).fit(X)
            distances, indices = nbrs.kneighbors(X)
            need = E + 1  # E+1 neighbors for simplex in E dimensions
            y_pred = []
            valid = True
            for i in range(n_pts):
                # Exclude self and temporally close neighbors (Theiler window)
                mask = (indices[i] != i) & (np.abs(indices[i] - i) > theiler)
                d_filt = distances[i][mask]
                idx_filt = indices[i][mask]
                if len(d_filt) < need:
                    valid = False
                    break
                dists = d_filt[:need]
                idxs = idx_filt[:need]
                min_d = max(np.min(dists), 1e-9)
                weights = np.exp(-dists / min_d)
                weights /= np.sum(weights)
                pred = np.dot(weights, y_true[idxs])
                y_pred.append(pred)
            if not valid:
                rhos.append(0)
                continue
            rho = safe_pearsonr(y_true, np.array(y_pred))
            rhos.append(rho)
        optimal_e = dims[np.argmax(rhos)]
        return optimal_e, np.max(rhos)

    def optimize(self) -> Tuple[int, int]:
        tau_mi = self.find_tau_mutual_info()
        e_simplex, _ = self.find_e_simplex(tau=tau_mi, max_dim=self.max_dim)
        return int(e_simplex), int(tau_mi)


def get_optimal_embedding(time_series):
    optimizer = EmbeddingOptimizer(time_series)
    E, tau = optimizer.optimize()
    return E, tau
