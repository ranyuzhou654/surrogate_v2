from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import networkx as nx
import numpy as np
import warnings

from utils import add_observation_noise_snr, safe_zscore_rows


def effective_dt(system_type: str, cfg) -> float:
    base_dt = cfg.integration.base_dt_by_system.get(system_type, 1.0)
    ds = max(1, int(cfg.system.downsample_factor))
    return float(base_dt) * float(ds)


def make_dynamic_scale(T: int, strength: float, smooth_win: int, rng: np.random.Generator) -> np.ndarray:
    """Positive time-varying scale with mean 1."""
    T = int(T)
    if T <= 0:
        return np.ones(0, dtype=float)
    if strength <= 0:
        return np.ones(T, dtype=float)
    win = int(max(1, smooth_win))
    z = rng.standard_normal(T)
    if win > 1:
        kernel = np.ones(win, dtype=float) / float(win)
        z = np.convolve(z, kernel, mode="same")
    scale = np.exp(strength * z)
    scale = scale / float(np.mean(scale))
    return scale.astype(float)


def ar1_process(T: int, n_series: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) with stationary variance 1. Returns shape [n_series, T]."""
    T = int(T)
    n_series = int(n_series)
    if T <= 0 or n_series <= 0:
        return np.zeros((n_series, T), dtype=float)
    phi = float(np.clip(phi, -0.999, 0.999))
    eps = rng.standard_normal((n_series, T))
    x = np.zeros((n_series, T), dtype=float)
    x[:, 0] = eps[:, 0]
    s = float(np.sqrt(max(1e-12, 1.0 - phi * phi)))
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + s * eps[:, t]
    return x


@dataclass
class ProcessNoise:
    mode: str
    sigma: float
    scale_t: np.ndarray  # [T]
    eta: np.ndarray  # [n_nodes, T]

    def step(self, t: int, dt: float) -> np.ndarray:
        """Return noise increment for all nodes at step t (variance sigma^2 * dt)."""
        if self.mode == "none" or self.sigma <= 0.0:
            return np.zeros(self.eta.shape[0], dtype=float)
        scale = float(self.scale_t[t]) if self.scale_t.size else 1.0
        return self.sigma * scale * self.eta[:, t] * float(np.sqrt(max(dt, 0.0)))


@dataclass
class ClipStats:
    clipped: bool = False
    non_finite: bool = False
    max_abs: float = 0.0


def _clip_state(state: np.ndarray, limit: float, stats: ClipStats) -> np.ndarray:
    if state.size:
        finite = np.isfinite(state)
        if not np.all(finite):
            stats.non_finite = True
            stats.clipped = True
        if np.any(finite):
            max_abs = float(np.max(np.abs(state[finite])))
            stats.max_abs = max(stats.max_abs, max_abs)
            if max_abs > limit:
                stats.clipped = True
        else:
            stats.max_abs = max(stats.max_abs, float("inf"))
            stats.clipped = True
    return np.clip(state, -limit, limit)


def _warn_if_clipped(stats: ClipStats, system_name: str, limit: float) -> None:
    if not stats.clipped:
        return
    extra = "non-finite values detected" if stats.non_finite else "magnitude exceeded limit"
    warnings.warn(
        f"{system_name}: state clipped to +/-{limit} ({extra}); max_abs={stats.max_abs:.3g}",
        RuntimeWarning,
    )


def build_process_noise(
    cfg,
    T: int,
    n_nodes: int,
    rng: np.random.Generator,
) -> ProcessNoise:
    """Precompute process noise drivers for reproducibility."""
    mode = str(cfg.robustness.dyn_noise_mode).lower()
    sigma = cfg.robustness.dyn_noise_std
    strength = cfg.robustness.dyn_noise_strength
    smooth_win = cfg.robustness.dyn_noise_smooth_win
    phi = cfg.robustness.dyn_noise_ar1_phi

    if mode in {"none", "off", "false", "0"} or sigma <= 0.0:
        return ProcessNoise(mode="none", sigma=0.0, scale_t=np.ones(int(T)), eta=np.zeros((int(n_nodes), int(T))))

    use_dynamic = "dynamic" in mode
    use_colored = "colored" in mode

    scale_t = make_dynamic_scale(T, strength=strength if use_dynamic else 0.0, smooth_win=smooth_win, rng=rng)
    if use_colored:
        eta = ar1_process(T, n_nodes, phi=phi, rng=rng)
    else:
        eta = rng.standard_normal((int(n_nodes), int(T)))
    return ProcessNoise(mode=mode, sigma=float(sigma), scale_t=scale_t, eta=eta)


def generate_adjacency(n: int, prob: float, topology: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a directed adjacency matrix A[src, tgt] in {0,1} with no self-loops."""
    n = int(n)
    prob = float(prob)
    topology = str(topology).upper()

    max_edges = n * (n - 1)
    if max_edges <= 0:
        return np.zeros((n, n), dtype=int)

    if topology == "ER":
        if prob <= 0.0:
            return np.zeros((n, n), dtype=int)
        if prob >= 1.0:
            A = np.ones((n, n), dtype=int)
            np.fill_diagonal(A, 0)
            return A

    attempts = 0
    while attempts < 200:
        attempts += 1
        if topology == "ER":
            A = (rng.random((n, n)) < prob).astype(int)
            np.fill_diagonal(A, 0)
        elif topology == "BA":
            m = int(max(1, min(n - 1, round(prob * (n - 1)))))
            G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(0, 2**32 - 1)))
            A = np.zeros((n, n), dtype=int)
            for u, v in sorted(G.edges()):
                if rng.random() < 0.5:
                    A[u, v] = 1
                else:
                    A[v, u] = 1
        elif topology == "SF":
            G = nx.scale_free_graph(n, seed=int(rng.integers(0, 2**32 - 1)))
            A = nx.to_numpy_array(G, dtype=int)
            A = (A > 0).astype(int)
            np.fill_diagonal(A, 0)
        else:
            raise ValueError(f"Unknown topology type: {topology}")

        e = int(A.sum())
        if 0 < e < max_edges:
            return A

    A = np.zeros((n, n), dtype=int)
    if n >= 2:
        A[0, 1] = 1
    return A


def rk4_step(state: np.ndarray, dt: float, drift, *drift_args) -> np.ndarray:
    """One RK4 step for state shaped [d, n]."""
    k1 = drift(state, *drift_args)
    k2 = drift(state + 0.5 * dt * k1, *drift_args)
    k3 = drift(state + 0.5 * dt * k2, *drift_args)
    k4 = drift(state + dt * k3, *drift_args)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _diffusive_coupling_x(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Incoming diffusive coupling on x: (mean_in(x_j) - x_i)."""
    deg_in = np.sum(A, axis=0).astype(float)
    sum_in = A.T @ x
    return (sum_in - deg_in * x) / np.maximum(1.0, deg_in)


def simulate_lorenz(
    cfg,
    n: int,
    t_steps: int,
    epsilon: float,
    prob: float,
    rng: np.random.Generator,
    topology: str,
    fixed_adj: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = max(1, int(cfg.system.downsample_factor))
    A = fixed_adj if fixed_adj is not None else generate_adjacency(n, prob, topology, rng)
    dt = cfg.integration.base_dt_by_system["Lorenz"]
    internal_steps = int(t_steps) * ds
    warm_steps = int(cfg.integration.warmup_steps) * ds

    state = rng.uniform(low=-10.0, high=10.0, size=(3, n)).astype(float)
    clip_stats = ClipStats()

    def drift(s: np.ndarray, A_: np.ndarray, eps: float) -> np.ndarray:
        x, y, z = s
        dx = 10.0 * (y - x)
        dy = x * (28.0 - z) - y
        dz = x * y - (8.0 / 3.0) * z
        dx = dx + eps * _diffusive_coupling_x(x, A_)
        return np.vstack([dx, dy, dz])

    pn = build_process_noise(cfg, warm_steps + internal_steps, n_nodes=n, rng=rng)

    for t in range(warm_steps):
        state = rk4_step(state, dt, drift, np.zeros_like(A), 0.0)
        if cfg.system.noise_inject_target == "x":
            state[0] += pn.step(t, dt)
        state = _clip_state(state, 1e6, clip_stats)

    xs = np.zeros((n, internal_steps), dtype=float)
    for t in range(internal_steps):
        idx = warm_steps + t
        state = rk4_step(state, dt, drift, A, float(epsilon))
        if cfg.system.noise_inject_target == "x":
            state[0] += pn.step(idx, dt)
        state = _clip_state(state, 1e6, clip_stats)
        xs[:, t] = state[0]

    xs = xs[:, ::ds]
    xs = add_observation_noise_snr(xs, cfg.system.obs_noise_snr_db, rng)
    xs = safe_zscore_rows(xs)
    _warn_if_clipped(clip_stats, "Lorenz", 1e6)
    return xs, A


def simulate_rossler(
    cfg,
    n: int,
    t_steps: int,
    epsilon: float,
    prob: float,
    rng: np.random.Generator,
    topology: str,
    hetero_sigma: float = 0.0,
    fixed_adj: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = max(1, int(cfg.system.downsample_factor))
    A = fixed_adj if fixed_adj is not None else generate_adjacency(n, prob, topology, rng)
    dt = cfg.integration.base_dt_by_system["Rossler"]
    internal_steps = int(t_steps) * ds
    warm_steps = int(cfg.integration.warmup_steps) * ds

    a, b, c0 = 0.2, 0.2, 5.7
    c = c0 + rng.normal(0.0, float(hetero_sigma), size=n)

    state = rng.uniform(low=-5.0, high=5.0, size=(3, n)).astype(float)
    clip_stats = ClipStats()

    def drift(s: np.ndarray, A_: np.ndarray, eps: float) -> np.ndarray:
        x, y, z = s
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        dx = dx + eps * _diffusive_coupling_x(x, A_)
        return np.vstack([dx, dy, dz])

    pn = build_process_noise(cfg, warm_steps + internal_steps, n_nodes=n, rng=rng)

    for t in range(warm_steps):
        state = rk4_step(state, dt, drift, np.zeros_like(A), 0.0)
        if cfg.system.noise_inject_target == "x":
            state[0] += pn.step(t, dt)
        state = _clip_state(state, 1e6, clip_stats)

    xs = np.zeros((n, internal_steps), dtype=float)
    for t in range(internal_steps):
        idx = warm_steps + t
        state = rk4_step(state, dt, drift, A, float(epsilon))
        if cfg.system.noise_inject_target == "x":
            state[0] += pn.step(idx, dt)
        state = _clip_state(state, 1e6, clip_stats)
        xs[:, t] = state[0]

    xs = xs[:, ::ds]
    xs = add_observation_noise_snr(xs, cfg.system.obs_noise_snr_db, rng)
    xs = safe_zscore_rows(xs)
    _warn_if_clipped(clip_stats, "Rossler", 1e6)
    return xs, A


def simulate_hindmarsh_rose(
    cfg,
    n: int,
    t_steps: int,
    epsilon: float,
    prob: float,
    rng: np.random.Generator,
    topology: str,
    fixed_adj: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = max(1, int(cfg.system.downsample_factor))
    A = fixed_adj if fixed_adj is not None else generate_adjacency(n, prob, topology, rng)
    dt = cfg.integration.base_dt_by_system["HindmarshRose"]
    internal_steps = int(t_steps) * ds
    warm_steps = int(cfg.integration.warmup_steps) * ds

    a, b, c, d = 1.0, 3.0, 1.0, 5.0
    r, s, x_R = 0.006, 4.0, -1.6
    I = 3.25

    state = rng.uniform(low=-2.0, high=2.0, size=(3, n)).astype(float)
    clip_stats = ClipStats()

    def drift(st: np.ndarray, A_: np.ndarray, eps: float) -> np.ndarray:
        x, y, z = st
        dx = y - a * x**3 + b * x**2 - z + I
        dy = c - d * x**2 - y
        dz = r * (s * (x - x_R) - z)
        dx = dx + eps * _diffusive_coupling_x(x, A_)
        return np.vstack([dx, dy, dz])

    pn = build_process_noise(cfg, warm_steps + internal_steps, n_nodes=n, rng=rng)

    for t in range(warm_steps):
        state = rk4_step(state, dt, drift, np.zeros_like(A), 0.0)
        if cfg.system.noise_inject_target == "x":
            state[0] += pn.step(t, dt)
        state = _clip_state(state, 1e6, clip_stats)

    xs = np.zeros((n, internal_steps), dtype=float)
    for t in range(internal_steps):
        idx = warm_steps + t
        state = rk4_step(state, dt, drift, A, float(epsilon))
        if cfg.system.noise_inject_target == "x":
            state[0] += pn.step(idx, dt)
        state = _clip_state(state, 1e6, clip_stats)
        xs[:, t] = state[0]

    xs = xs[:, ::ds]
    xs = add_observation_noise_snr(xs, cfg.system.obs_noise_snr_db, rng)
    xs = safe_zscore_rows(xs)
    _warn_if_clipped(clip_stats, "HindmarshRose", 1e6)
    return xs, A


def simulate_fitzhugh_nagumo(
    cfg,
    n: int,
    t_steps: int,
    epsilon: float,
    prob: float,
    rng: np.random.Generator,
    topology: str,
    fixed_adj: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """FitzHugh–Nagumo neuronal network (2D per node)."""
    ds = max(1, int(cfg.system.downsample_factor))
    A = fixed_adj if fixed_adj is not None else generate_adjacency(n, prob, topology, rng)
    dt = cfg.integration.base_dt_by_system["FitzHughNagumo"]
    internal_steps = int(t_steps) * ds
    warm_steps = int(cfg.integration.warmup_steps) * ds

    a, b, tau = 0.7, 0.8, 12.5
    I = 0.5

    state = rng.uniform(low=-2.0, high=2.0, size=(2, n)).astype(float)
    clip_stats = ClipStats()

    def drift(st: np.ndarray, A_: np.ndarray, eps: float) -> np.ndarray:
        v, w = st
        dv = v - (v**3) / 3.0 - w + I
        dw = (v + a - b * w) / tau
        dv = dv + eps * _diffusive_coupling_x(v, A_)
        return np.vstack([dv, dw])

    pn = build_process_noise(cfg, warm_steps + internal_steps, n_nodes=n, rng=rng)

    for t in range(warm_steps):
        state = rk4_step(state, dt, drift, np.zeros_like(A), 0.0)
        if cfg.system.noise_inject_target == "x":
            state[0] += pn.step(t, dt)
        state = _clip_state(state, 1e6, clip_stats)

    vs = np.zeros((n, internal_steps), dtype=float)
    for t in range(internal_steps):
        idx = warm_steps + t
        state = rk4_step(state, dt, drift, A, float(epsilon))
        if cfg.system.noise_inject_target == "x":
            state[0] += pn.step(idx, dt)
        state = _clip_state(state, 1e6, clip_stats)
        vs[:, t] = state[0]

    vs = vs[:, ::ds]
    vs = add_observation_noise_snr(vs, cfg.system.obs_noise_snr_db, rng)
    vs = safe_zscore_rows(vs)
    _warn_if_clipped(clip_stats, "FitzHughNagumo", 1e6)
    return vs, A


def simulate_kuramoto(
    cfg,
    n: int,
    t_steps: int,
    epsilon: float,
    prob: float,
    rng: np.random.Generator,
    topology: str,
    fixed_adj: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kuramoto oscillator network with heterogeneous intrinsic frequencies."""
    ds = max(1, int(cfg.system.downsample_factor))
    A = fixed_adj if fixed_adj is not None else generate_adjacency(n, prob, topology, rng)
    dt = cfg.integration.base_dt_by_system["Kuramoto"]
    internal_steps = int(t_steps) * ds
    warm_steps = int(cfg.integration.warmup_steps) * ds

    omega = rng.normal(float(cfg.integration.kuramoto_omega_mean), float(cfg.integration.kuramoto_omega_sigma), size=n)
    theta = rng.uniform(low=-np.pi, high=np.pi, size=n).astype(float)

    pn = build_process_noise(cfg, warm_steps + internal_steps, n_nodes=n, rng=rng)

    def drift_theta(th: np.ndarray, A_: np.ndarray, eps: float) -> np.ndarray:
        diff = th[:, None] - th[None, :]
        coupling_mat = A_ * np.sin(diff)
        sum_in = np.sum(coupling_mat, axis=0)
        deg_in = np.sum(A_, axis=0).astype(float)
        coupling = sum_in / np.maximum(1.0, deg_in)
        return omega + eps * coupling

    for t in range(warm_steps):
        k1 = drift_theta(theta, np.zeros_like(A), 0.0)
        k2 = drift_theta(theta + 0.5 * dt * k1, np.zeros_like(A), 0.0)
        k3 = drift_theta(theta + 0.5 * dt * k2, np.zeros_like(A), 0.0)
        k4 = drift_theta(theta + dt * k3, np.zeros_like(A), 0.0)
        theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if cfg.system.noise_inject_target == "x":
            theta = theta + pn.step(t, dt)

    obs = np.zeros((n, internal_steps), dtype=float)
    for t in range(internal_steps):
        idx = warm_steps + t
        k1 = drift_theta(theta, A, float(epsilon))
        k2 = drift_theta(theta + 0.5 * dt * k1, A, float(epsilon))
        k3 = drift_theta(theta + 0.5 * dt * k2, A, float(epsilon))
        k4 = drift_theta(theta + dt * k3, A, float(epsilon))
        theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if cfg.system.noise_inject_target == "x":
            theta = theta + pn.step(idx, dt)

        if str(cfg.integration.kuramoto_observable).lower() == "phase":
            obs[:, t] = theta
        else:
            obs[:, t] = np.sin(theta)

    obs = obs[:, ::ds]
    obs = add_observation_noise_snr(obs, cfg.system.obs_noise_snr_db, rng)
    obs = safe_zscore_rows(obs)
    return obs, A


def simulate_logistic(
    cfg,
    n: int,
    t_steps: int,
    epsilon: float,
    prob: float,
    rng: np.random.Generator,
    topology: str,
    fixed_adj: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = max(1, int(cfg.system.downsample_factor))
    A = fixed_adj if fixed_adj is not None else generate_adjacency(n, prob, topology, rng)
    r = 3.8

    internal_steps = int(t_steps) * ds
    warm_steps = int(cfg.integration.warmup_steps) * ds

    x = rng.random(n)
    pn = build_process_noise(cfg, warm_steps + internal_steps, n_nodes=n, rng=rng)

    for t in range(warm_steps):
        x = r * x * (1.0 - x)
        x = x + pn.step(t, dt=1.0)
        x = np.clip(x, 0.0, 1.0)

    xs = np.zeros((n, internal_steps), dtype=float)
    for t in range(internal_steps):
        idx = warm_steps + t
        coupling = _diffusive_coupling_x(x, A)
        x = r * x * (1.0 - x) + float(epsilon) * coupling
        x = x + pn.step(idx, dt=1.0)
        x = np.clip(x, 0.0, 1.0)
        xs[:, t] = x

    xs = xs[:, ::ds]
    xs = add_observation_noise_snr(xs, cfg.system.obs_noise_snr_db, rng)
    xs = safe_zscore_rows(xs)
    return xs, A


def simulate_henon(
    cfg,
    n: int,
    t_steps: int,
    epsilon: float,
    prob: float,
    rng: np.random.Generator,
    topology: str,
    fixed_adj: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Coupled Henon map network (2D per node; observe x)."""
    ds = max(1, int(cfg.system.downsample_factor))
    A = fixed_adj if fixed_adj is not None else generate_adjacency(n, prob, topology, rng)
    a, b = 1.4, 0.3

    internal_steps = int(t_steps) * ds
    warm_steps = int(cfg.integration.warmup_steps) * ds

    x = rng.uniform(low=-1.0, high=1.0, size=n)
    y = rng.uniform(low=-1.0, high=1.0, size=n)

    pn = build_process_noise(cfg, warm_steps + internal_steps, n_nodes=n, rng=rng)

    for t in range(warm_steps):
        x = np.clip(x, -5.0, 5.0)
        y = np.clip(y, -5.0, 5.0)
        x_new = 1.0 - a * x * x + y
        y_new = b * x
        x = x_new + pn.step(t, dt=1.0)
        y = y_new
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    xs = np.zeros((n, internal_steps), dtype=float)
    for t in range(internal_steps):
        idx = warm_steps + t
        x = np.clip(x, -5.0, 5.0)
        y = np.clip(y, -5.0, 5.0)
        coupling = _diffusive_coupling_x(x, A)
        x_new = 1.0 - a * x * x + y + float(epsilon) * coupling
        y_new = b * x
        x = x_new + pn.step(idx, dt=1.0)
        y = y_new
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xs[:, t] = x

    xs = xs[:, ::ds]
    xs = add_observation_noise_snr(xs, cfg.system.obs_noise_snr_db, rng)
    xs = safe_zscore_rows(xs)
    return xs, A


SIMULATORS = {
    "Lorenz": simulate_lorenz,
    "Rossler": simulate_rossler,
    "HindmarshRose": simulate_hindmarsh_rose,
    "FitzHughNagumo": simulate_fitzhugh_nagumo,
    "Kuramoto": simulate_kuramoto,
    "Logistic": simulate_logistic,
    "Henon": simulate_henon,
}


def generate_synthetic_data(
    cfg,
    n: int,
    t_steps: int,
    epsilon: float,
    prob: float,
    rng: np.random.Generator,
    topology: str,
    return_metadata: bool = False,
    fixed_adj: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray | int]]:
    """Generate observed time series data and ground-truth adjacency."""
    system_type = str(cfg.system.type)
    n_obs = int(n)
    if fixed_adj is not None:
        n_total = fixed_adj.shape[0]
    else:
        n_total = int(n_obs + max(0, int(cfg.system.n_hidden_nodes)))

    if system_type not in SIMULATORS:
        raise ValueError(f"Unknown system type: {system_type}")

    data_all, A_all = SIMULATORS[system_type](cfg, n_total, t_steps, epsilon, prob, rng, topology, fixed_adj=fixed_adj)
    data_obs = data_all[:n_obs]
    A_obs = A_all[:n_obs, :n_obs]
    if not return_metadata:
        return data_obs, A_obs

    metadata = {
        "adj_all": A_all,
        "n_obs": n_obs,
        "n_hidden": int(max(0, n_total - n_obs)),
    }
    return data_obs, A_obs, metadata
