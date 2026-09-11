from __future__ import annotations

"""
Synthetic open-network experiments for distributed online stochastic learning.

This script contains the numerical experiments corresponding to Section V-A.

Experiments
-----------
A. Effect of join/leave frequency
B. Effect of graph topology
C. Baseline comparisons
D. Parameter sensitivity (step size and discount factor)
E. Non-Gaussian / heavy-tailed gradient noise

The default execution runs Experiments A--E and saves the generated figures
as PNG and EPS files.

Notes
-----
- Experiments A--D use noise-free local gradients by default.
- Experiment C enables alternative update rules for baseline comparisons.
- Experiment E adds synthetic zero-mean gradient noise.
- The warm-start implementation initializes a (re)connecting agent from the
  mean of currently available agent states plus a small perturbation; it does
  not restore the agent's own stored checkpoint.
"""

# ---------------------------------------------------------------------------
# Copyright and acknowledgment
# ---------------------------------------------------------------------------
# Copyright (c) 2026 The University of Osaka
#
# This work was partially supported by Japan Society for the Promotion of
# Science KAKENHI Grant Number JP26K07551.

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("./output_open_network_v3")
LOG_FLOOR = 1e-12

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """Configuration for one synthetic open-network scenario."""

    # Network size
    max_active_agents: int
    total_agents: int | None = None
    min_active_agents: int | None = None

    # Algorithm
    horizon: int = 1000
    gamma: float = 0.99
    alpha0: float | None = None
    alpha_min: float | None = None

    # Algorithm variant (Experiment C)
    #   dsgd           : proposed method
    #   no_consensus   : local SGD only
    #   periodic       : consensus every avg_period iterations
    #   dual_averaging : naive open-network dual averaging
    algorithm: str = "dsgd"
    avg_period: int = 10

    # Gradient-noise model (Experiment E)
    #   none      : noise-free local gradients
    #   gaussian  : additive Gaussian noise
    #   student_t : finite-variance Student-t noise
    #   outlier   : Gaussian base noise + rare large outliers
    noise_model: str = "none"
    noise_std: float = 0.0
    t_dof: float = 3.0
    outlier_prob: float = 0.05
    outlier_scale: float = 8.0

    # Join / leave dynamics
    p_join: float = 0.005
    p_leave: float = 0.004
    min_dwell: int = 120
    max_dwell: int = 200

    # Graph topology: random | ring | complete | star
    topology: str = "random"
    extra_edge_prob: float = 0.20

    # Initialization of arriving / reconnecting agents
    warm_start: bool = True
    init_perturb: float = 0.05
    init_low: float = -1.0
    init_high: float = 1.0

    # Local objective:
    # f_i(x) = (mu_i / 2) x^2 + a_i (1 - cos x) + c_i x
    mu_low: float = 0.5
    mu_high: float = 1.0
    a_low: float = 1.5
    a_high: float = 2.0
    c_low: float = -0.6
    c_high: float = -0.4

    # True: draw each agent's coefficients once and keep them fixed.
    fix_coefficients: bool = True

    # Repeated trials
    num_trials: int = 3
    seed: int = 3407
    label: str = ""

    def resolved_alpha0(self) -> float:
        return 0.05 if self.alpha0 is None else self.alpha0

    def resolved_alpha_min(self) -> float:
        if self.alpha_min is not None:
            return self.alpha_min
        return self.resolved_alpha0() / 10.0

    def resolved_total_agents(self) -> int:
        if self.total_agents is not None:
            return self.total_agents
        return int(np.ceil(1.5 * self.max_active_agents))

    def resolved_min_active_agents(self) -> int:
        if self.min_active_agents is not None:
            return self.min_active_agents
        return max(3, self.max_active_agents // 5)

    def get_label(self) -> str:
        if self.label:
            return self.label
        return f"max_active={self.max_active_agents}"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    population_trace: np.ndarray
    cumulative_regi_trace: np.ndarray
    time_avg_regi_trace: np.ndarray
    instantaneous_stat_trace: np.ndarray
    cost_trace: np.ndarray
    residual_trace: np.ndarray
    max_abs_x_trace: np.ndarray
    total_regret_by_agent: np.ndarray
    episode_counts_by_agent: np.ndarray
    final_active_count: int


@dataclass
class ScenarioResult:
    config: Dict
    population_trace_mean: np.ndarray
    population_trace_representative: np.ndarray
    representative_trial_index: int
    cumulative_regi_trace_mean: np.ndarray
    time_avg_regi_trace_mean: np.ndarray
    instantaneous_stat_trace_mean: np.ndarray
    cost_trace_mean: np.ndarray
    residual_trace_mean: np.ndarray
    max_abs_x_trace_mean: np.ndarray
    total_regret_by_agent_mean: np.ndarray
    episode_counts_by_agent_mean: np.ndarray
    final_active_count_mean: float


# ---------------------------------------------------------------------------
# Local objective
# ---------------------------------------------------------------------------

def sample_local_coefficients(
    n: int,
    rng: np.random.Generator,
    cfg: ScenarioConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample coefficients satisfying 0 < mu_i < a_i."""
    mu = rng.uniform(cfg.mu_low, cfg.mu_high, size=n)
    a = rng.uniform(cfg.a_low, cfg.a_high, size=n)
    c = rng.uniform(cfg.c_low, cfg.c_high, size=n)

    assert np.all(mu > 0.0) and np.all(mu < a), (
        "coefficient ranges must satisfy 0 < mu_i < a_i"
    )
    return mu, a, c


def local_value(
    x: np.ndarray,
    mu: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Evaluate the local objective."""
    return 0.5 * mu * x**2 + a * (1.0 - np.cos(x)) + c * x


def local_gradient(
    x: np.ndarray,
    mu: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Evaluate the local objective gradient."""
    return mu * x + a * np.sin(x) + c


def aggregate_global_min(mu: float, a: float, c: float) -> float:
    """
    Compute the global minimum value of

        f(x) = (mu/2) x^2 + a (1 - cos x) + c x,

    using a grid search plus bisection over stationary-point brackets.
    """
    radius = (a + abs(c)) / mu + 1.0
    xs = np.linspace(-radius, radius, 2001)
    gs = mu * xs + a * np.sin(xs) + c

    def fval(x: np.ndarray | float) -> np.ndarray | float:
        return 0.5 * mu * x**2 + a * (1.0 - np.cos(x)) + c * x

    best = float(np.min(fval(xs)))

    sign_change = np.flatnonzero(
        np.sign(gs[:-1]) * np.sign(gs[1:]) < 0
    )

    for idx in sign_change:
        lo, hi = xs[idx], xs[idx + 1]
        glo = mu * lo + a * np.sin(lo) + c

        for _ in range(60):
            mid = 0.5 * (lo + hi)
            gmid = mu * mid + a * np.sin(mid) + c

            if glo * gmid <= 0.0:
                hi = mid
            else:
                lo, glo = mid, gmid

        best = min(
            best,
            float(fval(0.5 * (lo + hi))),
        )

    for x0 in xs[gs == 0.0]:
        best = min(best, float(fval(x0)))

    return best


# ---------------------------------------------------------------------------
# Synthetic gradient noise (Experiment E)
# ---------------------------------------------------------------------------

def gradient_noise(
    size: int,
    cfg: ScenarioConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate additive zero-mean gradient noise."""
    if cfg.noise_model == "none" or cfg.noise_std <= 0.0:
        return np.zeros(size)

    if cfg.noise_model == "gaussian":
        return rng.normal(
            0.0,
            cfg.noise_std,
            size=size,
        )

    if cfg.noise_model == "student_t":
        raw = rng.standard_t(
            cfg.t_dof,
            size=size,
        )
        scale = cfg.noise_std / np.sqrt(
            cfg.t_dof / (cfg.t_dof - 2.0)
        )
        return scale * raw

    if cfg.noise_model == "outlier":
        base = rng.normal(
            0.0,
            cfg.noise_std,
            size=size,
        )
        mask = rng.random(size) < cfg.outlier_prob
        spikes = (
            rng.choice([-1.0, 1.0], size=size)
            * cfg.outlier_scale
            * cfg.noise_std
        )
        return np.where(
            mask,
            base + spikes,
            base,
        )

    raise ValueError(
        f"unknown noise_model: {cfg.noise_model}"
    )


# ---------------------------------------------------------------------------
# Population dynamics
# ---------------------------------------------------------------------------

def initialize_population(
    cfg: ScenarioConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize the active population and dwell timers."""
    total = cfg.resolved_total_agents()

    n_init = min(
        max(
            cfg.resolved_min_active_agents(),
            int(round(0.6 * cfg.max_active_agents)),
        ),
        cfg.max_active_agents,
    )

    active = np.zeros(total, dtype=bool)
    ids = rng.choice(
        total,
        size=n_init,
        replace=False,
    )
    active[ids] = True

    dwell = np.zeros(total, dtype=int)
    dwell[ids] = rng.integers(
        cfg.min_dwell,
        cfg.max_dwell + 1,
        size=n_init,
    )

    return active, dwell


def evolve_active_set(
    active: np.ndarray,
    dwell: np.ndarray,
    cfg: ScenarioConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the active set at the next iteration."""
    total = len(active)
    min_act = cfg.resolved_min_active_agents()
    max_act = cfg.max_active_agents

    active_set = set(
        np.flatnonzero(active).tolist()
    )
    inactive_set = set(range(total)) - active_set

    free_active = [
        i for i in active_set
        if dwell[i] == 0
    ]
    free_inactive = [
        i for i in inactive_set
        if dwell[i] == 0
    ]

    leavers = {
        i for i in free_active
        if rng.random() < cfg.p_leave
    }
    joiners = {
        i for i in free_inactive
        if rng.random() < cfg.p_join
    }

    def planned() -> int:
        return (
            len(active_set)
            - len(leavers)
            + len(joiners)
        )

    while planned() > max_act:
        excess = planned() - max_act

        cands = (
            sorted(joiners)
            if joiners
            else sorted(
                set(free_active) - leavers
            )
        )

        drop = rng.choice(
            cands,
            size=min(excess, len(cands)),
            replace=False,
        )

        if joiners:
            joiners -= set(drop.tolist())
        else:
            leavers |= set(drop.tolist())

    while planned() < min_act:
        shortage = min_act - planned()

        if leavers:
            cancel = rng.choice(
                sorted(leavers),
                size=min(
                    shortage,
                    len(leavers),
                ),
                replace=False,
            )
            leavers -= set(cancel.tolist())

        elif free_inactive:
            cands = sorted(
                set(free_inactive) - joiners
            )

            if not cands:
                break

            extra = rng.choice(
                cands,
                size=min(
                    shortage,
                    len(cands),
                ),
                replace=False,
            )
            joiners |= set(extra.tolist())

        else:
            break

    # Ensure the remaining set is nonempty.
    remaining = active_set - leavers

    if active_set and not remaining:
        rescued = int(
            rng.choice(sorted(leavers))
        )
        leavers.remove(rescued)

        if planned() > max_act and joiners:
            joiners.remove(
                int(rng.choice(sorted(joiners)))
            )

    next_active = active.copy()

    if leavers:
        next_active[list(leavers)] = False

    if joiners:
        next_active[list(joiners)] = True

    next_dwell = np.maximum(
        dwell - 1,
        0,
    )

    changed = list(
        leavers | joiners
    )

    if changed:
        next_dwell[changed] = rng.integers(
            cfg.min_dwell,
            cfg.max_dwell + 1,
            size=len(changed),
        )

    return next_active, next_dwell


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    ids: np.ndarray,
    rng: np.random.Generator,
    topology: str,
    extra_edge_prob: float,
) -> Dict[int, set]:
    """Build the requested undirected communication graph."""
    m = len(ids)

    neighbors: Dict[int, set] = {
        int(i): set()
        for i in ids
    }

    if m <= 1:
        return neighbors

    if topology == "ring":
        for p in range(m):
            u = int(ids[p])
            v = int(ids[(p + 1) % m])
            neighbors[u].add(v)
            neighbors[v].add(u)

    elif topology == "complete":
        for a in range(m):
            for b in range(a + 1, m):
                i = int(ids[a])
                j = int(ids[b])
                neighbors[i].add(j)
                neighbors[j].add(i)

    elif topology == "star":
        hub = int(ids[0])

        for p in range(1, m):
            spoke = int(ids[p])
            neighbors[hub].add(spoke)
            neighbors[spoke].add(hub)

    else:
        # Random spanning tree + Erdos-Renyi extra edges.
        perm = ids[rng.permutation(m)]

        for p in range(1, m):
            u = int(perm[p])
            v = int(
                perm[rng.integers(0, p)]
            )
            neighbors[u].add(v)
            neighbors[v].add(u)

        for a in range(m):
            for b in range(a + 1, m):
                i = int(ids[a])
                j = int(ids[b])

                if (
                    j not in neighbors[i]
                    and rng.random() < extra_edge_prob
                ):
                    neighbors[i].add(j)
                    neighbors[j].add(i)

    return neighbors


def metropolis_weights(
    ids: np.ndarray,
    neighbors: Dict[int, set],
) -> np.ndarray:
    """Construct the Metropolis weight matrix."""
    m = len(ids)
    W = np.zeros((m, m))

    pos = {
        int(agent): p
        for p, agent in enumerate(ids)
    }
    degree = {
        agent: len(neighbors[agent])
        for agent in neighbors
    }

    for i in ids:
        row = pos[int(i)]

        for j in neighbors[int(i)]:
            col = pos[int(j)]

            W[row, col] = (
                1.0
                / (
                    1.0
                    + max(
                        degree[int(i)],
                        degree[int(j)],
                    )
                )
            )

        W[row, row] = (
            1.0 - W[row].sum()
        )

    return W


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def simulate_trial(
    cfg: ScenarioConfig,
    seed: int,
) -> TrialResult:
    """Run one trial for the proposed method or an Experiment-C baseline."""
    rng = np.random.default_rng(seed)

    total = cfg.resolved_total_agents()
    alpha0 = cfg.resolved_alpha0()
    alpha_min = cfg.resolved_alpha_min()

    active, dwell = initialize_population(
        cfg,
        rng,
    )
    prev_active = np.zeros_like(active)

    x = np.full(
        total,
        np.nan,
    )

    def initialize_agents(
        ids: np.ndarray,
    ) -> None:
        if ids.size == 0:
            return

        carried = ~np.isnan(x)
        carried[ids] = False

        if cfg.warm_start and carried.any():
            anchor = float(
                np.mean(x[carried])
            )
            x[ids] = (
                anchor
                + rng.uniform(
                    -cfg.init_perturb,
                    cfg.init_perturb,
                    size=ids.size,
                )
            )
        else:
            x[ids] = rng.uniform(
                cfg.init_low,
                cfg.init_high,
                size=ids.size,
            )

    arr0 = active & ~prev_active
    initialize_agents(
        np.flatnonzero(arr0)
    )

    # Used only by the dual-averaging baseline.
    z_dual = np.zeros(total)

    # Fixed per-agent local-objective coefficients.
    coef_mu = np.zeros(total)
    coef_a = np.zeros(total)
    coef_c = np.zeros(total)
    coef_ready = np.zeros(
        total,
        dtype=bool,
    )

    def ensure_coefficients(
        ids: np.ndarray,
    ) -> None:
        new = ids[
            ~coef_ready[ids]
        ]

        if new.size:
            mu, a, c = (
                sample_local_coefficients(
                    new.size,
                    rng,
                    cfg,
                )
            )

            coef_mu[new] = mu
            coef_a[new] = a
            coef_c[new] = c
            coef_ready[new] = True

    if cfg.fix_coefficients:
        ensure_coefficients(
            np.flatnonzero(active)
        )

    # Episode-wise regret bookkeeping.
    in_episode = np.zeros(
        total,
        dtype=bool,
    )
    ep_acc = np.zeros(total)
    total_regi = np.zeros(total)
    ep_count = np.zeros(
        total,
        dtype=int,
    )

    K = cfg.horizon

    pop_trace = np.zeros(K)
    cum_regi_trace = np.zeros(K)
    tavg_regi_trace = np.zeros(K)
    stat_trace = np.zeros(K)
    cost_trace = np.zeros(K)
    resid_trace = np.zeros(K)
    maxx_trace = np.zeros(K)

    for k in range(K):
        alpha_k = max(
            alpha0 / np.sqrt(k + 1.0),
            alpha_min,
        )

        pop_trace[k] = active.sum()

        # Initialize newly arrived agents.
        arr_k = active & ~prev_active

        if arr_k.any():
            initialize_agents(
                np.flatnonzero(arr_k)
            )

            z_dual[arr_k] = 0.0

            if cfg.fix_coefficients:
                ensure_coefficients(
                    np.flatnonzero(arr_k)
                )

        next_active, next_dwell = (
            evolve_active_set(
                active,
                dwell,
                cfg,
                rng,
            )
        )

        remaining = (
            active & next_active
        )
        leaving = (
            active & ~next_active
        )

        active_ids = np.flatnonzero(
            active
        )

        active_pos = {
            int(agent): p
            for p, agent in enumerate(active_ids)
        }

        x_active = x[active_ids]

        neighbors = build_graph(
            active_ids,
            rng,
            cfg.topology,
            cfg.extra_edge_prob,
        )

        W = metropolis_weights(
            active_ids,
            neighbors,
        )

        if cfg.fix_coefficients:
            mu = coef_mu[active_ids]
            a = coef_a[active_ids]
            c = coef_c[active_ids]
        else:
            mu, a, c = (
                sample_local_coefficients(
                    len(active_ids),
                    rng,
                    cfg,
                )
            )

        # Local gradients; Experiment E optionally adds synthetic noise.
        grads_active = local_gradient(
            x_active,
            mu,
            a,
            c,
        )
        grads_active = (
            grads_active
            + gradient_noise(
                len(active_ids),
                cfg,
                rng,
            )
        )

        # ---------------------------------------------------------------
        # Update rule
        # ---------------------------------------------------------------
        if cfg.algorithm == "dsgd":
            z_active = W @ x_active
            x_next_active = (
                z_active
                - alpha_k * grads_active
            )

        elif cfg.algorithm == "no_consensus":
            x_next_active = (
                x_active
                - alpha_k * grads_active
            )

        elif cfg.algorithm == "periodic":
            if k % cfg.avg_period == 0:
                z_active = W @ x_active
            else:
                z_active = x_active

            x_next_active = (
                z_active
                - alpha_k * grads_active
            )

        elif cfg.algorithm == "dual_averaging":
            z_active_dual = (
                W @ z_dual[active_ids]
                + grads_active
            )

            x_next_active = (
                -alpha_k
                * z_active_dual
            )

            z_next_dual = np.zeros(total)
            z_next_dual[
                active_ids
            ] = z_active_dual
            z_dual = z_next_dual

        else:
            raise ValueError(
                f"unknown algorithm: {cfg.algorithm}"
            )

        # ---------------------------------------------------------------
        # Regret, cost, and residual
        # ---------------------------------------------------------------
        remaining_ids = np.flatnonzero(
            remaining
        )

        if remaining_ids.size:
            remaining_pos = np.array([
                active_pos[int(agent)]
                for agent in remaining_ids
            ])

            x_remaining = x[
                remaining_ids
            ]

            r_mu = mu[
                remaining_pos
            ]
            r_a = a[
                remaining_pos
            ]
            r_c = c[
                remaining_pos
            ]

            mu_bar = float(
                r_mu.mean()
            )
            a_bar = float(
                r_a.mean()
            )
            c_bar = float(
                r_c.mean()
            )

            stat_sq = (
                local_gradient(
                    x_remaining,
                    mu_bar,
                    a_bar,
                    c_bar,
                )
                ** 2
            )

            stat_trace[k] = float(
                stat_sq.mean()
            )

            fk_vals = local_value(
                x_remaining,
                mu_bar,
                a_bar,
                c_bar,
            )

            fk_star = aggregate_global_min(
                mu_bar,
                a_bar,
                c_bar,
            )

            cost_trace[k] = float(
                fk_vals.mean()
            )

            resid_trace[k] = float(
                np.maximum(
                    fk_vals - fk_star,
                    0.0,
                ).mean()
            )

            for agent, sq in zip(
                remaining_ids.tolist(),
                stat_sq.tolist(),
            ):
                if not in_episode[agent]:
                    ep_acc[agent] = 0.0
                    in_episode[agent] = True

                ep_acc[agent] = (
                    cfg.gamma
                    * ep_acc[agent]
                    + sq
                )

        else:
            stat_trace[k] = 0.0

            cost_trace[k] = (
                cost_trace[k - 1]
                if k > 0
                else 0.0
            )

            resid_trace[k] = (
                resid_trace[k - 1]
                if k > 0
                else 0.0
            )

        # Close episodes for departing agents.
        for agent in np.flatnonzero(
            leaving
        ).tolist():
            if in_episode[agent]:
                total_regi[agent] += (
                    ep_acc[agent]
                )
                ep_count[agent] += 1
                ep_acc[agent] = 0.0
                in_episode[agent] = False

        current_total = (
            total_regi
            + np.where(
                in_episode,
                ep_acc,
                0.0,
            )
        )

        participated = (
            (ep_count > 0)
            | in_episode
        )

        if participated.any():
            cum_val = float(
                current_total[
                    participated
                ].mean()
            )
        elif k > 0:
            cum_val = (
                cum_regi_trace[k - 1]
            )
        else:
            cum_val = 0.0

        cum_regi_trace[k] = cum_val
        tavg_regi_trace[k] = (
            cum_val / (k + 1.0)
        )

        # Carry states forward for agents active at k+1.
        x_next = np.full_like(
            x,
            np.nan,
        )

        for p, agent in enumerate(
            active_ids.tolist()
        ):
            if next_active[agent]:
                x_next[agent] = (
                    x_next_active[p]
                )

        x = x_next

        if next_active.any():
            maxx_trace[k] = float(
                np.nanmax(
                    np.abs(
                        x[next_active]
                    )
                )
            )

        prev_active = active.copy()
        active = next_active
        dwell = next_dwell

    # Close episodes still active at the horizon.
    for agent in np.flatnonzero(
        in_episode
    ).tolist():
        total_regi[agent] += (
            ep_acc[agent]
        )
        ep_count[agent] += 1

    return TrialResult(
        population_trace=pop_trace,
        cumulative_regi_trace=cum_regi_trace,
        time_avg_regi_trace=tavg_regi_trace,
        instantaneous_stat_trace=stat_trace,
        cost_trace=cost_trace,
        residual_trace=resid_trace,
        max_abs_x_trace=maxx_trace,
        total_regret_by_agent=total_regi,
        episode_counts_by_agent=ep_count,
        final_active_count=int(
            active.sum()
        ),
    )


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(
    cfg: ScenarioConfig,
) -> ScenarioResult:
    """Run all trials for one scenario and average the recorded metrics."""
    trials = [
        simulate_trial(
            cfg,
            (
                cfg.seed
                + 1000 * trial_index
                + 37 * cfg.max_active_agents
            ),
        )
        for trial_index in range(
            cfg.num_trials
        )
    ]

    def mean_stack(
        attr: str,
    ) -> np.ndarray:
        return np.stack([
            getattr(trial, attr)
            for trial in trials
        ]).mean(0)

    pop_mean = mean_stack(
        "population_trace"
    )

    distances = [
        np.linalg.norm(
            trial.population_trace
            - pop_mean
        )
        for trial in trials
    ]

    representative_index = int(
        np.argmin(distances)
    )

    return ScenarioResult(
        config=asdict(cfg),
        population_trace_mean=pop_mean,
        population_trace_representative=(
            trials[
                representative_index
            ].population_trace.copy()
        ),
        representative_trial_index=(
            representative_index
        ),
        cumulative_regi_trace_mean=mean_stack(
            "cumulative_regi_trace"
        ),
        time_avg_regi_trace_mean=mean_stack(
            "time_avg_regi_trace"
        ),
        instantaneous_stat_trace_mean=mean_stack(
            "instantaneous_stat_trace"
        ),
        cost_trace_mean=mean_stack(
            "cost_trace"
        ),
        residual_trace_mean=mean_stack(
            "residual_trace"
        ),
        max_abs_x_trace_mean=mean_stack(
            "max_abs_x_trace"
        ),
        total_regret_by_agent_mean=mean_stack(
            "total_regret_by_agent"
        ),
        episode_counts_by_agent_mean=mean_stack(
            "episode_counts_by_agent"
        ),
        final_active_count_mean=float(
            np.mean([
                trial.final_active_count
                for trial in trials
            ])
        ),
    )


def run_config_group(
    configs: List[ScenarioConfig],
) -> List[ScenarioResult]:
    """Run a list of configurations and print concise progress messages."""
    results: List[ScenarioResult] = []

    for cfg in configs:
        print(f"  {cfg.get_label()}")

        result = run_scenario(cfg)

        print(
            "    -> representative trial index: "
            f"{result.representative_trial_index}"
        )

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Figure utilities
# ---------------------------------------------------------------------------

def save_fig(
    fig: plt.Figure,
    stem: Path,
) -> None:
    """Save a figure as PNG (200 dpi) and EPS."""
    fig.savefig(
        str(stem) + ".png",
        dpi=200,
        bbox_inches="tight",
    )
    fig.savefig(
        str(stem) + ".eps",
        format="eps",
        bbox_inches="tight",
    )
    plt.close(fig)


def make_comparison_fig(
    results: List[ScenarioResult],
    labels: List[str],
    stem: Path,
    exp_id: str = "",
) -> None:
    """
    Save separate comparison figures for one experiment.

    Generated metrics:
    - cumulative E[Reg_i]
    - time-averaged regret
    - active-agent population
    - aggregate cost
    - aggregate cost residual (log scale)
    """
    stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    panels = [
        (
            "cumulative_Regi",
            r"Cumulative $\mathrm{E}[\mathrm{Reg}_i]$",
            lambda result: result.cumulative_regi_trace_mean,
            "linear",
            True,
            True,
        ),
        (
            "time_avg_Regi",
            r"Population-averaged regret $\bar{T}(k)$",
            lambda result: result.time_avg_regi_trace_mean,
            "linear",
            True,
            True,
        ),
        (
            "population",
            "$|\\mathcal{N}_k|$  (active agents)",
            lambda result: result.population_trace_representative,
            "linear",
            True,
            True,
        ),
        (
            "cost",
            r"$f_k(x_i(k))$",
            lambda result: result.cost_trace_mean,
            "linear",
            False,
            False,
        ),
        (
            "residual_log",
            r"Population-averaged cost residual $\bar{E}(k)$",
            lambda result: np.maximum(
                result.residual_trace_mean,
                LOG_FLOOR,
            ),
            "log",
            False,
            False,
        ),
    ]

    max_len = max(
        result.cumulative_regi_trace_mean.size
        for result in results
    )

    id_suffix = (
        f"_{exp_id}"
        if exp_id
        else ""
    )

    for (
        suffix,
        ylabel,
        data_fn,
        yscale,
        integer_ticks,
        bottom_zero,
    ) in panels:
        fig, ax = plt.subplots()

        for index, (
            result,
            label,
        ) in enumerate(
            zip(results, labels)
        ):
            iterations = np.arange(
                1,
                result.cumulative_regi_trace_mean.size + 1,
            )

            ax.plot(
                iterations,
                data_fn(result),
                label=label,
                color=COLORS[
                    index % len(COLORS)
                ],
            )

        ax.set_xlim(
            0,
            max_len,
        )

        if yscale == "log":
            ax.set_yscale("log")
        else:
            if integer_ticks:
                ax.yaxis.set_major_locator(
                    MaxNLocator(integer=True)
                )

            if bottom_zero:
                ax.set_ylim(bottom=0)

        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel(ylabel)
        ax.legend()

        fig.tight_layout()

        save_fig(
            fig,
            Path(
                str(stem)
                + f"_{suffix}{id_suffix}"
            ),
        )


def save_scenario_outputs(
    result: ScenarioResult,
    out: Path,
) -> None:
    """
    Save detailed CSV/JSON summaries and per-scenario figures.

    This utility is provided for optional detailed inspection and is not
    invoked by the default Experiment A--E runner.
    """
    cfg = result.config
    max_active = int(
        cfg["max_active_agents"]
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    iterations = np.arange(
        1,
        result.population_trace_mean.size + 1,
    )

    np.savetxt(
        out
        / f"scenario_{max_active}_timeseries.csv",
        np.column_stack([
            iterations,
            result.population_trace_mean,
            result.cumulative_regi_trace_mean,
            result.time_avg_regi_trace_mean,
            result.instantaneous_stat_trace_mean,
            result.cost_trace_mean,
            result.residual_trace_mean,
            result.max_abs_x_trace_mean,
        ]),
        delimiter=",",
        header=(
            "k,active,cum_Regi,tavg_Regi,"
            "stat,cost,residual,max_abs_x"
        ),
        comments="",
    )

    with open(
        out
        / f"scenario_{max_active}_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "config": result.config,
                "final_active_mean": (
                    result.final_active_count_mean
                ),
                "final_cum_Regi": float(
                    result.cumulative_regi_trace_mean[-1]
                ),
                "final_tavg_Regi": float(
                    result.time_avg_regi_trace_mean[-1]
                ),
                "final_cost": float(
                    result.cost_trace_mean[-1]
                ),
                "final_residual": float(
                    result.residual_trace_mean[-1]
                ),
            },
            file,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Experiment A: join / leave frequency
# ---------------------------------------------------------------------------

def exp_A(base: Path) -> None:
    """Compare slow, medium, and fast membership changes."""
    print(
        "\n=== Experiment A: "
        "Join/Leave Frequency ==="
    )

    configs = [
        ScenarioConfig(
            max_active_agents=20,
            p_join=0.005,
            p_leave=0.004,
            min_dwell=120,
            max_dwell=200,
            label="Slow",
        ),
        ScenarioConfig(
            max_active_agents=20,
            p_join=0.02,
            p_leave=0.016,
            min_dwell=60,
            max_dwell=100,
            label="Medium",
        ),
        ScenarioConfig(
            max_active_agents=20,
            p_join=0.05,
            p_leave=0.04,
            min_dwell=30,
            max_dwell=60,
            label="Fast",
        ),
    ]

    results = run_config_group(
        configs
    )

    make_comparison_fig(
        results,
        [
            cfg.get_label()
            for cfg in configs
        ],
        stem=(
            base
            / "A_join_leave_frequency"
            / "comparison"
        ),
        exp_id="A",
    )

    print(
        "  -> "
        f"{base / 'A_join_leave_frequency'}"
    )


# ---------------------------------------------------------------------------
# Experiment B: graph topology
# ---------------------------------------------------------------------------

def exp_B(base: Path) -> None:
    """Compare several communication-graph topologies."""
    print(
        "\n=== Experiment B: "
        "Graph Topology ==="
    )

    common = dict(
        max_active_agents=20
    )

    configs = [
        ScenarioConfig(
            **common,
            topology="ring",
            label="Ring",
        ),
        ScenarioConfig(
            **common,
            topology="random",
            extra_edge_prob=0.20,
            label="Random (sparse)",
        ),
        ScenarioConfig(
            **common,
            topology="random",
            extra_edge_prob=0.60,
            label="Random (dense)",
        ),
        ScenarioConfig(
            **common,
            topology="complete",
            label="Complete",
        ),
        ScenarioConfig(
            **common,
            topology="star",
            label="Star",
        ),
    ]

    results = run_config_group(
        configs
    )

    make_comparison_fig(
        results,
        [
            cfg.get_label()
            for cfg in configs
        ],
        stem=(
            base
            / "B_graph_topology"
            / "comparison"
        ),
        exp_id="B",
    )

    print(
        "  -> "
        f"{base / 'B_graph_topology'}"
    )


# ---------------------------------------------------------------------------
# Experiment C: baseline comparisons
# ---------------------------------------------------------------------------

def exp_C(base: Path) -> None:
    """Compare the proposed method against three baseline variants."""
    print(
        "\n=== Experiment C: "
        "Baseline Comparisons ==="
    )

    common = dict(
        max_active_agents=20,
        p_join=0.005,
        p_leave=0.004,
        min_dwell=120,
        max_dwell=200,
    )

    configs = [
        ScenarioConfig(
            **common,
            algorithm="dsgd",
            label="Proposed (Algorithm 1)",
        ),
        ScenarioConfig(
            **common,
            algorithm="no_consensus",
            label="No-consensus SGD",
        ),
        ScenarioConfig(
            **common,
            algorithm="periodic",
            avg_period=10,
            label="Periodic averaging (every 10)",
        ),
        ScenarioConfig(
            **common,
            algorithm="dual_averaging",
            label="Open dual averaging (naive)",
        ),
    ]

    results = run_config_group(
        configs
    )

    make_comparison_fig(
        results,
        [
            cfg.get_label()
            for cfg in configs
        ],
        stem=(
            base
            / "C_baseline_comparison"
            / "comparison"
        ),
        exp_id="C",
    )

    print(
        "  -> "
        f"{base / 'C_baseline_comparison'}"
    )


# ---------------------------------------------------------------------------
# Experiment D: parameter sensitivity
# ---------------------------------------------------------------------------

def exp_D(base: Path) -> None:
    """Study sensitivity to alpha0 (eta) and gamma."""
    print(
        "\n=== Experiment D: "
        "Parameter Sensitivity ==="
    )

    common = dict(
        max_active_agents=20,
        p_join=0.005,
        p_leave=0.004,
        min_dwell=120,
        max_dwell=200,
    )

    # D-1: step-size coefficient eta
    etas = [
        0.01,
        0.05,
        0.1,
    ]

    eta_configs = [
        ScenarioConfig(
            **common,
            alpha0=eta,
            gamma=0.99,
            label=rf"$\eta={eta}$",
        )
        for eta in etas
    ]

    eta_results = run_config_group(
        eta_configs
    )

    make_comparison_fig(
        eta_results,
        [
            cfg.get_label()
            for cfg in eta_configs
        ],
        stem=(
            base
            / "D_parameter_sensitivity"
            / "stepsize_eta"
        ),
        exp_id="D",
    )

    # D-2: discount factor gamma
    gammas = [
        0.90,
        0.99,
        0.999,
    ]

    gamma_configs = [
        ScenarioConfig(
            **common,
            alpha0=0.05,
            gamma=gamma,
            label=rf"$\gamma={gamma}$",
        )
        for gamma in gammas
    ]

    gamma_results = run_config_group(
        gamma_configs
    )

    make_comparison_fig(
        gamma_results,
        [
            cfg.get_label()
            for cfg in gamma_configs
        ],
        stem=(
            base
            / "D_parameter_sensitivity"
            / "discount_gamma"
        ),
        exp_id="D",
    )

    print(
        "  -> "
        f"{base / 'D_parameter_sensitivity'}"
    )


# ---------------------------------------------------------------------------
# Experiment E: non-Gaussian / heavy-tailed noise
# ---------------------------------------------------------------------------

def exp_E(base: Path) -> None:
    """Compare Gaussian, Student-t, and rare-outlier gradient noise."""
    print(
        "\n=== Experiment E: "
        "Non-Gaussian / Heavy-Tailed Noise ==="
    )

    sigma = 2.0

    common = dict(
        max_active_agents=20,
        p_join=0.005,
        p_leave=0.004,
        min_dwell=120,
        max_dwell=200,
        noise_std=sigma,
    )

    configs = [
        ScenarioConfig(
            **common,
            noise_model="gaussian",
            label=rf"Gaussian ($\sigma={sigma}$)",
        ),
        ScenarioConfig(
            **common,
            noise_model="student_t",
            t_dof=3.0,
            label=(
                r"Student-$t$ "
                r"($\nu=3$, same variance)"
            ),
        ),
        ScenarioConfig(
            **common,
            noise_model="outlier",
            outlier_prob=0.05,
            outlier_scale=8.0,
            label=(
                r"Rare outliers "
                r"($p=0.05$)"
            ),
        ),
    ]

    results = run_config_group(
        configs
    )

    make_comparison_fig(
        results,
        [
            cfg.get_label()
            for cfg in configs
        ],
        stem=(
            base
            / "E_gradient_noise"
            / "comparison"
        ),
        exp_id="E",
    )

    print(
        "  -> "
        f"{base / 'E_gradient_noise'}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_experiments(
    base: Path,
) -> None:
    """Run Experiments A--E in sequence."""
    exp_A(base)
    exp_B(base)
    exp_C(base)
    exp_D(base)
    exp_E(base)


def main() -> None:
    """Entry point for reproducing all synthetic experiments."""
    run_all_experiments(
        DEFAULT_OUTPUT_DIR
    )

    print(
        "\nAll outputs (PNG + EPS) saved under: "
        f"{DEFAULT_OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
