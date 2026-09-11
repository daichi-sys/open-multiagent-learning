from __future__ import annotations

"""
MNIST experiment for distributed stochastic gradient descent over an open network.

This script implements the numerical experiment with:
- dynamically joining and leaving agents,
- a k-nearest ring communication graph,
- mini-batch stochastic gradients,
- warm-start initialization for arriving agents,
- episode-wise discounted regret,
- MNIST training/test accuracy evaluation,
- linear- and log-scale loss plots,
- an approximate loss-residual plot.

Important
---------
The residual plotted in this script uses

    f_star = min_k f_k(x_i(k))

along each simulation run as an empirical reference value. Therefore, the
residual is an observed-loss residual, not the exact quantity
f_k(x_i(k)) - f_k(x*(k)) unless that minimum coincides with the true optimum.
"""

# ---------------------------------------------------------------------------
# Copyright and acknowledgment
# ---------------------------------------------------------------------------
# Copyright (c) 2026 The University of Osaka
#
# This work was partially supported by Japan Society for the Promotion of
# Science KAKENHI Grant Number JP26K07551.

from dataclasses import dataclass
from pathlib import Path
import math
import os
import random
from typing import Dict, List, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the MNIST open-network experiment."""

    num_agents: int = 20
    num_iterations: int = 600

    radius: float = 5000.0
    alpha_init: float = 0.8
    batch_size: int = 256
    gamma: float = 0.97

    w_bar: float = 0.01
    num_neighbors: int = 3

    fluctuation_period: int = 30
    core_count: int = 0
    min_active: int = 10
    warm_start: bool = True

    seed: int = 340
    device: str | None = None

    output_dir: str = "./output_agent0_regret_cases"


FLUCTUATION_RATIOS = (0.1, 0.3, 0.5)
RATIO_LABELS = (r"$\rho=0.1$", r"$\rho=0.3$", r"$\rho=0.5$")
LOG_FLOOR = 1e-8


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def fix_random_seed(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset and model utilities
# ---------------------------------------------------------------------------

def load_mnist(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """Download (if necessary) and load the MNIST training and test sets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    return train_dataset, test_dataset


def build_model() -> nn.Module:
    """Return the linear classifier used by each agent."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10),
    )


def create_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    """Create a shuffled mini-batch data loader."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 256,
    device: torch.device | str = "cpu",
) -> float:
    """Evaluate classification accuracy on one dataset."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            predicted = output.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    return correct / total


# ---------------------------------------------------------------------------
# Open-network management
# ---------------------------------------------------------------------------

def manage_open_network(
    num_agents: int,
    agent_active_prev: np.ndarray,
    agent_timer: np.ndarray,
    fluctuation_period: int = 30,
    fluctuation_ratio: float = 0.1,
    core_count: int | None = None,
    min_active: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Set[int], Set[int], Set[int], Set[int]]:
    """
    Update the active-agent set.

    Core agents are always active. Non-core agents may join or leave. After
    joining, an agent remains active for a random dwell time in
    [fluctuation_period, fluctuation_period + 9].

    Returns
    -------
    agent_active:
        Current activity array (1 = active, 0 = inactive).
    agent_timer:
        Updated dwell timers.
    S_a:
        Arriving agents, N_k \\ N_{k-1}.
    S_d:
        Departing agents, N_{k-1} \\ N_k.
    S_r:
        Remaining agents, N_k ∩ N_{k-1}.
    N_k:
        Current active-agent set.
    """
    agent_active = agent_active_prev.copy()

    if core_count is None:
        core_count = int(0.75 * num_agents)

    for i in range(core_count):
        agent_active[i] = 1

    candidate_indices = list(range(core_count, num_agents))
    fluctuation_size = max(
        1,
        int(round(fluctuation_ratio * num_agents)),
    )

    flip_indices = np.array([], dtype=int)

    if candidate_indices:
        flip_indices = np.random.choice(
            candidate_indices,
            min(fluctuation_size, len(candidate_indices)),
            replace=False,
        )

        for i in flip_indices:
            if agent_active[i] == 1:
                if agent_timer[i] > 0:
                    agent_timer[i] -= 1
                elif np.random.randint(0, 2) == 0:
                    agent_active[i] = 0
            else:
                if np.random.randint(0, 2) == 1:
                    agent_active[i] = 1
                    if fluctuation_period > 0:
                        agent_timer[i] = int(
                            np.random.randint(
                                fluctuation_period,
                                fluctuation_period + 10,
                            )
                        )
                    else:
                        agent_timer[i] = 0

    selected = set(int(i) for i in flip_indices)
    for i in candidate_indices:
        if (
            agent_active[i] == 1
            and i not in selected
            and agent_timer[i] > 0
        ):
            agent_timer[i] -= 1

    curr_active_count = int(agent_active.sum())
    if curr_active_count < min_active:
        inactive = [
            i for i in range(num_agents)
            if agent_active[i] == 0
        ]
        np.random.shuffle(inactive)

        needed = min_active - curr_active_count
        for i in inactive[:needed]:
            agent_active[i] = 1
            if fluctuation_period > 0:
                agent_timer[i] = int(
                    np.random.randint(
                        fluctuation_period,
                        fluctuation_period + 10,
                    )
                )

    prev_set = {
        i for i in range(num_agents)
        if agent_active_prev[i] == 1
    }
    curr_set = {
        i for i in range(num_agents)
        if agent_active[i] == 1
    }

    S_a = curr_set - prev_set
    S_d = prev_set - curr_set
    S_r = curr_set & prev_set

    return agent_active, agent_timer, S_a, S_d, S_r, curr_set


# ---------------------------------------------------------------------------
# Communication graph
# ---------------------------------------------------------------------------

def build_ring_graph_weights(
    active_list: Sequence[int],
    w_bar: float = 0.01,
    num_neighbors: int = 1,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """Build a k-nearest ring graph and communication weight matrix."""
    n_active = len(active_list)

    if n_active == 1:
        return np.array([[1.0]]), {0: []}

    k = min(num_neighbors, (n_active - 1) // 2)
    if k < 1:
        k = 1

    neighbors: Dict[int, List[int]] = {}

    for idx in range(n_active):
        nbrs = set()
        for offset in range(1, k + 1):
            nbrs.add((idx - offset) % n_active)
            nbrs.add((idx + offset) % n_active)
        neighbors[idx] = sorted(nbrs)

    W = np.zeros((n_active, n_active))

    for i in range(n_active):
        for j in neighbors[i]:
            d_i = len(neighbors[i])
            d_j = len(neighbors[j])
            W[i, j] = max(
                w_bar,
                1.0 / (0.5 + max(d_i, d_j)),
            )

        W[i, i] = 1.0 - np.sum(W[i])

    return W, neighbors


# ---------------------------------------------------------------------------
# Parameter-vector utilities
# ---------------------------------------------------------------------------

def projection_ball(x: np.ndarray, radius: float) -> np.ndarray:
    """Project x onto the Euclidean ball of the specified radius."""
    norm = np.linalg.norm(x)

    if norm > radius:
        return radius * x / norm

    return x


def make_alpha(k: int, alpha_init: float) -> float:
    """Return alpha(k) = alpha_init / sqrt(k + 1)."""
    return alpha_init / math.sqrt(k + 1)


def unflatten_parameters(
    flat_params: torch.Tensor,
    param_shapes: Sequence[torch.Size],
    param_elements: Sequence[int],
) -> List[torch.Tensor]:
    """Convert one flat parameter vector into model-shaped tensors."""
    params = []
    offset = 0

    for shape, elements in zip(param_shapes, param_elements):
        param = flat_params[offset:offset + elements].view(shape)
        params.append(param)
        offset += elements

    return params


def assign_params_to_model(
    model: nn.Module,
    params: Sequence[torch.Tensor],
) -> None:
    """Copy parameter tensors into a PyTorch model."""
    with torch.no_grad():
        for param, new_param in zip(model.parameters(), params):
            param.copy_(new_param)


# ---------------------------------------------------------------------------
# Regret evaluation
# ---------------------------------------------------------------------------

def compute_aggregate_grad_norm_sq_and_loss(
    param_shapes: Sequence[torch.Size],
    param_elements: Sequence[int],
    train_datasets: Sequence[Dataset],
    remaining_agents: Set[int],
    x_i: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Approximate ||grad f_k(x_i(k))||^2 and f_k(x_i(k)).

    One mini-batch from each remaining agent is used to approximate the
    aggregate objective and aggregate gradient.
    """
    if not remaining_agents:
        return 0.0, 0.0

    grad_sum = torch.zeros_like(x_i)
    total_loss = 0.0

    for agent_id in remaining_agents:
        model_tmp = build_model().to(device)

        params = unflatten_parameters(
            x_i.clone(),
            param_shapes,
            param_elements,
        )
        assign_params_to_model(model_tmp, params)

        dataloader = create_dataloader(
            train_datasets[agent_id],
            batch_size,
        )

        inputs, labels = next(iter(dataloader))
        inputs = inputs.to(device)
        labels = labels.to(device)

        output = model_tmp(inputs)
        loss = nn.CrossEntropyLoss()(output, labels)

        model_tmp.zero_grad()
        loss.backward()

        grad = torch.cat([
            p.grad.view(-1)
            for p in model_tmp.parameters()
            if p.grad is not None
        ])

        grad_sum += grad.detach()
        total_loss += loss.item()

    avg_grad = grad_sum / len(remaining_agents)
    avg_loss = total_loss / len(remaining_agents)

    return torch.norm(avg_grad).item() ** 2, avg_loss


def compute_episode_wise_discounted_regret(
    episodes_grad_norms: Sequence[Sequence[float]],
    gamma: float,
) -> float:
    """Compute episode-wise discounted regret."""
    total_regret = 0.0

    for episode_norms in episodes_grad_norms:
        episode_length = len(episode_norms)

        for t, norm_sq in enumerate(episode_norms):
            exponent = episode_length - 1 - t
            total_regret += gamma ** exponent * norm_sq

    return total_regret


def compute_cumulative_regret_array(
    episode_data: Sequence[Sequence[Tuple[int, float]]],
    gamma: float,
    num_iterations: int,
) -> np.ndarray:
    """Return cumulative discounted regret at each iteration."""
    contributions = np.zeros(num_iterations)

    for episode in episode_data:
        episode_length = len(episode)

        for t, (iteration, norm_sq) in enumerate(episode):
            exponent = episode_length - 1 - t
            contributions[iteration] = gamma ** exponent * norm_sq

    return np.cumsum(contributions)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def experiment(
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    num_agents: int = 20,
    num_iterations: int = 600,
    radius: float = 5000.0,
    alpha_init: float = 0.8,
    batch_size: int = 256,
    gamma: float = 0.97,
    w_bar: float = 0.01,
    fluctuation_period: int = 30,
    fluctuation_ratio: float = 0.1,
    core_count: int | None = None,
    min_active: int = 1,
    num_neighbors: int = 1,
    warm_start: bool = False,
    device: torch.device | None = None,
) -> Dict[str, object]:
    """Run the open-network distributed mini-batch SGD experiment."""
    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    train_size = len(train_dataset)
    test_size = len(test_dataset)

    train_per_agent = train_size // num_agents
    test_per_agent = test_size // num_agents

    train_sizes = [train_per_agent] * num_agents
    test_sizes = [test_per_agent] * num_agents

    train_sizes[-1] += train_size - sum(train_sizes)
    test_sizes[-1] += test_size - sum(test_sizes)

    train_agent_datasets = random_split(
        train_dataset,
        train_sizes,
    )
    test_agent_datasets = random_split(
        test_dataset,
        test_sizes,
    )

    agent_models: Dict[int, nn.Module] = {}
    x: Dict[int, torch.Tensor] = {}

    for i in range(num_agents):
        model = build_model().to(device)
        agent_models[i] = model

        x[i] = torch.cat([
            p.view(-1)
            for p in model.parameters()
        ]).detach().cpu()

    param_shapes = [
        p.shape for p in agent_models[0].parameters()
    ]
    param_elements = [
        p.numel() for p in agent_models[0].parameters()
    ]

    loss_function = nn.CrossEntropyLoss()

    agent_active = np.ones(num_agents)
    agent_timer = np.zeros(num_agents)

    agent0_episodes: List[List[Tuple[int, float]]] = []
    agent0_current_episode: List[Tuple[int, float]] = []
    agent0_was_active = True

    all_grad_norms_agent0: List[float] = []
    all_losses_agent0: List[float] = []
    num_active_agents_list: List[int] = []

    train_accuracies = {
        i: [] for i in range(num_agents)
    }
    test_accuracies = {
        i: [] for i in range(num_agents)
    }

    data_iters: Dict[int, object] = {}

    def get_minibatch(agent_id: int):
        if agent_id not in data_iters:
            data_iters[agent_id] = iter(
                create_dataloader(
                    train_agent_datasets[agent_id],
                    batch_size,
                )
            )

        try:
            return next(data_iters[agent_id])
        except StopIteration:
            data_iters[agent_id] = iter(
                create_dataloader(
                    train_agent_datasets[agent_id],
                    batch_size,
                )
            )
            return next(data_iters[agent_id])

    for k in range(num_iterations):
        alpha_k = make_alpha(k, alpha_init)

        (
            agent_active,
            agent_timer,
            arriving_agents,
            _departing_agents,
            remaining_agents,
            active_set,
        ) = manage_open_network(
            num_agents,
            agent_active,
            agent_timer,
            fluctuation_period=fluctuation_period,
            fluctuation_ratio=fluctuation_ratio,
            core_count=core_count,
            min_active=min_active,
        )

        num_active = len(active_set)
        num_active_agents_list.append(num_active)

        if num_active == 0:
            all_grad_norms_agent0.append(0.0)
            all_losses_agent0.append(0.0)

            for i in range(num_agents):
                train_accuracies[i].append(
                    train_accuracies[i][-1]
                    if train_accuracies[i]
                    else 0.0
                )
                test_accuracies[i].append(
                    test_accuracies[i][-1]
                    if test_accuracies[i]
                    else 0.0
                )
            continue

        for i in arriving_agents:
            model = build_model().to(device)
            agent_models[i] = model

            if warm_start and remaining_agents:
                donor = random.choice(list(remaining_agents))
                x[i] = x[donor].clone()
            else:
                x[i] = torch.cat([
                    p.view(-1)
                    for p in model.parameters()
                ]).detach().cpu()

        active_list = sorted(active_set)
        global_to_local = {
            global_id: local_id
            for local_id, global_id in enumerate(active_list)
        }

        W, neighbors = build_ring_graph_weights(
            active_list,
            w_bar=w_bar,
            num_neighbors=num_neighbors,
        )

        agent_gradients: Dict[int, torch.Tensor] = {}

        for i in active_list:
            model = agent_models[i]

            params_i = unflatten_parameters(
                x[i].to(device),
                param_shapes,
                param_elements,
            )
            assign_params_to_model(model, params_i)

            inputs, labels = get_minibatch(i)
            inputs = inputs.to(device)
            labels = labels.to(device)

            model.zero_grad()
            output = model(inputs)
            loss = loss_function(output, labels)
            loss.backward()

            gradient = torch.cat([
                p.grad.view(-1)
                for p in model.parameters()
                if p.grad is not None
            ])

            agent_gradients[i] = gradient.detach().cpu()

        x_new: Dict[int, torch.Tensor] = {}

        for i in active_list:
            local_i = global_to_local[i]
            x_i_np = x[i].numpy().copy()

            z_i = x_i_np.copy()

            for local_j in neighbors[local_i]:
                global_j = active_list[local_j]
                z_i += (
                    W[local_i, local_j]
                    * (x[global_j].numpy() - x_i_np)
                )

            x_next = (
                z_i
                - alpha_k * agent_gradients[i].numpy()
            )

            x_next = projection_ball(x_next, radius)
            x_new[i] = torch.from_numpy(
                x_next.astype(np.float32)
            )

        for i in active_list:
            x[i] = x_new[i]

            params_i = unflatten_parameters(
                x[i].to(device),
                param_shapes,
                param_elements,
            )
            assign_params_to_model(
                agent_models[i],
                params_i,
            )

        # The v2 implementation uses the current remaining set S_r(k)
        # as an approximation of S^r_{k+1}.
        remaining_for_regret = (
            remaining_agents
            if remaining_agents
            else active_set
        )

        if 0 in active_set:
            norm_sq, loss_val = (
                compute_aggregate_grad_norm_sq_and_loss(
                    param_shapes,
                    param_elements,
                    train_agent_datasets,
                    remaining_for_regret,
                    x[0].to(device),
                    batch_size,
                    device,
                )
            )
        else:
            norm_sq = 0.0
            loss_val = 0.0

        all_grad_norms_agent0.append(norm_sq)
        all_losses_agent0.append(loss_val)

        if 0 in active_set:
            if not agent0_was_active:
                if agent0_current_episode:
                    agent0_episodes.append(
                        agent0_current_episode
                    )
                agent0_current_episode = []

            agent0_current_episode.append(
                (k, norm_sq)
            )
            agent0_was_active = True

        else:
            if (
                agent0_was_active
                and agent0_current_episode
            ):
                agent0_episodes.append(
                    agent0_current_episode
                )
                agent0_current_episode = []

            agent0_was_active = False

        for i in range(num_agents):
            if agent_active[i] == 1:
                train_acc = evaluate_model(
                    agent_models[i],
                    train_agent_datasets[i],
                    batch_size,
                    device,
                )
                test_acc = evaluate_model(
                    agent_models[i],
                    test_agent_datasets[i],
                    batch_size,
                    device,
                )

                train_accuracies[i].append(train_acc)
                test_accuracies[i].append(test_acc)

            else:
                train_accuracies[i].append(
                    train_accuracies[i][-1]
                    if train_accuracies[i]
                    else 0.0
                )
                test_accuracies[i].append(
                    test_accuracies[i][-1]
                    if test_accuracies[i]
                    else 0.0
                )

        if (k + 1) % 20 == 0:
            print(
                f"  Iteration {k + 1}/{num_iterations}, "
                f"|N_k|={num_active}, "
                f"alpha={alpha_k:.4f}"
            )

    if agent0_current_episode:
        agent0_episodes.append(
            agent0_current_episode
        )

    mean_train_acc = np.mean(
        [
            train_accuracies[i]
            for i in range(num_agents)
        ],
        axis=0,
    )
    mean_test_acc = np.mean(
        [
            test_accuracies[i]
            for i in range(num_agents)
        ],
        axis=0,
    )

    episode_norms = [
        [norm_sq for _, norm_sq in episode]
        for episode in agent0_episodes
    ]

    regret_agent0 = compute_episode_wise_discounted_regret(
        episode_norms,
        gamma,
    )

    cumulative_regret = compute_cumulative_regret_array(
        agent0_episodes,
        gamma,
        num_iterations,
    )

    return {
        "mean_train_acc": mean_train_acc,
        "mean_test_acc": mean_test_acc,
        "grad_norms": np.asarray(all_grad_norms_agent0),
        "regret": regret_agent0,
        "num_active": np.asarray(num_active_agents_list),
        "losses": np.asarray(all_losses_agent0),
        "cumulative_regret": cumulative_regret,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_standard_comparison(
    iterations: np.ndarray,
    results: Dict[str, Dict[str, object]],
    labels: Sequence[str],
    key: str,
    ylabel: str,
    output_stem: Path,
    *,
    ylim: Tuple[float, float] | None = None,
    bottom_zero: bool = False,
    yticks: np.ndarray | None = None,
) -> None:
    """Save one standard comparison plot as PNG and EPS."""
    fig, ax = plt.subplots()

    for label in labels:
        ax.plot(
            iterations,
            results[label][key],
            label=label,
        )

    ax.set_xlim(0, len(iterations) + 1)

    if ylim is not None:
        ax.set_ylim(*ylim)
    elif bottom_zero:
        ax.set_ylim(bottom=0)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.grid(True)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(ylabel)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=150)
    fig.savefig(output_stem.with_suffix(".eps"), format="eps")
    plt.close(fig)


def save_accuracy_comparison(
    iterations: np.ndarray,
    results: Dict[str, Dict[str, object]],
    labels: Sequence[str],
    output_stem: Path,
) -> None:
    """Save train/test accuracy comparison."""
    fig, ax = plt.subplots()

    for label in labels:
        ax.plot(
            iterations,
            results[label]["mean_train_acc"],
            label=f"{label} (Train)",
        )
        ax.plot(
            iterations,
            results[label]["mean_test_acc"],
            "--",
            label=f"{label} (Test)",
        )

    ax.set_xlim(0, len(iterations) + 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Accuracy")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=150)
    fig.savefig(output_stem.with_suffix(".eps"), format="eps")
    plt.close(fig)


def save_log_loss_comparison(
    iterations: np.ndarray,
    results: Dict[str, Dict[str, object]],
    labels: Sequence[str],
    output_stem: Path,
) -> None:
    """Save f_k(x_i(k)) on a logarithmic vertical axis."""
    fig, ax = plt.subplots()

    for label in labels:
        losses = np.asarray(
            results[label]["losses"],
            dtype=float,
        )
        ax.semilogy(
            iterations,
            np.maximum(losses, LOG_FLOOR),
            label=label,
        )

    ax.set_xlim(0, len(iterations) + 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$f_k(x_i(k))$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=150)
    fig.savefig(output_stem.with_suffix(".eps"), format="eps")
    plt.close(fig)


def save_empirical_residual_comparison(
    iterations: np.ndarray,
    results: Dict[str, Dict[str, object]],
    labels: Sequence[str],
    output_stem: Path,
) -> None:
    """
    Save empirical loss residuals on a logarithmic vertical axis.

    For each run, the reference value is the minimum observed loss:
        f_min_obs = min_k f_k(x_i(k)).
    """
    fig, ax = plt.subplots()

    for label in labels:
        losses = np.asarray(
            results[label]["losses"],
            dtype=float,
        )

        f_min_obs = float(losses.min())
        residual = np.maximum(
            losses - f_min_obs,
            LOG_FLOOR,
        )

        ax.semilogy(
            iterations,
            residual,
            label=label,
        )

    ax.set_xlim(0, len(iterations) + 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(
        r"$f_k(x_i(k)) - f_{\mathrm{min,obs}}$"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=150)
    fig.savefig(output_stem.with_suffix(".eps"), format="eps")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_fluctuation_ratio_comparison(
    train_dataset: Dataset,
    test_dataset: Dataset,
    cfg: ExperimentConfig,
) -> None:
    """Compare fluctuation ratios rho = 0.1, 0.3, and 0.5."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(cfg.device)
        if cfg.device is not None
        else torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    print(
        "\n[MNIST: N=20, no core agents, warm start, "
        "dense ring graph]"
    )
    print(f"Device: {device}")

    results: Dict[str, Dict[str, object]] = {}

    for ratio, label in zip(
        FLUCTUATION_RATIOS,
        RATIO_LABELS,
    ):
        print(f"--- {label} ---")

        fix_random_seed(cfg.seed)

        results[label] = experiment(
            train_dataset,
            test_dataset,
            num_agents=cfg.num_agents,
            num_iterations=cfg.num_iterations,
            radius=cfg.radius,
            alpha_init=cfg.alpha_init,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            w_bar=cfg.w_bar,
            fluctuation_period=cfg.fluctuation_period,
            fluctuation_ratio=ratio,
            core_count=cfg.core_count,
            min_active=cfg.min_active,
            num_neighbors=cfg.num_neighbors,
            warm_start=cfg.warm_start,
            device=device,
        )

    iterations = np.arange(
        1,
        cfg.num_iterations + 1,
    )

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
    })

    save_standard_comparison(
        iterations,
        results,
        RATIO_LABELS,
        "cumulative_regret",
        r"$\mathrm{Reg}_i$",
        output_dir / "case4_regret_vs_ratio",
        bottom_zero=True,
    )

    save_standard_comparison(
        iterations,
        results,
        RATIO_LABELS,
        "losses",
        "Loss",
        output_dir / "case4_loss_vs_ratio",
        bottom_zero=True,
    )

    save_accuracy_comparison(
        iterations,
        results,
        RATIO_LABELS,
        output_dir / "case4_accuracy_vs_ratio",
    )

    save_standard_comparison(
        iterations,
        results,
        RATIO_LABELS,
        "num_active",
        r"$|N_k|$",
        output_dir / "case4_num_agents_vs_ratio",
        bottom_zero=True,
        yticks=np.arange(
            0,
            cfg.num_agents + 1,
            step=5,
        ),
    )

    save_log_loss_comparison(
        iterations,
        results,
        RATIO_LABELS,
        output_dir / "case4_loss_log_vs_ratio",
    )

    save_empirical_residual_comparison(
        iterations,
        results,
        RATIO_LABELS,
        output_dir / "case4_residual_log_vs_ratio",
    )

    print("\n" + "=" * 60)
    print("Episode-wise Discounted Regret (Agent 0)")
    print("=" * 60)
    print(f"{'Setting':30s} {'Regret':>12s}")
    print("-" * 60)

    for label in RATIO_LABELS:
        print(
            f"{label:30s} "
            f"{results[label]['regret']:12.2f}"
        )

    print("=" * 60)
    print(f"Outputs saved to: {output_dir.resolve()}")


def main() -> None:
    """Entry point."""
    cfg = ExperimentConfig()

    fix_random_seed(cfg.seed)
    train_dataset, test_dataset = load_mnist()

    run_fluctuation_ratio_comparison(
        train_dataset,
        test_dataset,
        cfg,
    )


if __name__ == "__main__":
    main()
