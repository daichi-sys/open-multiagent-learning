# Distributed Online Stochastic Learning over Open Multiagent Networks

This repository contains the Python code used for the numerical experiments
associated with a manuscript on distributed online stochastic learning over
open multiagent networks.

The repository includes two groups of experiments:

1. **Synthetic open-network experiments**
2. **MNIST experiment**

The associated manuscript is currently under review.

---

## Repository Structure

```text
ieee_cyb_open/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── synthetic_experiment/
│   └── synthetic_open_network_experiments.py
└── mnist_experiment/
    └── mnist_open_network_experiment.py
```

---

## Environment

The experiments were tested with the following software versions:

- Python 3.12.14
- NumPy 2.5.2
- Matplotlib 3.11.1
- PyTorch 2.13.0
- torchvision 0.28.0

### Using Conda

If Anaconda or Miniconda is available, create the environment from
`environment.yml`:

```bash
conda env create -f environment.yml
conda activate ieee_cyb_code
```

### Using pip

Alternatively, create a Python 3.12 environment and install the required
packages with:

```bash
python -m pip install -r requirements.txt
```

---

## 1. Synthetic Open-Network Experiments

The synthetic experiments are implemented in:

```text
synthetic_experiment/synthetic_open_network_experiments.py
```

The script contains the following experiments:

- **Experiment A:** Effect of join/leave frequency
- **Experiment B:** Effect of graph topology
- **Experiment C:** Baseline comparisons
- **Experiment D:** Parameter sensitivity to the step-size coefficient and
  discount factor
- **Experiment E:** Non-Gaussian and heavy-tailed gradient noise

### Experiment A: Join/Leave Frequency

This experiment compares different rates of agent arrivals and departures:

- Slow
- Medium
- Fast

The comparison evaluates the influence of network openness on the learning
performance.

### Experiment B: Graph Topology

This experiment compares several communication-graph structures:

- Ring
- Random sparse graph
- Random dense graph
- Complete graph
- Star graph

### Experiment C: Baseline Comparisons

The proposed distributed stochastic-gradient method is compared with the
following baseline variants:

- No-consensus SGD
- Periodic averaging
- Naive open-network dual averaging

### Experiment D: Parameter Sensitivity

This experiment studies sensitivity to:

- the step-size coefficient
  \(\eta \in \{0.01, 0.05, 0.1\}\)
- the discount factor
  \(\gamma \in \{0.90, 0.99, 0.999\}\)

### Experiment E: Gradient-Noise Models

This experiment compares the following synthetic gradient-noise models:

- Gaussian noise
- Student-\(t\) noise with finite variance
- Rare-outlier noise

The default settings of Experiments A--D use noise-free local gradients.
Experiment E explicitly adds synthetic stochastic gradient noise.

### Running the Synthetic Experiments

Run the script from the repository root:

```bash
python synthetic_experiment/synthetic_open_network_experiments.py
```

By default, Experiments A--E are executed sequentially.

Generated figures are saved under:

```text
output_open_network_v3/
```

The outputs include comparisons of quantities such as:

- cumulative discounted regret
- time-averaged regret
- number of active agents
- aggregate objective value
- aggregate cost residual

Figures are saved in both PNG and EPS formats.

### Warm-Start Initialization

When warm-start initialization is enabled, a newly arriving or reconnecting
agent is initialized from the mean of the currently available agent states,
with a small perturbation.

The implementation does not restore the reconnecting agent's own previously
stored state.

---

## 2. MNIST Experiment

The MNIST experiment is implemented in:

```text
mnist_experiment/mnist_open_network_experiment.py
```

The experiment considers distributed mini-batch stochastic gradient descent
over a dynamically changing network using a linear classifier for the MNIST
handwritten-digit dataset.

The current experiment compares the fluctuation ratios

```text
rho = 0.1, 0.3, 0.5
```

with the following main settings:

- 20 agents
- 600 iterations
- mini-batch size 256
- warm-start initialization for arriving agents
- ring-based communication graph
- episode-wise discounted regret evaluation

### Running the MNIST Experiment

Run the script from the repository root:

```bash
python mnist_experiment/mnist_open_network_experiment.py
```

The MNIST dataset is downloaded automatically by `torchvision` if it is not
already available locally.

Generated figures are saved under:

```text
output_agent0_regret_cases/
```

The outputs include:

- cumulative discounted regret
- loss
- log-scale loss
- empirical loss residual
- training and test accuracy
- number of active agents

### Empirical Loss Residual

The residual shown in the MNIST experiment uses the minimum observed loss
along each simulation run as the reference value:

```math
f_{\mathrm{min,obs}}
=
\min_k f_k(x_i(k)).
```

Accordingly, the plotted residual is

```math
f_k(x_i(k)) - f_{\mathrm{min,obs}}.
```

This is an empirical reference quantity and is not necessarily equal to the
difference from the exact time-varying optimal objective value.

---

## Reproducibility

Random seeds are fixed in the experiment scripts to improve reproducibility.

For the most faithful reproduction, use the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate ieee_cyb_code
```

Then run the experiment scripts from the repository root.

Exact numerical results can depend on the operating system, hardware,
PyTorch backend, and numerical-library implementation. The provided
environment files record the software versions used for the experiments.

---

## Data and Generated Files

The MNIST dataset is not included in this repository because it can be
downloaded automatically by `torchvision`.

Generated output directories, figures, Python cache files, and local
development files are excluded from version control through `.gitignore`.

---

## Citation

This repository contains the code associated with the following manuscript, which is currently under review:

Daichi Ishikawa, Hinano Yasuda, Naoki Hayashi and Masahiro Inuiguchi,
"Distributed Online Stochastic Learning for Open Multiagent Systems",
manuscript under review.

Please check this repository for updated citation information after publication.

Replace the placeholder above with the final bibliographic information before public release.

---

## License

A license has not yet been specified.

Before public release, the authors will select an appropriate license in
accordance with institutional, coauthor, and applicable research-policy
requirements.
