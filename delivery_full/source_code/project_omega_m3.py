"""
================================================================================
PROJECT OMEGA — D3QN RESEARCH PIPELINE
================================================================================
Author  : Aryan Singh Chandel
Module  : 3 — Neural Architecture
            · SumTree Prioritised Experience Replay (PER)
            · Dueling Double Deep Q-Network (D3QN)
            · Soft target-network update
            · Double Q-learning Bellman target
            · Ablation-flag-aware agent (dueling / use_per toggles)
Depends : project_omega_m1_m2.py  (must be exec'd / imported first)
================================================================================
"""

from __future__ import annotations

import math
import random
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── import from Module 1/2 (already exec'd in the same Colab session) ─────────
from project_omega_m1_m2 import RLConfig   # noqa: E402


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3-A  DEVICE SELECTION                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_device() -> torch.device:
    """Return the best available torch device (CUDA preferred for Colab T4).

    Returns:
        ``torch.device`` pointing at CUDA if available, else CPU.
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[Device] CUDA not found — using CPU")
    return dev


DEVICE: torch.device = get_device()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3-B  SUMTREE — O(log n) PRIORITY REPLAY BUFFER                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class SumTree:
    """Binary segment tree for O(log n) priority sampling in PER.

    Internal layout (1-indexed array of size ``2*capacity``)::

        Index 1          = root (sum of all priorities)
        Indices 2..cap   = internal nodes
        Indices cap..2*cap-1 = leaves (one per experience slot)

    The leaf at position ``i`` (0-based slot) is stored at array index
    ``capacity + i``.

    Args:
        capacity: Maximum number of transitions to store (must be a power of 2
                  for efficient tree addressing; padded automatically).
    """

    def __init__(self, capacity: int) -> None:
        # Round up to next power of two so the tree is perfectly balanced
        self.capacity: int = int(2 ** math.ceil(math.log2(max(capacity, 2))))
        self._tree: np.ndarray = np.zeros(2 * self.capacity, dtype=np.float64)
        self._data: np.ndarray = np.empty(self.capacity, dtype=object)
        self._write: int = 0        # next write pointer (circular)
        self._size:  int = 0        # current number of stored transitions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propagate(self, idx: int, delta: float) -> None:
        """Propagate a priority change up the tree from leaf to root.

        Args:
            idx:   1-based tree index of the *leaf* that changed.
            delta: Signed change in priority value.
        """
        parent = idx >> 1
        while parent >= 1:
            self._tree[parent] += delta
            parent >>= 1

    def _leaf_index(self, slot: int) -> int:
        """Convert a 0-based slot to a 1-based tree leaf index.

        Args:
            slot: 0-based circular buffer position.

        Returns:
            1-based leaf index in ``self._tree``.
        """
        return self.capacity + slot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_priority(self) -> float:
        """Sum of all stored priorities (root of the SumTree)."""
        return float(self._tree[1])

    @property
    def max_priority(self) -> float:
        """Maximum leaf priority; falls back to 1.0 when tree is empty."""
        if self._size == 0:
            return 1.0
        leaves = self._tree[self.capacity: self.capacity + self._size]
        return float(np.max(leaves))

    @property
    def min_priority(self) -> float:
        """Minimum *non-zero* leaf priority; used for IS weight normalisation."""
        if self._size == 0:
            return 1.0
        leaves = self._tree[self.capacity: self.capacity + self._size]
        nonzero = leaves[leaves > 0]
        return float(np.min(nonzero)) if len(nonzero) else 1.0

    def add(self, priority: float, data: object) -> None:
        """Insert a transition with the given priority.

        Args:
            priority: TD-error-derived priority (already raised to alpha).
            data:     The transition tuple to store.
        """
        leaf = self._leaf_index(self._write)
        delta = priority - self._tree[leaf]
        self._tree[leaf] = priority
        self._propagate(leaf, delta)
        self._data[self._write] = data
        self._write = (self._write + 1) % self.capacity
        self._size  = min(self._size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float) -> None:
        """Update the priority of an existing leaf in-place.

        Args:
            tree_idx: 1-based tree index (as returned by :meth:`sample`).
            priority: New priority value (already raised to alpha).
        """
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def sample(self, value: float) -> Tuple[int, float, object]:
        """Retrieve a transition whose cumulative priority covers ``value``.

        Uses tree descent: at each node, go left if value ≤ left-child sum,
        else subtract left-child sum and go right.

        Args:
            value: A uniformly-sampled scalar in ``[0, total_priority)``.

        Returns:
            Tuple of (tree_idx, priority, data).
        """
        idx = 1  # start at root
        while idx < self.capacity:   # while not a leaf
            left  = 2 * idx
            right = left + 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx    = right
        return idx, self._tree[idx], self._data[idx - self.capacity]

    def __len__(self) -> int:
        return self._size


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3-C  PRIORITISED EXPERIENCE REPLAY BUFFER                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Transition(NamedTuple):
    """A single stored environment transition.

    Attributes:
        state:      Normalised observation at time t.
        action:     Discrete action taken.
        reward:     Shaped scalar reward received.
        next_state: Normalised observation at time t+1.
        done:       Terminal flag (float: 1.0 if done else 0.0).
    """
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       float


class PERBuffer:
    """Prioritised Experience Replay using a SumTree backend.

    Implements the full PER algorithm from Schaul et al. (2016):
    - Priority assignment:  p_i = (|δ_i| + ε)^α
    - Sampling probability: P(i) = p_i / Σ p_k
    - IS weight:            w_i  = (1 / (N · P(i)))^β  normalised by max w

    When ``use_per=False`` the buffer degrades to uniform random sampling
    (standard experience replay), making it suitable for ablation runs.

    Args:
        cfg: Hyperparameter config carrying capacity, alpha, beta, eps, use_per.
    """

    def __init__(self, cfg: RLConfig) -> None:
        self.cfg       = cfg
        self.tree      = SumTree(cfg.buffer_capacity)
        self._beta     = cfg.per_beta_start
        self._beta_end = cfg.per_beta_end
        self._alpha    = cfg.per_alpha
        self._eps      = cfg.per_eps
        self._use_per  = cfg.use_per

    # ------------------------------------------------------------------
    def _priority(self, td_error: float) -> float:
        """Compute α-scaled priority from a raw TD error.

        Args:
            td_error: Absolute Bellman residual ``|δ|``.

        Returns:
            Priority value ``(|δ| + ε)^α``.
        """
        return (abs(td_error) + self._eps) ** self._alpha

    # ------------------------------------------------------------------
    def push(self, transition: Transition, td_error: float = 1.0) -> None:
        """Store a transition; new transitions get max priority by default.

        Args:
            transition: The :class:`Transition` namedtuple to store.
            td_error:   Optional initial TD error; uses max priority if omitted.
        """
        if self._use_per:
            priority = max(self._priority(td_error), self.tree.max_priority)
        else:
            priority = 1.0          # uniform
        self.tree.add(priority, transition)

    # ------------------------------------------------------------------
    def sample(
        self, batch_size: int, global_step: int, total_steps: int
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample a batch using prioritised or uniform sampling.

        Beta is linearly annealed from ``per_beta_start`` → ``per_beta_end``
        over ``total_steps`` to reduce the IS-weight correction bias.

        Args:
            batch_size:  Number of transitions to sample.
            global_step: Current training step (for beta annealing).
            total_steps: Total expected training steps (for beta annealing).

        Returns:
            Tuple of:
            - ``transitions``: List of :class:`Transition` namedtuples.
            - ``tree_indices``: 1-based SumTree indices (needed for updates).
            - ``is_weights``:  Normalised importance-sampling weights, shape (B,).
        """
        # Anneal beta linearly
        frac        = min(global_step / max(total_steps, 1), 1.0)
        self._beta  = self.cfg.per_beta_start + frac * (
            self.cfg.per_beta_end - self.cfg.per_beta_start
        )

        transitions:  List[Transition] = []
        tree_indices: List[int]        = []
        priorities:   List[float]      = []

        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            if self._use_per:
                lo  = segment * i
                hi  = segment * (i + 1)
                val = random.uniform(lo, hi)
                idx, prio, data = self.tree.sample(val)
            else:
                # Uniform fallback: pick a random occupied leaf
                slot = random.randint(0, len(self.tree) - 1)
                idx  = self.tree.capacity + slot
                prio = 1.0
                data = self.tree._data[slot]

            if data is None:        # guard against uninitialised leaves
                continue
            transitions.append(data)
            tree_indices.append(idx)
            priorities.append(max(prio, 1e-12))

        # IS weights: w_i = (1 / (N * P(i)))^beta, normalised by max w
        n    = len(self.tree)
        if self._use_per:
            total = self.tree.total_priority
            probs = np.array(priorities, dtype=np.float64) / total
        else:
            probs = np.ones(len(priorities), dtype=np.float64) / n

        raw_weights = (n * probs) ** (-self._beta)
        is_weights  = (raw_weights / raw_weights.max()).astype(np.float32)

        return transitions, np.array(tree_indices), is_weights

    # ------------------------------------------------------------------
    def update_priorities(
        self, tree_indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        """Recompute and store priorities after a learning update.

        Args:
            tree_indices: 1-based tree indices returned from :meth:`sample`.
            td_errors:    Per-sample absolute TD errors from the learning step.
        """
        if not self._use_per:
            return
        for idx, err in zip(tree_indices, td_errors):
            self.tree.update(int(idx), self._priority(float(err)))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tree)

    @property
    def beta(self) -> float:
        """Current IS-weight exponent beta (read-only)."""
        return self._beta


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3-D  DUELING NETWORK ARCHITECTURE                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class NoisyLinear(nn.Module):
    """Factorised NoisyNet linear layer (optional; used as a drop-in for Linear).

    Adds learnable Gaussian noise to weights and biases for implicit exploration.
    Factorised parametrisation from Fortunato et al. (2017): only
    ``p + q`` noise scalars are stored instead of ``p*q``.

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        std_init:     Initial standard deviation of noise parameters (σ₀).
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        std_init:     float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init

        # Learnable parameters
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not parameters)
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.register_buffer("bias_eps",   torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    # ------------------------------------------------------------------
    def reset_parameters(self) -> None:
        """Initialise mu with uniform bounds and sigma with std_init/√fan_in."""
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    # ------------------------------------------------------------------
    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        """Factorised noise function: sgn(x) · √|x|."""
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Sample fresh factorised noise into the buffers."""
        eps_i = self._f(torch.randn(self.in_features))
        eps_j = self._f(torch.randn(self.out_features))
        self.weight_eps.copy_(eps_j.outer(eps_i))
        self.bias_eps.copy_(eps_j)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with (train) or without (eval) noise injection.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)


# ──────────────────────────────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    """Dueling Double Deep Q-Network.

    Architecture::

        Input (state_dim)
            │
        [Shared trunk: Linear → LayerNorm → SiLU] × len(hidden_dims)
            │
          ┌─┴─────────────────┐
          │                   │
      Value stream         Advantage stream
      Linear → SiLU        Linear → SiLU
      Linear → V(s)        Linear → A(s,a)  [action_dim]
          │                   │
          └──── Q = V + (A − mean(A)) ────┘
                │
            Q-values  (action_dim,)

    When ``dueling=False`` the value/advantage split is disabled and the
    network degrades to a standard MLP Q-network (ablation mode).

    Args:
        cfg:    Hyperparameter config (uses state_dim, action_dim, hidden_dims,
                dueling).
        noisy:  If True, replace output-layer Linear with NoisyLinear.
    """

    def __init__(self, cfg: RLConfig, noisy: bool = False) -> None:
        super().__init__()
        self.cfg    = cfg
        self.noisy  = noisy
        self.dueling = cfg.dueling

        dims = [cfg.state_dim] + list(cfg.hidden_dims)

        # ── Shared trunk ──────────────────────────────────────────────────────
        trunk_layers: List[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            trunk_layers.extend([
                nn.Linear(in_d, out_d, bias=True),
                nn.LayerNorm(out_d),
                nn.SiLU(),
            ])
        self.trunk = nn.Sequential(*trunk_layers)

        feat_dim = dims[-1]   # output dim of trunk

        if self.dueling:
            # ── Value stream ──────────────────────────────────────────────────
            self.value_hidden = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.SiLU(),
            )
            self.value_out = (
                NoisyLinear(feat_dim // 2, 1) if noisy
                else nn.Linear(feat_dim // 2, 1)
            )

            # ── Advantage stream ──────────────────────────────────────────────
            self.adv_hidden = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.SiLU(),
            )
            self.adv_out = (
                NoisyLinear(feat_dim // 2, cfg.action_dim) if noisy
                else nn.Linear(feat_dim // 2, cfg.action_dim)
            )
        else:
            # Standard MLP Q-head (ablation: no dueling)
            self.q_head = nn.Linear(feat_dim, cfg.action_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Orthogonal initialisation for all Linear layers; NoisyLinear handled internally."""
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, NoisyLinear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Args:
            x: State tensor of shape ``(B, state_dim)`` or ``(state_dim,)``.

        Returns:
            Q-value tensor of shape ``(B, action_dim)``.
        """
        feat = self.trunk(x)

        if self.dueling:
            # V(s):   (B, 1)
            v = self.value_out(self.value_hidden(feat))
            # A(s,a): (B, action_dim)
            a = self.adv_out(self.adv_hidden(feat))
            # Aggregation: Q = V + (A − mean_a A)   [Wang et al. 2016, Eq. 9]
            q = v + (a - a.mean(dim=-1, keepdim=True))
        else:
            q = self.q_head(feat)

        return q

    # ------------------------------------------------------------------
    def get_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Return trunk (penultimate) activations for t-SNE analysis.

        Args:
            x: State batch of shape ``(B, state_dim)``.

        Returns:
            Activation tensor of shape ``(B, last_hidden_dim)``.
        """
        return self.trunk(x)

    # ------------------------------------------------------------------
    def reset_noise(self) -> None:
        """Re-sample noise for all NoisyLinear layers (call before each step)."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3-E  D3QN AGENT — DOUBLE Q-LEARNING + SOFT TARGET UPDATE                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class D3QNAgent:
    """Dueling Double DQN agent with Prioritised Experience Replay.

    Implements:
    - **Double Q-learning**: online network selects action, target network
      evaluates it, eliminating maximisation bias.
    - **Dueling streams**: separate value and advantage heads.
    - **PER sampling**: TD-error-prioritised replay with IS-weight correction.
    - **Soft target update**: θ_target ← τ·θ_online + (1−τ)·θ_target.
    - **Gradient clipping**: prevents exploding gradients in early training.
    - **LR scheduling**: multiplicative decay with a minimum floor.

    When ``cfg.dueling=False`` and ``cfg.use_per=False`` the agent behaves as
    a vanilla DQN (for ablation comparisons).

    Args:
        cfg:    Hyperparameter config.
        device: torch.device to place networks and tensors on.
    """

    def __init__(self, cfg: RLConfig, device: torch.device) -> None:
        self.cfg    = cfg
        self.device = device

        # Online and target networks
        self.online = DuelingDQN(cfg).to(device)
        self.target = DuelingDQN(cfg).to(device)
        self._hard_copy()   # initialise target = online

        self.optimizer = optim.Adam(
            self.online.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            eps=1e-5,
        )
        self.scheduler = optim.lr_scheduler.MultiplicativeLR(
            self.optimizer,
            lr_lambda=lambda _ep: max(
                cfg.lr_decay,
                cfg.lr_min / (cfg.lr + 1e-12)
            ),
        )

        self.buffer  = PERBuffer(cfg)
        self.epsilon = cfg.eps_start

        # Step counters
        self._learn_step:  int = 0
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _hard_copy(self) -> None:
        """Copy online weights to target exactly (used at init)."""
        self.target.load_state_dict(self.online.state_dict())

    def _soft_update(self) -> None:
        """Polyak-average target network toward online network.

        θ_target ← τ·θ_online + (1−τ)·θ_target
        """
        tau = self.cfg.tau
        for t_param, o_param in zip(
            self.target.parameters(), self.online.parameters()
        ):
            t_param.data.copy_(tau * o_param.data + (1.0 - tau) * t_param.data)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Select an action using ε-greedy policy (or purely greedy).

        Args:
            state:  Normalised observation array of shape ``(state_dim,)``.
            greedy: If True, always pick the argmax action (evaluation mode).

        Returns:
            Discrete action index in ``[0, action_dim)``.
        """
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.cfg.action_dim)

        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.online.eval()
        with torch.no_grad():
            q_values = self.online(state_t)
        self.online.train()
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store(self, transition: Transition, td_error: float = 1.0) -> None:
        """Push one transition into the replay buffer.

        Args:
            transition: Completed :class:`Transition` namedtuple.
            td_error:   Optional initial priority hint.
        """
        self.buffer.push(transition, td_error)
        self._global_step += 1

    # ------------------------------------------------------------------
    def learn(self) -> Dict[str, float]:
        """Draw a batch and perform one gradient update.

        Returns a metrics dict with keys:
        ``loss``, ``mean_q``, ``max_q``, ``mean_td_error``, ``beta``.

        Returns:
            Dict of scalar training metrics for this update step.

        Raises:
            RuntimeError: If the buffer has fewer samples than ``warmup_steps``.
        """
        cfg  = self.cfg
        B    = cfg.batch_size
        S    = cfg.state_dim

        # ── Sample ───────────────────────────────────────────────────────────
        transitions, tree_idx, is_weights = self.buffer.sample(
            B,
            global_step=self._global_step,
            total_steps=cfg.max_episodes * cfg.max_steps_per_ep,
        )
        if len(transitions) < B:
            return {}

        # ── Batch construction ────────────────────────────────────────────────
        states      = torch.tensor(
            np.stack([t.state      for t in transitions]), dtype=torch.float32
        ).to(self.device)
        next_states = torch.tensor(
            np.stack([t.next_state for t in transitions]), dtype=torch.float32
        ).to(self.device)
        actions     = torch.tensor(
            [t.action for t in transitions], dtype=torch.long
        ).to(self.device)
        rewards     = torch.tensor(
            [t.reward for t in transitions], dtype=torch.float32
        ).to(self.device)
        dones       = torch.tensor(
            [t.done   for t in transitions], dtype=torch.float32
        ).to(self.device)
        is_w        = torch.tensor(is_weights, dtype=torch.float32).to(self.device)

        # ── Double Q-learning Bellman target ──────────────────────────────────
        #   1. Online selects greedy action in s'
        #   2. Target evaluates Q(s', a*)
        #   → prevents overestimation bias (Van Hasselt et al. 2016)
        with torch.no_grad():
            # Online network → best action for s'
            online_q_next   = self.online(next_states)                  # (B, A)
            best_actions    = online_q_next.argmax(dim=1, keepdim=True) # (B, 1)
            # Target network → Q-value of that action
            target_q_next   = self.target(next_states)                  # (B, A)
            target_q_chosen = target_q_next.gather(1, best_actions).squeeze(1)  # (B,)
            # Bellman equation
            td_targets = rewards + cfg.gamma * target_q_chosen * (1.0 - dones)

        # ── Current Q-values for taken actions ────────────────────────────────
        current_q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── TD errors & IS-weighted Huber loss ───────────────────────────────
        td_errors  = (current_q - td_targets).detach().abs().cpu().numpy()
        elementwise_loss = F.smooth_l1_loss(current_q, td_targets, reduction="none")
        loss = (is_w * elementwise_loss).mean()

        # ── Optimiser step ────────────────────────────────────────────────────
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
        self.optimizer.step()

        # ── Update priorities in SumTree ──────────────────────────────────────
        self.buffer.update_priorities(tree_idx, td_errors)

        # ── Soft target update ────────────────────────────────────────────────
        self._soft_update()
        self._learn_step += 1

        # ── Metrics ──────────────────────────────────────────────────────────
        with torch.no_grad():
            all_q = self.online(states)

        return {
            "loss":          float(loss.item()),
            "mean_q":        float(all_q.mean().item()),
            "max_q":         float(all_q.max().item()),
            "mean_td_error": float(td_errors.mean()),
            "beta":          self.buffer.beta,
            "lr":            self.optimizer.param_groups[0]["lr"],
        }

    # ------------------------------------------------------------------
    def decay_epsilon(self) -> None:
        """Apply multiplicative epsilon decay (call once per episode)."""
        self.epsilon = max(
            self.cfg.eps_end,
            self.epsilon * self.cfg.eps_decay,
        )

    def step_scheduler(self) -> None:
        """Step the LR scheduler (call once per episode)."""
        if self.optimizer.param_groups[0]["lr"] > self.cfg.lr_min:
            self.scheduler.step()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Serialise agent state to disk.

        Saves online/target weights, optimiser state, epsilon, and step counts.

        Args:
            path: Full file path for the ``.pth`` checkpoint.
        """
        torch.save({
            "online_state_dict":  self.online.state_dict(),
            "target_state_dict":  self.target.state_dict(),
            "optimizer_state":    self.optimizer.state_dict(),
            "epsilon":            self.epsilon,
            "learn_step":         self._learn_step,
            "global_step":        self._global_step,
            "cfg":                self.cfg.to_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Restore agent state from a checkpoint file.

        Args:
            path: Path to a ``.pth`` checkpoint written by :meth:`save_checkpoint`.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online_state_dict"])
        self.target.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epsilon      = ckpt.get("epsilon",     self.cfg.eps_end)
        self._learn_step  = ckpt.get("learn_step",  0)
        self._global_step = ckpt.get("global_step", 0)

    # ------------------------------------------------------------------
    @property
    def learn_step(self) -> int:
        """Total number of gradient updates performed so far."""
        return self._learn_step

    @property
    def global_step(self) -> int:
        """Total environment steps taken so far."""
        return self._global_step


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SELF-TEST                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 72)
    print("PROJECT OMEGA — Module 3 Self-Test")
    print("=" * 72)

    cfg = RLConfig()
    dev = get_device()

    # ── SumTree ──────────────────────────────────────────────────────────────
    tree = SumTree(capacity=8)
    for i in range(8):
        tree.add(float(i + 1), f"data_{i}")
    assert len(tree) == 8
    assert abs(tree.total_priority - sum(range(1, 9))) < 1e-6
    idx, prio, dat = tree.sample(0.5)
    assert dat is not None
    print(f"[SumTree] total={tree.total_priority:.1f}  sample → idx={idx} prio={prio:.2f}")

    # ── PERBuffer ─────────────────────────────────────────────────────────────
    buf = PERBuffer(cfg)
    dummy_obs = np.zeros(cfg.state_dim, dtype=np.float32)
    for _ in range(cfg.warmup_steps + 1):
        t = Transition(dummy_obs, 0, 0.0, dummy_obs, 0.0)
        buf.push(t, td_error=abs(np.random.randn()))
    transitions, tidx, isw = buf.sample(cfg.batch_size, 100, 10_000)
    assert len(transitions) == cfg.batch_size
    assert isw.min() > 0.0 and isw.max() <= 1.0 + 1e-6
    print(f"[PERBuffer] size={len(buf)}  batch={len(transitions)}  IS weight range=[{isw.min():.3f}, {isw.max():.3f}]")

    # ── DuelingDQN — forward pass ─────────────────────────────────────────────
    net = DuelingDQN(cfg).to(dev)
    x   = torch.randn(32, cfg.state_dim, device=dev)
    q   = net(x)
    assert q.shape == (32, cfg.action_dim)
    acts = net.get_activations(x)
    assert acts.shape[0] == 32
    print(f"[DuelingDQN] output shape: {tuple(q.shape)}  activations: {tuple(acts.shape)}")

    # ── Ablation: no-dueling mode ─────────────────────────────────────────────
    cfg_noduel     = RLConfig(); cfg_noduel.dueling = False
    net_noduel     = DuelingDQN(cfg_noduel).to(dev)
    q_noduel       = net_noduel(x)
    assert q_noduel.shape == (32, cfg.action_dim)
    print(f"[DuelingDQN ablation] no-dueling output shape: {tuple(q_noduel.shape)}")

    # ── D3QNAgent — act + store + learn ──────────────────────────────────────
    agent = D3QNAgent(cfg, dev)
    state = np.zeros(cfg.state_dim, dtype=np.float32)
    for _ in range(cfg.warmup_steps + cfg.batch_size):
        action = agent.act(state)
        t      = Transition(state, action, 0.1, state, 0.0)
        agent.store(t)
    metrics = agent.learn()
    assert "loss" in metrics and metrics["loss"] >= 0.0
    print(f"[D3QNAgent] learn step OK  loss={metrics['loss']:.4f}  mean_q={metrics['mean_q']:.4f}")

    # ── Checkpoint round-trip ─────────────────────────────────────────────────
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp = f.name
    agent.save_checkpoint(tmp)
    agent2 = D3QNAgent(cfg, dev)
    agent2.load_checkpoint(tmp)
    os.unlink(tmp)
    assert agent2.epsilon == agent.epsilon
    print(f"[Checkpoint] save/load round-trip OK  epsilon={agent2.epsilon:.4f}")

    # ── NoisyLinear ──────────────────────────────────────────────────────────
    nl = NoisyLinear(128, 64)
    nl.train()
    out_train = nl(torch.randn(8, 128))
    nl.eval()
    out_eval  = nl(torch.randn(8, 128))
    assert out_train.shape == out_eval.shape == (8, 64)
    print(f"[NoisyLinear] train/eval shapes OK: {tuple(out_train.shape)}")

    print("=" * 72)
    print("Module 3 PASSED. Ready for Training Loop.")
    print("=" * 72)
