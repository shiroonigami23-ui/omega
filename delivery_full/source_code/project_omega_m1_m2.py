"""
================================================================================
PROJECT OMEGA — D3QN RESEARCH PIPELINE
================================================================================
Author  : Aryan Singh Chandel
Style   : NeurIPS / ICML
Target  : Google Colab (T4 GPU), overnight training
Modules : 1 — Setup, Config, Logging, Google Drive
          2 — Environment Wrapper (LunarLander-v2), Reward Shaping, Normalization
================================================================================

COLAB SETUP CELL (run this first in a separate cell):
    !pip install -q gymnasium[box2d] fpdf2 reportlab python-pptx wandb imageio[ffmpeg] moviepy

Then in the next cell:
    from google.colab import drive
    drive.mount('/content/drive')
    # (or let the pipeline mount it automatically via MODULE 1)

Then run: exec(open("project_omega_m1_m2.py").read())
================================================================================
"""

from __future__ import annotations

# ── stdlib ──────────────────────────────────────────────────────────────────
import hashlib
import logging
import math
import os
import random
import shutil
import time
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── optional experiment tracking (imported lazily in training loop) ──────────
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 1 — ENTERPRISE SETUP, CONFIGURATION, & LOGGING                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# 1-A  Directory scaffold
# ─────────────────────────────────────────────────────────────────────────────

_COLAB_DRIVE_ROOT: Path = Path("/content/drive/MyDrive")
_PROJECT_NAME: str = "Project_Omega_D3QN"


def build_directory_tree(base: Path) -> Dict[str, Path]:
    """Create the full project directory tree and return a path registry.

    Args:
        base: Root directory under which the entire tree is created.

    Returns:
        A dict mapping logical name → resolved Path for every directory.
    """
    dirs: Dict[str, Path] = {
        "root":                  base,
        "config":                base / "config",
        "logs":                  base / "logs",
        "models":                base / "models",
        "checkpoints":           base / "models" / "checkpoints",
        "plots":                 base / "plots",
        "plots_distributions":   base / "plots" / "distributions",
        "plots_performance":     base / "plots" / "performance",
        "plots_ablation":        base / "plots" / "ablation",
        "reports":               base / "reports",
        "video":                 base / "video",
        "output":                base / "output",
        "tensorboard":           base / "logs" / "tensorboard",
    }
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def mount_google_drive() -> bool:
    """Attempt to mount Google Drive inside Colab.

    Returns:
        True if mounted (or already mounted), False otherwise.
    """
    try:
        from google.colab import drive  # type: ignore
        if not _COLAB_DRIVE_ROOT.exists():
            drive.mount("/content/drive")
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1-B  Hyperparameter dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RLConfig:
    """Complete hyperparameter registry for the D3QN pipeline.

    All values are chosen for overnight Colab T4 training on LunarLander-v2.
    """
    # ── Environment ──────────────────────────────────────────────────────────
    env_id: str             = "LunarLander-v2"
    seed: int               = 42
    max_episodes: int       = 2000
    max_steps_per_ep: int   = 1000
    eval_episodes: int      = 100
    solve_threshold: float  = 200.0
    early_stop_window: int  = 50       # consecutive episodes above threshold

    # ── Network architecture ──────────────────────────────────────────────────
    hidden_dims: Tuple[int, ...]  = (512, 512, 256)
    dueling: bool                 = True   # Dueling streams on/off
    state_dim: int                = 8      # LunarLander-v2 observation dim
    action_dim: int               = 4      # LunarLander-v2 discrete actions

    # ── Optimiser ────────────────────────────────────────────────────────────
    lr: float               = 3e-4
    lr_decay: float         = 0.9995    # multiplicative decay per episode
    lr_min: float           = 1e-5
    grad_clip: float        = 10.0
    weight_decay: float     = 1e-5

    # ── RL core ──────────────────────────────────────────────────────────────
    gamma: float            = 0.99
    tau: float              = 5e-3      # soft target-network update coefficient
    target_update_every: int = 4        # hard-update fallback (not used w/ soft)

    # ── Exploration (ε-greedy) ────────────────────────────────────────────────
    eps_start: float        = 1.0
    eps_end: float          = 0.01
    eps_decay: float        = 0.9975    # multiplicative per episode

    # ── Replay buffer / PER ──────────────────────────────────────────────────
    buffer_capacity: int    = 200_000
    batch_size: int         = 128
    warmup_steps: int       = 10_000    # steps before first learning update
    per_alpha: float        = 0.6       # priority exponent
    per_beta_start: float   = 0.4       # IS-weight exponent (annealed → 1.0)
    per_beta_end: float     = 1.0
    per_eps: float          = 1e-6      # small constant to avoid zero priority
    use_per: bool           = True      # PER on/off (ablation flag)

    # ── Reward shaping ───────────────────────────────────────────────────────
    fuel_penalty: float     = 0.05      # penalty per thruster firing
    hover_penalty: float    = 0.01      # penalty per step not on ground
    shape_rewards: bool     = True

    # ── Checkpointing & logging ───────────────────────────────────────────────
    checkpoint_every: int   = 100       # episodes
    log_every: int          = 10        # episodes
    rolling_window: int     = 100       # window for moving average

    # ── Normalisation running stats (updated online) ─────────────────────────
    obs_norm_clip: float    = 10.0      # clip normalised observations

    # ── Experiment tracking ───────────────────────────────────────────────────
    use_wandb: bool         = True
    use_tensorboard: bool   = True
    wandb_project: str      = "Project-Omega-D3QN"
    wandb_entity: str       = ""        # set your W&B username here

    # ── Author metadata ───────────────────────────────────────────────────────
    author: str             = "Aryan Singh Chandel"
    paper_style: str        = "NeurIPS/ICML"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise config to a plain dict (W&B / JSON-friendly)."""
        d = asdict(self)
        d["hidden_dims"] = list(d["hidden_dims"])
        return d


# ─────────────────────────────────────────────────────────────────────────────
# 1-C  Logging setup
# ─────────────────────────────────────────────────────────────────────────────

class ColorFormatter(logging.Formatter):
    """ANSI-coloured console formatter for rich Colab output."""

    _COLOURS: Dict[int, str] = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[32m",   # green
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[35m",   # magenta
    }
    _RESET: str = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        colour = self._COLOURS.get(record.levelno, self._RESET)
        record.levelname = f"{colour}{record.levelname:8s}{self._RESET}"
        return super().format(record)


def build_logger(log_dir: Path, run_id: str) -> logging.Logger:
    """Construct a dual-handler (file + console) logger.

    Args:
        log_dir: Directory where ``training_run.log`` is written.
        run_id:  Unique identifier appended to the logger name.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(f"omega.{run_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — plain text, DEBUG level
    fh = logging.FileHandler(log_dir / "training_run.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Console handler — coloured, INFO level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColorFormatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# 1-D  Experiment tracking facade
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentTracker:
    """Thin wrapper that writes to W&B and TensorBoard simultaneously.

    Either backend degrades gracefully if unavailable or disabled.

    Args:
        cfg:     Hyperparameter config.
        paths:   Directory path registry from :func:`build_directory_tree`.
        run_id:  Unique run identifier string.
    """

    def __init__(self, cfg: RLConfig, paths: Dict[str, Path], run_id: str) -> None:
        self._cfg   = cfg
        self._wandb = None
        self._tb    = None

        if cfg.use_wandb and _WANDB_AVAILABLE:
            try:
                self._wandb = wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity or None,
                    name=run_id,
                    config=cfg.to_dict(),
                    reinit=True,
                )
            except Exception:
                pass

        if cfg.use_tensorboard and _TB_AVAILABLE:
            try:
                self._tb = _TBWriter(log_dir=str(paths["tensorboard"]))
            except Exception:
                pass

    # ------------------------------------------------------------------
    def log(self, metrics: Dict[str, float], step: int) -> None:
        """Log a metrics dict to all active backends.

        Args:
            metrics: Key-value pairs to log.
            step:    Global training step / episode index.
        """
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:
                pass
        if self._tb is not None:
            for k, v in metrics.items():
                try:
                    self._tb.add_scalar(k, v, step)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def finish(self) -> None:
        """Flush and close all tracking backends."""
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
        if self._tb is not None:
            try:
                self._tb.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# 1-E  Pipeline bootstrap
# ─────────────────────────────────────────────────────────────────────────────

class PipelineSetup:
    """Bootstrap the entire Project Omega run environment.

    Orchestrates Drive mounting, directory creation, seeding, logger
    construction, and tracker initialisation.

    Args:
        cfg:          Hyperparameter config.
        project_base: Root directory for outputs (defaults to Drive if mounted,
                      else ``/content/Project_Omega_D3QN``).

    Attributes:
        paths:   Directory registry dict.
        logger:  Configured logger.
        tracker: :class:`ExperimentTracker` instance.
        run_id:  Unique run identifier string.
        cfg:     Resolved config reference.
    """

    def __init__(
        self,
        cfg: RLConfig,
        project_base: Optional[Path] = None,
    ) -> None:
        self.cfg    = cfg
        self.run_id = f"omega_{time.strftime('%Y%m%d_%H%M%S')}"

        # ── Google Drive ─────────────────────────────────────────────────────
        drive_ok = mount_google_drive()
        if project_base is None:
            if drive_ok and _COLAB_DRIVE_ROOT.exists():
                project_base = _COLAB_DRIVE_ROOT / _PROJECT_NAME
            else:
                project_base = Path("/content") / _PROJECT_NAME

        self.paths  = build_directory_tree(project_base)
        self.logger = build_logger(self.paths["logs"], self.run_id)

        if drive_ok:
            self.logger.info("Google Drive mounted → %s", project_base)
        else:
            self.logger.warning("Google Drive not mounted → using %s", project_base)

        # ── Reproducibility ──────────────────────────────────────────────────
        self._seed_everything(cfg.seed)
        self.logger.info("Global seed set to %d", cfg.seed)

        # ── Experiment trackers ──────────────────────────────────────────────
        self.tracker = ExperimentTracker(cfg, self.paths, self.run_id)
        if _WANDB_AVAILABLE and cfg.use_wandb:
            self.logger.info("W&B tracking active → project: %s", cfg.wandb_project)
        if _TB_AVAILABLE and cfg.use_tensorboard:
            self.logger.info("TensorBoard active → %s", self.paths["tensorboard"])

        self.logger.info("Pipeline bootstrap complete | run_id=%s", self.run_id)

    # ------------------------------------------------------------------
    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Seed Python, NumPy, and PyTorch (if available).

        Args:
            seed: Integer seed value.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark     = False
        except ImportError:
            pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 2 — ENVIRONMENT WRAPPER, REWARD SHAPING, NORMALISATION               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# 2-A  Running statistics for online observation normalisation
# ─────────────────────────────────────────────────────────────────────────────

class RunningNormalizer:
    """Welford's online algorithm for streaming mean/variance normalisation.

    Maintains per-feature running mean and variance and applies
    standardisation with optional symmetric clipping.

    Args:
        shape: Shape of a single observation vector.
        clip:  Symmetric clip value applied after normalisation.
        eps:   Small constant added to std to prevent division by zero.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        clip: float = 10.0,
        eps: float  = 1e-8,
    ) -> None:
        self.shape = shape
        self.clip  = clip
        self.eps   = eps
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = 0

    # ------------------------------------------------------------------
    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a new observation.

        Args:
            x: Single observation array of shape ``self.shape``.
        """
        self.count += 1
        delta       = x - self.mean
        self.mean  += delta / self.count
        self.var   += delta * (x - self.mean)

    # ------------------------------------------------------------------
    @property
    def std(self) -> np.ndarray:
        """Per-feature standard deviation (minimum ``self.eps``)."""
        variance = self.var / max(self.count - 1, 1)
        return np.sqrt(np.maximum(variance, self.eps ** 2))

    # ------------------------------------------------------------------
    def normalise(self, x: np.ndarray) -> np.ndarray:
        """Normalise a single observation using running statistics.

        Args:
            x: Raw observation.

        Returns:
            Standardised (and clipped) observation as float32.
        """
        normed = (x - self.mean) / self.std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """Serialise normaliser state for checkpointing."""
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        """Restore normaliser state from a checkpoint dict."""
        self.mean  = d["mean"].copy()
        self.var   = d["var"].copy()
        self.count = d["count"]


# ─────────────────────────────────────────────────────────────────────────────
# 2-B  Frame buffer for GIF rendering
# ─────────────────────────────────────────────────────────────────────────────

class FrameBuffer:
    """Accumulate RGB frames for GIF export.

    Args:
        max_frames: Safety cap on stored frames to bound memory usage.
    """

    def __init__(self, max_frames: int = 2000) -> None:
        self._frames: List[np.ndarray] = []
        self._max    = max_frames

    def capture(self, frame: Optional[np.ndarray]) -> None:
        """Append a frame if the buffer is not full.

        Args:
            frame: RGB array (H, W, 3) or None (skipped silently).
        """
        if frame is not None and len(self._frames) < self._max:
            self._frames.append(frame.astype(np.uint8))

    def frames(self) -> List[np.ndarray]:
        """Return accumulated frames."""
        return self._frames

    def clear(self) -> None:
        """Flush all stored frames."""
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


# ─────────────────────────────────────────────────────────────────────────────
# 2-C  LunarLander wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LunarLanderWrapper(gym.Wrapper):
    """Custom gymnasium.Wrapper for LunarLander-v2.

    Applies three transformations on top of the base environment:

    1. **Online state normalisation** — Welford running mean/std per feature,
       clipped to ``cfg.obs_norm_clip``.
    2. **Reward shaping** — penalises fuel consumption (non-zero action proxy)
       and hover time (not-landed penalty per step).
    3. **Frame capture** — stores ``render_mode="rgb_array"`` frames into a
       :class:`FrameBuffer` for GIF export during evaluation episodes.

    Args:
        cfg:            Hyperparameter config.
        render_mode:    Passed to :func:`gymnasium.make`.
        record_frames:  If True, capture frames into the internal buffer.
    """

    # LunarLander action semantics (discrete):
    #   0 = do nothing, 1 = fire left engine,
    #   2 = fire main engine, 3 = fire right engine
    _FUEL_ACTIONS: frozenset = frozenset({1, 2, 3})

    def __init__(
        self,
        cfg:           RLConfig,
        render_mode:   str  = "rgb_array",
        record_frames: bool = False,
    ) -> None:
        env = gym.make(cfg.env_id, render_mode=render_mode)
        super().__init__(env)
        self.cfg            = cfg
        self.record_frames  = record_frames
        self.normaliser     = RunningNormalizer(
            shape=(cfg.state_dim,),
            clip=cfg.obs_norm_clip,
        )
        self.frame_buffer   = FrameBuffer()
        self._episode_steps = 0
        self._landed        = False

    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed:    Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return a normalised initial observation.

        Args:
            seed:    Optional RNG seed forwarded to the base env.
            options: Optional reset options dict.

        Returns:
            Tuple of (normalised_obs, info).
        """
        obs, info           = self.env.reset(seed=seed, options=options)
        self._episode_steps = 0
        self._landed        = False
        if self.record_frames:
            self.frame_buffer.clear()
            self._maybe_capture()
        self.normaliser.update(obs)
        return self.normaliser.normalise(obs), info

    # ------------------------------------------------------------------
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment, apply reward shaping, and normalise the state.

        Args:
            action: Discrete action index (0-3).

        Returns:
            Tuple of (normalised_obs, shaped_reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_steps += 1

        shaped_reward = float(reward)

        if self.cfg.shape_rewards:
            # Fuel penalty: penalise every non-noop action
            if action in self._FUEL_ACTIONS:
                shaped_reward -= self.cfg.fuel_penalty

            # Hover penalty: encourage fast landing
            if not terminated:
                shaped_reward -= self.cfg.hover_penalty

            # Detect landing from base reward signal (>= 100 reward spikes)
            if terminated and reward >= 100:
                self._landed = True

        info["raw_reward"]    = float(reward)
        info["shaped_reward"] = shaped_reward
        info["landed"]        = self._landed
        info["episode_steps"] = self._episode_steps

        self.normaliser.update(obs)
        normed_obs = self.normaliser.normalise(obs)

        if self.record_frames:
            self._maybe_capture()

        return normed_obs, shaped_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _maybe_capture(self) -> None:
        """Render a frame and append it to the buffer (no-op on failure)."""
        if not self.record_frames:
            return
        try:
            frame = self.env.render()
            self.frame_buffer.capture(frame)
        except Exception:
            pass

    # ------------------------------------------------------------------
    @staticmethod
    def make_eval_env(cfg: RLConfig) -> "LunarLanderWrapper":
        """Convenience factory for a frame-recording evaluation environment.

        Args:
            cfg: Hyperparameter config.

        Returns:
            A :class:`LunarLanderWrapper` with ``record_frames=True``.
        """
        return LunarLanderWrapper(cfg, render_mode="rgb_array", record_frames=True)

    @staticmethod
    def make_train_env(cfg: RLConfig) -> "LunarLanderWrapper":
        """Convenience factory for a lightweight training environment.

        Args:
            cfg: Hyperparameter config.

        Returns:
            A :class:`LunarLanderWrapper` with frame recording disabled.
        """
        return LunarLanderWrapper(cfg, render_mode="rgb_array", record_frames=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2-D  Ablation environment registry
# ─────────────────────────────────────────────────────────────────────────────

class AblationConfig:
    """Factory that returns the four ablation variant configs.

    Ablation variants (all share the same base RLConfig except flagged fields):
        - ``vanilla``  : No Dueling, No PER (baseline DQN)
        - ``per_only`` : No Dueling, PER enabled
        - ``dueling``  : Dueling, No PER
        - ``d3qn``     : Dueling + PER (full model)

    Args:
        base_cfg: Master :class:`RLConfig` from which variants are derived.

    Returns:
        Dict mapping variant name → :class:`RLConfig`.
    """

    @staticmethod
    def all_variants(base_cfg: RLConfig) -> Dict[str, RLConfig]:
        """Return all four ablation variant configs.

        Args:
            base_cfg: Template config; not mutated.

        Returns:
            Ordered dict of variant name → config.
        """
        import copy

        variants: Dict[str, RLConfig] = {}

        # Vanilla DQN
        v = copy.deepcopy(base_cfg)
        v.dueling = False
        v.use_per = False
        variants["vanilla_dqn"] = v

        # DQN + PER
        v = copy.deepcopy(base_cfg)
        v.dueling = False
        v.use_per = True
        variants["dqn_per"] = v

        # Dueling DQN (no PER)
        v = copy.deepcopy(base_cfg)
        v.dueling = True
        v.use_per = False
        variants["dueling_dqn"] = v

        # Full D3QN
        v = copy.deepcopy(base_cfg)
        v.dueling = True
        v.use_per = True
        variants["d3qn"] = v

        return variants


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SELF-TEST  (runs only when this file is executed directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("PROJECT OMEGA — Module 1 & 2 Self-Test")
    print("=" * 72)

    # Bootstrap
    cfg   = RLConfig()
    setup = PipelineSetup(cfg)
    log   = setup.logger

    log.info("Directory tree OK → %s", setup.paths["root"])
    log.info("Run ID: %s", setup.run_id)

    # Environment smoke-test
    env = LunarLanderWrapper.make_train_env(cfg)
    obs, info = env.reset(seed=cfg.seed)
    log.info("Env reset OK → obs shape: %s, dtype: %s", obs.shape, obs.dtype)

    for step in range(10):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()

    log.info("10-step rollout OK → last shaped reward: %.4f", rew)

    # Ablation configs
    variants = AblationConfig.all_variants(cfg)
    log.info("Ablation variants: %s", list(variants.keys()))

    # Normaliser check
    norm = RunningNormalizer(shape=(8,), clip=10.0)
    for _ in range(100):
        norm.update(np.random.randn(8))
    sample = norm.normalise(np.random.randn(8))
    log.info("Normaliser output range: [%.3f, %.3f]", sample.min(), sample.max())

    env.close()
    setup.tracker.finish()
    print("=" * 72)
    print("Module 1 & 2 PASSED. Ready for Neural Architecture.")
    print("=" * 72)
