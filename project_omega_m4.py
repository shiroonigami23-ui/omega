"""
================================================================================
PROJECT OMEGA — D3QN RESEARCH PIPELINE
================================================================================
Author  : Aryan Singh Chandel
Module  : 4 — Training Loop, Checkpointing, Evaluation & GIF Rendering
            · Full D3QN training loop with early stopping
            · Ablation runner (4 variants × full training)
            · Post-training greedy evaluation (100 episodes)
            · Best-episode GIF export via imageio
            · Metrics DataFrame saved as CSV
            · W&B + TensorBoard logging hooks
Depends : project_omega_m1_m2.py, project_omega_m3.py
================================================================================
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import pandas as pd
import torch

from project_omega_m1_m2 import (
    AblationConfig,
    LunarLanderWrapper,
    PipelineSetup,
    RLConfig,
)
from project_omega_m3 import (
    D3QNAgent,
    DEVICE,
    Transition,
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4-A  METRICS CONTAINER                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TrainingMetrics:
    """Accumulate and expose all per-episode training statistics.

    Attributes collected per episode:
        - raw reward, shaped reward, episode length
        - rolling average reward (window=cfg.rolling_window)
        - TD loss, mean Q-value, max Q-value, mean TD error
        - epsilon, beta (PER IS exponent), learning rate

    Args:
        cfg: Hyperparameter config (uses ``rolling_window``).
    """

    def __init__(self, cfg: RLConfig) -> None:
        self.cfg = cfg
        self._window = deque(maxlen=cfg.rolling_window)

        # Per-episode lists
        self.episodes:          List[int]   = []
        self.raw_rewards:       List[float] = []
        self.shaped_rewards:    List[float] = []
        self.rolling_avg:       List[float] = []
        self.episode_lengths:   List[int]   = []
        self.td_losses:         List[float] = []
        self.mean_qs:           List[float] = []
        self.max_qs:            List[float] = []
        self.mean_td_errors:    List[float] = []
        self.epsilons:          List[float] = []
        self.betas:             List[float] = []
        self.learning_rates:    List[float] = []
        self.timestamps:        List[float] = []

        self._t0 = time.time()

    # ------------------------------------------------------------------
    def record(
        self,
        episode:        int,
        raw_reward:     float,
        shaped_reward:  float,
        ep_length:      int,
        learn_metrics:  Dict[str, float],
        epsilon:        float,
    ) -> float:
        """Record one episode's statistics and return the current rolling avg.

        Args:
            episode:       Episode index (0-based).
            raw_reward:    Undiscounted sum of environment rewards.
            shaped_reward: Undiscounted sum of shaped rewards.
            ep_length:     Number of steps taken.
            learn_metrics: Dict from ``D3QNAgent.learn()`` (may be empty).
            epsilon:       Current exploration rate.

        Returns:
            Current rolling-average reward.
        """
        self._window.append(raw_reward)
        avg = float(np.mean(self._window))

        self.episodes.append(episode)
        self.raw_rewards.append(raw_reward)
        self.shaped_rewards.append(shaped_reward)
        self.rolling_avg.append(avg)
        self.episode_lengths.append(ep_length)
        self.td_losses.append(learn_metrics.get("loss",          0.0))
        self.mean_qs.append(  learn_metrics.get("mean_q",        0.0))
        self.max_qs.append(   learn_metrics.get("max_q",         0.0))
        self.mean_td_errors.append(learn_metrics.get("mean_td_error", 0.0))
        self.epsilons.append(epsilon)
        self.betas.append(   learn_metrics.get("beta",           0.0))
        self.learning_rates.append(learn_metrics.get("lr",       0.0))
        self.timestamps.append(time.time() - self._t0)

        return avg

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Serialise all recorded metrics to a tidy :class:`pandas.DataFrame`.

        Returns:
            DataFrame with one row per episode and all metric columns.
        """
        return pd.DataFrame({
            "episode":        self.episodes,
            "raw_reward":     self.raw_rewards,
            "shaped_reward":  self.shaped_rewards,
            "rolling_avg":    self.rolling_avg,
            "episode_length": self.episode_lengths,
            "td_loss":        self.td_losses,
            "mean_q":         self.mean_qs,
            "max_q":          self.max_qs,
            "mean_td_error":  self.mean_td_errors,
            "epsilon":        self.epsilons,
            "beta":           self.betas,
            "learning_rate":  self.learning_rates,
            "elapsed_sec":    self.timestamps,
        })

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics over all training episodes.

        Returns:
            Dict with keys: ``max_reward``, ``final_avg``, ``best_avg``,
            ``solved``, ``solve_episode``, ``total_episodes``,
            ``win_rate``, ``avg_ep_length``.
        """
        rr  = np.array(self.raw_rewards)
        avg = np.array(self.rolling_avg)
        thr = self.cfg.solve_threshold
        solved_eps = np.where(avg >= thr)[0]
        win_rate   = float(np.mean(rr >= thr)) if len(rr) else 0.0

        return {
            "max_reward":     float(rr.max()) if len(rr) else 0.0,
            "final_avg":      float(avg[-1])  if len(avg) else 0.0,
            "best_avg":       float(avg.max()) if len(avg) else 0.0,
            "solved":         len(solved_eps) > 0,
            "solve_episode":  int(solved_eps[0]) if len(solved_eps) else -1,
            "total_episodes": len(self.episodes),
            "win_rate":       win_rate,
            "avg_ep_length":  float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
        }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4-B  SINGLE-VARIANT TRAINER                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Trainer:
    """Full training loop for one D3QN variant.

    Orchestrates:
    - Episode rollout with the :class:`LunarLanderWrapper`
    - Warmup period (no learning until buffer has ``cfg.warmup_steps`` samples)
    - Per-step ``agent.learn()`` calls
    - Early stopping when rolling avg > ``cfg.solve_threshold`` for
      ``cfg.early_stop_window`` consecutive episodes
    - Checkpoint saving every ``cfg.checkpoint_every`` episodes IF rolling
      average improved
    - W&B + TensorBoard metric streaming
    - Final greedy evaluation (100 episodes)
    - Best-episode GIF export

    Args:
        cfg:     Hyperparameter config for this variant.
        setup:   :class:`PipelineSetup` instance (paths, logger, tracker).
        variant: Variant label string (e.g. ``"d3qn"``).
    """

    def __init__(
        self,
        cfg:     RLConfig,
        setup:   PipelineSetup,
        variant: str = "d3qn",
    ) -> None:
        self.cfg     = cfg
        self.setup   = setup
        self.variant = variant
        self.log     = setup.logger
        self.paths   = setup.paths
        self.tracker = setup.tracker
        self.device  = DEVICE

        self.agent   = D3QNAgent(cfg, self.device)
        self.metrics = TrainingMetrics(cfg)

        # Best rolling average seen so far (for conditional checkpointing)
        self._best_avg:  float = -float("inf")
        self._best_ckpt: Path  = self.paths["checkpoints"] / f"best_{variant}.pth"

        # Early-stop counter
        self._above_threshold: int = 0

    # ------------------------------------------------------------------
    def train(self) -> TrainingMetrics:
        """Execute the full training loop.

        Returns:
            :class:`TrainingMetrics` populated with all episode statistics.
        """
        cfg = self.cfg
        env = LunarLanderWrapper.make_train_env(cfg)
        env.reset(seed=cfg.seed)

        self.log.info(
            "[%s] Training started — %d episodes, device=%s",
            self.variant, cfg.max_episodes, self.device,
        )

        global_step   = 0
        episode_losses: List[float] = []
        episode_qs:     List[float] = []

        for ep in range(cfg.max_episodes):
            obs, info = env.reset()
            ep_raw        = 0.0
            ep_shaped     = 0.0
            ep_steps      = 0
            episode_losses.clear()
            episode_qs.clear()

            # ── Episode rollout ───────────────────────────────────────────────
            for _ in range(cfg.max_steps_per_ep):
                action = self.agent.act(obs)
                next_obs, shaped_r, terminated, truncated, info = env.step(action)
                done = float(terminated or truncated)

                self.agent.store(
                    Transition(obs, action, shaped_r, next_obs, done)
                )
                obs        = next_obs
                ep_raw    += info["raw_reward"]
                ep_shaped += shaped_r
                ep_steps  += 1
                global_step += 1

                # ── Learn once buffer is warm ─────────────────────────────────
                if (
                    len(self.agent.buffer) >= cfg.warmup_steps
                    and global_step % 1 == 0          # every step
                ):
                    lm = self.agent.learn()
                    if lm:
                        episode_losses.append(lm["loss"])
                        episode_qs.append(lm["mean_q"])

                if terminated or truncated:
                    break

            # ── Post-episode bookkeeping ──────────────────────────────────────
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            avg_q    = float(np.mean(episode_qs))     if episode_qs     else 0.0

            learn_metrics = {
                "loss":          avg_loss,
                "mean_q":        avg_q,
                "max_q":         float(np.max(episode_qs)) if episode_qs else 0.0,
                "mean_td_error": avg_loss,
                "beta":          self.agent.buffer.beta,
                "lr":            self.agent.optimizer.param_groups[0]["lr"],
            }

            rolling = self.metrics.record(
                episode       = ep,
                raw_reward    = ep_raw,
                shaped_reward = ep_shaped,
                ep_length     = ep_steps,
                learn_metrics = learn_metrics,
                epsilon       = self.agent.epsilon,
            )

            self.agent.decay_epsilon()
            self.agent.step_scheduler()

            # ── Logging ───────────────────────────────────────────────────────
            if ep % cfg.log_every == 0:
                self.log.info(
                    "[%s] ep=%4d | raw=% 8.2f | avg=%8.2f | ε=%.3f | "
                    "loss=%.4f | Q=%.3f | buf=%d",
                    self.variant, ep, ep_raw, rolling,
                    self.agent.epsilon, avg_loss, avg_q,
                    len(self.agent.buffer),
                )

            # W&B + TensorBoard
            self.tracker.log(
                {
                    f"{self.variant}/raw_reward":    ep_raw,
                    f"{self.variant}/rolling_avg":   rolling,
                    f"{self.variant}/epsilon":       self.agent.epsilon,
                    f"{self.variant}/td_loss":       avg_loss,
                    f"{self.variant}/mean_q":        avg_q,
                    f"{self.variant}/buffer_size":   float(len(self.agent.buffer)),
                    f"{self.variant}/beta":          self.agent.buffer.beta,
                    f"{self.variant}/lr":            learn_metrics["lr"],
                },
                step=ep,
            )

            # ── Conditional checkpointing ─────────────────────────────────────
            if (
                ep > 0
                and ep % cfg.checkpoint_every == 0
                and rolling > self._best_avg
            ):
                self._best_avg = rolling
                self.agent.save_checkpoint(str(self._best_ckpt))
                self.log.info(
                    "[%s] ✓ Checkpoint saved (avg=%.2f) → %s",
                    self.variant, rolling, self._best_ckpt.name,
                )

            # ── Early stopping ────────────────────────────────────────────────
            if rolling >= cfg.solve_threshold:
                self._above_threshold += 1
                if self._above_threshold >= cfg.early_stop_window:
                    self.log.info(
                        "[%s] ✓ Solved at episode %d (avg=%.2f for %d eps)",
                        self.variant, ep, rolling, cfg.early_stop_window,
                    )
                    break
            else:
                self._above_threshold = 0

        env.close()
        self.log.info("[%s] Training complete — %d episodes", self.variant, ep + 1)
        return self.metrics

    # ------------------------------------------------------------------
    def evaluate(self, n_episodes: int = 100) -> "EvalResult":
        """Run greedy evaluation episodes and collect states/actions/rewards.

        Loads the best checkpoint if it exists; otherwise uses current weights.

        Args:
            n_episodes: Number of purely greedy evaluation episodes.

        Returns:
            :class:`EvalResult` with all collected data.
        """
        if self._best_ckpt.exists():
            self.agent.load_checkpoint(str(self._best_ckpt))
            self.log.info("[%s] Loaded best checkpoint for eval", self.variant)

        eval_env  = LunarLanderWrapper.make_eval_env(self.cfg)
        result    = EvalResult(self.variant)
        best_ep_reward = -float("inf")
        best_ep_frames: List[np.ndarray] = []

        for ep in range(n_episodes):
            obs, _     = eval_env.reset(seed=self.cfg.seed + ep)
            ep_reward  = 0.0
            ep_states: List[np.ndarray] = []
            ep_actions: List[int]       = []

            for _ in range(self.cfg.max_steps_per_ep):
                action = self.agent.act(obs, greedy=True)
                next_obs, r, term, trunc, info = eval_env.step(action)

                ep_states.append(obs.copy())
                ep_actions.append(action)
                ep_reward += info["raw_reward"]
                obs = next_obs

                if term or trunc:
                    break

            result.record_episode(ep_reward, ep_states, ep_actions)

            # Track the best episode for GIF export
            if ep_reward > best_ep_reward:
                best_ep_reward = ep_reward
                best_ep_frames = list(eval_env.frame_buffer.frames())

        eval_env.close()

        self.log.info(
            "[%s] Eval complete — mean=%.2f  max=%.2f  win_rate=%.1f%%",
            self.variant,
            result.mean_reward,
            result.max_reward,
            result.win_rate * 100,
        )

        # Save best-episode GIF
        gif_path = self.paths["video"] / f"best_agent_{self.variant}.gif"
        _save_gif(best_ep_frames, gif_path, fps=30)
        self.log.info("[%s] GIF saved → %s (%d frames)", self.variant, gif_path.name, len(best_ep_frames))
        result.gif_path = gif_path

        return result

    # ------------------------------------------------------------------
    def save_metrics_csv(self) -> Path:
        """Persist training metrics DataFrame to CSV.

        Returns:
            Path to the written CSV file.
        """
        df   = self.metrics.to_dataframe()
        path = self.paths["plots_performance"] / f"training_metrics_{self.variant}.csv"
        df.to_csv(path, index=False)
        self.log.info("[%s] Metrics CSV → %s", self.variant, path.name)
        return path


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4-C  EVALUATION RESULT CONTAINER                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class EvalResult:
    """Stores post-training evaluation data for one agent variant.

    Args:
        variant: String label for this variant (e.g. ``"d3qn"``).
    """

    def __init__(self, variant: str) -> None:
        self.variant  = variant
        self.rewards:         List[float]            = []
        self.all_states:      List[List[np.ndarray]] = []
        self.all_actions:     List[List[int]]        = []
        self.gif_path: Optional[Path]                = None

    # ------------------------------------------------------------------
    def record_episode(
        self,
        reward:  float,
        states:  List[np.ndarray],
        actions: List[int],
    ) -> None:
        """Append one evaluation episode's data.

        Args:
            reward:  Total undiscounted raw reward for the episode.
            states:  Ordered list of normalised observations.
            actions: Ordered list of actions taken.
        """
        self.rewards.append(reward)
        self.all_states.append(states)
        self.all_actions.append(actions)

    # ------------------------------------------------------------------
    @property
    def mean_reward(self) -> float:
        """Mean reward over all evaluation episodes."""
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def max_reward(self) -> float:
        """Peak reward across all evaluation episodes."""
        return float(np.max(self.rewards)) if self.rewards else 0.0

    @property
    def std_reward(self) -> float:
        """Standard deviation of rewards across evaluation episodes."""
        return float(np.std(self.rewards)) if self.rewards else 0.0

    @property
    def win_rate(self) -> float:
        """Fraction of episodes with raw reward >= solve threshold (200)."""
        if not self.rewards:
            return 0.0
        return float(np.mean(np.array(self.rewards) >= 200.0))

    @property
    def flat_states(self) -> np.ndarray:
        """All states from all eval episodes, shape ``(N, state_dim)``."""
        return np.vstack([s for ep in self.all_states for s in ep])

    @property
    def flat_actions(self) -> np.ndarray:
        """All actions from all eval episodes, shape ``(N,)``."""
        return np.array([a for ep in self.all_actions for a in ep])

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a flat dict of scalar summary statistics.

        Returns:
            Dict with keys: variant, mean_reward, max_reward, std_reward,
            win_rate, n_episodes.
        """
        return {
            "variant":      self.variant,
            "mean_reward":  self.mean_reward,
            "max_reward":   self.max_reward,
            "std_reward":   self.std_reward,
            "win_rate":     self.win_rate,
            "n_episodes":   len(self.rewards),
        }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4-D  GIF EXPORT UTILITY                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _save_gif(
    frames:   List[np.ndarray],
    out_path: Path,
    fps:      int = 30,
    max_frames: int = 600,
) -> None:
    """Write a list of RGB arrays to an animated GIF via imageio.

    Downsamples to ``max_frames`` if the episode is very long to keep
    the file size manageable.

    Args:
        frames:     List of uint8 RGB arrays, each shape ``(H, W, 3)``.
        out_path:   Destination path for the ``.gif`` file.
        fps:        Frames per second for the output GIF.
        max_frames: Hard cap on number of frames written.
    """
    if not frames:
        return

    # Uniform downsampling if needed
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames  = [frames[i] for i in indices]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(
            str(out_path),
            frames,
            fps=fps,
            loop=0,          # loop forever
        )
    except Exception as exc:
        # Non-fatal: GIF export failing should not crash the pipeline
        import logging
        logging.getLogger("omega").warning("GIF export failed: %s", exc)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4-E  ABLATION RUNNER                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class AblationRunner:
    """Train and evaluate all four ablation variants sequentially.

    Variants:
        - ``vanilla_dqn``  : No Dueling, no PER
        - ``dqn_per``      : No Dueling, PER enabled
        - ``dueling_dqn``  : Dueling, no PER
        - ``d3qn``         : Full D3QN (Dueling + PER)

    Args:
        base_cfg: Master :class:`RLConfig`; variants are derived from it.
        setup:    :class:`PipelineSetup` instance.
    """

    def __init__(self, base_cfg: RLConfig, setup: PipelineSetup) -> None:
        self.base_cfg = base_cfg
        self.setup    = setup
        self.log      = setup.logger

        self.variants:      Dict[str, RLConfig]    = AblationConfig.all_variants(base_cfg)
        self.trainers:      Dict[str, Trainer]     = {}
        self.all_metrics:   Dict[str, TrainingMetrics] = {}
        self.all_eval:      Dict[str, EvalResult]  = {}

    # ------------------------------------------------------------------
    def run_all(self) -> Tuple[Dict[str, TrainingMetrics], Dict[str, EvalResult]]:
        """Train and evaluate every variant in sequence.

        Returns:
            Tuple of (metrics_dict, eval_dict) keyed by variant name.
        """
        for name, cfg in self.variants.items():
            self.log.info("=" * 60)
            self.log.info("ABLATION VARIANT: %s", name.upper())
            self.log.info("  dueling=%s  use_per=%s", cfg.dueling, cfg.use_per)
            self.log.info("=" * 60)

            trainer = Trainer(cfg, self.setup, variant=name)
            metrics = trainer.train()
            eval_r  = trainer.evaluate(n_episodes=self.base_cfg.eval_episodes)
            trainer.save_metrics_csv()

            self.trainers[name]    = trainer
            self.all_metrics[name] = metrics
            self.all_eval[name]    = eval_r

        self._save_combined_csv()
        self._save_eval_summary_csv()
        return self.all_metrics, self.all_eval

    # ------------------------------------------------------------------
    def _save_combined_csv(self) -> None:
        """Write a single CSV combining metrics from all variants."""
        frames = []
        for name, m in self.all_metrics.items():
            df = m.to_dataframe()
            df["variant"] = name
            frames.append(df)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            path = self.setup.paths["plots_performance"] / "all_variants_metrics.csv"
            combined.to_csv(path, index=False)
            self.log.info("Combined ablation CSV → %s", path.name)

    # ------------------------------------------------------------------
    def _save_eval_summary_csv(self) -> None:
        """Write a summary CSV of eval statistics for all variants."""
        rows = [r.to_summary_dict() for r in self.all_eval.values()]
        df   = pd.DataFrame(rows)
        path = self.setup.paths["plots_performance"] / "eval_summary.csv"
        df.to_csv(path, index=False)
        self.log.info("Eval summary CSV → %s", path.name)

    # ------------------------------------------------------------------
    def build_final_metrics_table(self) -> pd.DataFrame:
        """Compile the Module-5-ready final metrics summary table.

        Includes training stats + eval stats for every variant.

        Returns:
            DataFrame with one row per variant and all summary columns.
        """
        rows = []
        for name in self.variants:
            m   = self.all_metrics.get(name)
            ev  = self.all_eval.get(name)
            tr  = self.trainers.get(name)

            train_sum = m.summary() if m else {}
            eval_sum  = ev.to_summary_dict() if ev else {}

            rows.append({
                "variant":         name,
                "dueling":         self.variants[name].dueling,
                "use_per":         self.variants[name].use_per,
                # Training
                "train_max_reward":  train_sum.get("max_reward",     0.0),
                "train_best_avg":    train_sum.get("best_avg",       0.0),
                "train_final_avg":   train_sum.get("final_avg",      0.0),
                "solved":            train_sum.get("solved",         False),
                "solve_episode":     train_sum.get("solve_episode",  -1),
                "total_train_eps":   train_sum.get("total_episodes", 0),
                # Eval
                "eval_mean_reward":  eval_sum.get("mean_reward",     0.0),
                "eval_max_reward":   eval_sum.get("max_reward",      0.0),
                "eval_std_reward":   eval_sum.get("std_reward",      0.0),
                "eval_win_rate":     eval_sum.get("win_rate",        0.0),
            })

        return pd.DataFrame(rows)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SELF-TEST                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 72)
    print("PROJECT OMEGA — Module 4 Self-Test (SMOKE: 5 episodes)")
    print("=" * 72)

    # ── Smoke config (very short training for self-test) ──────────────────────
    cfg                     = RLConfig()
    cfg.max_episodes        = 5
    cfg.max_steps_per_ep    = 200
    cfg.eval_episodes       = 3
    cfg.warmup_steps        = 100
    cfg.batch_size          = 32
    cfg.buffer_capacity     = 2_000
    cfg.checkpoint_every    = 2
    cfg.log_every           = 1
    cfg.use_wandb           = False
    cfg.use_tensorboard     = False

    # Bootstrap (no Drive in local test)
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        setup = PipelineSetup(cfg, project_base=Path(tmpdir))
        log   = setup.logger

        # ── Single trainer smoke test ─────────────────────────────────────────
        trainer = Trainer(cfg, setup, variant="d3qn")
        metrics = trainer.train()
        assert len(metrics.episodes) == 5
        log.info("Trainer smoke OK — %d episodes recorded", len(metrics.episodes))

        # ── Metrics DataFrame ─────────────────────────────────────────────────
        df = metrics.to_dataframe()
        assert set(["episode", "raw_reward", "rolling_avg", "td_loss"]).issubset(df.columns)
        log.info("Metrics DataFrame OK — shape %s", df.shape)

        # ── Summary dict ──────────────────────────────────────────────────────
        s = metrics.summary()
        assert "max_reward" in s and "win_rate" in s
        log.info("Summary OK — max_reward=%.2f  win_rate=%.2f", s["max_reward"], s["win_rate"])

        # ── Evaluation ────────────────────────────────────────────────────────
        result = trainer.evaluate(n_episodes=3)
        assert len(result.rewards) == 3
        assert result.flat_states.ndim == 2
        log.info("Eval OK — mean_reward=%.2f  states_shape=%s",
                 result.mean_reward, result.flat_states.shape)

        # ── CSV export ────────────────────────────────────────────────────────
        csv_path = trainer.save_metrics_csv()
        assert csv_path.exists()
        log.info("CSV export OK → %s", csv_path.name)

        # ── EvalResult helpers ────────────────────────────────────────────────
        sum_d = result.to_summary_dict()
        assert sum_d["variant"] == "d3qn"
        assert 0.0 <= sum_d["win_rate"] <= 1.0
        log.info("EvalResult summary OK")

        # ── TrainingMetrics record boundary ──────────────────────────────────
        tm  = TrainingMetrics(cfg)
        avg = tm.record(0, 100.0, 95.0, 50, {"loss": 0.5, "mean_q": 1.2}, 0.9)
        assert abs(avg - 100.0) < 1e-6
        log.info("TrainingMetrics record OK — rolling_avg=%.2f", avg)

        # ── GIF export (small dummy frames) ──────────────────────────────────
        dummy_frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(10)]
        gif_out = Path(tmpdir) / "test.gif"
        _save_gif(dummy_frames, gif_out, fps=10)
        assert gif_out.exists() and gif_out.stat().st_size > 0
        log.info("GIF export OK — size=%d bytes", gif_out.stat().st_size)

        print("=" * 72)
        print("Module 4 PASSED. Ready for Visualization Engine.")
        print("=" * 72)
