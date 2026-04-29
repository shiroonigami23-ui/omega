"""
================================================================================
PROJECT OMEGA -- D3QN RESEARCH PIPELINE
================================================================================
Author  : Aryan Singh Chandel
Module  : 8 -- Packaging, SHA-256 Hashing & Main Orchestrator
            . Recursive ZIP of /models /plots /reports /video /logs
            . SHA-256 checksum -> manifest_hash.txt
            . Google Drive sync of final bundle
            . Full if __name__ == "__main__" pipeline runner
            . Colab install cell printed at startup
            . Graceful error handling -- pipeline never crashes silently
Depends : project_omega_m1_m2.py
          project_omega_m3.py
          project_omega_m4.py
          project_omega_m5.py
          project_omega_m6.py
          project_omega_m7.py
================================================================================

COLAB QUICK-START
=================
Cell 1 (install):
    !pip install -q gymnasium[box2d] torch fpdf2 reportlab python-pptx \
                    wandb imageio moviepy scikit-learn seaborn scipy pandas \
                    matplotlib numpy

Cell 2 (upload all 8 .py files to /content/, then):
    exec(open("project_omega_m8.py").read())
================================================================================
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
#  8-A  ARTEFACT PACKAGER
# ============================================================================

class ArtefactPackager:
    """Recursively zip selected project directories and produce a SHA-256 manifest.

    Directories included in the ZIP:
        /models, /plots, /reports, /video, /logs

    Args:
        root:      Project root (parent of all subdirectories).
        out_dir:   Directory where the ZIP and manifest are written.
        logger:    Logger instance for status messages.
    """

    ZIP_NAME      = "Project_Omega_Release.zip"
    MANIFEST_NAME = "manifest_hash.txt"

    # Subdirectories to include (relative to root)
    INCLUDE_DIRS: List[str] = [
        "models",
        "plots",
        "reports",
        "video",
        "logs",
    ]

    def __init__(
        self,
        root:    Path,
        out_dir: Path,
        logger:  logging.Logger,
    ) -> None:
        self.root    = root
        self.out_dir = out_dir
        self.log     = logger
        out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    @property
    def zip_path(self) -> Path:
        """Full path to the output ZIP file."""
        return self.out_dir / self.ZIP_NAME

    @property
    def manifest_path(self) -> Path:
        """Full path to the SHA-256 manifest text file."""
        return self.out_dir / self.MANIFEST_NAME

    # ------------------------------------------------------------------
    def pack(self) -> Tuple[Path, str]:
        """Create the ZIP archive and compute its SHA-256 digest.

        Walks each included directory recursively, skipping
        .pyc / __pycache__ files. Files are stored with paths
        relative to ``self.root`` to keep the archive self-contained.

        Returns:
            Tuple of (zip_path, sha256_hex_digest).
        """
        self.log.info("Packaging artefacts -> %s", self.ZIP_NAME)
        file_count = 0

        with zipfile.ZipFile(
            str(self.zip_path),
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as zf:
            for subdir in self.INCLUDE_DIRS:
                abs_dir = self.root / subdir
                if not abs_dir.exists():
                    self.log.debug("Skipping missing dir: %s", subdir)
                    continue

                for file_path in sorted(abs_dir.rglob("*")):
                    if not file_path.is_file():
                        continue
                    if "__pycache__" in file_path.parts:
                        continue
                    if file_path.suffix in {".pyc", ".pyo"}:
                        continue

                    arc_name = file_path.relative_to(self.root)
                    zf.write(str(file_path), arcname=str(arc_name))
                    file_count += 1

        size_mb = self.zip_path.stat().st_size / 1e6
        self.log.info(
            "ZIP created: %d files, %.2f MB -> %s",
            file_count, size_mb, self.zip_path.name,
        )
        return self.zip_path, self._sha256(self.zip_path)

    # ------------------------------------------------------------------
    @staticmethod
    def _sha256(path: Path, chunk: int = 1 << 20) -> str:
        """Compute SHA-256 digest of a file in streaming chunks.

        Args:
            path:  Path to the file to hash.
            chunk: Read chunk size in bytes (default 1 MiB).

        Returns:
            Lowercase hex-encoded SHA-256 digest string.
        """
        h = hashlib.sha256()
        with open(str(path), "rb") as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    # ------------------------------------------------------------------
    def write_manifest(self, digest: str) -> Path:
        """Write the SHA-256 hash and metadata to ``manifest_hash.txt``.

        The manifest includes:
        - Archive filename and size
        - SHA-256 digest
        - Generation timestamp
        - Author
        - Verification instructions

        Args:
            digest: Hex SHA-256 digest of the ZIP file.

        Returns:
            Path to the written manifest file.
        """
        size_bytes = self.zip_path.stat().st_size
        timestamp  = time.strftime("%Y-%m-%d %H:%M:%S UTC")

        manifest_text = (
            "=" * 72 + "\n"
            "PROJECT OMEGA -- ARTEFACT INTEGRITY MANIFEST\n"
            "=" * 72 + "\n\n"
            f"Archive   : {self.ZIP_NAME}\n"
            f"Size      : {size_bytes:,} bytes  ({size_bytes / 1e6:.3f} MB)\n"
            f"Algorithm : SHA-256\n"
            f"Digest    : {digest}\n\n"
            f"Generated : {timestamp}\n"
            f"Author    : Aryan Singh Chandel\n"
            f"Pipeline  : Project Omega -- D3QN Research Pipeline\n\n"
            "VERIFICATION\n"
            "------------\n"
            "Linux / macOS:\n"
            f"    sha256sum {self.ZIP_NAME}\n"
            "    # Expected: " + digest + "\n\n"
            "Python:\n"
            "    import hashlib\n"
            f"    h = hashlib.sha256(open('{self.ZIP_NAME}','rb').read()).hexdigest()\n"
            f"    assert h == '{digest}', 'Integrity check FAILED'\n"
            "    print('Integrity check PASSED')\n\n"
            "=" * 72 + "\n"
        )

        self.manifest_path.write_text(manifest_text, encoding="utf-8")
        self.log.info("Manifest written -> %s", self.manifest_path.name)
        self.log.info("SHA-256: %s", digest)
        return self.manifest_path

    # ------------------------------------------------------------------
    def sync_to_drive(self, drive_root: Path) -> bool:
        """Copy ZIP and manifest to Google Drive.

        Args:
            drive_root: Google Drive root path (e.g. /content/drive/MyDrive).

        Returns:
            True if copy succeeded, False otherwise.
        """
        if not drive_root.exists():
            self.log.warning("Drive not mounted -- skipping Drive sync")
            return False

        dest = drive_root / "Project_Omega_D3QN" / "output"
        dest.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(str(self.zip_path),      str(dest / self.ZIP_NAME))
            shutil.copy2(str(self.manifest_path), str(dest / self.MANIFEST_NAME))
            self.log.info("Drive sync OK -> %s", dest)
            return True
        except Exception as exc:
            self.log.error("Drive sync failed: %s", exc)
            return False


# ============================================================================
#  8-B  PIPELINE TIMER
# ============================================================================

class PipelineTimer:
    """Track wall-clock time for each pipeline stage.

    Args:
        logger: Logger instance.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.log    = logger
        self._times: Dict[str, float] = {}
        self._start: Optional[float]  = None
        self._stage_start: Optional[float] = None
        self._current_stage: Optional[str] = None

    def start_pipeline(self) -> None:
        """Record pipeline start time."""
        self._start = time.time()
        self.log.info("=" * 60)
        self.log.info("PROJECT OMEGA PIPELINE STARTED")
        self.log.info("=" * 60)

    def start_stage(self, name: str) -> None:
        """Mark the beginning of a named stage.

        Args:
            name: Human-readable stage name.
        """
        self._stage_start   = time.time()
        self._current_stage = name
        self.log.info("-" * 40)
        self.log.info("STAGE: %s", name)
        self.log.info("-" * 40)

    def end_stage(self) -> float:
        """Mark the end of the current stage and return elapsed seconds.

        Returns:
            Elapsed time in seconds for the current stage.
        """
        if self._stage_start is None or self._current_stage is None:
            return 0.0
        elapsed = time.time() - self._stage_start
        self._times[self._current_stage] = elapsed
        self.log.info(
            "STAGE COMPLETE: %s -- %.1f s (%.1f min)",
            self._current_stage, elapsed, elapsed / 60,
        )
        return elapsed

    def summary(self) -> str:
        """Format a timing summary table for all completed stages.

        Returns:
            Multi-line string with stage names and elapsed times.
        """
        if not self._times:
            return "No stages recorded."
        total = time.time() - (self._start or time.time())
        lines = [
            "",
            "=" * 50,
            "PIPELINE TIMING SUMMARY",
            "=" * 50,
        ]
        for stage, secs in self._times.items():
            lines.append(f"  {stage:<35s}  {secs/60:6.1f} min")
        lines += [
            "-" * 50,
            f"  {'TOTAL':<35s}  {total/60:6.1f} min",
            "=" * 50,
            "",
        ]
        return "\n".join(lines)


# ============================================================================
#  8-C  COMPLETE PIPELINE ORCHESTRATOR
# ============================================================================

def run_pipeline(
    cfg_overrides: Optional[Dict[str, Any]] = None,
    skip_ablation: bool = False,
    project_base:  Optional[Path] = None,
) -> None:
    """Execute the full Project Omega pipeline end-to-end.

    Stage sequence:
    1.  Bootstrap (M1): directories, logger, Drive mount, trackers
    2.  Environment smoke-test (M2): verify LunarLander-v3 works
    3.  Ablation training (M3+M4): train 4 variants sequentially
    4.  Visualisation (M5): generate all 8 plots + CSV
    5.  PDF report (M6): compile 11-page academic PDF
    6.  PPTX deck (M7): build 5-slide executive presentation
    7.  Packaging (M8): ZIP + SHA-256 manifest + Drive sync

    Args:
        cfg_overrides: Optional dict of RLConfig field overrides
                       (e.g. ``{"max_episodes": 100}`` for a quick test run).
        skip_ablation: If True, only trains the full D3QN variant (faster).
        project_base:  Override for the project root directory.
    """
    # ------------------------------------------------------------------ #
    #  STAGE 0 -- Imports (deferred so syntax errors surface cleanly)     #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("PROJECT OMEGA -- D3QN RESEARCH PIPELINE")
    print("Author: Aryan Singh Chandel")
    print("=" * 72)
    _print_colab_install_hint()

    try:
        from project_omega_m1_m2 import (
            RLConfig, PipelineSetup,
            LunarLanderWrapper, AblationConfig,
        )
        from project_omega_m3 import DEVICE
        from project_omega_m4 import AblationRunner, Trainer, TrainingMetrics, EvalResult
        from project_omega_m5 import VisualizationEngine
        from project_omega_m6 import PDFReportCompiler
        from project_omega_m7 import PPTXCompiler
    except ImportError as exc:
        print(f"\n[FATAL] Missing module: {exc}")
        print("Ensure all 8 project_omega_m*.py files are in the same directory.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    #  STAGE 1 -- Bootstrap                                               #
    # ------------------------------------------------------------------ #
    cfg = RLConfig()
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    setup = PipelineSetup(cfg, project_base=project_base)
    log   = setup.logger
    timer = PipelineTimer(log)
    timer.start_pipeline()

    log.info("Device: %s", DEVICE)
    log.info("Config: max_episodes=%d  batch=%d  buffer=%d",
             cfg.max_episodes, cfg.batch_size, cfg.buffer_capacity)

    # ------------------------------------------------------------------ #
    #  STAGE 2 -- Environment smoke-test                                  #
    # ------------------------------------------------------------------ #
    timer.start_stage("Environment Verification")
    try:
        env = LunarLanderWrapper.make_train_env(cfg)
        obs, _ = env.reset(seed=cfg.seed)
        for _ in range(5):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset()
        env.close()
        log.info("LunarLander-v3 smoke-test PASSED -- obs shape: %s", obs.shape)
    except Exception as exc:
        log.error("Environment smoke-test FAILED: %s", exc)
        log.error(traceback.format_exc())
        log.error("Install: pip install gymnasium[box2d]")
        sys.exit(1)
    timer.end_stage()

    # ------------------------------------------------------------------ #
    #  STAGE 3 -- Training (ablation or single variant)                   #
    # ------------------------------------------------------------------ #
    timer.start_stage("Ablation Training (4 variants)")
    all_metrics: Dict[str, TrainingMetrics] = {}
    all_eval:    Dict[str, EvalResult]      = {}

    try:
        if skip_ablation:
            # Quick mode: train full D3QN only
            log.info("skip_ablation=True -- training D3QN variant only")
            trainer = Trainer(cfg, setup, variant="d3qn")
            metrics = trainer.train()
            result  = trainer.evaluate(n_episodes=cfg.eval_episodes)
            trainer.save_metrics_csv()
            all_metrics["d3qn"] = metrics
            all_eval["d3qn"]    = result

            # Synthetic stand-ins for the other 3 variants
            # (so visualisations don't crash on missing keys)
            for v in ["vanilla_dqn", "dqn_per", "dueling_dqn"]:
                all_metrics[v] = metrics
                all_eval[v]    = result

            final_table = _single_variant_table(cfg, metrics, result)

        else:
            runner = AblationRunner(cfg, setup)
            all_metrics, all_eval = runner.run_all()
            final_table = runner.build_final_metrics_table()

        # Save master final_metrics.csv
        csv_path = setup.paths["plots_performance"] / "final_metrics.csv"
        final_table.to_csv(str(csv_path), index=False)
        log.info("Master final_metrics.csv saved")

    except Exception as exc:
        log.error("Training stage FAILED: %s", exc)
        log.error(traceback.format_exc())
        _emergency_checkpoint(setup.paths["logs"])
        raise
    timer.end_stage()

    # ------------------------------------------------------------------ #
    #  STAGE 4 -- Visualisation                                           #
    # ------------------------------------------------------------------ #
    timer.start_stage("Visualization Engine (8 plots)")
    plot_paths: Dict[str, Path] = {}
    try:
        engine     = VisualizationEngine(setup, all_metrics, all_eval, cfg)
        plot_paths = engine.generate_all(final_table)
        log.info("Plots generated: %d", len(plot_paths))
    except Exception as exc:
        log.error("Visualization stage FAILED: %s", exc)
        log.error(traceback.format_exc())
        # Non-fatal: continue to PDF/PPTX with whatever plots exist
    timer.end_stage()

    # ------------------------------------------------------------------ #
    #  STAGE 5 -- Academic PDF                                            #
    # ------------------------------------------------------------------ #
    timer.start_stage("Academic PDF Report (FPDF2 + ReportLab)")
    pdf_path: Optional[Path] = None
    try:
        compiler = PDFReportCompiler(
            setup, cfg, all_metrics, all_eval, plot_paths, final_table
        )
        pdf_path = compiler.compile()
    except Exception as exc:
        log.error("PDF stage FAILED: %s", exc)
        log.error(traceback.format_exc())
    timer.end_stage()

    # ------------------------------------------------------------------ #
    #  STAGE 6 -- Executive PPTX                                          #
    # ------------------------------------------------------------------ #
    timer.start_stage("Executive PPTX Deck (5 slides)")
    pptx_path: Optional[Path] = None
    try:
        pptx_compiler = PPTXCompiler(
            setup, cfg, all_metrics, all_eval, plot_paths, final_table
        )
        pptx_path = pptx_compiler.compile()
    except Exception as exc:
        log.error("PPTX stage FAILED: %s", exc)
        log.error(traceback.format_exc())
    timer.end_stage()

    # ------------------------------------------------------------------ #
    #  STAGE 7 -- Packaging & SHA-256                                     #
    # ------------------------------------------------------------------ #
    timer.start_stage("ZIP Packaging & SHA-256 Manifest")
    zip_path: Optional[Path]     = None
    sha256:   Optional[str]      = None
    try:
        packager          = ArtefactPackager(
            root    = setup.paths["root"],
            out_dir = setup.paths["output"],
            logger  = log,
        )
        zip_path, sha256  = packager.pack()
        packager.write_manifest(sha256)

        # Sync to Google Drive
        drive_root = Path("/content/drive/MyDrive")
        packager.sync_to_drive(drive_root)

    except Exception as exc:
        log.error("Packaging stage FAILED: %s", exc)
        log.error(traceback.format_exc())
    timer.end_stage()

    # ------------------------------------------------------------------ #
    #  FINAL SUMMARY                                                       #
    # ------------------------------------------------------------------ #
    log.info(timer.summary())
    _print_final_banner(
        log         = log,
        cfg         = cfg,
        all_metrics = all_metrics,
        all_eval    = all_eval,
        pdf_path    = pdf_path,
        pptx_path   = pptx_path,
        zip_path    = zip_path,
        sha256      = sha256,
        root        = setup.paths["root"],
    )

    # Flush experiment trackers
    setup.tracker.finish()


# ============================================================================
#  8-D  HELPERS
# ============================================================================

def _print_colab_install_hint() -> None:
    """Print the Colab pip install cell to stdout (once, at startup)."""
    print("\n# ---- COLAB INSTALL (run in a separate cell if needed) ----")
    print("# !pip install -q gymnasium[box2d] torch fpdf2 reportlab \\")
    print("#               python-pptx wandb imageio moviepy \\")
    print("#               scikit-learn seaborn scipy pandas matplotlib numpy")
    print("# -----------------------------------------------------------\n")


def _single_variant_table(
    cfg:     Any,
    metrics: Any,
    result:  Any,
) -> pd.DataFrame:
    """Build a minimal final_table when only D3QN was trained.

    Args:
        cfg:     RLConfig instance.
        metrics: TrainingMetrics for D3QN.
        result:  EvalResult for D3QN.

    Returns:
        Single-row DataFrame compatible with Module 5/6/7 consumers.
    """
    s  = metrics.summary()
    es = result.to_summary_dict()
    return pd.DataFrame([{
        "variant":           "d3qn",
        "dueling":           cfg.dueling,
        "use_per":           cfg.use_per,
        "train_best_avg":    s["best_avg"],
        "train_final_avg":   s["final_avg"],
        "solved":            s["solved"],
        "solve_episode":     s["solve_episode"],
        "total_train_eps":   s["total_episodes"],
        "eval_mean_reward":  es["mean_reward"],
        "eval_max_reward":   es["max_reward"],
        "eval_std_reward":   es["std_reward"],
        "eval_win_rate":     es["win_rate"],
    }])


def _emergency_checkpoint(log_dir: Path) -> None:
    """Write an emergency timestamp file if training crashes.

    Args:
        log_dir: Directory to write the crash marker into.
    """
    try:
        marker = log_dir / "CRASH_MARKER.txt"
        marker.write_text(
            f"Pipeline crashed at {time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            "Check training_run.log for full traceback.\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def _print_final_banner(
    log:         logging.Logger,
    cfg:         Any,
    all_metrics: Dict[str, Any],
    all_eval:    Dict[str, Any],
    pdf_path:    Optional[Path],
    pptx_path:   Optional[Path],
    zip_path:    Optional[Path],
    sha256:      Optional[str],
    root:        Path,
) -> None:
    """Log the final results banner summarising all pipeline outputs.

    Args:
        log:         Logger instance.
        cfg:         RLConfig used for training.
        all_metrics: Dict of TrainingMetrics per variant.
        all_eval:    Dict of EvalResult per variant.
        pdf_path:    Path to the generated PDF, or None.
        pptx_path:   Path to the generated PPTX, or None.
        zip_path:    Path to the generated ZIP, or None.
        sha256:      SHA-256 hex digest of the ZIP, or None.
        root:        Project root directory.
    """
    d3qn_m = all_metrics.get("d3qn", next(iter(all_metrics.values())))
    d3qn_e = all_eval.get("d3qn",    next(iter(all_eval.values())))
    ts     = d3qn_m.summary()
    ev     = d3qn_e

    lines = [
        "",
        "=" * 72,
        "  PROJECT OMEGA -- PIPELINE COMPLETE",
        "=" * 72,
        f"  Author       : Aryan Singh Chandel",
        f"  Environment  : {cfg.env_id}",
        f"  Device       : (see logs)",
        "",
        "  D3QN RESULTS",
        f"  Solved       : {ts['solved']}",
        f"  Solve episode: {ts['solve_episode']}",
        f"  Best avg     : {ts['best_avg']:.2f}",
        f"  Eval mean    : {ev.mean_reward:.2f} +/- {ev.std_reward:.2f}",
        f"  Win rate     : {ev.win_rate * 100:.1f}%",
        f"  Peak reward  : {ev.max_reward:.2f}",
        "",
        "  ARTEFACTS",
        f"  PDF report   : {pdf_path.name  if pdf_path  else 'FAILED'}",
        f"  PPTX deck    : {pptx_path.name if pptx_path else 'FAILED'}",
        f"  ZIP archive  : {zip_path.name  if zip_path  else 'FAILED'}",
        f"  SHA-256      : {sha256[:32] + '...' if sha256 else 'N/A'}",
        f"  Output dir   : {root}",
        "=" * 72,
        "",
    ]
    for line in lines:
        log.info(line)

    # Also print to stdout for Colab cell output visibility
    print("\n".join(lines))


# ============================================================================
#  8-E  SELF-TEST
# ============================================================================

def _run_self_test() -> None:
    """Run Module 8 unit tests (packaging and manifest logic only)."""
    import tempfile
    print("=" * 72)
    print("PROJECT OMEGA -- Module 8 Self-Test")
    print("=" * 72)

    # -- SumTree round-trip already tested in M3, skip here
    # -- Focus: ArtefactPackager end-to-end

    with tempfile.TemporaryDirectory() as tmpdir:
        root    = Path(tmpdir) / "project"
        out_dir = root / "output"

        # Create fake artefact tree
        for subdir in ["models/checkpoints", "plots/performance",
                       "reports", "video", "logs"]:
            (root / subdir).mkdir(parents=True)

        (root / "models" / "checkpoints" / "best_d3qn.pth").write_bytes(b"fake_weights" * 100)
        (root / "plots"  / "performance" / "performance_curve.png").write_bytes(b"fake_png" * 500)
        (root / "reports" / "D3QN_Research_Paper.pdf").write_bytes(b"fake_pdf" * 1000)
        (root / "video"  / "best_agent_d3qn.gif").write_bytes(b"fake_gif" * 200)
        (root / "logs"   / "training_run.log").write_text("INFO | training\n" * 100)

        log = logging.getLogger("omega.m8_test")
        log.setLevel(logging.INFO)
        if not log.handlers:
            log.addHandler(logging.StreamHandler())

        packager = ArtefactPackager(root=root, out_dir=out_dir, logger=log)

        # -- pack()
        zip_p, digest = packager.pack()
        assert zip_p.exists(), "ZIP not created"
        assert len(digest) == 64, f"Bad digest length: {len(digest)}"
        assert zip_p.stat().st_size > 100, "ZIP not created"

        # -- verify contents
        with zipfile.ZipFile(str(zip_p)) as zf:
            names = zf.namelist()
        assert any("best_d3qn.pth" in n for n in names), "Model not in ZIP"
        assert any("performance_curve.png" in n for n in names), "Plot not in ZIP"
        assert any("D3QN_Research_Paper.pdf" in n for n in names), "PDF not in ZIP"
        assert any("best_agent_d3qn.gif" in n for n in names), "GIF not in ZIP"
        assert any("training_run.log" in n for n in names), "Log not in ZIP"
        print(f"  ZIP contents: {len(names)} files  ({zip_p.stat().st_size//1024} KB)")

        # -- write_manifest()
        man_p = packager.write_manifest(digest)
        assert man_p.exists()
        man_text = man_p.read_text()
        assert digest in man_text
        assert "SHA-256" in man_text
        assert "Aryan Singh Chandel" in man_text
        assert "sha256sum" in man_text
        print(f"  Manifest: {man_p.stat().st_size} bytes  OK")

        # -- integrity: re-hash and verify
        rehash = ArtefactPackager._sha256(zip_p)
        assert rehash == digest, "Re-hash mismatch!"
        print(f"  SHA-256 integrity verified: {digest[:32]}...")

        # -- Drive sync (no Drive mounted -- should return False gracefully)
        ok = packager.sync_to_drive(Path("/nonexistent/drive"))
        assert ok is False
        print(f"  Drive sync graceful failure: OK")

        # -- PipelineTimer
        import logging as _logging
        tlog = _logging.getLogger("omega.timer_test")
        tlog.addHandler(_logging.StreamHandler())
        tlog.setLevel(_logging.INFO)
        pt = PipelineTimer(tlog)
        pt.start_pipeline()
        pt.start_stage("Test Stage")
        time.sleep(0.05)
        elapsed = pt.end_stage()
        assert elapsed >= 0.04
        summary = pt.summary()
        assert "Test Stage" in summary
        assert "TOTAL" in summary
        print(f"  PipelineTimer: elapsed={elapsed:.3f}s  summary OK")

    print("=" * 72)
    print("Module 8 PASSED.")
    print("=" * 72)


# ============================================================================
#  MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Project Omega -- D3QN Research Pipeline"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run Module 8 self-test only (no training)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke run: 10 episodes, D3QN only, no ablation",
    )
    parser.add_argument(
        "--skip-ablation", action="store_true",
        help="Train D3QN only (skip the 3 ablation variants)",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override max_episodes (e.g. 500 for a short run)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--no-tb", action="store_true",
        help="Disable TensorBoard logging",
    )
    args = parser.parse_args()

    # ---- Self-test mode ----
    if args.test:
        _run_self_test()
        sys.exit(0)

    # ---- Build config overrides ----
    overrides: Dict[str, Any] = {}

    if args.quick:
        overrides.update({
            "max_episodes":     10,
            "max_steps_per_ep": 200,
            "eval_episodes":    3,
            "warmup_steps":     200,
            "batch_size":       32,
            "buffer_capacity":  2_000,
            "use_wandb":        False,
            "use_tensorboard":  False,
        })
        skip_abl = True
    else:
        skip_abl = args.skip_ablation

    if args.episodes is not None:
        overrides["max_episodes"] = args.episodes
    if args.no_wandb:
        overrides["use_wandb"] = False
    if args.no_tb:
        overrides["use_tensorboard"] = False

    # ---- Run the full pipeline ----
    run_pipeline(
        cfg_overrides = overrides if overrides else None,
        skip_ablation = skip_abl,
    )
