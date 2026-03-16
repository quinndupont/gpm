#!/usr/bin/env python3
"""Lightweight timing infrastructure for training loops.

Provides hierarchical timing with minimal overhead for tracking training performance.
"""
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Optional


class TrainingTimer:
    """Hierarchical timer with running statistics for training phases."""

    def __init__(self):
        self.timings: Dict[str, list[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        self.epoch_start: Optional[float] = None
        self.step_start: Optional[float] = None
        self.global_start: Optional[float] = None

    @contextmanager
    def phase(self, name: str):
        """Time a phase of computation with context manager.

        Args:
            name: Name of the phase (e.g., "generation", "scoring", "gradients")

        Example:
            with timer.phase("generation"):
                outputs = model.generate(...)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[name].append(elapsed)

    def start_epoch(self):
        """Mark the start of a new epoch."""
        self.epoch_start = time.perf_counter()
        if self.global_start is None:
            self.global_start = self.epoch_start

    def start_step(self):
        """Mark the start of a new training step."""
        self.step_start = time.perf_counter()

    def get_avg(self, name: str, last_n: int = 100) -> float:
        """Get average time for phase over last N measurements.

        Args:
            name: Name of the phase
            last_n: Number of recent measurements to average (default: 100)

        Returns:
            Average time in seconds, or 0.0 if no measurements exist
        """
        times = self.timings[name]
        if not times:
            return 0.0
        return sum(times[-last_n:]) / min(len(times), last_n)

    def get_total(self, name: str) -> float:
        """Get total cumulative time for a phase.

        Args:
            name: Name of the phase

        Returns:
            Total time in seconds
        """
        return sum(self.timings[name])

    def format_summary(
        self,
        step: int,
        total_steps: int,
        metrics: dict,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> str:
        """Format timing summary for logging.

        Args:
            step: Current step number
            total_steps: Total steps in epoch
            metrics: Dictionary of training metrics (loss, reward, kl, etc.)
            epoch: Current epoch number (optional)
            total_epochs: Total epochs (optional)

        Returns:
            Formatted string with metrics and timing information

        Example output:
            [1/2] Step 10/764 | loss=-8.2625 | reward=0.643 | kl=-1.1061 |
            gen=4.32s | score=0.18s | grad=2.45s | 7.12s/step (0.140 steps/s) | ETA 89m
        """
        parts = []

        # Epoch info (if provided)
        if epoch is not None and total_epochs is not None:
            parts.append(f"[{epoch}/{total_epochs}]")

        # Step info
        parts.append(f"step {step}/{total_steps}")

        # Training metrics
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")

        # Timing breakdown for major phases
        gen_time = self.get_avg("generation", last_n=10)
        score_time = self.get_avg("scoring", last_n=10)
        grad_time = self.get_avg("gradients", last_n=10)

        timing_parts = []
        if gen_time > 0:
            timing_parts.append(f"gen={gen_time:.2f}s")
        if score_time > 0:
            timing_parts.append(f"score={score_time:.2f}s")
        if grad_time > 0:
            timing_parts.append(f"grad={grad_time:.2f}s")

        if timing_parts:
            parts.append(" | ".join(timing_parts))

        # Step throughput
        if self.step_start:
            step_time = time.perf_counter() - self.step_start
            if step_time > 0:
                steps_per_sec = 1.0 / step_time
                parts.append(f"{step_time:.2f}s/step ({steps_per_sec:.3f} steps/s)")

        # ETA calculation
        if self.epoch_start and step > 0:
            elapsed = time.perf_counter() - self.epoch_start
            avg_step_time = elapsed / step
            remaining_steps = total_steps - step
            eta_sec = avg_step_time * remaining_steps
            if eta_sec >= 60:
                eta_min = int(eta_sec / 60)
                parts.append(f"ETA {eta_min}m")
            else:
                parts.append(f"ETA {int(eta_sec)}s")

        return " | ".join(parts)

    def format_epoch_summary(self, epoch: int, total_epochs: int, metrics: dict) -> str:
        """Format end-of-epoch summary with total timing.

        Args:
            epoch: Current epoch number
            total_epochs: Total epochs
            metrics: Dictionary of epoch-level metrics

        Returns:
            Formatted string with epoch metrics and total time

        Example output:
            Epoch 1/2 complete | loss=-8.2156 | reward=0.652 | kl=-1.0983 |
            epoch_time=89.5m | total_time=89.5m
        """
        parts = [f"Epoch {epoch}/{total_epochs} complete"]

        # Metrics
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")

        # Epoch timing
        if self.epoch_start:
            epoch_time = time.perf_counter() - self.epoch_start
            if epoch_time >= 60:
                parts.append(f"epoch_time={epoch_time/60:.1f}m")
            else:
                parts.append(f"epoch_time={epoch_time:.1f}s")

        # Total training time
        if self.global_start:
            total_time = time.perf_counter() - self.global_start
            if total_time >= 60:
                parts.append(f"total_time={total_time/60:.1f}m")
            else:
                parts.append(f"total_time={total_time:.1f}s")

        return " | ".join(parts)

    def get_phase_breakdown(self) -> dict:
        """Get percentage breakdown of time spent in each phase.

        Returns:
            Dictionary mapping phase names to percentage of total time
        """
        total_time = sum(sum(times) for times in self.timings.values())
        if total_time == 0:
            return {}

        breakdown = {}
        for phase, times in self.timings.items():
            phase_total = sum(times)
            breakdown[phase] = (phase_total / total_time) * 100

        return breakdown
