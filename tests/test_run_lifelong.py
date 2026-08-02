"""Tests for the lifelong self-learning driver script.

Run as a subprocess (scripts/ is not a package): this is the same interface
a user actually invokes, so a test passing here means the CLI genuinely
works, not just the functions behind it in isolation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_lifelong.py"


def run_script(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, timeout=timeout,
    )


def test_runs_without_crashing_over_a_few_episodes():
    result = run_script("--episodes", "6", "--seed", "0", "--block", "3")
    assert result.returncode == 0, result.stderr
    assert "confirmed" in result.stdout.lower()


def test_reports_the_summary_table_and_final_counts():
    result = run_script("--episodes", "6", "--seed", "0", "--block", "3")
    assert "interactions/new" in result.stdout
    assert "Goal-directed pursuit began at episode:" in result.stdout
    assert "concepts formed" in result.stdout


def test_is_deterministic_given_the_same_seed():
    a = run_script("--episodes", "5", "--seed", "1", "--block", "5")
    b = run_script("--episodes", "5", "--seed", "1", "--block", "5")
    assert a.stdout == b.stdout


def test_different_seeds_diverge():
    a = run_script("--episodes", "5", "--seed", "1", "--block", "5")
    b = run_script("--episodes", "5", "--seed", "2", "--block", "5")
    assert a.stdout != b.stdout
