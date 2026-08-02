"""Tests for the baseline comparison script's mechanics.

Deliberately does not assert which policy wins at what -- that is a research
finding, reported in docs/RESEARCH_FINDINGS.md from a specific, larger run,
and it should be free to shift as the underlying code changes without a test
suite treating a legitimate research update as a regression. What these
pin down is that the comparison itself is trustworthy: every policy runs
through the loop without crashing, results are well-formed, and it is
reproducible given a fixed seed -- the actual prerequisites for any
comparison drawn from it to mean something.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_baseline_comparison.py"


def run_script(*args: str, timeout: float = 90.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, timeout=timeout,
    )


def test_all_four_policies_run_without_crashing(tmp_path):
    result = run_script("--episodes", "3", "--seeds", "2",
                        "--output", str(tmp_path / "comparison.json"))
    assert result.returncode == 0, result.stderr
    for name in ("random", "novelty", "uncertainty_only", "lama"):
        assert name in result.stdout


def test_reports_the_headline_secondary_recall_column(tmp_path):
    result = run_script("--episodes", "3", "--seeds", "2",
                        "--output", str(tmp_path / "comparison.json"))
    assert "secondary_recall" in result.stdout


def test_writes_well_formed_json_output(tmp_path):
    output = tmp_path / "comparison.json"
    result = run_script("--episodes", "3", "--seeds", "2", "--output", str(output))
    assert result.returncode == 0, result.stderr
    assert output.exists()

    import json
    data = json.loads(output.read_text())
    assert set(data) == {"config", "raw_runs", "summary"}
    for policy in ("random", "novelty", "uncertainty_only", "lama"):
        assert policy in data["summary"]
        assert len(data["raw_runs"][policy]) == 2  # --seeds 2
        for metric in ("precision", "recall", "secondary_recall"):
            assert 0.0 <= data["summary"][policy][metric]["mean"] <= 1.0


def test_is_deterministic_given_the_same_seeds(tmp_path):
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    run_script("--episodes", "3", "--seeds", "2", "--output", str(out_a))
    run_script("--episodes", "3", "--seeds", "2", "--output", str(out_b))
    import json
    assert json.loads(out_a.read_text())["raw_runs"] == json.loads(out_b.read_text())["raw_runs"]
