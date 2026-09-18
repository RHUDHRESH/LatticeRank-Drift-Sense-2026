from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

import register


def _pairs(path: Path, ids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("pair_id", "reference_path", "search_path"))
        writer.writeheader()
        for pair_id in ids:
            writer.writerow({"pair_id": pair_id, "reference_path": "r.png", "search_path": "s.png"})


def test_phase2_output_schema_and_duplicate_ids(tmp_path: Path, monkeypatch) -> None:
    source, output = tmp_path / "pairs.csv", tmp_path / "predictions.csv"
    _pairs(source, ["a", "a", "b"])
    monkeypatch.setattr(register, "load_image", lambda _path: np.zeros((32, 32), np.uint8))
    monkeypatch.setattr(register, "solve_edge", lambda *_args: {
        "x": 4.5, "y": 7.5, "theta": 1.0, "scale": 10.0,
        "found": 1, "score": 0.8,
    })
    assert register.main(["--input", str(source), "--output", str(output)]) == 0
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert handle.closed is False
    assert list(rows[0]) == list(register.COLUMNS)
    assert [row["pair_id"] for row in rows] == ["a", "b"]


def test_phase2_failure_is_finite_and_zeroed(tmp_path: Path, monkeypatch) -> None:
    source, output = tmp_path / "pairs.csv", tmp_path / "predictions.csv"
    _pairs(source, ["broken"])
    monkeypatch.setattr(register, "load_image", lambda _path: np.zeros((32, 32), np.uint8))
    monkeypatch.setattr(register, "solve_edge", lambda *_args: (_ for _ in ()).throw(ValueError("bad")))
    assert register.main(["--input", str(source), "--output", str(output)]) == 0
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["found"] == "0"
    assert all(float(row[key]) == 0.0 for key in ("x", "y", "theta", "scale"))
    assert math.isfinite(float(row["score"]))


def test_import_surface_has_no_training_stack() -> None:
    project = str(Path(__file__).resolve().parents[1])
    code = (f"import sys; sys.path.insert(0, {project!r}); import register; "
            "assert not any(x == 'sklearn' or x.startswith('sklearn.') for x in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_judge_facing_phase_folder_exposes_entry_point() -> None:
    # This phase ships standalone, so the contract is checked against its own
    # entry point only -- nothing here may depend on a sibling phase folder.
    project = Path(__file__).resolve().parents[1]
    for relative in ("register.py",):
        result = subprocess.run(
            [sys.executable, str(project / relative), "--help"],
            cwd=project, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "--input" in result.stdout
        assert "--output" in result.stdout
