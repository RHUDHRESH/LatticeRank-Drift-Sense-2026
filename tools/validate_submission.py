#!/usr/bin/env python3
"""Prove a phase folder is submittable before it is pushed.

This runs the phase the way the organizers said they run it -- one entry
point, the published arguments, paths relative to the dataset root, the
blind-split pairs.csv whose last two columns are empty, and no network --
then checks predictions.csv against the published output contract.

    python tools/validate_submission.py --phase 3 \
        --dataset <root containing pairs.csv>

Checks, in order:

  1  the phase folder is self-contained          (copied alone to a temp dir)
  2  requirements.txt pins every dependency      (== on every requirement)
  3  the entry point answers --help              (exit 0)
  4  it runs on a BLIND pairs.csv                (reference_sem_path and
                                                  params_json_path emptied)
  5  it runs with the network disabled           (socket blocked in-process)
  6  predictions.csv has the exact header        pair_id,x,y,theta,scale,found,score
  7  one row per pair_id, exactly once           (a missing row scores zero)
  8  every value is finite and in range          found in {0,1}, score in [0,1]
  9  rejected rows are zeroed                    x=y=theta=scale=0 when found=0
 10  a second run reproduces every scored field  (well inside the rubric bands)

Exit status is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CONTRACT = ["pair_id", "x", "y", "theta", "scale", "found", "score"]
BLIND_COLUMNS = ("reference_sem_path", "params_json_path")
ENTRY = {2: "register.py", 3: "phase3.py"}

# Dropped into the working directory of the scored run. Python imports
# sitecustomize automatically at start-up, so this disables outbound
# networking for the entry point and everything it imports.
NO_NETWORK = """
import socket


class _Blocked(socket.socket):
    def __init__(self, *a, **k):
        raise OSError("network access is disabled during validation")


socket.socket = _Blocked
socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(
    OSError("network access is disabled during validation"))
"""


class Report:
    def __init__(self):
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((bool(ok), name, detail))
        return bool(ok)

    def render(self) -> bool:
        width = max(len(name) for _, name, _ in self.rows)
        print()
        for ok, name, detail in self.rows:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name.ljust(width)}  {detail}")
        passed = sum(ok for ok, _, _ in self.rows)
        good = passed == len(self.rows)
        print(f"\n  {passed}/{len(self.rows)} checks passed -- "
              f"{'SUBMITTABLE' if good else 'NOT SUBMITTABLE'}\n")
        return good


def blind_copy(pairs: Path, destination: Path) -> int:
    """Rewrite pairs.csv with the training-only columns emptied, exactly as
    the blind split ships. Code that needs them fails here, not on the
    scored run."""
    with pairs.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        for column in BLIND_COLUMNS:
            if column in row:
                row[column] = ""
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run(command, cwd, env=None, timeout=1800):
    return subprocess.run(command, cwd=str(cwd), env=env, timeout=timeout,
                          capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", type=int, choices=(2, 3), required=True)
    ap.add_argument("--dataset", type=Path, required=True,
                    help="dataset root containing pairs.csv")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--keep", action="store_true", help="keep the temp workspace")
    args = ap.parse_args()

    phase_dir = args.repo / f"phase_{args.phase}"
    entry = ENTRY[args.phase]
    report = Report()

    if not report.check(phase_dir.is_dir(), "phase folder exists", str(phase_dir)):
        return 1 if not report.render() else 0

    workspace = Path(tempfile.mkdtemp(prefix=f"validate_phase{args.phase}_"))
    try:
        # 1 -- self-contained: only this phase folder is copied
        sandbox = workspace / f"phase_{args.phase}"
        shutil.copytree(phase_dir, sandbox,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "assets"))
        report.check((sandbox / entry).is_file(), "self-contained phase folder",
                     f"{entry} present, copied without the rest of the repository")

        # 2 -- pinned dependencies
        req = sandbox / "requirements.txt"
        lines = [l.strip() for l in req.read_text().splitlines()
                 if l.strip() and not l.startswith("#")] if req.is_file() else []
        unpinned = [l for l in lines if "==" not in l]
        report.check(bool(lines) and not unpinned, "dependencies pinned",
                     f"{len(lines)} requirements" + (f", unpinned: {unpinned}" if unpinned else ""))

        # 3 -- the entry point answers --help
        helped = run([args.python, entry, "--help"], sandbox)
        report.check(helped.returncode == 0, "entry point --help",
                     f"exit {helped.returncode}")

        # 4/5 -- blind pairs.csv, network disabled
        dataset = workspace / "dataset"
        shutil.copytree(args.dataset, dataset,
                        ignore=shutil.ignore_patterns("ground_truth.csv", "manifest_jury.csv",
                                                      "*.json", "predictions*.csv"))
        n_pairs = blind_copy(args.dataset / "pairs.csv", dataset / "pairs.csv")
        (dataset / "sitecustomize.py").write_text(NO_NETWORK)

        env = dict(**{k: v for k, v in __import__("os").environ.items()})
        env["PYTHONPATH"] = str(dataset)          # picks up sitecustomize.py
        entry_abs = sandbox / entry

        first = run([args.python, str(entry_abs), "--input", "pairs.csv",
                     "--output", "predictions.csv"], dataset, env)
        report.check(first.returncode == 0,
                     "runs on the blind split",
                     f"exit {first.returncode}; {n_pairs} pairs; "
                     f"{BLIND_COLUMNS[0]} and {BLIND_COLUMNS[1]} empty")
        report.check("network access is disabled" not in (first.stderr or ""),
                     "no network at run time", "socket blocked for the whole run")
        if first.returncode != 0:
            print((first.stderr or "")[-2000:], file=sys.stderr)
            return 0 if report.render() else 1

        # 6/7/8/9 -- the output contract
        out = dataset / "predictions.csv"
        with out.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            rows = list(reader)
        report.check(header == CONTRACT, "exact output header", ",".join(header))

        wanted = [r["pair_id"] for r in csv.DictReader(
            (args.dataset / "pairs.csv").open(newline="", encoding="utf-8-sig"))]
        got = [r["pair_id"] for r in rows]
        report.check(got == wanted or sorted(got) == sorted(wanted),
                     "one row per pair_id",
                     f"{len(got)} rows for {len(wanted)} pairs; "
                     f"missing={sorted(set(wanted) - set(got)) or 'none'}; "
                     f"duplicate={sorted({p for p in got if got.count(p) > 1}) or 'none'}")

        bad_value, bad_zero = [], []
        for r in rows:
            try:
                values = {k: float(r[k]) for k in ("x", "y", "theta", "scale", "score")}
                found = int(float(r["found"]))
            except (TypeError, ValueError):
                bad_value.append(r["pair_id"])
                continue
            if (any(not math.isfinite(v) for v in values.values())
                    or found not in (0, 1) or not 0.0 <= values["score"] <= 1.0):
                bad_value.append(r["pair_id"])
            elif not found and any(values[k] != 0.0 for k in ("x", "y", "theta", "scale")):
                bad_zero.append(r["pair_id"])
        report.check(not bad_value, "values finite and in range",
                     f"found in {{0,1}}, score in [0,1]" if not bad_value else f"offenders {bad_value}")
        report.check(not bad_zero, "rejected rows zeroed",
                     "x=y=theta=scale=0 when found=0" if not bad_zero else f"offenders {bad_zero}")

        # 10 -- determinism
        second = run([args.python, str(entry_abs), "--input", "pairs.csv",
                      "--output", "predictions_again.csv"], dataset, env)
        again = list(csv.DictReader(
            (dataset / "predictions_again.csv").open(newline="", encoding="utf-8-sig")))
        # Tolerances sit far below the finest band the rubric can see: its
        # tightest localization tier is 1 px and its published pose tolerances
        # are 1% of scale and 0.25 deg. A disagreement under these bounds
        # cannot move a single point. `found` must agree exactly.
        TOL = {"x": 1e-2, "y": 1e-2, "theta": 1e-2, "scale": 1e-4}
        by_id = {r["pair_id"]: r for r in again}
        drift, mismatched, worst = 0.0, [], {}
        for r in rows:
            other = by_id.get(r["pair_id"])
            if other is None or r["found"] != other["found"]:
                mismatched.append(r["pair_id"])
                continue
            for field, bound in TOL.items():
                delta = abs(float(r[field]) - float(other[field]))
                worst[field] = max(worst.get(field, 0.0), delta)
                if delta > bound:
                    mismatched.append(r["pair_id"])
                    break
            drift = max(drift, abs(float(r["score"]) - float(other["score"])))
        # Every field the rubric scores must repeat exactly. The confidence
        # column is a float reduction over many terms, so its last digits move
        # with summation order; a drift this small cannot change a ranking,
        # a threshold or a point, and we report it rather than hide it.
        report.check(second.returncode == 0 and not mismatched and drift < 1e-6,
                     "reproducible",
                     f"found identical; x/y within {max(worst.get('x',0), worst.get('y',0)):.1e} px, "
                     f"theta within {worst.get('theta', 0):.1e} deg, "
                     f"scale within {worst.get('scale', 0):.1e}, score within {drift:.1e}"
                     + (f"; mismatched {mismatched}" if mismatched else ""))

        ok = report.render()
        print(f"  entry point as the judges run it:  python {entry} "
              f"--input pairs.csv --output predictions.csv")
        if args.keep:
            print(f"  workspace kept at {workspace}")
        return 0 if ok else 1
    finally:
        if not args.keep:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
