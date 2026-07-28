#!/usr/bin/env python3
"""
Full pipeline runner with per-case parameter control.

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --cases venezuela iran
    python scripts/run_full_pipeline.py --cases venezuela --skip-embeddings
    python scripts/run_full_pipeline.py --cases russia --model llama --skip-embeddings --skip-windows --skip-alignment --skip-themes
"""

import argparse
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
log = logging.getLogger(__name__)

# ── Per-case defaults ──────────────────────────────────────────────────────────
CASE_PARAMS = {
    "venezuela": dict(window=4,   step=2,  mcs=5, samples=2, eps=0.25, cap=10),
    "iran":      dict(window=24,  step=8,  mcs=5, samples=3, eps=0.25, cap=15),
    "russia":    dict(window=168, step=24, mcs=5, samples=8, eps=0.25, cap=25),
}

# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Full pipeline runner")
    p.add_argument("--cases", nargs="+", default=["venezuela", "iran", "russia"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--skip-embeddings",  action="store_true")
    p.add_argument("--skip-windows",     action="store_true")
    p.add_argument("--skip-alignment",   action="store_true")
    p.add_argument("--skip-themes",      action="store_true")
    p.add_argument("--skip-stance",      action="store_true")
    p.add_argument("--skip-metrics",     action="store_true")
    p.add_argument("--skip-plots",       action="store_true")
    p.add_argument("--similarity-threshold", type=float, default=0.70)

    # Per-case parameter overrides
    for case in CASE_PARAMS:
        p.add_argument(f"--{case}-window",   type=int)
        p.add_argument(f"--{case}-step",     type=int)
        p.add_argument(f"--{case}-mcs",      type=int)
        p.add_argument(f"--{case}-samples",  type=int)
        p.add_argument(f"--{case}-eps",      type=float)
        p.add_argument(f"--{case}-cap",      type=int)

    return p.parse_args()

# ── Runner ─────────────────────────────────────────────────────────────────────
def run(description, cmd, log_file=None):
    log.info(f"START: {description}")
    log.info(f"  CMD: {' '.join(str(c) for c in cmd)}")
    
    try:
        result = subprocess.run(
            [str(c) for c in cmd],
            capture_output=True,
            text=True
        )
        
        if log_file:
            Path(log_file).write_text(result.stdout + "\n" + result.stderr)
        
        if result.returncode != 0:
            log.error(f"FAILED: {description}")
            log.error(f"  STDOUT: {result.stdout[-2000:] if result.stdout else '(empty)'}")
            log.error(f"  STDERR: {result.stderr[-2000:] if result.stderr else '(empty)'}")
            return False
        
        # Print last few lines of output for progress visibility
        output_lines = (result.stdout or "").strip().split("\n")
        for line in output_lines[-5:]:
            if line.strip():
                log.info(f"  {line}")
        
        log.info(f"DONE:  {description}")
        return True

    except Exception as e:
        log.error(f"EXCEPTION in {description}: {e}")
        return False

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Apply per-case overrides
    params = {}
    for case in args.cases:
        if case not in CASE_PARAMS:
            log.error(f"Unknown case: {case}")
            sys.exit(1)
        params[case] = dict(CASE_PARAMS[case])
        for key in ["window", "step", "mcs", "samples", "eps", "cap"]:
            val = getattr(args, f"{case}_{key}", None)
            if val is not None:
                params[case][key] = val

    log.info("=" * 60)
    log.info(f"Pipeline starting — cases: {args.cases}")
    log.info("=" * 60)

    results = {}

    for case in args.cases:
        p = params[case]
        log.info("=" * 60)
        log.info(f"Processing case: {case.upper()}")
        log.info(f"  window={p['window']}h  step={p['step']}h  mcs={p['mcs']}  "
                 f"samples={p['samples']}  eps={p['eps']}  cap={p['cap']}")
        log.info("=" * 60)

        case_results = {}

        # Stage 1: Prepare data
        ok = run(
            f"{case}: prepare data",
            ["python", "scripts/prepare_processed_data.py", "--case", case],
            log_file=f"logs/{case}_prepare.log"
        )
        case_results["prepare"] = ok
        if not ok:
            log.error(f"Stopping {case} pipeline at prepare stage")
            results[case] = case_results
            continue

        # Stage 2: Embeddings
        if not args.skip_embeddings:
            ok = run(
                f"{case}: build embeddings",
                ["python", "scripts/build_representation.py",
                 "--case", case, "--device", args.device],
                log_file=f"logs/{case}_embeddings.log"
            )
            case_results["embeddings"] = ok
            if not ok:
                log.error(f"Stopping {case} pipeline at embeddings stage")
                results[case] = case_results
                continue
        else:
            log.info(f"SKIP: {case} embeddings")

        # Stage 3: Rolling windows — clear old outputs first
        if not args.skip_windows:
            old_files = list(Path(f"data/evaluated/{case}").glob("*.parquet"))
            if old_files:
                for f in old_files:
                    f.unlink()
                log.info(f"  Cleared {len(old_files)} old parquet files for {case}")

            ok = run(
                f"{case}: rolling windows",
                ["python", "scripts/run_rolling_windows.py",
                 "--case", case,
                 "--window-hours", p["window"],
                 "--step-hours",   p["step"],
                 "--min-cluster-size", p["mcs"],
                 "--min-samples",  p["samples"],
                 "--cluster-epsilon", p["eps"],
                 "--mcs-cap",      p["cap"]],
                log_file=f"logs/{case}_windows.log"
            )
            case_results["windows"] = ok
            if not ok:
                log.error(f"Stopping {case} pipeline at windows stage")
                results[case] = case_results
                continue
        else:
            log.info(f"SKIP: {case} rolling windows")

        # Stage 4: Alignment
        if not args.skip_alignment:
            ok = run(
                f"{case}: alignment",
                ["python", "scripts/run_alignment.py", "--case", case],
                log_file=f"logs/{case}_alignment.log"
            )
            case_results["alignment"] = ok
            if not ok:
                log.error(f"Stopping {case} pipeline at alignment stage")
                results[case] = case_results
                continue
        else:
            log.info(f"SKIP: {case} alignment")

        # Stage 5: Theme labeling
        if not args.skip_themes:
            ok = run(
                f"{case}: theme labeling",
                ["python", "scripts/run_theme_labeling.py", "--case", case],
                log_file=f"logs/{case}_themes.log"
            )
            case_results["themes"] = ok
            if not ok:
                log.error(f"Stopping {case} pipeline at themes stage")
                results[case] = case_results
                continue
        else:
            log.info(f"SKIP: {case} themes")

        # Stage 6: Stance classification
        if not args.skip_stance:
            for mode in ["cluster", "topic"]:
                cmd = ["python", "scripts/run_stance_classification.py",
                       "--case", case,
                       "--mode", mode,
                       "--model", args.model,
                       "--overwrite"]
                ok = run(
                    f"{case}: stance {mode}",
                    cmd,
                    log_file=f"logs/{case}_stance_{mode}.log"
                )
                case_results[f"stance_{mode}"] = ok
                if not ok:
                    log.error(f"Stopping {case} pipeline at stance {mode} stage")
                    break
            if not ok:
                results[case] = case_results
                continue
        else:
            log.info(f"SKIP: {case} stance")

        # Stage 7: Recompute metrics
        if not args.skip_metrics:
            ok = run(
                f"{case}: recompute metrics",
                ["python", "scripts/recompute_stance_metrics.py", "--case", case],
                log_file=f"logs/{case}_metrics.log"
            )
            case_results["metrics"] = ok
        else:
            log.info(f"SKIP: {case} metrics")

        results[case] = case_results
        log.info(f"COMPLETE: {case.upper()}")

    # Stage 8: Plots
    if not args.skip_plots:
        ok = run(
            "generate comparison plots",
            ["python", "analysis/compare_cases.py"],
            log_file="logs/plots.log"
        )
        results["plots"] = ok

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 60)
    for case, stage_results in results.items():
        if isinstance(stage_results, dict):
            failed = [s for s, ok in stage_results.items() if not ok]
            status = "✓ ALL OK" if not failed else f"✗ FAILED: {failed}"
            log.info(f"  {case:<12} {status}")
        else:
            log.info(f"  {case:<12} {'✓' if stage_results else '✗'}")

if __name__ == "__main__":
    main()
