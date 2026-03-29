#!/usr/bin/env python3
"""
Validation script for comparing training convergence between the legacy
(Foxy + PyTorch 1.10) and modernized (Jazzy + PyTorch 2.7 + Zenoh) stacks.

Reads training log CSV files, computes key metrics, generates comparison
plots, and produces a pass/fail summary report.

Usage:
    # Compare a new DDPG run against the baseline example:
    python3 util/validate_convergence.py \
        --baseline examples/ddpg_0_stage9 \
        --candidate ddpg_0 \
        --interval 100

    # Compare multiple algorithms:
    python3 util/validate_convergence.py \
        --baseline examples/ddpg_0_stage9 examples/td3_0_stage9 \
        --candidate ddpg_0 td3_0 \
        --interval 100

    # Custom tolerance (default: candidate must reach 90% of baseline peak reward):
    python3 util/validate_convergence.py \
        --baseline examples/ddpg_0_stage9 \
        --candidate ddpg_0 \
        --tolerance 0.85
"""

import argparse
import glob
import os
import socket
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================================== #
#                               Configuration                                  #
# =========================================================================== #

DEFAULT_TOLERANCE = 0.90        # Candidate must reach 90% of baseline peak
DEFAULT_INTERVAL = 100          # Averaging window (episodes)
DEFAULT_MIN_EPISODES = 500      # Minimum episodes for a valid run
SUCCESS_RATE_WINDOW = 100       # Window for success rate calculation

OUTCOME_SUCCESS = 1
OUTCOME_COLLISION_WALL = 2
OUTCOME_COLLISION_OBSTACLE = 3
OUTCOME_TIMEOUT = 4
OUTCOME_TUMBLE = 5


# =========================================================================== #
#                             Helper Functions                                 #
# =========================================================================== #

def find_model_dir(model_name: str) -> str:
    """Resolve model directory path (handles examples/ and hostname-prefixed paths)."""
    base = os.getenv("DRLNAV_BASE_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_root = os.path.join(base, "src", "turtlebot3_drl", "model")

    if "examples" in model_name:
        return os.path.join(model_root, model_name)

    hostname_dir = os.path.join(model_root, socket.gethostname(), model_name)
    if os.path.isdir(hostname_dir):
        return hostname_dir

    direct_dir = os.path.join(model_root, model_name)
    if os.path.isdir(direct_dir):
        return direct_dir

    raise FileNotFoundError(f"Model directory not found for: {model_name}")


def load_training_log(model_name: str) -> pd.DataFrame:
    """Load the training log CSV for a model."""
    model_dir = find_model_dir(model_name)
    logfiles = glob.glob(os.path.join(model_dir, "_train_*.txt"))
    if not logfiles:
        raise FileNotFoundError(f"No training log found in: {model_dir}")
    if len(logfiles) > 1:
        print(f"  WARNING: Multiple training logs found for {model_name}, using latest")
        logfiles.sort(key=os.path.getmtime)
    df = pd.read_csv(logfiles[-1])
    df.columns = df.columns.str.strip()
    return df


def load_test_log(model_name: str) -> pd.DataFrame:
    """Load the most recent test log CSV for a model."""
    model_dir = find_model_dir(model_name)
    logfiles = glob.glob(os.path.join(model_dir, "_test_*.txt"))
    if not logfiles:
        return None
    logfiles.sort(key=os.path.getmtime)
    df = pd.read_csv(logfiles[-1])
    df.columns = df.columns.str.strip()
    return df


def compute_metrics(df: pd.DataFrame, interval: int) -> dict:
    """Compute key training metrics from a log DataFrame."""
    rewards = df["reward"].values
    successes = df["success"].values
    episodes = len(rewards)

    # Windowed average reward
    n_windows = episodes // interval
    avg_rewards = []
    for i in range(n_windows):
        window = rewards[i * interval : (i + 1) * interval]
        avg_rewards.append(np.mean(window))
    avg_rewards = np.array(avg_rewards)

    # Success rate (rolling window)
    success_rates = []
    for i in range(n_windows):
        window = successes[i * interval : (i + 1) * interval]
        rate = np.sum(window == OUTCOME_SUCCESS) / len(window) * 100
        success_rates.append(rate)
    success_rates = np.array(success_rates)

    # Peak metrics
    peak_reward = np.max(avg_rewards) if len(avg_rewards) > 0 else 0.0
    peak_reward_ep = (np.argmax(avg_rewards) + 1) * interval if len(avg_rewards) > 0 else 0
    peak_success = np.max(success_rates) if len(success_rates) > 0 else 0.0
    peak_success_ep = (np.argmax(success_rates) + 1) * interval if len(success_rates) > 0 else 0

    # Final window metrics (last 10% of training)
    final_n = max(1, len(avg_rewards) // 10)
    final_reward = np.mean(avg_rewards[-final_n:]) if len(avg_rewards) > 0 else 0.0
    final_success = np.mean(success_rates[-final_n:]) if len(success_rates) > 0 else 0.0

    return {
        "episodes": episodes,
        "avg_rewards": avg_rewards,
        "success_rates": success_rates,
        "peak_reward": peak_reward,
        "peak_reward_episode": peak_reward_ep,
        "peak_success_rate": peak_success,
        "peak_success_episode": peak_success_ep,
        "final_avg_reward": final_reward,
        "final_avg_success": final_success,
    }


def compute_test_metrics(df: pd.DataFrame) -> dict:
    """Compute test metrics from a test log DataFrame."""
    if df is None:
        return None
    outcomes = df["outcome"].values
    total = len(outcomes)
    return {
        "total_episodes": total,
        "success_rate": np.sum(outcomes == OUTCOME_SUCCESS) / total * 100,
        "collision_wall_rate": np.sum(outcomes == OUTCOME_COLLISION_WALL) / total * 100,
        "collision_obs_rate": np.sum(outcomes == OUTCOME_COLLISION_OBSTACLE) / total * 100,
        "timeout_rate": np.sum(outcomes == OUTCOME_TIMEOUT) / total * 100,
    }


# =========================================================================== #
#                            Plotting Functions                                 #
# =========================================================================== #

def plot_comparison(baselines: dict, candidates: dict, interval: int, output_path: str):
    """Generate side-by-side reward and success rate comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Reward plot
    ax_reward = axes[0]
    for name, metrics in baselines.items():
        x = np.arange(1, len(metrics["avg_rewards"]) + 1) * interval
        ax_reward.plot(x, metrics["avg_rewards"], label=f"{name} (baseline)", linestyle="--", alpha=0.7)
    for name, metrics in candidates.items():
        x = np.arange(1, len(metrics["avg_rewards"]) + 1) * interval
        ax_reward.plot(x, metrics["avg_rewards"], label=f"{name} (candidate)", linewidth=2)

    ax_reward.set_xlabel("Episode", fontsize=16, fontweight="bold")
    ax_reward.set_ylabel("Average Reward", fontsize=16, fontweight="bold")
    ax_reward.set_title("Reward Convergence", fontsize=18, fontweight="bold")
    ax_reward.legend(fontsize=12)
    ax_reward.grid(True, linestyle="--", alpha=0.5)
    ax_reward.tick_params(labelsize=12)

    # Success rate plot
    ax_success = axes[1]
    for name, metrics in baselines.items():
        x = np.arange(1, len(metrics["success_rates"]) + 1) * interval
        ax_success.plot(x, metrics["success_rates"], label=f"{name} (baseline)", linestyle="--", alpha=0.7)
    for name, metrics in candidates.items():
        x = np.arange(1, len(metrics["success_rates"]) + 1) * interval
        ax_success.plot(x, metrics["success_rates"], label=f"{name} (candidate)", linewidth=2)

    ax_success.set_xlabel("Episode", fontsize=16, fontweight="bold")
    ax_success.set_ylabel("Success Rate (%)", fontsize=16, fontweight="bold")
    ax_success.set_title("Success Rate Convergence", fontsize=18, fontweight="bold")
    ax_success.legend(fontsize=12)
    ax_success.grid(True, linestyle="--", alpha=0.5)
    ax_success.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {output_path}")
    plt.close()


# =========================================================================== #
#                           Validation Logic                                   #
# =========================================================================== #

def validate_pair(baseline_name: str, candidate_name: str,
                  baseline_metrics: dict, candidate_metrics: dict,
                  tolerance: float) -> dict:
    """Validate a candidate model against a baseline. Returns pass/fail results."""
    results = {"model": candidate_name, "baseline": baseline_name, "checks": []}

    # Check 1: Minimum episodes
    min_eps = DEFAULT_MIN_EPISODES
    passed = candidate_metrics["episodes"] >= min_eps
    results["checks"].append({
        "name": "Minimum episodes",
        "passed": passed,
        "detail": f"{candidate_metrics['episodes']} episodes (need >= {min_eps})",
    })

    # Check 2: Peak reward within tolerance of baseline
    baseline_peak = baseline_metrics["peak_reward"]
    candidate_peak = candidate_metrics["peak_reward"]
    threshold = baseline_peak * tolerance
    passed = candidate_peak >= threshold
    results["checks"].append({
        "name": "Peak reward",
        "passed": passed,
        "detail": f"candidate={candidate_peak:.1f}, baseline={baseline_peak:.1f}, "
                  f"threshold={threshold:.1f} ({tolerance:.0%})",
    })

    # Check 3: Final reward within tolerance
    baseline_final = baseline_metrics["final_avg_reward"]
    candidate_final = candidate_metrics["final_avg_reward"]
    threshold = baseline_final * tolerance
    passed = candidate_final >= threshold
    results["checks"].append({
        "name": "Final avg reward",
        "passed": passed,
        "detail": f"candidate={candidate_final:.1f}, baseline={baseline_final:.1f}, "
                  f"threshold={threshold:.1f} ({tolerance:.0%})",
    })

    # Check 4: Peak success rate within tolerance
    baseline_sr = baseline_metrics["peak_success_rate"]
    candidate_sr = candidate_metrics["peak_success_rate"]
    threshold = baseline_sr * tolerance
    passed = candidate_sr >= threshold
    results["checks"].append({
        "name": "Peak success rate",
        "passed": passed,
        "detail": f"candidate={candidate_sr:.1f}%, baseline={baseline_sr:.1f}%, "
                  f"threshold={threshold:.1f}%",
    })

    results["overall_passed"] = all(c["passed"] for c in results["checks"])
    return results


# =========================================================================== #
#                              Report Generation                               #
# =========================================================================== #

def print_report(all_results: list, baselines: dict, candidates: dict,
                 output_path: str = None):
    """Print (and optionally save) the validation report."""
    lines = []
    lines.append("=" * 72)
    lines.append("  TRAINING CONVERGENCE VALIDATION REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)

    # Summary table
    lines.append("\n--- Metrics Summary ---\n")
    lines.append(f"{'Model':<30} {'Episodes':>8} {'Peak Reward':>12} {'Peak SR%':>9} {'Final Reward':>13} {'Final SR%':>9}")
    lines.append("-" * 85)
    for name, m in {**baselines, **candidates}.items():
        tag = " (baseline)" if name in baselines else " (candidate)"
        lines.append(
            f"{name + tag:<30} {m['episodes']:>8} {m['peak_reward']:>12.1f} "
            f"{m['peak_success_rate']:>8.1f}% {m['final_avg_reward']:>13.1f} "
            f"{m['final_avg_success']:>8.1f}%"
        )

    # Validation results
    lines.append("\n--- Validation Checks ---\n")
    all_passed = True
    for result in all_results:
        status = "PASS" if result["overall_passed"] else "FAIL"
        all_passed = all_passed and result["overall_passed"]
        lines.append(f"[{status}] {result['model']} vs {result['baseline']}")
        for check in result["checks"]:
            icon = "  [+]" if check["passed"] else "  [-]"
            lines.append(f"  {icon} {check['name']}: {check['detail']}")
        lines.append("")

    # Overall verdict
    lines.append("=" * 72)
    verdict = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
    lines.append(f"  VERDICT: {verdict}")
    lines.append("=" * 72)

    report = "\n".join(lines)
    print(report)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")

    return all_passed


# =========================================================================== #
#                                   Main                                       #
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Validate DRL training convergence against baseline models"
    )
    parser.add_argument(
        "--baseline", nargs="+", required=True,
        help="Baseline model name(s) (e.g. examples/ddpg_0_stage9)"
    )
    parser.add_argument(
        "--candidate", nargs="+", required=True,
        help="Candidate model name(s) to validate"
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help=f"Averaging interval in episodes (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help=f"Fraction of baseline performance candidate must reach (default: {DEFAULT_TOLERANCE})"
    )
    parser.add_argument(
        "--output-dir", default="util/graphs",
        help="Directory for output plots and reports"
    )
    args = parser.parse_args()

    if len(args.baseline) != len(args.candidate):
        if len(args.baseline) == 1:
            args.baseline = args.baseline * len(args.candidate)
        else:
            sys.exit("ERROR: --baseline and --candidate must have the same number of entries "
                     "(or provide a single baseline to compare all candidates against)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and compute metrics
    baselines = {}
    candidates = {}
    for bname in set(args.baseline):
        print(f"Loading baseline: {bname}")
        df = load_training_log(bname)
        baselines[bname] = compute_metrics(df, args.interval)

    for cname in args.candidate:
        print(f"Loading candidate: {cname}")
        df = load_training_log(cname)
        candidates[cname] = compute_metrics(df, args.interval)

    # Validate each pair
    all_results = []
    for bname, cname in zip(args.baseline, args.candidate):
        result = validate_pair(bname, cname, baselines[bname], candidates[cname], args.tolerance)
        all_results.append(result)

    # Generate comparison plot
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    plot_path = os.path.join(args.output_dir, f"validation_{dt}.png")
    plot_comparison(baselines, candidates, args.interval, plot_path)

    # Generate report
    report_path = os.path.join(args.output_dir, f"validation_{dt}.txt")
    passed = print_report(all_results, baselines, candidates, report_path)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
