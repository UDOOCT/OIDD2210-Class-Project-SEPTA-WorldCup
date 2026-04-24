"""
main.py
-------
Entry point for SEPTA World Cup optimization pipeline.

Usage:
  python main.py                    # full bilevel pipeline
  python main.py --mode upper_only  # upper level ILP only (simpler, like Excel model)
  python main.py --mode sensitivity # Optuna sensitivity sweep (OptQuest equivalent)
  python main.py --no-worldcup      # normal day optimization (no World Cup demand)
"""

import argparse
import os
import numpy as np
from data.demand import get_total_demand
from models.upper_level import solve as upper_solve, LNAMES, TBLOCKS, TBLOCK_RANGES
from models.bilevel import run_bilevel
from models.sensitivity import run_sensitivity


def print_results(result: dict):
    print("\n" + "="*70)
    print(f"  PROFIT:      ${result['profit']:>12,.0f}")
    print(f"  REVENUE:     ${result['revenue']:>12,.0f}")
    print(f"  FIXED COST:  ${result['fixed_cost']:>12,.0f}")
    print(f"  VAR COST:    ${result['var_cost']:>12,.0f}")
    print(f"  TOTAL PAX:   {result['total_pax']:>13,.0f}")
    print("="*70)
    print(f"\n{'Line':<25} {'Block':<10} {'Trains':>6} {'Avg Fare':>9} {'Pax':>8} {'Util':>6} {'Eq?':>4}")
    print("-"*70)
    for l in LNAMES:
        lr = result["lines"][l]
        for blk in TBLOCKS:
            idxs     = list(TBLOCK_RANGES[blk])
            f_blk    = lr["f"][idxs]
            p_blk    = lr["p"][idxs]
            x_blk    = lr["x"][idxs]
            util_blk = lr["util"][idxs]
            eq_blk   = lr["equity_ok"][idxs]
            total_f  = f_blk.sum()
            total_x  = x_blk.sum()
            avg_p    = float(np.average(p_blk, weights=np.maximum(x_blk, 1e-6)))
            avg_util = float(np.mean(util_blk))
            eq       = "✓" if bool(np.all(eq_blk)) else "✗"
            print(f"{l:<25} {blk:<10} {total_f:>6.0f} ${avg_p:>8.2f} {total_x:>8,.0f} "
                  f"{avg_util:>5.1%} {eq:>4}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["upper_only", "bilevel", "sensitivity"],
                        default="bilevel")
    parser.add_argument("--no-worldcup", action="store_true")
    args = parser.parse_args()

    print("SEPTA Regional Rail — World Cup 2026 Optimization")
    print(f"Mode: {args.mode} | World Cup demand: {not args.no_worldcup}")
    print("-"*50)

    demand = get_total_demand(worldcup=not args.no_worldcup)
    total_d = sum(demand[l].sum() for l in demand)
    print(f"Total demand across all lines/blocks: {total_d:,.0f} passengers")

    if args.mode == "upper_only":
        print("\nRunning upper-level ILP (scipy SLSQP)...")
        result = upper_solve(demand)
        print_results(result)

    elif args.mode == "bilevel":
        print("\nRunning bilevel optimization (iterative best-response)...")
        result = run_bilevel(demand, verbose=True)
        print_results(result)
        if "iterations" in result:
            print(f"\nConverged: {result['converged']} in {len(result['iterations'])} iterations")

    elif args.mode == "sensitivity":
        print("\nRunning sensitivity analysis (Optuna TPE, 200 trials × 100 MC scenarios)...")
        res = run_sensitivity(n_trials=200, n_mc_samples=100)
        bp = res["best_policy"]
        print(f"\nOptimal 4-block policy:")
        print(f"  Morning fare:  ${bp['fare_morning']:.2f}   freq: {bp['freq_morning']} trains/slot")
        print(f"  Midday  fare:  ${bp['fare_midday']:.2f}   freq: {bp['freq_midday']} trains/slot")
        print(f"  Evening fare:  ${bp['fare_evening']:.2f}   freq: {bp['freq_evening']} trains/slot")
        print(f"  Night   fare:  ${bp['fare_night']:.2f}   freq: {bp['freq_night']} trains/slot")
        print(f"  Expected profit: ${res['best_value']:,.0f}")
        os.makedirs("outputs", exist_ok=True)
        res["trials_df"].to_csv("outputs/sensitivity_results.csv", index=False)
        print("  Saved: outputs/sensitivity_results.csv")


if __name__ == "__main__":
    main()
