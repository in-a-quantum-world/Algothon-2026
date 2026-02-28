"""
IMC Full Pipeline Runner
=========================
  1. Generate synthetic data
  2. Run backtest on all strategies
  3. Run parameter optimisation
  4. Print full summary

Usage:
  cd imc/
  python run_pipeline.py
  python run_pipeline.py --steps 2000 --plot
"""

from __future__ import annotations
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="IMC Full Pipeline")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of synthetic timesteps")
    parser.add_argument("--plot", action="store_true",
                        help="Show PnL chart after backtest")
    parser.add_argument("--skip-optimise", action="store_true",
                        help="Skip parameter optimisation (faster)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "historical_data.csv")

    # ── Step 1: Generate data ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Generating synthetic data")
    print("=" * 60)
    from generate_data import generate
    generate(steps=args.steps, out_path=data_path)

    # ── Step 2: Backtest ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Running backtest")
    print("=" * 60)
    from backtester import Backtester
    bt = Backtester(data_path, verbose=False)
    pnl = bt.run()
    bt.print_summary()

    if args.plot:
        bt.plot()

    # ── Step 3: Optimise ──────────────────────────────────────────────
    if not args.skip_optimise:
        print("\n" + "=" * 60)
        print("STEP 3: Parameter optimisation")
        print("=" * 60)
        from optimiser import run_optimisation
        best = run_optimisation(data_path)

        # ── Step 4: Re-run backtest with best params ──────────────────
        print("\n" + "=" * 60)
        print("STEP 4: Backtest with best parameters")
        print("=" * 60)

        import trader as trader_mod

        if "FixedFairMM" in best:
            spread = best["FixedFairMM"]["spread"]
            for sym in trader_mod.FIXED_FAIR_PRODUCTS:
                trader_mod.FIXED_FAIR_PRODUCTS[sym]["spread"] = spread
            print(f"  Applied: FixedFairMM spread={spread}")

        if "RollingMidMM" in best:
            window = best["RollingMidMM"]["window"]
            for sym in trader_mod.ROLLING_MID_PRODUCTS:
                trader_mod.ROLLING_MID_PRODUCTS[sym]["window"] = window
            print(f"  Applied: RollingMidMM window={window}")

        if "SpreadArb" in best:
            z = best["SpreadArb"]["z_threshold"]
            for sym in trader_mod.BASKET_CONFIGS:
                trader_mod.BASKET_CONFIGS[sym]["z_threshold"] = z
            print(f"  Applied: SpreadArb z_threshold={z}")

        bt2 = Backtester(data_path, verbose=False)
        bt2.trader = trader_mod.Trader()
        bt2.run()
        bt2.print_summary()

        if args.plot:
            bt2.plot()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
