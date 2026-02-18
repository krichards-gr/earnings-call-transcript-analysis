#!/usr/bin/env python3
"""
benchmark_parallel.py

Benchmark script to compare sequential vs parallel processing performance.
Run this to see the speedup from parallelization on your system.

Usage:
    python benchmark_parallel.py --companies AAPL,MSFT --limit 100
"""

import argparse
import time
import subprocess
import sys
from datetime import datetime


def run_benchmark(companies, limit, runs=2):
    """
    Run benchmark comparing sequential vs parallel processing.

    Args:
        companies: Comma-separated company symbols
        limit: Number of records to process
        runs: Number of benchmark runs to average
    """
    print("=" * 80)
    print("PARALLEL PROCESSING BENCHMARK")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Companies: {companies}")
    print(f"  Limit: {limit} records")
    print(f"  Benchmark runs: {runs}")
    print("\n" + "=" * 80)

    results = {
        'sequential': [],
        'parallel': []
    }

    # Benchmark sequential processing
    print(f"\n[1/2] Benchmarking SEQUENTIAL processing...")
    print("-" * 80)
    for run in range(runs):
        print(f"\nRun {run + 1}/{runs}:")
        start = time.time()

        cmd = [
            sys.executable,
            "cli_analysis.py",
            "--companies", companies,
            "--limit", str(limit),
            "--no-parallel",
            "--no-content"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        elapsed = time.time() - start
        results['sequential'].append(elapsed)

        print(f"  Time: {elapsed:.2f}s")

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return

    # Benchmark parallel processing
    print(f"\n[2/2] Benchmarking PARALLEL processing...")
    print("-" * 80)
    for run in range(runs):
        print(f"\nRun {run + 1}/{runs}:")
        start = time.time()

        cmd = [
            sys.executable,
            "cli_analysis.py",
            "--companies", companies,
            "--limit", str(limit),
            "--no-content"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        elapsed = time.time() - start
        results['parallel'].append(elapsed)

        print(f"  Time: {elapsed:.2f}s")

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return

    # Calculate statistics
    avg_sequential = sum(results['sequential']) / len(results['sequential'])
    avg_parallel = sum(results['parallel']) / len(results['parallel'])
    speedup = avg_sequential / avg_parallel

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nSequential Processing:")
    print(f"  Average time: {avg_sequential:.2f}s")
    print(f"  Runs: {results['sequential']}")

    print(f"\nParallel Processing:")
    print(f"  Average time: {avg_parallel:.2f}s")
    print(f"  Runs: {results['parallel']}")

    print(f"\nSpeedup:")
    print(f"  {speedup:.2f}x faster with parallel processing")
    print(f"  Time saved: {avg_sequential - avg_parallel:.2f}s ({(1 - avg_parallel/avg_sequential)*100:.1f}%)")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark parallel vs sequential processing performance"
    )

    parser.add_argument('--companies', type=str, default='AAPL,MSFT',
                       help='Comma-separated company symbols (default: AAPL,MSFT)')
    parser.add_argument('--limit', type=int, default=100,
                       help='Number of records to process (default: 100)')
    parser.add_argument('--runs', type=int, default=2,
                       help='Number of benchmark runs to average (default: 2)')

    args = parser.parse_args()

    run_benchmark(args.companies, args.limit, args.runs)


if __name__ == "__main__":
    main()
