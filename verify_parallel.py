#!/usr/bin/env python3
"""
verify_parallel.py

Quick verification script to check that parallel processing is properly configured.
Run this after installation to verify everything is working.
"""

import sys

def verify_imports():
    """Verify all required imports are available."""
    print("Verifying imports...")

    try:
        import tqdm
        print("  [OK] tqdm installed")
    except ImportError:
        print("  [FAIL] tqdm missing - run: pip install tqdm")
        return False

    try:
        import psutil
        print("  [OK] psutil installed")
    except ImportError:
        print("  [FAIL] psutil missing - run: pip install psutil")
        return False

    try:
        from parallel_analyzer import ParallelAnalyzer, get_optimal_config
        print("  [OK] parallel_analyzer module available")
    except ImportError as e:
        print(f"  [FAIL] parallel_analyzer import failed: {e}")
        return False

    return True


def verify_system_config():
    """Check system configuration."""
    print("\nChecking system configuration...")

    try:
        from parallel_analyzer import get_optimal_config
        config = get_optimal_config()
        return True
    except Exception as e:
        print(f"  [FAIL] Error getting config: {e}")
        return False


def verify_models():
    """Check that model paths exist."""
    print("\nVerifying model paths...")

    import os
    current_dir = os.getcwd()

    models = {
        "Interaction": os.path.join(current_dir, "models", "eng_type_class_v1"),
        "Role": os.path.join(current_dir, "models", "role_class_v1"),
        "Embedding": os.path.join(current_dir, "models", "all-MiniLM-L6-v2"),
        "Sentiment": os.path.join(current_dir, "models", "deberta-v3-base-absa-v1.1")
    }

    all_exist = True
    for name, path in models.items():
        if os.path.exists(path):
            print(f"  [OK] {name} model found")
        else:
            print(f"  [FAIL] {name} model missing at: {path}")
            all_exist = False

    return all_exist


def main():
    print("=" * 80)
    print("PARALLEL PROCESSING VERIFICATION")
    print("=" * 80)
    print()

    checks = [
        ("Imports", verify_imports),
        ("System Config", verify_system_config),
        ("Models", verify_models)
    ]

    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False
            print(f"\n[FAIL] {name} check failed")
        else:
            print(f"\n[OK] {name} check passed")

    print("\n" + "=" * 80)
    if all_passed:
        print("[SUCCESS] All checks passed! Parallel processing is ready to use.")
        print("\nTry running:")
        print("  python cli_analysis.py --companies AAPL --limit 50 --test")
    else:
        print("[WARNING] Some checks failed. Please fix the issues above.")
        print("\nFor model setup, run:")
        print("  python download_models.py")
        print("\nFor dependencies, run:")
        print("  pip install -r requirements.txt")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
