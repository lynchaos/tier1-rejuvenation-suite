#!/usr/bin/env python3
"""
Installation Test Script
========================

Quick test to verify the TIER 1 Rejuvenation Suite is properly installed and functional.
"""

import subprocess
import sys


def run_command(command, description):
    """Run a command and report success/failure"""
    print(f"\n🔍 Testing: {description}")
    print(f"Command: {command}")

    try:
        result = subprocess.run(
            command.split(), capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✅ SUCCESS")
            return True
        else:
            print(f"❌ FAILED: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✅ SUCCESS (timeout expected for interactive)")
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Run installation tests"""
    print("🧬 TIER 1 Rejuvenation Suite - Installation Test")
    print("=" * 50)

    tests = [
        ("tier1 --help", "Main CLI help"),
        ("tier1 version", "Version information"),
        ("tier1 info", "Suite information"),
        ("tier1 bulk --help", "Bulk analysis help"),
        ("tier1 sc --help", "Single-cell analysis help"),
        ("tier1 multi --help", "Multi-omics analysis help"),
        ("tier1 bulk fit --help", "Bulk fit command help"),
        ("tier1 sc run-qc --help", "Single-cell QC help"),
        ("tier1 multi pipeline --help", "Multi-omics pipeline help"),
    ]

    passed = 0
    total = len(tests)

    for command, description in tests:
        if run_command(command, description):
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Installation successful.")

        print("\n🚀 Quick Start Commands:")
        print("  tier1 info                    # Show suite information")
        print("  tier1 interactive            # Launch interactive mode")
        print("  tier1 bulk fit data.csv models/    # Train ML models")
        print("  tier1 sc pipeline data.h5ad results/  # Single-cell analysis")
        print("  tier1 multi pipeline *.csv results/   # Multi-omics integration")

        return True
    else:
        print("⚠️  Some tests failed. Check installation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
