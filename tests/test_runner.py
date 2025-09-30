"""
Test runner for TIER 1 Rejuvenation Suite.
Comprehensive test execution with reporting and coverage analysis.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pytest


def run_test_suite(
    test_categories: Optional[List[str]] = None,
    verbose: bool = True,
    coverage: bool = True,
    generate_report: bool = True,
) -> Dict[str, any]:
    """
    Run comprehensive test suite with optional coverage and reporting.

    Parameters:
    -----------
    test_categories : List[str], optional
        Specific test categories to run. If None, runs all tests.
        Options: ['transforms', 'leakage', 'determinism', 'schemas', 'integration']
    verbose : bool
        Whether to run tests in verbose mode
    coverage : bool
        Whether to generate coverage report
    generate_report : bool
        Whether to generate HTML test report

    Returns:
    --------
    Dict with test results and metrics
    """

    # Suppress warnings during testing
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Set environment variables for reproducible tests
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"

    # Configure test arguments
    pytest_args = []

    # Add test files based on categories
    if test_categories is None:
        test_categories = ["transforms", "leakage", "determinism", "schemas"]

    test_files = []
    for category in test_categories:
        test_file = f"tests/test_{category}.py"
        if Path(test_file).exists():
            test_files.append(test_file)
        else:
            print(f"Warning: Test file {test_file} not found")

    pytest_args.extend(test_files)

    # Configure verbosity
    if verbose:
        pytest_args.extend(["-v", "-s"])

    # Configure coverage
    if coverage:
        pytest_args.extend(
            [
                "--cov=tier1_suite",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-report=xml",
            ]
        )

    # Configure HTML report
    if generate_report:
        pytest_args.extend(["--html=test_report.html", "--self-contained-html"])

    # Add other useful options
    pytest_args.extend(
        [
            "--tb=short",  # Shorter traceback format
            "--strict-markers",  # Strict marker usage
            "--disable-warnings",  # Disable warnings in output
        ]
    )

    print(f"Running tests with categories: {test_categories}")
    print(f"Test files: {test_files}")
    print(f"Pytest args: {' '.join(pytest_args)}")

    # Run tests
    try:
        exit_code = pytest.main(pytest_args)

        # Collect results
        results = {
            "exit_code": exit_code,
            "success": exit_code == 0,
            "test_categories": test_categories,
            "coverage_enabled": coverage,
            "report_generated": generate_report,
        }

        # Parse coverage results if available
        if coverage and Path("htmlcov/index.html").exists():
            results["coverage_report"] = "htmlcov/index.html"

        if generate_report and Path("test_report.html").exists():
            results["test_report"] = "test_report.html"

        return results

    except Exception as e:
        print(f"Error running tests: {e}")
        return {
            "exit_code": -1,
            "success": False,
            "error": str(e),
            "test_categories": test_categories,
        }


def validate_test_environment() -> Dict[str, bool]:
    """
    Validate that the testing environment is properly configured.
    """
    validation_results = {}

    # Check required packages
    required_packages = [
        "pytest",
        "numpy",
        "pandas",
        "scikit-learn",
        "scanpy",
        "anndata",
        "matplotlib",
        "seaborn",
    ]

    for package in required_packages:
        try:
            __import__(package)
            validation_results[f"{package}_available"] = True
        except ImportError:
            validation_results[f"{package}_available"] = False

    # Check optional packages for enhanced testing
    optional_packages = ["pytest-cov", "pytest-html", "pytest-xdist"]

    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
            validation_results[f"{package}_available"] = True
        except ImportError:
            validation_results[f"{package}_available"] = False

    # Check file system permissions
    test_dir = Path("tests")
    validation_results["test_directory_exists"] = test_dir.exists()
    validation_results["test_directory_readable"] = test_dir.is_dir() and os.access(
        test_dir, os.R_OK
    )

    # Check for test files
    expected_test_files = [
        "test_transforms.py",
        "test_leakage.py",
        "test_determinism.py",
        "test_schemas.py",
    ]

    for test_file in expected_test_files:
        file_path = test_dir / test_file
        validation_results[f"{test_file}_exists"] = file_path.exists()

    # Check memory availability (rough estimate)
    try:
        import psutil

        memory_gb = psutil.virtual_memory().total / (1024**3)
        validation_results["sufficient_memory"] = memory_gb >= 2  # 2GB minimum
        validation_results["memory_gb"] = memory_gb
    except ImportError:
        validation_results["memory_check_available"] = False

    return validation_results


def generate_test_summary(results: Dict) -> str:
    """
    Generate a summary report of test results.
    """
    summary = []
    summary.append("=" * 60)
    summary.append("TIER 1 Rejuvenation Suite - Test Summary")
    summary.append("=" * 60)

    if results["success"]:
        summary.append("✅ ALL TESTS PASSED")
    else:
        summary.append("❌ SOME TESTS FAILED")

    summary.append(f"Exit Code: {results['exit_code']}")
    summary.append(f"Test Categories: {', '.join(results['test_categories'])}")

    if "coverage_report" in results:
        summary.append(f"Coverage Report: {results['coverage_report']}")

    if "test_report" in results:
        summary.append(f"HTML Test Report: {results['test_report']}")

    if "error" in results:
        summary.append(f"Error: {results['error']}")

    summary.append("=" * 60)

    return "\n".join(summary)


def run_critical_tests_only() -> bool:
    """
    Run only the most critical tests for CI/CD or quick validation.

    Returns:
    --------
    bool : True if all critical tests pass
    """

    # Define critical test markers or specific tests
    critical_tests = [
        "tests/test_leakage.py::TestDataLeakage::test_scaler_leakage_detection",
        "tests/test_determinism.py::TestSeedDeterminism::test_random_forest_determinism",
        "tests/test_schemas.py::TestDataSchemas::test_bulk_data_schema_validation",
    ]

    pytest_args = critical_tests + ["-v", "--tb=short"]

    print("Running critical tests only...")
    exit_code = pytest.main(pytest_args)

    return exit_code == 0


def benchmark_test_performance() -> Dict[str, float]:
    """
    Benchmark test execution time for performance monitoring.
    """
    import time

    test_categories = ["transforms", "leakage", "determinism", "schemas"]
    performance_results = {}

    for category in test_categories:
        test_file = f"tests/test_{category}.py"
        if not Path(test_file).exists():
            continue

        start_time = time.time()

        # Run single category with minimal output
        exit_code = pytest.main([test_file, "-q", "--tb=no"])

        execution_time = time.time() - start_time
        performance_results[category] = {
            "execution_time_seconds": execution_time,
            "success": exit_code == 0,
        }

    return performance_results


if __name__ == "__main__":
    """
    Command-line interface for test runner.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="TIER 1 Rejuvenation Suite Test Runner"
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        choices=["transforms", "leakage", "determinism", "schemas", "integration"],
        help="Test categories to run",
    )
    parser.add_argument(
        "--no-coverage", action="store_true", help="Disable coverage reporting"
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Disable HTML report generation"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--critical-only", action="store_true", help="Run only critical tests"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate test environment only"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark test performance"
    )

    args = parser.parse_args()

    if args.validate:
        print("Validating test environment...")
        validation = validate_test_environment()

        print("\nValidation Results:")
        for check, result in validation.items():
            status = "✅" if result else "❌"
            print(f"{status} {check}: {result}")

        # Check if critical requirements are met
        critical_checks = [
            "pytest_available",
            "numpy_available",
            "pandas_available",
            "scikit-learn_available",
            "test_directory_exists",
        ]

        all_critical_passed = all(
            validation.get(check, False) for check in critical_checks
        )

        if all_critical_passed:
            print("\n✅ Test environment validation passed!")
            sys.exit(0)
        else:
            print("\n❌ Test environment validation failed!")
            sys.exit(1)

    if args.benchmark:
        print("Benchmarking test performance...")
        performance = benchmark_test_performance()

        print("\nPerformance Results:")
        for category, metrics in performance.items():
            status = "✅" if metrics["success"] else "❌"
            time_str = f"{metrics['execution_time_seconds']:.2f}s"
            print(f"{status} {category}: {time_str}")

        sys.exit(0)

    if args.critical_only:
        success = run_critical_tests_only()
        sys.exit(0 if success else 1)

    # Run full test suite
    results = run_test_suite(
        test_categories=args.categories,
        verbose=not args.quiet,
        coverage=not args.no_coverage,
        generate_report=not args.no_report,
    )

    # Print summary
    summary = generate_test_summary(results)
    print(summary)

    # Exit with appropriate code
    sys.exit(results["exit_code"])
