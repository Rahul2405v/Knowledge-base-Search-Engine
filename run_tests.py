#!/usr/bin/env python3
"""
Test runner script for the Knowledge Base Search Backend

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --verbose    # Run with verbose output
    python run_tests.py --coverage   # Run with coverage report
"""

import sys
import subprocess
import os

def run_tests(verbose=False, coverage=False):
    """Run the test suite"""
    
    # Make sure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term"])
    
    cmd.append("tests/")
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Please install it with: pip install pytest")
        return 1

if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    coverage = "--coverage" in sys.argv or "--cov" in sys.argv
    
    exit_code = run_tests(verbose=verbose, coverage=coverage)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    sys.exit(exit_code)