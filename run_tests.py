#!/usr/bin/env python3
"""
Test runner for Arbor-o1 unit tests.

This script runs all unit tests and provides a comprehensive test report.
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tests():
    """Run all tests with comprehensive reporting."""
    
    print("🧪 Arbor-o1 Test Suite")
    print("=" * 50)
    
    # Test configuration
    test_args = [
        "-v",                    # Verbose output
        "-s",                    # Don't capture stdout
        "--tb=short",            # Short traceback format
        "--color=yes",           # Colored output
        "--durations=10",        # Show 10 slowest tests
    ]
    
    # Add test discovery paths
    test_dirs = [
        "tests/test_layers.py",
        "tests/test_growth_manager.py", 
        "tests/test_smoke_train.py",
    ]
    
    # Check if test files exist
    existing_tests = []
    for test_file in test_dirs:
        if os.path.exists(test_file):
            existing_tests.append(test_file)
            print(f"✓ Found: {test_file}")
        else:
            print(f"⚠️  Missing: {test_file}")
    
    if not existing_tests:
        print("❌ No test files found!")
        return 1
    
    print(f"\n🚀 Running {len(existing_tests)} test modules...")
    print("-" * 50)
    
    # Run tests
    exit_code = pytest.main(test_args + existing_tests)
    
    print("-" * 50)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code: {exit_code}")
    
    return exit_code


def run_specific_test(test_name):
    """Run a specific test module."""
    
    test_files = {
        "layers": "tests/test_layers.py",
        "growth": "tests/test_growth_manager.py",
        "smoke": "tests/test_smoke_train.py",
    }
    
    if test_name not in test_files:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_files.keys())}")
        return 1
    
    test_file = test_files[test_name]
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return 1
    
    print(f"🧪 Running {test_name} tests...")
    print("=" * 50)
    
    exit_code = pytest.main(["-v", "-s", "--tb=short", test_file])
    
    if exit_code == 0:
        print(f"✅ {test_name} tests passed!")
    else:
        print(f"❌ {test_name} tests failed!")
    
    return exit_code


def main():
    """Main test runner entry point."""
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        exit_code = run_specific_test(test_name)
    else:
        exit_code = run_tests()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
