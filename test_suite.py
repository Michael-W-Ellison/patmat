#!/usr/bin/env python3
"""
Simplified test suite focusing on end-to-end functionality.
Tests that all 9 games can train successfully.
"""

import os
import subprocess
import tempfile
from pathlib import Path

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'


def run_test(test_name, command, expected_success=True):
    """Run a single test command"""
    print(f"\n{BLUE}Testing: {test_name}{RESET}")
    print(f"Command: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd='/home/user/patmat/chess_pattern_ai'
        )

        if result.returncode == 0:
            print(f"{GREEN}✓ PASSED{RESET}")
            return True
        else:
            print(f"{RED}✗ FAILED (exit code {result.returncode}){RESET}")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"{RED}✗ TIMEOUT{RESET}")
        return False
    except Exception as e:
        print(f"{RED}✗ ERROR: {e}{RESET}")
        return False


def main():
    """Run all tests"""
    print(f"{BLUE}{'='*70}")
    print("PATTERN RECOGNITION AI - FUNCTIONAL TEST SUITE")
    print(f"{'='*70}{RESET}\n")

    results = {}

    # Test 1: Checkers (3 games)
    results['Checkers'] = run_test(
        'Checkers (3 games)',
        'python3 checkers/checkers_headless_trainer.py 3'
    )

    # Test 2: Go 9x9 (2 games)
    results['Go (9x9)'] = run_test(
        'Go 9x9 (2 games)',
        'python3 go/go_headless_trainer.py 2 --size 9'
    )

    # Test 3: Othello (3 games)
    results['Othello'] = run_test(
        'Othello (3 games)',
        'python3 othello/othello_headless_trainer.py 3'
    )

    # Test 4: Connect Four (5 games - fast)
    results['Connect Four'] = run_test(
        'Connect Four (5 games)',
        'python3 connect4/connect4_headless_trainer.py 5'
    )

    # Test 5: Gomoku (3 games)
    results['Gomoku'] = run_test(
        'Gomoku (3 games)',
        'python3 gomoku/gomoku_headless_trainer.py 3'
    )

    # Test 6: Hex (3 games)
    results['Hex'] = run_test(
        'Hex (3 games)',
        'python3 hex/hex_headless_trainer.py 3'
    )

    # Test 7: Dots and Boxes (5 games - fast)
    results['Dots and Boxes'] = run_test(
        'Dots and Boxes (5 games)',
        'python3 dots_boxes/dots_boxes_headless_trainer.py 5'
    )

    # Test 8: Breakthrough (3 games)
    results['Breakthrough'] = run_test(
        'Breakthrough (3 games)',
        'python3 breakthrough/breakthrough_headless_trainer.py 3'
    )

    # Test 9: Pentago (5 games)
    results['Pentago'] = run_test(
        'Pentago (5 games)',
        'python3 pentago/pentago_headless_trainer.py 5'
    )

    # Test 10: Nine Men's Morris (5 games)
    results['Nine Men\'s Morris'] = run_test(
        'Nine Men\'s Morris (5 games)',
        'python3 morris/morris_headless_trainer.py 5'
    )

    # Test 11: Lines of Action (5 games)
    results['Lines of Action'] = run_test(
        'Lines of Action (5 games)',
        'python3 loa/loa_headless_trainer.py 5'
    )

    # Test 12: Show learned patterns (Checkers)
    results['Pattern Display'] = run_test(
        'Show learned patterns',
        'python3 checkers/checkers_headless_trainer.py --show-patterns 5'
    )

    # Print summary
    print(f"\n{BLUE}{'='*70}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*70}{RESET}\n")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    failed = total - passed

    for name, result in results.items():
        status = f"{GREEN}PASSED{RESET}" if result else f"{RED}FAILED{RESET}"
        print(f"{name:25} {status}")

    print(f"\n{BLUE}{'='*70}")
    print(f"Total: {total} | Passed: {GREEN}{passed}{RESET} | Failed: {RED}{failed}{RESET}")
    print(f"{'='*70}{RESET}\n")

    if failed == 0:
        print(f"{GREEN}✓ ALL TESTS PASSED!{RESET}\n")
        return 0
    else:
        print(f"{RED}✗ {failed} TEST(S) FAILED{RESET}\n")
        return 1


if __name__ == '__main__':
    exit(main())
