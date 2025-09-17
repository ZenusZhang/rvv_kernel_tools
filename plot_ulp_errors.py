#!/usr/bin/env python3
"""
Script to parse ULP error data from RVV kernel validation logs and generate plots.

Usage:
    python plot_ulp_errors.py <operation>

Example:
    python plot_ulp_errors.py exp

This will parse build/exp.log and generate plots showing:
1. x vs ULP vs f64
2. x vs ULP vs golden
"""

import re
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    """Parse log file to extract ULP error data."""
    data = {
        'x_values': [],
        'ulp_vs_f64': [],
        'ulp_vs_golden': []
    }

    # Pattern to match problematic cases
    problem_pattern = r'=== PROBLEMATIC CASE \((\w+)\) x = ([+-]?\d+\.\d+e[+-]?\d+) ==='
    ulp_pattern = r'ULP vs f64: (\d+), ULP vs golden\(\w+\): (\d+)'

    with open(log_path, 'r') as f:
        content = f.read()

    # Find all problematic cases
    problem_matches = re.finditer(problem_pattern, content)

    for match in problem_matches:
        operation = match.group(1)
        x_value = float(match.group(2))

        # Find the corresponding ULP line (should be the next line)
        start_pos = match.end()
        remaining_content = content[start_pos:]
        ulp_match = re.search(ulp_pattern, remaining_content)

        if ulp_match:
            ulp_f64 = int(ulp_match.group(1))
            ulp_golden = int(ulp_match.group(2))

            data['x_values'].append(x_value)
            data['ulp_vs_f64'].append(ulp_f64)
            data['ulp_vs_golden'].append(ulp_golden)

    return data

def create_plots(data, operation):
    """Create plots for ULP errors."""
    if not data['x_values']:
        print(f"No problematic cases found for operation: {operation}")
        return

    # Convert to numpy arrays for easier manipulation
    x_values = np.array(data['x_values'])
    ulp_f64 = np.array(data['ulp_vs_f64'])
    ulp_golden = np.array(data['ulp_vs_golden'])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: x vs ULP vs f64
    ax1.scatter(x_values, ulp_f64, alpha=0.7, s=20, color='blue')
    ax1.set_xlabel('x value')
    ax1.set_ylabel('ULP vs f64')
    ax1.set_title(f'{operation.upper()} - ULP Error vs f64 Reference')
    ax1.grid(True, alpha=0.3)

    # Plot 2: x vs ULP vs golden
    ax2.scatter(x_values, ulp_golden, alpha=0.7, s=20, color='red')
    ax2.set_xlabel('x value')
    ax2.set_ylabel('ULP vs golden')
    ax2.set_title(f'{operation.upper()} - ULP Error vs Golden Reference')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_path = f'{operation}_ulp_errors.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_path}")

    # Also create a combined plot
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.scatter(x_values, ulp_f64, alpha=0.7, s=20, color='blue', label='ULP vs f64')
    ax.scatter(x_values, ulp_golden, alpha=0.7, s=20, color='red', label='ULP vs golden')
    ax.set_xlabel('x value')
    ax.set_ylabel('ULP Error')
    ax.set_title(f'{operation.upper()} - ULP Errors Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    combined_output_path = f'{operation}_ulp_errors_combined.png'
    plt.savefig(combined_output_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved as: {combined_output_path}")

    # Print summary statistics
    print(f"\nSummary for {operation.upper()}:")
    print(f"Total problematic cases: {len(x_values)}")
    print(f"X value range: [{x_values.min():.6e}, {x_values.max():.6e}]")
    print(f"ULP vs f64 - Min: {ulp_f64.min()}, Max: {ulp_f64.max()}, Mean: {ulp_f64.mean():.2f}")
    print(f"ULP vs golden - Min: {ulp_golden.min()}, Max: {ulp_golden.max()}, Mean: {ulp_golden.mean():.2f}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_ulp_errors.py <operation>")
        print("Example: python plot_ulp_errors.py exp")
        sys.exit(1)

    operation = sys.argv[1]
    log_path = f"build/{operation}.log"

    if not os.path.exists(log_path):
        print(f"Error: Log file not found: {log_path}")
        print("Make sure you're running this script from the rvv_kernel_tools directory")
        print("and that the log file exists in the build/ directory.")
        sys.exit(1)

    print(f"Parsing log file: {log_path}")
    data = parse_log_file(log_path)

    if data['x_values']:
        create_plots(data, operation)
    else:
        print(f"No ULP error data found in {log_path}")

if __name__ == "__main__":
    main()