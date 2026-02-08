#!/usr/bin/env python3
"""
Quick syntax check for the view analysis implementation.
"""

import ast
import sys

def check_syntax(filepath):
    """Check if Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        
        ast.parse(code)
        print(f"✓ {filepath}: Syntax OK")
        return True
    except SyntaxError as e:
        print(f"✗ {filepath}: Syntax Error")
        print(f"  Line {e.lineno}: {e.msg}")
        print(f"  {e.text}")
        return False

if __name__ == "__main__":
    files_to_check = [
        "m2f_train_multiview.py",
    ]
    
    all_ok = True
    for filepath in files_to_check:
        if not check_syntax(filepath):
            all_ok = False
    
    if all_ok:
        print("\n✓ All files passed syntax check!")
        print("\nNew features added:")
        print("  1. compute_view_metrics() - Per-view loss and class probability analysis")
        print("  2. CSVMetricsLogger.log_view_analysis() - Log view-level metrics to CSV")
        print("  3. MultiViewTrainer.run_view_analysis() - Standalone view analysis mode")
        print("  4. --view-analysis flag - Run detailed analysis during eval")
        print("\nCSV outputs:")
        print("  - training_metrics.csv: Training losses every 20 iterations")
        print("  - evaluation_metrics.csv: PQ/SQ/RQ after each evaluation")
        print("  - view_analysis_metrics.csv: Per-view losses and class consistency")
        print("\nUsage:")
        print("  # Run view analysis on evaluation set")
        print("  python m2f_train_multiview.py \\")
        print("      --eval-only --view-analysis \\")
        print("      --config-file configs/my_config.yaml \\")
        print("      MODEL.WEIGHTS output/model_final.pth")
        sys.exit(0)
    else:
        print("\n✗ Some files have syntax errors!")
        sys.exit(1)
