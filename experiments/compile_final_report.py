"""
experiments/compile_final_report.py
Phase 7 Final Execution Script

This script triggers the report generator to aggregate all existing 
experiment outputs into the final master document.
If the prior experiment scripts have not been run, it will indicate missing data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.evaluation.report_generator import ReportGenerator

logger = get_logger("compile_final_report")


def main():
    logger.info("=" * 65)
    logger.info("PHASE 7: Final Report Compilation")
    logger.info("=" * 65)
    
    generator = ReportGenerator(output_dir="reports")
    report_path = generator.compile_report()
    
    print("\n" + "=" * 65)
    print(f"Compilation Complete.")
    print(f"Master Report available at: {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
