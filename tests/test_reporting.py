"""
tests/test_reporting.py
Unit tests for Phase 7: Final Reporting.

Verifies:
  1. Report Generator builds the document without throwing exceptions.
  2. Missing tables are handled gracefully via _load_table_safe.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.report_generator import ReportGenerator

def test_report_compilation_with_missing_files(tmp_path):
    # Use a temporary directory with no tables generated yet
    generator = ReportGenerator(output_dir=str(tmp_path))
    
    report_path = generator.compile_report()
    
    assert Path(report_path).exists()
    content = Path(report_path).read_text(encoding="utf-8")
    
    assert "File not found" in content
    assert "DISCLAIMER" in content
