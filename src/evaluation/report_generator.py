"""
src/evaluation/report_generator.py
Aggregates experimental outputs (tables, charts) into a comprehensive final Markdown report.

DATA STREAM: Handles synthetic in-silico simulation results.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Compiles individual experiment results into a master research report.
    """
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.report_path = self.output_dir / "FINAL_RESEARCH_REPORT.md"

    def compile_report(self) -> str:
        """
        Reads generated markdown tables and combines them with narrative text.
        """
        logger.info("Compiling final research report...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        content = []
        content.append("# AI-Based Time-Aware Emergency Blood Donor Recommendation System")
        content.append(f"**Date Compiled:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("\n> **DISCLAIMER:** All spatial and simulation findings in this report are based on the **SYNTHETIC_SIMULATION** data stream. They represent in-silico computational experiments, not real-world clinical or operational evidence. The UCI dataset stream was strictly isolated to ML behavioral benchmarking.")
        
        content.append("\n## Executive Summary")
        content.append("This report aggregates the findings from a 7-phase experimental pipeline evaluating the Unified Donor Index (UDI) against traditional dispatch baselines.")
        
        content.append("\n## Phase 1 & 2: Machine Learning Behavioral Benchmarking (UCI Stream)")
        content.append("Evaluated LR, RF, SVM, and XGBoost on the isolated UCI donor behavioral dataset.")
        content.append("- *Refer to `reports/tables/exp1_ml_metrics.md` if generated.*")

        content.append("\n## Phase 3: Spatial Feasibility (Synthetic Stream)")
        content.append("Evaluated Network Grid vs Haversine vs Euclidean travel times under dynamic urban congestion.")
        content.append(self._load_table_safe("tables/spatial_feasibility_summary.md"))
        
        content.append("\n## Phase 4: UDI Multi-Criteria Ranking")
        content.append("Benchmarked UDI vs Proximity-First, Probability-First, and Random baselines.")
        content.append(self._load_table_safe("tables/udi_ranking_summary.md"))

        content.append("\n## Phase 5: Dynamic Recruitment Simulation")
        content.append("Simulated mass broadcast vs dynamic wave dispatch policies.")
        content.append(self._load_table_safe("tables/exp4_simulation_summary.md"))

        content.append("\n## Phase 6: Ablation & Sensitivity")
        content.append("Evaluated the impact of individual UDI components and alert fatigue decay parameters.")
        content.append("- *Refer to EXP-6 and EXP-7 console outputs for detailed metrics.*")
        
        content.append("\n## Conclusion")
        content.append("The UDI engine, combined with dynamic wave dispatch, demonstrates theoretical improvements in fulfilling emergency blood requests while actively mitigating donor alert fatigue within simulated environments.")

        final_markdown = "\n".join(content)
        self.report_path.write_text(final_markdown, encoding="utf-8")
        logger.info(f"Final report saved to: {self.report_path}")
        return str(self.report_path)

    def _load_table_safe(self, rel_path: str) -> str:
        """Safely attempt to read an intermediate report table."""
        full_path = self.output_dir / rel_path
        if full_path.exists():
            return f"\n{full_path.read_text(encoding='utf-8')}\n"
        return f"\n*(File not found: {rel_path}. Run the respective experiment first.)*\n"
