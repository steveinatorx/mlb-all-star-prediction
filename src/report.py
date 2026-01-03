"""Report generation module."""

from pathlib import Path
from typing import Optional

import polars as pl
from loguru import logger

from src.config import config


def generate_markdown_report(
    experiments_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate a markdown report summarizing experiments.

    Args:
        experiments_dir: Directory containing experiment results
        output_path: Output path for report

    Returns:
        Path to generated report
    """
    experiments_dir = experiments_dir or config.experiments_dir
    output_path = output_path or config.reports_dir / "experiment_report.md"

    logger.info("Generating markdown report")

    # Load evaluation summary
    summary_path = config.tables_dir / "evaluation_summary.csv"
    if summary_path.exists():
        summary_df = pl.read_csv(summary_path)
    else:
        logger.warning("Evaluation summary not found, creating empty report")
        summary_df = None

    # Generate report
    report_lines = [
        "# MLB All-Star Prediction - Experiment Report",
        "",
        "## Overview",
        "",
        "This report summarizes the results of predicting MLB All-Star pitchers from "
        "Minor League data.",
        "",
        "## Data Summary",
        "",
        f"- **Training Period**: Up to {config.train_end_year}",
        f"- **Validation Period**: {config.train_end_year + 1} - {config.val_end_year}",
        f"- **Test Period**: After {config.val_end_year}",
        "",
        "## Model Performance",
        "",
    ]

    if summary_df is not None:
        report_lines.append("### Metrics Summary")
        report_lines.append("")
        report_lines.append("| Model | PR-AUC | ROC-AUC | Recall@Top10 | Recall@Top25 | Recall@Top50 |")
        report_lines.append("|-------|--------|---------|--------------|--------------|--------------|")

        for row in summary_df.iter_rows(named=True):
            model = row["model"]
            pr_auc = row.get("pr_auc", 0.0)
            roc_auc = row.get("roc_auc", 0.0)
            recall_10 = row.get("recall_at_top_10", 0.0)
            recall_25 = row.get("recall_at_top_25", 0.0)
            recall_50 = row.get("recall_at_top_50", 0.0)

            report_lines.append(
                f"| {model} | {pr_auc:.4f} | {roc_auc:.4f} | {recall_10:.4f} | "
                f"{recall_25:.4f} | {recall_50:.4f} |"
            )

        report_lines.append("")

    # Add figures section
    report_lines.extend(
        [
            "## Figures",
            "",
            "### Precision-Recall Curves",
            "",
        ]
    )

    figures_dir = config.figures_dir
    pr_curves = sorted(figures_dir.glob("pr_curve_*.png"))
    for curve_path in pr_curves:
        model_name = curve_path.stem.replace("pr_curve_", "")
        report_lines.append(f"![PR Curve: {model_name}]({curve_path.relative_to(config.project_root)})")
        report_lines.append("")

    report_lines.extend(
        [
            "### ROC Curves",
            "",
        ]
    )

    roc_curves = sorted(figures_dir.glob("roc_curve_*.png"))
    for curve_path in roc_curves:
        model_name = curve_path.stem.replace("roc_curve_", "")
        report_lines.append(f"![ROC Curve: {model_name}]({curve_path.relative_to(config.project_root)})")
        report_lines.append("")

    # SHAP plots
    shap_plots = sorted(figures_dir.glob("shap_*.png"))
    if shap_plots:
        report_lines.extend(
            [
                "### SHAP Feature Importance",
                "",
            ]
        )
        for shap_path in shap_plots:
            model_name = shap_path.stem.replace("shap_", "")
            report_lines.append(f"![SHAP: {model_name}]({shap_path.relative_to(config.project_root)})")
            report_lines.append("")

    # Coefficient plots
    coef_plots = sorted(figures_dir.glob("coefficients_*.png"))
    if coef_plots:
        report_lines.extend(
            [
                "### Logistic Regression Coefficients",
                "",
            ]
        )
        for coef_path in coef_plots:
            model_name = coef_path.stem.replace("coefficients_", "")
            report_lines.append(f"![Coefficients: {model_name}]({coef_path.relative_to(config.project_root)})")
            report_lines.append("")

    # Tables section
    report_lines.extend(
        [
            "## Tables",
            "",
        ]
    )

    tables_dir = config.tables_dir
    csv_files = sorted(tables_dir.glob("*.csv"))
    for csv_path in csv_files:
        table_name = csv_path.stem
        report_lines.append(f"### {table_name}")
        report_lines.append("")
        report_lines.append(f"See: [{table_name}.csv]({csv_path.relative_to(config.project_root)})")
        report_lines.append("")

    # Write report
    report_content = "\n".join(report_lines)
    output_path.write_text(report_content)

    logger.info(f"Saved report to {output_path}")

    return output_path

