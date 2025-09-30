#!/usr/bin/env python3
"""
Bulk Data Analysis CLI
=====================

Command-line interface for bulk RNA-seq and other omics data analysis.
Provides ML model fitting, prediction, and biomarker validation.
"""

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from rich.console import Console

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

bulk_app = typer.Typer(
    name="bulk",
    help="Bulk data analysis and machine learning operations",
    no_args_is_help=True,
)
console = Console()


@bulk_app.command()
def fit(
    input_file: str = typer.Argument(..., help="Input data file (CSV/TSV/H5)"),
    output_dir: str = typer.Argument(..., help="Output directory for trained models"),
    target_column: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target column name"
    ),
    validation_split: float = typer.Option(
        0.2, "--val-split", help="Validation split ratio"
    ),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Cross-validation folds"),
    models: Optional[List[str]] = typer.Option(
        None, "--models", "-m", help="Models to train (rf,xgb,lgb,svm)"
    ),
    biomarker_validation: bool = typer.Option(
        True, "--biomarker-val/--no-biomarker-val", help="Enable biomarker validation"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Fit machine learning models on bulk omics data.

    Trains ensemble ML models with comprehensive biomarker validation
    and cross-validation for robust rejuvenation scoring.
    """
    console.print("🔬 [bold blue]Starting bulk data model fitting...[/bold blue]")

    try:
        # Import the bulk analyzer
        from tier1_suite.bulk import BulkAnalyzer

        analyzer = BulkAnalyzer(
            validation_split=validation_split,
            cv_folds=cv_folds,
            biomarker_validation=biomarker_validation,
            verbose=verbose,
        )

        # Load data
        console.print(f"📂 Loading data from: [cyan]{input_file}[/cyan]")
        data = analyzer.load_data(input_file)

        # Fit models
        console.print("🤖 Training ensemble models...")
        results = analyzer.fit_models(
            data=data,
            target_column=target_column,
            output_dir=output_dir,
            models=models or ["rf", "xgb", "lgb"],
        )

        console.print("✅ [green]Models trained successfully![/green]")
        console.print(f"📊 Model performance: {results['cv_scores']}")
        console.print(f"💾 Models saved to: [cyan]{output_dir}[/cyan]")

    except Exception as e:
        console.print(f"❌ [red]Error during model fitting: {e}[/red]")
        raise typer.Exit(1)


@bulk_app.command()
def predict(
    input_file: str = typer.Argument(..., help="Input data file for prediction"),
    model_dir: str = typer.Argument(..., help="Directory containing trained models"),
    output_file: str = typer.Argument(..., help="Output file for predictions"),
    ensemble_method: str = typer.Option(
        "voting", "--ensemble", help="Ensemble method (voting/stacking/mean)"
    ),
    confidence_intervals: bool = typer.Option(
        True, "--ci/--no-ci", help="Calculate confidence intervals"
    ),
    biomarker_analysis: bool = typer.Option(
        True, "--biomarker/--no-biomarker", help="Include biomarker analysis"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Make predictions using trained bulk analysis models.

    Generates rejuvenation scores with ensemble predictions,
    confidence intervals, and biomarker-level analysis.
    """
    console.print("🔮 [bold blue]Starting bulk data prediction...[/bold blue]")

    try:
        from tier1_suite.bulk import BulkAnalyzer

        analyzer = BulkAnalyzer(verbose=verbose)

        # Load data and models
        console.print(f"📂 Loading data from: [cyan]{input_file}[/cyan]")
        data = analyzer.load_data(input_file)

        console.print(f"🤖 Loading models from: [cyan]{model_dir}[/cyan]")
        analyzer.load_models(model_dir)

        # Make predictions
        console.print("🔮 Generating predictions...")
        predictions = analyzer.predict(
            data=data,
            ensemble_method=ensemble_method,
            confidence_intervals=confidence_intervals,
            biomarker_analysis=biomarker_analysis,
        )

        # Save results
        console.print(f"💾 Saving predictions to: [cyan]{output_file}[/cyan]")
        predictions.to_csv(output_file, index=False)

        console.print("✅ [green]Predictions generated successfully![/green]")
        console.print(f"📊 Predicted {len(predictions)} samples")

    except Exception as e:
        console.print(f"❌ [red]Error during prediction: {e}[/red]")
        raise typer.Exit(1)


@bulk_app.command()
def validate(
    predictions_file: str = typer.Argument(..., help="Predictions file to validate"),
    ground_truth_file: Optional[str] = typer.Option(
        None, "--truth", help="Ground truth file"
    ),
    output_dir: str = typer.Option(
        "validation_results", "--output", "-o", help="Output directory"
    ),
    generate_report: bool = typer.Option(
        True, "--report/--no-report", help="Generate validation report"
    ),
):
    """
    Validate bulk analysis predictions against ground truth or biological knowledge.

    Performs comprehensive validation using biological pathways,
    known aging signatures, and statistical tests.
    """
    console.print("✅ [bold blue]Starting prediction validation...[/bold blue]")

    try:
        from tier1_suite.bulk import BulkAnalyzer

        analyzer = BulkAnalyzer()

        # Load predictions
        console.print(f"📂 Loading predictions from: [cyan]{predictions_file}[/cyan]")
        predictions = pd.read_csv(predictions_file)

        # Validate
        console.print("🔬 Running biological validation...")
        validation_results = analyzer.validate_predictions(
            predictions=predictions,
            ground_truth_file=ground_truth_file,
            output_dir=output_dir,
            generate_report=generate_report,
        )

        console.print("✅ [green]Validation completed![/green]")
        console.print(f"📊 Validation metrics: {validation_results['summary']}")

    except Exception as e:
        console.print(f"❌ [red]Error during validation: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    bulk_app()
