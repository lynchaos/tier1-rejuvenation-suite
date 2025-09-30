#!/usr/bin/env python3
"""
Multi-Omics Integration CLI
==========================

Command-line interface for multi-omics data integration,
fusion analysis, and comprehensive evaluation.
"""

import typer
from rich.console import Console
from rich.progress import track
from typing import Optional, List
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

multi_app = typer.Typer(
    name="multi", help="Multi-omics integration and evaluation", no_args_is_help=True
)
console = Console()


@multi_app.command()
def fit(
    data_files: List[str] = typer.Argument(..., help="Input multi-omics data files"),
    output_dir: str = typer.Argument(
        ..., help="Output directory for integration models"
    ),
    data_types: Optional[List[str]] = typer.Option(
        None, "--types", help="Data types (rna,protein,metabolite,etc)"
    ),
    integration_method: str = typer.Option(
        "mofa", "--method", help="Integration method (mofa/ae/vae/pca)"
    ),
    n_factors: int = typer.Option(10, "--factors", help="Number of latent factors"),
    batch_correction: bool = typer.Option(
        True, "--batch/--no-batch", help="Enable batch correction"
    ),
    biomarker_guided: bool = typer.Option(
        True, "--biomarker/--no-biomarker", help="Use biomarker guidance"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Fit multi-omics integration models.

    Integrates multiple omics datasets using advanced fusion methods
    with biological pathway guidance and batch correction.
    """
    console.print(
        f"🔬 [bold blue]Starting multi-omics integration fitting...[/bold blue]"
    )

    try:
        from tier1_suite.multi_omics import MultiOmicsAnalyzer

        analyzer = MultiOmicsAnalyzer(
            integration_method=integration_method,
            n_factors=n_factors,
            batch_correction=batch_correction,
            biomarker_guided=biomarker_guided,
            verbose=verbose,
        )

        # Load and integrate data
        console.print(f"📂 Loading {len(data_files)} omics datasets...")
        integration_results = analyzer.fit_integration(
            data_files=data_files, data_types=data_types, output_dir=output_dir
        )

        console.print(f"✅ [green]Integration models fitted successfully![/green]")
        console.print(f"📊 Latent factors: {n_factors}")
        console.print(f"💾 Models saved to: [cyan]{output_dir}[/cyan]")

    except Exception as e:
        console.print(f"❌ [red]Error during integration fitting: {e}[/red]")
        raise typer.Exit(1)


@multi_app.command()
def embed(
    data_files: List[str] = typer.Argument(..., help="Input multi-omics data files"),
    model_dir: str = typer.Argument(
        ..., help="Directory with trained integration models"
    ),
    output_file: str = typer.Argument(..., help="Output file for embeddings"),
    embedding_dim: int = typer.Option(50, "--dim", help="Embedding dimensionality"),
    umap_embed: bool = typer.Option(
        True, "--umap/--no-umap", help="Compute UMAP embedding"
    ),
    pathway_analysis: bool = typer.Option(
        True, "--pathway/--no-pathway", help="Include pathway analysis"
    ),
    generate_plots: bool = typer.Option(
        True, "--plots/--no-plots", help="Generate embedding plots"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Generate multi-omics embeddings using trained models.

    Creates integrated embeddings from multiple omics datasets
    with pathway analysis and visualization.
    """
    console.print(f"🎯 [bold blue]Starting multi-omics embedding...[/bold blue]")

    try:
        from tier1_suite.multi_omics import MultiOmicsAnalyzer

        analyzer = MultiOmicsAnalyzer(verbose=verbose)

        # Load models and generate embeddings
        console.print(f"🤖 Loading integration models from: [cyan]{model_dir}[/cyan]")
        analyzer.load_models(model_dir)

        console.print(f"🎯 Generating integrated embeddings...")
        embedding_results = analyzer.generate_embeddings(
            data_files=data_files,
            embedding_dim=embedding_dim,
            umap_embed=umap_embed,
            pathway_analysis=pathway_analysis,
            generate_plots=generate_plots,
        )

        # Save results
        console.print(f"💾 Saving embeddings to: [cyan]{output_file}[/cyan]")
        embedding_results.to_csv(output_file, index=False)

        console.print(f"✅ [green]Embeddings generated successfully![/green]")

    except Exception as e:
        console.print(f"❌ [red]Error during embedding: {e}[/red]")
        raise typer.Exit(1)


@multi_app.command()
def eval(
    embeddings_file: str = typer.Argument(..., help="Embeddings file to evaluate"),
    reference_data: Optional[str] = typer.Option(
        None, "--reference", help="Reference dataset for validation"
    ),
    output_dir: str = typer.Option(
        "evaluation_results", "--output", "-o", help="Output directory"
    ),
    metrics: Optional[List[str]] = typer.Option(
        ["silhouette", "ari", "nmi"], "--metrics", help="Evaluation metrics"
    ),
    biomarker_validation: bool = typer.Option(
        True, "--biomarker/--no-biomarker", help="Biomarker-based validation"
    ),
    pathway_enrichment: bool = typer.Option(
        True, "--pathway/--no-pathway", help="Pathway enrichment analysis"
    ),
    generate_report: bool = typer.Option(
        True, "--report/--no-report", help="Generate evaluation report"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Evaluate multi-omics integration quality.

    Comprehensive evaluation using clustering metrics, biological validation,
    and pathway enrichment analysis.
    """
    console.print(f"📊 [bold blue]Starting multi-omics evaluation...[/bold blue]")

    try:
        from tier1_suite.multi_omics import MultiOmicsAnalyzer
        import pandas as pd

        analyzer = MultiOmicsAnalyzer(verbose=verbose)

        # Load embeddings
        console.print(f"📂 Loading embeddings from: [cyan]{embeddings_file}[/cyan]")
        embeddings = pd.read_csv(embeddings_file)

        # Run evaluation
        console.print("📊 Running comprehensive evaluation...")
        eval_results = analyzer.evaluate_integration(
            embeddings=embeddings,
            reference_data=reference_data,
            output_dir=output_dir,
            metrics=metrics,
            biomarker_validation=biomarker_validation,
            pathway_enrichment=pathway_enrichment,
            generate_report=generate_report,
        )

        console.print(f"✅ [green]Evaluation completed![/green]")
        console.print(f"📊 Evaluation scores: {eval_results['summary']}")
        console.print(f"📂 Results saved to: [cyan]{output_dir}[/cyan]")

    except Exception as e:
        console.print(f"❌ [red]Error during evaluation: {e}[/red]")
        raise typer.Exit(1)


@multi_app.command()
def discover_biomarkers(
    data_files: List[str] = typer.Argument(..., help="Input multi-omics data files"),
    embeddings_file: str = typer.Argument(..., help="Integration embeddings file"),
    output_dir: str = typer.Argument(
        ..., help="Output directory for biomarker results"
    ),
    discovery_method: str = typer.Option(
        "integrated_shap", "--method", help="Discovery method (shap/lime/gradient)"
    ),
    top_n: int = typer.Option(
        100, "--top-n", help="Number of top biomarkers to report"
    ),
    pathway_filter: bool = typer.Option(
        True, "--pathway-filter/--no-filter", help="Filter by known pathways"
    ),
    validation_split: float = typer.Option(
        0.3, "--val-split", help="Validation split for biomarker testing"
    ),
    generate_plots: bool = typer.Option(
        True, "--plots/--no-plots", help="Generate biomarker plots"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Discover novel biomarkers from multi-omics integration.

    Uses integrated analysis to identify cross-omics biomarker signatures
    with biological validation and pathway analysis.
    """
    console.print(f"🔍 [bold blue]Starting biomarker discovery...[/bold blue]")

    try:
        from tier1_suite.multi_omics import MultiOmicsAnalyzer
        import pandas as pd

        analyzer = MultiOmicsAnalyzer(verbose=verbose)

        # Load data
        console.print(f"📂 Loading embeddings from: [cyan]{embeddings_file}[/cyan]")
        embeddings = pd.read_csv(embeddings_file)

        # Discover biomarkers
        console.print("🔍 Discovering cross-omics biomarkers...")
        discovery_results = analyzer.discover_biomarkers(
            data_files=data_files,
            embeddings=embeddings,
            output_dir=output_dir,
            discovery_method=discovery_method,
            top_n=top_n,
            pathway_filter=pathway_filter,
            validation_split=validation_split,
            generate_plots=generate_plots,
        )

        console.print(f"✅ [green]Biomarker discovery completed![/green]")
        console.print(
            f"🎯 Discovered {len(discovery_results['top_biomarkers'])} biomarkers"
        )
        console.print(f"📂 Results saved to: [cyan]{output_dir}[/cyan]")

    except Exception as e:
        console.print(f"❌ [red]Error during biomarker discovery: {e}[/red]")
        raise typer.Exit(1)


@multi_app.command()
def pipeline(
    data_files: List[str] = typer.Argument(..., help="Input multi-omics data files"),
    output_dir: str = typer.Argument(..., help="Output directory for all results"),
    config_file: Optional[str] = typer.Option(
        None, "--config", help="Configuration file (YAML)"
    ),
    integration_method: str = typer.Option(
        "mofa", "--method", help="Integration method"
    ),
    skip_fitting: bool = typer.Option(False, "--skip-fit", help="Skip model fitting"),
    skip_embedding: bool = typer.Option(
        False, "--skip-embed", help="Skip embedding generation"
    ),
    skip_evaluation: bool = typer.Option(False, "--skip-eval", help="Skip evaluation"),
    skip_discovery: bool = typer.Option(
        False, "--skip-discovery", help="Skip biomarker discovery"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Run complete multi-omics integration pipeline.

    Executes the full pipeline: Integration → Embedding → Evaluation → Discovery
    with comprehensive biological validation and reporting.
    """
    console.print(
        f"🚀 [bold blue]Starting complete multi-omics pipeline...[/bold blue]"
    )

    try:
        from tier1_suite.multi_omics import MultiOmicsAnalyzer

        analyzer = MultiOmicsAnalyzer(verbose=verbose)

        # Run pipeline
        console.print("🚀 Running complete multi-omics pipeline...")
        pipeline_results = analyzer.run_complete_pipeline(
            data_files=data_files,
            output_dir=output_dir,
            config_file=config_file,
            integration_method=integration_method,
            skip_fitting=skip_fitting,
            skip_embedding=skip_embedding,
            skip_evaluation=skip_evaluation,
            skip_discovery=skip_discovery,
        )

        console.print(f"✅ [green]Complete multi-omics pipeline finished![/green]")
        console.print(f"📂 Results saved to: [cyan]{output_dir}[/cyan]")

    except Exception as e:
        console.print(f"❌ [red]Error in multi-omics pipeline: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    multi_app()
