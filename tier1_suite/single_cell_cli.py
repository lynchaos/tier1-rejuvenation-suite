#!/usr/bin/env python3
"""
Single-Cell Analysis CLI
=======================

Command-line interface for single-cell RNA-seq analysis including QC,
dimensionality reduction, clustering, and trajectory analysis.
"""

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

sc_app = typer.Typer(
    name="sc", help="Single-cell analysis pipeline", no_args_is_help=True
)
console = Console()


@sc_app.command()
def run_qc(
    input_file: str = typer.Argument(
        ..., help="Input single-cell data file (H5AD/H5/CSV)"
    ),
    output_file: str = typer.Argument(..., help="Output file after QC"),
    min_genes: int = typer.Option(200, "--min-genes", help="Minimum genes per cell"),
    max_genes: int = typer.Option(5000, "--max-genes", help="Maximum genes per cell"),
    min_cells: int = typer.Option(3, "--min-cells", help="Minimum cells per gene"),
    mito_threshold: float = typer.Option(
        20.0, "--mito-thresh", help="Mitochondrial gene threshold (%)"
    ),
    doublet_detection: bool = typer.Option(
        True, "--doublets/--no-doublets", help="Enable doublet detection"
    ),
    generate_plots: bool = typer.Option(
        True, "--plots/--no-plots", help="Generate QC plots"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Run quality control on single-cell data.

    Performs comprehensive QC including filtering, doublet detection,
    and mitochondrial gene analysis with biological validation.
    """
    console.print("🔬 [bold blue]Starting single-cell quality control...[/bold blue]")

    try:
        from tier1_suite.single_cell import SingleCellAnalyzer

        analyzer = SingleCellAnalyzer(verbose=verbose)

        # Load data
        console.print(f"📂 Loading data from: [cyan]{input_file}[/cyan]")
        adata = analyzer.load_data(input_file)

        # Run QC
        console.print("🔍 Running quality control analysis...")
        analyzer.run_qc(
            adata=adata,
            min_genes=min_genes,
            max_genes=max_genes,
            min_cells=min_cells,
            mito_threshold=mito_threshold,
            doublet_detection=doublet_detection,
            generate_plots=generate_plots,
        )

        # Save results
        console.print(f"💾 Saving QC data to: [cyan]{output_file}[/cyan]")
        adata.write(output_file)

        console.print("✅ [green]Quality control completed![/green]")
        console.print(f"📊 Cells remaining: {adata.n_obs}")
        console.print(f"📊 Genes remaining: {adata.n_vars}")

    except Exception as e:
        console.print(f"❌ [red]Error during QC: {e}[/red]")
        raise typer.Exit(1)


@sc_app.command()
def run_embed(
    input_file: str = typer.Argument(..., help="Input single-cell data file (H5AD)"),
    output_file: str = typer.Argument(..., help="Output file with embeddings"),
    n_pcs: int = typer.Option(50, "--n-pcs", help="Number of principal components"),
    n_neighbors: int = typer.Option(
        15, "--neighbors", help="Number of neighbors for UMAP"
    ),
    resolution: float = typer.Option(0.5, "--resolution", help="Clustering resolution"),
    methods: Optional[List[str]] = typer.Option(
        ["umap", "tsne"], "--methods", help="Embedding methods"
    ),
    batch_correct: bool = typer.Option(
        False, "--batch-correct/--no-batch", help="Enable batch correction"
    ),
    batch_key: Optional[str] = typer.Option(
        None, "--batch-key", help="Batch correction key"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Run dimensionality reduction and embedding.

    Performs PCA, UMAP/t-SNE embedding with optional batch correction
    and biologically-informed parameter selection.
    """
    console.print(
        "🎯 [bold blue]Starting dimensionality reduction and embedding...[/bold blue]"
    )

    try:
        from tier1_suite.single_cell import SingleCellAnalyzer

        analyzer = SingleCellAnalyzer(verbose=verbose)

        # Load data
        console.print(f"📂 Loading data from: [cyan]{input_file}[/cyan]")
        adata = analyzer.load_data(input_file)

        # Run embedding
        console.print("🎯 Computing embeddings...")
        analyzer.run_embedding(
            adata=adata,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            resolution=resolution,
            methods=methods,
            batch_correct=batch_correct,
            batch_key=batch_key,
        )

        # Save results
        console.print(f"💾 Saving embedded data to: [cyan]{output_file}[/cyan]")
        adata.write(output_file)

        console.print("✅ [green]Embedding completed![/green]")
        console.print(f"📊 Embeddings: {', '.join(methods)}")

    except Exception as e:
        console.print(f"❌ [red]Error during embedding: {e}[/red]")
        raise typer.Exit(1)


@sc_app.command()
def cluster(
    input_file: str = typer.Argument(..., help="Input single-cell data file (H5AD)"),
    output_file: str = typer.Argument(..., help="Output file with clusters"),
    method: str = typer.Option(
        "leiden", "--method", help="Clustering method (leiden/louvain)"
    ),
    resolution: float = typer.Option(0.5, "--resolution", help="Clustering resolution"),
    key_added: str = typer.Option("clusters", "--key", help="Key for storing clusters"),
    biomarker_annotation: bool = typer.Option(
        True, "--annotate/--no-annotate", help="Annotate with biomarkers"
    ),
    generate_plots: bool = typer.Option(
        True, "--plots/--no-plots", help="Generate cluster plots"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Perform clustering analysis on single-cell data.

    Clusters cells using Leiden or Louvain algorithms with
    biological validation and automated cell type annotation.
    """
    console.print("🎪 [bold blue]Starting clustering analysis...[/bold blue]")

    try:
        from tier1_suite.single_cell import SingleCellAnalyzer

        analyzer = SingleCellAnalyzer(verbose=verbose)

        # Load data
        console.print(f"📂 Loading data from: [cyan]{input_file}[/cyan]")
        adata = analyzer.load_data(input_file)

        # Run clustering
        console.print("🎪 Performing clustering...")
        analyzer.cluster_cells(
            adata=adata,
            method=method,
            resolution=resolution,
            key_added=key_added,
            biomarker_annotation=biomarker_annotation,
            generate_plots=generate_plots,
        )

        # Save results
        console.print(f"💾 Saving clustered data to: [cyan]{output_file}[/cyan]")
        adata.write(output_file)

        console.print("✅ [green]Clustering completed![/green]")
        console.print(f"📊 Number of clusters: {len(adata.obs[key_added].unique())}")

    except Exception as e:
        console.print(f"❌ [red]Error during clustering: {e}[/red]")
        raise typer.Exit(1)


@sc_app.command()
def paga(
    input_file: str = typer.Argument(..., help="Input single-cell data file (H5AD)"),
    output_file: str = typer.Argument(..., help="Output file with PAGA results"),
    cluster_key: str = typer.Option(
        "clusters", "--cluster-key", help="Clustering key for PAGA"
    ),
    root_cluster: Optional[str] = typer.Option(
        None, "--root", help="Root cluster for trajectory"
    ),
    compute_pseudotime: bool = typer.Option(
        True, "--pseudotime/--no-pseudotime", help="Compute pseudotime"
    ),
    rejuvenation_analysis: bool = typer.Option(
        True, "--rejuv/--no-rejuv", help="Rejuvenation trajectory analysis"
    ),
    generate_plots: bool = typer.Option(
        True, "--plots/--no-plots", help="Generate PAGA plots"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Perform PAGA trajectory analysis.

    Runs Partition-based Graph Abstraction (PAGA) for trajectory inference
    with specialized rejuvenation pathway analysis.
    """
    console.print("🛤️ [bold blue]Starting PAGA trajectory analysis...[/bold blue]")

    try:
        from tier1_suite.single_cell import SingleCellAnalyzer

        analyzer = SingleCellAnalyzer(verbose=verbose)

        # Load data
        console.print(f"📂 Loading data from: [cyan]{input_file}[/cyan]")
        adata = analyzer.load_data(input_file)

        # Run PAGA
        console.print("🛤️ Computing PAGA trajectories...")
        analyzer.run_paga_analysis(
            adata=adata,
            cluster_key=cluster_key,
            root_cluster=root_cluster,
            compute_pseudotime=compute_pseudotime,
            rejuvenation_analysis=rejuvenation_analysis,
            generate_plots=generate_plots,
        )

        # Save results
        console.print(f"💾 Saving PAGA results to: [cyan]{output_file}[/cyan]")
        adata.write(output_file)

        console.print("✅ [green]PAGA analysis completed![/green]")
        if compute_pseudotime:
            console.print("📊 Pseudotime computed for trajectory analysis")

    except Exception as e:
        console.print(f"❌ [red]Error during PAGA analysis: {e}[/red]")
        raise typer.Exit(1)


@sc_app.command()
def pipeline(
    input_file: str = typer.Argument(..., help="Input single-cell data file"),
    output_dir: str = typer.Argument(..., help="Output directory for all results"),
    config_file: Optional[str] = typer.Option(
        None, "--config", help="Configuration file (YAML)"
    ),
    skip_qc: bool = typer.Option(False, "--skip-qc", help="Skip quality control"),
    skip_embedding: bool = typer.Option(False, "--skip-embed", help="Skip embedding"),
    skip_clustering: bool = typer.Option(
        False, "--skip-cluster", help="Skip clustering"
    ),
    skip_trajectories: bool = typer.Option(
        False, "--skip-traj", help="Skip trajectory analysis"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Run complete single-cell analysis pipeline.

    Executes the full pipeline: QC → Embedding → Clustering → Trajectory analysis
    with biologically validated parameters and comprehensive reporting.
    """
    console.print(
        "🚀 [bold blue]Starting complete single-cell pipeline...[/bold blue]"
    )

    try:
        from tier1_suite.single_cell import SingleCellAnalyzer

        analyzer = SingleCellAnalyzer(verbose=verbose)

        # Run pipeline
        console.print("🚀 Running complete pipeline...")
        analyzer.run_complete_pipeline(
            input_file=input_file,
            output_dir=output_dir,
            config_file=config_file,
            skip_qc=skip_qc,
            skip_embedding=skip_embedding,
            skip_clustering=skip_clustering,
            skip_trajectories=skip_trajectories,
        )

        console.print("✅ [green]Complete pipeline finished![/green]")
        console.print(f"📂 Results saved to: [cyan]{output_dir}[/cyan]")

    except Exception as e:
        console.print(f"❌ [red]Error in pipeline: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    sc_app()
