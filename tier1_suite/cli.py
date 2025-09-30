#!/usr/bin/env python3
"""
TIER 1 Suite - Main CLI Entry Point
==================================

Command-line interface for the TIER 1 Cellular Rejuvenation Suite.
Provides access to bulk analysis, single-cell analysis, and multi-omics integration.
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tier1_suite.bulk_cli import bulk_app
from tier1_suite.multi_omics_cli import multi_app
from tier1_suite.single_cell_cli import sc_app

# Initialize Typer app and Rich console
app = typer.Typer(
    name="tier1",
    help="🧬 TIER 1 Cellular Rejuvenation Suite - Biologically validated analysis tools",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()

# Add subcommands
app.add_typer(bulk_app, name="bulk", help="Bulk data analysis and ML model operations")
app.add_typer(sc_app, name="sc", help="Single-cell analysis pipeline")
app.add_typer(multi_app, name="multi", help="Multi-omics integration and evaluation")


@app.command()
def version():
    """Show version information"""
    console.print("🧬 [bold blue]TIER 1 Cellular Rejuvenation Suite[/bold blue]")
    console.print("Version: [green]1.0.0[/green]")
    console.print("Author: [yellow]Kemal Yaylali[/yellow]")


@app.command()
def info():
    """Show detailed information about the suite"""
    console.print("\n🧬 [bold blue]TIER 1 Cellular Rejuvenation Suite[/bold blue]\n")

    table = Table(title="Suite Components")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Features", style="green")

    table.add_row(
        "tier1 bulk",
        "Bulk data analysis",
        "• ML model fit/predict\n• Biomarker validation\n• Statistical analysis",
    )
    table.add_row(
        "tier1 sc",
        "Single-cell analysis",
        "• Quality control\n• Dimensionality reduction\n• Clustering & trajectories",
    )
    table.add_row(
        "tier1 multi",
        "Multi-omics integration",
        "• Data integration\n• Pathway analysis\n• Biomarker discovery",
    )

    console.print(table)

    console.print("\n📚 [bold]Key Features:[/bold]")
    console.print("• 110+ peer-reviewed aging biomarkers")
    console.print("• Biologically validated scoring systems")
    console.print("• Ensemble ML models with cross-validation")
    console.print("• Age-stratified statistical analysis")
    console.print("• Comprehensive scientific reporting")


@app.command()
def interactive():
    """Launch interactive analysis interface"""
    console.print("🚀 [bold green]Launching interactive interface...[/bold green]")
    try:
        # Import and run the interactive interface
        from tier1_interactive import main as interactive_main

        interactive_main()
    except ImportError as e:
        console.print(f"❌ [red]Error launching interactive mode: {e}[/red]")
        console.print("Make sure the interactive module is properly installed.")


def main():
    """Main CLI entry point"""
    app()


if __name__ == "__main__":
    main()
