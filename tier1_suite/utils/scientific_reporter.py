"""
Scientific Report Generator for TIER 1 Rejuvenation Suite.
Generates comprehensive, reproducible HTML/PDF reports with methods, metrics, and visualizations.
"""

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Statistical and visualization imports
from scipy import stats
from sklearn.calibration import calibration_curve

# Try to import optional dependencies
try:
    import weasyprint

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import umap as _umap  # Import but assign to indicate conditional usage

    UMAP_AVAILABLE = True
except ImportError:
    _umap = None
    UMAP_AVAILABLE = False


@dataclass
class ReportMetadata:
    """Metadata for the scientific report."""

    title: str
    author: str
    institution: str
    date: str
    run_id: str
    pipeline_version: str
    description: str
    keywords: List[str]


@dataclass
class MethodsSection:
    """Methods section data."""

    data_description: str
    preprocessing_steps: List[str]
    algorithms_used: List[Dict[str, Any]]
    hyperparameters: Dict[str, Any]
    validation_strategy: str
    random_seeds: Dict[str, int]
    software_versions: Dict[str, str]


@dataclass
class MetricsTable:
    """Structured metrics with confidence intervals."""

    metric_name: str
    values: List[float]
    ci_lower: List[float]
    ci_upper: List[float]
    mean: float
    std: float
    method: str = "bootstrap"
    confidence_level: float = 0.95


@dataclass
class Figure:
    """Figure metadata and data."""

    id: str
    title: str
    caption: str
    file_path: str
    figure_type: str  # 'calibration', 'feature_importance', 'umap', 'metrics', 'custom'
    data: Optional[Dict] = None


class ScientificReporter:
    """
    Comprehensive scientific report generator with Jinja2 templating.
    """

    def __init__(
        self, output_dir: Union[str, Path] = "reports", run_name: Optional[str] = None
    ):
        """
        Initialize the scientific reporter.

        Parameters:
        -----------
        output_dir : str or Path
            Base directory for reports
        run_name : str, optional
            Name for this run (defaults to timestamp)
        """
        self.output_dir = Path(output_dir)
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create run directory structure
        self.run_dir = (
            self.output_dir
            / "runs"
            / datetime.now().strftime("%Y-%m-%d")
            / self.run_name
        )
        self.figures_dir = self.run_dir / "figures"
        self.data_dir = self.run_dir / "data"
        self.templates_dir = self.run_dir / "templates"

        # Create directories
        for dir_path in [
            self.run_dir,
            self.figures_dir,
            self.data_dir,
            self.templates_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize Jinja2 environment
        self._setup_templates()

        # Initialize report components
        self.metadata: Optional[ReportMetadata] = None
        self.methods: Optional[MethodsSection] = None
        self.metrics_tables: List[MetricsTable] = []
        self.figures: List[Figure] = []
        self.custom_sections: Dict[str, str] = {}

        # Set up plotting style
        self._setup_plotting()

    def _setup_templates(self):
        """Set up Jinja2 template environment."""
        # Create template directory if it doesn't exist
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)

        # Create default template if it doesn't exist
        self._create_default_template()

        self.jinja_env = Environment(
            loader=FileSystemLoader([str(template_dir), str(self.templates_dir)]),
            autoescape=select_autoescape(["html", "xml"]),
        )

        # Add custom filters
        self.jinja_env.filters["scientific_notation"] = self._scientific_notation_filter
        self.jinja_env.filters["format_ci"] = self._format_ci_filter

    def _setup_plotting(self):
        """Set up matplotlib and seaborn styling."""
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("husl")

        # Set figure parameters
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 16,
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
            }
        )

    def _create_default_template(self):
        """Create default HTML template."""
        template_dir = Path(__file__).parent / "templates"
        template_path = template_dir / "scientific_report.html"

        if not template_path.exists():
            template_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ metadata.title }}</title>
    <style>
        body {
            font-family: 'Times New Roman', serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }

        .header {
            text-align: center;
            border-bottom: 2px solid #333;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }

        .metadata {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }

        .section {
            margin-bottom: 40px;
        }

        .section h2 {
            color: #2c3e50;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }

        .methods-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        .methods-table th, .methods-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }

        .methods-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }

        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        .metrics-table th, .metrics-table td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }

        .metrics-table th {
            background-color: #3498db;
            color: white;
        }

        .figure {
            margin: 30px 0;
            page-break-inside: avoid;
        }

        .figure img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }

        .figure-caption {
            font-style: italic;
            text-align: center;
            margin-top: 10px;
            color: #666;
        }

        .code-block {
            background-color: #f8f8f8;
            border-left: 4px solid #2c3e50;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
        }

        .highlight {
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }

        .footer {
            margin-top: 50px;
            border-top: 1px solid #ddd;
            padding-top: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }

        @media print {
            body { font-size: 12pt; }
            .section { page-break-inside: avoid; }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <h1>{{ metadata.title }}</h1>
        <p><strong>{{ metadata.author }}</strong> - {{ metadata.institution }}</p>
        <p>{{ metadata.date }} | Run ID: {{ metadata.run_id }}</p>
    </div>

    <!-- Metadata -->
    <div class="metadata">
        <h3>Study Metadata</h3>
        <p><strong>Description:</strong> {{ metadata.description }}</p>
        <p><strong>Keywords:</strong> {{ metadata.keywords | join(', ') }}</p>
        <p><strong>Pipeline Version:</strong> {{ metadata.pipeline_version }}</p>
    </div>

    <!-- Abstract -->
    {% if abstract %}
    <div class="section">
        <h2>Abstract</h2>
        {{ abstract }}
    </div>
    {% endif %}

    <!-- Methods -->
    <div class="section">
        <h2>Methods</h2>

        <h3>Data Description</h3>
        <p>{{ methods.data_description }}</p>

        <h3>Preprocessing Steps</h3>
        <ol>
        {% for step in methods.preprocessing_steps %}
            <li>{{ step }}</li>
        {% endfor %}
        </ol>

        <h3>Algorithms and Parameters</h3>
        <table class="methods-table">
            <tr><th>Algorithm</th><th>Version</th><th>Parameters</th></tr>
            {% for algo in methods.algorithms_used %}
            <tr>
                <td>{{ algo.name }}</td>
                <td>{{ algo.version }}</td>
                <td>{{ algo.parameters | tojson }}</td>
            </tr>
            {% endfor %}
        </table>

        <h3>Validation Strategy</h3>
        <p>{{ methods.validation_strategy }}</p>

        <h3>Reproducibility Information</h3>
        <div class="highlight">
            <strong>Random Seeds:</strong><br>
            {% for seed_name, seed_value in methods.random_seeds.items() %}
                {{ seed_name }}: {{ seed_value }}<br>
            {% endfor %}
        </div>

        <h3>Software Versions</h3>
        <table class="methods-table">
            <tr><th>Package</th><th>Version</th></tr>
            {% for package, version in methods.software_versions.items() %}
            <tr><td>{{ package }}</td><td>{{ version }}</td></tr>
            {% endfor %}
        </table>
    </div>

    <!-- Results -->
    <div class="section">
        <h2>Results</h2>

        <!-- Metrics Tables -->
        {% for table in metrics_tables %}
        <h3>{{ table.metric_name }}</h3>
        <table class="metrics-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>{{ table.confidence_level * 100 }}% CI</th>
                <th>Method</th>
            </tr>
            {% for i in range(table.values|length) %}
            <tr>
                <td>{{ table.metric_name }}</td>
                <td>{{ table.values[i] | scientific_notation }}</td>
                <td>{{ table | format_ci(i) }}</td>
                <td>{{ table.method }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endfor %}

        <!-- Figures -->
        {% for figure in figures %}
        <div class="figure">
            <h3>{{ figure.title }}</h3>
            <img src="{{ figure.file_path }}" alt="{{ figure.title }}">
            <div class="figure-caption">
                <strong>Figure {{ loop.index }}:</strong> {{ figure.caption }}
            </div>
        </div>
        {% endfor %}
    </div>

    <!-- Custom Sections -->
    {% for section_name, section_content in custom_sections.items() %}
    <div class="section">
        <h2>{{ section_name }}</h2>
        {{ section_content }}
    </div>
    {% endfor %}

    <!-- Footer -->
    <div class="footer">
        <p>Generated by TIER 1 Rejuvenation Suite Scientific Reporter</p>
        <p>Report generated on {{ generation_time }}</p>
    </div>
</body>
</html>"""

            template_dir.mkdir(exist_ok=True)
            with open(template_path, "w") as f:
                f.write(template_content)

    def _scientific_notation_filter(self, value: float, precision: int = 3) -> str:
        """Jinja2 filter for scientific notation."""
        if abs(value) < 0.001 or abs(value) >= 1000:
            return f"{value:.{precision}e}"
        else:
            return f"{value:.{precision}f}"

    def _format_ci_filter(self, table: MetricsTable, index: int) -> str:
        """Format confidence interval for Jinja2 template."""
        if not hasattr(table, "ci_lower") or not hasattr(table, "ci_upper"):
            return "N/A"
        if (
            not table.ci_lower
            or not table.ci_upper
            or index >= len(table.ci_lower)
            or index >= len(table.ci_upper)
        ):
            return "N/A"
        return f"[{table.ci_lower[index]:.3f}, {table.ci_upper[index]:.3f}]"

    def set_metadata(
        self,
        title: str,
        author: str,
        institution: str,
        description: str,
        keywords: List[str],
        pipeline_version: str = "1.0.0",
    ) -> None:
        """Set report metadata."""
        self.metadata = ReportMetadata(
            title=title,
            author=author,
            institution=institution,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_id=self.run_name,
            pipeline_version=pipeline_version,
            description=description,
            keywords=keywords,
        )

    def set_methods(
        self,
        data_description: str,
        preprocessing_steps: List[str],
        algorithms_used: List[Dict[str, Any]],
        hyperparameters: Dict[str, Any],
        validation_strategy: str,
        random_seeds: Dict[str, int],
        software_versions: Dict[str, str],
    ) -> None:
        """Set methods section."""
        self.methods = MethodsSection(
            data_description=data_description,
            preprocessing_steps=preprocessing_steps,
            algorithms_used=algorithms_used,
            hyperparameters=hyperparameters,
            validation_strategy=validation_strategy,
            random_seeds=random_seeds,
            software_versions=software_versions,
        )

    def add_metrics_table(
        self,
        metric_name: str,
        values: List[float],
        ci_method: str = "bootstrap",
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> None:
        """
        Add metrics table with confidence intervals.

        Parameters:
        -----------
        metric_name : str
            Name of the metric
        values : List[float]
            Metric values (e.g., from cross-validation)
        ci_method : str
            Method for CI calculation ('bootstrap', 'normal', 'percentile')
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap : int
            Number of bootstrap samples
        """
        # Calculate confidence intervals
        ci_lower, ci_upper = self._calculate_confidence_intervals(
            values, ci_method, confidence_level, n_bootstrap
        )

        # Calculate summary statistics
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # Sample standard deviation

        metrics_table = MetricsTable(
            metric_name=metric_name,
            values=values,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            mean=mean_val,
            std=std_val,
            method=ci_method,
            confidence_level=confidence_level,
        )

        self.metrics_tables.append(metrics_table)

    def _calculate_confidence_intervals(
        self,
        values: List[float],
        method: str,
        confidence_level: float,
        n_bootstrap: int,
    ) -> Tuple[List[float], List[float]]:
        """Calculate confidence intervals for metrics."""
        values = np.array(values)
        alpha = 1 - confidence_level

        if method == "bootstrap":
            # Bootstrap confidence intervals
            bootstrap_means = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(
                    values, size=len(values), replace=True
                )
                bootstrap_means.append(np.mean(bootstrap_sample))

            ci_lower = [np.percentile(bootstrap_means, 100 * alpha / 2)]
            ci_upper = [np.percentile(bootstrap_means, 100 * (1 - alpha / 2))]

        elif method == "normal":
            # Normal approximation
            mean = np.mean(values)
            se = stats.sem(values)
            ci = se * stats.t.ppf(1 - alpha / 2, len(values) - 1)
            ci_lower = [mean - ci]
            ci_upper = [mean + ci]

        elif method == "percentile":
            # Percentile method
            ci_lower = [np.percentile(values, 100 * alpha / 2)]
            ci_upper = [np.percentile(values, 100 * (1 - alpha / 2))]

        else:
            raise ValueError(f"Unknown CI method: {method}")

        return ci_lower, ci_upper

    def add_calibration_plot(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "Calibration Plot",
        n_bins: int = 10,
    ) -> str:
        """
        Create and add calibration plot.

        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_prob : np.ndarray
            Predicted probabilities
        title : str
            Plot title
        n_bins : int
            Number of bins for calibration

        Returns:
        --------
        str : Figure ID
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Calculate calibration
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        # Plot calibration curve
        ax.plot(mean_pred, fraction_pos, "s-", label="Model", linewidth=2, markersize=8)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

        # Add histogram of predictions
        ax2 = ax.twinx()
        ax2.hist(y_prob, bins=50, alpha=0.3, color="gray", density=True)
        ax2.set_ylabel("Density")

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Save figure
        figure_id = f"calibration_{len(self.figures)}"
        figure_path = self.figures_dir / f"{figure_id}.png"
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Add to figures list
        figure = Figure(
            id=figure_id,
            title=title,
            caption="Calibration plot showing the relationship between predicted probabilities and actual outcomes. Perfect calibration would follow the diagonal line.",
            file_path=str(figure_path.relative_to(self.run_dir)),
            figure_type="calibration",
        )

        self.figures.append(figure)
        return figure_id

    def add_feature_importance_stability(
        self,
        feature_importances: Dict[str, List[float]],
        feature_names: List[str],
        title: str = "Feature Importance Stability",
    ) -> str:
        """
        Create feature importance stability plot.

        Parameters:
        -----------
        feature_importances : Dict[str, List[float]]
            Dictionary mapping method names to lists of feature importances
        feature_names : List[str]
            Names of features
        title : str
            Plot title

        Returns:
        --------
        str : Figure ID
        """
        n_methods = len(feature_importances)
        fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4 * n_methods))

        if n_methods == 1:
            axes = [axes]

        for i, (method, importances_list) in enumerate(feature_importances.items()):
            importances_array = np.array(importances_list)

            # Calculate mean and std
            mean_imp = np.mean(importances_array, axis=0)
            std_imp = np.std(importances_array, axis=0)

            # Sort by mean importance
            sorted_idx = np.argsort(mean_imp)[::-1]

            # Plot top 20 features
            top_idx = sorted_idx[:20]

            x_pos = np.arange(len(top_idx))
            axes[i].bar(
                x_pos, mean_imp[top_idx], yerr=std_imp[top_idx], capsize=5, alpha=0.7
            )
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(
                [feature_names[idx] for idx in top_idx], rotation=45, ha="right"
            )
            axes[i].set_ylabel("Importance")
            axes[i].set_title(f"{method} - Feature Importance (±1 SD)")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        figure_id = f"feature_stability_{len(self.figures)}"
        figure_path = self.figures_dir / f"{figure_id}.png"
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Add to figures list
        figure = Figure(
            id=figure_id,
            title=title,
            caption="Feature importance stability across multiple runs. Error bars show ±1 standard deviation.",
            file_path=str(figure_path.relative_to(self.run_dir)),
            figure_type="feature_importance",
        )

        self.figures.append(figure)
        return figure_id

    def add_umap_stability_heatmap(
        self,
        embeddings_list: List[np.ndarray],
        labels: Optional[np.ndarray] = None,
        title: str = "UMAP Stability Heatmap",
    ) -> str:
        """
        Create UMAP stability heatmap showing embedding consistency.

        Parameters:
        -----------
        embeddings_list : List[np.ndarray]
            List of UMAP embeddings from different runs
        labels : np.ndarray, optional
            Labels for coloring points
        title : str
            Plot title

        Returns:
        --------
        str : Figure ID
        """
        if not UMAP_AVAILABLE:
            print("Warning: UMAP not available, skipping UMAP stability plot")
            return ""

        n_runs = len(embeddings_list)

        # Calculate pairwise distances between runs
        distance_matrix = np.zeros((n_runs, n_runs))

        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                # Use Procrustes analysis to align embeddings
                aligned_i, aligned_j = self._procrustes_align(
                    embeddings_list[i], embeddings_list[j]
                )

                # Calculate mean pairwise distance
                distances = np.linalg.norm(aligned_i - aligned_j, axis=1)
                distance_matrix[i, j] = distance_matrix[j, i] = np.mean(distances)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(distance_matrix, cmap="RdYlBu_r", aspect="auto")
        ax.set_xticks(range(n_runs))
        ax.set_yticks(range(n_runs))
        ax.set_xticklabels([f"Run {i + 1}" for i in range(n_runs)])
        ax.set_yticklabels([f"Run {i + 1}" for i in range(n_runs)])

        # Add text annotations
        for i in range(n_runs):
            for j in range(n_runs):
                ax.text(
                    j,
                    i,
                    f"{distance_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Mean Pairwise Distance")

        # Save figure
        figure_id = f"umap_stability_{len(self.figures)}"
        figure_path = self.figures_dir / f"{figure_id}.png"
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Add to figures list
        figure = Figure(
            id=figure_id,
            title=title,
            caption="UMAP embedding stability heatmap showing mean pairwise distances between runs after Procrustes alignment. Lower values indicate more stable embeddings.",
            file_path=str(figure_path.relative_to(self.run_dir)),
            figure_type="umap",
        )

        self.figures.append(figure)
        return figure_id

    def _procrustes_align(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align two point sets using Procrustes analysis."""
        # Center the point sets
        X_centered = X - np.mean(X, axis=0)
        Y_centered = Y - np.mean(Y, axis=0)

        # Calculate optimal rotation
        H = X_centered.T @ Y_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Apply rotation to Y
        Y_aligned = Y_centered @ R.T

        return X_centered, Y_aligned

    def add_custom_figure(
        self,
        figure_path: Union[str, Path],
        title: str,
        caption: str,
        figure_type: str = "custom",
    ) -> str:
        """
        Add a custom figure to the report.

        Parameters:
        -----------
        figure_path : str or Path
            Path to the figure file
        title : str
            Figure title
        caption : str
            Figure caption
        figure_type : str
            Type of figure for organization

        Returns:
        --------
        str : Figure ID
        """
        figure_id = f"custom_{len(self.figures)}"

        # Copy figure to run directory
        source_path = Path(figure_path)
        dest_path = self.figures_dir / f"{figure_id}{source_path.suffix}"
        shutil.copy2(source_path, dest_path)

        figure = Figure(
            id=figure_id,
            title=title,
            caption=caption,
            file_path=str(dest_path.relative_to(self.run_dir)),
            figure_type=figure_type,
        )

        self.figures.append(figure)
        return figure_id

    def add_custom_section(self, section_name: str, content: str) -> None:
        """Add a custom section to the report."""
        self.custom_sections[section_name] = content

    def generate_html_report(
        self,
        template_name: str = "scientific_report.html",
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        Generate HTML report.

        Parameters:
        -----------
        template_name : str
            Name of the Jinja2 template
        output_filename : str, optional
            Output filename (defaults to run_name.html)

        Returns:
        --------
        Path : Path to generated HTML file
        """
        if not output_filename:
            output_filename = f"{self.run_name}_report.html"

        output_path = self.run_dir / output_filename

        # Load template
        template = self.jinja_env.get_template(template_name)

        # Render template
        html_content = template.render(
            metadata=self.metadata,
            methods=self.methods,
            metrics_tables=self.metrics_tables,
            figures=self.figures,
            custom_sections=self.custom_sections,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path

    def generate_pdf_report(
        self,
        html_report_path: Optional[Path] = None,
        output_filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Generate PDF report from HTML.

        Parameters:
        -----------
        html_report_path : Path, optional
            Path to HTML report (generates if not provided)
        output_filename : str, optional
            Output filename (defaults to run_name.pdf)

        Returns:
        --------
        Path : Path to generated PDF file, or None if PDF generation unavailable
        """
        if not PDF_AVAILABLE:
            print("Warning: weasyprint not available, cannot generate PDF")
            return None

        if html_report_path is None:
            html_report_path = self.generate_html_report()

        if not output_filename:
            output_filename = f"{self.run_name}_report.pdf"

        output_path = self.run_dir / output_filename

        # Generate PDF
        html_doc = weasyprint.HTML(filename=str(html_report_path))
        html_doc.write_pdf(str(output_path))

        return output_path

    def save_run_metadata(self) -> Path:
        """Save run metadata and parameters to JSON."""
        metadata_path = self.run_dir / "run_metadata.json"

        metadata_dict = {
            "metadata": asdict(self.metadata) if self.metadata else None,
            "methods": asdict(self.methods) if self.methods else None,
            "metrics_tables": [asdict(table) for table in self.metrics_tables],
            "figures": [asdict(figure) for figure in self.figures],
            "custom_sections": self.custom_sections,
            "run_directory": str(self.run_dir),
            "generation_time": datetime.now().isoformat(),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2, default=str)

        return metadata_path

    def generate_complete_report(
        self, include_pdf: bool = True, template_name: str = "scientific_report.html"
    ) -> Dict[str, Path]:
        """
        Generate complete report with HTML, optional PDF, and metadata.

        Parameters:
        -----------
        include_pdf : bool
            Whether to generate PDF version
        template_name : str
            Template name to use

        Returns:
        --------
        Dict[str, Path] : Dictionary with paths to generated files
        """
        results = {}

        # Generate HTML report
        html_path = self.generate_html_report(template_name)
        results["html"] = html_path

        # Generate PDF report if requested
        if include_pdf:
            pdf_path = self.generate_pdf_report(html_path)
            if pdf_path:
                results["pdf"] = pdf_path

        # Save metadata
        metadata_path = self.save_run_metadata()
        results["metadata"] = metadata_path

        print("📊 Report generated successfully!")
        print(f"   HTML: {html_path}")
        if "pdf" in results:
            print(f"   PDF: {results['pdf']}")
        print(f"   Metadata: {metadata_path}")
        print(f"   Run directory: {self.run_dir}")

        return results
