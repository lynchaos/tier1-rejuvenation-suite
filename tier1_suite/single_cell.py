#!/usr/bin/env python3
"""
Single-Cell Analyzer
====================

Main analyzer class for single-cell RNA-seq data processing,
quality control, dimensionality reduction, clustering, and trajectory analysis.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import anndata
import warnings

warnings.filterwarnings("ignore")

# Configure scanpy
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=80, facecolor="white")

# Import existing components
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from SingleCellRejuvenationAtlas.python.biologically_validated_analyzer import (
    BiologicallyValidatedAnalyzer,
)


class SingleCellAnalyzer:
    """
    Comprehensive single-cell analysis with biological validation.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.bio_analyzer = BiologicallyValidatedAnalyzer()

        if verbose:
            sc.settings.verbosity = 3
        else:
            sc.settings.verbosity = 1

    def load_data(self, input_file: str) -> anndata.AnnData:
        """Load single-cell data from various formats."""
        file_path = Path(input_file)

        if file_path.suffix.lower() == ".h5ad":
            return sc.read_h5ad(input_file)
        elif file_path.suffix.lower() == ".h5":
            return sc.read_10x_h5(input_file, genome=None, gex_only=True)
        elif file_path.suffix.lower() == ".csv":
            df = pd.read_csv(input_file, index_col=0)
            return anndata.AnnData(df.T)  # Transpose: genes as vars, cells as obs
        elif file_path.suffix.lower() in [".tsv", ".txt"]:
            df = pd.read_csv(input_file, sep="\t", index_col=0)
            return anndata.AnnData(df.T)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def run_qc(
        self,
        adata: anndata.AnnData,
        min_genes: int = 200,
        max_genes: int = 5000,
        min_cells: int = 3,
        mito_threshold: float = 20.0,
        doublet_detection: bool = True,
        generate_plots: bool = True,
    ) -> Dict[str, Any]:
        """Run comprehensive quality control analysis."""

        if self.verbose:
            print(f"Starting QC with {adata.n_obs} cells and {adata.n_vars} genes")

        # Make variable names unique
        adata.var_names_unique()

        # Calculate QC metrics
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

        # Add mitochondrial gene percentage
        adata.obs["pct_counts_mt"] = adata.obs["pct_counts_mt"]

        # Store original counts
        adata.raw = adata

        # Filter cells and genes
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)

        # Filter by gene counts and mitochondrial percentage
        adata = adata[adata.obs.n_genes_by_counts < max_genes, :]
        adata = adata[adata.obs.pct_counts_mt < mito_threshold, :]

        # Doublet detection if requested
        if doublet_detection:
            try:
                import scrublet as scr

                scrub = scr.Scrublet(adata.X)
                doublet_scores, predicted_doublets = scrub.scrub_doublets(
                    verbose=self.verbose
                )
                adata.obs["doublet_score"] = doublet_scores
                adata.obs["predicted_doublet"] = predicted_doublets

                # Filter doublets
                adata = adata[~adata.obs["predicted_doublet"], :]

            except ImportError:
                if self.verbose:
                    print("Scrublet not available, skipping doublet detection")

        # Biological QC validation
        qc_validation = self.bio_analyzer.validate_qc_metrics(adata)

        qc_results = {
            "cells_after_qc": adata.n_obs,
            "genes_after_qc": adata.n_vars,
            "doublets_removed": (
                sum(adata.obs.get("predicted_doublet", [])) if doublet_detection else 0
            ),
            "biological_validation": qc_validation,
        }

        if self.verbose:
            print(f"QC completed: {adata.n_obs} cells, {adata.n_vars} genes")

        return qc_results

    def run_embedding(
        self,
        adata: anndata.AnnData,
        n_pcs: int = 50,
        n_neighbors: int = 15,
        resolution: float = 0.5,
        methods: List[str] = ["umap", "tsne"],
        batch_correct: bool = False,
        batch_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run dimensionality reduction and embedding."""

        # Normalize and log-transform
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]

        # Scale data
        sc.pp.scale(adata, max_value=10)

        # Principal component analysis
        sc.tl.pca(adata, svd_solver="arpack", n_comps=n_pcs)

        # Batch correction if requested
        if batch_correct and batch_key:
            try:
                import scanpy.external as sce

                sce.pp.harmony_integrate(adata, key=batch_key)
                if self.verbose:
                    print(f"Applied Harmony batch correction using key: {batch_key}")
            except ImportError:
                if self.verbose:
                    print("Harmony not available, skipping batch correction")

        # Compute neighborhood graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

        # Generate embeddings
        embedding_results = {}

        if "umap" in methods:
            sc.tl.umap(adata)
            embedding_results["umap"] = True

        if "tsne" in methods:
            sc.tl.tsne(adata, n_pcs=n_pcs)
            embedding_results["tsne"] = True

        # Biological embedding validation
        embedding_validation = self.bio_analyzer.validate_embeddings(adata)
        embedding_results["biological_validation"] = embedding_validation

        if self.verbose:
            print(f"Embedding completed with methods: {methods}")

        return embedding_results

    def cluster_cells(
        self,
        adata: anndata.AnnData,
        method: str = "leiden",
        resolution: float = 0.5,
        key_added: str = "clusters",
        biomarker_annotation: bool = True,
        generate_plots: bool = True,
    ) -> Dict[str, Any]:
        """Perform clustering analysis."""

        # Clustering
        if method == "leiden":
            sc.tl.leiden(adata, resolution=resolution, key_added=key_added)
        elif method == "louvain":
            sc.tl.louvain(adata, resolution=resolution, key_added=key_added)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        n_clusters = len(adata.obs[key_added].unique())

        # Find marker genes
        sc.tl.rank_genes_groups(adata, key_added, method="wilcoxon")

        cluster_results = {
            "n_clusters": n_clusters,
            "method": method,
            "resolution": resolution,
        }

        # Biomarker-based annotation if requested
        if biomarker_annotation:
            annotations = self.bio_analyzer.annotate_clusters_with_biomarkers(
                adata, key_added
            )
            cluster_results["biomarker_annotations"] = annotations

        if self.verbose:
            print(f"Clustering completed: {n_clusters} clusters using {method}")

        return cluster_results

    def run_paga_analysis(
        self,
        adata: anndata.AnnData,
        cluster_key: str = "clusters",
        root_cluster: Optional[str] = None,
        compute_pseudotime: bool = True,
        rejuvenation_analysis: bool = True,
        generate_plots: bool = True,
    ) -> Dict[str, Any]:
        """Perform PAGA trajectory analysis."""

        # PAGA analysis
        sc.tl.paga(adata, groups=cluster_key)

        # Initialize root if not provided
        if root_cluster is None and compute_pseudotime:
            # Use biological knowledge to select root
            root_cluster = self.bio_analyzer.select_trajectory_root(adata, cluster_key)

        paga_results = {"paga_computed": True, "root_cluster": root_cluster}

        # Compute pseudotime if requested
        if compute_pseudotime and root_cluster:
            adata.uns["iroot"] = np.flatnonzero(adata.obs[cluster_key] == root_cluster)[
                0
            ]
            sc.tl.dpt(adata)
            paga_results["pseudotime_computed"] = True

        # Rejuvenation-specific trajectory analysis
        if rejuvenation_analysis:
            rejuv_analysis = self.bio_analyzer.analyze_rejuvenation_trajectories(
                adata, cluster_key
            )
            paga_results["rejuvenation_analysis"] = rejuv_analysis

        if self.verbose:
            print("PAGA trajectory analysis completed")

        return paga_results

    def run_complete_pipeline(
        self,
        input_file: str,
        output_dir: str,
        config_file: Optional[str] = None,
        skip_qc: bool = False,
        skip_embedding: bool = False,
        skip_clustering: bool = False,
        skip_trajectories: bool = False,
    ) -> Dict[str, Any]:
        """Run complete single-cell analysis pipeline."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pipeline_results = {}

        # Load data
        if self.verbose:
            print("Loading single-cell data...")
        adata = self.load_data(input_file)

        # Quality Control
        if not skip_qc:
            if self.verbose:
                print("Running quality control...")
            qc_results = self.run_qc(adata)
            pipeline_results["qc"] = qc_results
            adata.write(output_path / "data_after_qc.h5ad")

        # Embedding
        if not skip_embedding:
            if self.verbose:
                print("Computing embeddings...")
            embedding_results = self.run_embedding(adata)
            pipeline_results["embedding"] = embedding_results
            adata.write(output_path / "data_with_embeddings.h5ad")

        # Clustering
        if not skip_clustering:
            if self.verbose:
                print("Performing clustering...")
            cluster_results = self.cluster_cells(adata)
            pipeline_results["clustering"] = cluster_results
            adata.write(output_path / "data_with_clusters.h5ad")

        # Trajectory Analysis
        if not skip_trajectories:
            if self.verbose:
                print("Running trajectory analysis...")
            paga_results = self.run_paga_analysis(adata)
            pipeline_results["trajectories"] = paga_results
            adata.write(output_path / "data_final.h5ad")

        # Generate summary report
        summary_file = output_path / "pipeline_summary.txt"
        with open(summary_file, "w") as f:
            f.write("Single-Cell Analysis Pipeline Summary\n")
            f.write("=" * 40 + "\n\n")
            for stage, results in pipeline_results.items():
                f.write(f"{stage.upper()}:\n")
                if isinstance(results, dict):
                    for key, value in results.items():
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

        if self.verbose:
            print(f"Pipeline completed. Results saved to: {output_dir}")

        return pipeline_results
