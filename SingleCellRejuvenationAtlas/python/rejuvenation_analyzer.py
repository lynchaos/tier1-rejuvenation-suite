"""
Single-Cell Rejuvenation Atlas: Scanpy-based Analysis Pipeline
============================================================
Core trajectory inference and rejuvenation analysis using Scanpy
"""

import numpy as np
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier

# Configure scanpy
sc.settings.verbosity = 3  # verbosity level
sc.settings.set_figure_params(dpi=80, facecolor="white")


class RejuvenationAnalyzer:
    def __init__(self, adata):
        self.adata = adata

    def preprocess_data(self):
        """Preprocess single-cell data with robust filtering"""
        print("Starting Single-Cell Rejuvenation Analysis...")

        # Basic filtering with safety checks
        sc.pp.filter_cells(self.adata, min_genes=200)
        sc.pp.filter_genes(self.adata, min_cells=3)

        # Calculate QC metrics
        self.adata.var["mt"] = self.adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(
            self.adata, percent_top=None, log1p=False, inplace=True
        )

        # Normalization and log transformation
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)

        # Find highly variable genes with robust parameters
        try:
            sc.pp.highly_variable_genes(
                self.adata, min_mean=0.0125, max_mean=3, min_disp=0.5
            )
        except Exception as e:
            print(f"Using fallback highly variable gene detection: {e}")
            # Fallback: select top variable genes
            import numpy as np

            gene_var = np.var(self.adata.X.toarray(), axis=0)
            n_top_genes = min(2000, len(gene_var))
            top_genes_idx = np.argsort(gene_var)[-n_top_genes:]
            self.adata.var["highly_variable"] = False
            self.adata.var.iloc[
                top_genes_idx, self.adata.var.columns.get_loc("highly_variable")
            ] = True

        # Keep raw data
        self.adata.raw = self.adata

        # Filter to highly variable genes
        if "highly_variable" in self.adata.var.columns:
            self.adata = self.adata[:, self.adata.var.highly_variable]

        # Scale data
        sc.pp.scale(self.adata, max_value=10)

        print("Preprocessing complete")

    def trajectory_inference(self):
        """Aging → rejuvenation trajectory analysis"""
        # Principal component analysis
        sc.tl.pca(self.adata, svd_solver="arpack")

        # Neighborhood graph
        sc.pp.neighbors(self.adata, n_neighbors=10, n_pcs=40)

        # UMAP embedding
        sc.tl.umap(self.adata)

        # Clustering
        sc.tl.leiden(self.adata, resolution=0.5)

        # Trajectory inference (only if we have multiple clusters)
        n_clusters = len(self.adata.obs["leiden"].unique())
        if n_clusters > 1:
            sc.tl.paga(self.adata, groups="leiden")
            sc.pl.paga(self.adata, plot=False)
            sc.tl.umap(self.adata, init_pos="paga")
        else:
            print(f"Skipping PAGA: only {n_clusters} cluster found")

        print("Trajectory inference complete")
        return self.adata

    def cellular_reprogramming_predictor(self):
        """Predict optimal conditions for state transitions"""
        # Extract features for ML prediction
        X = self.adata.X.toarray() if hasattr(self.adata.X, "toarray") else self.adata.X

        # Mock aging labels (would come from experimental metadata)
        y_aging = np.random.choice(["young", "aged", "rejuvenated"], size=X.shape[0])

        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y_aging)

        # Predict reprogramming potential
        reprogramming_scores = clf.predict_proba(X)
        self.adata.obs["reprogramming_potential"] = reprogramming_scores[
            :, 2
        ]  # rejuvenated class

        print("Reprogramming prediction complete")
        return self.adata

    def senescence_marker_analysis(self):
        """Detect and visualize senescence markers"""
        senescence_markers = ["CDKN1A", "CDKN2A", "TP53", "RB1", "GLB1"]
        available_markers = [m for m in senescence_markers if m in self.adata.var_names]

        if available_markers:
            sc.pl.dotplot(
                self.adata,
                available_markers,
                groupby="leiden",
                save="_senescence_markers.pdf",
            )

            # Calculate senescence score
            sc.tl.score_genes(
                self.adata, available_markers, score_name="senescence_score"
            )

        print(f"Senescence analysis complete for {len(available_markers)} markers")
        return self.adata

    def stem_cell_pluripotency_scoring(self):
        """Score stem cell pluripotency"""
        pluripotency_markers = ["POU5F1", "SOX2", "KLF4", "MYC", "NANOG"]
        available_markers = [
            m for m in pluripotency_markers if m in self.adata.var_names
        ]

        if available_markers:
            sc.tl.score_genes(
                self.adata, available_markers, score_name="pluripotency_score"
            )

        print(f"Pluripotency scoring complete for {len(available_markers)} markers")
        return self.adata

    def generate_plots(self):
        """Generate key visualization plots"""
        # UMAP plot
        sc.pl.umap(
            self.adata,
            color=["leiden", "reprogramming_potential"],
            save="_rejuvenation_umap.pdf",
        )

        # Trajectory plot (only if PAGA was computed)
        if "paga" in self.adata.uns:
            sc.pl.paga(self.adata, color=["leiden"], save="_trajectory.pdf")

        # Marker plots
        if "senescence_score" in self.adata.obs.columns:
            sc.pl.umap(self.adata, color="senescence_score", save="_senescence.pdf")

        if "pluripotency_score" in self.adata.obs.columns:
            sc.pl.umap(self.adata, color="pluripotency_score", save="_pluripotency.pdf")

        print("Plots generated in figures/ directory")

    def run_full_analysis(self):
        """Execute complete analysis pipeline"""
        print("Starting Single-Cell Rejuvenation Analysis...")

        self.preprocess_data()
        self.trajectory_inference()
        self.cellular_reprogramming_predictor()
        self.senescence_marker_analysis()
        self.stem_cell_pluripotency_scoring()
        self.generate_plots()

        print("Analysis pipeline complete!")
        return self.adata


# Example usage
if __name__ == "__main__":
    # Load example data (would be real single-cell data)
    adata = sc.datasets.pbmc68k_reduced()  # Example dataset

    # Run analysis
    analyzer = RejuvenationAnalyzer(adata)
    result = analyzer.run_full_analysis()

    # Save results
    result.write("rejuvenation_analysis.h5ad")
