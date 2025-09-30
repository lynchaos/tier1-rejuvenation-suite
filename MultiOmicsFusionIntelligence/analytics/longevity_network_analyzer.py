"""
Longevity Gene Network Analyzer
==============================
Advanced network analysis for prioritizing intervention targets in aging and rejuvenation
"""

import warnings
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class LongevityNetworkAnalyzer:
    """
    Network-based analysis for longevity gene prioritization
    """

    def __init__(self):
        self.gene_network = None
        self.expression_data = None
        self.pathway_database = None

    def load_gene_network(self, network_path: str = None) -> nx.Graph:
        """
        Load or create gene interaction network
        """
        if network_path:
            # Load existing network (e.g., STRING database)
            print(f"Loading gene network from {network_path}")
            # In practice, would load from file
            # For demo, create synthetic network

        print("Creating synthetic longevity gene network...")

        # Create synthetic network with known longevity genes
        longevity_genes = [
            "SIRT1",
            "SIRT3",
            "SIRT6",
            "FOXO1",
            "FOXO3",
            "TP53",
            "mTOR",
            "IGF1R",
            "APOE",
            "KLOTHO",
            "TERT",
            "TERF1",
            "CDKN2A",
            "CDKN1A",
            "NRF2",
            "PGC1A",
            "AMPK",
            "PTEN",
            "RB1",
            "p16",
            "p21",
            "ATM",
            "WRN",
            "BLM",
            "LMNA",
        ]

        # Add aging-related genes
        aging_genes = [
            "TNF",
            "IL6",
            "IL1B",
            "NFKB1",
            "JUN",
            "FOS",
            "MYC",
            "RAS",
            "AKT1",
            "PIK3CA",
            "VEGFA",
            "HIF1A",
            "EGFR",
            "PDGFRA",
            "TGFB1",
            "SMAD3",
        ]

        all_genes = longevity_genes + aging_genes + [f"Gene_{i}" for i in range(100)]

        # Create random network
        G = nx.erdos_renyi_graph(len(all_genes), 0.05)

        # Relabel nodes with gene names
        mapping = dict(enumerate(all_genes))
        G = nx.relabel_nodes(G, mapping)

        # Add edge weights (interaction confidence)
        for edge in G.edges():
            G.edges[edge]["weight"] = np.random.beta(2, 2)

        # Add node attributes
        for node in G.nodes():
            G.nodes[node]["gene_type"] = (
                "longevity"
                if node in longevity_genes
                else "aging"
                if node in aging_genes
                else "other"
            )

        self.gene_network = G
        print(
            f"Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
        return G

    def calculate_network_centrality(self) -> pd.DataFrame:
        """
        Calculate various centrality measures for gene prioritization
        """
        print("Calculating network centrality measures...")

        if self.gene_network is None:
            raise ValueError("Gene network not loaded. Please load network first.")

        G = self.gene_network

        centrality_measures = {
            "degree_centrality": nx.degree_centrality(G),
            "betweenness_centrality": nx.betweenness_centrality(G, weight="weight"),
            "closeness_centrality": nx.closeness_centrality(G, distance="weight"),
            "eigenvector_centrality": nx.eigenvector_centrality(G, weight="weight"),
            "pagerank": nx.pagerank(G, weight="weight"),
        }

        # Create DataFrame
        centrality_df = pd.DataFrame(centrality_measures)

        # Add composite score
        scaler = StandardScaler()
        scaled_centralities = scaler.fit_transform(centrality_df)
        centrality_df["composite_score"] = np.mean(scaled_centralities, axis=1)

        # Add gene types
        gene_types = [G.nodes[node]["gene_type"] for node in centrality_df.index]
        centrality_df["gene_type"] = gene_types

        # Sort by composite score
        centrality_df = centrality_df.sort_values("composite_score", ascending=False)

        print("Centrality calculation completed!")
        return centrality_df

    def identify_network_modules(self, n_clusters: int = 10) -> Dict[str, List[str]]:
        """
        Identify functional modules in the gene network
        """
        print("Identifying network modules...")

        if self.gene_network is None:
            raise ValueError("Gene network not loaded.")

        G = self.gene_network

        # Get adjacency matrix
        adj_matrix = nx.adjacency_matrix(G, weight="weight")

        # Spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=42
        )
        cluster_labels = clustering.fit_predict(adj_matrix)

        # Organize genes by clusters
        modules = {}
        gene_names = list(G.nodes())

        for i in range(n_clusters):
            cluster_genes = [
                gene_names[j] for j, label in enumerate(cluster_labels) if label == i
            ]
            modules[f"Module_{i}"] = cluster_genes

        print(f"Identified {len(modules)} network modules")
        return modules

    def pathway_impact_scoring(
        self, gene_expression: pd.DataFrame, comparison_groups: List[str]
    ) -> pd.DataFrame:
        """
        Cross-omics pathway impact scoring using GSEA + network topology
        """
        print("Calculating pathway impact scores...")

        # Mock pathway analysis (would use actual GSEA in practice)
        pathways = {
            "DNA_REPAIR": ["TP53", "ATM", "BRCA1", "BRCA2", "WRN", "BLM"],
            "CELLULAR_SENESCENCE": ["CDKN1A", "CDKN2A", "RB1", "TP53"],
            "AUTOPHAGY": ["mTOR", "AMPK", "BECN1", "ATG5", "ATG7"],
            "OXIDATIVE_STRESS": ["NRF2", "SOD1", "SOD2", "CAT", "GPX1"],
            "INFLAMMATION": ["TNF", "IL6", "IL1B", "NFKB1"],
            "METABOLISM": ["SIRT1", "SIRT3", "PGC1A", "AMPK"],
            "TELOMERE_MAINTENANCE": ["TERT", "TERF1", "TERF2", "RTEL1"],
            "INSULIN_SIGNALING": ["IGF1R", "FOXO1", "FOXO3", "AKT1", "PIK3CA"],
        }

        pathway_scores = []

        for pathway_name, pathway_genes in pathways.items():
            # Calculate pathway activity (simplified)
            available_genes = [g for g in pathway_genes if g in gene_expression.columns]

            if len(available_genes) > 0:
                pathway_activity = gene_expression[available_genes].mean(axis=1)

                # Calculate differential activity between groups
                group_means = {}
                for group in comparison_groups:
                    group_samples = [s for s in gene_expression.index if group in s]
                    if group_samples:
                        group_means[group] = pathway_activity[group_samples].mean()

                # Calculate impact score
                if len(group_means) >= 2:
                    group_values = list(group_means.values())
                    impact_score = max(group_values) - min(group_values)
                else:
                    impact_score = 0

                # Network topology bonus
                if self.gene_network:
                    network_genes = [
                        g for g in available_genes if g in self.gene_network.nodes()
                    ]
                    if network_genes:
                        subgraph = self.gene_network.subgraph(network_genes)
                        topology_score = nx.density(subgraph)
                        impact_score *= 1 + topology_score

                pathway_scores.append(
                    {
                        "pathway": pathway_name,
                        "impact_score": impact_score,
                        "num_genes": len(available_genes),
                        "network_genes": len(network_genes) if self.gene_network else 0,
                    }
                )

        pathway_df = pd.DataFrame(pathway_scores)
        pathway_df = pathway_df.sort_values("impact_score", ascending=False)

        print("Pathway impact scoring completed!")
        return pathway_df

    def prioritize_intervention_targets(
        self,
        centrality_df: pd.DataFrame,
        pathway_df: pd.DataFrame,
        druggability_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Prioritize genes for therapeutic intervention
        """
        print("Prioritizing intervention targets...")

        # Start with centrality scores
        target_scores = centrality_df[["composite_score", "gene_type"]].copy()
        target_scores["centrality_rank"] = range(1, len(target_scores) + 1)

        # Add pathway membership scores
        pathway_membership = {}
        for _, row in pathway_df.iterrows():
            row["pathway"]
            impact_score = row["impact_score"]
            # This is simplified - would map genes to pathways properly
            for gene in target_scores.index[:10]:  # Top genes as example
                if gene not in pathway_membership:
                    pathway_membership[gene] = 0
                pathway_membership[gene] += impact_score * 0.1

        target_scores["pathway_score"] = [
            pathway_membership.get(gene, 0) for gene in target_scores.index
        ]

        # Add druggability score (mock data)
        np.random.seed(42)
        target_scores["druggability_score"] = np.random.beta(2, 3, len(target_scores))

        # Calculate final intervention priority
        target_scores["intervention_priority"] = (
            target_scores["composite_score"] * 0.4
            + target_scores["pathway_score"] * 0.3
            + target_scores["druggability_score"] * 0.3
        )

        # Sort by priority
        target_scores = target_scores.sort_values(
            "intervention_priority", ascending=False
        )

        print("Intervention target prioritization completed!")
        return target_scores

    def visualize_network(self, centrality_df: pd.DataFrame, top_n: int = 20) -> None:
        """
        Visualize gene network with prioritization highlights
        """
        print("Creating network visualization...")

        if self.gene_network is None:
            raise ValueError("Gene network not loaded.")

        # Get top genes
        top_genes = centrality_df.head(top_n).index.tolist()
        subgraph = self.gene_network.subgraph(top_genes)

        # Set up plot
        plt.figure(figsize=(15, 10))

        # Node colors based on gene type
        color_map = {
            "longevity": "lightgreen",
            "aging": "lightcoral",
            "other": "lightblue",
        }
        node_colors = [
            color_map[subgraph.nodes[node]["gene_type"]] for node in subgraph.nodes()
        ]

        # Node sizes based on centrality scores
        node_sizes = [
            centrality_df.loc[node, "composite_score"] * 1000
            for node in subgraph.nodes()
        ]

        # Layout
        pos = nx.spring_layout(subgraph, k=3, iterations=50)

        # Draw network
        nx.draw_networkx_nodes(
            subgraph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7
        )
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, width=0.5)
        nx.draw_networkx_labels(subgraph, pos, font_size=8)

        plt.title("Top Longevity Gene Network\n(Node size = Centrality Score)")
        plt.axis("off")

        # Legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=gene_type,
            )
            for gene_type, color in color_map.items()
        ]
        plt.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        plt.savefig("longevity_gene_network.png", dpi=300, bbox_inches="tight")
        plt.show()

    def run_longevity_analysis(
        self, expression_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Complete longevity gene network analysis pipeline
        """
        print("Starting Longevity Gene Network Analysis Pipeline...")

        # Load network
        self.load_gene_network()

        # Calculate centrality measures
        centrality_df = self.calculate_network_centrality()

        # Identify modules
        modules = self.identify_network_modules()

        # Pathway analysis (if expression data provided)
        if expression_data is not None:
            comparison_groups = ["young", "aged", "rejuvenated"]
            pathway_df = self.pathway_impact_scoring(expression_data, comparison_groups)

            # Prioritize targets
            target_priorities = self.prioritize_intervention_targets(
                centrality_df, pathway_df
            )
        else:
            pathway_df = pd.DataFrame()
            target_priorities = centrality_df

        # Visualize network
        self.visualize_network(centrality_df)

        results = {
            "centrality_scores": centrality_df,
            "network_modules": modules,
            "pathway_scores": pathway_df,
            "intervention_targets": target_priorities,
        }

        print("Longevity analysis pipeline completed!")
        return results


# Example usage
if __name__ == "__main__":
    # Create analyzer
    analyzer = LongevityNetworkAnalyzer()

    # Run complete analysis
    results = analyzer.run_longevity_analysis()

    # Display top intervention targets
    print("\nTop 10 Intervention Targets:")
    print(
        results["intervention_targets"].head(10)[["gene_type", "intervention_priority"]]
    )

    # Display top pathways
    if not results["pathway_scores"].empty:
        print("\nTop Impacted Pathways:")
        print(results["pathway_scores"].head())
