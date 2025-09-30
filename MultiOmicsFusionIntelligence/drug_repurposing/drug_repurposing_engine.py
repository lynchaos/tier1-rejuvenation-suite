"""
Drug Repurposing and Therapeutic Target Discovery
===============================================
AI-powered identification of drug repurposing candidates and therapeutic targets for aging and rejuvenation
"""

import warnings
from collections import defaultdict
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


class DrugRepurposingEngine:
    """
    Comprehensive drug repurposing and therapeutic target discovery platform
    """

    def __init__(self):
        self.drug_database = None
        self.target_database = None
        self.expression_signatures = None
        self.connectivity_scores = None

    def load_drug_databases(self) -> Dict[str, pd.DataFrame]:
        """
        Load drug databases and create comprehensive drug profiles
        """
        print("Loading drug databases...")

        # Simulate drug database (in practice, would load from DrugBank, ChEMBL, etc.)
        np.random.seed(42)

        # Create drug profile database
        drugs = [
            "Metformin",
            "Rapamycin",
            "Resveratrol",
            "Nicotinamide",
            "Spermidine",
            "Lithium",
            "Aspirin",
            "Statins",
            "ACE_inhibitors",
            "Metoprolol",
            "Dasatinib",
            "Quercetin",
            "Fisetin",
            "Curcumin",
            "NAD_precursors",
            "Senolytics",
            "mTOR_inhibitors",
            "AMPK_activators",
            "Sirtuin_activators",
            "Antioxidants",
            "Anti_inflammatories",
            "Autophagy_inducers",
        ]

        # Drug properties
        drug_properties = pd.DataFrame(
            {
                "drug_name": drugs,
                "molecular_weight": np.random.normal(300, 100, len(drugs)),
                "logP": np.random.normal(2, 1, len(drugs)),
                "bioavailability": np.random.beta(3, 2, len(drugs)),
                "half_life": np.random.exponential(10, len(drugs)),
                "toxicity_score": np.random.beta(2, 5, len(drugs)),
                "clinical_trial_phase": np.random.choice(
                    ["Preclinical", "Phase I", "Phase II", "Phase III", "FDA Approved"],
                    len(drugs),
                ),
            }
        )

        # Drug-target interactions
        targets = [
            "mTOR",
            "AMPK",
            "SIRT1",
            "SIRT3",
            "FOXO1",
            "FOXO3",
            "TP53",
            "NFE2L2",
            "NFKB1",
            "IGF1R",
            "PIK3CA",
            "AKT1",
            "PTEN",
            "CDKN1A",
            "CDKN2A",
            "TERT",
            "KLOTHO",
            "TNF",
            "IL6",
            "IL1B",
            "SOD1",
            "SOD2",
            "CAT",
        ]

        # Create drug-target interaction matrix
        drug_target_interactions = np.random.choice(
            [0, 1], size=(len(drugs), len(targets)), p=[0.8, 0.2]
        )
        drug_target_df = pd.DataFrame(
            drug_target_interactions, index=drugs, columns=targets
        )

        # Drug mechanisms of action
        mechanisms = [
            "Autophagy induction",
            "Senescence clearance",
            "Oxidative stress reduction",
            "Inflammation suppression",
            "Metabolic enhancement",
            "DNA repair activation",
            "Telomere maintenance",
            "Protein homeostasis",
            "Mitochondrial function",
            "Insulin sensitivity",
            "Growth factor modulation",
            "Cell cycle regulation",
        ]

        drug_mechanisms = pd.DataFrame(
            np.random.choice([0, 1], size=(len(drugs), len(mechanisms)), p=[0.7, 0.3]),
            index=drugs,
            columns=mechanisms,
        )

        self.drug_database = {
            "properties": drug_properties,
            "targets": drug_target_df,
            "mechanisms": drug_mechanisms,
            "target_list": targets,
        }

        print(f"Loaded {len(drugs)} drugs with {len(targets)} targets")
        return self.drug_database

    def create_expression_signatures(
        self, expression_data: pd.DataFrame, condition_labels: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Create differential expression signatures for aging/rejuvenation conditions
        """
        print("Creating expression signatures...")

        signatures = {}
        conditions = condition_labels.unique()

        # Calculate differential expression signatures
        for condition in conditions:
            condition_samples = condition_labels[condition_labels == condition].index
            other_samples = condition_labels[condition_labels != condition].index

            if len(condition_samples) > 0 and len(other_samples) > 0:
                # Calculate fold changes
                condition_mean = expression_data.loc[condition_samples].mean()
                other_mean = expression_data.loc[other_samples].mean()

                # Log2 fold change
                fold_change = np.log2((condition_mean + 1) / (other_mean + 1))

                # T-test p-values
                p_values = []
                for gene in expression_data.columns:
                    condition_values = expression_data.loc[condition_samples, gene]
                    other_values = expression_data.loc[other_samples, gene]
                    _, p_val = stats.ttest_ind(condition_values, other_values)
                    p_values.append(p_val)

                # Create signature
                signature = pd.DataFrame(
                    {
                        "gene": expression_data.columns,
                        "log2fc": fold_change.values,
                        "p_value": p_values,
                    }
                )

                # Rank by significance and fold change
                signature["score"] = -np.log10(signature["p_value"] + 1e-10) * np.sign(
                    signature["log2fc"]
                )
                signature = signature.sort_values("score", ascending=False)

                signatures[condition] = signature.set_index("gene")["score"]

        self.expression_signatures = signatures
        print(f"Created signatures for {len(signatures)} conditions")
        return signatures

    def calculate_connectivity_scores(
        self, drug_signatures: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Calculate connectivity scores between disease signatures and drug signatures
        """
        print("Calculating drug-disease connectivity scores...")

        if self.expression_signatures is None:
            raise ValueError(
                "Expression signatures not created. Please run create_expression_signatures first."
            )

        connectivity_matrix = []
        drug_names = []
        condition_names = list(self.expression_signatures.keys())

        # For each drug, calculate connectivity with each condition
        for drug_name, drug_signature in drug_signatures.items():
            drug_connectivity = []

            for (
                _condition_name,
                condition_signature,
            ) in self.expression_signatures.items():
                # Find common genes
                common_genes = drug_signature.index.intersection(
                    condition_signature.index
                )

                if len(common_genes) > 50:  # Minimum overlap
                    drug_common = drug_signature[common_genes]
                    condition_common = condition_signature[common_genes]

                    # Calculate correlation (connectivity score)
                    connectivity_score = drug_common.corr(condition_common)

                    # Negative correlation suggests drug reverses disease signature
                    drug_connectivity.append(
                        -connectivity_score
                    )  # Flip sign for therapeutic potential
                else:
                    drug_connectivity.append(0)

            connectivity_matrix.append(drug_connectivity)
            drug_names.append(drug_name)

        connectivity_df = pd.DataFrame(
            connectivity_matrix, index=drug_names, columns=condition_names
        )

        self.connectivity_scores = connectivity_df
        print(f"Calculated connectivity scores for {len(drug_names)} drugs")
        return connectivity_df

    def identify_repurposing_candidates(
        self, target_condition: str = "aged", top_n: int = 20
    ) -> pd.DataFrame:
        """
        Identify top drug repurposing candidates for specific condition
        """
        print(f"Identifying repurposing candidates for {target_condition}...")

        if self.connectivity_scores is None:
            raise ValueError("Connectivity scores not calculated.")

        if target_condition not in self.connectivity_scores.columns:
            raise ValueError(
                f"Condition {target_condition} not found in connectivity scores."
            )

        # Get connectivity scores for target condition
        condition_scores = self.connectivity_scores[target_condition].sort_values(
            ascending=False
        )

        # Combine with drug properties for ranking
        drug_properties = self.drug_database["properties"].set_index("drug_name")

        repurposing_candidates = []

        for drug in condition_scores.head(top_n).index:
            if drug in drug_properties.index:
                drug_info = drug_properties.loc[drug]

                # Calculate repurposing score
                connectivity = condition_scores[drug]
                safety_score = 1 - drug_info["toxicity_score"]  # Higher is safer
                development_score = {
                    "Preclinical": 0.2,
                    "Phase I": 0.4,
                    "Phase II": 0.6,
                    "Phase III": 0.8,
                    "FDA Approved": 1.0,
                }[drug_info["clinical_trial_phase"]]

                repurposing_score = (
                    connectivity * 0.5 + safety_score * 0.3 + development_score * 0.2
                )

                repurposing_candidates.append(
                    {
                        "drug": drug,
                        "connectivity_score": connectivity,
                        "safety_score": safety_score,
                        "development_score": development_score,
                        "repurposing_score": repurposing_score,
                        "molecular_weight": drug_info["molecular_weight"],
                        "bioavailability": drug_info["bioavailability"],
                        "clinical_phase": drug_info["clinical_trial_phase"],
                    }
                )

        repurposing_df = pd.DataFrame(repurposing_candidates)
        repurposing_df = repurposing_df.sort_values(
            "repurposing_score", ascending=False
        )

        print(f"Identified {len(repurposing_df)} repurposing candidates")
        return repurposing_df

    def analyze_therapeutic_targets(
        self, repurposing_candidates: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze therapeutic targets based on drug repurposing results
        """
        print("Analyzing therapeutic targets...")

        drug_targets = self.drug_database["targets"]
        target_scores = defaultdict(list)

        # Calculate target importance based on repurposing scores
        for _, candidate in repurposing_candidates.iterrows():
            drug = candidate["drug"]
            repurposing_score = candidate["repurposing_score"]

            if drug in drug_targets.index:
                drug_target_profile = drug_targets.loc[drug]

                for target in drug_target_profile.index:
                    if drug_target_profile[target] == 1:  # Drug targets this protein
                        target_scores[target].append(repurposing_score)

        # Aggregate target scores
        target_analysis = []
        for target, scores in target_scores.items():
            target_analysis.append(
                {
                    "target": target,
                    "mean_repurposing_score": np.mean(scores),
                    "max_repurposing_score": np.max(scores),
                    "num_drugs": len(scores),
                    "total_score": np.sum(scores),
                }
            )

        target_df = pd.DataFrame(target_analysis)
        target_df = target_df.sort_values("total_score", ascending=False)

        # Add target annotations
        target_categories = {
            "mTOR": "Growth signaling",
            "AMPK": "Energy metabolism",
            "SIRT1": "Longevity",
            "SIRT3": "Longevity",
            "FOXO1": "Stress response",
            "FOXO3": "Stress response",
            "TP53": "DNA damage",
            "NFE2L2": "Antioxidant response",
            "NFKB1": "Inflammation",
            "IGF1R": "Growth signaling",
            "TNF": "Inflammation",
            "IL6": "Inflammation",
        }

        target_df["category"] = target_df["target"].map(
            lambda x: target_categories.get(x, "Other")
        )

        print(f"Analyzed {len(target_df)} therapeutic targets")
        return target_df

    def create_drug_target_network(
        self, repurposing_candidates: pd.DataFrame, target_analysis: pd.DataFrame
    ) -> nx.Graph:
        """
        Create drug-target interaction network
        """
        print("Creating drug-target network...")

        G = nx.Graph()

        # Add drug nodes
        for _, candidate in repurposing_candidates.head(15).iterrows():
            G.add_node(
                candidate["drug"],
                node_type="drug",
                repurposing_score=candidate["repurposing_score"],
                size=candidate["repurposing_score"] * 100,
            )

        # Add target nodes
        for _, target in target_analysis.head(10).iterrows():
            G.add_node(
                target["target"],
                node_type="target",
                total_score=target["total_score"],
                category=target["category"],
                size=target["total_score"] * 50,
            )

        # Add drug-target edges
        drug_targets = self.drug_database["targets"]

        for _, candidate in repurposing_candidates.head(15).iterrows():
            drug = candidate["drug"]
            if drug in drug_targets.index:
                for target in target_analysis.head(10)["target"]:
                    if (
                        target in drug_targets.columns
                        and drug_targets.loc[drug, target] == 1
                    ):
                        G.add_edge(drug, target, weight=candidate["repurposing_score"])

        print(
            f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )
        return G

    def visualize_repurposing_results(
        self, repurposing_candidates: pd.DataFrame, target_analysis: pd.DataFrame
    ) -> None:
        """
        Create comprehensive visualization of repurposing results
        """
        print("Generating visualization...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Drug Repurposing Analysis Results", fontsize=16, fontweight="bold"
        )

        # 1. Top repurposing candidates
        top_candidates = repurposing_candidates.head(10)
        axes[0, 0].barh(range(len(top_candidates)), top_candidates["repurposing_score"])
        axes[0, 0].set_yticks(range(len(top_candidates)))
        axes[0, 0].set_yticklabels(top_candidates["drug"])
        axes[0, 0].set_xlabel("Repurposing Score")
        axes[0, 0].set_title("Top 10 Drug Repurposing Candidates")

        # 2. Score components
        score_components = top_candidates[
            ["connectivity_score", "safety_score", "development_score"]
        ].head(8)
        score_components.plot(kind="bar", ax=axes[0, 1])
        axes[0, 1].set_xticklabels(top_candidates["drug"].head(8), rotation=45)
        axes[0, 1].set_title("Score Components for Top Candidates")
        axes[0, 1].legend()

        # 3. Clinical phase distribution
        phase_counts = repurposing_candidates["clinical_phase"].value_counts()
        axes[0, 2].pie(
            phase_counts.values, labels=phase_counts.index, autopct="%1.1f%%"
        )
        axes[0, 2].set_title("Clinical Development Phase Distribution")

        # 4. Target importance
        top_targets = target_analysis.head(10)
        axes[1, 0].barh(range(len(top_targets)), top_targets["total_score"])
        axes[1, 0].set_yticks(range(len(top_targets)))
        axes[1, 0].set_yticklabels(top_targets["target"])
        axes[1, 0].set_xlabel("Total Target Score")
        axes[1, 0].set_title("Top 10 Therapeutic Targets")

        # 5. Target categories
        category_counts = target_analysis["category"].value_counts()
        axes[1, 1].bar(category_counts.index, category_counts.values)
        axes[1, 1].set_title("Target Categories")
        axes[1, 1].tick_params(axis="x", rotation=45)

        # 6. Drug properties scatter
        axes[1, 2].scatter(
            repurposing_candidates["bioavailability"],
            repurposing_candidates["repurposing_score"],
            s=repurposing_candidates["molecular_weight"] / 5,
            alpha=0.6,
        )
        axes[1, 2].set_xlabel("Bioavailability")
        axes[1, 2].set_ylabel("Repurposing Score")
        axes[1, 2].set_title("Drug Properties vs Repurposing Score")

        plt.tight_layout()
        plt.savefig("drug_repurposing_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def generate_repurposing_report(
        self, repurposing_candidates: pd.DataFrame, target_analysis: pd.DataFrame
    ) -> str:
        """
        Generate comprehensive drug repurposing report
        """
        report = []
        report.append("=" * 80)
        report.append("DRUG REPURPOSING AND THERAPEUTIC TARGET ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total drug candidates analyzed: {len(repurposing_candidates)}")
        report.append(
            f"Top repurposing score: {repurposing_candidates['repurposing_score'].max():.3f}"
        )
        report.append(
            f"Average repurposing score: {repurposing_candidates['repurposing_score'].mean():.3f}"
        )
        report.append(f"Therapeutic targets identified: {len(target_analysis)}")
        report.append("")

        # Top drug candidates
        report.append("TOP 10 DRUG REPURPOSING CANDIDATES")
        report.append("-" * 40)
        for i, (_, candidate) in enumerate(
            repurposing_candidates.head(10).iterrows(), 1
        ):
            report.append(
                f"{i:2d}. {candidate['drug']:<15} "
                f"(Score: {candidate['repurposing_score']:.3f}, "
                f"Phase: {candidate['clinical_phase']}, "
                f"Bioavailability: {candidate['bioavailability']:.2f})"
            )
        report.append("")

        # Top therapeutic targets
        report.append("TOP 10 THERAPEUTIC TARGETS")
        report.append("-" * 40)
        for i, (_, target) in enumerate(target_analysis.head(10).iterrows(), 1):
            report.append(
                f"{i:2d}. {target['target']:<10} "
                f"(Category: {target['category']:<18}, "
                f"Score: {target['total_score']:.3f}, "
                f"Drugs: {target['num_drugs']})"
            )
        report.append("")

        # Clinical development insights
        phase_dist = repurposing_candidates["clinical_phase"].value_counts()
        report.append("CLINICAL DEVELOPMENT PHASE DISTRIBUTION")
        report.append("-" * 40)
        for phase, count in phase_dist.items():
            percentage = count / len(repurposing_candidates) * 100
            report.append(f"{phase:<15}: {count:3d} drugs ({percentage:5.1f}%)")
        report.append("")

        # Target category insights
        category_dist = target_analysis["category"].value_counts()
        report.append("THERAPEUTIC TARGET CATEGORIES")
        report.append("-" * 40)
        for category, count in category_dist.items():
            percentage = count / len(target_analysis) * 100
            report.append(f"{category:<20}: {count:2d} targets ({percentage:5.1f}%)")
        report.append("")

        report.append("=" * 80)

        report_text = "\n".join(report)

        # Save report
        with open("drug_repurposing_report.txt", "w") as f:
            f.write(report_text)

        print("Drug repurposing report generated!")
        return report_text

    def run_repurposing_pipeline(
        self,
        expression_data: pd.DataFrame,
        condition_labels: pd.Series,
        target_condition: str = "aged",
    ) -> Dict:
        """
        Complete drug repurposing analysis pipeline
        """
        print("Starting Drug Repurposing Pipeline...")

        # Load databases
        self.load_drug_databases()

        # Create expression signatures
        signatures = self.create_expression_signatures(
            expression_data, condition_labels
        )

        # Simulate drug signatures (in practice, would come from drug databases)
        drug_signatures = {}
        for drug in self.drug_database["properties"]["drug_name"]:
            # Create synthetic drug signature
            signature = pd.Series(
                np.random.normal(0, 1, len(expression_data.columns)),
                index=expression_data.columns,
            )
            drug_signatures[drug] = signature

        # Calculate connectivity scores
        connectivity_scores = self.calculate_connectivity_scores(drug_signatures)

        # Identify repurposing candidates
        repurposing_candidates = self.identify_repurposing_candidates(target_condition)

        # Analyze therapeutic targets
        target_analysis = self.analyze_therapeutic_targets(repurposing_candidates)

        # Create network
        drug_target_network = self.create_drug_target_network(
            repurposing_candidates, target_analysis
        )

        # Visualization
        self.visualize_repurposing_results(repurposing_candidates, target_analysis)

        # Generate report
        report = self.generate_repurposing_report(
            repurposing_candidates, target_analysis
        )

        results = {
            "repurposing_candidates": repurposing_candidates,
            "target_analysis": target_analysis,
            "connectivity_scores": connectivity_scores,
            "drug_target_network": drug_target_network,
            "expression_signatures": signatures,
            "report": report,
        }

        print("Drug repurposing pipeline completed!")
        return results


# Example usage
def simulate_aging_expression_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Simulate expression data for aging/rejuvenation analysis
    """
    print("Simulating aging expression data...")

    n_samples = 150
    n_genes = 1000
    np.random.seed(42)

    # Create sample labels
    conditions = ["young"] * 50 + ["aged"] * 50 + ["rejuvenated"] * 50
    sample_names = [f"Sample_{i}" for i in range(n_samples)]

    # Generate expression data with aging signal
    expression_data = np.random.lognormal(mean=1, sigma=0.5, size=(n_samples, n_genes))

    # Add aging signal to specific genes
    aging_genes_idx = np.arange(0, 100)  # First 100 genes
    for i, condition in enumerate(conditions):
        if condition == "aged":
            expression_data[i, aging_genes_idx] *= 1.5  # Upregulated in aging
        elif condition == "rejuvenated":
            expression_data[i, aging_genes_idx] *= 0.7  # Downregulated in rejuvenation

    expression_df = pd.DataFrame(
        expression_data,
        index=sample_names,
        columns=[f"Gene_{i}" for i in range(n_genes)],
    )

    condition_series = pd.Series(conditions, index=sample_names)

    return expression_df, condition_series


if __name__ == "__main__":
    # Simulate data
    expression_data, condition_labels = simulate_aging_expression_data()

    # Run repurposing analysis
    engine = DrugRepurposingEngine()
    results = engine.run_repurposing_pipeline(expression_data, condition_labels, "aged")

    print("\nTop 5 drug repurposing candidates:")
    print(
        results["repurposing_candidates"].head()[
            ["drug", "repurposing_score", "clinical_phase"]
        ]
    )

    print("\nTop 5 therapeutic targets:")
    print(results["target_analysis"].head()[["target", "category", "total_score"]])
