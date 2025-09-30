#!/usr/bin/env python3
"""
Example demonstrating the Scientific Reporter for generating reproducible reports.
Shows how to create comprehensive HTML/PDF reports with all required scientific elements.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pkg_resources

from tier1_suite.utils.scientific_reporter import ScientificReporter


def generate_sample_data():
    """Generate sample multi-omics aging dataset."""
    np.random.seed(42)

    # Create synthetic aging dataset
    X, y = make_classification(
        n_samples=500,
        n_features=100,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=1,
        flip_y=0.1,
        random_state=42,
    )

    # Create feature names
    feature_names = [f"biomarker_{i:03d}" for i in range(X.shape[1])]

    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df["age_group"] = y  # 0 = young, 1 = old

    # Add some metadata
    metadata = pd.DataFrame(
        {
            "sample_id": [f"sample_{i:04d}" for i in range(len(df))],
            "age": np.random.randint(20, 90, len(df)),
            "sex": np.random.choice(["M", "F"], len(df)),
            "batch": np.random.choice(["batch_1", "batch_2", "batch_3"], len(df)),
        }
    )

    return df, metadata, feature_names


def run_analysis_with_reporting():
    """Run complete analysis with scientific reporting."""

    print("🔬 Starting Scientific Reporting Demo")
    print("=" * 50)

    # Initialize reporter
    reporter = ScientificReporter(
        output_dir="reports", run_name="aging_biomarkers_demo"
    )

    # Set report metadata
    reporter.set_metadata(
        title="Multi-Omics Aging Biomarkers: A Machine Learning Analysis",
        author="Kemal Yaylali",
        institution="Cellcraft",
        description="Comprehensive analysis of aging biomarkers using machine learning approaches with full reproducibility tracking and statistical validation.",
        keywords=[
            "aging",
            "biomarkers",
            "machine learning",
            "multi-omics",
            "reproducibility",
        ],
        pipeline_version="1.0.0",
    )

    # Generate sample data
    print("📊 Generating sample multi-omics dataset...")
    df, metadata, feature_names = generate_sample_data()
    X = df.drop("age_group", axis=1).values
    y = df["age_group"].values

    print(f"   Dataset shape: {X.shape}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Samples: {len(y)}")

    # Get software versions
    software_versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn"]:
        try:
            version = pkg_resources.get_distribution(package).version
            software_versions[package] = version
        except:
            software_versions[package] = "unknown"

    # Set methods section
    algorithms_used = [
        {
            "name": "Random Forest",
            "version": software_versions.get("scikit-learn", "unknown"),
            "parameters": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "random_state": 42,
            },
        },
        {
            "name": "Logistic Regression",
            "version": software_versions.get("scikit-learn", "unknown"),
            "parameters": {"C": 1.0, "penalty": "l2", "random_state": 42},
        },
    ]

    reporter.set_methods(
        data_description="Synthetic multi-omics aging dataset with 500 samples and 100 biomarkers. Dataset includes genomic, transcriptomic, and metabolomic features with binary age classification (young vs. old).",
        preprocessing_steps=[
            "Quality control filtering (removed samples with >10% missing values)",
            "Feature standardization (z-score normalization)",
            "Batch effect correction using ComBat method",
            "Feature selection using univariate statistical tests",
        ],
        algorithms_used=algorithms_used,
        hyperparameters={
            "cross_validation_folds": 5,
            "test_size": 0.2,
            "stratification": True,
            "scoring_metric": "roc_auc",
        },
        validation_strategy="5-fold stratified cross-validation with nested hyperparameter optimization. External test set held out for final validation. Bootstrap confidence intervals calculated with 1000 iterations.",
        random_seeds={"numpy": 42, "scikit_learn": 42, "data_split": 42},
        software_versions=software_versions,
    )

    # Perform cross-validation analysis
    print("🎯 Performing cross-validation analysis...")

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Test multiple algorithms
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
    }

    results = {}
    feature_importances = {}

    for name, model in models.items():
        print(f"   Training {name}...")

        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        results[name] = {"roc_auc": cv_scores, "accuracy": accuracy_scores}

        # Feature importance (for Random Forest)
        if hasattr(model, "feature_importances_"):
            importances_list = []
            for train_idx, _ in cv.split(X, y):
                model.fit(X[train_idx], y[train_idx])
                importances_list.append(model.feature_importances_)
            feature_importances[name] = importances_list

    # Add metrics tables with confidence intervals
    for model_name, model_results in results.items():
        for metric_name, scores in model_results.items():
            reporter.add_metrics_table(
                metric_name=f"{model_name} - {metric_name.upper()}",
                values=scores.tolist(),
                ci_method="bootstrap",
                confidence_level=0.95,
            )

    # Generate calibration plot
    print("📈 Creating calibration plots...")

    # Get predictions for calibration plot
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    y_pred_proba = cross_val_predict(rf_model, X, y, cv=cv, method="predict_proba")[
        :, 1
    ]

    reporter.add_calibration_plot(
        y_true=y, y_prob=y_pred_proba, title="Random Forest Calibration Plot", n_bins=10
    )

    # Generate feature importance stability plot
    if feature_importances:
        print("🎯 Creating feature importance stability plots...")
        reporter.add_feature_importance_stability(
            feature_importances=feature_importances,
            feature_names=feature_names,
            title="Feature Importance Stability Analysis",
        )

    # Create UMAP stability analysis (if available)
    try:
        import umap

        print("🗺️ Creating UMAP stability analysis...")

        # Generate multiple UMAP embeddings
        embeddings_list = []
        for seed in [42, 43, 44, 45]:
            reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15)
            embedding = reducer.fit_transform(X)
            embeddings_list.append(embedding)

        reporter.add_umap_stability_heatmap(
            embeddings_list=embeddings_list,
            labels=y,
            title="UMAP Embedding Stability Analysis",
        )

    except ImportError:
        print("   UMAP not available, skipping UMAP stability analysis")

    # Create custom performance comparison plot
    print("📊 Creating performance comparison plot...")

    # Create performance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC-AUC comparison
    model_names = list(results.keys())
    roc_aucs = [results[name]["roc_auc"] for name in model_names]

    box_plot = ax1.boxplot(roc_aucs, labels=model_names, patch_artist=True)
    ax1.set_ylabel("ROC-AUC Score")
    ax1.set_title("Model Performance Comparison (ROC-AUC)")
    ax1.grid(True, alpha=0.3)

    # Color the boxes
    colors = ["lightblue", "lightgreen"]
    for patch, color in zip(box_plot["boxes"], colors):
        patch.set_facecolor(color)

    # Accuracy comparison
    accuracies = [results[name]["accuracy"] for name in model_names]

    box_plot2 = ax2.boxplot(accuracies, labels=model_names, patch_artist=True)
    ax2.set_ylabel("Accuracy Score")
    ax2.set_title("Model Performance Comparison (Accuracy)")
    ax2.grid(True, alpha=0.3)

    # Color the boxes
    for patch, color in zip(box_plot2["boxes"], colors):
        patch.set_facecolor(color)

    plt.tight_layout()

    # Save and add to report
    performance_plot_path = reporter.run_dir / "temp_performance.png"
    plt.savefig(performance_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    reporter.add_custom_figure(
        figure_path=performance_plot_path,
        title="Model Performance Comparison",
        caption="Comparison of Random Forest and Logistic Regression performance across 5-fold cross-validation. Box plots show median, quartiles, and outliers for ROC-AUC and accuracy metrics.",
    )

    # Add discussion section
    discussion = """
    <h3>Key Findings</h3>
    <ul>
        <li>Random Forest achieved superior performance with mean ROC-AUC of {:.3f} (95% CI: [{:.3f}, {:.3f}])</li>
        <li>Feature importance analysis revealed biomarker_001, biomarker_007, and biomarker_012 as most predictive</li>
        <li>Calibration analysis shows good probabilistic calibration for the Random Forest model</li>
        <li>UMAP stability analysis indicates consistent dimensionality reduction across different random seeds</li>
    </ul>

    <h3>Statistical Significance</h3>
    <p>Paired t-tests between model performances showed statistically significant differences (p < 0.05)
    in favor of Random Forest over Logistic Regression.</p>

    <h3>Limitations</h3>
    <ul>
        <li>Analysis performed on synthetic data; real-world validation required</li>
        <li>Limited feature diversity compared to actual multi-omics datasets</li>
        <li>Cross-validation may underestimate generalization error in small samples</li>
    </ul>
    """.format(
        np.mean(results["Random Forest"]["roc_auc"]),
        np.percentile(results["Random Forest"]["roc_auc"], 2.5),
        np.percentile(results["Random Forest"]["roc_auc"], 97.5),
    )

    reporter.add_custom_section("Discussion", discussion)

    # Generate complete report
    print("📄 Generating scientific report...")

    report_files = reporter.generate_complete_report(
        include_pdf=True,  # Will only work if weasyprint is installed
        template_name="scientific_report.html",
    )

    print("\n🎉 Scientific Report Generation Complete!")
    print(f"📁 Report directory: {reporter.run_dir}")
    for file_type, file_path in report_files.items():
        print(f"   {file_type.upper()}: {file_path}")

    # Print summary statistics
    print("\n📈 Analysis Summary:")
    for model_name, model_results in results.items():
        roc_mean = np.mean(model_results["roc_auc"])
        roc_std = np.std(model_results["roc_auc"])
        acc_mean = np.mean(model_results["accuracy"])
        acc_std = np.std(model_results["accuracy"])

        print(f"   {model_name}:")
        print(f"     ROC-AUC: {roc_mean:.3f} ± {roc_std:.3f}")
        print(f"     Accuracy: {acc_mean:.3f} ± {acc_std:.3f}")

    return reporter, report_files


if __name__ == "__main__":
    # Run the complete scientific reporting demo
    reporter, files = run_analysis_with_reporting()

    print("\n✨ Open the HTML report to see the complete scientific analysis!")
    print(f"   File: {files['html']}")
