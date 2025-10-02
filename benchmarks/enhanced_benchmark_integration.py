"""
Enhanced Benchmarking Script for TIER 1 Rejuvenation Suite
==========================================================

Integration of developer proposals with current implementation.
Provides comprehensive benchmarking against synthetic public datasets
with standardized metrics and performance evaluation.

Author: TIER 1 Enhancement Team
Date: 2025-10-02
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import current TIER 1 components
import sys
sys.path.append('/home/pi/projects')
from utils.advanced_statistics import AdvancedStatistics
from MultiOmicsFusionIntelligence.late_fusion_multiomics import LateFusionMultiOmics

@dataclass
class BenchmarkDataset:
    """Enhanced dataset container with metadata"""
    name: str
    description: str
    expression_data: pd.DataFrame
    metadata: pd.DataFrame
    source: str
    species: str
    tissue: str
    dataset_type: str  # 'cross_sectional', 'longitudinal', 'intervention'
    n_samples: int
    n_genes: int
    age_column: str = "age"
    
    def __post_init__(self):
        if self.n_samples is None:
            self.n_samples, self.n_genes = self.expression_data.shape

class EnhancedTIER1Benchmark:
    """
    Comprehensive benchmarking framework integrating developer proposals
    """
    
    def __init__(self, biomarker_panel_path=None, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Load enhanced biomarker panel
        self.biomarker_panel_path = biomarker_panel_path or "/home/pi/projects/data/validated_biomarkers.yaml"
        self.biomarker_panel = self._load_biomarker_panel()
        
        # Initialize statistics engine
        self.stats_engine = AdvancedStatistics(random_state=random_state)
        
        # Results storage
        self.benchmark_results = {}
        self.performance_metrics = {}
        
    def _load_biomarker_panel(self):
        """Load the enhanced biomarker panel"""
        import yaml
        try:
            with open(self.biomarker_panel_path, 'r') as f:
                panel = yaml.safe_load(f)
            print(f"✅ Loaded enhanced biomarker panel: {len(self._extract_all_genes(panel))} genes")
            return panel
        except FileNotFoundError:
            print(f"⚠️ Biomarker panel not found, using default")
            return self._create_default_panel()
    
    def _extract_all_genes(self, panel):
        """Extract all gene symbols from biomarker panel"""
        genes = set()
        if 'biomarkers' in panel:
            for pathway, data in panel['biomarkers'].items():
                if 'markers' in data:
                    for marker in data['markers']:
                        genes.add(marker['gene'])
        return genes
    
    def _create_default_panel(self):
        """Fallback biomarker panel"""
        return {
            'biomarkers': {
                'cellular_senescence': {
                    'markers': [
                        {'gene': 'CDKN2A', 'effect': 'up_with_age'},
                        {'gene': 'CDKN1A', 'effect': 'up_with_age'},
                        {'gene': 'TP53', 'effect': 'context_dependent'}
                    ]
                }
            }
        }
    
    def generate_enhanced_gtex_dataset(self, n_samples=250, n_genes=1000, tissue="liver"):
        """
        Generate GTEx-like dataset with realistic age-gene correlations
        Based on developer proposals with enhanced biological realism
        """
        print(f"🧬 Generating enhanced GTEx-like dataset ({tissue})...")
        
        # Realistic age distribution (GTEx characteristics)
        ages = np.concatenate([
            np.random.beta(2, 5, n_samples//3) * 40 + 20,  # Younger skew
            np.random.uniform(40, 70, n_samples//3),        # Middle age
            np.random.beta(5, 2, n_samples - 2*(n_samples//3)) * 20 + 60  # Older
        ])
        
        # Demographics
        sex = np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4])  # GTEx male bias
        
        # Get biomarker genes and add random genes
        biomarker_genes = list(self._extract_all_genes(self.biomarker_panel))[:50]
        random_genes = [f"ENSG{i:08d}" for i in range(n_genes - len(biomarker_genes))]
        all_genes = biomarker_genes + random_genes
        
        # Generate expression data
        expression_data = np.random.lognormal(5.5, 1.2, (n_samples, n_genes))
        
        # Add age-dependent effects to biomarker genes
        age_normalized = (ages - ages.mean()) / ages.std()
        
        for i, gene in enumerate(all_genes[:len(biomarker_genes)]):
            if gene in biomarker_genes:
                # Get expected effect from panel
                effect_direction = self._get_gene_effect_direction(gene)
                
                if 'up' in effect_direction.lower():
                    correlation_strength = np.random.uniform(0.25, 0.65)
                elif 'down' in effect_direction.lower():
                    correlation_strength = np.random.uniform(-0.65, -0.25)
                else:  # context_dependent
                    correlation_strength = np.random.uniform(-0.3, 0.3)
                
                # Add biological noise and non-linear effects
                noise = np.random.normal(0, 0.25, n_samples)
                age_effect = correlation_strength * age_normalized
                
                # Tissue-specific modulation
                if tissue == "brain":
                    age_effect *= 1.2  # Brain shows stronger aging effects
                elif tissue == "muscle":
                    age_effect *= 0.8  # Muscle more resilient
                
                expression_data[:, i] *= (1 + age_effect + noise)
        
        # Create DataFrames
        expr_df = pd.DataFrame(expression_data, columns=all_genes)
        
        metadata_df = pd.DataFrame({
            'sample_id': [f"GTEX_{tissue.upper()}_{i:04d}" for i in range(n_samples)],
            'age': ages.round().astype(int),
            'sex': sex,
            'tissue': tissue,
            'batch': np.random.choice(['batch1', 'batch2', 'batch3'], n_samples),
            'dataset_type': 'cross_sectional'
        })
        
        return BenchmarkDataset(
            name=f"GTEx_{tissue}",
            description=f"GTEx-like {tissue} aging dataset with {n_samples} samples",
            expression_data=expr_df,
            metadata=metadata_df,
            source="synthetic_gtex",
            species="human",
            tissue=tissue,
            dataset_type="cross_sectional",
            n_samples=n_samples,
            n_genes=n_genes
        )
    
    def generate_enhanced_intervention_dataset(self, n_samples=120, intervention="rapamycin"):
        """
        Generate intervention study dataset with realistic treatment effects
        """
        print(f"💊 Generating enhanced intervention dataset ({intervention})...")
        
        n_control = n_samples // 2
        n_treated = n_samples - n_control
        
        # Age-matched groups (older subjects for intervention studies)
        base_ages = np.random.normal(65, 8, n_samples)
        ages = np.clip(base_ages, 50, 85)
        
        # Treatment assignment
        treatment = np.concatenate([
            ['control'] * n_control,
            [intervention] * n_treated
        ])
        
        # Shuffle to avoid systematic bias
        shuffle_idx = np.random.permutation(n_samples)
        ages = ages[shuffle_idx]
        treatment = treatment[shuffle_idx]
        
        # Generate expression with intervention effects
        biomarker_genes = list(self._extract_all_genes(self.biomarker_panel))[:30]
        other_genes = [f"GENE_{i:04d}" for i in range(50 - len(biomarker_genes))]
        all_genes = biomarker_genes + other_genes
        
        expression_data = np.random.lognormal(4.5, 1.0, (n_samples, 50))
        
        # Apply intervention effects
        treatment_mask = treatment == intervention
        
        for i, gene in enumerate(biomarker_genes):
            effect_direction = self._get_gene_effect_direction(gene)
            
            # Intervention should counteract aging
            if 'up' in effect_direction.lower():
                # If gene increases with age, intervention should decrease it
                effect_size = np.random.uniform(-0.45, -0.15)
            elif 'down' in effect_direction.lower():
                # If gene decreases with age, intervention should increase it
                effect_size = np.random.uniform(0.15, 0.45)
            else:
                effect_size = np.random.uniform(-0.2, 0.2)
            
            # Add individual variation
            individual_effects = np.random.normal(effect_size, abs(effect_size) * 0.3, 
                                                np.sum(treatment_mask))
            
            expression_data[treatment_mask, i] *= (1 + individual_effects)
        
        # Create DataFrames
        expr_df = pd.DataFrame(expression_data, columns=all_genes)
        
        metadata_df = pd.DataFrame({
            'sample_id': [f"INT_{intervention.upper()}_{i:04d}" for i in range(n_samples)],
            'age': ages.round().astype(int),
            'treatment': treatment,
            'sex': np.random.choice(['M', 'F'], n_samples),
            'study_duration': np.random.choice([6, 12, 24], n_samples),  # months
            'dataset_type': 'intervention'
        })
        
        return BenchmarkDataset(
            name=f"intervention_{intervention}",
            description=f"{intervention.capitalize()} intervention study with {n_samples} samples",
            expression_data=expr_df,
            metadata=metadata_df,
            source="synthetic_intervention",
            species="human",
            tissue="multiple",
            dataset_type="intervention",
            n_samples=n_samples,
            n_genes=50
        )
    
    def generate_longitudinal_dataset(self, n_individuals=40, n_timepoints=4):
        """
        Generate longitudinal aging dataset
        """
        print(f"📈 Generating longitudinal dataset ({n_individuals} individuals, {n_timepoints} timepoints)...")
        
        data_rows = []
        biomarker_genes = list(self._extract_all_genes(self.biomarker_panel))[:20]
        other_genes = [f"GENE_{i:04d}" for i in range(30 - len(biomarker_genes))]
        all_genes = biomarker_genes + other_genes
        
        for individual in range(n_individuals):
            baseline_age = np.random.uniform(35, 55)
            sex = np.random.choice(['M', 'F'])
            
            # Individual baseline expression profile
            individual_baseline = np.random.lognormal(4.2, 0.4, len(all_genes))
            
            for timepoint in range(n_timepoints):
                current_age = baseline_age + timepoint * 3  # 3-year intervals
                
                # Calculate aging trajectory
                aging_factor = timepoint * 0.08  # 8% change per timepoint
                
                expression = individual_baseline.copy()
                
                for i, gene in enumerate(biomarker_genes):
                    effect_direction = self._get_gene_effect_direction(gene)
                    
                    if 'up' in effect_direction.lower():
                        age_effect = aging_factor
                    elif 'down' in effect_direction.lower():
                        age_effect = -aging_factor
                    else:
                        age_effect = np.random.uniform(-0.02, 0.02)
                    
                    # Add measurement noise
                    noise = np.random.normal(0, 0.12)
                    expression[i] *= (1 + age_effect + noise)
                
                # Create sample data
                sample_data = {gene: expression[i] for i, gene in enumerate(all_genes)}
                sample_data.update({
                    'sample_id': f"LONG_{individual:03d}_T{timepoint}",
                    'individual_id': f"SUBJ_{individual:03d}",
                    'timepoint': timepoint,
                    'age': round(current_age, 1),
                    'sex': sex,
                    'years_from_baseline': timepoint * 3,
                    'dataset_type': 'longitudinal'
                })
                
                data_rows.append(sample_data)
        
        # Create DataFrames
        full_df = pd.DataFrame(data_rows)
        expr_cols = [col for col in full_df.columns if col.startswith(('GENE_', 'CDKN', 'TP53', 'ATM', 'BRCA', 'SIRT', 'PPARGC', 'TERT', 'ATG', 'IL', 'TNF', 'MTOR', 'FOXO', 'DNMT', 'TET', 'MMP', 'COL', 'NANOG', 'SOX', 'POU', 'KLF', 'MYC', 'NLRP', 'CXCL', 'HSPA', 'POT', 'BECN', 'SQSTM', 'HDAC'))]
        
        expr_df = full_df[expr_cols]
        metadata_df = full_df.drop(columns=expr_cols)
        
        return BenchmarkDataset(
            name="longitudinal_aging",
            description=f"Longitudinal aging study with {n_individuals} individuals over {n_timepoints} timepoints",
            expression_data=expr_df,
            metadata=metadata_df,
            source="synthetic_longitudinal",
            species="human",
            tissue="blood",
            dataset_type="longitudinal",
            n_samples=len(data_rows),
            n_genes=len(expr_cols)
        )
    
    def _get_gene_effect_direction(self, gene):
        """Get expected effect direction for a gene from biomarker panel"""
        if 'biomarkers' not in self.biomarker_panel:
            return 'context_dependent'
        
        for pathway, data in self.biomarker_panel['biomarkers'].items():
            if 'markers' in data:
                for marker in data['markers']:
                    if marker['gene'] == gene:
                        return marker.get('effect', 'context_dependent')
        
        return 'context_dependent'
    
    def calculate_enhanced_rejuvenation_score(self, dataset: BenchmarkDataset):
        """
        Calculate rejuvenation scores using enhanced biomarker panel
        """
        biomarker_genes = list(self._extract_all_genes(self.biomarker_panel))
        available_biomarkers = [gene for gene in biomarker_genes 
                              if gene in dataset.expression_data.columns]
        
        if len(available_biomarkers) == 0:
            print("⚠️ No biomarker genes found in dataset")
            return np.random.random(len(dataset.expression_data)) * 0.5 + 0.25
        
        scores = []
        
        for idx in range(len(dataset.expression_data)):
            score = 0.5  # Neutral baseline
            
            for gene in available_biomarkers:
                expr_value = dataset.expression_data.iloc[idx][gene]
                age = dataset.metadata.iloc[idx]['age']
                
                # Normalize expression relative to age expectation
                effect_direction = self._get_gene_effect_direction(gene)
                
                # Enhanced scoring logic
                age_factor = (age - 25) / 50  # 0 to 1 for ages 25-75
                
                if 'up' in effect_direction.lower():
                    # Lower than expected = more rejuvenated
                    expected_expr = np.log(age) * 2.5
                    deviation = (expected_expr - expr_value) / expected_expr
                    score += deviation * 0.05
                elif 'down' in effect_direction.lower():
                    # Higher than expected = more rejuvenated
                    expected_expr = np.log(80 - age + 1) * 2.5
                    deviation = (expr_value - expected_expr) / expected_expr
                    score += deviation * 0.05
                else:
                    # Context-dependent: small random contribution
                    score += np.random.uniform(-0.02, 0.02)
            
            scores.append(np.clip(score, 0, 1))
        
        return np.array(scores)
    
    def evaluate_dataset_performance(self, dataset: BenchmarkDataset):
        """
        Comprehensive evaluation of dataset performance using enhanced statistics
        """
        print(f"\n📊 Evaluating {dataset.name}...")
        
        # Calculate rejuvenation scores
        scores = self.calculate_enhanced_rejuvenation_score(dataset)
        ages = dataset.metadata['age'].values
        
        # Basic correlations
        from scipy.stats import pearsonr, spearmanr
        age_corr_pearson, age_p_pearson = pearsonr(scores, ages)
        age_corr_spearman, age_p_spearman = spearmanr(scores, ages)
        
        # Enhanced statistical analysis
        results = {
            'dataset_name': dataset.name,
            'n_samples': dataset.n_samples,
            'n_genes': dataset.n_genes,
            'dataset_type': dataset.dataset_type,
            'basic_stats': {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'score_range': np.max(scores) - np.min(scores)
            },
            'age_correlations': {
                'pearson_r': age_corr_pearson,
                'pearson_p': age_p_pearson,
                'spearman_r': age_corr_spearman,
                'spearman_p': age_p_spearman
            }
        }
        
        # Dataset-specific analyses
        if dataset.dataset_type == 'intervention':
            results['intervention_analysis'] = self._analyze_intervention_effect(dataset, scores)
        elif dataset.dataset_type == 'longitudinal':
            results['longitudinal_analysis'] = self._analyze_longitudinal_trends(dataset, scores)
        
        # Age stratification analysis
        results['age_stratification'] = self._analyze_age_groups(ages, scores)
        
        return results
    
    def _analyze_intervention_effect(self, dataset: BenchmarkDataset, scores):
        """Analyze intervention effects using enhanced statistics"""
        treatment_col = 'treatment'
        if treatment_col not in dataset.metadata.columns:
            return None
        
        control_mask = dataset.metadata[treatment_col] == 'control'
        treated_mask = ~control_mask
        
        control_scores = scores[control_mask]
        treated_scores = scores[treated_mask]
        
        if len(control_scores) == 0 or len(treated_scores) == 0:
            return None
        
        # Use enhanced statistics for comparison
        comparison_results = self.stats_engine.comprehensive_group_comparison(
            treated_scores, control_scores,
            group_names=("Treated", "Control"),
            n_permutations=5000,
            n_bootstrap=2000
        )
        
        return {
            'control_mean': np.mean(control_scores),
            'treated_mean': np.mean(treated_scores),
            'difference': np.mean(treated_scores) - np.mean(control_scores),
            'enhanced_comparison': comparison_results,
            'practical_significance': abs(comparison_results['effect_size']['hedges_g']) > 0.5
        }
    
    def _analyze_longitudinal_trends(self, dataset: BenchmarkDataset, scores):
        """Analyze longitudinal aging trends"""
        if 'individual_id' not in dataset.metadata.columns:
            return None
        
        individual_trends = []
        
        for individual in dataset.metadata['individual_id'].unique():
            mask = dataset.metadata['individual_id'] == individual
            individual_scores = scores[mask]
            individual_ages = dataset.metadata.loc[mask, 'age'].values
            
            if len(individual_scores) > 2:  # Need at least 3 timepoints
                # Calculate aging trajectory slope
                slope, intercept = np.polyfit(individual_ages, individual_scores, 1)
                individual_trends.append(slope)
        
        if len(individual_trends) > 0:
            return {
                'n_individuals': len(individual_trends),
                'mean_aging_slope': np.mean(individual_trends),
                'std_aging_slope': np.std(individual_trends),
                'positive_aging_fraction': np.mean(np.array(individual_trends) > 0),
                'slope_distribution': individual_trends
            }
        
        return None
    
    def _analyze_age_groups(self, ages, scores):
        """Analyze performance across age groups"""
        # Define age groups
        young_mask = ages < 40
        middle_mask = (ages >= 40) & (ages < 65)
        old_mask = ages >= 65
        
        groups = {
            'young': scores[young_mask],
            'middle': scores[middle_mask],
            'old': scores[old_mask]
        }
        
        # Remove empty groups
        groups = {name: scores_group for name, scores_group in groups.items() 
                 if len(scores_group) > 0}
        
        if len(groups) < 2:
            return None
        
        results = {}
        for name, group_scores in groups.items():
            results[name] = {
                'n': len(group_scores),
                'mean': np.mean(group_scores),
                'std': np.std(group_scores)
            }
        
        # Pairwise comparisons using enhanced statistics
        if len(groups) >= 2:
            group_names = list(groups.keys())
            for i in range(len(group_names)):
                for j in range(i+1, len(group_names)):
                    name1, name2 = group_names[i], group_names[j]
                    comparison = self.stats_engine.comprehensive_group_comparison(
                        groups[name1], groups[name2],
                        group_names=(name1, name2),
                        n_permutations=2000,
                        n_bootstrap=1000
                    )
                    results[f'{name1}_vs_{name2}'] = comparison
        
        return results
    
    def run_comprehensive_benchmark(self):
        """
        Run complete benchmark suite with enhanced datasets and analysis
        """
        print("🚀 TIER 1 Enhanced Benchmarking Suite")
        print("=" * 60)
        
        # Generate enhanced datasets
        datasets = []
        
        # GTEx-like datasets (multiple tissues)
        for tissue in ['liver', 'muscle', 'brain']:
            datasets.append(self.generate_enhanced_gtex_dataset(tissue=tissue))
        
        # Intervention studies
        for intervention in ['rapamycin', 'senolytics']:
            datasets.append(self.generate_enhanced_intervention_dataset(intervention=intervention))
        
        # Longitudinal study
        datasets.append(self.generate_longitudinal_dataset())
        
        # Evaluate each dataset
        all_results = {}
        
        for dataset in datasets:
            results = self.evaluate_dataset_performance(dataset)
            all_results[dataset.name] = results
            self._print_dataset_summary(results)
        
        # Generate comprehensive report
        self._generate_benchmark_report(all_results)
        
        return all_results
    
    def _print_dataset_summary(self, results):
        """Print summary for individual dataset"""
        print(f"\n📋 {results['dataset_name']} Results:")
        print(f"  Samples: {results['n_samples']}, Genes: {results['n_genes']}")
        print(f"  Mean score: {results['basic_stats']['mean_score']:.3f} ± {results['basic_stats']['std_score']:.3f}")
        print(f"  Score range: {results['basic_stats']['min_score']:.3f} - {results['basic_stats']['max_score']:.3f}")
        print(f"  Age correlation (Pearson): r = {results['age_correlations']['pearson_r']:.3f}, p = {results['age_correlations']['pearson_p']:.4f}")
        
        if 'intervention_analysis' in results and results['intervention_analysis']:
            intervention = results['intervention_analysis']
            print(f"  Intervention effect: {intervention['difference']:.3f} (Hedges' g = {intervention['enhanced_comparison']['effect_size']['hedges_g']:.3f})")
    
    def _generate_benchmark_report(self, all_results):
        """Generate comprehensive benchmark report"""
        print(f"\n📊 COMPREHENSIVE BENCHMARK REPORT")
        print("=" * 60)
        
        # Summary statistics
        total_samples = sum(r['n_samples'] for r in all_results.values())
        mean_correlations = [r['age_correlations']['pearson_r'] for r in all_results.values()]
        
        print(f"Total samples analyzed: {total_samples}")
        print(f"Mean age correlation: {np.mean(mean_correlations):.3f} ± {np.std(mean_correlations):.3f}")
        print(f"Correlation range: {np.min(mean_correlations):.3f} to {np.max(mean_correlations):.3f}")
        
        # Dataset type performance
        print(f"\n📈 Performance by Dataset Type:")
        type_performance = {}
        for name, results in all_results.items():
            dtype = results['dataset_type']
            if dtype not in type_performance:
                type_performance[dtype] = []
            type_performance[dtype].append(results['age_correlations']['pearson_r'])
        
        for dtype, correlations in type_performance.items():
            print(f"  {dtype}: {np.mean(correlations):.3f} ± {np.std(correlations):.3f} (n={len(correlations)})")
        
        # Intervention effects summary
        intervention_effects = []
        for name, results in all_results.items():
            if 'intervention_analysis' in results and results['intervention_analysis']:
                effect = results['intervention_analysis']['enhanced_comparison']['effect_size']['hedges_g']
                intervention_effects.append(abs(effect))
        
        if intervention_effects:
            print(f"\n💊 Intervention Effects (|Hedges' g|):")
            print(f"  Mean effect size: {np.mean(intervention_effects):.3f}")
            print(f"  Large effects (>0.8): {np.sum(np.array(intervention_effects) > 0.8)}/{len(intervention_effects)}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"/home/pi/projects/benchmarks/results/enhanced_benchmark_{timestamp}.json"
        
        # Make directory if it doesn't exist
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_results_for_json(all_results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        print("\n✅ Enhanced benchmarking completed successfully!")
    
    def _prepare_results_for_json(self, results):
        """Prepare results for JSON serialization"""
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        return convert_numpy(results)

if __name__ == "__main__":
    # Run enhanced benchmarking
    benchmark = EnhancedTIER1Benchmark()
    results = benchmark.run_comprehensive_benchmark()