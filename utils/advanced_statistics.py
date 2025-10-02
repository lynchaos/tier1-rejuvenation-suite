"""
Advanced Statistical Methods for TIER 1 Rejuvenation Suite
========================================================

Implements permutation tests, BCa bootstrap confidence intervals,
and other advanced statistical methods for robust analysis.

References:
- Good, P. (2000). Permutation Tests: A Practical Guide to Resampling Methods
- Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap
- DiCiccio, T.J. & Efron, B. (1996). Bootstrap confidence intervals
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

class AdvancedStatistics:
    """
    Collection of advanced statistical methods for rejuvenation analysis.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def permutation_test_two_samples(self, x, y, statistic='mean_diff', n_permutations=10000):
        """
        Permutation test for comparing two independent samples.
        
        Parameters:
        -----------
        x, y : array-like
            Two samples to compare
        statistic : str or callable
            Test statistic ('mean_diff', 'median_diff', or custom function)
        n_permutations : int
            Number of permutations
            
        Returns:
        --------
        result : dict
            Contains observed statistic, p-value, and permutation distribution
        """
        x, y = np.asarray(x), np.asarray(y)
        
        # Define test statistic
        if statistic == 'mean_diff':
            stat_func = lambda a, b: np.mean(a) - np.mean(b)
        elif statistic == 'median_diff':
            stat_func = lambda a, b: np.median(a) - np.median(b)
        elif callable(statistic):
            stat_func = statistic
        else:
            raise ValueError("statistic must be 'mean_diff', 'median_diff', or callable")
        
        # Observed test statistic
        observed = stat_func(x, y)
        
        # Combine samples
        combined = np.concatenate([x, y])
        n_x, n_y = len(x), len(y)
        
        # Permutation distribution
        perm_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_x = combined[:n_x]
            perm_y = combined[n_x:]
            perm_stats.append(stat_func(perm_x, perm_y))
        
        perm_stats = np.array(perm_stats)
        
        # Two-tailed p-value
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
        
        return {
            'observed_statistic': observed,
            'p_value': p_value,
            'permutation_distribution': perm_stats,
            'n_permutations': n_permutations,
            'statistic_type': str(statistic)
        }
    
    def permutation_test_correlation(self, x, y, n_permutations=10000):
        """
        Permutation test for Pearson correlation coefficient.
        
        Parameters:
        -----------
        x, y : array-like
            Variables to test for correlation
        n_permutations : int
            Number of permutations
            
        Returns:
        --------
        result : dict
            Contains observed correlation, p-value, and permutation distribution
        """
        x, y = np.asarray(x), np.asarray(y)
        
        # Observed correlation
        observed_r = np.corrcoef(x, y)[0, 1]
        
        # Permutation distribution
        perm_correlations = []
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y)
            perm_r = np.corrcoef(x, y_perm)[0, 1]
            perm_correlations.append(perm_r)
        
        perm_correlations = np.array(perm_correlations)
        
        # Two-tailed p-value
        p_value = np.mean(np.abs(perm_correlations) >= np.abs(observed_r))
        
        return {
            'observed_correlation': observed_r,
            'p_value': p_value,
            'permutation_distribution': perm_correlations,
            'n_permutations': n_permutations
        }
    
    def bootstrap_bca_ci(self, data, statistic=np.mean, confidence_level=0.95, n_bootstrap=2000):
        """
        Bias-corrected and accelerated (BCa) bootstrap confidence interval.
        
        Parameters:
        -----------
        data : array-like
            Sample data
        statistic : callable
            Statistic function to compute CI for
        confidence_level : float
            Confidence level (0 < confidence_level < 1)
        n_bootstrap : int
            Number of bootstrap samples
            
        Returns:
        --------
        result : dict
            Contains CI bounds, bias correction, and acceleration
        """
        try:
            from scipy.stats import bootstrap as scipy_bootstrap
            
            # Use SciPy's implementation if available (preferred)
            res = scipy_bootstrap(
                (data,), 
                statistic, 
                method='BCa', 
                confidence_level=confidence_level,
                n_resamples=n_bootstrap,
                random_state=self.random_state
            )
            
            return {
                'confidence_interval': (res.confidence_interval.low, res.confidence_interval.high),
                'confidence_level': confidence_level,
                'method': 'BCa_scipy',
                'n_bootstrap': n_bootstrap
            }
            
        except (ImportError, AttributeError):
            # Fallback to manual implementation
            return self._manual_bca_ci(data, statistic, confidence_level, n_bootstrap)
    
    def _manual_bca_ci(self, data, statistic, confidence_level, n_bootstrap):
        """
        Manual implementation of BCa bootstrap CI.
        """
        data = np.asarray(data)
        n = len(data)
        
        # Original statistic
        theta_hat = statistic(data)
        
        # Bootstrap samples
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = resample(data, n_samples=n, random_state=None)
            bootstrap_stats.append(statistic(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Bias correction (z_0)
        z_0 = stats.norm.ppf(np.mean(bootstrap_stats < theta_hat))
        
        # Acceleration (a_hat) using jackknife
        jackknife_stats = []
        for i in range(n):
            jackknife_sample = np.delete(data, i)
            jackknife_stats.append(statistic(jackknife_sample))
        
        jackknife_stats = np.array(jackknife_stats)
        jackknife_mean = np.mean(jackknife_stats)
        
        # Acceleration parameter
        numerator = np.sum((jackknife_mean - jackknife_stats)**3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats)**2))**1.5
        
        if denominator == 0:
            a_hat = 0
        else:
            a_hat = numerator / denominator
        
        # Adjusted percentiles
        alpha = 1 - confidence_level
        z_alpha_2 = stats.norm.ppf(alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
        
        # BCa percentiles
        alpha_1 = stats.norm.cdf(z_0 + (z_0 + z_alpha_2) / (1 - a_hat * (z_0 + z_alpha_2)))
        alpha_2 = stats.norm.cdf(z_0 + (z_0 + z_1_alpha_2) / (1 - a_hat * (z_0 + z_1_alpha_2)))
        
        # Clip to valid range
        alpha_1 = np.clip(alpha_1, 0.001, 0.999)
        alpha_2 = np.clip(alpha_2, 0.001, 0.999)
        
        # Compute CI
        lower_bound = np.percentile(bootstrap_stats, alpha_1 * 100)
        upper_bound = np.percentile(bootstrap_stats, alpha_2 * 100)
        
        return {
            'confidence_interval': (lower_bound, upper_bound),
            'confidence_level': confidence_level,
            'method': 'BCa_manual',
            'bias_correction': z_0,
            'acceleration': a_hat,
            'n_bootstrap': n_bootstrap
        }
    
    def calculate_cohen_d(self, group1, group2):
        """
        Calculate Cohen's d effect size.
        
        Parameters:
        -----------
        group1, group2 : array-like
            Two groups to compare
            
        Returns:
        --------
        cohens_d : float
            Cohen's d effect size
        """
        x1, x2 = np.asarray(group1), np.asarray(group2)
        
        # Calculate means and standard deviations
        mean1, mean2 = np.mean(x1), np.mean(x2)
        std1, std2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(x1), len(x2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        return cohens_d
    
    def calculate_hedges_g(self, group1, group2):
        """
        Calculate Hedges' g effect size (bias-corrected Cohen's d).
        Better for small samples than Cohen's d.
        
        Parameters:
        -----------
        group1, group2 : array-like
            Two groups to compare
            
        Returns:
        --------
        dict : Effect size results including Hedges' g, Cohen's d, and interpretation
        """
        x1, x2 = np.asarray(group1), np.asarray(group2)
        n1, n2 = len(x1), len(x2)
        
        # Calculate Cohen's d first
        cohens_d = self.calculate_cohen_d(x1, x2)
        
        # Bias correction factor for Hedges' g
        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
        hedges_g = cohens_d * correction_factor
        
        # Effect size interpretation
        g_magnitude = abs(hedges_g)
        if g_magnitude < 0.2:
            interpretation = "negligible"
        elif g_magnitude < 0.5:
            interpretation = "small"  
        elif g_magnitude < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'cohens_d': cohens_d,
            'hedges_g': hedges_g,
            'correction_factor': correction_factor,
            'interpretation': interpretation,
            'sample_sizes': (n1, n2),
            'magnitude': g_magnitude
        }
    
    def comprehensive_group_comparison(self, group1, group2, 
                                     group_names=("Group 1", "Group 2"),
                                     n_permutations=10000, n_bootstrap=2000):
        """
        Comprehensive statistical comparison between two groups.
        
        Includes permutation tests, bootstrap CIs, effect sizes with CIs,
        and traditional parametric tests for comparison.
        
        Parameters:
        -----------
        group1, group2 : array-like
            Two groups to compare
        group_names : tuple
            Names for the two groups
        n_permutations : int
            Number of permutations for permutation test
        n_bootstrap : int
            Number of bootstrap samples for CIs
            
        Returns:
        --------
        dict : Comprehensive comparison results
        """
        x1, x2 = np.asarray(group1), np.asarray(group2)
        
        # Descriptive statistics
        descriptive = {
            group_names[0]: {
                'n': len(x1),
                'mean': np.mean(x1),
                'std': np.std(x1, ddof=1),
                'median': np.median(x1),
                'iqr': np.percentile(x1, 75) - np.percentile(x1, 25)
            },
            group_names[1]: {
                'n': len(x2),
                'mean': np.mean(x2),
                'std': np.std(x2, ddof=1),
                'median': np.median(x2),
                'iqr': np.percentile(x2, 75) - np.percentile(x2, 25)
            }
        }
        
        # Permutation test
        perm_result = self.permutation_test_two_samples(x1, x2, n_permutations=n_permutations)
        
        # Effect sizes with bootstrap CIs
        effect_result = self.calculate_hedges_g(x1, x2)
        
        # Bootstrap CI for effect size
        effect_bootstrap = []
        for _ in range(n_bootstrap):
            boot1 = resample(x1, random_state=self.random_state + _)
            boot2 = resample(x2, random_state=self.random_state + _ + 1000)
            boot_effect = self.calculate_hedges_g(boot1, boot2)
            effect_bootstrap.append(boot_effect['hedges_g'])
        
        effect_ci_lower = np.percentile(effect_bootstrap, 2.5)
        effect_ci_upper = np.percentile(effect_bootstrap, 97.5)
        
        # Bootstrap CIs for means
        mean1_ci = self.bootstrap_bca_ci(x1, statistic=np.mean, n_bootstrap=n_bootstrap)
        mean2_ci = self.bootstrap_bca_ci(x2, statistic=np.mean, n_bootstrap=n_bootstrap)
        
        # Traditional tests for comparison
        t_stat, t_pval = stats.ttest_ind(x1, x2)
        u_stat, u_pval = stats.mannwhitneyu(x1, x2, alternative='two-sided')
        
        return {
            'group_names': group_names,
            'descriptive_statistics': descriptive,
            'permutation_test': {
                'statistic': perm_result['observed_statistic'],
                'p_value': perm_result['p_value'],
                'method': 'two-sided permutation test'
            },
            'effect_size': {
                **effect_result,
                'confidence_interval': (effect_ci_lower, effect_ci_upper),
                'method': 'Hedges\' g with bootstrap CI'
            },
            'confidence_intervals': {
                group_names[0]: mean1_ci,
                group_names[1]: mean2_ci
            },
            'traditional_tests': {
                't_test': {'statistic': t_stat, 'p_value': t_pval},
                'mann_whitney': {'statistic': u_stat, 'p_value': u_pval}
            },
            'recommendations': self._generate_comparison_recommendations(perm_result, effect_result, len(x1), len(x2))
        }
    
    def _generate_comparison_recommendations(self, perm_result, effect_result, n1, n2):
        """Generate recommendations based on statistical results"""
        recommendations = []
        
        # Sample size recommendations
        if n1 < 20 or n2 < 20:
            recommendations.append("Small sample sizes: Hedges' g preferred over Cohen's d")
            recommendations.append("Consider permutation test over t-test for robustness")
        
        # Effect size interpretation
        if effect_result['magnitude'] < 0.2:
            recommendations.append("Effect size is negligible - practical significance questionable")
        elif effect_result['magnitude'] > 0.8:
            recommendations.append("Large effect size detected - likely practical significance")
        
        # Statistical significance
        if perm_result['p_value'] < 0.001:
            recommendations.append("Highly significant difference (p < 0.001)")
        elif perm_result['p_value'] < 0.05:
            recommendations.append("Statistically significant difference detected")
        else:
            recommendations.append("No significant difference detected")
        
        return recommendations
    
    def multiple_testing_correction(self, p_values, method='fdr_bh'):
        """
        Multiple testing correction for p-values.
        
        Parameters:
        -----------
        p_values : array-like
            Uncorrected p-values
        method : str
            Correction method ('bonferroni', 'fdr_bh', 'fdr_by')
            
        Returns:
        --------
        result : dict
            Contains corrected p-values and significant tests
        """
        p_values = np.asarray(p_values)
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            corrected_p = np.minimum(p_values * n_tests, 1.0)
            
        elif method == 'fdr_bh':
            # Benjamini-Hochberg procedure
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            # Calculate BH critical values
            corrected_p = np.zeros_like(p_values)
            for i in range(n_tests):
                corrected_p[sorted_indices[i]] = min(
                    sorted_p[i] * n_tests / (i + 1),
                    1.0
                )
            
            # Ensure monotonicity
            sorted_corrected = corrected_p[sorted_indices]
            for i in range(n_tests - 2, -1, -1):
                if sorted_corrected[i] > sorted_corrected[i + 1]:
                    sorted_corrected[i] = sorted_corrected[i + 1]
            
            corrected_p[sorted_indices] = sorted_corrected
            
        elif method == 'fdr_by':
            # Benjamini-Yekutieli procedure
            c_n = np.sum(1.0 / np.arange(1, n_tests + 1))
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected_p = np.zeros_like(p_values)
            for i in range(n_tests):
                corrected_p[sorted_indices[i]] = min(
                    sorted_p[i] * n_tests * c_n / (i + 1),
                    1.0
                )
        
        else:
            raise ValueError("method must be 'bonferroni', 'fdr_bh', or 'fdr_by'")
        
        # Determine significance (alpha = 0.05)
        significant = corrected_p < 0.05
        
        return {
            'corrected_p_values': corrected_p,
            'significant': significant,
            'method': method,
            'n_significant': np.sum(significant),
            'n_tests': n_tests
        }
    
    def robust_correlation(self, x, y, method='spearman'):
        """
        Calculate robust correlation with confidence intervals.
        
        Parameters:
        -----------
        x, y : array-like
            Variables to correlate
        method : str
            Correlation method ('spearman', 'kendall', 'pearson')
            
        Returns:
        --------
        result : dict
            Contains correlation, p-value, and bootstrap CI
        """
        x, y = np.asarray(x), np.asarray(y)
        
        # Calculate correlation
        if method == 'pearson':
            corr, p_value = stats.pearsonr(x, y)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(x, y)
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(x, y)
        else:
            raise ValueError("method must be 'pearson', 'spearman', or 'kendall'")
        
        # Bootstrap CI for correlation
        def correlation_statistic(data):
            x_boot, y_boot = data[0], data[1]
            if method == 'pearson':
                return stats.pearsonr(x_boot, y_boot)[0]
            elif method == 'spearman':
                return stats.spearmanr(x_boot, y_boot)[0]
            elif method == 'kendall':
                return stats.kendalltau(x_boot, y_boot)[0]
        
        ci_result = self.bootstrap_bca_ci(
            np.column_stack([x, y]),
            statistic=correlation_statistic
        )
        
        return {
            'correlation': corr,
            'p_value': p_value,
            'method': method,
            'confidence_interval': ci_result['confidence_interval'],
            'sample_size': len(x)
        }

# Testing and validation functions
def validate_statistical_methods():
    """
    Validate statistical methods with known distributions.
    """
    print("Validating Advanced Statistical Methods")
    print("=" * 40)
    
    stats_engine = AdvancedStatistics(random_state=42)
    
    # Test 1: Permutation test with known difference
    np.random.seed(42)
    x1 = np.random.normal(0, 1, 100)
    x2 = np.random.normal(0.5, 1, 100)  # Known difference
    
    perm_result = stats_engine.permutation_test_two_samples(x1, x2)
    print(f"Permutation test (known difference):")
    print(f"  Observed difference: {perm_result['observed_statistic']:.3f}")
    print(f"  P-value: {perm_result['p_value']:.4f}")
    
    # Test 2: BCa bootstrap CI
    data = np.random.gamma(2, 2, 200)  # Skewed distribution
    ci_result = stats_engine.bootstrap_bca_ci(data, statistic=np.mean)
    print(f"\nBCa Bootstrap CI (mean of gamma distribution):")
    print(f"  Sample mean: {np.mean(data):.3f}")
    print(f"  95% CI: ({ci_result['confidence_interval'][0]:.3f}, {ci_result['confidence_interval'][1]:.3f})")
    
    # Test 3: Effect size
    effect_result = stats_engine.effect_size_cohens_d(x1, x2)
    print(f"\nEffect size (Cohen's d):")
    print(f"  Cohen's d: {effect_result['cohens_d']:.3f}")
    print(f"  Interpretation: {effect_result['interpretation']}")
    
    # Test 4: Multiple testing correction
    p_vals = np.random.uniform(0, 0.1, 20)  # Mix of significant/non-significant
    mt_result = stats_engine.multiple_testing_correction(p_vals, method='fdr_bh')
    print(f"\nMultiple testing correction (FDR-BH):")
    print(f"  Original significant: {np.sum(p_vals < 0.05)}")
    print(f"  After correction: {mt_result['n_significant']}")

if __name__ == "__main__":
    validate_statistical_methods()