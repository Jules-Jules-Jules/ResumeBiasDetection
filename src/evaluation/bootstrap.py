from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_mean(
    data: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    return {
        'mean': float(np.mean(data)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'std': float(np.std(data)),
        'n': len(data),
        'n_bootstrap': n_bootstrap,
        'confidence': confidence
    }


def bootstrap_difference(
    data1: np.ndarray,
    data2: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    
    bootstrap_diffs = []
    n1, n2 = len(data1), len(data2)
    
    for _ in range(n_bootstrap):
        sample1 = rng.choice(data1, size=n1, replace=True)
        sample2 = rng.choice(data2, size=n2, replace=True)
        diff = np.mean(sample1) - np.mean(sample2)
        bootstrap_diffs.append(diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
    ci_upper = np.percentile(bootstrap_diffs, upper_percentile)
    
    observed_diff = np.mean(data1) - np.mean(data2)
    
    pooled_std = np.sqrt(((n1 - 1) * np.var(data1, ddof=1) + (n2 - 1) * np.var(data2, ddof=1)) / (n1 + n2 - 2))
    cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0.0
    
    return {
        'difference': float(observed_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'cohens_d': float(cohens_d),
        'n1': n1,
        'n2': n2,
        'n_bootstrap': n_bootstrap,
        'confidence': confidence,
        'significant': not (ci_lower <= 0 <= ci_upper)
    }


def bootstrap_selection_rate_gap(
    df: pd.DataFrame,
    group1: str,
    group2: str,
    k: int,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    df_k = df[df['k'] == k].copy()
    
    rates1 = df_k[df_k['demographic_group'] == group1]['selection_rate'].values
    rates2 = df_k[df_k['demographic_group'] == group2]['selection_rate'].values
    
    if len(rates1) == 0 or len(rates2) == 0:
        return {
            'gap': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'significant': False,
            'error': 'Missing data for one or both groups'
        }
    
    # Use bootstrap_difference
    result = bootstrap_difference(rates1, rates2, n_bootstrap, confidence, seed)
    result['gap'] = result.pop('difference')
    result['k'] = k
    result['group1'] = group1
    result['group2'] = group2
    
    return result


def bootstrap_counterfactual_deltas(
    deltas_df: pd.DataFrame,
    delta_col: str,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute bootstrap CI for mean counterfactual score delta.
    
    Args:
        deltas_df: Deltas dataframe (from compute_within_base_deltas)
        delta_col: Column with score deltas
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed
        
    Returns:
        Dictionary with mean delta and CI
    """
    deltas = deltas_df[delta_col].dropna().values
    
    if len(deltas) == 0:
        return {
            'mean': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'error': 'No valid deltas'
        }
    
    return bootstrap_mean(deltas, n_bootstrap, confidence, seed)


def compute_bootstrap_summary(
    selection_rates_df: pd.DataFrame,
    deltas_df: pd.DataFrame,
    comparisons: List[Tuple[str, str]] = [
        ('white_male', 'black_female'),
        ('white_male', 'black_male'),
        ('white_female', 'black_female')
    ],
    k_values: List[int] = [1, 3, 5, 8],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, List[Dict]]:
    results = {
        'selection_gaps': [],
        'counterfactual_deltas': []
    }
    
    print(f"\nComputing bootstrap CIs ({n_bootstrap} samples, {int(confidence*100)}% confidence)")
    
    print("\n1. Selection rate gaps:")
    for k in k_values:
        for group1, group2 in comparisons:
            print(f"   k={k}, {group1} vs {group2}")
            gap_result = bootstrap_selection_rate_gap(
                selection_rates_df, group1, group2, k,
                n_bootstrap, confidence, seed
            )
            results['selection_gaps'].append(gap_result)
    
    print("\n2. Counterfactual score deltas:")
    delta_cols = [col for col in deltas_df.columns if '_minus_' in col]
    
    for delta_col in delta_cols:
        print(f"   {delta_col}...")
        delta_result = bootstrap_counterfactual_deltas(
            deltas_df, delta_col,
            n_bootstrap, confidence, seed
        )
        delta_result['delta_type'] = delta_col
        results['counterfactual_deltas'].append(delta_result)
    
    # Also compute CI for range
    if 'range' in deltas_df.columns:
        print(f"   range (max-min)...")
        range_result = bootstrap_mean(
            deltas_df['range'].dropna().values,
            n_bootstrap, confidence, seed
        )
        range_result['delta_type'] = 'range'
        results['counterfactual_deltas'].append(range_result)
    
    return results


if __name__ == "__main__":
    from ..config.paths import DATA_DIR
    import json
    
    print("Loading data")
    selection_rates = pd.read_csv(DATA_DIR / "processed" / "phase3_score_embed_selection_rates.csv")
    deltas = pd.read_csv(DATA_DIR / "processed" / "phase3_score_embed_deltas.csv")
    
    results = compute_bootstrap_summary(selection_rates, deltas)
    
    out_path = DATA_DIR / "processed" / "phase3_bias_bootstrap.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved bootstrap results to {out_path.name}")
