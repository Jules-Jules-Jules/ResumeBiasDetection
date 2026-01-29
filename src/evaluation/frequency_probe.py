from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from scipy import stats


def compute_name_features(names: List[str]) -> pd.DataFrame:
    results = []
    
    for name in names:
        first_name = name.split()[0] if ' ' in name else name
        
        results.append({
            'full_name': name,
            'first_name': first_name,
            'name_length': len(first_name),
            'n_syllables_proxy': sum(1 for c in first_name.lower() if c in 'aeiou'),
            'has_uppercase_mid': any(c.isupper() for c in first_name[1:]),
            'n_words': len(name.split()),
        })
    
    return pd.DataFrame(results)


def test_length_bias(
    deltas_df: pd.DataFrame,
    tokenization_df: pd.DataFrame,
    score_col: str = 'white_male_minus_white_female'
) -> Dict:
    grouped = tokenization_df.groupby('demographic_group').agg({
        'n_subtokens': 'mean',
        'name_length': 'mean',
        'is_single_token': lambda x: (x.sum() / len(x)) * 100
    }).reset_index()
    
    # Compute mean deltas from counterfactual analysis
    if score_col in deltas_df.columns:
        mean_delta = deltas_df[score_col].mean()
        median_delta = deltas_df[score_col].median()
    else:
        mean_delta = None
        median_delta = None
    
    results = {
        'tokenization_summary': grouped.to_dict('records'),
        'mean_delta': mean_delta,
        'median_delta': median_delta,
    }
    
    return results


def analyze_subgroup_bias(
    deltas_df: pd.DataFrame,
    tokenization_df: pd.DataFrame,
    groupby_col: str = 'n_subtokens',
    delta_col: str = 'range'
) -> pd.DataFrame:
    """
    Analyze bias magnitude by subgroups (e.g., single-token vs multi-token names).
    
    Args:
        deltas_df: Counterfactual deltas
        tokenization_df: Tokenization results
        groupby_col: Column to group by
        delta_col: Which delta to analyze
        
    Returns:
        DataFrame with subgroup statistics
    """
    # This would require joining on name/demographic group
    # For now, return summary by tokenization group
    
    summary = tokenization_df.groupby([groupby_col, 'demographic_group']).size().reset_index(name='count')
    
    return summary


def run_frequency_probe(
    deltas_path: Path,
    tokenization_path: Path,
    output_path: Path = None
) -> Dict:
    """
    Run complete frequency probe.
    
    Args:
        deltas_path: Path to phase3_*_deltas.csv
        tokenization_path: Path to phase3_tokenization_probe.csv
        output_path: Where to save results
        
    Returns:
        Dictionary with probe results
    """
    print("=" * 70)
    print("FREQUENCY PROBE")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading deltas from {deltas_path.name}...")
    deltas_df = pd.read_csv(deltas_path)
    
    print(f"Loading tokenization from {tokenization_path.name}...")
    tok_df = pd.read_csv(tokenization_path)
    
    # Compute correlations
    results = {}
    
    # 1. Tokenization complexity by demographic group
    print("\n1. TOKENIZATION COMPLEXITY BY DEMOGRAPHIC GROUP")
    print("-" * 70)
    
    tok_summary = tok_df.groupby('demographic_group').agg({
        'n_subtokens': ['mean', 'std', 'min', 'max'],
        'name_length': ['mean', 'std'],
        'is_single_token': lambda x: (x.sum() / len(x)) * 100
    }).round(3)
    
    print(tok_summary)
    results['tokenization_by_group'] = tok_summary.to_dict()
    
    # 2. Delta magnitudes
    print("\n2. COUNTERFACTUAL DELTA MAGNITUDES")
    print("-" * 70)
    
    delta_cols = [col for col in deltas_df.columns if 'minus' in col or col == 'range']
    
    for col in delta_cols:
        if col in deltas_df.columns:
            print(f"\n{col}:")
            print(f"  Mean:   {deltas_df[col].mean():.6f}")
            print(f"  Median: {deltas_df[col].median():.6f}")
            print(f"  Std:    {deltas_df[col].std():.6f}")
            print(f"  95th %: {deltas_df[col].quantile(0.95):.6f}")
    
    # 3. Hypothesis: Multi-token names get lower scores?
    print("\n3. HYPOTHESIS TEST: Multi-token names penalized?")
    print("-" * 70)
    
    # Count multi-token names by group
    multi_token = tok_df[tok_df['n_subtokens'] > 1].groupby('demographic_group').size()
    single_token = tok_df[tok_df['n_subtokens'] == 1].groupby('demographic_group').size()
    
    print("\nMulti-token name counts by group:")
    for group in multi_token.index:
        total = len(tok_df[tok_df['demographic_group'] == group])
        pct = (multi_token[group] / total) * 100 if group in multi_token else 0
        print(f"  {group:20s}: {multi_token.get(group, 0):2d}/{total:2d} ({pct:.1f}%)")
    
    results['multi_token_counts'] = multi_token.to_dict()
    
    # 4. Find most fragmented names
    print("\n4. MOST FRAGMENTED NAMES (>2 subtokens)")
    print("-" * 70)
    
    fragmented = tok_df[tok_df['n_subtokens'] > 2].sort_values('n_subtokens', ascending=False)
    if len(fragmented) > 0:
        print(fragmented[['name', 'demographic_group', 'n_subtokens', 'tokens']].head(10).to_string(index=False))
        results['most_fragmented'] = fragmented[['name', 'demographic_group', 'n_subtokens', 'tokens']].head(10).to_dict('records')
    else:
        print("  None found (all names ≤2 subtokens)")
    
    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        import json
        with open(output_path, 'w') as f:
            # Convert to serializable format
            serializable = {}
            for k, v in results.items():
                if isinstance(v, (pd.DataFrame, pd.Series)):
                    serializable[k] = str(v)
                elif isinstance(v, dict):
                    # Convert dict with non-string keys
                    serializable[k] = {str(key): val for key, val in v.items()}
                else:
                    serializable[k] = v
            json.dump(serializable, f, indent=2)
        
        print(f"\n✓ Saved frequency probe to {output_path.name}")
    
    print("\n" + "=" * 70)
    print("FREQUENCY PROBE COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    from src.config.paths import DATA_DIR
    
    parser = argparse.ArgumentParser(description="Run frequency probe")
    parser.add_argument(
        '--deltas',
        type=Path,
        default=DATA_DIR / "processed" / "phase3_score_embed_deltas.csv",
        help="Path to counterfactual deltas CSV"
    )
    parser.add_argument(
        '--tokenization',
        type=Path,
        default=DATA_DIR / "processed" / "phase3_tokenization_probe.csv",
        help="Path to tokenization probe CSV"
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=DATA_DIR / "processed" / "phase3_frequency_probe.json",
        help="Output path"
    )
    
    args = parser.parse_args()
    
    # Run probe
    results = run_frequency_probe(
        deltas_path=args.deltas,
        tokenization_path=args.tokenization,
        output_path=args.out
    )
    
    print(f"\n✅ Complete! Results saved to {args.out}")
