from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from collections import defaultdict


def compute_topk_selection_rates(
    df: pd.DataFrame,
    score_col: str,
    k_values: List[int] = [1, 3, 5, 8],
    group_col: str = 'demographic_group'
) -> pd.DataFrame:
    results = []
    
    for job_id, job_df in df.groupby('job_id'):
        job_df = job_df.sort_values(score_col, ascending=False).reset_index(drop=True)
        
        n_candidates = len(job_df)
        
        for k in k_values:
            if k > n_candidates:
                continue
                
            top_k = job_df.head(k)
            group_counts = top_k[group_col].value_counts()
            
            for group in df[group_col].unique():
                count = group_counts.get(group, 0)
                rate = count / k
                
                results.append({
                    'job_id': job_id,
                    'k': k,
                    'demographic_group': group,
                    'n_selected': count,
                    'selection_rate': rate,
                    'score_col': score_col
                })
    
    return pd.DataFrame(results)


def compute_within_base_deltas(
    df: pd.DataFrame,
    score_col: str,
    reference_group: str = 'white_male'
) -> pd.DataFrame:
    results = []
    
    for (job_id, base_id), group in df.groupby(['job_id', 'base_resume_id']):
        if len(group) != 4:
            continue
        
        group_dict = dict(zip(group['demographic_group'], group[score_col]))
        
        if len(group_dict) != 4:
            continue
        
        ref_score = group_dict.get(reference_group, None)
        if ref_score is None:
            continue
        
        deltas = {}
        for demo_group, score in group_dict.items():
            if demo_group != reference_group:
                deltas[f'{reference_group}_minus_{demo_group}'] = ref_score - score
        
        scores = list(group_dict.values())
        score_range = max(scores) - min(scores)
        
        results.append({
            'job_id': job_id,
            'base_resume_id': base_id,
            'score_col': score_col,
            'range': score_range,
            **deltas,
            'occupation_match': group['occupation_match'].iloc[0],
            'job_occupation': group['job_occupation'].iloc[0],
            'resume_occupation': group['resume_occupation'].iloc[0]
        })
    
    return pd.DataFrame(results)


def compute_rank_flips(
    df: pd.DataFrame,
    score_col: str
) -> pd.DataFrame:
    results = []
    
    for job_id, job_df in df.groupby('job_id'):
        job_df = job_df.sort_values(score_col, ascending=False).reset_index(drop=True)
        job_df['rank'] = range(1, len(job_df) + 1)
        
        top_candidate = job_df.iloc[0]
        top_base_id = top_candidate['base_resume_id']
        top_group = top_candidate['demographic_group']
        
        variants = job_df[job_df['base_resume_id'] == top_base_id]
        
        if len(variants) < 4:
            continue
        
        rank_range = variants['rank'].max() - variants['rank'].min()
        rank_positions = variants['rank'].values
        n_would_be_top = sum(rank_positions == 1)
        
        results.append({
            'job_id': job_id,
            'score_col': score_col,
            'top_base_id': top_base_id,
            'top_group': top_group,
            'rank_range_of_top_base': rank_range,
            'n_variants_would_be_top': n_would_be_top,
            'top_position_stable': n_would_be_top == 4
        })
    
    return pd.DataFrame(results)


def aggregate_selection_rates(
    selection_df: pd.DataFrame,
    by_occupation: bool = False
) -> pd.DataFrame:
    if by_occupation:
        group_cols = ['k', 'demographic_group', 'job_occupation']
    else:
        group_cols = ['k', 'demographic_group']
    
    if 'job_occupation' not in selection_df.columns and by_occupation:
        print("Warning: job_occupation not in dataframe, cannot aggregate by occupation")
        by_occupation = False
        group_cols = ['k', 'demographic_group']
    
    agg_df = selection_df.groupby(group_cols).agg({
        'selection_rate': ['mean', 'std', 'count'],
        'n_selected': 'sum'
    }).reset_index()
    
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    
    return agg_df


def compute_bias_summary(
    df: pd.DataFrame,
    score_cols: List[str] = ['score_tfidf', 'score_embed'],
    k_values: List[int] = [1, 3, 5, 8]
) -> Dict[str, pd.DataFrame]:
    results = {}
    
    for score_col in score_cols:
        print(f"\nComputing bias metrics for {score_col}")
        
        print(f"  Computing top-k selection rates (k={k_values})")
        selection_rates = compute_topk_selection_rates(df, score_col, k_values)
        results[f'{score_col}_selection_rates'] = selection_rates
        
        agg_selection = aggregate_selection_rates(selection_rates)
        results[f'{score_col}_selection_agg'] = agg_selection
        
        print(f"  Computing within-base counterfactual deltas")
        deltas = compute_within_base_deltas(df, score_col)
        results[f'{score_col}_deltas'] = deltas
        
        print(f"  Computing rank flip metrics")
        rank_flips = compute_rank_flips(df, score_col)
        results[f'{score_col}_rank_flips'] = rank_flips
    
    return results


if __name__ == "__main__":
    from ..config.paths import DATA_DIR
    
    pairs_path = DATA_DIR / "processed" / "pairs_scored_phase2.csv"
    print(f"Loading pairs from {pairs_path}")
    df = pd.read_csv(pairs_path)
    
    print(f"Loaded {len(df)} pairs")
    
    results = compute_bias_summary(df)
    
    out_dir = DATA_DIR / "processed"
    
    for key, df_result in results.items():
        out_path = out_dir / f"phase3_{key}.csv"
        df_result.to_csv(out_path, index=False)
        print(f"Saved: {out_path.name}")
    
    print("\nBias metrics computation complete")
