from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# calculate cohens d effect size
def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1 = len(group1)
    n2 = len(group2)
    
    # need at least 2 samples in each group
    if n1 < 2 or n2 < 2:
        return np.nan
    
    # calculate pooled standard deviation
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return np.nan
    
    # return the effect size
    mean_diff = np.mean(group1) - np.mean(group2)
    return mean_diff / pooled_std

# function to compute metrics for retrieval
def compute_retrieval_metrics(df: pd.DataFrame, score_col: str, target_col: str = 'occupation_match', group_by: str = None) -> Dict[str, float]:
    # get rid of any NaN values
    valid = df[[score_col, target_col]].notna().all(axis=1)
    df_clean = df[valid].copy()
    
    # handle empty dataframe
    if len(df_clean) == 0:
        return {
            'n_pairs': 0,
            'auc': np.nan,
            'gap': np.nan,
            'cohens_d': np.nan
        }
    
    scores = df_clean[score_col].values
    targets = df_clean[target_col].values
    
    # calculate AUC
    try:
        auc = roc_auc_score(targets, scores)
    except:
        auc = np.nan
    
    # split into match and mismatch scores
    match_scores = scores[targets == 1]
    mismatch_scores = scores[targets == 0]
    
    # calculate gap and cohens d
    if len(match_scores) > 0 and len(mismatch_scores) > 0:
        gap = float(np.mean(match_scores) - np.mean(mismatch_scores))
        d = cohens_d(match_scores, mismatch_scores)
    else:
        gap = np.nan
        d = np.nan
    
    # return all the metrics
    results_dict = {
        'n_pairs': len(df_clean),
        'n_match': int(np.sum(targets == 1)),
        'n_mismatch': int(np.sum(targets == 0)),
        'auc': float(auc),
        'gap': float(gap),
        'cohens_d': float(d),
        'mean_score_match': float(np.mean(match_scores)) if len(match_scores) > 0 else np.nan,
        'mean_score_mismatch': float(np.mean(mismatch_scores)) if len(mismatch_scores) > 0 else np.nan,
    }
    return results_dict

# main function to evaluate retrieval performance
def evaluate_retrieval(pairs_path: Path, output_path: Path = None, score_cols: List[str] = ['score_tfidf', 'score_embed']) -> pd.DataFrame:
    print("=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)
    
    # load the scored pairs
    print(f"\nLoading scored pairs from {pairs_path.name}...")
    df = pd.read_csv(pairs_path)
    print(f"  Total pairs: {len(df):,}")
    
    results = []
    
    # compute overall metrics first
    print("\nComputing overall metrics...")
    for score_col in score_cols:
        metrics = compute_retrieval_metrics(df, score_col)
        metrics['scope'] = 'overall'
        metrics['occupation'] = 'all'
        metrics['score_type'] = score_col
        results.append(metrics)
        
        # print the results
        print(f"\n{score_col}:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Gap (match - mismatch): {metrics['gap']:.6f}")
        print(f"  Cohen's d: {metrics['cohens_d']:.4f}")
    
    # now compute per-occupation metrics
    print("\nComputing per-occupation metrics...")
    occupations = df['job_occupation'].unique()
    print(f"  Occupations: {len(occupations)}")
    
    # loop through each score type and occupation
    for score_col in score_cols:
        for occ in occupations:
            df_occ = df[df['job_occupation'] == occ]
            if len(df_occ) < 10:
                continue
                
            metrics = compute_retrieval_metrics(df_occ, score_col)
            metrics['scope'] = 'by_occupation'
            metrics['occupation'] = occ
            metrics['score_type'] = score_col
            results.append(metrics)
    
    # convert results list to dataframe
    df_metrics = pd.DataFrame(results)
    
    # save to file if user wants
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(output_path, index=False)
        print(f"\nSaved metrics to {output_path.name}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    return df_metrics

# test the code if run directly
if __name__ == "__main__":
    from ..config.paths import DATA_DIR
    
    # run evaluation
    pairs_path = DATA_DIR / "processed" / "pairs_scored_phase2.csv"
    output_path = DATA_DIR / "processed" / "phase2_retrieval_metrics.csv"
    
    df_metrics = evaluate_retrieval(pairs_path, output_path)
    
    # show top and bottom occupations
    print("\nTop 5 occupations by embedding AUC:")
    embed_metrics = df_metrics[(df_metrics['scope'] == 'by_occupation') & (df_metrics['score_type'] == 'score_embed')].sort_values('auc', ascending=False)
    
    print(embed_metrics[['occupation', 'auc', 'gap', 'n_pairs']].head())
    
    print("\nBottom 5 occupations by embedding AUC:")
    print(embed_metrics[['occupation', 'auc', 'gap', 'n_pairs']].tail())
