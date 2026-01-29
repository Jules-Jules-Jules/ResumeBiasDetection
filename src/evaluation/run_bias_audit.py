import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.bias_metrics import compute_bias_summary
from src.evaluation.bootstrap import compute_bootstrap_summary
from src.config.paths import DATA_DIR
from src.utils import get_logger

logger = get_logger(__name__)


def run_bias_audit(
    pairs_path: Path,
    out_dir: Path,
    score_cols: list = ['score_tfidf', 'score_embed'],
    topk: list = [1, 3, 5, 8],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42
):
    logger.info("=" * 70)
    logger.info("PHASE 3: BIAS AUDIT")
    logger.info("=" * 70)
    
    # Load data
    logger.info(f"\nLoading scored pairs from {pairs_path.name}...")
    df = pd.read_csv(pairs_path)
    logger.info(f"  Loaded {len(df):,} pairs")
    logger.info(f"  Jobs: {df['job_id'].nunique()}")
    logger.info(f"  Base resumes: {df['base_resume_id'].nunique()}")
    logger.info(f"  Demographic groups: {df['demographic_group'].unique().tolist()}")
    
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute bias metrics
    logger.info(f"\nComputing bias metrics for {len(score_cols)} score types...")
    results = compute_bias_summary(df, score_cols, topk)
    
    # Save all metrics
    logger.info(f"\nSaving bias metrics to {out_dir}...")
    for key, df_result in results.items():
        out_path = out_dir / f"phase3_{key}.csv"
        df_result.to_csv(out_path, index=False)
        logger.info(f"  ✓ Saved: {out_path.name} ({len(df_result)} rows)")
    
    # Compute bootstrap CIs for primary score column
    primary_score = score_cols[0] if score_cols else 'score_embed'
    logger.info(f"\nComputing bootstrap confidence intervals for {primary_score}...")
    
    selection_rates_df = results[f'{primary_score}_selection_rates']
    deltas_df = results[f'{primary_score}_deltas']
    
    bootstrap_results = compute_bootstrap_summary(
        selection_rates_df,
        deltas_df,
        comparisons=[
            ('white_male', 'black_female'),
            ('white_male', 'black_male'),
            ('white_female', 'black_female'),
            ('white_female', 'black_male')
        ],
        k_values=topk,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed
    )
    
    # Save bootstrap results
    bootstrap_path = out_dir / "phase3_bias_bootstrap.json"
    with open(bootstrap_path, 'w') as f:
        json.dump(bootstrap_results, f, indent=2)
    logger.info(f"\n✓ Saved bootstrap results to {bootstrap_path.name}")
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("BIAS AUDIT SUMMARY")
    logger.info("=" * 70)
    
    # Selection rate gaps
    logger.info("\n1. SELECTION RATE GAPS (with 95% CI):")
    for gap in bootstrap_results['selection_gaps']:
        if 'error' not in gap:
            sig_marker = "*" if gap['significant'] else " "
            logger.info(f"   k={gap['k']}, {gap['group1']} vs {gap['group2']}:")
            logger.info(f"     Gap: {gap['gap']:.4f} [{gap['ci_lower']:.4f}, {gap['ci_upper']:.4f}] {sig_marker}")
    
    # Counterfactual deltas
    logger.info("\n2. COUNTERFACTUAL SCORE DELTAS (with 95% CI):")
    for delta in bootstrap_results['counterfactual_deltas']:
        if 'error' not in delta and 'delta_type' in delta:
            delta_type = delta['delta_type']
            logger.info(f"   {delta_type}:")
            logger.info(f"     Mean: {delta['mean']:.6f} [{delta['ci_lower']:.6f}, {delta['ci_upper']:.6f}]")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ BIAS AUDIT COMPLETE")
    logger.info("=" * 70)
    
    logger.info(f"\nGenerated files in {out_dir}:")
    logger.info(f"  - phase3_*_selection_rates.csv (top-k selection rates per job)")
    logger.info(f"  - phase3_*_selection_agg.csv (aggregated selection rates)")
    logger.info(f"  - phase3_*_deltas.csv (counterfactual score deltas)")
    logger.info(f"  - phase3_*_rank_flips.csv (rank flip metrics)")
    logger.info(f"  - phase3_bias_bootstrap.json (bootstrap CIs)")
    
    logger.info(f"\nNext: Review results in notebooks/03_phase3_bias_audit.ipynb")


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 3 bias audit pipeline"
    )
    parser.add_argument(
        '--pairs',
        type=Path,
        default=DATA_DIR / "processed" / "pairs_scored_phase2.csv",
        help="Path to scored pairs CSV"
    )
    parser.add_argument(
        '--out_dir',
        type=Path,
        default=DATA_DIR / "processed",
        help="Output directory for results"
    )
    parser.add_argument(
        '--score_col',
        type=str,
        default='score_embed',
        help="Score column to analyze (score_tfidf or score_embed)"
    )
    parser.add_argument(
        '--topk',
        type=int,
        nargs='+',
        default=[1, 3, 5, 8],
        help="Top-k values for selection rates"
    )
    parser.add_argument(
        '--n_bootstrap',
        type=int,
        default=2000,
        help="Number of bootstrap samples"
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.95,
        help="Confidence level for CIs"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Run audit
    run_bias_audit(
        pairs_path=args.pairs,
        out_dir=args.out_dir,
        score_cols=[args.score_col],
        topk=args.topk,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
