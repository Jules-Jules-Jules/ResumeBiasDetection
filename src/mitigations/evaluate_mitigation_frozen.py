import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.embeddings.encoder import SentenceEncoder


def compute_counterfactual_flips(predictions_df: pd.DataFrame) -> pd.DataFrame:
    grouped = predictions_df.groupby(['job_id', 'base_resume_id'])
    
    flip_data = []
    
    for (job_id, base_resume_id), group in grouped:
        if len(group) != 4:
            continue
        
        probs = {}
        preds = {}
        for _, row in group.iterrows():
            demo = row['demographic_group']
            probs[demo] = row['pred_prob']
            preds[demo] = row['pred_label']
        
        required_groups = ['white_male', 'white_female', 'black_male', 'black_female']
        if not all(g in probs for g in required_groups):
            continue
        
        prob_values = list(probs.values())
        pred_values = list(preds.values())
        
        max_prob = max(prob_values)
        min_prob = min(prob_values)
        prob_range = max_prob - min_prob
        
        max_group = max(probs.items(), key=lambda x: x[1])[0]
        min_group = min(probs.items(), key=lambda x: x[1])[0]
        
        flip_occurred = len(set(pred_values)) > 1
        
        flip_data.append({
            'job_id': job_id,
            'job_occupation': group.iloc[0]['job_occupation'],
            'base_resume_id': base_resume_id,
            'white_male_prob': probs['white_male'],
            'white_female_prob': probs['white_female'],
            'black_male_prob': probs['black_male'],
            'black_female_prob': probs['black_female'],
            'white_male_pred': preds['white_male'],
            'white_female_pred': preds['white_female'],
            'black_male_pred': preds['black_male'],
            'black_female_pred': preds['black_female'],
            'max_prob': max_prob,
            'min_prob': min_prob,
            'prob_range': prob_range,
            'max_group': max_group,
            'min_group': min_group,
            'flip_occurred': flip_occurred,
        })
    
    return pd.DataFrame(flip_data)


def evaluate_mitigation_frozen(mitigation_file: Path, output_dir: Path):
    print("="*70)
    print("EVALUATING FROZEN MINILM (COSINE SIMILARITY) ON MITIGATION")
    print("="*70)
    
    mitigation_type = 'M1' if 'masked' in mitigation_file.name else 'M2'
    text_col = 'resume_text_masked' if mitigation_type == 'M1' else 'resume_text_freqnorm'
    
    print(f"\nMitigation: {mitigation_type}")
    print(f"Text column: {text_col}")
    
    print(f"\nLoading data from: {mitigation_file}")
    df = pd.read_csv(mitigation_file)
    print(f"Loaded {len(df)} pairs")
    
    print("\nLoading frozen MiniLM encoder")
    encoder = SentenceEncoder('all-MiniLM-L6-v2')
    
    print(f"\nGenerating embeddings")
    job_texts = df['job_text'].tolist()
    resume_texts = df[text_col].tolist()
    
    job_embeddings = encoder.encode(job_texts, show_progress=True)
    resume_embeddings = encoder.encode(resume_texts, show_progress=True)
    
    print(f"Job embeddings shape: {job_embeddings.shape}")
    print(f"Resume embeddings shape: {resume_embeddings.shape}")
    
    print(f"\nComputing cosine similarities")
    similarities = []
    for i in range(len(df)):
        sim = cosine_similarity(
            job_embeddings[i:i+1], 
            resume_embeddings[i:i+1]
        )[0, 0]
        similarities.append(sim)
    
    threshold = np.median(similarities)
    print(f"Using median as threshold: {threshold:.4f}")
    
    predictions = []
    for i in range(len(df)):
        sim = similarities[i]
        pred = 1 if sim >= threshold else 0
        
        predictions.append({
            'pair_id': df.iloc[i]['pair_id'],
            'job_id': df.iloc[i]['job_id'],
            'job_occupation': df.iloc[i]['job_occupation'],
            'base_resume_id': df.iloc[i]['base_resume_id'],
            'resume_id': df.iloc[i]['resume_id'],
            'demographic_group': df.iloc[i]['demographic_group'],
            'match': df.iloc[i]['match'],
            'pred_prob': sim,
            'pred_label': pred,
        })
    
    predictions_df = pd.DataFrame(predictions)
    
    print(f"\nComputing metrics")
    y_true = df['match'].values
    y_pred = predictions_df['pred_label'].values
    y_prob = predictions_df['pred_prob'].values
    
    metrics = {
        'auc': float(roc_auc_score(y_true, y_prob)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nComputing counterfactual flips")
    flips_df = compute_counterfactual_flips(predictions_df)
    
    flip_rate = flips_df['flip_occurred'].mean() * 100
    mean_prob_range = flips_df['prob_range'].mean()
    
    print(f"\nCounterfactual Analysis:")
    print(f"  Total quartets: {len(flips_df)}")
    print(f"  Flips occurred: {flips_df['flip_occurred'].sum()}")
    print(f"  Flip rate: {flip_rate:.1f}%")
    print(f"  Mean prob range: {mean_prob_range:.4f}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
    flips_df.to_csv(output_dir / 'counterfactual_flips.csv', index=False)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    summary = {
        'model': 'minilm_frozen',
        'mitigation': mitigation_type,
        'metrics': metrics,
        'flip_rate': float(flip_rate),
        'mean_prob_range': float(mean_prob_range),
        'num_quartets': len(flips_df),
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print("="*70)
    
    return summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate mitigation on frozen MiniLM')
    parser.add_argument('--mitigation', type=str, required=True, choices=['m1', 'm2'],
                        help='Mitigation to evaluate (m1=masking, m2=frequency)')
    
    args = parser.parse_args()
    
    # Determine mitigation file
    if args.mitigation == 'm1':
        mitigation_file = Path('data/processed/mitigations/m1_masked_test.csv')
    else:
        mitigation_file = Path('data/processed/mitigations/m2_freqnorm_test.csv')
    
    if not mitigation_file.exists():
        print(f"Error: Mitigation file not found: {mitigation_file}")
        print("Run apply_m1_masking.py or apply_m2_frequency_normalization.py first")
        sys.exit(1)
    
    # Output directory
    output_dir = Path(f'data/processed/mitigations/minilm_frozen_{args.mitigation}')
    
    # Run evaluation
    evaluate_mitigation_frozen(mitigation_file, output_dir)
