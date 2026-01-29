import sys
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.embeddings.encoder import SentenceEncoder
from src.models.classifier import ClassifierWrapper, build_classifier
from src.models.features import build_features_batch, get_feature_dim
from src.config import CLASSIFIER_CONFIG, EXPECTED_EMBED_DIMS


def load_model(model_name: str, architecture: str = 'mlp', device: str = 'cpu'):
    encoder_map = {
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
        'e5small': 'intfloat/e5-small-v2'
    }
    
    if model_name not in encoder_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    encoder_full_name = encoder_map[model_name]
    print(f"Loading encoder: {encoder_full_name}")
    encoder = SentenceEncoder(encoder_full_name)
    
    embedding_dim = EXPECTED_EMBED_DIMS[encoder_full_name]
    print(f"Embedding dimension: {embedding_dim}")
    
    feature_type = 'rich'
    feature_dim = get_feature_dim(embedding_dim, feature_type)
    print(f"Feature dimension: {feature_dim} (type: {feature_type})")
    
    if architecture == 'linear':
        hidden_dims = None
    else:
        hidden_dims = CLASSIFIER_CONFIG["hidden_dims"]
    
    classifier = build_classifier(
        feature_dim,
        hidden_dims=hidden_dims,
        dropout=CLASSIFIER_CONFIG["dropout"]
    )
    
    model_path = Path(f'models/classifier_head_{model_name}.pt')
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading classifier from: {model_path}")
    wrapper = ClassifierWrapper(classifier, device=device)
    wrapper.load(str(model_path))
    wrapper.model.eval()
    
    return encoder, wrapper


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


def evaluate_mitigation_on_classifier(model_name: str, mitigation_file: Path, output_dir: Path, architecture: str = 'mlp'):
    print("="*70)
    print(f"EVALUATING {model_name.upper()} ON MITIGATION")
    print("="*70)
    
    mitigation_type = 'M1' if 'masked' in mitigation_file.name else 'M2'
    text_col = 'resume_text_masked' if mitigation_type == 'M1' else 'resume_text_freqnorm'
    
    print(f"\nMitigation: {mitigation_type}")
    print(f"Text column: {text_col}")
    
    print(f"\nLoading data from: {mitigation_file}")
    df = pd.read_csv(mitigation_file)
    print(f"Loaded {len(df)} pairs")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    encoder, classifier_wrapper = load_model(model_name, architecture, device)
    
    print(f"\nGenerating embeddings")
    job_texts = df['job_text'].tolist()
    resume_texts = df[text_col].tolist()
    
    job_embeddings = encoder.encode(job_texts, show_progress=True)
    resume_embeddings = encoder.encode(resume_texts, show_progress=True)
    
    print(f"\nComputing predictions")
    predictions = []
    
    classifier_wrapper.model.eval()
    with torch.no_grad():
        for i in range(len(df)):
            # Build features (using 'rich' features like Phase 4 training)
            features = build_features_batch(
                job_embeddings[i:i+1],
                resume_embeddings[i:i+1],
                'rich'  # Same as training: job resume job*resume |job-resume|
            )
            
            # Get prediction
            features_tensor = torch.FloatTensor(features).to(device)
            prob_output = classifier_wrapper.predict_proba(features_tensor)
            # Handle both scalar and array outputs
            prob = prob_output.item() if hasattr(prob_output, 'item') else float(prob_output)
            pred = 1 if prob >= 0.5 else 0
            
            predictions.append({
                'pair_id': df.iloc[i]['pair_id'],
                'job_id': df.iloc[i]['job_id'],
                'job_occupation': df.iloc[i]['job_occupation'],
                'base_resume_id': df.iloc[i]['base_resume_id'],
                'resume_id': df.iloc[i]['resume_id'],
                'demographic_group': df.iloc[i]['demographic_group'],
                'match': df.iloc[i]['match'],
                'pred_prob': prob,
                'pred_label': pred,
            })
    
    predictions_df = pd.DataFrame(predictions)
    
    # Compute metrics
    print(f"\nComputing metrics...")
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
    
    # Compute counterfactual flips
    print(f"\nComputing counterfactual flips...")
    flips_df = compute_counterfactual_flips(predictions_df)
    
    flip_rate = flips_df['flip_occurred'].mean() * 100
    mean_prob_range = flips_df['prob_range'].mean()
    
    print(f"\nCounterfactual Analysis:")
    print(f"  Total quartets: {len(flips_df)}")
    print(f"  Flips occurred: {flips_df['flip_occurred'].sum()}")
    print(f"  Flip rate: {flip_rate:.1f}%")
    print(f"  Mean prob range: {mean_prob_range:.4f}")
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
    flips_df.to_csv(output_dir / 'counterfactual_flips.csv', index=False)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    summary = {
        'model': model_name,
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
    
    parser = argparse.ArgumentParser(description='Evaluate mitigation on classifier model')
    parser.add_argument('--model', type=str, required=True, choices=['minilm', 'e5small'],
                        help='Model to evaluate')
    parser.add_argument('--mitigation', type=str, required=True, choices=['m1', 'm2'],
                        help='Mitigation to evaluate (m1=masking, m2=frequency)')
    parser.add_argument('--architecture', type=str, default='mlp', choices=['linear', 'mlp'],
                        help='Classifier architecture (default: mlp)')
    
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
    output_dir = Path(f'data/processed/mitigations/{args.model}_{args.mitigation}')
    
    # Run evaluation
    evaluate_mitigation_on_classifier(args.model, mitigation_file, output_dir, args.architecture)
