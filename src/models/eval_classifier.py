import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from tqdm import tqdm

# add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    EMBED_MODELS, MODEL_TAGS, CLASSIFIER_CONFIG, DECISION_THRESHOLD,
    get_phase4_paths, DATA_PROCESSED, EXPECTED_EMBED_DIMS
)
from src.models.features import build_features_batch, get_feature_dim
from src.models.classifier import build_classifier, ClassifierWrapper
from src.models.train_classifier import (
    load_data, get_or_compute_embeddings, create_data_loader, evaluate
)

# load a trained classifier from a checkpoint file
def load_trained_model(checkpoint_path: Path, feature_dim: int, architecture: str, device: str) -> ClassifierWrapper:
    print(f"Loading model from {checkpoint_path}...")
    
    # figure out if we're using logistic or mlp
    if architecture == "logistic":
        hidden_dims = None
    else:
        hidden_dims = CLASSIFIER_CONFIG["hidden_dims"]
    
    # build the classifier
    classifier = build_classifier(feature_dim, hidden_dims=hidden_dims, dropout=CLASSIFIER_CONFIG["dropout"])
    
    # wrap it and load the weights
    wrapper = ClassifierWrapper(classifier, device=device)
    wrapper.load(str(checkpoint_path))
    
    print(f"Model loaded successfully")
    return wrapper

# compute all the test metrics
def compute_test_metrics(model: ClassifierWrapper, test_loader, device: str) -> dict:
    print("\nComputing test metrics...")
    
    # set model to eval mode
    model.model.eval()
    all_probs = []
    all_labels = []
    
    # get predictions for all test samples
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Testing"):
            features = features.to(device)
            probs = model.predict_proba(features)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # convert to numpy arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= DECISION_THRESHOLD).astype(int)
    
    # compute all the metrics
    auc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # get confusion matrix values
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    # put everything in a dictionary
    metrics = {
        'auc': float(auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        },
        'num_samples': len(all_labels),
        'match_rate': float(all_labels.mean())
    }
    
    # print the results
    print(f"\nTest Metrics:")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    
    return metrics, all_probs, all_preds

# analyze decision flips across demographic variants (counterfactuals)
def analyze_counterfactual_flips(df: pd.DataFrame, model: ClassifierWrapper, job_emb_map: dict, resume_emb_map: dict, feature_type: str, device: str) -> pd.DataFrame:
    print("\nAnalyzing counterfactual decision flips...")
    
    # filter to just test set
    test_df = df[df['split'] == 'test'].copy()
    
    # check if we have the demographic columns
    if 'base_resume_id' not in test_df.columns or 'demographic_group' not in test_df.columns:
        print("Warning: Missing base_resume_id or demographic_group columns")
        print("Skipping counterfactual flip analysis")
        return pd.DataFrame()
    
    # group by job and base resume to get the 4 demographic variants
    grouped = test_df.groupby(['job_id', 'base_resume_id'])
    
    flip_results = []
    
    # go through each quartet
    for (job_id, base_resume_id), group in tqdm(grouped, desc="Computing flips"):
        # skip if we don't have all 4 variants
        if len(group) != 4:
            continue
        
        # get the job embedding (same for all variants)
        job_text = group.iloc[0]['job_text']
        job_emb = job_emb_map[job_text]
        
        # get predictions for each demographic variant
        variant_results = []
        
        for _, row in group.iterrows():
            resume_text = row['resume_text']
            resume_emb = resume_emb_map[resume_text]
            
            # build the feature vector
            features = build_features_batch(job_emb.reshape(1, -1), resume_emb.reshape(1, -1), feature_type)
            
            # get model prediction
            features_tensor = torch.FloatTensor(features).to(device)
            prob = model.predict_proba(features_tensor).item()
            pred = int(prob >= DECISION_THRESHOLD)
            
            # save the result for this variant
            variant_results.append({
                'demographic_group': row['demographic_group'],
                'prob': prob,
                'pred': pred
            })
        
        # analyze the 4 predictions
        probs = np.array([v['prob'] for v in variant_results])
        preds = np.array([v['pred'] for v in variant_results])
        groups = [v['demographic_group'] for v in variant_results]
        
        # compute some statistics
        prob_range = probs.max() - probs.min()
        prob_mean = probs.mean()
        prob_std = probs.std()
        
        # figure out which group got highest and lowest probability
        max_idx = probs.argmax()
        min_idx = probs.argmin()
        max_group = groups[max_idx]
        min_group = groups[min_idx]
        
        # check if there was a decision flip
        flip_occurred = len(set(preds)) > 1
        num_accept = preds.sum()
        
        # make dictionaries for each group
        group_probs = {g: p for g, p in zip(groups, probs)}
        group_preds = {g: p for g, p in zip(groups, preds)}
        
        # save all the info for this quartet
        flip_results.append({
            'job_id': job_id,
            'base_resume_id': base_resume_id,
            'job_occupation': group.iloc[0].get('job_occupation', 'unknown'),
            'prob_range': prob_range,
            'prob_mean': prob_mean,
            'prob_std': prob_std,
            'max_prob': probs.max(),
            'min_prob': probs.min(),
            'max_group': max_group,
            'min_group': min_group,
            'flip_occurred': flip_occurred,
            'num_accept': int(num_accept),
            'white_male_prob': group_probs.get('white_male', None),
            'white_female_prob': group_probs.get('white_female', None),
            'black_male_prob': group_probs.get('black_male', None),
            'black_female_prob': group_probs.get('black_female', None),
            'white_male_pred': group_preds.get('white_male', None),
            'white_female_pred': group_preds.get('white_female', None),
            'black_male_pred': group_preds.get('black_male', None),
            'black_female_pred': group_preds.get('black_female', None),
        })
    
    # convert to dataframe
    df_flips = pd.DataFrame(flip_results)
    
    # print summary stats
    print(f"\nCounterfactual Flip Analysis:")
    print(f"  Total quartets analyzed: {len(df_flips):,}")
    print(f"  Decision flips occurred: {df_flips['flip_occurred'].sum():,} ({df_flips['flip_occurred'].mean():.1%})")
    print(f"  Mean prob range:         {df_flips['prob_range'].mean():.4f}")
    print(f"  Max prob range:          {df_flips['prob_range'].max():.4f}")
    print(f"\nMost favored group (max prob):")
    print(df_flips['max_group'].value_counts())
    print(f"\nLeast favored group (min prob):")
    print(df_flips['min_group'].value_counts())
    
    return df_flips

# compute selection rate gaps by demographic group
def compute_selection_rate_gaps(df: pd.DataFrame, model: ClassifierWrapper, job_emb_map: dict, resume_emb_map: dict, feature_type: str, device: str) -> pd.DataFrame:
    print("\nComputing selection rate gaps...")
    
    # filter to test set
    test_df = df[df['split'] == 'test'].copy()
    
    # check if we have demographic info
    if 'demographic_group' not in test_df.columns:
        print("Warning: Missing demographic_group column")
        return pd.DataFrame()
    
    # get predictions for all test pairs
    print("Predicting for all test pairs...")
    
    predictions = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        job_emb = job_emb_map[row['job_text']]
        resume_emb = resume_emb_map[row['resume_text']]
        
        # build features
        features = build_features_batch(job_emb.reshape(1, -1), resume_emb.reshape(1, -1), feature_type)
        
        # get prediction
        features_tensor = torch.FloatTensor(features).to(device)
        prob = model.predict_proba(features_tensor).item()
        
        predictions.append(prob)
    
    # add predictions to dataframe
    test_df['pred_prob'] = predictions
    
    # compute selection rates for each job
    selection_results = []
    
    for job_id in test_df['job_id'].unique():
        job_data = test_df[test_df['job_id'] == job_id].copy()
        
        # rank candidates by predicted probability
        job_data = job_data.sort_values('pred_prob', ascending=False)
        job_data['rank'] = range(1, len(job_data) + 1)
        
        # compute selection rates at different top k values
        for k in [1, 3, 5, 8]:
            top_k = job_data[job_data['rank'] <= k]
            
            if len(top_k) == 0:
                continue
            
            # compute for each demographic group
            for group in ['white_male', 'white_female', 'black_male', 'black_female']:
                group_data = job_data[job_data['demographic_group'] == group]
                
                if len(group_data) == 0:
                    continue
                
                # how many from this group are in top k
                selected = sum(group_data['rank'] <= k)
                rate = selected / len(group_data)
                
                selection_results.append({
                    'job_id': job_id,
                    'job_occupation': job_data.iloc[0].get('job_occupation', 'unknown'),
                    'k': k,
                    'demographic_group': group,
                    'selection_rate': rate,
                    'num_selected': selected,
                    'num_total': len(group_data)
                })
    
    # convert to dataframe
    df_selection = pd.DataFrame(selection_results)
    
    # print the gaps
    print(f"\nSelection Rate Gaps (k=1):")
    k1_data = df_selection[df_selection['k'] == 1]
    agg = k1_data.groupby('demographic_group')['selection_rate'].mean()
    print(agg)
    
    gap = agg.max() - agg.min()
    print(f"\nMax gap: {gap:.4f} ({gap*100:.1f}pp)")
    
    return df_selection

# main function to evaluate a trained model
def evaluate_model(model_tag: str, model_name: str, feature_type: str = "rich", architecture: str = "mlp", device: str = None):
    # figure out device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # print header
    print(f"\n{'='*80}")
    print(f"PHASE 4: Evaluating Classifier Head")
    print(f"{'='*80}")
    print(f"Model:        {model_name}")
    print(f"Model tag:    {model_tag}")
    print(f"Features:     {feature_type}")
    print(f"Architecture: {architecture}")
    print(f"Device:       {device}")
    print(f"{'='*80}\n")
    
    # get file paths
    paths = get_phase4_paths(model_tag)
    
    # load the data
    data_path = DATA_PROCESSED / "pairs_scored_phase2.csv"
    df = load_data(data_path)
    
    # get embeddings for all unique texts
    cache_dir = DATA_PROCESSED.parent / "cache" / "embeddings"
    
    unique_jobs = df['job_text'].unique()
    unique_resumes = df['resume_text'].unique()
    
    print(f"\nLoading embeddings...")
    job_embeddings = get_or_compute_embeddings(unique_jobs.tolist(), model_name, cache_dir)
    resume_embeddings = get_or_compute_embeddings(unique_resumes.tolist(), model_name, cache_dir)
    
    # make dictionaries for easy lookup
    job_emb_map = {text: emb for text, emb in zip(unique_jobs, job_embeddings)}
    resume_emb_map = {text: emb for text, emb in zip(unique_resumes, resume_embeddings)}
    
    # load the trained model
    embed_dim = EXPECTED_EMBED_DIMS[model_name]
    feature_dim = get_feature_dim(embed_dim, feature_type)
    
    model = load_trained_model(paths['checkpoint'], feature_dim, architecture, device)
    
    # prepare test data
    test_df = df[df['split'] == 'test']
    test_job_embs = np.array([job_emb_map[text] for text in test_df['job_text']])
    test_resume_embs = np.array([resume_emb_map[text] for text in test_df['resume_text']])
    test_features = build_features_batch(test_job_embs, test_resume_embs, feature_type)
    test_labels = test_df['match'].values.astype(np.float32)
    
    # create test data loader
    test_loader = create_data_loader(test_features, test_labels, CLASSIFIER_CONFIG['batch_size'], shuffle=False)
    
    # compute test metrics
    test_metrics, all_probs, all_preds = compute_test_metrics(model, test_loader, device)
    
    # save test metrics to json
    with open(paths['test_metrics'], 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\nSaved test metrics to {paths['test_metrics']}")
    
    # save predictions to csv
    predictions_df = test_df.copy()
    predictions_df['pred_prob'] = all_probs
    predictions_df['pred_label'] = all_preds
    predictions_df.to_csv(paths['predictions'], index=False)
    
    print(f"Saved predictions to {paths['predictions']}")
    
    # analyze counterfactual flips
    df_flips = analyze_counterfactual_flips(df, model, job_emb_map, resume_emb_map, feature_type, device)
    
    if len(df_flips) > 0:
        df_flips.to_csv(paths['counterfactual_flips'], index=False)
        print(f"Saved counterfactual flips to {paths['counterfactual_flips']}")
    
    # compute selection rate gaps
    df_selection = compute_selection_rate_gaps(df, model, job_emb_map, resume_emb_map, feature_type, device)
    
    if len(df_selection) > 0:
        df_selection.to_csv(paths['selection_gaps'], index=False)
        print(f"Saved selection gaps to {paths['selection_gaps']}")
    
    # print footer
    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}\n")

# main function to parse arguments and run evaluation
def main():
    # setup argument parser
    parser = argparse.ArgumentParser(description="Evaluate classifier head")
    parser.add_argument('--model', type=str, required=True, choices=['minilm', 'e5small'], help="Model to evaluate")
    parser.add_argument('--feature-type', type=str, default='rich', choices=['cosine', 'concat', 'rich'], help="Type of features used")
    parser.add_argument('--architecture', type=str, default='mlp', choices=['logistic', 'mlp'], help="Classifier architecture")
    parser.add_argument('--device', type=str, default=None, help="Device to use (default: auto-detect)")
    
    # parse arguments
    args = parser.parse_args()
    
    # map model tag to full name
    model_map = {
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
        'e5small': 'intfloat/e5-small-v2'
    }
    
    model_name = model_map[args.model]
    
    # run evaluation
    evaluate_model(model_tag=args.model, model_name=model_name, feature_type=args.feature_type, architecture=args.architecture, device=args.device)

# run main if this file is executed
if __name__ == "__main__":
    main()
