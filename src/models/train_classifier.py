import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm

# add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    EMBED_MODELS, MODEL_TAGS, CLASSIFIER_CONFIG,
    get_phase4_paths, DATA_PROCESSED, EXPECTED_EMBED_DIMS
)
from src.models.features import build_features_batch, get_feature_dim
from src.models.classifier import build_classifier, ClassifierWrapper
from src.embeddings import SentenceEncoder

# early stopping class to stop training when validation stops improving
class EarlyStopping:
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    # check if we should stop training
    def __call__(self, val_score: float) -> bool:
        # first time, just save the score
        if self.best_score is None:
            self.best_score = val_score
            return False
        
        # if score improved, reset counter
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            # no improvement, increment counter
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False

# load the data from phase 2
def load_data(data_path: Path) -> pd.DataFrame:
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # rename columns to match what we expect
    df = df.rename(columns={'resume_variant_id': 'resume_id', 'occupation_match': 'match'})
    
    # check that all required columns exist
    required = ['job_id', 'resume_id', 'job_text', 'resume_text', 'match', 'split']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # print summary stats
    print(f"Loaded {len(df):,} pairs")
    print(f"  Train: {sum(df['split'] == 'train'):,}")
    print(f"  Val:   {sum(df['split'] == 'val'):,}")
    print(f"  Test:  {sum(df['split'] == 'test'):,}")
    print(f"  Match rate: {df['match'].mean():.1%}")
    
    return df

# get embeddings from cache or compute them
def get_or_compute_embeddings(texts: list, model_name: str, cache_dir: Path, batch_size: int = 32) -> np.ndarray:
    print(f"Computing embeddings with {model_name}...")
    
    # create encoder
    encoder = SentenceEncoder(model_name=model_name, cache_dir=cache_dir, normalize=True)
    
    # encode the texts
    embeddings = encoder.encode(texts, batch_size=batch_size, show_progress=True)
    
    return embeddings

# prepare features for all the data splits
def prepare_features(df: pd.DataFrame, model_name: str, feature_type: str, cache_dir: Path) -> dict:
    # get all unique texts
    unique_jobs = df['job_text'].unique()
    unique_resumes = df['resume_text'].unique()
    
    # compute embeddings for unique jobs
    print(f"\nEmbedding {len(unique_jobs):,} unique jobs...")
    job_embeddings = get_or_compute_embeddings(unique_jobs.tolist(), model_name, cache_dir)
    
    # compute embeddings for unique resumes
    print(f"Embedding {len(unique_resumes):,} unique resumes...")
    resume_embeddings = get_or_compute_embeddings(unique_resumes.tolist(), model_name, cache_dir)
    
    # make dictionaries for easy lookup
    job_emb_map = {text: emb for text, emb in zip(unique_jobs, job_embeddings)}
    resume_emb_map = {text: emb for text, emb in zip(unique_resumes, resume_embeddings)}
    
    # build features for each split (train, val, test)
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        split_df = df[df['split'] == split_name].copy()
        
        # skip if no data
        if len(split_df) == 0:
            print(f"Warning: No data for split '{split_name}'")
            continue
        
        print(f"\nBuilding features for {split_name} ({len(split_df):,} pairs)...")
        
        # get embeddings for this split
        job_embs = np.array([job_emb_map[text] for text in split_df['job_text']])
        resume_embs = np.array([resume_emb_map[text] for text in split_df['resume_text']])
        
        # build feature vectors
        features = build_features_batch(job_embs, resume_embs, feature_type)
        labels = split_df['match'].values.astype(np.float32)
        
        print(f"  Features: {features.shape}")
        print(f"  Labels:   {labels.shape}, match rate: {labels.mean():.1%}")
        
        # save to dictionary
        splits[split_name] = {
            'features': features,
            'labels': labels,
            'indices': split_df.index.values
        }
    
    return splits

# create a pytorch data loader
def create_data_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    # convert to pytorch tensors
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.FloatTensor(labels)
    
    # create dataset and loader
    dataset = TensorDataset(features_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader

# evaluate model on validation or test set
def evaluate(model: ClassifierWrapper, loader: DataLoader, criterion: nn.Module, device: str) -> dict:
    # set to eval mode
    model.model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    # no gradients needed for evaluation
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            
            # forward pass
            logits = model.model(features).squeeze()
            loss = criterion(logits, labels)
            
            # get predictions
            probs = torch.sigmoid(logits)
            
            # accumulate results
            total_loss += loss.item() * len(labels)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # compute metrics
    avg_loss = total_loss / len(all_labels)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds >= 0.5)
    f1 = f1_score(all_labels, all_preds >= 0.5)
    
    # return dictionary with all metrics
    results = {
        'loss': avg_loss,
        'auc': auc,
        'accuracy': acc,
        'f1': f1
    }
    return results

# train for one epoch
def train_epoch(model: ClassifierWrapper, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str) -> dict:
    # set to train mode
    model.model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # go through all batches
    for features, labels in tqdm(loader, desc="Training", leave=False):
        features = features.to(device)
        labels = labels.to(device)
        
        # zero the gradients
        optimizer.zero_grad()
        
        # forward pass
        logits = model.model(features).squeeze()
        loss = criterion(logits, labels)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
        # track metrics
        total_loss += loss.item() * len(labels)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # calculate metrics
    avg_loss = total_loss / len(all_labels)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds >= 0.5)
    
    # return metrics
    metrics = {
        'loss': avg_loss,
        'auc': auc,
        'accuracy': acc
    }
    return metrics

# main training function
def train(model_tag: str, model_name: str, feature_type: str = "rich", architecture: str = "mlp", device: str = None):
    # figure out device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # print header
    print(f"\n{'='*80}")
    print(f"PHASE 4: Training Classifier Head")
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
    
    # prepare features for all splits
    cache_dir = DATA_PROCESSED.parent / "cache" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    splits = prepare_features(df, model_name, feature_type, cache_dir)
    
    # build the classifier model
    embed_dim = EXPECTED_EMBED_DIMS[model_name]
    feature_dim = get_feature_dim(embed_dim, feature_type)
    
    # choose architecture
    if architecture == "logistic":
        hidden_dims = None
    else:
        hidden_dims = CLASSIFIER_CONFIG["hidden_dims"]
    
    # create the classifier
    classifier = build_classifier(feature_dim, hidden_dims=hidden_dims, dropout=CLASSIFIER_CONFIG["dropout"])
    
    # wrap it
    wrapper = ClassifierWrapper(classifier, device=device)
    
    # print model info
    print(f"\nModel architecture:")
    print(classifier)
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # create data loaders for train and val
    train_loader = create_data_loader(splits['train']['features'], splits['train']['labels'], CLASSIFIER_CONFIG['batch_size'], shuffle=True)
    
    val_loader = create_data_loader(splits['val']['features'], splits['val']['labels'], CLASSIFIER_CONFIG['batch_size'], shuffle=False)
    
    # setup training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=CLASSIFIER_CONFIG['learning_rate'], weight_decay=CLASSIFIER_CONFIG['weight_decay'])
    
    # learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=CLASSIFIER_CONFIG['lr_factor'], patience=CLASSIFIER_CONFIG['lr_patience'])
    
    # early stopping
    early_stopping = EarlyStopping(patience=CLASSIFIER_CONFIG['patience'], min_delta=CLASSIFIER_CONFIG['min_delta'])
    
    # start training
    print(f"\n{'='*80}")
    print("Training...")
    print(f"{'='*80}\n")
    
    # history to track metrics
    history = {
        'train_loss': [],
        'train_auc': [],
        'val_loss': [],
        'val_auc': [],
        'val_accuracy': [],
        'val_f1': [],
        'learning_rates': []
    }
    
    # track best model
    best_val_auc = 0.0
    best_epoch = 0
    
    # training loop
    for epoch in range(CLASSIFIER_CONFIG['max_epochs']):
        print(f"\nEpoch {epoch+1}/{CLASSIFIER_CONFIG['max_epochs']}")
        print("-" * 80)
        
        # train for one epoch
        train_metrics = train_epoch(wrapper, train_loader, criterion, optimizer, device)
        
        # evaluate on validation set
        val_metrics = evaluate(wrapper, val_loader, criterion, device)
        
        # update learning rate
        scheduler.step(val_metrics['auc'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # save metrics to history
        history['train_loss'].append(train_metrics['loss'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['learning_rates'].append(current_lr)
        
        # print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch + 1
            wrapper.save(str(paths['checkpoint']))
            print(f"Saved best model (AUC: {best_val_auc:.4f})")
        
        # check early stopping
        if early_stopping(val_metrics['auc']):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # training finished
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Best validation AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"{'='*80}\n")
    
    # save training history
    with open(paths['training_curves'], 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Saved training curves to {paths['training_curves']}")
    
    # save training metrics
    num_params = sum(p.numel() for p in classifier.parameters())
    train_final = {
        'epochs_trained': len(history['train_loss']),
        'best_epoch': best_epoch,
        'final_train_loss': history['train_loss'][-1],
        'final_train_auc': history['train_auc'][-1],
        'model_name': model_name,
        'model_tag': model_tag,
        'feature_type': feature_type,
        'architecture': architecture,
        'feature_dim': feature_dim,
        'num_params': num_params,
        'config': CLASSIFIER_CONFIG
    }
    
    with open(paths['train_metrics'], 'w') as f:
        json.dump(train_final, f, indent=2)
    
    # save validation metrics
    val_final = {
        'best_val_auc': best_val_auc,
        'best_val_loss': history['val_loss'][best_epoch-1],
        'best_val_accuracy': history['val_accuracy'][best_epoch-1],
        'best_val_f1': history['val_f1'][best_epoch-1],
    }
    
    with open(paths['val_metrics'], 'w') as f:
        json.dump(val_final, f, indent=2)
    
    print(f"Saved metrics to {paths['model_dir']}")
    print(f"\nCheckpoint saved to {paths['checkpoint']}")

# main function to parse arguments and start training
def main():
    # setup argument parser
    parser = argparse.ArgumentParser(description="Train classifier head on frozen embeddings")
    parser.add_argument('--model', type=str, required=True, choices=['minilm', 'e5small'], help="Model to use")
    parser.add_argument('--feature-type', type=str, default='rich', choices=['cosine', 'concat', 'rich'], help="Type of features to build")
    parser.add_argument('--architecture', type=str, default='mlp', choices=['logistic', 'mlp'], help="Classifier architecture")
    parser.add_argument('--device', type=str, default=None, help="Device to train on (default: auto-detect)")
    
    # parse arguments
    args = parser.parse_args()
    
    # map model tag to full name
    model_map = {
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
        'e5small': 'intfloat/e5-small-v2'
    }
    
    model_name = model_map[args.model]
    
    # start training
    train(model_tag=args.model, model_name=model_name, feature_type=args.feature_type, architecture=args.architecture, device=args.device)

# run main if this file is executed
if __name__ == "__main__":
    main()
