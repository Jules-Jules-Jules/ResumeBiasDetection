from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..models import LinearHead, MLPHead
from ..embeddings import SentenceEncoder
from ..config.paths import DATA_DIR

# load embeddings for a specific data split (train, val, or test)
def load_embeddings_for_split(df: pd.DataFrame, encoder: SentenceEncoder, split: str) -> Tuple[np.ndarray, np.ndarray]:
    # filter to the split we want
    df_split = df[df['split'] == split].copy()
    
    print(f"\nPreparing {split} split ({len(df_split):,} pairs)")
    
    # get unique jobs and resumes
    unique_jobs = df_split[['job_id', 'job_text']].drop_duplicates('job_id')
    unique_resumes = df_split[['resume_variant_id', 'resume_text']].drop_duplicates('resume_variant_id')
    
    # encode jobs
    print(f"  Encoding {len(unique_jobs):,} unique jobs")
    job_embeds = encoder.encode(unique_jobs['job_text'].fillna("").tolist(), show_progress=True)
    
    # encode resumes
    print(f"  Encoding {len(unique_resumes):,} unique resumes")
    resume_embeds = encoder.encode(unique_resumes['resume_text'].fillna("").tolist(), show_progress=True)
    
    # create lookup dictionaries
    job_id_to_idx = {jid: i for i, jid in enumerate(unique_jobs['job_id'])}
    resume_id_to_idx = {rid: i for i, rid in enumerate(unique_resumes['resume_variant_id'])}
    
    # build feature matrix for all pairs
    n_pairs = len(df_split)
    embed_dim = job_embeds.shape[1]
    
    # concatenate job and resume embeddings
    X = np.zeros((n_pairs, embed_dim * 2), dtype=np.float32)
    
    print(f"  Building feature matrix")
    for i, (_, row) in enumerate(df_split.iterrows()):
        job_idx = job_id_to_idx[row['job_id']]
        resume_idx = resume_id_to_idx[row['resume_variant_id']]
        
        # put job embedding in first half
        X[i, :embed_dim] = job_embeds[job_idx]
        # put resume embedding in second half
        X[i, embed_dim:] = resume_embeds[resume_idx]
    
    # get labels
    y = df_split['occupation_match'].values.astype(np.float32)
    
    return X, y

# train a linear classifier (logistic regression)
def train_linear_head(pairs_path: Path = DATA_DIR / "processed" / "job_resume_pairs_phase1.csv", model_path: Path = DATA_DIR / "models" / "linear_head.joblib", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", C_values: list = [0.01, 0.1, 1.0, 10.0], use_cached_embeddings: bool = True) -> LinearHead:
    print("=" * 60)
    print("TRAINING LINEAR CLASSIFIER HEAD")
    print("=" * 60)
    
    # load the pairs data
    print(f"\nLoading pairs from {pairs_path.name}")
    df = pd.read_csv(pairs_path)
    print(f"  Total pairs: {len(df):,}")
    print(f"  Train: {(df['split'] == 'train').sum():,}")
    print(f"  Val: {(df['split'] == 'val').sum():,}")
    print(f"  Test: {(df['split'] == 'test').sum():,}")
    
    # setup encoder
    cache_dir = DATA_DIR / "processed" / "embedding_cache" if use_cached_embeddings else None
    encoder = SentenceEncoder(model_name=embedding_model, cache_dir=cache_dir)
    
    # prepare train and validation data
    X_train, y_train = load_embeddings_for_split(df, encoder, 'train')
    X_val, y_val = load_embeddings_for_split(df, encoder, 'val')
    
    print(f"\nFeature shape: {X_train.shape}")
    print(f"Train: {len(y_train):,} (pos: {y_train.sum():.0f}, {100*y_train.mean():.1f}%)")
    print(f"Val: {len(y_val):,} (pos: {y_val.sum():.0f}, {100*y_val.mean():.1f}%)")
    
    # try different C values to find best one
    print(f"\nTuning regularization strength (C)")
    print(f"  Trying C values: {C_values}")
    
    best_val_acc = 0
    best_C = None
    best_model = None
    
    # loop through each C value
    for C in C_values:
        print(f"\n  C={C}:")
        model = LinearHead(C=C)
        model.fit(X_train, y_train, X_val, y_val)
        
        # check validation accuracy
        val_acc = model.model.score(model.scaler.transform(X_val), y_val)
        
        # save if it's the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_C = C
            best_model = model
            print(f"    -> New best!")
    
    print(f"\nBest C: {best_C} (val_acc={best_val_acc:.4f})")
    
    # save the best model
    best_model.save(model_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return best_model

# train an MLP classifier (neural network)
def train_mlp_head(pairs_path: Path = DATA_DIR / "processed" / "job_resume_pairs_phase1.csv", model_path: Path = DATA_DIR / "models" / "mlp_head.pt", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", hidden_dim: int = 128, dropout: float = 0.2, lr: float = 0.001, epochs: int = 20, batch_size: int = 256, use_cached_embeddings: bool = True) -> MLPHead:
    print("=" * 60)
    print("TRAINING MLP CLASSIFIER HEAD")
    print("=" * 60)
    
    # figure out device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # load pairs
    print(f"\nLoading pairs from {pairs_path.name}")
    df = pd.read_csv(pairs_path)
    
    # setup encoder
    cache_dir = DATA_DIR / "processed" / "embedding_cache" if use_cached_embeddings else None
    encoder = SentenceEncoder(model_name=embedding_model, cache_dir=cache_dir)
    
    # prepare data
    X_train, y_train = load_embeddings_for_split(df, encoder, 'train')
    X_val, y_val = load_embeddings_for_split(df, encoder, 'val')
    
    input_dim = X_train.shape[1]
    print(f"\nInput dimension: {input_dim}")
    
    # create datasets
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    
    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # create model
    model = MLPHead(input_dim, hidden_dim, dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # training loop
    print(f"\nTraining for {epochs} epochs")
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # zero gradients
            optimizer.zero_grad()
            
            # forward pass
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            
            # backward pass
            loss.backward()
            optimizer.step()
            
            # accumulate loss
            train_loss += loss.item() * len(X_batch)
        
        # average train loss
        train_loss /= len(train_dataset)
        
        # validation phase
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # get predictions
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(X_batch)
                
                # count correct predictions
                correct += ((preds > 0.5) == y_batch).sum().item()
        
        # average validation metrics
        val_loss /= len(val_dataset)
        val_acc = correct / len(val_dataset)
        
        # print metrics
        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # check if we should stop early
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # save best model
            checkpoint = {
                'state_dict': model.state_dict(),
                'config': model.config
            }
            torch.save(checkpoint, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # load the best model
    model = MLPHead.load(model_path)
    print(f"\nSaved best model to {model_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return model

# main code to train both models
if __name__ == "__main__":
    print("Training classifier heads on frozen embeddings\n")
    
    # train linear classifier
    linear_model = train_linear_head()
    
    print("\n" + "=" * 60 + "\n")
    
    # train MLP classifier
    mlp_model = train_mlp_head()
