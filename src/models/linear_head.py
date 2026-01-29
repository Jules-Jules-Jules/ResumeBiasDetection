import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# linear classifier using logistic regression
class LinearHead:
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        # setup logistic regression model
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state, solver='lbfgs')
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # save the config
        self.config = {
            'model_type': 'linear',
            'C': C,
            'max_iter': max_iter,
            'random_state': random_state
        }
    
    # train the classifier
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        # scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # train the model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # print training accuracy
        train_acc = self.model.score(X_scaled, y)
        print(f"  Training accuracy: {train_acc:.4f}")
        
        # print validation accuracy if we have validation data
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_acc = self.model.score(X_val_scaled, y_val)
            print(f"  Validation accuracy: {val_acc:.4f}")
    
    # predict probabilities
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # make sure model is fitted
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # scale features and predict
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return probs
    
    # predict class labels
    def predict(self, X: np.ndarray) -> np.ndarray:
        # make sure model is fitted
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # scale and predict
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    # save model to disk
    def save(self, path: Path):
        import joblib
        
        # make sure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # put everything in a dictionary
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config
        }
        
        # save it
        joblib.dump(model_data, path)
        print(f"  Saved model to {path}")
    
    # load model from disk
    @classmethod
    def load(cls, path: Path) -> 'LinearHead':
        import joblib
        
        # load the saved data
        model_data = joblib.load(path)
        
        # recreate the object
        head = cls(**model_data['config'])
        head.model = model_data['model']
        head.scaler = model_data['scaler']
        head.is_fitted = True
        
        return head

# MLP classifier (neural network with hidden layers)
class MLPHead(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        # build the network layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # save config
        self.config = {
            'model_type': 'mlp',
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'dropout': dropout
        }
    
    # forward pass through the network
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x).squeeze(-1)
        return output
    
    # save the model
    def save(self, path: Path):
        # make sure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # save state dict and config
        checkpoint_data = {
            'state_dict': self.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint_data, path)
        print(f"  Saved model to {path}")
    
    # load the model
    @classmethod
    def load(cls, path: Path) -> 'MLPHead':
        # load checkpoint
        checkpoint = torch.load(path, weights_only=False)
        
        # get config without model_type
        config = {k: v for k, v in checkpoint['config'].items() if k != 'model_type'}
        
        # create model and load weights
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        return model
