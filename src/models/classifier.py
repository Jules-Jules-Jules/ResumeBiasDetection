import torch
import torch.nn as nn
from typing import List, Optional

# simple logistic regression classifier
class LogisticClassifier(nn.Module):
    
    def __init__(self, input_dim: int):
        super().__init__()
        # just one linear layer for logistic regression
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # run the input through the linear layer
        return self.linear(x)

# multi-layer perceptron classifier
class MLPClassifier(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()
        
        # build the layers list
        layers = []
        prev_dim = input_dim
        
        # add hidden layers with relu and dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # add output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        # combine all layers into sequential
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pass input through the network
        return self.network(x)

# function to build either logistic or MLP classifier
def build_classifier(input_dim: int, hidden_dims: Optional[List[int]] = None, dropout: float = 0.3) -> nn.Module:
    # if no hidden dims, use logistic regression
    if hidden_dims is None or len(hidden_dims) == 0:
        return LogisticClassifier(input_dim)
    else:
        # otherwise use MLP
        return MLPClassifier(input_dim, hidden_dims, dropout)

# wrapper class to make using the classifier easier
class ClassifierWrapper:
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        # save model and device
        self.model = model.to(device)
        self.device = device
        
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        # get probability predictions
        self.model.eval()
        with torch.no_grad():
            features = features.to(self.device)
            logits = self.model(features)
            probs = torch.sigmoid(logits).squeeze()
        return probs
    
    def predict(self, features: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        # get binary predictions using threshold
        probs = self.predict_proba(features)
        predictions = (probs >= threshold).long()
        return predictions
    
    def save(self, path: str):
        # save the model to a file
        checkpoint_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
        }
        torch.save(checkpoint_dict, path)
    
    def load(self, path: str):
        # load model from file
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

# test the classifiers if running this file directly
if __name__ == "__main__":
    print("Testing classifiers:")
    
    # setup test parameters
    input_dim = 1536
    batch_size = 32
    
    # test logistic regression
    log_model = build_classifier(input_dim, hidden_dims=None)
    num_params_log = sum(p.numel() for p in log_model.parameters())
    print(f"\nLogistic: {num_params_log} params")
    
    # test MLP
    mlp_model = build_classifier(input_dim, hidden_dims=[256, 128], dropout=0.3)
    num_params_mlp = sum(p.numel() for p in mlp_model.parameters())
    print(f"MLP:      {num_params_mlp} params")
    
    # test forward pass
    x = torch.randn(batch_size, input_dim)
    
    log_out = log_model(x)
    mlp_out = mlp_model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {log_out.shape} (logistic)")
    print(f"Output shape: {mlp_out.shape} (MLP)")
    
    # test wrapper
    wrapper = ClassifierWrapper(log_model)
    probs = wrapper.predict_proba(x)
    preds = wrapper.predict(x)
    
    print(f"\nProbabilities: {probs.shape}, range=[{probs.min():.3f}, {probs.max():.3f}]")
    print(f"Predictions: {preds.shape}, values={preds.unique().tolist()}")
