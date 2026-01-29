from pathlib import Path
from .paths import DATA_PROCESSED, MODELS_DIR

# the embedding models being tested
EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-small-v2",
]

# shorter names for each model to use in file names
MODEL_TAGS = {
    "sentence-transformers/all-MiniLM-L6-v2": "minilm",
    "intfloat/e5-small-v2": "e5small",
}

# expected dimensions for the embeddings
EXPECTED_EMBED_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "intfloat/e5-small-v2": 384,
}

# settings for training the classifier
CLASSIFIER_CONFIG = {
    "hidden_dims": [256, 128],
    "dropout": 0.3,
    "batch_size": 128,
    "max_epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "patience": 5,
    "min_delta": 0.001,
    "lr_scheduler": "plateau",
    "lr_factor": 0.5,
    "lr_patience": 3,
}

# where we save phase 4 output files
PHASE4_DIR = DATA_PROCESSED / "classifier"
PHASE4_DIR.mkdir(parents=True, exist_ok=True)

# function that returns all the file paths for a given model
def get_phase4_paths(model_tag: str) -> dict:
    # make a folder for this model
    model_dir = PHASE4_DIR / model_tag
    model_dir.mkdir(exist_ok=True)
    
    # create dictionary with all the paths
    path_dict = {
        "model_dir": model_dir,
        "checkpoint": MODELS_DIR / f"classifier_head_{model_tag}.pt",
        "train_metrics": model_dir / "train_metrics.json",
        "val_metrics": model_dir / "val_metrics.json",
        "test_metrics": model_dir / "test_metrics.json",
        "predictions": model_dir / "predictions_test.csv",
        "counterfactual_flips": model_dir / "counterfactual_flips.csv",
        "selection_gaps": model_dir / "selection_gaps.csv",
        "training_curves": model_dir / "training_curves.json",
    }
    return path_dict

# threshold for classifying as accept/reject
DECISION_THRESHOLD = 0.5

# number of bootstrap samples
N_BOOTSTRAP = 1000
