from pathlib import Path

# setting up the main folders for the project
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"

# make sure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# list of the embedding models
EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-small-v2",
]

# model tags
MODEL_TAGS = {
    "sentence-transformers/all-MiniLM-L6-v2": "minilm",
    "intfloat/e5-small-v2": "e5small",
}

# expected embedding dimensions for each model
EXPECTED_EMBED_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "intfloat/e5-small-v2": 384,
}

# config settings for training the classifier
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

# where to save phase 4 results
PHASE4_DIR = DATA_DIR / "processed" / "classifier"
PHASE4_DIR.mkdir(parents=True, exist_ok=True)

# function to get all the file paths for a model
def get_phase4_paths(model_tag: str) -> dict:
    # create folder for this model
    model_dir = PHASE4_DIR / model_tag
    model_dir.mkdir(exist_ok=True)
    
    # return dictionary with all the paths
    paths_dict = {
        "model_dir": model_dir,
        "checkpoint": MODEL_DIR / f"classifier_head_{model_tag}.pt",
        "train_metrics": model_dir / "train_metrics.json",
        "val_metrics": model_dir / "val_metrics.json",
        "test_metrics": model_dir / "test_metrics.json",
        "predictions": model_dir / "predictions_test.csv",
        "counterfactual_flips": model_dir / "counterfactual_flips.csv",
        "selection_gaps": model_dir / "selection_gaps.csv",
        "training_curves": model_dir / "training_curves.json",
    }
    return paths_dict

# threshold for deciding if something is accept or reject
DECISION_THRESHOLD = 0.5

# how many bootstrap samples to use
N_BOOTSTRAP = 1000
