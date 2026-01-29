"""Shared constants for the experiments"""

SEED = 42

# Demographic groups we care about in the resume experiments
# (following Bertrand & Mullainathan 2004)
DEMOGRAPHIC_GROUPS = [
    "white_male",
    "white_female",
    "black_male",
    "black_female",
]

# First-name pools used to generate counterfactual resumes
# Pulled from distinctive names in labor market bias studies
NAME_POOLS = {
    "white_male": ["Brad", "Brendan", "Geoffrey", "Greg", "Brett", "Jay", "Matthew", "Neil"],
    "white_female": ["Allison", "Anne", "Carrie", "Emily", "Jill", "Laurie", "Kristen", "Meredith"],
    "black_male": ["Darnell", "Hakim", "Jermaine", "Kareem", "Jamal", "Leroy", "Rasheed", "Tremayne"],
    "black_female": ["Aisha", "Ebony", "Keisha", "Kenya", "Latonya", "Lakisha", "Latoya", "Tamika"],
}

# Which top-k values to track when looking at ranking metrics
TOPK_LIST = [1, 3, 5, 10]

# train/val/test split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Settings for bootstrap confidence intervals
BOOTSTRAP_N_SAMPLES = 2000
BOOTSTRAP_CONFIDENCE = 0.95

# Default text embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQUENCE_LENGTH = 512

# Rough batch sizes - can be tweeked locally
BATCH_SIZE_SMALL = 32
BATCH_SIZE_MEDIUM = 64
BATCH_SIZE_LARGE = 128
