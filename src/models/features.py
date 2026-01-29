import numpy as np
from typing import Tuple, Dict

# compute cosine similarity between two vectors
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # make sure they are 2d arrays
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    
    # normalize the vectors
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    
    # compute dot product
    sim = np.sum(a_norm * b_norm, axis=1)
    
    # return scalar if only one pair
    if len(sim) == 1:
        return sim[0]
    else:
        return sim

# build feature vector from job and resume embeddings
def build_pair_features(job_emb: np.ndarray, resume_emb: np.ndarray, feature_type: str = "rich") -> np.ndarray:
    # make sure they are 2d
    if job_emb.ndim == 1:
        job_emb = job_emb.reshape(1, -1)
    if resume_emb.ndim == 1:
        resume_emb = resume_emb.reshape(1, -1)
    
    # cosine similarity only
    if feature_type == "cosine":
        sim = cosine_similarity(job_emb, resume_emb)
        return sim.reshape(-1, 1)
    
    # just concatenate the two embeddings
    elif feature_type == "concat":
        return np.concatenate([job_emb, resume_emb], axis=1)
    
    # rich features with concat, difference, and product
    elif feature_type == "rich":
        concat = np.concatenate([job_emb, resume_emb], axis=1)
        abs_diff = np.abs(job_emb - resume_emb)
        product = job_emb * resume_emb
        
        # put them all together
        all_features = np.concatenate([concat, abs_diff, product], axis=1)
        return all_features
    
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

# calculate the feature dimension based on embedding dimension and feature type
def get_feature_dim(embed_dim: int, feature_type: str) -> int:
    # cosine similarity is just 1 number
    if feature_type == "cosine":
        return 1
    # concat is 2x the embedding dimension
    elif feature_type == "concat":
        return 2 * embed_dim
    # rich is 4x (concat + diff + product)
    elif feature_type == "rich":
        return 4 * embed_dim
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

# build features for a batch of job-resume pairs
def build_features_batch(job_embs: np.ndarray, resume_embs: np.ndarray, feature_type: str = "rich") -> np.ndarray:
    # just use the same function for batch
    return build_pair_features(job_embs, resume_embs, feature_type)

# get metadata about the feature configuration
def get_feature_metadata(embed_dim: int, feature_type: str) -> Dict:
    # calculate the feature dimension
    feature_dim = get_feature_dim(embed_dim, feature_type)
    
    # create metadata dictionary
    metadata = {
        "embed_dim": embed_dim,
        "feature_type": feature_type,
        "feature_dim": feature_dim,
        "components": {
            "cosine": feature_type == "cosine",
            "concat": feature_type in ["concat", "rich"],
            "abs_diff": feature_type == "rich",
            "product": feature_type == "rich",
        }
    }
    
    return metadata

# test the feature functions
if __name__ == "__main__":
    # create some dummy embeddings
    job_emb = np.random.randn(5, 384)
    resume_emb = np.random.randn(5, 384)
    
    print("Testing feature builders:")
    print(f"Input shape: job={job_emb.shape}, resume={resume_emb.shape}")
    
    # test each feature type
    for ftype in ["cosine", "concat", "rich"]:
        features = build_pair_features(job_emb, resume_emb, ftype)
        expected_dim = get_feature_dim(384, ftype)
        print(f"\n{ftype:10s}: {features.shape}")
        print(f"  Expected: {expected_dim}")
        
        # get the metadata
        metadata = get_feature_metadata(384, ftype)
        print(f"  Metadata: {metadata}")
