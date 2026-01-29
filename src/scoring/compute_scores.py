from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..embeddings import SentenceEncoder, TfidfEncoder
from ..config.paths import DATA_DIR

# main function to score job-resume pairs
def score_pairs(pairs_path: Path = DATA_DIR / "processed" / "job_resume_pairs_phase1.csv", output_path: Path = DATA_DIR / "processed" / "pairs_scored_phase2.csv", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", cache_embeddings: bool = True, batch_size: int = 32) -> pd.DataFrame:
    print("=" * 60)
    print("PHASE 2: SCORING PIPELINE")
    print("=" * 60)
    
    # load the pairs data
    print(f"\n1. Loading pairs from {pairs_path.name}...")
    df = pd.read_csv(pairs_path)
    print(f"   Total pairs: {len(df):,}")
    print(f"   Unique jobs: {df['job_id'].nunique():,}")
    print(f"   Unique resumes: {df['resume_variant_id'].nunique():,}")
    
    # fill in any missing text with empty string
    df['job_text'] = df['job_text'].fillna("")
    df['resume_text'] = df['resume_text'].fillna("")
    
    # get the unique texts so we only encode each one once
    unique_job_texts = df[['job_id', 'job_text']].drop_duplicates('job_id')
    unique_resume_texts = df[['resume_variant_id', 'resume_text']].drop_duplicates('resume_variant_id')
    
    print(f"   Unique job texts to encode: {len(unique_job_texts):,}")
    print(f"   Unique resume texts to encode: {len(unique_resume_texts):,}")
    
    # compute TF-IDF baseline scores
    print("\n2. Computing TF-IDF baseline scores")
    tfidf_encoder = TfidfEncoder()
    
    # combine all texts together for fitting
    all_texts = pd.concat([unique_job_texts['job_text'], unique_resume_texts['resume_text']])
    tfidf_encoder.fit(all_texts.tolist())
    
    # encode the jobs and resumes
    print("   Encoding jobs")
    job_tfidf = tfidf_encoder.encode(unique_job_texts['job_text'].tolist())
    print("   Encoding resumes")
    resume_tfidf = tfidf_encoder.encode(unique_resume_texts['resume_text'].tolist())
    
    # make lookup dictionaries to map IDs to indices
    job_id_to_idx = {jid: i for i, jid in enumerate(unique_job_texts['job_id'])}
    resume_id_to_idx = {rid: i for i, rid in enumerate(unique_resume_texts['resume_variant_id'])}
    
    # now compute scores for each pair
    print("   Computing pairwise TF-IDF scores")
    scores_tfidf = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Scoring"):
        # get the indices for this job and resume
        job_idx = job_id_to_idx[row['job_id']]
        resume_idx = resume_id_to_idx[row['resume_variant_id']]
        
        # compute cosine similarity using dot product
        score = np.dot(job_tfidf[job_idx], resume_tfidf[resume_idx])
        scores_tfidf.append(float(score))
    
    # add scores to dataframe
    df['score_tfidf'] = scores_tfidf
    print(f"   TF-IDF score range: [{min(scores_tfidf):.4f}, {max(scores_tfidf):.4f}]")
    
    # now do the neural embedding scores
    print(f"\n3. Computing embedding scores ({embedding_model})")
    
    # set up caching if needed
    cache_dir = DATA_DIR / "processed" / "embedding_cache" if cache_embeddings else None
    encoder = SentenceEncoder(model_name=embedding_model, cache_dir=cache_dir, normalize=True)
    
    # encode jobs using the embedding model
    print("   Encoding jobs")
    job_embeds = encoder.encode(unique_job_texts['job_text'].tolist(), batch_size=batch_size, show_progress=True)
    
    # encode resumes
    print("   Encoding resumes")
    resume_embeds = encoder.encode(unique_resume_texts['resume_text'].tolist(), batch_size=batch_size, show_progress=True)
    
    # compute embedding scores for each pair
    print("   Computing pairwise embedding scores")
    scores_embed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Scoring"):
        # look up the indices
        job_idx = job_id_to_idx[row['job_id']]
        resume_idx = resume_id_to_idx[row['resume_variant_id']]
        
        # compute similarity with dot product
        score = np.dot(job_embeds[job_idx], resume_embeds[resume_idx])
        scores_embed.append(float(score))
    
    # add embedding scores to dataframe
    df['score_embed'] = scores_embed
    print(f"   Embedding score range: [{min(scores_embed):.4f}, {max(scores_embed):.4f}]")
    
    # save the results
    print(f"\n4. Saving scored pairs to {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   Saved {len(df):,} scored pairs")
    
    print("\n" + "~" * 60)
    print("SCORING COMPLETE")
    print("=" * 60)
    
    return df

# run the scoring if this file is executed directly
if __name__ == "__main__":
    # run the scoring pipeline
    df_scored = score_pairs()
    
    # print some quick stats
    print("\nQuick Statistics:")
    print(f"  TF-IDF mean: {df_scored['score_tfidf'].mean():.4f}")
    print(f"  Embedding mean: {df_scored['score_embed'].mean():.4f}")
    print(f"  Correlation: {df_scored['score_tfidf'].corr(df_scored['score_embed']):.4f}")
