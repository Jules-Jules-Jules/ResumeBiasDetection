import pandas as pd
import random
from pathlib import Path
from typing import Optional
from collections import Counter

from src.config import DATA_PROCESSED, SEED
from src.utils import get_logger, read_jsonl, write_csv, set_seed
from src.data.normalize_occupations import is_occupation_match

logger = get_logger(__name__)


def sample_jobs_per_occupation(
    job_records: list[dict],
    n_jobs_per_occupation: int = 20,
) -> list[dict]:
    df = pd.DataFrame(job_records)
    
    # Sample within each occupation
    sampled_list = []
    for occupation, group in df.groupby("occupation"):
        n_sample = min(n_jobs_per_occupation, len(group))
        sampled_group = group.sample(n=n_sample, random_state=SEED)
        sampled_list.append(sampled_group)
    
    sampled = pd.concat(sampled_list, ignore_index=True)
    sampled_records = sampled.to_dict('records')
    
    logger.info(f"Sampled {len(sampled_records):,} jobs from {df['occupation'].nunique()} occupations")
    
    return sampled_records


def build_pairs_for_job(
    job_record: dict,
    resume_variants: list[dict],
    n_match_base_resumes: int = 4,
    n_mismatch_base_resumes: int = 4,
) -> list[dict]:
    job_occupation = job_record["occupation"]
    job_id = job_record["job_id"]
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(resume_variants)
    
    # Get unique base resume IDs for matching and mismatching occupations
    matching_df = df[df.apply(lambda row: is_occupation_match(row["occupation"], job_occupation), axis=1)]
    mismatching_df = df[~df.apply(lambda row: is_occupation_match(row["occupation"], job_occupation), axis=1)]
    
    matching_base_ids = matching_df["base_resume_id"].unique()
    mismatching_base_ids = mismatching_df["base_resume_id"].unique()
    
    # Sample base resume IDs
    if len(matching_base_ids) < n_match_base_resumes:
        logger.warning(f"Job {job_id} ({job_occupation}): Only {len(matching_base_ids)} matching resumes available (requested {n_match_base_resumes})")
        selected_match_ids = matching_base_ids
    else:
        selected_match_ids = random.sample(list(matching_base_ids), n_match_base_resumes)
    
    if len(mismatching_base_ids) < n_mismatch_base_resumes:
        logger.warning(f"Job {job_id} ({job_occupation}): Only {len(mismatching_base_ids)} mismatching resumes available (requested {n_mismatch_base_resumes})")
        selected_mismatch_ids = mismatching_base_ids
    else:
        selected_mismatch_ids = random.sample(list(mismatching_base_ids), n_mismatch_base_resumes)
    
    # Get all 4 variants for each selected base resume
    selected_base_ids = list(selected_match_ids) + list(selected_mismatch_ids)
    selected_variants = df[df["base_resume_id"].isin(selected_base_ids)].to_dict('records')
    
    # Create pairs
    pairs = []
    for resume_variant in selected_variants:
        occupation_match = is_occupation_match(resume_variant["occupation"], job_occupation)
        
        pair = {
            "pair_id": f"{job_id}_{resume_variant['variant_id']}",
            "job_id": job_id,
            "job_text": job_record["text"],
            "job_occupation": job_occupation,
            "base_resume_id": resume_variant["base_resume_id"],
            "resume_variant_id": resume_variant["variant_id"],
            "resume_text": resume_variant["text"],
            "resume_occupation": resume_variant["occupation"],
            "demographic_group": resume_variant["demographic_group"],
            "occupation_match": 1 if occupation_match else 0,
            "split": resume_variant["split"],
        }
        
        pairs.append(pair)
    
    return pairs


def build_all_pairs(
    job_records: list[dict],
    resume_variants: list[dict],
    n_match_base_resumes: int = 4,
    n_mismatch_base_resumes: int = 4,
) -> list[dict]:
    logger.info(f"Building pairs for {len(job_records):,} jobs")
    logger.info(f" Match base resumes per job: {n_match_base_resumes}")
    logger.info(f" Mismatch base resumes per job: {n_mismatch_base_resumes}")
    logger.info(f" Expected pairs per job: {(n_match_base_resumes + n_mismatch_base_resumes) * 4}")
    
    all_pairs = []
    
    for job_record in job_records:
        pairs = build_pairs_for_job(
            job_record,
            resume_variants,
            n_match_base_resumes,
            n_mismatch_base_resumes,
        )
        all_pairs.extend(pairs)
    
    logger.info(f"Created {len(all_pairs):,} total pairs")
    
    return all_pairs


def verify_pair_balance(pairs: list[dict]):
    df = pd.DataFrame(pairs)
    logger.info("\n" + "=" * 70)
    logger.info("PAIR BALANCE VERIFICATION")
    logger.info("=" * 70)
    # Occupation match balance
    match_counts = df["occupation_match"].value_counts()
    logger.info("\nOccupation match balance:")
    logger.info(f"  Match (1):    {match_counts.get(1, 0):,} ({100*match_counts.get(1, 0)/len(df):.1f}%)")
    logger.info(f"  Mismatch (0): {match_counts.get(0, 0):,} ({100*match_counts.get(0, 0)/len(df):.1f}%)")
    # Demographic group balance
    group_counts = df["demographic_group"].value_counts()
    logger.info("\nDemographic group balance:")
    for group in sorted(group_counts.index):
        count = group_counts[group]
        logger.info(f"  {group:20s}: {count:,} ({100*count/len(df):.1f}%)")
    
    # Split distribution
    split_counts = df["split"].value_counts()
    logger.info("\nSplit distribution:")
    for split in ["train", "val", "test"]:
        count = split_counts.get(split, 0)
        logger.info(f"  {split:10s}: {count:,} ({100*count/len(df):.1f}%)")
    # Candidate set size per job
    pairs_per_job = df.groupby("job_id").size()
    logger.info("\nCandidate set size per job:")
    logger.info(f"  Min: {pairs_per_job.min()}")
    logger.info(f"  Median: {pairs_per_job.median():.0f}")
    logger.info(f"  Max: {pairs_per_job.max()}")
    logger.info(f"  Std: {pairs_per_job.std():.2f}")
    
    # Counterfactual quartets check
    pairs_per_job_base = df.groupby(["job_id", "base_resume_id"]).size()
    logger.info("\nCounterfactual quartet completeness:")
    logger.info("  Expected: 4 variants per (job_id, base_resume_id)")
    logger.info(f"  Actual range: {pairs_per_job_base.min()} to {pairs_per_job_base.max()}")
    if pairs_per_job_base.nunique() == 1 and pairs_per_job_base.iloc[0] == 4:
        logger.info(" All quartets complete (4 variants each)")
    else:
        logger.warning(" Some quartets incomplete")

    logger.info("=" * 70 + "\n")


def save_pairs(
    pairs: list[dict],
    output_path: Optional[Path] = None,
):
    if output_path is None:
        output_path = DATA_PROCESSED / "job_resume_pairs_phase1.csv"
    
    df = pd.DataFrame(pairs)
    
    logger.info(f"Saving {len(pairs):,} pairs to: {output_path}")
    write_csv(output_path, df, index=False)
    logger.info("Saved successfully")


if __name__ == "__main__":
    set_seed(SEED)
    
    # Load data
    logger.info("Loading data")
    jobs = read_jsonl(DATA_PROCESSED / "jobs_clean.jsonl")
    resume_variants = read_jsonl(DATA_PROCESSED / "resumes_augmented.jsonl")
    
    logger.info(f"  Jobs: {len(jobs):,}")
    logger.info(f"  Resume variants: {len(resume_variants):,}")
    
    # Sample jobs (use subset for manageable size)
    # For full dataset, increase n_jobs_per_occupation
    sampled_jobs = sample_jobs_per_occupation(jobs, n_jobs_per_occupation=20)
    
    # Build pairs
    # Each job gets 4 matching + 4 mismatching base resumes
    # Each base resume has 4 demographic variants
    # So each job has (4+4)*4 = 32 candidates
    pairs = build_all_pairs(
        sampled_jobs,
        resume_variants,
        n_match_base_resumes=4,
        n_mismatch_base_resumes=4,
    )
    
    # Verify balance
    verify_pair_balance(pairs)
    save_pairs(pairs)
    
    print("\n Pair building complete")
