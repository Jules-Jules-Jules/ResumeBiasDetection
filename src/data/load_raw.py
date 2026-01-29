import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import DATA_RAW
from src.utils import get_logger

logger = get_logger(__name__)


def load_resume_dataset(
    filepath: Optional[str | Path] = None,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    if filepath is None:
        filepath = DATA_RAW / "resume-dataset.csv"
    
    logger.info(f"Loading resume dataset from {filepath}")
    
    df = pd.read_csv(filepath)
    
    expected_cols = ["ID", "Resume_str", "Resume_html", "Category"]
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    
    logger.info(f"Loaded {len(df)} resumes")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Categories: {df['Category'].nunique()} unique values")
    
    if sample_n is not None:
        logger.info(f"Sampling first {sample_n} rows")
        df = df.head(sample_n)
    
    return df


def load_jobs_dataset(
    filepath: Optional[str | Path] = None,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    if filepath is None:
        filepath = DATA_RAW / "jobs.csv"
    
    logger.info(f"Loading jobs dataset from {filepath}")
    
    df = pd.read_csv(filepath)
    
    required_cols = ["description"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info(f"Loaded {len(df)} job postings")
    logger.info(f"Columns: {list(df.columns)}")
    
    if "major_job" in df.columns:
        logger.info(f"Major job categories: {df['major_job'].nunique()} unique values")
    
    if sample_n is not None:
        logger.info(f"Sampling first {sample_n} rows")
        df = df.head(sample_n)
    
    return df


def get_resume_categories(df: pd.DataFrame) -> list[str]:
    return sorted(df["Category"].unique().tolist())


def get_job_categories(df: pd.DataFrame) -> list[str]:
    if "major_job" in df.columns:
        return sorted(df["major_job"].dropna().unique().tolist())
    return []


def print_dataset_summary(resume_df: pd.DataFrame, jobs_df: pd.DataFrame):
    logger.info("DATASET SUMMARY")
    logger.info("Resumes:")
    logger.info(f"Total records: {len(resume_df)}")
    logger.info(f"Unique categories: {resume_df['Category'].nunique()}")
    logger.info(f"Avg resume length: {resume_df['Resume_str'].str.len().mean():.0f} chars")
    logger.info(f"Null Resume_str: {resume_df['Resume_str'].isna().sum()}")

    logger.info("Jobs:")
    logger.info(f"Total records: {len(jobs_df)}")
    if "major_job" in jobs_df.columns:
        logger.info(f"Unique major_job: {jobs_df['major_job'].nunique()}")
    logger.info(f"Avg description length: {jobs_df['description'].str.len().mean():.0f} chars")
    logger.info(f"Null descriptions: {jobs_df['description'].isna().sum()}")


if __name__ == "__main__":
    resumes = load_resume_dataset()
    jobs = load_jobs_dataset()
    print_dataset_summary(resumes, jobs)
    print("Sample resume categories:")
    for cat in get_resume_categories(resumes)[:10]:
        print(cat)

    print("Sample job categories:")
    for cat in get_job_categories(jobs)[:10]:
        print(cat)
