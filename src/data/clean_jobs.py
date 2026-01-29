import re
import pandas as pd
from pathlib import Path
from typing import Optional
import ast

from src.config import DATA_PROCESSED, SEED
from src.utils import get_logger, write_jsonl, set_seed

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    if text.startswith('[') and text.endswith(']'):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                text = '\n'.join(str(item) for item in parsed if item)
        except (ValueError, SyntaxError):
            text = text.strip('[]')
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\n+', '\n\n', text)
    text = text.strip()
    
    return text


def normalize_major_job(major_job: str) -> str:
    if pd.isna(major_job):
        return "unknown"
    normalized = major_job.lower()
    normalized = re.sub(r'[-\s]+', '_', normalized)
    
    return normalized


def clean_jobs(
    df: pd.DataFrame,
    min_text_length: int = 50,
    sample_n: Optional[int] = None,
) -> list[dict]:
    logger.info(f"Cleaning {len(df):,} job postings...")
    
    df = df.reset_index(drop=True)
    df['job_id'] = df.index
    records = []
    dropped_empty = 0
    dropped_short = 0
    
    for idx, row in df.iterrows():
        text = row.get("description", "")
        text_clean = clean_text(text)
        if len(text_clean) == 0:
            dropped_empty += 1
            continue
        if len(text_clean) < min_text_length:
            dropped_short += 1
            continue
        record = {
            "job_id": int(row["job_id"]),
            "occupation": normalize_major_job(row.get("major_job", "unknown")),
            "position": str(row.get("position", "")).strip() if pd.notna(row.get("position")) else "",
            "location": str(row.get("location", "")).strip() if pd.notna(row.get("location")) else "",
            "text": text_clean,
            "text_length": len(text_clean),
        }
        
        records.append(record)
    
    logger.info(f"Cleaned job postings: {len(records)}")
    logger.info(f"Dropped empty: {dropped_empty}")
    logger.info(f"Dropped too short (< {min_text_length} chars): {dropped_short}")
    logger.info(f"Retention rate: {100 * len(records) / len(df):.1f}%")
    if sample_n is not None:
        logger.info(f"Sampling up to {sample_n} jobs per occupation")
        df_records = pd.DataFrame(records)
        sampled = df_records.groupby("occupation").apply(
            lambda x: x.sample(n=min(sample_n, len(x)), random_state=SEED)
        ).reset_index(drop=True)
        records = sampled.to_dict('records')
        logger.info(f"After sampling: {len(records)} jobs")
    lengths = [r["text_length"] for r in records]
    logger.info(f"Text length min={min(lengths)}, median={int(pd.Series(lengths).median())}, max={max(lengths)}")
    
    return records


def save_cleaned_jobs(
    records: list[dict],
    output_path: Optional[Path] = None,
):
    if output_path is None:
        output_path = DATA_PROCESSED / "jobs_clean.jsonl"
    
    logger.info(f"Saving {len(records)} cleaned jobs to {output_path}")
    write_jsonl(output_path, records)
    logger.info("Saved jobs")


def get_occupation_counts(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    counts = df.groupby("occupation").size().reset_index(name="count")
    counts = counts.sort_values("count", ascending=False)
    return counts


if __name__ == "__main__":
    from src.data.load_raw import load_jobs_dataset
    set_seed(SEED)
    df = load_jobs_dataset()
    records = clean_jobs(df, min_text_length=50, sample_n=300)
    save_cleaned_jobs(records)
    counts = get_occupation_counts(records)
    print("Occupation distribution:")
    print(counts.to_string(index=False))
