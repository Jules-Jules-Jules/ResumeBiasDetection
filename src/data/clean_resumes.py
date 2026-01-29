import re
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Optional

from src.config import DATA_PROCESSED, SEED
from src.utils import get_logger, write_jsonl, set_seed

logger = get_logger(__name__)


def strip_html(text: str) -> str:
    if pd.isna(text):
        return ""
    
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    
    text = strip_html(text)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\n+', '\n\n', text)
    text = text.strip()
    
    return text


def normalize_category(category: str) -> str:
    if pd.isna(category):
        return "unknown"
    
    normalized = category.lower()
    normalized = re.sub(r'[-\s]+', '_', normalized)
    
    return normalized


def clean_resumes(
    df: pd.DataFrame,
    min_text_length: int = 50,
) -> list[dict]:
    logger.info(f"Cleaning {len(df):,} resumes...")
    records = []
    dropped_empty = 0
    dropped_short = 0
    
    for idx, row in df.iterrows():
        text = row.get("Resume_str", "")
        if pd.isna(text) or len(str(text).strip()) == 0:
            text = row.get("Resume_html", "")
        text_clean = clean_text(text)
        if len(text_clean) == 0:
            dropped_empty += 1
            continue
        if len(text_clean) < min_text_length:
            dropped_short += 1
            continue
        record = {
            "resume_id": int(row["ID"]),
            "occupation": normalize_category(row["Category"]),
            "text": text_clean,
            "text_length": len(text_clean),
        }
        
        records.append(record)
    
    logger.info(f"Cleaned resumes: {len(records)}")
    logger.info(f"Dropped empty: {dropped_empty}")
    logger.info(f"Dropped too short (< {min_text_length} chars): {dropped_short}")
    logger.info(f"Retention rate: {100 * len(records) / len(df):.1f}%")
    lengths = [r["text_length"] for r in records]
    logger.info(f"Text length min={min(lengths)}, median={int(pd.Series(lengths).median())}, max={max(lengths)}")
    
    return records


def save_cleaned_resumes(
    records: list[dict],
    output_path: Optional[Path] = None,
):
    if output_path is None:
        output_path = DATA_PROCESSED / "resumes_clean.jsonl"
    
    logger.info(f"Saving {len(records)} cleaned resumes to {output_path}")
    write_jsonl(output_path, records)
    logger.info("Saved resumes")


def get_occupation_counts(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    counts = df.groupby("occupation").size().reset_index(name="count")
    counts = counts.sort_values("count", ascending=False)
    return counts


if __name__ == "__main__":
    from src.data.load_raw import load_resume_dataset
    set_seed(SEED)
    df = load_resume_dataset()
    records = clean_resumes(df, min_text_length=50)
    save_cleaned_resumes(records)
    counts = get_occupation_counts(records)
    print("Occupation distribution:")
    print(counts.to_string(index=False))
