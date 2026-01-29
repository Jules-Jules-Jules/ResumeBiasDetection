import pandas as pd
from pathlib import Path
from typing import Optional
from collections import Counter

from src.config import DATA_PROCESSED, SEED, TRAIN_RATIO, TEST_RATIO
from src.utils import get_logger, read_jsonl, write_json, set_seed

logger = get_logger(__name__)


def create_stratified_splits(
    resume_ids: list[int],
    occupations: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = SEED,
) -> dict[str, list[int]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
    
    set_seed(random_seed)
    
    df = pd.DataFrame({
        "resume_id": resume_ids,
        "occupation": occupations,
    })
    
    splits = {"train": [], "val": [], "test": []}
    
    for occupation, group in df.groupby("occupation"):
        n = len(group)
        ids = group["resume_id"].tolist()
        
        shuffled_df = group.sample(frac=1, random_state=random_seed)
        shuffled_ids = shuffled_df["resume_id"].tolist()
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits["train"].extend(shuffled_ids[:n_train])
        splits["val"].extend(shuffled_ids[n_train:n_train+n_val])
        splits["test"].extend(shuffled_ids[n_train+n_val:])
    
    logger.info("Split sizes:")
    logger.info(f"Train: {len(splits['train'])} ({100*len(splits['train'])/len(resume_ids):.1f}%)")
    logger.info(f"Val: {len(splits['val'])} ({100*len(splits['val'])/len(resume_ids):.1f}%)")
    logger.info(f"Test: {len(splits['test'])} ({100*len(splits['test'])/len(resume_ids):.1f}%)")
    
    return splits


def verify_no_leakage(splits: dict[str, list[int]]):
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    assert len(train_val_overlap) == 0, f"Train/val overlap: {len(train_val_overlap)} IDs"
    assert len(train_test_overlap) == 0, f"Train/test overlap: {len(train_test_overlap)} IDs"
    assert len(val_test_overlap) == 0, f"Val/test overlap: {len(val_test_overlap)} IDs"
    
    logger.info("No overlap between splits")


def verify_stratification(
    splits: dict[str, list[int]],
    resume_records: list[dict],
):
    id_to_occ = {r["resume_id"]: r["occupation"] for r in resume_records}
    
    logger.info("Occupation distribution per split:")
    for split_name in ["train", "val", "test"]:
        occs = [id_to_occ[rid] for rid in splits[split_name]]
        occ_counts = Counter(occs)
        logger.info(f"\n{split_name.upper()} ({len(occs)} resumes):")
        for occ, count in occ_counts.most_common(5):
            logger.info(f"{occ}: {count}")


def save_splits(
    splits: dict[str, list[int]],
    output_path: Optional[Path] = None,
):
    if output_path is None:
        output_path = DATA_PROCESSED / "split_ids.json"
    logger.info(f"Saving splits to {output_path}")
    write_json(output_path, splits, indent=2)
    logger.info("Saved splits")


if __name__ == "__main__":
    set_seed(SEED)
    
    resumes = read_jsonl(DATA_PROCESSED / "resumes_clean.jsonl")
    logger.info(f"Loaded {len(resumes)} cleaned resumes")
    
    resume_ids = [r["resume_id"] for r in resumes]
    occupations = [r["occupation"] for r in resumes]
    
    logger.info(f"Creating splits with ratios train={TRAIN_RATIO}, test={TEST_RATIO}")
    val_ratio = 1.0 - TRAIN_RATIO - TEST_RATIO
    
    splits = create_stratified_splits(
        resume_ids=resume_ids,
        occupations=occupations,
        train_ratio=TRAIN_RATIO,
        val_ratio=val_ratio,
        test_ratio=TEST_RATIO,
        random_seed=SEED,
    )
    
    verify_no_leakage(splits)
    verify_stratification(splits, resumes)
    save_splits(splits)
    print("Splits created")
