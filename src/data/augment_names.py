import pandas as pd
import random
from pathlib import Path
from typing import Optional

from src.config import DATA_PROCESSED, SEED, NAME_POOLS, DEMOGRAPHIC_GROUPS
from src.utils import get_logger, read_jsonl, write_jsonl, write_json, set_seed

logger = get_logger(__name__)
LAST_NAME = "Williams"


def prepend_name(first_name: str, last_name: str, text: str) -> str:
    return f"{first_name} {last_name}\n\n{text}"


def augment_resume(
    base_record: dict,
    split: str,
) -> list[dict]:
    variants = []
    
    for group in DEMOGRAPHIC_GROUPS:
        # Randomly select a name from this group's pool
        first_name = random.choice(NAME_POOLS[group])
        
        # Create variant record
        variant = {
            "base_resume_id": base_record["resume_id"],
            "variant_id": f"{base_record['resume_id']}_{group}",
            "demographic_group": group,
            "name_first": first_name,
            "name_last": LAST_NAME,
            "text": prepend_name(first_name, LAST_NAME, base_record["text"]),
            "text_base": base_record["text"],  # Store original for verification
            "occupation": base_record["occupation"],
            "split": split,
        }
        
        variants.append(variant)
    
    return variants


def augment_all_resumes(
    resume_records: list[dict],
    splits: dict[str, list[int]],
) -> list[dict]:
    # Create lookup from resume_id to split
    id_to_split = {}
    for split_name, ids in splits.items():
        for rid in ids:
            id_to_split[rid] = split_name
    
    logger.info(f"Augmenting {len(resume_records):,} resumes with {len(DEMOGRAPHIC_GROUPS)} demographic variants each...")
    
    all_variants = []
    
    for record in resume_records:
        resume_id = record["resume_id"]
        split = id_to_split.get(resume_id, "unknown")
        
        # Create variants
        variants = augment_resume(record, split)
        all_variants.extend(variants)
    
    logger.info(f"Created {len(all_variants):,} augmented resumes ({len(DEMOGRAPHIC_GROUPS)}x expansion)")
    
    # Verify balance
    df = pd.DataFrame(all_variants)
    group_counts = df.groupby("demographic_group").size()
    logger.info(f"\nDemographic group balance:")
    for group in DEMOGRAPHIC_GROUPS:
        count = group_counts.get(group, 0)
        logger.info(f"  {group}: {count:,}")
    
    assert group_counts.nunique() <= 1, "Groups are not balanced!"
    logger.info("✓ All demographic groups have equal counts")
    
    return all_variants


def verify_counterfactual_property(variants: list[dict], n_samples: int = 5):
    df = pd.DataFrame(variants)
    base_ids = df["base_resume_id"].unique()
    
    logger.info(f"\n{'='*70}")
    logger.info("COUNTERFACTUAL VERIFICATION")
    logger.info(f"{'='*70}")
    
    # Sample some base IDs
    sampled_base_ids = random.sample(list(base_ids), min(n_samples, len(base_ids)))
    
    for base_id in sampled_base_ids:
        base_variants = df[df["base_resume_id"] == base_id].to_dict('records')
        
        logger.info(f"\nBase resume ID: {base_id}")
        logger.info(f"Occupation: {base_variants[0]['occupation']}")
        
        # Check that base text is identical across all variants
        base_texts = set(v["text_base"] for v in base_variants)
        assert len(base_texts) == 1, f"Base texts differ for resume {base_id}!"
        
        # Show name differences
        for variant in base_variants:
            name = f"{variant['name_first']} {variant['name_last']}"
            first_line = variant['text'].split('\n')[0]
            logger.info(f"  {variant['demographic_group']:20s} -> {first_line}")
        
        logger.info("  ✓ Base text identical across all variants")
    
    logger.info(f"{'='*70}\n")


def create_name_lookup_metadata() -> dict:
    metadata = {
        "last_name": LAST_NAME,
        "demographic_groups": DEMOGRAPHIC_GROUPS,
        "name_pools": NAME_POOLS,
        "names_per_group": {group: len(names) for group, names in NAME_POOLS.items()},
        "random_seed": SEED,
        "methodology": "Following Bertrand & Mullainathan (2004), using distinctive names as demographic proxies",
    }
    
    return metadata


def save_augmented_resumes(
    variants: list[dict],
    output_path: Optional[Path] = None,
):
    if output_path is None:
        output_path = DATA_PROCESSED / "resumes_augmented.jsonl"
    
    logger.info(f"Saving {len(variants):,} augmented resumes to: {output_path}")
    write_jsonl(output_path, variants)
    logger.info("✓ Saved successfully")


def save_name_lookup(
    metadata: dict,
    output_path: Optional[Path] = None,
):
    if output_path is None:
        output_path = DATA_PROCESSED / "name_lookup.json"
    
    logger.info(f"Saving name lookup metadata to: {output_path}")
    write_json(output_path, metadata, indent=2)
    logger.info("✓ Saved successfully")


if __name__ == "__main__":
    set_seed(SEED)
    
    # Load data
    resumes = read_jsonl(DATA_PROCESSED / "resumes_clean.jsonl")
    from src.utils import read_json
    splits = read_json(DATA_PROCESSED / "split_ids.json")
    
    logger.info(f"Loaded {len(resumes):,} base resumes")
    
    # Augment
    variants = augment_all_resumes(resumes, splits)
    
    # Verify counterfactual property
    verify_counterfactual_property(variants, n_samples=3)
    
    # Save
    save_augmented_resumes(variants)
    
    # Save name lookup metadata
    metadata = create_name_lookup_metadata()
    save_name_lookup(metadata)
    
    print("\n Resume augmentation completed")
