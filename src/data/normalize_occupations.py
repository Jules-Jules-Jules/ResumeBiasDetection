import pandas as pd
import json
from pathlib import Path
from typing import Optional

from src.config import DATA_PROCESSED
from src.utils import get_logger, read_jsonl, write_json

logger = get_logger(__name__)

OCCUPATION_MAPPING = {
    "information_technology": ["information_and_communications_technology_professionals", "information_and_communications_technicians"],
    "digital_media": ["information_and_communications_technology_professionals"],
    "engineering": ["science_and_engineering_professionals", "science_and_engineering_associate_professionals"],
    "aviation": ["science_and_engineering_professionals"],
    "automobile": ["science_and_engineering_professionals", "drivers_and_mobile_plant_operators"],
    "business_development": ["business_and_administration_professionals", "business_and_administration_associate_professionals", "administrative_and_commercial_managers"],
    "consultant": ["business_and_administration_professionals", "administrative_and_commercial_managers"],
    "finance": ["business_and_administration_professionals"],
    "accountant": ["business_and_administration_professionals"],
    "banking": ["business_and_administration_professionals"],
    "sales": ["sales_workers", "business_and_administration_associate_professionals"],
    "public_relations": ["business_and_administration_professionals"],
    "healthcare": ["health_professionals", "health_associate_professionals"],
    "fitness": ["health_professionals", "health_associate_professionals"],
    "chef": ["cleaners_and_helpers"],
    "hr": ["business_and_administration_professionals"],
    "bpo": ["customer_services_clerks"],
    "teacher": ["teaching_professionals"],
    "advocate": ["business_and_administration_professionals"],
    "designer": ["business_and_administration_professionals"],
    "arts": ["business_and_administration_professionals"],
    "apparel": ["sales_workers"],
    "agriculture": ["market_oriented_skilled_agricultural_workers"],
    "construction": ["science_and_engineering_associate_professionals"],
}


def get_matching_job_occupations(resume_occupation: str) -> list[str]:
    return OCCUPATION_MAPPING.get(resume_occupation, [])


def is_occupation_match(resume_occupation: str, job_occupation: str) -> bool:
    matching_jobs = get_matching_job_occupations(resume_occupation)
    return job_occupation in matching_jobs


def create_occupation_map(
    resume_occupations: list[str],
    job_occupations: list[str],
) -> dict:
    occupation_map = {
        "resume_occupations": sorted(resume_occupations),
        "job_occupations": sorted(job_occupations),
        "mapping": OCCUPATION_MAPPING,
        "coverage": {},
    }
    
    for res_occ in resume_occupations:
        matching_jobs = get_matching_job_occupations(res_occ)
        has_match = len(matching_jobs) > 0
        occupation_map["coverage"][res_occ] = {
            "has_matches": has_match,
            "matching_job_occupations": matching_jobs,
        }
    
    total = len(resume_occupations)
    with_matches = sum(1 for v in occupation_map["coverage"].values() if v["has_matches"])
    logger.info(f"Occupation coverage: {with_matches}/{total} ({100*with_matches/total:.1f}%) resume occupations have job matches")
    
    return occupation_map


def save_occupation_map(
    occupation_map: dict,
    output_path: Optional[Path] = None,
):
    if output_path is None:
        output_path = DATA_PROCESSED / "occupation_map.json"
    
    logger.info(f"Saving occupation map to {output_path}")
    write_json(output_path, occupation_map, indent=2)
    logger.info("Saved occupation map")


def save_occupation_counts(
    resume_counts: pd.DataFrame,
    job_counts: pd.DataFrame,
    output_path: Optional[Path] = None,
):
    if output_path is None:
        output_path = DATA_PROCESSED / "occupation_counts.csv"
    
    resume_counts = resume_counts.rename(columns={"count": "resume_count"})
    job_counts = job_counts.rename(columns={"count": "job_count"})
    
    merged = pd.merge(
        resume_counts,
        job_counts,
        on="occupation",
        how="outer"
    ).fillna(0)
    
    merged = merged.sort_values("resume_count", ascending=False)
    
    logger.info(f"Saving occupation counts to {output_path}")
    merged.to_csv(output_path, index=False)
    logger.info("Saved occupation counts")


if __name__ == "__main__":
    resumes = read_jsonl(DATA_PROCESSED / "resumes_clean.jsonl")
    jobs = read_jsonl(DATA_PROCESSED / "jobs_clean.jsonl")
    
    resume_occs = sorted(set(r["occupation"] for r in resumes))
    job_occs = sorted(set(j["occupation"] for j in jobs))
    
    logger.info(f"Resume occupations: {len(resume_occs)}")
    logger.info(f"Job occupations: {len(job_occs)}")
    
    occ_map = create_occupation_map(resume_occs, job_occs)
    save_occupation_map(occ_map)
    
    from src.data.clean_resumes import get_occupation_counts as get_resume_counts
    from src.data.clean_jobs import get_occupation_counts as get_job_counts
    
    resume_counts = get_resume_counts(resumes)
    job_counts = get_job_counts(jobs)
    
    save_occupation_counts(resume_counts, job_counts)
    
    print("Occupation mapping created")
    print("Resume occupations without job matches:")
    for res_occ in resume_occs:
        if not get_matching_job_occupations(res_occ):
            print(f"  {res_occ}")
