"""
Utils: I/O Operations

Consistent file reading/writing across the project.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def ensure_parent_dir(filepath: str | Path):
    """
    Ensure parent directory exists for a file path.
    
    Args:
        filepath: Path to file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(filepath: str | Path) -> List[Dict[str, Any]]:
    """
    Read JSONL file (one JSON object per line).
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(filepath: str | Path, data: List[Dict[str, Any]]):
    """
    Write JSONL file (one JSON object per line).
    
    Args:
        filepath: Path to output JSONL file
        data: List of dictionaries to write
    """
    ensure_parent_dir(filepath)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def read_json(filepath: str | Path) -> Dict[str, Any]:
    """
    Read standard JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(filepath: str | Path, data: Dict[str, Any], indent: int = 2):
    """
    Write standard JSON file.
    
    Args:
        filepath: Path to output JSON file
        data: Dictionary to write
        indent: Indentation level (default: 2)
    """
    ensure_parent_dir(filepath)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_csv(filepath: str | Path, **kwargs) -> pd.DataFrame:
    """
    Read CSV file using pandas.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
        
    Returns:
        DataFrame
    """
    return pd.read_csv(filepath, **kwargs)


def write_csv(filepath: str | Path, df: pd.DataFrame, **kwargs):
    """
    Write CSV file using pandas.
    
    Args:
        filepath: Path to output CSV file
        df: DataFrame to write
        **kwargs: Additional arguments to pass to df.to_csv
    """
    ensure_parent_dir(filepath)
    df.to_csv(filepath, **kwargs)
