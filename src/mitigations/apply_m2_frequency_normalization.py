import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mitigations.name_frequency_sets import (
    get_frequency_matched_name,
    replace_name_with_frequency_matched,
    get_name_from_resume_text
)


def apply_m2_frequency_normalization():
    print("~"*70)
    print("M2 MITIGATION: NAME FREQUENCY NORMALIZATION")
    print("~"*70)
    
    test_path = Path('data/processed/classifier/minilm/predictions_test.csv')
    print(f"\nLoading test set from: {test_path}")
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test set not found: {test_path}")
    
    df = pd.read_csv(test_path)
    print(f"Loaded {len(df)} test pairs")
    
    print("\nApplying frequency normalization")
    
    def normalize_name_in_text(row):
        text = row['resume_text']
        demo_group = row['demographic_group']
        
        original_name = get_name_from_resume_text(text, demo_group)
        
        if original_name is None:
            return text
        
        seed = hash(row['resume_id']) % 10000
        new_name = get_frequency_matched_name(demo_group, original_name, seed=seed)
        
        normalized_text = replace_name_with_frequency_matched(text, original_name, new_name)
        
        return normalized_text
    
    df['resume_text_freqnorm'] = df.apply(normalize_name_in_text, axis=1)
    
    changed_count = (df['resume_text'] != df['resume_text_freqnorm']).sum()
    print(f"Successfully normalized {changed_count}/{len(df)} resumes")
    
    output_dir = Path('data/processed/mitigations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'm2_freqnorm_test.csv'
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved frequency-normalized test set to: {output_path}")
    
    print("EXAMPLE TRANSFORMATIONS:")
    print("~"*70)
    
    for group in ['white_male', 'white_female', 'black_male', 'black_female']:
        sample = df[df['demographic_group'] == group].iloc[0]
        print(f"\n{group}:")
        print(f"  Original (first 200 chars):")
        print(f"    {sample['resume_text'][:200]}...")
        print(f"  Normalized (first 200 chars):")
        print(f"    {sample['resume_text_freqnorm'][:200]}...")
    
    print("~"*70)
    print("M2 FREQUENCY NORMALIZATION COMPLETE")
    print("~"*70)


if __name__ == '__main__':
    apply_m2_frequency_normalization()
