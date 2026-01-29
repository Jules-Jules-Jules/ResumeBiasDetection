import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mitigations.name_masking import mask_resume_text


def apply_m1_masking():
    print("~"*70)
    print("M1 MITIGATION: NAME MASKING")
    print("~"*70)
    
    test_path = Path('data/processed/classifier/minilm/predictions_test.csv')
    print(f"\nLoading test set from: {test_path}")
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test set not found: {test_path}")
    
    df = pd.read_csv(test_path)
    print(f"Loaded {len(df)} test pairs")
    
    print("\nApplying name masking")
    df['resume_text_masked'] = df['resume_text'].apply(mask_resume_text)
    
    masked_count = df['resume_text_masked'].str.contains(r'\[NAME\]').sum()
    print(f"Successfully masked {masked_count}/{len(df)} resumes")
    
    output_dir = Path('data/processed/mitigations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'm1_masked_test.csv'
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved masked test set to: {output_path}")
    
    print("~"*70)
    print("EXAMPLE TRANSFORMATION:")
    print("~"*70)
    sample_idx = df[df['resume_text'].str.contains('Brad|Darnell|Emily|Aisha')].index[0]
    print("\nOriginal (first 300 chars):")
    print(df.loc[sample_idx, 'resume_text'][:300])
    print("\nMasked (first 300 chars):")
    print(df.loc[sample_idx, 'resume_text_masked'][:300])
    
    print("M1 MASKING COMPLETE")
    print("~"*70)


if __name__ == '__main__':
    apply_m1_masking()
