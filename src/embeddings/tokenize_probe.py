from pathlib import Path
from typing import Dict, List
import json

import pandas as pd
import numpy as np
from transformers import AutoTokenizer


def tokenize_names(
    names: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> pd.DataFrame:
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    results = []
    
    for name in names:
        tokens = tokenizer.tokenize(name)
        token_ids = tokenizer.encode(name, add_special_tokens=False)
        
        results.append({
            'name': name,
            'n_subtokens': len(tokens),
            'tokens': tokens,
            'token_ids': token_ids,
            'name_length': len(name),
            'is_single_token': len(tokens) == 1,
            'first_token': tokens[0] if tokens else None
        })
    
    return pd.DataFrame(results)


def analyze_tokenization_bias(
    tokenization_df: pd.DataFrame,
    deltas_df: pd.DataFrame,
    demographic_group: str,
    name_col: str = 'name'
) -> Dict:
    results = {
        'demographic_group': demographic_group,
        'n_names': len(tokenization_df),
        'mean_subtokens': tokenization_df['n_subtokens'].mean(),
        'std_subtokens': tokenization_df['n_subtokens'].std(),
        'pct_single_token': (tokenization_df['is_single_token'].sum() / len(tokenization_df)) * 100,
        'max_subtokens': tokenization_df['n_subtokens'].max(),
        'min_subtokens': tokenization_df['n_subtokens'].min(),
    }
    
    weird_names = tokenization_df[tokenization_df['n_subtokens'] > 2]
    if len(weird_names) > 0:
        results['weird_tokenizations'] = weird_names[['name', 'tokens', 'n_subtokens']].to_dict('records')
    
    return results


def run_tokenization_probe(
    name_lookup_path: Path,
    model_names: List[str] = ["sentence-transformers/all-MiniLM-L6-v2"],
    output_path: Path = None
) -> pd.DataFrame:
    print("=" * 70)
    print("TOKENIZATION PROBE")
    print("=" * 70)
    
    print(f"Loading names from {name_lookup_path.name}")
    with open(name_lookup_path) as f:
        name_lookup = json.load(f)
    
    if 'name_pools' in name_lookup:
        name_pools = name_lookup['name_pools']
    else:
        raise ValueError("name_lookup.json must have 'name_pools' key")
    
    all_results = []
    
    for model_name in model_names:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")
        
        for demo_group, names in name_pools.items():
            first_names = names
            
            print(f"{demo_group}: {len(first_names)} names")
            
            tok_df = tokenize_names(first_names, model_name)
            tok_df['demographic_group'] = demo_group
            tok_df['model'] = model_name
            
            print(f"Mean subtokens: {tok_df['n_subtokens'].mean():.2f}")
            print(f"Single token: {(tok_df['is_single_token'].sum() / len(tok_df)) * 100:.1f}%")
            
            weird = tok_df[tok_df['n_subtokens'] > 2]
            if len(weird) > 0:
                print(f"{len(weird)} names split into >2 subtokens:")
                for _, row in weird.head(5).iterrows():
                    print(f"  {row['name']:15s} -> {row['tokens']}")
            
            all_results.append(tok_df)
    
    df_final = pd.concat(all_results, ignore_index=True)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path, index=False)
        print(f"Saved tokenization probe to {output_path.name}")
    
    print("=" * 70)
    print("TOKENIZATION PROBE COMPLETE")
    print("=" * 70)
    
    return df_final


if __name__ == "__main__":
    import argparse
    from src.config.paths import DATA_DIR
    
    parser = argparse.ArgumentParser(description="Run tokenization probe")
    parser.add_argument(
        '--name_lookup',
        type=Path,
        default=DATA_DIR / "processed" / "name_lookup.json",
        help="Path to name lookup JSON"
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=["sentence-transformers/all-MiniLM-L6-v2"],
        help="Model identifiers to test"
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=DATA_DIR / "processed" / "phase3_tokenization_probe.csv",
        help="Output path"
    )
    
    args = parser.parse_args()
    
    # Run probe
    df_results = run_tokenization_probe(
        name_lookup_path=args.name_lookup,
        model_names=args.models,
        output_path=args.out
    )
    
    print(f"Results saved to {args.out}")
