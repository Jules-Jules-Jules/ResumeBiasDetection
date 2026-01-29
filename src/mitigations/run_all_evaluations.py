#!/usr/bin/env python3

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.mitigations.evaluate_mitigation_frozen import evaluate_mitigation_frozen
from src.mitigations.evaluate_mitigation_classifier import evaluate_mitigation_on_classifier


def main():
    print("="*70)
    print("PHASE 5: MITIGATION EVALUATION")
    print("="*70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check that mitigation files exist
    m1_file = Path('data/processed/mitigations/m1_masked_test.csv')
    m2_file = Path('data/processed/mitigations/m2_freqnorm_test.csv')
    
    if not m1_file.exists():
        print(f"\nError: M1 file not found: {m1_file}")
        print("Run: python src/mitigations/apply_m1_masking.py")
        return
    
    if not m2_file.exists():
        print(f"\nError: M2 file not found: {m2_file}")
        print("Run: python src/mitigations/apply_m2_frequency_normalization.py")
        return
    
    print(f"\nM1 masked test file found")
    print(f"M2 frequency-normalized test file found")
    
    evaluations = [
        ('minilm_frozen', 'm1', evaluate_mitigation_frozen, m1_file, None),
        ('minilm_frozen', 'm2', evaluate_mitigation_frozen, m2_file, None),
        ('minilm', 'm1', evaluate_mitigation_on_classifier, m1_file, 'mlp'),
        ('minilm', 'm2', evaluate_mitigation_on_classifier, m2_file, 'mlp'),
        ('e5small', 'm1', evaluate_mitigation_on_classifier, m1_file, 'mlp'),
        ('e5small', 'm2', evaluate_mitigation_on_classifier, m2_file, 'mlp'),
    ]
    
    results = []
    
    print(f"\n\nRunning {len(evaluations)} evaluations")
    print("="*70)
    
    for i, (model, mitigation, eval_func, mitigation_file, architecture) in enumerate(evaluations, 1):
        print(f"\n\n[{i}/{len(evaluations)}] Evaluating {model.upper()} on {mitigation.upper()}")
        print("-"*70)
        
        try:
            output_dir = Path(f'data/processed/mitigations/{model}_{mitigation}')
            
            if model == 'minilm_frozen':
                summary = eval_func(mitigation_file, output_dir)
            else:
                summary = eval_func(model, mitigation_file, output_dir, architecture)
            
            results.append(summary)
            print(f"\nCompleted {model} + {mitigation}")
            
        except Exception as e:
            print(f"\nError in {model} + {mitigation}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("SUMMARY OF ALL EVALUATIONS")
    print("="*70)
    
    summary_file = Path('data/processed/mitigations/phase5_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted {len(results)}/{len(evaluations)} evaluations")
    print(f"Summary saved to: {summary_file}")
    
    print("\n\nFLIP RATE COMPARISON:")
    print("-"*70)
    print(f"{'Model':<20} {'M0 (baseline)':<15} {'M1 (masking)':<15} {'M2 (frequency)':<15}")
    print("-"*70)
    
    m0_results = {}
    for model in ['minilm_frozen', 'minilm', 'e5small']:
        if model == 'minilm_frozen':
            try:
                phase2_file = Path('data/processed/phase2_results/phase2_retrieval_metrics.csv')
                if phase2_file.exists():
                    import pandas as pd
                    df = pd.read_csv(phase2_file)
                    df = df[(df['model'] == 'all-MiniLM-L6-v2') & 
                           (df['score_type'] == 'score_embed') & 
                           (df['scope'] == 'overall')]
                    if len(df) > 0:
                        m0_results[model] = f"N/A (gap: {df.iloc[0]['gap']:.3f})"
                    else:
                        m0_results[model] = "N/A"
                else:
                    m0_results[model] = "N/A"
            except:
                m0_results[model] = "N/A"
        else:
            try:
                phase4_file = Path(f'data/processed/classifier/{model}/counterfactual_flips.csv')
                if phase4_file.exists():
                    import pandas as pd
                    df = pd.read_csv(phase4_file)
                    flip_rate = df['flip_occurred'].mean() * 100
                    m0_results[model] = f"{flip_rate:.1f}%"
                else:
                    m0_results[model] = "N/A"
            except:
                m0_results[model] = "N/A"
    
    m1_results = {r['model']: f"{r['flip_rate']:.1f}%" for r in results if r['mitigation'] == 'M1'}
    m2_results = {r['model']: f"{r['flip_rate']:.1f}%" for r in results if r['mitigation'] == 'M2'}
    
    for model in ['minilm_frozen', 'minilm', 'e5small']:
        model_display = model.replace('_', ' ').title()
        m0 = m0_results.get(model, "N/A")
        m1 = m1_results.get(model, "N/A")
        m2 = m2_results.get(model, "N/A")
        print(f"{model_display:<20} {m0:<15} {m1:<15} {m2:<15}")
    
    print("-"*70)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPhase 5 evaluation complete")
    print("="*70)


if __name__ == '__main__':
    main()
