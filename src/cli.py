import sys
import argparse
from pathlib import Path

from src.config import ensure_dirs, get_path_summary, SEED
from src.utils import set_seed, get_logger


def cmd_smoke():
    logger = get_logger(__name__)
    
    logger.info("Running smoke test")
    
    set_seed(SEED)
    logger.info(f"Set random seed to {SEED}")
    
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.info(f"Python version: {py_version}")
    
    try:
        import numpy as np
        logger.info(f"NumPy version: {np.__version__}")
    except ImportError:
        logger.error("NumPy not installed")
        return False
    
    try:
        import pandas as pd
        logger.info(f"Pandas version: {pd.__version__}")
    except ImportError:
        logger.error("Pandas not installed")
        return False
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        logger.error("PyTorch not installed")
        return False
    
    try:
        import sklearn
        logger.info(f"Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        logger.error("Scikit-learn not installed")
        return False
    
    try:
        import sentence_transformers
        logger.info(f"Sentence-transformers version: {sentence_transformers.__version__}")
    except ImportError:
        logger.error("Sentence-transformers not installed")
        return False
    
    ensure_dirs()
    logger.info("All required directories exist")
    
    paths = get_path_summary()
    logger.info("Path configuration:")
    for key, value in paths.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Smoke test passed")
    return True


def cmd_info():
    logger = get_logger(__name__)
    
    logger.info("System Information")
    logger.info(f"Python: {sys.version}")
    
    paths = get_path_summary()
    logger.info("\nProject Paths:")
    for key, value in paths.items():
        exists = "exists" if Path(value).exists() else "missing"
        logger.info(f"  {key}: {value} ({exists})")
    
    return True


def cmd_phase1():
    import subprocess
    logger = get_logger(__name__)
    logger.info("Phase 1: Data Processing")
    
    script_path = Path(__file__).parent.parent / "scripts" / "phase1_build_all.sh"
    
    if not script_path.exists():
        logger.error(f"Phase 1 script not found: {script_path}")
        return False
    
    logger.info(f"Running: {script_path}")
    result = subprocess.run([str(script_path)], shell=True)
    
    return result.returncode == 0


def cmd_phase2():
    logger = get_logger(__name__)
    logger.info("Phase 2: Scoring & Evaluation")
    
    try:
        logger.info("\nStep 1: Computing similarity scores")
        from src.scoring.compute_scores import score_pairs
        df_scored = score_pairs()
        
        logger.info("\nStep 2: Evaluating retrieval quality")
        from src.scoring.metrics import evaluate_retrieval
        from src.config.paths import DATA_DIR
        
        pairs_path = DATA_DIR / "processed" / "pairs_scored_phase2.csv"
        metrics_path = DATA_DIR / "processed" / "phase2_retrieval_metrics.csv"
        evaluate_retrieval(pairs_path, metrics_path)
        
        logger.info("\nStep 3: Training classifier heads")
        from src.models.train_head import train_linear_head, train_mlp_head
        
        logger.info("  Training linear head")
        train_linear_head()
        
        logger.info("  Training MLP head")
        train_mlp_head()
        
        logger.info("\nPhase 2 complete")
        logger.info("Next: Review results in notebooks/phase2_retrieval_eval.ipynb")
        
        return True
        
    except Exception as e:
        logger.error(f"Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cmd_phase3():
    logger = get_logger(__name__)
    logger.info("Phase 3: Bias Audit")
    
    try:
        from src.evaluation.run_bias_audit import run_bias_audit
        from src.config.paths import DATA_DIR
        
        for score_col in ['score_embed', 'score_tfidf']:
            logger.info(f"\nRunning bias audit for {score_col}")
            run_bias_audit(
                pairs_path=DATA_DIR / "processed" / "pairs_scored_phase2.csv",
                out_dir=DATA_DIR / "processed",
                score_cols=[score_col],
                topk=[1, 3, 5, 8],
                n_bootstrap=2000,
                confidence=0.95,
                seed=42
            )
        
        logger.info("\nPhase 3 complete")
        logger.info("Next: Review results in notebooks/03_phase3_bias_audit.ipynb")
        
        return True
        
    except Exception as e:
        logger.error(f"Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Resume Screening Audit CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    subparsers.add_parser("smoke", help="Run smoke test")
    subparsers.add_parser("info", help="Show system info")
    subparsers.add_parser("phase1", help="Run Phase 1")
    subparsers.add_parser("phase2", help="Run Phase 2")
    subparsers.add_parser("phase3", help="Run Phase 3")
    
    args = parser.parse_args()
    
    if args.command == "smoke":
        success = cmd_smoke()
    elif args.command == "info":
        success = cmd_info()
    elif args.command == "phase1":
        success = cmd_phase1()
    elif args.command == "phase2":
        success = cmd_phase2()
    elif args.command == "phase3":
        success = cmd_phase3()
    else:
        parser.print_help()
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
