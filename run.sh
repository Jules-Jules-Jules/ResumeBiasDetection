#!/usr/bin/env bash

set -e

COMMAND=${1:-help}

case "$COMMAND" in
  smoke)
    echo "Running smoke test"
    python -m src.cli smoke
    ;;
  
  info)
    echo "System information"
    python -m src.cli info
    ;;
  
  phase1)
    echo "Running Phase 1: Data Processing"
    ./scripts/phase1_build_all.sh
    ;;
  
  phase2)
    echo "Running Phase 2: Scoring & Evaluation"
    python -m src.cli phase2
    ;;
  
  phase3)
    echo "Running Phase 3: Bias Audit"
    python -m src.evaluation.run_bias_audit
    ;;
  
  phase4-train)
    echo "Training Phase 4 Classifiers"
    echo "Training MiniLM model"
    python -m src.models.train_classifier --model minilm --architecture mlp
    echo ""
    echo "Training E5-Small model"
    python -m src.models.train_classifier --model e5small --architecture mlp
    ;;
  
  phase4-eval)
    echo "Evaluating Phase 4 Classifiers"
    echo "Evaluating MiniLM model"
    python -m src.models.eval_classifier --model minilm --architecture mlp
    echo ""
    echo "Evaluating E5-Small model"
    python -m src.models.eval_classifier --model e5small --architecture mlp
    ;;
  
  phase4)
    echo "Running Full Phase 4: Classifier Head Experiments"
    ./run.sh phase4-train
    echo ""
    ./run.sh phase4-eval
    ;;
  
  phase5)
    echo "Running Phase 5: Mitigation Experiments"
    echo "Evaluating M1 (name masking) and M2 (frequency normalization)"
    python -m src.mitigations.run_all_evaluations
    ;;
  
  all)
    echo "Running All Phases"
    echo ""
    ./run.sh phase1
    echo ""
    ./run.sh phase2
    echo ""
    ./run.sh phase3
    echo ""
    ./run.sh phase4
    echo ""
    ./run.sh phase5
    echo ""
    echo "Completed All Phases"
    ;;
  
  test)
    echo "Running tests"
    pytest tests/ -v
    ;;
  
  export-reports)
    echo "Exporting notebooks to HTML reports"
    python scripts/export_notebooks.py
    ;;
  
  help|*)
    echo "Resume Screening Audit"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  all - run all phases (phase1 through phase5)"
    echo "  smoke - run smoke test"
    echo "  info - show system info"
    echo "  phase1 - data processing"
    echo "  phase2 - scoring & evaluation"
    echo "  phase3 - bias audit"
    echo "  phase4 - train + eval both models"
    echo "  phase4-train - train classifiers"
    echo "  phase4-eval - eval classifiers"
    echo "  phase5 - mitigation experiments"
    echo "  export-reports - export notebooks to HTML in reports/"
    echo "  test - run tests"
    echo "  help - show this help"
    ;;
esac
