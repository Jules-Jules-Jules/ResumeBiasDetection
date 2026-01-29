# Attribution

This project uses third‑party datasets, pretrained models, and open‑source libraries. Please review each source's license/terms before reuse or redistribution.

## Datasets

### Resume Dataset
- **Source**: Kaggle “Resume Dataset” by Sneha Anbhawal
- **URL**: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- **Used in repo as**: `data/raw/resume-dataset.csv`
- **Usage**: Base resume text dataset used for preprocessing, pairing, scoring, and bias audit experiments.
- **License / Terms**: Refer to the dataset page on Kaggle for the applicable license/terms.
- **Modifications**: Cleaned/normalized, split into train/val/test, and augmented with demographic variants for counterfactual analysis.

### Job Descriptions Dataset
- **Source**: Kaggle “Scraped Job Descriptions” by Marco Cavaco
- **URL**: https://www.kaggle.com/datasets/marcocavaco/scraped-job-descriptions
- **Used in repo as**: `data/raw/jobs.csv`
- **Usage**: Job description corpus used to build job–resume pairs and evaluate retrieval/classification.
- **License / Terms**: Refer to the dataset page on Kaggle for the applicable license/terms.

## Models

### Sentence-Transformers Models
- **all-MiniLM-L6-v2**: Default embedding model used throughout the pipeline
  - Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  - Library: `sentence-transformers`
  - License: Apache-2.0 (per model card)
  - Usage: Encoding resumes and job descriptions into vector embeddings.
  
- **intfloat/e5-small-v2**: Alternative embedding model used for comparisons
  - Model card: https://huggingface.co/intfloat/e5-small-v2
  - Library: `transformers` / `sentence-transformers`
  - License: Refer to the model card (license may vary by model version)
  - Usage: Comparison experiments for robustness/model choice.

### Scikit-learn TF-IDF
- Used as classical baseline for comparison with neural embeddings

## Open-source libraries (non-exhaustive)

This repository relies heavily on:
- PyTorch (`torch`)
- Hugging Face Transformers (`transformers`)
- Sentence-Transformers (`sentence-transformers`)
- NumPy, pandas, scikit-learn
- matplotlib, seaborn

See `requirements.txt` for the full set of Python dependencies.

## Research Methodology

### Primary Research Inspiration
- **Paper**: "Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval" (AIES 2024)
- **Authors**: Kyra Wilson, Aylin Caliskan
- **Method**: This project is inspired by the evaluation framing and bias-audit approach described in the paper.

## Figures and reports

Plots in `figures/` and rendered reports in `reports/` are generated outputs from this codebase unless otherwise noted.