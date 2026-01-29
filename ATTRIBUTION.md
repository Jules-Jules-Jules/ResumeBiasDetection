# Attribution

## Datasets

### Resume Dataset
- **Source**: Kaggle Resume Dataset by Sneha Anbhawal
- **URL**: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- **Usage**: Training and evaluation data for resume screening models
- **License**: Public dataset
- **Modifications**: Data was preprocessed, split into train/val/test sets, and augmented with demographic variants

### Job Descriptions Dataset
- **Source**: Scraped Job Descriptions by Marco Cavaco
- **URL**: https://www.kaggle.com/datasets/marcocavaco/scraped-job-descriptions
- **Usage**: Job description data for resume-job pair classification
- **License**: Public dataset

## Models

### Sentence-Transformers Models
- **all-MiniLM-L6-v2**: Baseline embedding model (90MB)
  - Source: sentence-transformers/all-MiniLM-L6-v2
  - HuggingFace: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  - License: Apache 2.0
  - Usage: Primary embedding model for resume and job description encoding
  
- **intfloat/e5-small-v2**: Secondary embedding model
  - Source: intfloat/e5-small-v2
  - HuggingFace: https://huggingface.co/intfloat/e5-small-v2
  - License: MIT
  - Usage: Alternative embedding model for comparison experiments

### Scikit-learn TF-IDF
- Used as classical baseline for comparison with neural embeddings

## Research Methodology

### Primary Research Inspiration
- **Paper**: "Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval" (AIES 2024)
- **Authors**: Kyra Wilson, Aylin Caliskan
- **Method**: This project is inspired by the methodology and approach presented in this paper for detecting bias in AI-powered resume screening systems

## AI-Generated Code

### GitHub Copilot
 While no complete methods or functions were AI-generated, In-line GitHub Copilot was used for autocompleting lines of code, syntax suggestions, error fixing, and improving code robustness