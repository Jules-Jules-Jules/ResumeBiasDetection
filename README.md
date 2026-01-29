# Bias in Embedding-Based Resume Screening Systems

This project investigates bias in AI resume screening systems by testing whether modern embedding-based models score or rank identical resumes differently when only the applicant's name changes.

## What it Doesin Embedding-Based Resume Screening Systems

This project investigates bias in AI resume screening systems by testing whether modern embedding-based models score or rank identical resumes differently when only the applicant’s name changes.

To study bias in a controlled way, each resume is duplicated into four counterfactual variants that differ only by name, corresponding to Black female, Black male, White female, and White male groups. Because the resume content is otherwise identical, any differences in scores, rankings, or decisions can be attributed to how the model processes name-related signals. The project measures bias at the scoring stage, examines how it changes once a decision boundary is added, and evaluates simple mitigation strategies that could realistically be applied at scale.

## Quick Start

First, install dependencies using the instructions in `SETUP.md`. 
Run `./run.sh all` to run phases 0-5.

The notebooks can be run end-to-end to regenerate plots and summary statistics.
The pre-run notebook data exists under `reports/`. You can open these html files in your browser to view the sample notebook outputs.

Finally, analysis of findings can be found in `docs/final_report.md`. This includes the project summary, methodology, and findings and conclusion.

## Evaluation

**Retrieval-style screening** was evaluated using AUC and score separation between matched and mismatched job–resume pairs. On this dataset, TF-IDF achieved an AUC of **0.6022**, while MiniLM embeddings achieved an AUC of **0.6347**, indicating better separation for embeddings in the retrieval setting. When a supervised classifier head was trained on frozen embeddings, discrimination increased further (MiniLM head **AUC = 0.7227**, E5-Small head **AUC = 0.7113**), showing that learning a decision boundary improved predictive performance relative to raw embedding similarity.

**Bias** was evaluated using counterfactual score comparisons and selection-rate analysis under identical resume content with only the name changed. Embeddings showed clear counterfactual sensitivity: the mean within-quartet score range was **0.031495** for embeddings versus approximately **0.000001** for TF-IDF. At the selection level, top-1 selection rates were not uniform across groups. For TF-IDF, the top-1 rate ranged from **31.9% (white_male)** to **15.0% (black_female)** (gap **16.9pp**). For embeddings, the top-1 rates ranged from **30.0% (black_female)** to **18.4% (white_female)** (gap **11.6pp**). After adding a classifier head, counterfactual differences sometimes crossed the decision boundary: decision flips occurred in **58/414 = 14.0%** of quartets for MiniLM and **62/414 = 15.0%** for E5-Small. The mean probability range across name variants was **0.1326** (MiniLM) and **0.1397** (E5-Small), with worst-case ranges reaching **0.9529** and **0.9718**, respectively.

**Mitigation experiments** tested whether simple inference-time changes reduce counterfactual instability. Name masking (replacing the first name with `[NAME]`) reduced flip rates to **0.0%** for both MiniLM and E5-Small and collapsed probability-range variability to approximately zero. In contrast, single-token/frequency-based name normalization did not reduce instability and increased flip rates to **16.4%** (MiniLM) and **25.4%** (E5-Small), with mean probability ranges **0.1529** and **0.2182**. Overall, the mitigation results indicate that removing the name signal directly was effective in this pipeline, while normalization-based approaches did not address the underlying sensitivity.


## Phases

- **Phase 0** – basic setup, smoke tests, checking paths
- **Phase 1** – clean raw resumes/jobs and save processed CSVs
- **Phase 2** – score job–resume pairs with embedding models and look at retrieval metrics
- **Phase 3** – build name-swapped resume quartets and measure how often scores/decisions flip
- **Phase 4** – train simple classifier heads (MiniLM + E5-Small) and check their flip rates
- **Phase 5** – run two mitigation ideas (name masking vs single-token normalization) and compare