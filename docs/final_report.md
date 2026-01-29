# Resume Screening Bias Audit via Retrieval and Counterfactual Name Swaps

## 1. Introduction and motivation

AI-assisted hiring is now common, and resume screening is one of the first stages where automation can affect who advances. A key concern is that screening systems may respond to proxy signals for protected attributes even when resume content is otherwise equivalent. This matters because screening is typically implemented as ranking (top-k) or thresholding (accept/reject), which can turn small score differences into different outcomes.

This project performs a controlled audit of that risk. I hold resume content constant and change only the applicant name (race and gender cue), then measure how similarity scores, rankings, and downstream decisions change. The main purpose is to quantify whether the pipeline is counterfactually invariant to names, identify where sensitivity enters the system, and test whether simple mitigations reduce that sensitivity.

## 2. Relation to prior work

This study is based on Wilson & Caliskan’s paper *Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval* (2024). Their work frames screening as a retrieval problem (job description as query, resumes as documents) and uses counterfactual name swaps to test whether embedding models show systematic preferences across demographic groups. They report large and frequent disparities across their models and tests, including strong evidence of race and intersectional bias in selection outcomes.

My project follows the same core audit logic (retrieval + counterfactual names), but differs in the specific embedding models used in the main pipeline, and the addition of a classifier-head stage and mitigation experiments.

## 3. Project plan and technical overview (Phases 1–5)

**Phase 1 (Data build):** Clean resumes and job descriptions, map them into shared occupation categories, and construct counterfactual resume quartets by prepending names associated with four demographic categories.

**Phase 2 (Retrieval evaluation):** Compute similarity scores for job–resume pairs using (1) TF-IDF cosine similarity and (2) embedding cosine similarity, and evaluate whether matched pairs score higher than mismatched pairs.

**Phase 3 (Bias audit):** Measure counterfactual score deltas under name swaps, simulate screening by selecting top-k resumes per job, and quantify group selection-rate disparities with statistical testing and bootstrap confidence intervals.

**Phase 4 (Classifier head):** Train a classifier head on frozen embeddings to predict match/accept decisions, then measure counterfactual decision flip rates and probability-range sensitivity across name variants.

**Phase 5 (Mitigations):** Test name masking and name normalization strategies and evaluate whether they reduce counterfactual instability and group gaps.

## 4. Data and preprocessing (Phase 1)

After cleaning, the dataset contains 2,483 base resumes and 3,307 job postings (with jobs capped per occupation to reduce imbalance). Each base resume is expanded into four counterfactual variants:
- White male
- White female
- Black male
- Black female

This produces 9,932 augmented resumes (exactly 4× the base count). Job–resume pairs are generated to create a final evaluation dataset of 9,920 pairs, with an occupation match rate of 48.4% and mismatch rate of 51.6%. The data is split into train/validation/test with no leakage.

The counterfactual design is important: within each quartet, the resume content is identical and only the name changes. This allows the audit to interpret differences as being caused by name-associated signals rather than differences in qualifications.

## 5. Methods and metrics

This project uses four main metric families:

**(1) Retrieval quality:** Discrimination between occupation-matched and mismatched pairs, summarized by AUC and related separation statistics.

**(2) Counterfactual score sensitivity:** For each quartet, compute the range of similarity scores across the four name variants and compute group mean score differences (e.g., White male minus Black male).

**(3) Selection-rate disparities (top-k):** Simulate screening by selecting the top-k candidates per job and measure how often each group appears in the selected set. Evaluate whether selection is uniform across groups using chi-square tests and estimate uncertainty with bootstrapping.

**(4) Decision instability (classifier head):** After training a classifier head, compute the probability range across name variants and the flip rate (fraction of quartets where predicted label changes across names).

## 6. Results

### 6.1 Retrieval evaluation (Phase 2)

Retrieval performance is modest but above random. TF-IDF achieves AUC = 0.6022, while the embedding model (MiniLM) achieves AUC = 0.6347. This confirms that the retrieval setup is capturing some meaningful matching signal, and embeddings perform better than TF-IDF on this dataset.

Performance varies substantially by occupation category. Some occupations show very high AUC, while others are much weaker, including at least one degenerate category slice where AUC could not be computed. This indicates that model behavior and any bias effects may be occupation-dependent.

### 6.2 Name sensitivity in similarity scores (Phases 2–3)

Embedding similarity scores are substantially more sensitive to name swaps than TF-IDF. The mean within-quartet score range across the four name variants is about 0.0315 for embeddings, compared to about 0.000001 for TF-IDF. This implies that embeddings encode the name strongly enough to shift cosine similarity in a measurable way even when the resume content is unchanged.

At the group level, mean score differences across demographic variants are small but consistently non-zero. However, the worst-case within-quartet gaps are much larger than the means, with a maximum observed gap around 0.22. This suggests the name effect has a heavy tail: most cases show small shifts, but a subset shows very large shifts.

### 6.3 Selection-rate disparities and statistical testing (Phase 3)

When simulating screening by selecting top-k resumes per job, selection outcomes are not uniform across groups.

For embeddings at top-1 selection, group selection rates are uneven, with a maximum gap of about 11.6 percentage points between the most- and least-selected group in the aggregated results. TF-IDF shows an even larger top-1 gap of about 16.9 percentage points, despite TF-IDF showing near-zero counterfactual score deltas in the score-sensitivity analysis.

A chi-square test strongly rejects uniform selection across groups (p < 1e-12) with a small-to-moderate effect size (Cramér’s V ≈ 0.076). Bootstrapped confidence intervals show that some group comparisons at k=1 are consistently non-zero, indicating that at least some disparities are robust to resampling noise.

These results suggest that selection-rate disparities can arise both from direct score sensitivity (as in embeddings) and from structural properties of ranking/tie-breaking (as suggested by TF-IDF’s large selection gap despite minimal score deltas).

### 6.4 Mechanism probe: tokenization (Phase 3)

Tokenization differs across name groups. White-associated names are predominantly single-token, while Black-associated names are frequently split into multiple subtokens. This provides a plausible low-level mechanism for some embedding sensitivity, because the name input is not represented equivalently across groups.

However, tokenization differences alone do not explain the behavior, because mitigation experiment M2 shows that forcing names into single-token forms does not reduce instability.

### 6.5 Classifier head: decision flips and probability instability (Phase 4)

Training a classifier head improves predictive performance but introduces clear decision-level sensitivity to names.

The classifier head achieves AUC = 0.7227 for MiniLM and AUC = 0.7113 for E5-Small. Despite this improved AUC, counterfactual decision flips are common:
- MiniLM flip rate ≈ 14.0%
- E5-Small flip rate ≈ 15.0%

Probability instability is also meaningful: mean within-quartet probability ranges are around 0.13–0.14, and there are extreme examples where the predicted probability spans almost the entire [0, 1] interval depending only on the name.

Selection-rate gaps also appear under the classifier head and can increase at larger k. This indicates that once a learned decision boundary is added, name-driven embedding perturbations can cross the boundary and produce different accept/reject outcomes for otherwise identical resumes.

### 6.6 Mitigations (Phase 5)

Two mitigation strategies were tested:

**(1) Name masking:** Replacing the first name with a neutral token ([NAME]) reduces the flip rate to 0.0% for both trained models and collapses the probability-range metric to approximately 0. This indicates that removing the name token eliminates counterfactual instability in this pipeline.

**(2) Single-token normalization:** Replacing names with single-token alternatives within demographic groups increases flip rates (to roughly 23–27% in the reported runs). This suggests that token count is not the main driver. Instead, specific name forms carry demographic associations that the model responds to.

Overall, masking is effective in this controlled audit, while normalization is not.

## 7. Discussion and comparison to Wilson & Caliskan

This project supports the same general concern highlighted by Wilson & Caliskan: retrieval-style resume screening can change outcomes under counterfactual name swaps, implying that demographic proxy signals can influence ranking and selection even when resume content is constant.

At the same time, my results differ in directionality. Wilson & Caliskan report more consistently “White-preferred” patterns across a wide set of tests and large MTE models. In my results, directionality depends more on the specific model, stage and metric: some aggregated top-k outcomes do not show a single consistently favored group, but statistical testing and worst-case behavior show that disparities are still present and can be large in individual cases. My classifier-head results also emphasize an additional point: converting retrieval signals into a hard decision boundary can create frequent accept/reject flips under name swaps, which is a more directly decision-focused failure mode than score deltas alone.

A major practical implication from this comparison is that bias evaluation depends heavily on where in the pipeline it is measured (raw similarity scores, ranked selection, or downstream classifier decisions). A pipeline can appear weakly biased by mean score deltas while still producing measurable disparities after ranking or thresholding.

## 8. Conclusion

Across all phases, the pipeline shows measurable sensitivity to demographic name cues. Embedding similarity scores shift under counterfactual name swaps, selection-rate disparities appear when simulating top-k screening, and a trained classifier head produces substantial decision instability (14–15% flip rates across identical resume quartets). Mitigation experiments show that masking names eliminates counterfactual decision flips in this setup, while single-token normalization does not help and can worsen instability. Overall, the results are consistent with prior work showing that retrieval-based resume screening systems should be audited using counterfactual tests. They show overall that decision boundaries can amplify small representation differences into different outcomes.