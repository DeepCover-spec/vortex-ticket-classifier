# Experiment log

Score log for the report. The EDA notebook (section 15) points here. `01 baseline model.ipynb` wrote the first row. `02_model_experiments.ipynb` appends one row each time a setup is scored. Keep this table at the bottom of the file so new rows stay in it.

## How a model is chosen

Train-set cross-validation is not the selection metric. Train tickets repeat near-duplicate sentences, so 5-fold CV macro-F1 sits near 0.99 while the same model scores about 0.66 on `validation.csv`. Every comparison uses **validation macro-F1**. Accuracy is written beside it. A CV number in the table is an optimistic reference only.

The score to beat is baseline-01: **validation macro-F1 0.6640** (0.6731 as actually served, since the API takes the calibrated head's argmax). If a later run does not beat that, the API keeps `model/model.joblib` (`baseline-tfidf-22cd5e2d`).

`app/predictor.py` looks the team up from the fixed category-to-team table. Team is not a model output. `spam_irrelevant` is always non-urgent and has no secondary category.

## Data notes that shaped the baseline

- 11 categories, imbalanced. Training uses `class_weight="balanced"`. The headline metric is macro-F1.
- Languages: English, Singlish, Sinhala, Tamil, Tanglish, mixed. The baseline uses word n-grams plus character n-grams so Sinhala and Tamil do not need a tokenizer.
- `app/textproc.py` masks reference codes and digits, lowercases, and leaves emoji and other scripts in place. Training and the API import that same function.
- About 10% of tickets have a secondary category. About 10% are urgent, mostly inside `safety_conduct`.
- The numbers in the table are fit on train and scored on validation. The file in `model/` was refit on train + validation after that score was recorded, so do not rescore `model.joblib` on `validation.csv` and treat it as a new result.
- Fitted with scikit-learn 1.9.0. The container has to use that same version.

## What `02_model_experiments.ipynb` tries

These rows appear only after that notebook is run:

- LinearSVC `C` in {0.1, 1.0, 2.0}, and logistic regression `C=4`, with the baseline features. Fit on train, score on validation. No cross-validation.
- Word n-grams `(1, 3)` and character n-grams `(3, 5)`, LinearSVC `C=0.5`.
- Frozen multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) plus logistic regression, aimed at Sinhala, Tamil, Singlish and Tanglish. A win is logged, and the TF-IDF model stays shipped until `predictor.py` can load the encoder offline. The container is not allowed to download weights at runtime.

## P3 findings (train_p3.py)

- **Validation is phrased differently from train.** Only 18.5% of validation tickets reuse a category-specific sentence seen in train. Train-to-validation macro-F1 is about 0.67, but training on train plus 4/5 of validation scores about 0.97 on the remaining 1/5. The spec's example tickets resemble neither set, so the hidden test set most likely has new phrasing. Every P3 choice is judged on train-to-validation, checked in the reverse direction (fit on validation, score train) to avoid tuning to one split.
- **Word bigrams memorise the train phrasing.** Dropping them for unigrams gave the first consistent gain in both directions.
- **The shipped baseline is the calibrated head, not the raw SVM.** Calibration's internal folds move the argmax: baseline-01 as served scores 0.6731, not 0.6640. P3-01 takes the label from the raw LinearSVC and only the confidence from the calibrated head: 0.6888 train-to-validation (+1.6 over served baseline) and 0.4876 validation-to-train (+3.5).
- **Did not help:** removing boilerplate sentences, negation marking, adding core-only copies, ComplementNB, logistic regression, soft-vote ensembles.
- **Secondary category:** predicting a second issue when P(none) < 0.65 lifts recall with almost no false positives. **Urgency:** baseline-01's head stayed best, so it is kept.

## Runs

| id | model_version | setup | train-CV macro-F1 (optimistic) | val macro-F1 | val acc | notes |
|---|---|---|---|---|---|---|
| baseline-01 | baseline-tfidf-22cd5e2d | word(1,2)+char_wb(2,5) TF-IDF + LinearSVC (C=0.5, balanced), sigmoid-calibrated | 0.9884 | 0.6640 | 0.6412 | urgent F1 0.892; secondary recall 0.550; ECE 0.063 |
| exp-02a | not-shipped | word(1,2)+char_wb(2,5) TF-IDF + LinearSVC (C=0.1, balanced) | n/a | 0.6366 | 0.6275 | below baseline; 17.6s; no CV |
| exp-02b | not-shipped | word(1,2)+char_wb(2,5) TF-IDF + LinearSVC (C=1.0, balanced) | n/a | 0.6694 | 0.6438 | beats baseline; 28.9s; no CV |
| exp-02c | not-shipped | word(1,2)+char_wb(2,5) TF-IDF + LinearSVC (C=2.0, balanced) | n/a | 0.6673 | 0.6400 | beats baseline; 33.3s; no CV |
| exp-02d | not-shipped | word(1,2)+char_wb(2,5) TF-IDF + LogReg (C=4.0, balanced) | n/a | 0.6151 | 0.5950 | below baseline; 22.6s; no CV |
| exp-03a | not-shipped | word(1,3)+char_wb(2,5) TF-IDF + LinearSVC (C=0.5, balanced) | n/a | 0.6554 | 0.6362 | below baseline; 12.6s; no CV |
| exp-03b | not-shipped | word(1,2)+char_wb(3,5) TF-IDF + LinearSVC (C=0.5, balanced) | n/a | 0.6676 | 0.6462 | beats baseline; 12.7s; no CV |
| p3-02a | not-shipped | word(1,1)+char_wb(2,5) TF-IDF + LinearSVC (C=0.5, balanced) | n/a | 0.6804 | 0.6613 | reverse val->train 0.4782 (baseline 0.4497) |
| p3-02b | not-shipped | p3-02a after removing sentences seen under 3+ categories | n/a | 0.6362 | 0.6188 | boilerplate removal hurts |
| p3-02c | not-shipped | p3-02a with negation marking (NOT_ prefix) | n/a | 0.6746 | 0.6550 | below p3-02a |
| p3-02d | not-shipped | word(1,1)+char_wb(3,6) + LinearSVC (C=0.5, balanced) | n/a | 0.6809 | 0.6625 | same as p3-02a |
| p3-02e | not-shipped | word(1,1) only + LinearSVC | n/a | 0.6487 | 0.6338 | char n-grams needed |
| p3-02f | not-shipped | char_wb(2,5) only + LinearSVC | n/a | 0.6622 | 0.6438 | word n-grams needed |
| p3-02g | not-shipped | word(1,2)+char_wb(2,5) + ComplementNB (alpha=0.3) | n/a | 0.5965 | 0.5925 | below baseline |
| p3-02h | not-shipped | soft vote: word LR + char LR + calibrated SVC | n/a | 0.6727 | 0.6462 | reverse 0.4571 |
| p3-02i | not-shipped | p3-02j plus boilerplate-free copies of each ticket | n/a | 0.6807 | n/a | reverse 0.5045; spec examples unchanged |
| p3-02j | not-shipped | word(1,1)+char_wb(2,5) + LinearSVC (C=0.5, no class weight) | n/a | 0.6888 | 0.6613 | reverse 0.4876; best in both directions |
| p3-01 | see model/labels.json | **shipped.** Label: word(1,1)+char_wb(2,5) + LinearSVC (C=0.5). Confidence: same features, sigmoid-calibrated (5-fold). Secondary: LR, predicted when P(none) < 0.65. Urgent: baseline-01 head | n/a | 0.6888 | 0.6613 | served baseline-01 scores 0.6731 (calibrated argmax); secondary recall 0.600 (was 0.550), FP 0.003; urgent F1 0.892; ECE 0.055 (was 0.063); confidence AUROC 0.771 (was 0.830) |
