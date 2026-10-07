# SustainX.Q Parkinson Voice Research Model Card

## Intended Use

This is an exploratory research prototype for comparing classical models with a small hybrid quantum-classical model on the UCI Parkinsons engineered voice-feature dataset. It is not intended for clinical decision-making, patient triage, diagnosis, treatment, or exclusion of disease.

The microphone workflow performs basic signal quality checks and supports a descriptive, session-only personal acoustic baseline. Microphone recordings are not scored by the UCI models because feature compatibility has not been demonstrated.

## Data

- Source: [UCI Parkinsons](https://archive.ics.uci.edu/dataset/174/parkinsons), DOI 10.24432/C59C74, CC BY 4.0.
- Validated downloaded table: 195 recordings, 22 engineered numeric voice features, 32 subject IDs, zero missing cells.
- Recording labels: 48 healthy, 147 Parkinson's. Person labels: 8 healthy, 24 Parkinson's.
- UCI descriptive text states 31 people (23 with Parkinson's), while the downloaded table contains 32 person IDs (24 with Parkinson's). The source-file counts are reported without alteration.
- Dataset checksum is recorded in the generated `artifacts/benchmark.json`.

## Models

- Logistic Regression: median imputation, standardization, class balancing.
- RBF SVM: median imputation, standardization, class balancing. Decision scores are monotonically rescaled to [0, 1] only for common metric interfaces; no probability calibration is performed.
- Random Forest: median imputation, 250 trees, minimum leaf size 2, class balancing.
- Quantum: fold-local median imputation and standardization, PCA to the selected qubit budget, fold-fitted min-max angle scaling to [-pi, pi], Y-axis angle embedding, two strongly entangling variational layers, Pauli-Z expectation readout, Adam optimizer, PennyLane `default.qubit` statevector simulator.

## Evaluation

Five-fold stratified group cross-validation uses the parsed subject ID. No subject appears in both training and test portions of a fold. Metrics are calculated after averaging each person's held-out recording scores. The report includes accuracy, sensitivity, specificity, precision, F1, ROC-AUC, mean training time, mean inference time, and fold mean/standard deviation.

The quantum utility label compares paired subject-level out-of-fold ROC-AUC with the RBF SVM, using a pre-set 0.05 practical margin and 2,000 class-stratified subject bootstrap resamples. The interval does not include all model-training uncertainty and is not external validation.

Classical feature importance uses held-out-fold permutation importance. It describes predictive association in this dataset and is not a causal or biological explanation.

## Results Caveats

Only 32 independent subjects are available, with just 8 healthy participants. Fold metrics are consequently unstable; the high fold standard deviations and wide bootstrap interval should be presented with the results. The data are not representative evidence of clinical performance or generalization to new populations.

Model bundles refit on the full dataset are saved for reproducibility only. Their predictions are not independent test results. Scores and threshold signals are not medically calibrated confidence. No clinically validated risk threshold is provided.

## Known Limitations

- No compatible labeled raw-audio training/evaluation set is included.
- No clinical validation or external cohort evaluation.
- No medically calibrated probabilities or diagnostic thresholds.
- Small classical-simulated quantum experiment; no quantum speedup or advantage is demonstrated.
- Quantum noise experiment is not included.
- Baseline data are temporary session state with explicit JSON export/import; not persistent browser storage.
- Alzheimer's/MCI, depression, skin screening, and other unsupported condition models are not included.
