# Experiments

- Baseline: rule-based relevance scoring by exact normalized description match, then same coarse modality within five years.
  - Public local accuracy: `0.669697`.
  - This over-predicted relevance and performed worse than an all-false baseline on accuracy because only about 24% of public priors are relevant.

- Random forest with engineered features.
  - Features: date distance, date buckets, exact description match, modality match, body-region match, token overlap, token Jaccard, laterality match, contrast match, current/prior modality, and current/prior body region.
  - Model: `RandomForestClassifier`, 300 trees, `min_samples_leaf=8`, `class_weight=balanced_subsample`, threshold `0.56`.
  - Public local accuracy: `0.947563`.
  - Held-out-by-case validation during development was approximately `0.9295` accuracy with F1 around `0.850`.

- What worked:
  - Batch inference over all priors in the request.
  - Compact clinical-ish text features rather than raw exact study-description categories.
  - Raising the probability threshold from `0.50` to `0.56` reduced false positives and improved accuracy.
  - Pinning `scikit-learn==1.7.2` and storing model metadata avoided sklearn pickle version mismatches.

- What failed:
  - The initial broad rule-based model missed too many relevant studies and produced many false positives.
  - Using unlimited sklearn workers caused local process pressure, so the deployed model now forces `n_jobs=1`.
  - Exact high-cardinality description categories made training unnecessarily heavy and less deployment-friendly.

- Error patterns:
  - False positives often involve breast/mammography studies with similar region but different procedure, side, or clinical purpose.
  - Older abdomen/pelvis and chest/cardiac studies can still be over-linked when the coarse body region matches.
  - False negatives often involve clinically related but textually dissimilar studies, such as mammography vs breast ultrasound or cardiac stress/CT coronary/echo relationships.

- Next improvements:
  - Add stronger mammography-specific features for screening vs diagnostic vs biopsy vs ultrasound and left/right/bilateral laterality.
  - Add a small domain mapping for clinically related cross-modality pairs, especially cardiac and breast imaging.
  - Tune threshold and features only on held-out-by-case validation to avoid overfitting public labels.
  - Consider a calibrated gradient boosting model or an ensemble of random forest plus a linear text model if private-split feedback remains below target.
