# Experiments

## Goal

The API decides which prior exams should be shown while a radiologist reads the current exam. I optimized for returning one prediction per prior quickly and consistently, then used case-level validation to estimate whether features generalize to unseen patients/cases.

## Data and Validation

- Public labeled split: 996 cases, 27,614 previous examinations.
- Final artifact is trained on `relevant_priors_public.json` because no other labeled training split is provided.
- Model selection used grouped validation by `case_id` through `analyze_oof_errors.py`, so priors from the same case do not leak between train and validation folds.
- `evaluate_public.py` is now a public labeled-eval command with explicit `--payload` and `--model` arguments; it is not treated as the primary estimate of hidden performance.

Reproduction commands:

```bash
python -B app/train.py --training-data relevant_priors_public.json --output-model models/model.joblib --threshold 0.65
python -B evaluate_public.py --payload relevant_priors_public.json --model models/model.joblib
python -B analyze_oof_errors.py --folds 5 --threshold 0.65
python -B -m unittest discover -s tests
```

## Ablation Summary

| Version | Main features | Validation method | Accuracy |
|---|---|---:|---:|
| Rule baseline | exact description, coarse modality, 5-year age rule | full public eval | 0.669697 |
| Random forest v1 | date buckets, modality/body region, token overlap, laterality, contrast | grouped held-out estimate | about 0.9295 |
| Random forest v1 | same as above | full public eval | 0.947563 |
| Random forest v2 | v1 plus clinical text families and related-family features | grouped OOF, threshold 0.65 | 0.944774 |
| Random forest v2 | same as above | full public eval | 0.957051 |

The main improvement came from handling textually different but clinically related studies: mammography and breast ultrasound, cardiac stress/echo/coronary CT, PET/CT oncology follow-up, chest/ribs/thoracic spine relationships, and renal/abdomen/pelvis relationships.

## Current Model

- `RandomForestClassifier`
- 300 trees
- `min_samples_leaf=8`
- `class_weight=balanced_subsample`
- `max_features=sqrt`
- `n_jobs=1` to avoid process pressure in small hosted containers
- decision threshold `0.65`

Features include:

- days between studies and date buckets
- exact normalized description match
- modality and body-region match
- token overlap count and Jaccard score
- laterality and contrast agreement
- clinical-family flags for current/prior/shared families
- related-family indicator for clinically connected text families

## Error Slices

Out-of-fold validation after clinical-family features:

- accuracy: 0.944774
- precision: 0.895761
- recall: 0.868890
- F1: 0.882121
- false positives: 664
- false negatives: 861

Remaining false positives cluster around:

- breast studies with same body region but mismatched side or procedure purpose, such as diagnostic right mammogram vs left screening/ultrasound
- older abdomen/pelvis studies where coarse body region matches but clinical question may differ
- older chest/cardiac or spine-related priors where anatomical overlap is broad

Remaining false negatives cluster around:

- renal/bladder ultrasound vs older abdomen/pelvis CT or urogram
- cardiac/coronary/echo/chest studies with very different wording
- chest radiograph vs CT thoracic spine/ribs/chest studies
- broad `other` descriptions that hide useful clinical relationships

## Radiologist Workflow Considerations

The most harmful failure is a false negative: a relevant prior is hidden, and the radiologist loses comparison context for interval change. False positives are less harmful but add cognitive load by cluttering the comparison list. For that reason, I chose a threshold that balances accuracy and recall rather than maximizing precision only.

In a production workflow, I would not only return a boolean. I would rank priors by relevance confidence and group them by clinical family, recency, and modality so the most useful comparisons appear first. I would validate usefulness with radiologists by measuring whether the surfaced priors reduce missed comparisons and whether the extra priors slow reads or create alert fatigue.

## What Worked

- Bulk inference over all priors in the request.
- Compact engineered features instead of external services or one call per prior.
- Grouped validation by case to reduce leakage.
- Clinical text-family features for common radiology wording shifts.
- Pinned sklearn version and model metadata to avoid pickle incompatibility.

## What Failed

- The first rule-based model over-predicted relevance and performed poorly.
- Full-public accuracy was too optimistic when treated as the headline validation metric.
- High-cardinality exact description categories made training heavier without improving the private score.
- Unlimited sklearn workers caused local resource pressure, so the final model forces single-worker inference/training.

## Next Improvements

- Add more radiology-specific procedure families for broad `other` studies.
- Split mammography features into screening, diagnostic, biopsy, localization, and ultrasound-guided procedures.
- Add a calibrated probability model or small ensemble after feature work, not before.
- Evaluate with a radiologist-facing ranking metric, not just binary accuracy, if the product were deployed beyond this challenge.
