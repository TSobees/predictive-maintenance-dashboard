# Synthetic demonstration evaluation

All results are generated from fictional data. This is not a report of field accuracy.

| Horizon | Test windows | Positive windows | Recall | Precision | Flagged | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 days | 423 | 38 | 57.9% | 15.9% | 32.6% | 0.736 | 0.170 |
| 14 days | 376 | 67 | 61.2% | 28.3% | 38.6% | 0.731 | 0.307 |
| 30 days | 235 | 91 | 82.4% | 52.4% | 60.9% | 0.682 | 0.499 |

One fixed configuration per horizon; no test-based model search. Thresholds are set on separate validation dates, targeting at least 80% recall there. Test recall can differ. Calibration uses an earlier separate period. Outcome windows are purged at boundaries.

These windows overlap and are not counts of unique physical failures. The simulation itself defines condition/failure associations. No unseen-equipment transfer result, real alert workload reduction or field validation is claimed. See MODEL_CARD.md and the machine-readable evaluation under maintenance_app/data/forecast_models/.
