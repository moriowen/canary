## swe

| Model | AUROC k=0 | AUROC k=5 | AUROC k=10 | Within-task k=5 | Within-task k=10 | Compute saved @ 2% lost | @ 5% lost | @ 10% lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (run features) | 0.515 | 0.699 | 0.717 | 0.542 | 0.603 | 40.9% | 51.4% | 64.2% |
| GRU monitor | 0.488 | 0.686 | 0.716 | 0.537 | 0.610 | 46.5% | 60.3% | 66.3% |
| LightGBM within-task ranker | 0.512 | 0.601 | 0.643 | 0.585 | 0.636 | 15.1% | 33.7% | 44.6% |
| LightGBM + issue features | 0.602 | 0.688 | 0.704 | 0.551 | 0.603 | 35.6% | 46.6% | 60.1% |
| Logistic regression | 0.515 | 0.667 | 0.703 | 0.535 | 0.596 | 34.8% | 50.6% | 63.0% |
| Issue text only | 0.595 | 0.596 | 0.575 | 0.500 | 0.500 | 3.1% | 3.1% | 3.1% |
| Oracle task difficulty | 0.942 | 0.940 | 0.934 | n/a | n/a | 75.7% | 78.8% | 82.6% |
| Rule: N consecutive errors | n/a | n/a | n/a | n/a | n/a | 19.6% | 25.8% | 36.1% |
| Rule: hard turn cap | n/a | n/a | n/a | n/a | n/a | 24.6% | 39.6% | 64.0% |
