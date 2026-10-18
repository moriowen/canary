## oh

| Model | AUROC k=0 | AUROC k=5 | AUROC k=10 | Within-task k=5 | Within-task k=10 | Compute saved @ 2% lost | @ 5% lost | @ 10% lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (all features) | 0.500 | 0.584 | 0.582 | 0.520 | 0.497 | 6.3% | 11.4% | 20.2% |
| Logistic regression | 0.500 | 0.550 | 0.567 | 0.506 | 0.501 | 0.0% | 6.1% | 15.7% |
| Issue text only | 0.563 | 0.563 | 0.563 | 0.500 | 0.500 | 2.8% | 6.1% | 11.9% |
| Oracle task difficulty | 0.974 | 0.974 | 0.974 | n/a | n/a | 49.8% | 53.1% | 60.2% |
| Rule: N consecutive errors | n/a | n/a | n/a | n/a | n/a | 1.0% | 2.2% | 5.5% |
| Rule: hard turn cap | n/a | n/a | n/a | n/a | n/a | 0.0% | 0.0% | 0.0% |

## swe

| Model | AUROC k=0 | AUROC k=5 | AUROC k=10 | Within-task k=5 | Within-task k=10 | Compute saved @ 2% lost | @ 5% lost | @ 10% lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (all features) | 0.515 | 0.699 | 0.717 | 0.542 | 0.603 | 40.9% | 51.4% | 64.2% |
| Logistic regression | 0.515 | 0.667 | 0.703 | 0.535 | 0.596 | 34.8% | 50.6% | 63.0% |
| Issue text only | 0.595 | 0.596 | 0.575 | 0.500 | 0.500 | 3.1% | 3.1% | 3.1% |
| Oracle task difficulty | 0.942 | 0.940 | 0.934 | n/a | n/a | 75.7% | 78.8% | 82.6% |
| Rule: N consecutive errors | n/a | n/a | n/a | n/a | n/a | 19.6% | 25.8% | 36.1% |
| Rule: hard turn cap | n/a | n/a | n/a | n/a | n/a | 24.6% | 39.6% | 64.0% |

## swe_to_oh

| Model | AUROC k=0 | AUROC k=5 | AUROC k=10 | Within-task k=5 | Within-task k=10 | Compute saved @ 2% lost | @ 5% lost | @ 10% lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (all features) | 0.500 | 0.498 | 0.505 | 0.494 | 0.496 | 0.0% | 7.8% | 14.4% |
| Logistic regression | 0.500 | 0.525 | 0.539 | 0.492 | 0.498 | 0.0% | 0.0% | 5.6% |
| Issue text only | 0.535 | 0.535 | 0.535 | 0.500 | 0.500 | 2.2% | 5.9% | 13.2% |
| Oracle task difficulty | 0.974 | 0.974 | 0.974 | n/a | n/a | 49.8% | 53.1% | 60.2% |
| Rule: N consecutive errors | n/a | n/a | n/a | n/a | n/a | 1.0% | 2.2% | 5.5% |
| Rule: hard turn cap | n/a | n/a | n/a | n/a | n/a | 0.0% | 0.0% | 0.0% |
