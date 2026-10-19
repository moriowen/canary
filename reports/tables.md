## oh

| Model | AUROC k=0 | AUROC k=5 | AUROC k=10 | Within-task k=5 | Within-task k=10 | Compute saved @ 2% lost | @ 5% lost | @ 10% lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ensemble (GBM + GRU + text) | 0.500 | 0.585 | 0.597 | 0.511 | 0.498 | 5.6% | 10.1% | 18.1% |
| LightGBM (run features) | 0.500 | 0.584 | 0.582 | 0.520 | 0.497 | 6.3% | 11.4% | 20.2% |
| GRU monitor | 0.500 | 0.566 | 0.590 | 0.511 | 0.497 | 5.1% | 9.3% | 16.5% |
| LightGBM within-task ranker | 0.500 | 0.499 | 0.495 | 0.498 | 0.462 | 0.0% | 0.0% | 9.9% |
| LightGBM + issue features | 0.561 | 0.577 | 0.588 | 0.509 | 0.505 | 2.4% | 10.3% | 17.5% |
| Logistic regression | 0.500 | 0.550 | 0.567 | 0.506 | 0.501 | 0.0% | 6.1% | 15.7% |
| Issue text only | 0.563 | 0.563 | 0.563 | 0.500 | 0.500 | 2.8% | 6.1% | 11.9% |
| Oracle task difficulty | 0.974 | 0.974 | 0.974 | n/a | n/a | 49.8% | 53.1% | 60.2% |
| Rule: N consecutive errors | n/a | n/a | n/a | n/a | n/a | 1.0% | 2.2% | 5.5% |
| Rule: hard turn cap | n/a | n/a | n/a | n/a | n/a | 0.0% | 0.0% | 0.0% |

## oh_portable

| Model | AUROC k=0 | AUROC k=5 | AUROC k=10 | Within-task k=5 | Within-task k=10 | Compute saved @ 2% lost | @ 5% lost | @ 10% lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (run features) | 0.500 | 0.578 | 0.578 | 0.515 | 0.482 | 0.0% | 11.4% | 15.8% |
| LightGBM within-task ranker | 0.500 | 0.494 | 0.496 | 0.485 | 0.463 | 0.0% | 0.0% | 9.8% |
| LightGBM + issue features | 0.566 | 0.583 | 0.592 | 0.512 | 0.495 | 3.6% | 9.3% | 17.1% |
| Logistic regression | 0.500 | 0.549 | 0.569 | 0.500 | 0.497 | 0.0% | 6.1% | 15.7% |
| Issue text only | 0.563 | 0.563 | 0.563 | 0.500 | 0.500 | 2.8% | 6.1% | 11.9% |
| Oracle task difficulty | 0.974 | 0.974 | 0.974 | n/a | n/a | 49.8% | 53.1% | 60.2% |
| Rule: N consecutive errors | n/a | n/a | n/a | n/a | n/a | 1.0% | 2.2% | 5.5% |
| Rule: hard turn cap | n/a | n/a | n/a | n/a | n/a | 0.0% | 0.0% | 0.0% |

## oh_portable_aligned

| Model | AUROC k=0 | AUROC k=5 | AUROC k=10 | Within-task k=5 | Within-task k=10 | Compute saved @ 2% lost | @ 5% lost | @ 10% lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (run features) | 0.500 | 0.582 | 0.578 | 0.525 | 0.477 | 0.0% | 11.7% | 16.2% |
| LightGBM within-task ranker | 0.500 | 0.490 | 0.505 | 0.487 | 0.473 | 0.0% | 0.0% | 6.1% |
| LightGBM + issue features | 0.570 | 0.586 | 0.600 | 0.497 | 0.507 | 4.3% | 8.1% | 16.3% |
| Logistic regression | 0.500 | 0.558 | 0.579 | 0.518 | 0.505 | 0.0% | 6.7% | 16.8% |
| Issue text only | 0.562 | 0.562 | 0.562 | 0.500 | 0.500 | 1.5% | 4.3% | 9.4% |
| Oracle task difficulty | 0.974 | 0.974 | 0.974 | n/a | n/a | 49.8% | 53.1% | 60.2% |
| Rule: N consecutive errors | n/a | n/a | n/a | n/a | n/a | 1.0% | 2.2% | 5.5% |
| Rule: hard turn cap | n/a | n/a | n/a | n/a | n/a | 0.0% | 0.0% | 0.0% |

## oh_to_swe

| Model | AUROC k=0 | AUROC k=5 | AUROC k=10 | Within-task k=5 | Within-task k=10 | Compute saved @ 2% lost | @ 5% lost | @ 10% lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (run features) | 0.500 | 0.579 | 0.624 | 0.527 | 0.582 | 31.4% | 41.7% | 54.6% |
| LightGBM within-task ranker | 0.500 | 0.573 | 0.611 | 0.513 | 0.533 | 13.5% | 20.0% | 31.9% |
| LightGBM + issue features | 0.568 | 0.591 | 0.642 | 0.522 | 0.559 | 14.7% | 28.6% | 41.4% |
| Logistic regression | 0.500 | 0.550 | 0.617 | 0.502 | 0.586 | 23.5% | 30.2% | 42.6% |
| Issue text only | 0.586 | 0.592 | 0.587 | 0.500 | 0.500 | 3.3% | 7.5% | 16.5% |
| Oracle task difficulty | 0.942 | 0.940 | 0.934 | n/a | n/a | 75.7% | 78.8% | 82.6% |
| Rule: N consecutive errors | n/a | n/a | n/a | n/a | n/a | 19.6% | 25.8% | 36.1% |
| Rule: hard turn cap | n/a | n/a | n/a | n/a | n/a | 24.6% | 39.6% | 64.0% |

## oh_to_swe_aligned

| Model | AUROC k=0 | AUROC k=5 | AUROC k=10 | Within-task k=5 | Within-task k=10 | Compute saved @ 2% lost | @ 5% lost | @ 10% lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (run features) | 0.500 | 0.554 | 0.627 | 0.499 | 0.544 | 8.7% | 19.4% | 27.9% |
| LightGBM within-task ranker | 0.500 | 0.551 | 0.588 | 0.515 | 0.542 | 15.4% | 25.4% | 32.7% |
| LightGBM + issue features | 0.593 | 0.613 | 0.632 | 0.512 | 0.543 | 6.5% | 12.6% | 24.7% |
| Logistic regression | 0.500 | 0.576 | 0.639 | 0.499 | 0.566 | 26.1% | 34.7% | 43.2% |
| Issue text only | 0.597 | 0.596 | 0.589 | 0.500 | 0.500 | 5.9% | 13.1% | 18.3% |
| Oracle task difficulty | 0.942 | 0.940 | 0.934 | n/a | n/a | 75.7% | 78.8% | 82.6% |
| Rule: N consecutive errors | n/a | n/a | n/a | n/a | n/a | 19.6% | 25.8% | 36.1% |
| Rule: hard turn cap | n/a | n/a | n/a | n/a | n/a | 24.6% | 39.6% | 64.0% |

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

## swe_portable

| Model | AUROC k=0 | AUROC k=5 | AUROC k=10 | Within-task k=5 | Within-task k=10 | Compute saved @ 2% lost | @ 5% lost | @ 10% lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (run features) | 0.500 | 0.696 | 0.712 | 0.540 | 0.602 | 38.0% | 51.7% | 61.3% |
| LightGBM within-task ranker | 0.500 | 0.592 | 0.641 | 0.551 | 0.611 | 22.7% | 36.5% | 47.8% |
| LightGBM + issue features | 0.598 | 0.686 | 0.705 | 0.549 | 0.605 | 35.6% | 46.6% | 58.2% |
| Logistic regression | 0.500 | 0.664 | 0.700 | 0.530 | 0.592 | 31.3% | 50.8% | 63.2% |
| Issue text only | 0.579 | 0.582 | 0.572 | 0.500 | 0.500 | 3.6% | 4.3% | 17.6% |
| Oracle task difficulty | 0.942 | 0.940 | 0.934 | n/a | n/a | 75.7% | 78.8% | 82.6% |
| Rule: N consecutive errors | n/a | n/a | n/a | n/a | n/a | 19.6% | 25.8% | 36.1% |
| Rule: hard turn cap | n/a | n/a | n/a | n/a | n/a | 24.6% | 39.6% | 64.0% |

## swe_to_oh

| Model | AUROC k=0 | AUROC k=5 | AUROC k=10 | Within-task k=5 | Within-task k=10 | Compute saved @ 2% lost | @ 5% lost | @ 10% lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (run features) | 0.500 | 0.498 | 0.505 | 0.494 | 0.496 | 0.0% | 7.8% | 14.4% |
| LightGBM within-task ranker | 0.500 | 0.515 | 0.506 | 0.502 | 0.492 | 0.0% | 0.0% | 8.1% |
| LightGBM + issue features | 0.528 | 0.529 | 0.528 | 0.496 | 0.499 | 0.0% | 6.5% | 11.2% |
| Logistic regression | 0.500 | 0.525 | 0.539 | 0.492 | 0.498 | 0.0% | 0.0% | 5.6% |
| Issue text only | 0.535 | 0.535 | 0.535 | 0.500 | 0.500 | 2.2% | 5.9% | 13.2% |
| Oracle task difficulty | 0.974 | 0.974 | 0.974 | n/a | n/a | 49.8% | 53.1% | 60.2% |
| Rule: N consecutive errors | n/a | n/a | n/a | n/a | n/a | 1.0% | 2.2% | 5.5% |
| Rule: hard turn cap | n/a | n/a | n/a | n/a | n/a | 0.0% | 0.0% | 0.0% |

## swe_to_oh_aligned

| Model | AUROC k=0 | AUROC k=5 | AUROC k=10 | Within-task k=5 | Within-task k=10 | Compute saved @ 2% lost | @ 5% lost | @ 10% lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (run features) | 0.500 | 0.522 | 0.533 | 0.481 | 0.468 | 0.0% | 0.0% | 8.9% |
| LightGBM within-task ranker | 0.500 | 0.506 | 0.527 | 0.476 | 0.484 | 0.0% | 0.0% | 5.7% |
| LightGBM + issue features | 0.541 | 0.547 | 0.553 | 0.478 | 0.458 | 0.0% | 0.0% | 7.8% |
| Logistic regression | 0.500 | 0.511 | 0.528 | 0.515 | 0.518 | 0.0% | 6.6% | 12.1% |
| Issue text only | 0.509 | 0.509 | 0.509 | 0.500 | 0.500 | 0.9% | 3.4% | 7.4% |
| Oracle task difficulty | 0.974 | 0.974 | 0.974 | n/a | n/a | 49.8% | 53.1% | 60.2% |
| Rule: N consecutive errors | n/a | n/a | n/a | n/a | n/a | 1.0% | 2.2% | 5.5% |
| Rule: hard turn cap | n/a | n/a | n/a | n/a | n/a | 0.0% | 0.0% | 0.0% |
