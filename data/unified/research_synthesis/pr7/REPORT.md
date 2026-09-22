# PR #7 findings and closure decision

Close PR #7 without merging its research runner. Preserve its component experiments, negative results, scoring comparisons and original evidence. Its claim of a universal champion is unsupported, and its temporal comparison contains a confirmed indexing defect.

The source is pinned at `d936d0f13703fda7b88071ce02b72955bd9097be`. The original nine phase-2 evidence files are preserved byte-for-byte in `original/`. Their conclusions are historical claims, not current acceptance decisions. `trials.csv` retains all 13 candidates. The existing [research synthesis](../REPORT.md) preserves the inherited PR #3, #4 and #5 findings; those findings remain part of this record.

## Findings to retain

| Experiment | Recorded stable raw score | Finding and present interpretation |
| --- | ---: | --- |
| Frozen P3 control | 0.886470436 | The generalised two-component implementation reproduced the frozen score exactly. |
| C5 control | 0.879365878 | Per-event weighting improved the fixed pair. The reproduced control differs from its rounded reference by 1.23e-11. |
| E1: replace empirical with t3; global weight | 0.884504954 | A stronger standalone component did not automatically improve the blend against C5. Its paired interval is entirely worse. |
| E2: replace empirical with t3; shrunk event weights | 0.877461791 | The mean improved, but its paired interval crosses zero. The claimed temporal rejection requires a corrected comparison. |
| E3: retain both empirical components | 0.873009869 | Complementary components improved the recorded cross-fitted raw score. Retain addition versus substitution as a research hypothesis. |
| N1: add training-only stratum mean | 0.875385583 | Shrinkage toward a simple positional/start-status comparator improved raw error. Check ranking and team decisions separately. |
| N3: cap the stratum-mean component | 0.875383867 | A maximum share of 0.5 retained the raw improvement. This cap applied to this candidate; it is not a general guarantee. |
| N2: t3 plus stratum mean | 0.875048590 | Improved empirical estimates and simple shrinkage could coexist. It did not surpass E3 or E4. |
| E4: v4, original empirical, t3, stratum mean | 0.869656146 | This was the preferred candidate for the reconstructed fantasy rubrics. It was not a verified deployed champion. |
| E5: add empirical t4 instead of stratum mean | 0.872857869 | The best standalone empirical variant did not yield the best combined model. |
| G1: starter/bench blend weights | 0.878790913 | This implementation had an unresolved gain. It does not disprove all role-aware modelling. |
| G2: position-group blend weights | 0.879174714 | This implementation also had an unresolved gain. Keep event type as the supported blend axis in this search. |
| E6: all five components | 0.870825751 | More components did not improve on E4. Complexity needs evidence of additional benefit. |
| C9: multiplicatively calibrate E4 | 0.863402411 | This was the raw-score optimum, but its NCR reconstructed MAE worsened by 0.048388 against E4. |
| C10: widen the calibration grid | 0.863768965 | Extra calibration freedom did not improve the raw optimum or restore the NCR MAE loss. |

All 13 raw headline means recompute from the saved 21-fold arrays. Every saved incumbent array equals the C5 control array. This verifies arithmetic, not fresh model fitting or independence from previous selection.

The recorded standalone controls are v4 0.913307525, empirical 0.939031042, t3 0.904537865 and t4 0.900753906. Better standalone performance does not establish better ensemble performance. The t3 mechanisms were empirical-Bayes minute shrinkage, event-specific shrinkage, a separate minutes head and removal of the opponent multiplier. These remain candidate mechanisms, not a package approved for production.

E4 improved the recorded raw score in seven of eight tournament families against C5. British and Irish Lions regressed by 1.23%. Both hemisphere averages improved. Its final all-fold stratum-mean shares reached 0.45. All-fold fitted shares describe the final research fit; they are not independent validation.

C9 illustrates a conflict between raw-event loss and fantasy decisions. Lower Poisson deviance does not guarantee lower fantasy MAE or better teams. Keep raw accuracy, fantasy accuracy, ranking and legal-team points as separate measures. Do not select a different winning cell for each competition.

The reusable evaluation ideas are component ablations, per-target simplex weights, nested selection of shrinkage strength, exact control reconstruction, paired fold uncertainty and explicit hemisphere/tournament/extended-event safeguards. Reusing cached component forecasts can reduce repeated fitting, but requires aligned fixture/player/team keys and source/configuration hashes. A cached forecast does not make a repeatedly used outcome independent. Preserve both the raw-score optimum and the fantasy-preferred candidate instead of deleting the inconvenient comparison.

## Claims that must not be carried forward

### The temporal improvement is invalid

`build_incumbent` creates C5 indices in its two-component pair grid. `temporal_check` then uses those integer indices in the candidate's different component table. For example, index 100 represents `[0.5, 0.5]` for C5, but `[0, 0.25, 0.25, 0.5]` for E4. An index is not a transferable weight vector.

E4's recorded temporal calculation is `0.880270661279173 - 0.9154705563515934 = -0.0351998950724206`. The second value is not C5. This defect affects E1, E2, E3, N1, N2, N3, E4, E5 and E6. E1 and E2 also change the empirical component, although their grid dimensions match.

G1, G2, C9 and C10 use the correct saved C5 scalar scores. However, C5's weights are leave-one-fold-out, rather than fitted only on the first ten folds. Therefore, none of the 13 saved temporal comparisons establishes the advertised fully chronological candidate-versus-C5 result.

C5's saved final-eleven-fold mean is 0.881621632556016. Subtracting it from E4's candidate score gives about -0.001351, but that is not a corrected temporal result. A valid comparison must refit both weight rules on the first ten folds and evaluate the same final eleven. The original temporal deltas, intervals and acceptance decisions that depend on them remain invalid pending that work. The primary cross-fitted raw means use the correct scalar C5 arrays and are unaffected by this specific indexing defect.

### Pooled scoring is not universal superiority

The board averages 98 historical slates under reconstructed scoring rules. It does not compare the complete deployed fantasy pipelines. It omits the full Six Nations champion and contains no price-constrained official-team replay.

E4 loses NCR-rubric MAE to empirical in the Nations Championship family: 6.3091 versus 6.2925. It also loses that measure in the Lions and Pacific Nations Cup families. Its pooled capture intervals against empirical cross zero under both rubrics.

The PR body claims every listed MAE and Spearman gap excludes zero. Its own board contradicts that statement. E4 versus C5 NCR Spearman has interval `[-0.00008, 0.00376]`. E4 versus t3 NCR MAE has interval `[-0.04673, 0.00402]`.

The E1 ledger says its positive interval includes zero. The interval `[0.000397, 0.009891]` actually excludes zero in the wrong direction for improvement. The rejection remains supported, but that sentence is incorrect.

## Integration decision

Preserve the evidence and hypotheses. Do not merge the 526-file experimental branch, its replacement empirical implementation or its faulty temporal runner. The original commit remains accessible through the archive tag and PR branch. The original phase-2 evidence and this corrected interpretation are stored on main.

Any renewed E4 work needs a corrected chronological baseline, a prediction-time interface, current leakage corrections, identity checks and the complete evaluation protocol. Repeated historical results cannot provide prospective confirmation. The existing NCR GW4–7 shadow protocol remains unchanged.
