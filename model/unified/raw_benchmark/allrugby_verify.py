"""End-to-end verification: the real event_metrics pipeline must agree with the
precomputed loss table, and extended events must be untouched by the candidate."""
import json, numpy as np, pandas as pd
import model.unified.raw_benchmark.allrugby as A
from model.unified.raw_benchmark.allrugby import WORK
from model.unified.raw_benchmark.allrugby_blend import BlendRule
from model.unified.raw_benchmark.allrugby_cache import build_cache
from model.unified.raw_benchmark.allrugby_table import build_table, GRID
from model.unified.raw_benchmark.config import EXTENDED_EVENTS
from model.unified.raw_benchmark.folds import build_folds
from model.unified.raw_benchmark.metrics import prediction_means
import model.unified.raw_benchmark.allrugby_fit as F

table = build_table(build_cache())
weights = json.load(open(WORK / "c5_weights.json"))
cross = weights["cross_fitted_by_fold"]

store = A._store()
rows, ext_rows = [], []
for fold in build_folds(store):
    evaluation, naive = A.evaluation_context(store, fold)
    emp = A.read_predictions(A._component_path(fold.label, "empirical"))
    v4 = A.read_predictions(A._component_path(fold.label, "v4"))
    frozen = BlendRule("p3_frozen", {}, 0.5).apply(emp, v4)
    cand = BlendRule("C5", cross[fold.label], 0.5).apply(emp, v4)
    ev_f, _ = A.score_predictions(evaluation, frozen, naive, engine="p3_frozen", fold=fold)
    ev_c, _ = A.score_predictions(evaluation, cand, naive, engine="C5", fold=fold)
    def stable(df):
        s = df[df.cohort.eq("all") & df.tier.isin(["stable", "minutes"])]
        return float(s.relative_loss.mean())
    rows.append({"fold": fold.label, "pipeline_frozen": stable(ev_f), "pipeline_C5": stable(ev_c)})
    # extended events must be bit-identical: they keep the frozen 0.5 weight
    for event in EXTENDED_EVENTS:
        a, b = prediction_means(frozen, event), prediction_means(cand, event)
        both = np.isfinite(a) & np.isfinite(b)
        ext_rows.append({"fold": fold.label, "event": event,
                         "max_abs_diff": float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0})
    e_f = ev_f[ev_f.tier.eq("extended") & ev_f.cohort.eq("all")].set_index("target")["loss"]
    e_c = ev_c[ev_c.tier.eq("extended") & ev_c.cohort.eq("all")].set_index("target")["loss"]
    for t in e_f.index:
        ext_rows.append({"fold": fold.label, "event": f"loss::{t}",
                         "max_abs_diff": float(abs(e_f[t] - e_c[t]))})
    print(f"[{fold.label}] verified", flush=True)

frame = pd.DataFrame(rows)
per_fold_table_frozen = table.stable("all", np.full(len(table.targets), table.index_of(0.5), int))
idx = [np.array([table.index_of(cross[l][t]) for t in table.targets]) for l in table.folds]
per_fold_table_c5 = F.all_fold_scores(table, idx)
frame["table_frozen"] = per_fold_table_frozen
frame["table_C5"] = per_fold_table_c5
frame.to_csv(WORK / "verification_pipeline_vs_table.csv", index=False)
ext = pd.DataFrame(ext_rows)
print()
print("pipeline vs table, max abs per-fold difference:")
print("  frozen:", float((frame.pipeline_frozen - frame.table_frozen).abs().max()))
print("  C5    :", float((frame.pipeline_C5 - frame.table_C5).abs().max()))
print("pipeline stable frozen: %.10f  C5: %.10f" % (frame.pipeline_frozen.mean(), frame.pipeline_C5.mean()))
print("table    stable frozen: %.10f  C5: %.10f" % (frame.table_frozen.mean(), frame.table_C5.mean()))
print()
print("EXTENDED EVENTS max abs difference across all folds/events:", float(ext.max_abs_diff.max()))
print("criterion (e) extended-event loss regression >5%:",
      "NONE - extended predictions are bit-identical" if ext.max_abs_diff.max() == 0.0 else "CHECK")
