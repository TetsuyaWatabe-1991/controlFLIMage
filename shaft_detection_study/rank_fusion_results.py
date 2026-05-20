# -*- coding: utf-8 -*-
"""Rank fusion sweep CSV (quick helper)."""
import csv
import sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results/AP5_pos6_256_4x_001/sweep_summary.csv"
rows = list(csv.DictReader(p.open(encoding="utf-8")))


def f(r, k):
    return float(r[k])


for r in rows:
    r["_score"] = f(r, "fused_skel2d_len_0_3") + 0.5 * f(r, "delta_skel2d_0_3")

rows.sort(key=lambda r: r["_score"], reverse=True)
print("Top 8 (fused skel2d@0.3 + 0.5*delta):")
for r in rows[:8]:
    print(
        f"  {r['combo_id']:32s}  skel2d={f(r,'fused_skel2d_len_0_3'):.0f}  "
        f"delta={f(r,'delta_skel2d_0_3'):+.0f}  frac_mip={f(r,'fused_frac_mip_gt_0_2'):.3f}  "
        f"n_comp={r['fused_n_components_mip_0_2']}"
    )

print("\nTop 5 delta_skel2d_0_3:")
for r in sorted(rows, key=lambda r: f(r, "delta_skel2d_0_3"), reverse=True)[:5]:
    print(
        f"  {r['combo_id']:32s}  delta={f(r,'delta_skel2d_0_3'):+.0f}  "
        f"skel2d={f(r,'fused_skel2d_len_0_3'):.0f}"
    )

best = rows[0]
print("\nRecommended:")
print(f"  combo_id={best['combo_id']}")
print(
    f"  IMAGE_FUSION_PERCENTILE={best['image_fusion_percentile']}  "
    f"IMAGE_FUSION_WEIGHT={best['image_fusion_weight']}  "
    f"DENDRITE_CLOSING_ITERATIONS={best['dendrite_closing_iterations']}"
)
print(f"\n  compare PNG: {p.parent / ('compare_' + best['combo_id'] + '.png')}")
