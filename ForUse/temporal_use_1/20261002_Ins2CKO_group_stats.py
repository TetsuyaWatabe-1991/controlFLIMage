# %%
"""Group statistics of the 25-35 min delta spine volume for 20261002_Ins2CKO.py.

Uses the same tables and filters as 20261002_Ins2CKO.py (its ``cfg``, groups split by
uncaging protocol). Per uncaging power:

A. All groups against each other
   - one-way ANOVA, then Tukey HSD for every pair
   - Kruskal-Wallis, then Mann-Whitney U for every pair with Holm correction
B. Every group against one reference group (default: Culture media, 0.488 Hz uncaging)
   - Dunnett's test (one-way ANOVA framework)
   - Mann-Whitney U against the reference with Holm correction

Tables (CSV) and figures (PNG/PDF) are written to <session>/summary/group_stats/, with the
tiled swarm figure of the LTP analysis (panel_<signal>_swarmplot.png) for the same sets.
"""
import importlib.util
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FORUSE_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(FORUSE_DIR))
sys.path.insert(0, FORUSE_DIR)

from ltp_group_analysis import (  # noqa: E402
    PROTOCOL_SEP,
    build_condition_group_specs,
    plot_swarm_panels,
    prepare_ltp_group_tables,
    protocol_condition_order,
    signal_columns,
    two_line_title,
)

# %% settings
# Reference group of test B: "<condition>, <uncaging protocol>" as in the plots.
REFERENCE_CONDITION = "Culture media"
REFERENCE_PROTOCOL_HZ = 0.488  # the 0.5 Hz uncaging (30 pulses, 2048 ms interval)
ALPHA = 0.05


def load_ins2cko_cfg():
    """The cfg of 20261002_Ins2CKO.py (its analysis runs only under __main__)."""
    spec = importlib.util.spec_from_file_location("ins2cko", os.path.join(HERE, "20261002_Ins2CKO.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cfg


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def stars(p: float) -> str:
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < ALPHA else "n.s."


def describe(groups: dict) -> pd.DataFrame:
    return pd.DataFrame([{"group": g, "n": len(v), "mean": np.mean(v), "sd": np.std(v, ddof=1),
                          "sem": stats.sem(v), "median": np.median(v)} for g, v in groups.items()])


def all_pairs(groups: dict) -> tuple[dict, pd.DataFrame]:
    names, vals = list(groups), list(groups.values())
    anova = stats.f_oneway(*vals)
    kw = stats.kruskal(*vals)
    tukey = stats.tukey_hsd(*vals)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mw = stats.mannwhitneyu(vals[i], vals[j], alternative="two-sided")
            rows.append({"group_a": names[i], "group_b": names[j],
                         "mean_diff_b_minus_a": np.mean(vals[j]) - np.mean(vals[i]),
                         "tukey_p": float(tukey.pvalue[i, j]), "mannwhitney_p_raw": float(mw.pvalue)})
    pairs = pd.DataFrame(rows)
    pairs["mannwhitney_p_holm"] = holm(pairs["mannwhitney_p_raw"])
    omnibus = {"anova_F": float(anova.statistic), "anova_p": float(anova.pvalue),
               "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue)}
    return omnibus, pairs


def versus_reference(groups: dict, reference: str) -> pd.DataFrame:
    others = [g for g in groups if g != reference]
    # Dunnett p comes from a randomized numerical integration: fixed seed for reproducible p
    dun = stats.dunnett(*[groups[g] for g in others], control=groups[reference], random_state=0)
    rows = []
    for k, g in enumerate(others):
        mw = stats.mannwhitneyu(groups[g], groups[reference], alternative="two-sided")
        rows.append({"reference": reference, "group": g,
                     "mean_diff_group_minus_ref": np.mean(groups[g]) - np.mean(groups[reference]),
                     "dunnett_p": float(dun.pvalue[k]), "mannwhitney_p_raw": float(mw.pvalue)})
    out = pd.DataFrame(rows)
    out["mannwhitney_p_holm"] = holm(out["mannwhitney_p_raw"])
    return out


def plot_groups(groups: dict, comparisons: list, title: str, ylabel: str, path: str) -> None:
    """Points + mean +/- SEM per group; brackets with Tukey/Dunnett (left) and Mann-Whitney Holm (right) p."""
    import matplotlib.pyplot as plt

    names = list(groups)
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 1.6, 4.2), dpi=200)
    rng = np.random.default_rng(0)
    for i, g in enumerate(names):
        v = groups[g]
        ax.scatter(i + rng.uniform(-0.1, 0.1, len(v)), v, s=16, c="0.4", edgecolors="k", linewidths=0.3, zorder=2)
        ax.errorbar(i, np.mean(v), yerr=stats.sem(v), fmt="o", color="r", capsize=4, ms=5, zorder=3)
    lo = min(np.min(v) for v in groups.values())
    hi = max(np.max(v) for v in groups.values())
    span = max(hi - lo, 0.2)
    y = hi + 0.08 * span
    for a, b, p_param, p_np in comparisons:
        i, j = names.index(a), names.index(b)
        ax.plot([i, i, j, j], [y, y + 0.03 * span, y + 0.03 * span, y], c="k", lw=0.8)
        ax.text((i + j) / 2, y + 0.035 * span, f"{p_param:.3f} / {p_np:.3f}", ha="center", va="bottom", fontsize=6.5)
        y += 0.11 * span
    ax.axhline(0, color="0.6", lw=0.6, ls="--", zorder=0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{two_line_title(g)}\n(n={len(groups[g])})" for g in names], fontsize=7)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(lo - 0.1 * span, y + 0.05 * span)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def assumption_checks(groups: dict) -> tuple[pd.DataFrame, dict]:
    """Normality per group (Shapiro-Wilk) and equal variances (Brown-Forsythe / Levene on medians)."""
    normal = pd.DataFrame([{"group": g, "n": len(v), "shapiro_W": stats.shapiro(v).statistic,
                            "shapiro_p": stats.shapiro(v).pvalue, "skewness": stats.skew(v)}
                           for g, v in groups.items() if len(v) >= 3])
    lev = stats.levene(*groups.values(), center="median")
    return normal, {"brown_forsythe_W": float(lev.statistic), "brown_forsythe_p": float(lev.pvalue)}


def main(reference_hz: float = REFERENCE_PROTOCOL_HZ, exclude_above: float | None = None) -> str:
    """exclude_above: drop sets whose value is above it as outliers (results in their own folder)."""
    matplotlib.use("Agg")
    cfg = load_ins2cko_cfg()
    summary_df, _ts, save_folder = prepare_ltp_group_tables(cfg)
    _y_ts, y_col, ylabel = signal_columns(cfg.signal, cfg.ch_1or2)
    order = protocol_condition_order(cfg.condition_order, summary_df) if cfg.split_by_uncaging_protocol \
        else list(cfg.condition_order)
    out_dir = os.path.join(save_folder, "group_stats")
    report = []
    if exclude_above is not None:
        out_dir = os.path.join(save_folder, f"group_stats_exclude_above_{exclude_above:g}")
        drop = summary_df[y_col] > exclude_above
        removed = summary_df.loc[drop, ["group_set_id", "condition", y_col]]
        report += [f"Outliers: {int(drop.sum())} sets with {y_col} > {exclude_above:g} excluded",
                   removed.round(3).to_string(index=False), ""]
        summary_df = summary_df[~drop]
    os.makedirs(out_dir, exist_ok=True)
    # the tiled swarm figure of the LTP analysis (summary/panel_<signal>_swarmplot.png) for these sets
    lo, hi = summary_df[y_col].min(), summary_df[y_col].max()
    plot_swarm_panels(
        summary_df,
        [(c, h) for h, c in build_condition_group_specs(summary_df, order)],
        sorted(summary_df["uncaging_power_coherent_mW"].dropna().unique()),
        y_col=y_col, ylabel=ylabel, ylim=[lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)],
        save_path=os.path.join(out_dir, f"panel_{cfg.signal}_swarmplot.png"),
    )
    for power in sorted(summary_df["uncaging_power_coherent_mW"].dropna().unique()):
        d = summary_df[summary_df["uncaging_power_coherent_mW"] == power]
        groups = {g: d.loc[d["condition"] == g, y_col].dropna().to_numpy(float) for g in order}
        groups = {g: v for g, v in groups.items() if len(v) >= 2}
        tag = f"{power:g}mW"
        desc = describe(groups)
        desc.to_csv(os.path.join(out_dir, f"descriptive_{tag}.csv"), index=False)

        omnibus, pairs = all_pairs(groups)
        pairs.to_csv(os.path.join(out_dir, f"A_all_pairs_{tag}.csv"), index=False)
        pd.DataFrame([omnibus]).to_csv(os.path.join(out_dir, f"A_omnibus_{tag}.csv"), index=False)
        plot_groups(groups, [(r.group_a, r.group_b, r.tukey_p, r.mannwhitney_p_holm) for r in pairs.itertuples()],
                    f"{power:g} mW, all pairs\np: Tukey HSD / Mann-Whitney (Holm)\n"
                    f"ANOVA p={omnibus['anova_p']:.3f}, Kruskal-Wallis p={omnibus['kruskal_p']:.3f}",
                    ylabel, os.path.join(out_dir, f"A_all_pairs_{tag}.png"))

        reference = next((g for g in groups if g.startswith(REFERENCE_CONDITION + PROTOCOL_SEP)
                          and g.endswith(f" {reference_hz:.3g} Hz")), None)
        if reference is None:
            print(f"{tag}: reference group not found ({REFERENCE_CONDITION}, {reference_hz} Hz)")
            continue
        ref_tag = f"ref_{REFERENCE_CONDITION.replace(' ', '_')}_{reference_hz:g}Hz"
        vs = versus_reference(groups, reference)
        vs.to_csv(os.path.join(out_dir, f"B_vs_{ref_tag}_{tag}.csv"), index=False)
        normal, var_eq = assumption_checks(groups)
        normal.to_csv(os.path.join(out_dir, f"assumptions_normality_{tag}.csv"), index=False)
        plot_groups(groups, [(reference, r.group, r.dunnett_p, r.mannwhitney_p_holm) for r in vs.itertuples()],
                    f"{power:g} mW, vs {two_line_title(reference).replace(chr(10), ' ')}\n"
                    "p: Dunnett / Mann-Whitney (Holm)", ylabel, os.path.join(out_dir, f"B_vs_{ref_tag}_{tag}.png"))

        pd.set_option("display.width", 200)
        report += [f"===== {power:g} mW: {y_col} =====", desc.round(3).to_string(index=False), "",
                   "A. all pairs", f"  one-way ANOVA F={omnibus['anova_F']:.2f}, p={omnibus['anova_p']:.4f}; "
                   f"Kruskal-Wallis H={omnibus['kruskal_H']:.2f}, p={omnibus['kruskal_p']:.4f}",
                   pairs.assign(tukey=pairs.tukey_p.map(stars), mw_holm=pairs.mannwhitney_p_holm.map(stars))
                   .round(4).to_string(index=False), "",
                   f"B. vs reference: {reference}",
                   vs.assign(dunnett=vs.dunnett_p.map(stars), mw_holm=vs.mannwhitney_p_holm.map(stars))
                   .round(4).to_string(index=False), "",
                   "Assumptions of the parametric tests (ANOVA, Tukey, Dunnett: normal data, equal variances)",
                   normal.round(4).to_string(index=False),
                   f"  equal variances, Brown-Forsythe: W={var_eq['brown_forsythe_W']:.2f}, "
                   f"p={var_eq['brown_forsythe_p']:.4f}", ""]
    text = "\n".join(report)
    with open(os.path.join(out_dir, f"group_stats_report_{ref_tag}.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"Saved to {out_dir}")
    return out_dir


# %%
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-hz", type=float, default=REFERENCE_PROTOCOL_HZ,
                    help="uncaging rate (Hz, as in the group name) of the reference Culture media group")
    ap.add_argument("--exclude-above", type=float, default=None,
                    help="treat sets above this value as outliers (e.g. 2); results go to group_stats_exclude_above_<x>")
    args = ap.parse_args()
    main(args.ref_hz, args.exclude_above)
