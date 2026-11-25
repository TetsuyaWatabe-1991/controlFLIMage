# %%
"""Within-solvent tests of the 25-35 min delta spine volume.

Water, AP5, and CPP were dissolved in water, so they are compared only with
each other. DMSO, Gavestinel, and MK801 were dissolved in DMSO and are
compared only with each other. The two solvent families are not tested
against one another.

Several groups are right-skewed (Shapiro-Wilk), so each family uses a
Kruskal-Wallis test and two-sided Dunn post hoc tests. Holm adjustment is
applied inside the family only. Mann-Whitney U with the same Holm adjustment
is written to the table as a second pairwise summary; the figure shows Dunn.

Reads the filtered summary already written by ``20261119_DrugNMDARi.py``.
Figures and tables go next to that summary, under ``solvent_group_stats/``.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FORUSE_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(FORUSE_DIR))
sys.path.insert(0, FORUSE_DIR)

from ltp_group_analysis import (  # noqa: E402
    PROTOCOL_SEP,
    signal_columns,
    summary_folder_name,
)

ALPHA = 0.05
# Vehicle first, then the two antagonists that share that vehicle.
SOLVENT_FAMILIES: list[tuple[str, list[str]]] = [
    ("Water", ["Water", "AP5", "CPP"]),
    ("DMSO", ["DMSO", "Gavestinel", "MK801"]),
]
DRUG_COLOR = {
    "Water": "#000000",
    "AP5": "#0072B2",
    "CPP": "#009E73",
    "DMSO": "#000000",
    "Gavestinel": "#CC79A7",
    "MK801": "#D55E00",
}
MEAN_COLOR = "#B2182B"


def load_session_module():
    """Import ``20261119_DrugNMDARi.py`` without running its analysis."""
    path = os.path.join(HERE, "20261119_DrugNMDARi.py")
    spec = importlib.util.spec_from_file_location("drug_nmdari", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses looks the class up in sys.modules while the body executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    n = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def stars(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < ALPHA:
        return "*"
    return "n.s."


def drug_name(condition: str) -> str:
    text = str(condition)
    return text.split(PROTOCOL_SEP, 1)[0] if PROTOCOL_SEP in text else text


def dunn_z_p(groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """All-pairs two-sided Dunn tests on one pooled ranking.

    Tie correction matches the form used by R's dunn.test: the null variance
    ``N(N+1)/12`` is multiplied by ``1 - sum(t^3 - t) / (N^3 - N)``.
    """
    names = list(groups)
    samples = [np.asarray(groups[name], dtype=float) for name in names]
    pooled = np.concatenate(samples)
    n_total = len(pooled)
    ranks = stats.rankdata(pooled, method="average")
    _values, tie_counts = np.unique(pooled, return_counts=True)
    tie_correction = 1.0 - float(np.sum(tie_counts**3 - tie_counts)) / (n_total**3 - n_total)
    sigma2 = n_total * (n_total + 1) / 12.0
    mean_rank = []
    sizes = []
    start = 0
    for sample in samples:
        mean_rank.append(float(ranks[start:start + len(sample)].mean()))
        sizes.append(len(sample))
        start += len(sample)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            se = np.sqrt(sigma2 * tie_correction * (1.0 / sizes[i] + 1.0 / sizes[j]))
            z = abs(mean_rank[i] - mean_rank[j]) / se
            rows.append({
                "group_a": names[i],
                "group_b": names[j],
                "dunn_z": float(z),
                "dunn_p_raw": float(2.0 * stats.norm.sf(z)),
            })
    return pd.DataFrame(rows)


def family_tests(groups: dict[str, np.ndarray], family: str, protocol: str, power: float) -> tuple[dict, pd.DataFrame]:
    """Kruskal-Wallis plus Dunn and Mann-Whitney pairs, Holm-adjusted within the family."""
    names = list(groups)
    samples = [groups[name] for name in names]
    kw = stats.kruskal(*samples)
    pairs = dunn_z_p(groups)
    mw_p = []
    mean_diff = []
    median_diff = []
    for row in pairs.itertuples(index=False):
        a = groups[row.group_a]
        b = groups[row.group_b]
        mw_p.append(float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue))
        mean_diff.append(float(np.mean(b) - np.mean(a)))
        median_diff.append(float(np.median(b) - np.median(a)))
    pairs["mannwhitney_p_raw"] = mw_p
    pairs["mean_diff_b_minus_a"] = mean_diff
    pairs["median_diff_b_minus_a"] = median_diff
    pairs["dunn_p_holm"] = holm(pairs["dunn_p_raw"].to_numpy())
    pairs["mannwhitney_p_holm"] = holm(pairs["mannwhitney_p_raw"].to_numpy())
    pairs.insert(0, "power_mW", power)
    pairs.insert(0, "uncaging_protocol", protocol)
    pairs.insert(0, "solvent", family)
    omnibus = {
        "solvent": family,
        "uncaging_protocol": protocol,
        "power_mW": power,
        "kruskal_H": float(kw.statistic),
        "kruskal_df": len(names) - 1,
        "kruskal_p": float(kw.pvalue),
        "n_groups": len(names),
        "n_spines": int(sum(len(v) for v in samples)),
    }
    return omnibus, pairs


def describe(groups: dict[str, np.ndarray], family: str, protocol: str, power: float) -> pd.DataFrame:
    rows = []
    for name, values in groups.items():
        rows.append({
            "solvent": family,
            "uncaging_protocol": protocol,
            "power_mW": power,
            "drug": name,
            "n": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "sem": float(stats.sem(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "shapiro_W": float(stats.shapiro(values).statistic) if len(values) >= 3 else np.nan,
            "shapiro_p": float(stats.shapiro(values).pvalue) if len(values) >= 3 else np.nan,
        })
    return pd.DataFrame(rows)


def _draw_points(ax, order: list[str], groups: dict[str, np.ndarray]) -> None:
    frame = pd.DataFrame({
        "drug": np.concatenate([[name] * len(groups[name]) for name in order]),
        "value": np.concatenate([groups[name] for name in order]),
    })
    sns.swarmplot(
        data=frame,
        x="drug",
        y="value",
        order=order,
        hue="drug",
        hue_order=order,
        palette=[DRUG_COLOR[name] for name in order],
        size=3.4,
        linewidth=0.35,
        edgecolor="white",
        ax=ax,
        legend=False,
        zorder=2,
    )
    for i, name in enumerate(order):
        values = groups[name]
        mean = float(np.mean(values))
        sem = float(stats.sem(values))
        ax.plot([i - 0.22, i + 0.22], [mean, mean], color=MEAN_COLOR, lw=1.6, zorder=4, solid_capstyle="butt")
        ax.errorbar(
            i, mean, yerr=sem, fmt="none", ecolor=MEAN_COLOR,
            elinewidth=1.05, capsize=3.2, capthick=1.05, zorder=4,
        )


def _draw_brackets(ax, order: list[str], pairs: pd.DataFrame, y_max: float, span: float) -> float:
    """Stack Dunn brackets. Shorter spans sit lower. Returns the top of the last label."""
    ranked = pairs.copy()
    ranked["span_groups"] = [
        abs(order.index(a) - order.index(b))
        for a, b in zip(ranked["group_a"], ranked["group_b"])
    ]
    ranked = ranked.sort_values(["span_groups", "group_a", "group_b"])
    y = y_max + 0.07 * span
    step = 0.14 * span
    tick = 0.022 * span
    top = y
    for row in ranked.itertuples(index=False):
        i = order.index(row.group_a)
        j = order.index(row.group_b)
        left, right = min(i, j), max(i, j)
        ax.plot(
            [left, left, right, right],
            [y, y + tick, y + tick, y],
            color="black", lw=0.8, clip_on=False, zorder=5,
        )
        label = stars(float(row.dunn_p_holm))
        ax.text(
            (left + right) / 2.0,
            y + tick + 0.008 * span,
            label,
            ha="center",
            va="bottom",
            fontsize=8 if label != "n.s." else 7,
            fontstyle="normal" if label != "n.s." else "italic",
            color="black",
            clip_on=False,
            zorder=5,
        )
        top = y + tick + 0.07 * span
        y += step
    return top


def plot_families(
    family_groups: list[tuple[str, list[str], dict[str, np.ndarray], pd.DataFrame, dict]],
    protocol: str,
    power: float,
    window_min: tuple[float, float],
    path: str,
) -> None:
    """Two-panel figure. Each panel is one solvent; brackets stay inside that panel."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, len(family_groups), figsize=(7.1, 4.15), dpi=300, sharey=True)
    if len(family_groups) == 1:
        axes = [axes]
    data_lo = min(float(np.min(v)) for _f, _o, groups, _p, _k in family_groups for v in groups.values())
    data_hi = max(float(np.max(v)) for _f, _o, groups, _p, _k in family_groups for v in groups.values())
    span = max(data_hi - data_lo, 0.2)
    tops = []
    for ax, (family, order, groups, pairs, omnibus) in zip(axes, family_groups):
        _draw_points(ax, order, groups)
        ax.axhline(0, color="0.75", lw=0.7, zorder=0)
        tops.append(_draw_brackets(ax, order, pairs, data_hi, span))
        ax.set_xlim(-0.55, len(order) - 0.45)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{name}\n(n = {len(groups[name])})" for name in order])
        kw_p = omnibus["kruskal_p"]
        kw_text = "p < 0.0001" if kw_p < 1e-4 else f"p = {kw_p:.3f}"
        ax.set_xlabel(f"Kruskal-Wallis {kw_text}", fontsize=7.5, labelpad=4, color="0.2")
        ax.set_title(f"{family} solvent", fontsize=10, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"$\Delta$ spine volume (F/F$_0$ $-$ 1)")
    y_top = max(tops) + 0.04 * span
    axes[0].set_ylim(data_lo - 0.08 * span, y_top)
    letters = "ABCDEFGH"
    for ax, letter in zip(axes, letters):
        ax.text(
            -0.16, 1.06, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left",
        )
    lo, hi = window_min
    caption = (
        f"{lo:.0f}\u2013{hi:.0f} min after uncaging ({protocol}, {power:g} mW). "
        "Each point is one spine; red mark is mean \u00b1 SEM. "
        "Brackets: Dunn after Kruskal-Wallis, Holm-adjusted within solvent. "
        "Solvents were not compared."
    )
    fig.supxlabel(caption, fontsize=6.5, color="0.15")
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.6)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def main() -> str:
    module = load_session_module()
    session = module.SESSIONS[0]
    cfg = module._cfg_for(session)
    _y_ts, y_col, ylabel = signal_columns(cfg.signal, cfg.ch_1or2)
    summary_dir = os.path.join(os.path.dirname(session.df_save_path), summary_folder_name(cfg.signal))
    summary_csv = os.path.join(summary_dir, "summary_df.csv")
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(summary_csv)
    summary = pd.read_csv(summary_csv)
    summary["drug"] = summary["condition"].map(drug_name)
    known = {drug for _family, drugs in SOLVENT_FAMILIES for drug in drugs}
    unknown = sorted(set(summary["drug"]) - known)
    if unknown:
        raise ValueError(f"Drugs with no solvent family: {unknown}")

    out_dir = os.path.join(summary_dir, "solvent_group_stats")
    os.makedirs(out_dir, exist_ok=True)
    descriptive_parts = []
    omnibus_parts = []
    pair_parts = []
    report = [
        "Within-solvent comparison of delta spine volume.",
        "Families are not compared with each other.",
        "Primary pairwise test on the figure: Dunn after Kruskal-Wallis, Holm within family.",
        "Mann-Whitney Holm is reported in the pairwise table only.",
        f"Endpoint: {y_col} ({ylabel}), window {tuple(cfg.ltp_window_min)} min.",
        "",
    ]
    grouped = summary.groupby(["uncaging_protocol", "uncaging_power_coherent_mW"], dropna=False)
    for (protocol, power), block in grouped:
        protocol_s = str(protocol)
        power_f = float(power)
        family_groups = []
        for family, order in SOLVENT_FAMILIES:
            present = [drug for drug in order if (block["drug"] == drug).any()]
            groups = {
                drug: block.loc[block["drug"] == drug, y_col].dropna().to_numpy(float)
                for drug in present
            }
            groups = {drug: values for drug, values in groups.items() if len(values) >= 2}
            if len(groups) < 2:
                report.append(f"{family}: fewer than 2 groups at {protocol_s}, {power_f:g} mW")
                continue
            ordered = [drug for drug in order if drug in groups]
            groups = {drug: groups[drug] for drug in ordered}
            desc = describe(groups, family, protocol_s, power_f)
            omnibus, pairs = family_tests(groups, family, protocol_s, power_f)
            descriptive_parts.append(desc)
            omnibus_parts.append(omnibus)
            pair_parts.append(pairs)
            family_groups.append((family, ordered, groups, pairs, omnibus))
            report.append(f"===== {family} solvent | {protocol_s} | {power_f:g} mW =====")
            report.append(desc.round(4).to_string(index=False))
            report.append(
                f"Kruskal-Wallis H({omnibus['kruskal_df']}) = {omnibus['kruskal_H']:.2f}, "
                f"p = {omnibus['kruskal_p']:.4g}"
            )
            shown = pairs.copy()
            shown["dunn"] = shown["dunn_p_holm"].map(stars)
            shown["mw"] = shown["mannwhitney_p_holm"].map(stars)
            report.append(shown.round(4).to_string(index=False))
            report.append("")
        if family_groups:
            tag = _slug(f"{protocol_s}_{power_f:g}mW")
            plot_families(
                family_groups,
                protocol_s,
                power_f,
                tuple(cfg.ltp_window_min),
                os.path.join(out_dir, f"within_solvent_{tag}.png"),
            )
    desc_df = pd.concat(descriptive_parts, ignore_index=True)
    omni_df = pd.DataFrame(omnibus_parts)
    pair_df = pd.concat(pair_parts, ignore_index=True)
    desc_df.to_csv(os.path.join(out_dir, "descriptive.csv"), index=False)
    omni_df.to_csv(os.path.join(out_dir, "kruskal_wallis.csv"), index=False)
    pair_df.to_csv(os.path.join(out_dir, "pairwise_dunn_mannwhitney.csv"), index=False)
    text = "\n".join(report)
    report_path = os.path.join(out_dir, "within_solvent_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(text)
    print(f"Saved to {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
