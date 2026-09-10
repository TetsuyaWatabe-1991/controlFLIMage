"""
Compare spine GCaMP F/F0 from uncaging titration across four conditions:
  Control, Grin1 KO, Control + Gavestinel, Grin1 KO + Gavestinel.

Publication-style swarmplot with Kruskal-Wallis and pairwise Mann-Whitney tests.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _SCRIPT_DIR / "output" / "20260703_spine_ff0_four_conditions"

CONDITIONS: list[tuple[Path, str]] = [
    (
        Path(r"G:\ImagingData\Tetsuya\20260703\cont_ko\plot\titration_result.csv"),
        "Control",
    ),
    (
        Path(r"G:\ImagingData\Tetsuya\20260703\grin1_ko\plot\titration_result.csv"),
        "Grin1 KO",
    ),
    (
        Path(
            r"G:\ImagingData\Tetsuya\20260703\cont_ko_gavestinel\plot\titration_result.csv"
        ),
        "Control + Gavestinel",
    ),
    (
        Path(
            r"G:\ImagingData\Tetsuya\20260703\grin1_ko_gavestinel\plot\titration_result.csv"
        ),
        "Grin1 KO + Gavestinel",
    ),
]

GROUP_ORDER = [label for _, label in CONDITIONS]
GROUP_COLORS = {
    "Control": "#3C5488",
    "Grin1 KO": "#E64B35",
    "Control + Gavestinel": "#00A087",
    "Grin1 KO + Gavestinel": "#F39B7F",
}
Y_LABEL = r"Spine GCaMP $F/F_0$"

# Brackets shown on the figure: (x0, x1, short label for legend table)
FIGURE_BRACKETS: list[tuple[int, int, str]] = [
    (0, 1, "vehicle"),
    (2, 3, "gavestinel"),
]


def _p_to_stars(p: float) -> str:
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def load_spine_ff0(csv_path: Path, condition: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "spine_F_F0" not in df.columns:
        raise ValueError(f"Missing spine_F_F0 column in {csv_path}")
    out = df[["group", "spine_F_F0", "pow_mw_round"]].copy()
    out["condition"] = condition
    out["spine_F_F0"] = pd.to_numeric(out["spine_F_F0"], errors="coerce")
    return out.dropna(subset=["spine_F_F0"])


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in GROUP_ORDER:
        vals = df.loc[df["condition"] == condition, "spine_F_F0"].to_numpy(dtype=float)
        n = len(vals)
        rows.append(
            {
                "condition": condition,
                "n": n,
                "mean": float(np.mean(vals)) if n else np.nan,
                "sem": float(stats.sem(vals)) if n > 1 else np.nan,
                "median": float(np.median(vals)) if n else np.nan,
                "std": float(np.std(vals, ddof=1)) if n > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_kruskal_wallis(df: pd.DataFrame) -> dict[str, float]:
    samples = [
        df.loc[df["condition"] == c, "spine_F_F0"].to_numpy(dtype=float)
        for c in GROUP_ORDER
    ]
    h_stat, p_val = stats.kruskal(*samples)
    return {"kruskal_h": float(h_stat), "kruskal_p": float(p_val)}


def run_pairwise_mannwhitney(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_tests = len(list(itertools.combinations(GROUP_ORDER, 2)))
    for g1, g2 in itertools.combinations(GROUP_ORDER, 2):
        v1 = df.loc[df["condition"] == g1, "spine_F_F0"].to_numpy(dtype=float)
        v2 = df.loc[df["condition"] == g2, "spine_F_F0"].to_numpy(dtype=float)
        mw = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        tt = stats.ttest_ind(v1, v2, equal_var=False)
        rows.append(
            {
                "group_a": g1,
                "group_b": g2,
                "n_a": len(v1),
                "n_b": len(v2),
                "mannwhitney_u": float(mw.statistic),
                "mannwhitney_p": float(mw.pvalue),
                "mannwhitney_p_bonferroni": min(float(mw.pvalue) * n_tests, 1.0),
                "welch_t": float(tt.statistic),
                "welch_p": float(tt.pvalue),
            }
        )
    return pd.DataFrame(rows)


def _apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _add_significance_bracket(
    ax, x0: float, x1: float, y: float, h: float, text: str
) -> None:
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=0.8, c="black")
    ax.text((x0 + x1) / 2, y + h * 0.15, text, ha="center", va="bottom", fontsize=8)


def _short_xlabel(condition: str, n: int) -> str:
    short = {
        "Control": "Ctrl",
        "Grin1 KO": "Grin1 KO",
        "Control + Gavestinel": "Ctrl\n+ Gavestinel",
        "Grin1 KO + Gavestinel": "Grin1 KO\n+ Gavestinel",
    }
    return f"{short.get(condition, condition)}\n(n={n})"


def plot_swarmplot(
    df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    kw_stats: dict[str, float],
    save_dir: Path,
) -> None:
    _apply_pub_style()
    save_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_groups(df)

    x_labels = [
        _short_xlabel(
            c, int(summary.loc[summary["condition"] == c, "n"].iloc[0])
        )
        for c in GROUP_ORDER
    ]

    fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=150)

    sns.swarmplot(
        data=df,
        x="condition",
        y="spine_F_F0",
        hue="condition",
        order=GROUP_ORDER,
        hue_order=GROUP_ORDER,
        palette=GROUP_COLORS,
        size=3.5,
        alpha=0.9,
        legend=False,
        ax=ax,
    )
    sns.boxplot(
        data=df,
        x="condition",
        y="spine_F_F0",
        order=GROUP_ORDER,
        width=0.45,
        showcaps=False,
        boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 0.8},
        whiskerprops={"linewidth": 0.8, "color": "black"},
        medianprops={"color": "black", "linewidth": 1.0},
        showfliers=False,
        ax=ax,
    )

    for i, condition in enumerate(GROUP_ORDER):
        row = summary.loc[summary["condition"] == condition].iloc[0]
        ax.hlines(row["mean"], i - 0.2, i + 0.2, colors="black", linewidth=1.2, zorder=10)

    ax.set_xlabel("")
    ax.set_ylabel(Y_LABEL)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(x_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_max = float(df["spine_F_F0"].max())
    y_pad = max(0.8, y_max * 0.06)
    bracket_h = y_pad * 0.25
    bracket_y = y_max + y_pad

    for level, (x0, x1, _tag) in enumerate(FIGURE_BRACKETS):
        g_a, g_b = GROUP_ORDER[x0], GROUP_ORDER[x1]
        row = pairwise_df[
            ((pairwise_df["group_a"] == g_a) & (pairwise_df["group_b"] == g_b))
            | ((pairwise_df["group_a"] == g_b) & (pairwise_df["group_b"] == g_a))
        ].iloc[0]
        p_mw = row["mannwhitney_p"]
        label = f"{_p_to_stars(p_mw)}\n$p={p_mw:.3g}$"
        y_level = bracket_y + level * (bracket_h * 3.2)
        _add_significance_bracket(ax, x0, x1, y_level, bracket_h, label)

    top = bracket_y + (len(FIGURE_BRACKETS) + 0.5) * bracket_h * 3.2
    ax.set_ylim(bottom=0, top=top)

    unc_pow = df["pow_mw_round"].dropna().unique()
    pow_str = ", ".join(f"{p:g}" for p in sorted(unc_pow))
    kw_p = kw_stats["kruskal_p"]
    ax.set_title(
        f"Uncaging titration ({pow_str} mW)\n"
        f"Kruskal-Wallis $p={kw_p:.3g}$",
        pad=10,
        fontsize=9,
    )

    fig.tight_layout()
    stem = "spine_gcamp_ff0_four_conditions_swarmplot"
    for ext in ("png", "pdf", "svg"):
        out = save_dir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close(fig)


def main() -> None:
    frames = []
    for csv_path, label in CONDITIONS:
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        frames.append(load_spine_ff0(csv_path, label))

    df = pd.concat(frames, ignore_index=True)
    df["condition"] = pd.Categorical(df["condition"], categories=GROUP_ORDER, ordered=True)

    summary = summarize_groups(df)
    kw_stats = run_kruskal_wallis(df)
    pairwise_df = run_pairwise_mannwhitney(df)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUTPUT_DIR / "combined_spine_ff0.csv", index=False)
    summary.to_csv(_OUTPUT_DIR / "summary_spine_ff0.csv", index=False)
    pd.DataFrame([kw_stats]).to_csv(_OUTPUT_DIR / "statistics_kruskal_wallis.csv", index=False)
    pairwise_df.to_csv(_OUTPUT_DIR / "statistics_pairwise.csv", index=False)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print("\n=== Kruskal-Wallis (4 groups) ===")
    print(f"H = {kw_stats['kruskal_h']:.4f}, p = {kw_stats['kruskal_p']:.4g}")
    print("\n=== Pairwise Mann-Whitney (Bonferroni in CSV) ===")
    print(pairwise_df.to_string(index=False))

    plot_swarmplot(df, pairwise_df, kw_stats, _OUTPUT_DIR)


if __name__ == "__main__":
    main()
