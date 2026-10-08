# %%
"""Per-set correlations of Spine lifetime and spine volume (for 20261002_Camui.py).

Same tables and filters as 20261002_Camui.py (its ``cfg``). Per set (Spine, channel ch):
  unc_dlt   mean delta lifetime (ns, lifetime - pre mean) of the uncaging frames 30-50 s
            after the first pulse (frames without a lifetime fit skipped; >= 3 frames)
  pre_lt    mean absolute lifetime (ns) of the pre frames
  post_lt   mean absolute lifetime (ns) 25-35 min after uncaging
  post_dlt  post_lt - pre_lt (ns)
  dvol      delta spine volume 25-35 min (F/F0 - 1)
Each pair in PAIRS is drawn as a scatter per uncaging power (as the GCaMP vs spine volume
plots) and with all powers pooled, with the linear fit and Pearson / Spearman correlation.
Figures, the per-set table and one correlation table are written to
<session>/summary_lifetime/lifetime_vs_volume_common_sets/ (COMMON_SETS_ONLY, same sets in every
plot) or lifetime_vs_volume/ (each pair with all sets it can use).
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

from ltp_group_analysis import build_condition_group_specs, prepare_ltp_group_tables, two_line_title  # noqa: E402

# %% settings
UNCAGING_WINDOW_SEC = (30.0, 50.0)  # after the first uncaging pulse
MIN_FRAMES_IN_WINDOW = 3  # unc_dlt needs at least this many lifetime frames in the window
# True: only sets with every value of PAIRS (same n in all plots); results in
# lifetime_vs_volume_common_sets/. False: each pair uses all sets it can (lifetime_vs_volume/).
COMMON_SETS_ONLY = True

LABELS = {
    "unc_dlt": "Spine Δlifetime (ns), uncaging {u0:g}-{u1:g} s",
    "pre_lt": "Spine lifetime (ns), pre",
    "post_lt": "Spine lifetime (ns), post {p0:g}-{p1:g} min",
    "post_dlt": "Spine Δlifetime (ns), post {p0:g}-{p1:g} min",
    "dvol": "Δspine volume (a.u.), {p0:g}-{p1:g} min",
}
# (x, y): one figure each
PAIRS = [
    ("unc_dlt", "dvol"),
    ("pre_lt", "unc_dlt"),
    ("pre_lt", "dvol"),
    ("post_lt", "dvol"),
    ("post_dlt", "dvol"),
    ("unc_dlt", "post_dlt"),
]


def load_camui_cfg():
    """The cfg of 20261002_Camui.py (its analysis runs only under __main__)."""
    spec = importlib.util.spec_from_file_location("camui", os.path.join(HERE, "20261002_Camui.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cfg


def per_set_table(summary_df: pd.DataFrame, ts: pd.DataFrame, ch: int) -> pd.DataFrame:
    lo, hi = UNCAGING_WINDOW_SEC
    unc = ts[(ts["phase"] == "unc") & ts["aligned_time_sec"].between(lo, hi)]
    g = unc.groupby("group_set_id")[f"Spine_Ch{ch}_lifetime_normalized"]
    out = summary_df.set_index("group_set_id")[["condition", "uncaging_power_coherent_mW"]].copy()
    out["unc_dlt"] = g.mean()
    out["unc_dlt_n_frames"] = g.count()
    out.loc[~(out["unc_dlt_n_frames"] >= MIN_FRAMES_IN_WINDOW), "unc_dlt"] = np.nan
    s = summary_df.set_index("group_set_id")
    out["pre_lt"] = s[f"Ch{ch}_pre_lifetime"]
    out["post_lt"] = s[f"Ch{ch}_post_lifetime"]
    out["post_dlt"] = s[f"delta_lifetime_ch{ch}"]
    out["dvol"] = s[f"delta_FF0_intensity_ch{ch}"]
    return out.reset_index()


def correlation(x: np.ndarray, y: np.ndarray) -> dict:
    pr, sr, fit = stats.pearsonr(x, y), stats.spearmanr(x, y), stats.linregress(x, y)
    return {"n": len(x), "pearson_r": pr.statistic, "pearson_p": pr.pvalue, "spearman_rho": sr.statistic,
            "spearman_p": sr.pvalue, "slope": fit.slope, "intercept": fit.intercept}


def draw(ax, d: pd.DataFrame, xc: str, yc: str, by_power: list | None = None) -> dict:
    """Scatter (markers per power when by_power), fit line, correlation text."""
    if by_power:
        for power, marker in zip(by_power, "osD^v"):
            dd = d[d["uncaging_power_coherent_mW"] == power]
            ax.scatter(dd[xc], dd[yc], s=18, marker=marker, edgecolors="k", linewidths=0.3, zorder=3,
                       label=f"{power:g} mW (n={len(dd)})")
    else:
        ax.scatter(d[xc], d[yc], s=18, c="#1f77b4", edgecolors="k", linewidths=0.3, zorder=3)
    res = {}
    if len(d) >= 3:
        x, y = d[xc].to_numpy(float), d[yc].to_numpy(float)
        res = correlation(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, res["intercept"] + res["slope"] * xs, color="r", lw=1, zorder=2)
        ax.text(0.02, 0.98, f"n={res['n']}\nPearson r={res['pearson_r']:.2f}, p={res['pearson_p']:.3g}\n"
                f"Spearman ρ={res['spearman_rho']:.2f}, p={res['spearman_p']:.3g}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7)
    for c, axline in ((xc, ax.axvline), (yc, ax.axhline)):
        if c != "pre_lt" and c != "post_lt":  # zero line only for changes
            axline(0, color="gray", lw=0.5, ls="--", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return res


def main() -> str:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_camui_cfg()
    ch = cfg.ch_1or2
    summary_df, ts, save_folder = prepare_ltp_group_tables(cfg)
    table = per_set_table(summary_df, ts, ch)
    out_dir = os.path.join(save_folder, "lifetime_vs_volume")
    if COMMON_SETS_ONLY:
        cols = sorted({c for pair in PAIRS for c in pair})
        n_all = len(table)
        table = table.dropna(subset=cols)
        out_dir += "_common_sets"
        print(f"Common sets only: {len(table)} of {n_all} sets have all of {cols}")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "per_set_lifetime_volume.csv"), index=False)
    fmt = dict(u0=UNCAGING_WINDOW_SEC[0], u1=UNCAGING_WINDOW_SEC[1], p0=cfg.ltp_window_min[0], p1=cfg.ltp_window_min[1])
    label = {k: v.format(**fmt) for k, v in LABELS.items()}
    pad = lambda v: (lambda lo, hi: [lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)])(np.nanmin(v), np.nanmax(v))  # noqa: E731
    rows = []
    for xc, yc in PAIRS:
        d_all = table.dropna(subset=[xc, yc])
        headers = [(c, h) for h, c in build_condition_group_specs(d_all, list(cfg.condition_order))]
        powers = sorted(d_all["uncaging_power_coherent_mW"].dropna().unique())
        xlim, ylim = pad(d_all[xc]), pad(d_all[yc])
        name = f"{yc}_vs_{xc}"
        # per power (rows) and condition (columns)
        fig, axes = plt.subplots(len(powers), len(headers), figsize=(3.6 * len(headers), 3.3 * len(powers)),
                                 sharex=True, sharey=True, squeeze=False, dpi=200)
        for r, power in enumerate(powers):
            for c, (cond, header) in enumerate(headers):
                d = d_all[(d_all["condition"] == cond) & (d_all["uncaging_power_coherent_mW"] == power)]
                ax = axes[r, c]
                if d.empty:
                    ax.axis("off")
                    continue
                res = draw(ax, d, xc, yc)
                rows.append({"x": xc, "y": yc, "condition": cond, "uncaging_power_coherent_mW": power, **res})
                ax.set_title(f"{two_line_title(header)}, {power:g} mW", fontsize=9)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel(label[xc] if r == len(powers) - 1 else "", fontsize=8)
                ax.set_ylabel(label[yc] if c == 0 else "", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"panel_{name}.png"), bbox_inches="tight", facecolor="white")
        fig.savefig(os.path.join(out_dir, f"panel_{name}.pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        # all powers pooled, one figure per condition
        for cond, header in headers:
            d = d_all[d_all["condition"] == cond]
            fig, ax = plt.subplots(figsize=(3.8, 3.4), dpi=200)
            res = draw(ax, d, xc, yc, by_power=powers)
            rows.append({"x": xc, "y": yc, "condition": cond, "uncaging_power_coherent_mW": "all", **res})
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(label[xc], fontsize=8)
            ax.set_ylabel(label[yc], fontsize=8)
            ax.set_title(f"{two_line_title(header)}, all powers", fontsize=9)
            ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{cond}_{name}_all_powers.png"), bbox_inches="tight", facecolor="white")
            plt.close(fig)
    corr = pd.DataFrame(rows)
    corr.to_csv(os.path.join(out_dir, "correlations.csv"), index=False)
    pd.set_option("display.width", 200)
    print(corr.drop(columns=["slope", "intercept"]).round(4).to_string(index=False))
    print(f"Saved to {out_dir}")
    return out_dir


# %%
if __name__ == "__main__":
    main()
