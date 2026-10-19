"""Figures 1-4 of Paper 3, built only from results/records.tsv.

Nothing here computes a statistic: every value plotted is looked up by its full record
key (design, estimand, subject, seed, quantity), so a figure cannot pair a number with the
wrong seed or estimand without the lookup failing. Intervals are the paired 95% intervals
stored in the records; the five seeds are plotted as separate points, never averaged into
an interval.

Usage: python make_figures.py [RECORDS=results/records.tsv] [OUT=manuscript/figures]
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SEEDS = range(5)
DESIGNS = (("E2", "E2: declared population"), ("E1_random", "E1-random: published split"),
           ("E1_disjoint", "E1-disjoint: published files"))
COL = {"drugban": "#0072B2", "moltrans": "#D55E00", "drugonly": "#8C8C8C",
       "biot5": "#009E73", "biot5sm": "#009E73"}
NAME = {"drugban": "DrugBAN", "moltrans": "MolTrans", "drugonly": "Drug-only (ECFP4)",
        "biot5": "BioT5 (released)", "biot5sm": "BioT5, softmax score"}
EST = (("pooled", "pooled"), ("pooled_subset", "pooled,\nqualified\nproteins"),
       ("within_pairweighted", "within\nprotein,\npair weights"),
       ("within_protein", "within\nprotein,\nequal weights"))

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
})


class Records:
    def __init__(self, path):
        self.v = {}
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                key = (r["design"], r["estimand"], r["subject"], r["seed"], r["quantity"])
                f = lambda x: float(x) if x != "" else None  # noqa: E731
                self.v[key] = (f(r["value"]), f(r["lo"]), f(r["hi"]))

    def __call__(self, design, estimand, subject, seed, quantity, part=0):
        val = self.v[(design, estimand, subject, str(seed), quantity)][part]
        if val is None:
            raise KeyError(f"no interval stored for {design}/{estimand}/{subject}/{seed}/{quantity}")
        return val

    def interval(self, *key):
        return tuple(self(*key, part=i) for i in range(3))


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=600 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1
def fig1(R, out):
    P = lambda q: R("E2", "population", "target", "", q)  # noqa: E731
    fig = plt.figure(figsize=(7.0, 2.9))
    ax = fig.add_axes([0.0, 0.0, 0.40, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.985, "a", fontweight="bold", fontsize=9, va="top")
    boxes = [
        (0.93, f"BindingDB_All, 2026-10 snapshot\n{P('snapshot_rows'):,.0f} rows"),
        (0.74, f"usable IC50, single chain, declared proteins\n{P('rows_kept'):,.0f} rows"),
        (0.55, f"labelled pairs (positive < 100 nM, negative > 10 µM)\n"
               f"{P('pairs_labelled'):,.0f}"),
        (0.36, f"clean pairs, readable by every model\n{P('pairs_clean'):,.0f}"),
    ]
    for y, txt in boxes:
        ax.add_patch(FancyBboxPatch((0.06, y - 0.065), 0.88, 0.12, boxstyle="round,pad=0.006",
                                    fc="#F2F2F2", ec="#666666", lw=0.5))
        ax.text(0.5, y - 0.005, txt, ha="center", va="center", fontsize=6.5)
    for y0, y1 in ((0.865, 0.805), (0.675, 0.615), (0.485, 0.425)):
        ax.annotate("", xy=(0.5, y1), xytext=(0.5, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#444444"))
    ax.text(0.5, 0.215, f"positives {P('positives'):,.0f}    negatives {P('negatives_N'):,.0f}",
            ha="center", va="center", fontsize=6.5)
    ax.annotate("", xy=(0.5, 0.245), xytext=(0.5, 0.295),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#444444"))
    ax.text(0.5, 0.09,
            f"negatives: O {P('negatives_O'):,.0f} (eligible drugs)  |  "
            f"U {P('negatives_U_block_disjoint'):,.0f} (excluded drugs)\n"
            f"{P('negatives_U_dropped_block_overlap'):,.0f} U pairs dropped: drug shares a "
            f"connectivity block with O", ha="center", va="center", fontsize=6.2)

    n_eval = P("O_pairs_in_test_split") + P("negatives_U_block_disjoint")
    bars = [
        ("b", f"Drugs: {P('drugs_ineligible') / P('drugs_after_merging_identical_inputs'):.1%} "
              f"carry one label type and are excluded "
              f"(of {P('drugs_after_merging_identical_inputs'):,.0f})",
         P("drugs_eligible"), P("drugs_ineligible"), "eligible", "excluded"),
        ("c", f"Measured negative pairs: W_rule = {P('W_rule'):.4f} lie in U "
              f"(of {P('negatives_N_block_disjoint'):,.0f})",
         P("negatives_O"), P("negatives_U_block_disjoint"), "O", "U"),
        ("d", f"Negatives this design can score: W_eval = {P('W_eval'):.4f} lie in U "
              f"(of {n_eval:,.0f})",
         P("O_pairs_in_test_split"), P("negatives_U_block_disjoint"), "O test", "U"),
    ]
    for i, (tag, head, keep, excl, lk, le) in enumerate(bars):
        a = fig.add_axes([0.47, 0.70 - i * 0.31, 0.51, 0.12])
        tot = keep + excl
        a.barh([0], [keep / tot], color="#BBBBBB", height=1.0)
        a.barh([0], [excl / tot], left=[keep / tot], color="#CC79A7", height=1.0)
        a.set_xlim(0, 1)
        a.set_ylim(-0.5, 0.5)
        a.set_axis_off()
        a.text(0, 1.18, head, transform=a.transAxes, ha="left", va="bottom", fontsize=6.6)
        a.text(-0.035, 1.18, tag, transform=a.transAxes, ha="right", va="bottom",
               fontweight="bold", fontsize=9)
        small = keep / tot < 0.15
        if small:
            a.text(0, -0.62, f"{lk}: {keep:,.0f}", ha="left", va="top", fontsize=6)
        else:
            a.text(keep / tot / 2, 0, f"{lk}\n{keep:,.0f}", ha="center", va="center", fontsize=6)
        a.text(keep / tot + excl / tot / 2, 0, f"{le}\n{excl:,.0f}",
               ha="center", va="center", fontsize=6, color="white")
    save(fig, out, "fig1_population")


# ---------------------------------------------------------------- Figure 2
def fig2(R, out):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharey="row")
    for row, (est, ylab) in enumerate((("pooled", "AUROC, pooled"),
                                       ("within_protein", "AUROC, within protein\n(equal weights)"))):
        for col, (d, title) in enumerate(DESIGNS):
            ax = axes[row, col]
            models = ["drugban", "moltrans", "drugonly"] + (["biot5"] if d == "E1_random" else [])
            for k, m in enumerate(models):
                off = (k - (len(models) - 1) / 2) * 0.07
                seeds = [0] if m.startswith("biot5") else SEEDS
                ys = [(R(d, est, m, s, "AUROC_O"), R(d, est, m, s, "AUROC_U")) for s in seeds]
                for o, u in ys:
                    ax.plot([0 + off, 1 + off], [o, u], color=COL[m], lw=0.5, alpha=0.55)
                    ax.scatter([0 + off, 1 + off], [o, u], s=6, color=COL[m], lw=0, zorder=3,
                               marker="D" if m.startswith("biot5") else "o")
                if m == "biot5":
                    o2, u2 = R(d, est, "biot5sm", 0, "AUROC_O"), R(d, est, "biot5sm", 0, "AUROC_U")
                    ax.scatter([0 + off + 0.035, 1 + off + 0.035], [o2, u2], s=9, facecolor="white",
                               edgecolor=COL[m], lw=0.6, marker="D", zorder=3)
                else:
                    mo = sum(o for o, _ in ys) / len(ys)
                    mu = sum(u for _, u in ys) / len(ys)
                    ax.plot([0 + off, 1 + off], [mo, mu], color=COL[m], lw=1.4, zorder=4,
                            label=NAME[m] if (row, col) == (0, 0) else None)
            ax.set_xlim(-0.35, 1.35)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["O: retained\nnegatives", "U: excluded\nnegatives"])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(ylab)
            if est == "within_protein":
                n = R(d, est, "drugban", 0, "n_proteins")
                ns = [R(d, est, "drugban", s, "n_proteins") for s in SEEDS]
                cs = [R(d, est, "drugban", s, "U_coverage") for s in SEEDS]
                txt = (f"{int(n)} proteins, {cs[0]:.1%} of sampled U" if len(set(ns)) == 1 else
                       f"{int(min(ns))}–{int(max(ns))} proteins, "
                       f"{min(cs):.1%}–{max(cs):.1%} of sampled U")
                ax.set_title(txt + " negatives", fontsize=6.3, color="#444444")
            ax.grid(axis="y", lw=0.3, color="#DDDDDD")
            ax.set_axisbelow(True)
    h, lab = axes[0, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], marker="D", ls="", color=COL["biot5"], ms=3.5),
          Line2D([], [], marker="D", ls="", mfc="white", mec=COL["biot5"], ms=3.5)]
    lab += ["BioT5, released checkpoint", "BioT5, softmax scoring"]
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "fig2_regional_auroc")


# ---------------------------------------------------------------- Figure 3
def fig3(R, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for col, (d, title) in enumerate(DESIGNS):
        ax = axes[col]
        ax.axhline(0, color="#444444", lw=0.6)
        for j, (est, lab) in enumerate(EST):
            for s in SEEDS:
                x = j + (s - 2) * 0.09
                v, lo, hi = R.interval(d, est, "drugban-moltrans", s, "G")
                excl = lo > 0 or hi < 0
                ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt="o", ms=2.6, lw=0.6,
                            color="#333333", mfc="#333333" if excl else "white", mew=0.6,
                            capsize=0)
        ax.set_xticks(range(len(EST)))
        ax.set_xticklabels([lab for _, lab in EST], fontsize=5.8)
        ns = [R(d, "within_protein", "drugban-moltrans", s, "n_proteins") for s in SEEDS]
        cs = [R(d, "within_protein", "drugban-moltrans", s, "U_coverage") for s in SEEDS]
        cov = (f"{int(ns[0])} qualified proteins, {cs[0]:.1%} of sampled U" if len(set(ns)) == 1
               else f"{int(min(ns))}–{int(max(ns))} qualified proteins, "
                    f"{min(cs):.1%}–{max(cs):.1%} of sampled U")
        ax.set_title(title + "\n", fontsize=7.5)
        ax.text(0.5, 1.0, cov, transform=ax.transAxes, ha="center", va="bottom", fontsize=6,
                color="#555555")
        ax.grid(axis="y", lw=0.3, color="#DDDDDD")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("change in DrugBAN − MolTrans gap\n"
                       "G = ΔU − ΔO (AUROC)")
    fig.tight_layout()
    save(fig, out, "fig3_gap_by_aggregation")


# ---------------------------------------------------------------- Figure 4
def fig4(R, out):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9), sharey="row",
                             gridspec_kw={"width_ratios": [5, 5, 1.6]})
    q_style = (("Delta_O", "o", "ΔO (retained)"), ("Delta_U", "s", "ΔU (excluded)"),
               ("G", "^", "G = ΔU − ΔO"))
    for row, (est, ylab) in enumerate((("pooled", "BioT5 − DrugBAN\n(pooled)"),
                                       ("within_protein", "BioT5 − DrugBAN\n(within protein)"))):
        for col, (subj, title) in enumerate((("biot5-drugban", "BioT5 raw-ratio score (as released)"),
                                             ("biot5sm-drugban", "BioT5 softmax score"))):
            ax = axes[row, col]
            ax.axhline(0, color="#444444", lw=0.6)
            for s in SEEDS:
                for k, (q, mk, _) in enumerate(q_style):
                    v, lo, hi = R.interval("E1_random", est, subj, s, q)
                    excl = lo > 0 or hi < 0
                    ax.errorbar([s + (k - 1) * 0.22], [v], yerr=[[v - lo], [hi - v]], fmt=mk,
                                ms=2.8, lw=0.6, color=COL["biot5"] if q != "G" else "#333333",
                                mfc=(COL["biot5"] if q != "G" else "#333333") if excl else "white",
                                mew=0.6, capsize=0)
            ax.set_xticks(list(SEEDS))
            ax.set_xticklabels([f"DrugBAN\nseed {s}" for s in SEEDS] if row == 1 else [""] * 5)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(ylab)
            ax.grid(axis="y", lw=0.3, color="#DDDDDD")
            ax.set_axisbelow(True)
        ax = axes[row, 2]
        ax.axhline(0, color="#444444", lw=0.6)
        for k, (q, mk, _) in enumerate(q_style):
            v, lo, hi = R.interval("E1_random", est, "biot5sm-biot5", 0, q)
            excl = lo > 0 or hi < 0
            ax.errorbar([(k - 1) * 0.3], [v], yerr=[[v - lo], [hi - v]], fmt=mk, ms=2.8, lw=0.6,
                        color="#7F7F7F", mfc="#7F7F7F" if excl else "white", mew=0.6, capsize=0)
        ax.set_xlim(-0.8, 0.8)
        ax.set_xticks([0])
        ax.set_xticklabels(["same\ncheckpoint" if row == 1 else ""])
        if row == 0:
            ax.set_title("softmax − raw\nratio", fontsize=6.8)
        ax.grid(axis="y", lw=0.3, color="#DDDDDD")
        ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=mk, ls="", color="#555555", ms=3.5) for _, mk, _ in q_style]
    h.append(Line2D([], [], marker="o", ls="", mfc="white", mec="#555555", ms=3.5))
    fig.legend(h, [lab for _, _, lab in q_style] + ["open: 95% interval includes 0"],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "fig4_biot5_drugban")


def main(records=os.path.join(ROOT, "results", "records.tsv"),
         out=os.path.join(ROOT, "manuscript", "figures")):
    R = Records(records)
    fig1(R, out)
    fig2(R, out)
    fig3(R, out)
    fig4(R, out)
    print(f"figures 1-4 -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
