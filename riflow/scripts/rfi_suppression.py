"""rfi-suppression — quantify RFI subtraction from nufft-gif light curves.

Compares a *before* light-curve column (the data with RFI present, e.g.
``real_fluxcal``, or the fitted RFI model ``tab_rfi``) against an *after*
residual column (``tab_res``) at each source aperture, using the residual's
per-timestep noise null (``E[max]`` and the central band saved by ``nufft-gif``)
as the floor.

For each source it reports, at the *peak-RFI* timestep (where the before signal
is brightest), the before flux, the residual flux, the noise floor, and the
suppression factor ``before / max(after, floor)``. When the residual is within
the noise band the source is subtracted into the noise, so the factor is a
**lower limit** (``≥``) — you cannot measure a suppression deeper than the floor.

Outputs:
  <out>_suppression_lightcurves.png : per-source before/after/floor (log y)
  <out>_suppression_summary.png     : peak suppression factor per source (bar)

Two ways to point it at the light-curve npz files:

  # explicit npz paths
  rfi-suppression -a .../tab_res/lc_data/<base>_light_curves.npz \\
                  -b .../real_fluxcal/lc_data/<base>_light_curves.npz

  # locate them from the imaging config + tab suffix (mfs)
  rfi-suppression -c img_*.yaml -tsx <suffix> --after-type tab_res \\
                  --before-type real_fluxcal
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_npz(image_path: str, tab_suffix: str, data_type: str) -> str:
    """Locate the single mfs light-curve npz for a data type under gifs/."""
    tab_dir = tab_suffix or ""
    base = os.path.join(image_path, "gifs", tab_dir, data_type) if tab_dir \
        else os.path.join(image_path, "gifs", data_type)
    hits = sorted(glob.glob(os.path.join(base, "lc_data", "*mfs*_light_curves.npz")))
    if not hits:
        raise FileNotFoundError(
            f"No mfs light_curves.npz for '{data_type}' under {base}/lc_data. "
            f"Run `nufft-gif -d {data_type} -mo mfs` (with -tsx) first.")
    return hits[-1]


def _suppression(before_pk, after_pk, floor_pk, band_hi_pk):
    """Peak suppression factor and whether the residual is noise-consistent."""
    into_noise = after_pk <= band_hi_pk
    factor = before_pk / max(after_pk, floor_pk)
    return factor, into_noise


def plot_suppression(before, after, out_prefix, before_label="data (before)"):
    """Build the two suppression figures + return the per-source summary rows."""
    t = after["times_sec"]
    titles = [str(x) for x in after["titles"]]
    aft_max = after["light_curves"][:, :, 1]   # (n_src, n_t) max|flux|
    bef_max = before["light_curves"][:, :, 1]
    if "exp_max_t" not in after:
        raise KeyError(
            "The --after npz lacks the noise-null keys (exp_max_t/max_hi_t). "
            "Re-image the residual with a current nufft-gif so the floor is saved.")
    exp = after["exp_max_t"]
    hi = after["max_hi_t"]
    level = float(after["band_level"]) if "band_level" in after else 0.95
    pct = int(round(level * 100))

    n = len(titles)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))

    # --- Figure 1: per-source before / after / floor on log scale ---
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 3.2 * nrow),
                             squeeze=False)
    summary = []
    for s, title in enumerate(titles):
        ax = axes[s // ncol][s % ncol]
        b, a = bef_max[s], aft_max[s]
        ax.fill_between(t, np.full_like(hi, np.nanmin(a) * 0.5), hi,
                        color="0.85", label=f"noise {pct}% band")
        ax.plot(t, exp, color="0.3", ls="--", lw=1.3, label="E[max] noise")
        ax.plot(t, b, color="tab:orange", lw=1.2, label=before_label)
        ax.plot(t, a, color="tab:blue", lw=1.2, label="residual (after)")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylabel("max|flux| [Jy/bm]")
        if s // ncol == nrow - 1:
            ax.set_xlabel("Time [s]")
        ax.grid(True, which="both", alpha=0.3)

        if np.all(np.isnan(b)):
            continue
        k = int(np.nanargmax(b))
        factor, into = _suppression(b[k], a[k], exp[k], hi[k])
        summary.append((title, b[k], a[k], exp[k], factor, into))
        ax.annotate(
            f"peak {b[k]:.0f}→{a[k]:.0f} Jy\n"
            f"{'≥' if into else ''}{factor:.0f}× "
            f"({'floor-limited' if into else 'residual'})",
            xy=(0.02, 0.97), xycoords="axes fraction", va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    f1 = f"{out_prefix}_suppression_lightcurves.png"
    fig.savefig(f1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f1)

    # --- Figure 2: peak suppression factor per source ---
    if summary:
        fig, ax = plt.subplots(figsize=(8, 0.6 * len(summary) + 1.5))
        names = [s[0] for s in summary]
        Rs = [s[4] for s in summary]
        y = np.arange(len(summary))
        colors = ["tab:green" if s[5] else "tab:red" for s in summary]
        ax.barh(y, Rs, color=colors, alpha=0.8)
        for i, s in enumerate(summary):
            ax.text(Rs[i] * 1.02, i,
                    f"{'≥' if s[5] else ''}{s[4]:.0f}×   "
                    f"({s[1]:.0f}→{s[2]:.0f} Jy, floor {s[3]:.0f})",
                    va="center", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xscale("log")
        ax.set_xlim(1, max(Rs) * 3)
        ax.set_xlabel("peak suppression factor  (before / max(after, noise floor))")
        ax.set_title("RFI suppression at peak  "
                     "(green = into noise → lower limit; red = residual)")
        ax.grid(True, axis="x", which="both", alpha=0.3)
        fig.tight_layout()
        f2 = f"{out_prefix}_suppression_summary.png"
        fig.savefig(f2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", f2)

    return summary


def main():
    p = argparse.ArgumentParser(
        description="Quantify RFI suppression: before vs residual light curves "
                    "against the residual's per-timestep noise floor.")
    p.add_argument("-a", "--after", help="Residual (tab_res) light_curves.npz.")
    p.add_argument("-b", "--before", help="Before (e.g. real_fluxcal) npz.")
    p.add_argument("-o", "--out", default=None,
                   help="Output path prefix (default: derived from --after).")
    p.add_argument("--before-label", default="data (before)",
                   help="Legend label for the before curve.")
    # Config-based location (alternative to -a/-b)
    p.add_argument("-c", "--config_path", default=None,
                   help="Imaging config YAML to locate npz by data type.")
    p.add_argument("-tsx", "--tab_suffix", default="",
                   help="Tab suffix subdir under gifs/ (with -c).")
    p.add_argument("--after-type", default="tab_res",
                   help="Data type of the residual column (with -c).")
    p.add_argument("--before-type", default="real_fluxcal",
                   help="Data type of the before column (with -c).")
    args = p.parse_args()

    after_npz, before_npz = args.after, args.before
    if (after_npz is None or before_npz is None):
        if args.config_path is None:
            p.error("provide both -a/--after and -b/--before, or -c/--config_path "
                    "with --after-type/--before-type to locate them.")
        from riflow import load_config
        config = load_config(args.config_path)
        image_path = os.path.abspath(config["data"]["image_path"])
        after_npz = after_npz or _find_npz(image_path, args.tab_suffix, args.after_type)
        before_npz = before_npz or _find_npz(image_path, args.tab_suffix, args.before_type)

    print(f"after  (residual): {after_npz}")
    print(f"before (data)    : {before_npz}")

    out = args.out
    if out is None:
        out = os.path.join(os.path.dirname(os.path.dirname(after_npz)),
                           os.path.basename(after_npz).replace("_light_curves.npz", ""))

    after = np.load(after_npz, allow_pickle=True)
    before = np.load(before_npz, allow_pickle=True)
    summary = plot_suppression(after=after, before=before, out_prefix=out,
                               before_label=args.before_label)

    print(f"\n{'source':12s} {'before':>9s} {'after':>9s} {'floor':>8s} "
          f"{'suppr':>8s}  verdict")
    for nm, bpk, apk, fl, R, into in summary:
        print(f"{nm:12s} {bpk:9.1f} {apk:9.1f} {fl:8.1f} "
              f"{('≥' if into else ' ') + f'{R:.0f}x':>8s}  "
              f"{'into noise' if into else 'residual'}")


if __name__ == "__main__":
    main()
