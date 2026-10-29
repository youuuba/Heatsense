"""One-figure run summary: heart rate, pace, estimated core temperature."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated default palette (light mode); identity never relies on color alone:
# every series is in a legend and the reference is also dashed.
BLUE = "#2a78d6"  # series 1: our signal / estimate
ORANGE = "#eb6834"  # series 2: reference measurement
FLAG = "#e34948"  # bad-data samples (always labeled)
# Status colors for strain zones; always paired with a text label in the strip.
ZONE_COLORS = {"green": "#0ca30c", "amber": "#fab219", "red": "#d03b3b", "check": "#c3c2b7"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SURFACE = "#fcfcfb"


def plot_run(
    seconds: pd.DataFrame,
    minutes: pd.DataFrame,
    estimate: pd.DataFrame,
    reference: pd.Series | None = None,
    reference_label: str = "Reference (core sensor)",
    title: str = "Run summary",
    estimate_label: str = "HeatSense estimate",
    note: str | None = None,
    path: str | None = None,
    baseline: pd.DataFrame | None = None,
    zones: pd.Series | None = None,
    profile_label: str | None = None,
):
    t_sec = (seconds.index - seconds.index[0]).total_seconds() / 60
    t_min = minutes.index.to_numpy() + 0.5

    rows = 4 if zones is not None else 3
    ratios = [1.2, 0.6, 1.4, 0.18][:rows]
    fig, axes = plt.subplots(rows, 1, figsize=(10, 9.2 if zones is not None else 8.5), sharex=True,
                             gridspec_kw={"height_ratios": ratios})
    fig.patch.set_facecolor(SURFACE)

    # 1. Heart rate.
    ax = axes[0]
    ax.plot(t_sec, seconds["heart_rate"], color=GRID, lw=1, label="Heart rate, per second")
    ax.plot(t_min, minutes["hr"], color=BLUE, lw=2, label="Heart rate, 1-min (clean samples)")
    if "hr_ok" in seconds:
        bad = ~seconds["hr_ok"] & seconds["heart_rate"].notna()
        if bad.any():
            ax.scatter(t_sec[bad], seconds["heart_rate"][bad], s=8, color=FLAG, zorder=3,
                       label=f"Flagged ({bad.mean():.1%} of samples)")
    ax.set_ylabel("Heart rate (bpm)")

    # 2. Pace.
    ax = axes[1]
    pace = (1000 / 60) / minutes["speed"].where(minutes["speed"] > 0.8)
    ax.plot(t_min, pace, color=BLUE, lw=2, label="Pace")
    ax.invert_yaxis()
    ax.set_ylabel("Pace (min/km)")

    # 3. Core temperature.
    ax = axes[2]
    ct, sd = estimate["ct"].to_numpy(), estimate["ct_sd"].to_numpy()
    ax.fill_between(t_min, ct - 2 * sd, ct + 2 * sd, color=BLUE, alpha=0.15, lw=0,
                    label="Estimate ± 2 SD")
    ax.plot(t_min, ct, color=BLUE, lw=2, label=estimate_label)
    if baseline is not None:
        ax.plot(t_min, baseline["ct"].to_numpy(), color=INK_2, lw=1.5, ls=":",
                label="Reference baseline: HR-only filter (Buller 2013)")
    if reference is not None and np.isfinite(reference).any():
        ax.plot(t_min, reference, color=ORANGE, lw=2, ls="--", label=reference_label)
    ax.set_ylabel("Core temperature (°C)")
    ax.set_xlabel("Elapsed time (min)")

    # 4. Strain zones strip (provisional thresholds).
    if zones is not None:
        ax = axes[3]
        z = zones.to_numpy()
        for name, color in ZONE_COLORS.items():
            on = z == name
            if on.any():
                ax.bar(minutes.index.to_numpy()[on] + 0.5, 1, width=1.0, color=color, lw=0,
                       label=f"{name.capitalize()} ({on.mean():.0%})")
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Zone", rotation=0, ha="right", va="center")
        ax.set_xlabel("Elapsed time (min)")
        axes[2].set_xlabel("")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.05), ncol=4, fontsize=8, frameon=False,
                  labelcolor=INK_2, handlelength=1.0, borderaxespad=0)

    # Shade paused / stopped minutes on every panel (one legend entry, on the top panel).
    if "paused" in minutes:
        p = minutes["paused"].to_numpy(dtype=bool)
        edges = np.flatnonzero(np.diff(np.r_[0, p.astype(int), 0]))
        for k, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
            for j, ax in enumerate(axes):
                ax.axvspan(minutes.index[a], minutes.index[b - 1] + 1, color=GRID, alpha=0.6,
                           lw=0, zorder=0, label="Paused / stopped" if (k == 0 and j == 0) else None)

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.yaxis.label.set_color(INK_2)
        ax.xaxis.label.set_color(INK_2)
        if ax is not (axes[3] if zones is not None else None) and len(ax.get_legend_handles_labels()[0]) > 1:
            ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.9,
                      facecolor=SURFACE, edgecolor=GRID, labelcolor=INK_2)

    fig.suptitle(title, x=0.06, ha="left", fontsize=13, color=INK, fontweight="bold")
    sub = "Research prototype, not a medical device. Zones are provisional. A low estimate never means you are safe."
    if profile_label:
        sub = f"Profile: {profile_label}. " + sub
    fig.text(0.06, 0.935, f"{note}\n{sub}" if note else sub, fontsize=8.5, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
    return fig
